"""
Lead-1 Training Data Backfill

The Open-Meteo *historical forecast* archive that built historical_forecasts
serves SAME-DAY (lead-0) forecasts, but the bot trades on NEXT-DAY (lead-1)
forecasts — so the trained blend weights were fit at the wrong lead time and
training errors understated live errors. (Live daily collection is already
lead-1: each date is first recorded the day before, as "tomorrow".)

This script re-sources the forecast columns from the Previous Runs API
(temperature_2m_previous_day1 = the value predicted 24h before valid time),
computing the daily max per local day. It is:

  - NON-DESTRUCTIVE: each source's original lead-0 value is preserved once
    in {src}_lead0_f before being overwritten (same pattern as era5_high_f).
  - IDEMPOTENT: re-running refetches the same lead-1 values; lead-0
    preservation only happens while {src}_lead0_f is NULL.
  - PARTIAL BY DESIGN: model coverage varies (GFS from 2021, ECMWF/ICON/
    best_match from ~2024). Sources without lead-1 data for a date keep
    their lead-0 value — better than dropping 2021-2023 entirely.

Usage:
    python backfill_lead1.py            # backfill all cities, then retrain
    python backfill_lead1.py --city chicago
    python backfill_lead1.py --no-retrain

REVERT SWITCH (mirrors config.py's SIZING_MODE pattern):
    python backfill_lead1.py --revert   # restore lead-0 data + retrain

--revert puts every preserved {src}_lead0_f value back into the forecast
columns, recomputes errors/spread, and retrains — regenerating fresh
models with the OLD (pre-lead-1) behavior, no network calls needed. This
survives the nightly retrain (which a git-restored .pkl would not: the
daily learner would overwrite it with a lead-1-trained model next
morning). Re-applying lead-1 later is just running this script forward
again (it refetches).
"""

from __future__ import annotations

import argparse
import sqlite3
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import requests

from weather import CITIES

DB_PATH = str(Path(__file__).parent / "kalshi_data.db")
PREV_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
HEADERS = {"User-Agent": "(kalshi-weather-bot, github.com/bnbi/kalshi-weather-trading)"}

# historical_forecasts column -> Open-Meteo model identifier
SOURCES = {
    "gfs": "gfs_seamless",
    "ecmwf": "ecmwf_ifs025",
    "blend": "best_match",
    "icon": "icon_seamless",
}
CHUNK_DAYS = 90


def ensure_lead0_columns(conn: sqlite3.Connection) -> None:
    for src in SOURCES:
        try:
            conn.execute(f"ALTER TABLE historical_forecasts "
                         f"ADD COLUMN {src}_lead0_f REAL")
            conn.commit()
        except sqlite3.OperationalError:
            pass  # already exists


def fetch_lead1_daily_max(city, model: str, start: str, end: str) -> dict:
    """{date: lead-1 daily max °F} from hourly previous_day1 values."""
    resp = requests.get(PREV_RUNS_URL, params={
        "latitude": city.lat,
        "longitude": city.lon,
        "hourly": "temperature_2m_previous_day1",
        "temperature_unit": "fahrenheit",
        "timezone": city.timezone,
        "start_date": start,
        "end_date": end,
        "models": model,
    }, headers=HEADERS, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    vals = hourly.get("temperature_2m_previous_day1", [])

    by_day = defaultdict(list)
    for t, v in zip(times, vals):
        if v is not None:
            by_day[t[:10]].append(v)
    # Require most of the day present — a couple of stray hours would give
    # a bogus "daily max".
    return {d: round(max(temps), 1) for d, temps in by_day.items()
            if len(temps) >= 18}


def backfill_city(conn: sqlite3.Connection, city_key: str) -> int:
    city = CITIES[city_key]
    row = conn.execute("""SELECT MIN(date), MAX(date) FROM historical_forecasts
                          WHERE city = ?""", (city_key,)).fetchone()
    if not row or row[0] is None:
        print(f"  [{city_key}] no historical rows — skipping")
        return 0
    start_all, end_all = row
    print(f"  [{city_key}] backfilling lead-1 forecasts {start_all}..{end_all}")

    # Fetch lead-1 daily maxes per source, chunked
    lead1: dict[str, dict] = {src: {} for src in SOURCES}
    chunk_start = datetime.strptime(start_all, "%Y-%m-%d")
    end_dt = datetime.strptime(end_all, "%Y-%m-%d")
    while chunk_start <= end_dt:
        chunk_end = min(chunk_start + timedelta(days=CHUNK_DAYS - 1), end_dt)
        cs, ce = chunk_start.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")
        for src, model in SOURCES.items():
            try:
                got = fetch_lead1_daily_max(city, model, cs, ce)
                lead1[src].update(got)
            except Exception as e:
                print(f"    {src} {cs}..{ce} failed: {e}")
            time.sleep(0.4)
        print(f"    {cs}..{ce}: " + ", ".join(
            f"{s}={sum(1 for d in lead1[s] if cs <= d <= ce)}" for s in SOURCES))
        chunk_start = chunk_end + timedelta(days=1)

    # Apply to rows
    rows = conn.execute("""
        SELECT date, actual_high_f,
               gfs_forecast_f, ecmwf_forecast_f, blend_forecast_f,
               icon_forecast_f,
               gfs_lead0_f, ecmwf_lead0_f, blend_lead0_f, icon_lead0_f
        FROM historical_forecasts WHERE city = ? ORDER BY date
    """, (city_key,)).fetchall()

    updated = 0
    for (date_str, actual, gfs, ecmwf, blend, icon,
         gfs0, ecmwf0, blend0, icon0) in rows:
        current = {"gfs": gfs, "ecmwf": ecmwf, "blend": blend, "icon": icon}
        saved0 = {"gfs": gfs0, "ecmwf": ecmwf0, "blend": blend0, "icon": icon0}

        new_vals = {}
        sets, params = [], []
        for src in SOURCES:
            v1 = lead1[src].get(date_str)
            if v1 is None:
                new_vals[src] = current[src]  # keep what we have
                continue
            # Preserve the original lead-0 value exactly once
            if saved0[src] is None and current[src] is not None:
                sets.append(f"{src}_lead0_f = ?")
                params.append(current[src])
            sets.append(f"{src}_forecast_f = ?")
            params.append(v1)
            new_vals[src] = v1
            if actual is not None:
                sets.append(f"{src}_error = ?")
                params.append(v1 - actual)

        if not sets:
            continue

        # Recompute spread from the three core sources actually stored
        core = [new_vals[s] for s in ("gfs", "ecmwf", "blend")
                if new_vals[s] is not None]
        if len(core) > 1:
            sets.append("model_spread = ?")
            params.append(max(core) - min(core))

        params += [date_str, city_key]
        conn.execute(f"""UPDATE historical_forecasts SET {', '.join(sets)}
                         WHERE date = ? AND city = ?""", params)
        updated += 1

    conn.commit()
    n_l1 = {s: len(lead1[s]) for s in SOURCES}
    print(f"    updated {updated} rows | lead-1 coverage: {n_l1}")
    return updated


def revert_city(conn: sqlite3.Connection, city_key: str) -> int:
    """
    Restore the preserved lead-0 values into the forecast columns and
    recompute errors/spread — the data-level undo of backfill_city.
    Offline (no API calls). The *_lead0_f columns are kept, so running
    the forward backfill again later still preserves-once correctly.
    """
    rows = conn.execute("""
        SELECT date, actual_high_f,
               gfs_forecast_f, ecmwf_forecast_f, blend_forecast_f,
               icon_forecast_f,
               gfs_lead0_f, ecmwf_lead0_f, blend_lead0_f, icon_lead0_f
        FROM historical_forecasts WHERE city = ? ORDER BY date
    """, (city_key,)).fetchall()

    reverted = 0
    for (date_str, actual, gfs, ecmwf, blend, icon,
         gfs0, ecmwf0, blend0, icon0) in rows:
        current = {"gfs": gfs, "ecmwf": ecmwf, "blend": blend, "icon": icon}
        saved0 = {"gfs": gfs0, "ecmwf": ecmwf0, "blend": blend0, "icon": icon0}

        sets, params, final = [], [], {}
        for src in SOURCES:
            if saved0[src] is None:
                final[src] = current[src]  # was never overwritten
                continue
            sets.append(f"{src}_forecast_f = ?")
            params.append(saved0[src])
            final[src] = saved0[src]
            if actual is not None:
                sets.append(f"{src}_error = ?")
                params.append(saved0[src] - actual)

        if not sets:
            continue

        core = [final[s] for s in ("gfs", "ecmwf", "blend")
                if final[s] is not None]
        if len(core) > 1:
            sets.append("model_spread = ?")
            params.append(max(core) - min(core))

        params += [date_str, city_key]
        conn.execute(f"""UPDATE historical_forecasts SET {', '.join(sets)}
                         WHERE date = ? AND city = ?""", params)
        reverted += 1

    conn.commit()
    print(f"  [{city_key}] restored lead-0 values on {reverted} rows")
    return reverted


def main():
    parser = argparse.ArgumentParser(description="Backfill lead-1 forecasts")
    parser.add_argument("--city", choices=list(CITIES.keys()), default=None)
    parser.add_argument("--no-retrain", action="store_true",
                        help="Skip retraining after the backfill")
    parser.add_argument("--revert", action="store_true",
                        help="Restore preserved lead-0 data and retrain "
                             "(full undo of the lead-1 migration; offline)")
    parser.add_argument("--db", default=DB_PATH,
                        help="Database path (default: the live DB)")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    ensure_lead0_columns(conn)
    cities = [args.city] if args.city else list(CITIES.keys())

    print("=" * 60)
    if args.revert:
        print("STEP 1/2: REVERT — restore lead-0 forecasts from *_lead0_f")
        print("=" * 60)
        for ck in cities:
            revert_city(conn, ck)
    else:
        print("STEP 1/2: Re-source forecasts at lead-1 (Previous Runs API)")
        print("=" * 60)
        for ck in cities:
            backfill_city(conn, ck)

    if not args.no_retrain:
        print("\n" + "=" * 60)
        print(f"STEP 2/2: Retrain models on "
              f"{'restored lead-0' if args.revert else 'lead-1'} data")
        print("=" * 60)
        from train_model import train_and_evaluate
        for ck in cities:
            try:
                train_and_evaluate(ck, conn)
            except Exception as e:
                print(f"  [{ck}] training failed: {e}")

    conn.close()
    if args.revert:
        print("\nDone. Models now retrained on the ORIGINAL lead-0 data.")
    else:
        print("\nDone. Lead-0 originals preserved in *_lead0_f columns.")


if __name__ == "__main__":
    main()
