"""
Historical Data Backfill & Model Bootstrap

    python backfill_history.py

It does four things, in order:
    1. Extends forecast history back to 2021-04-01 (~3x more training data)
       via Open-Meteo's historical forecast API.
    2. Adds ICON (German DWD model) as a 4th forecast source for all dates.
    3. Replaces ERA5 reanalysis "actuals" with the OFFICIAL settlement-station
       readings from NOAA GHCND (the numbers Kalshi actually settles on).
       The old ERA5 value is preserved in era5_high_f for reference.
    4. Retrains all city models on the corrected, enlarged dataset.

Safe to re-run: all writes are idempotent upserts/updates.
"""

from __future__ import annotations

import sqlite3
import time
from datetime import datetime, timedelta

from weather import CITIES
from historical_data import (
    init_historical_tables,
    fetch_all_historical,
    fetch_historical_forecasts,
)
from station_obs import fetch_ghcnd_daily_highs

DB_PATH = "kalshi_data.db"
BACKFILL_START = "2021-04-01"  # earliest reliable Open-Meteo forecast archive


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Add new columns for ICON forecasts and preserved ERA5 actuals."""
    init_historical_tables(conn)
    for col in ["icon_forecast_f", "icon_error", "era5_high_f"]:
        try:
            conn.execute(f"ALTER TABLE historical_forecasts ADD COLUMN {col} REAL")
            conn.commit()
        except sqlite3.OperationalError:
            pass  # column already exists


def extend_history(conn: sqlite3.Connection, city_key: str) -> None:
    """Backfill rows for dates before the current earliest date."""
    row = conn.execute(
        "SELECT MIN(date) FROM historical_forecasts WHERE city = ?",
        (city_key,)).fetchone()
    earliest = row[0]

    if earliest is None:
        end = datetime.now().strftime("%Y-%m-%d")
        fetch_all_historical(city_key, BACKFILL_START, end, conn)
        return

    if earliest <= BACKFILL_START:
        print(f"  [{city_key}] history already extends to {earliest}, skipping")
        return

    end = (datetime.strptime(earliest, "%Y-%m-%d")
           - timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"  [{city_key}] extending history {BACKFILL_START} -> {end}")
    fetch_all_historical(city_key, BACKFILL_START, end, conn)


def backfill_icon(conn: sqlite3.Connection, city_key: str) -> int:
    """Fetch ICON forecasts for every date that doesn't have one yet."""
    city = CITIES[city_key]
    rows = conn.execute("""
        SELECT MIN(date), MAX(date) FROM historical_forecasts
        WHERE city = ? AND icon_forecast_f IS NULL
    """, (city_key,)).fetchone()

    if rows[0] is None:
        print(f"  [{city_key}] ICON already backfilled")
        return 0

    start = datetime.strptime(rows[0], "%Y-%m-%d")
    end = datetime.strptime(rows[1], "%Y-%m-%d")
    updated = 0

    chunk_start = start
    while chunk_start <= end:
        chunk_end = min(chunk_start + timedelta(days=89), end)
        cs, ce = (chunk_start.strftime("%Y-%m-%d"),
                  chunk_end.strftime("%Y-%m-%d"))
        try:
            forecasts = fetch_historical_forecasts(city, cs, ce, "icon_seamless")
            for date_str, high in forecasts.items():
                cur = conn.execute("""
                    UPDATE historical_forecasts SET icon_forecast_f = ?
                    WHERE date = ? AND city = ?
                """, (high, date_str, city_key))
                updated += cur.rowcount
            conn.commit()
            print(f"  [{city_key}] ICON {cs}..{ce}: {len(forecasts)} days")
        except Exception as e:
            print(f"  [{city_key}] ICON {cs}..{ce} failed: {e}")
        time.sleep(0.5)
        chunk_start = chunk_end + timedelta(days=1)

    return updated


def fix_actuals_to_station(conn: sqlite3.Connection, city_key: str) -> dict:
    """
    Replace ERA5 'actuals' with official settlement-station TMAX (GHCND)
    and recompute all error columns against the corrected target.

    Returns stats on how far off ERA5 was — this quantifies the bias the
    model has been trained against.
    """
    rows = conn.execute("""
        SELECT date, actual_high_f, era5_high_f FROM historical_forecasts
        WHERE city = ? ORDER BY date
    """, (city_key,)).fetchall()
    if not rows:
        return {}

    start, end = rows[0][0], rows[-1][0]
    print(f"  [{city_key}] fetching official station TMAX {start}..{end}")

    # GHCND allows long ranges; fetch in 1-year chunks to be safe
    station_highs = {}
    chunk_start = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    while chunk_start <= end_dt:
        chunk_end = min(chunk_start + timedelta(days=364), end_dt)
        cs, ce = (chunk_start.strftime("%Y-%m-%d"),
                  chunk_end.strftime("%Y-%m-%d"))
        try:
            chunk = fetch_ghcnd_daily_highs(city_key, cs, ce)
            station_highs.update(chunk)
            print(f"    GHCND {cs}..{ce}: {len(chunk)} days", flush=True)
        except Exception as e:
            print(f"    GHCND {cs}..{ce} failed: {e}", flush=True)
        time.sleep(1.0)
        chunk_start = chunk_end + timedelta(days=1)

    print(f"    got {len(station_highs)} station days")

    replaced = 0
    diffs = []
    for date_str, current_actual, era5_saved in rows:
        station = station_highs.get(date_str)
        if station is None:
            continue

        # Preserve the original ERA5 value once (first run only)
        era5_value = era5_saved if era5_saved is not None else current_actual
        if current_actual is not None and era5_saved is None:
            diffs.append(era5_value - station)

        conn.execute("""
            UPDATE historical_forecasts SET
                actual_high_f = ?,
                era5_high_f = ?,
                gfs_error   = CASE WHEN gfs_forecast_f   IS NOT NULL
                              THEN gfs_forecast_f   - ? END,
                ecmwf_error = CASE WHEN ecmwf_forecast_f IS NOT NULL
                              THEN ecmwf_forecast_f - ? END,
                blend_error = CASE WHEN blend_forecast_f IS NOT NULL
                              THEN blend_forecast_f - ? END,
                icon_error  = CASE WHEN icon_forecast_f  IS NOT NULL
                              THEN icon_forecast_f  - ? END
            WHERE date = ? AND city = ?
        """, (station, era5_value, station, station, station, station,
              date_str, city_key))
        replaced += 1

    conn.commit()

    stats = {"replaced": replaced, "station_days": len(station_highs)}
    if diffs:
        mean_diff = sum(diffs) / len(diffs)
        mae_diff = sum(abs(d) for d in diffs) / len(diffs)
        stats["era5_vs_station_bias"] = round(mean_diff, 2)
        stats["era5_vs_station_mae"] = round(mae_diff, 2)
        print(f"    replaced {replaced} actuals | ERA5 was off from the "
              f"station by {mae_diff:.2f}°F MAE (bias {mean_diff:+.2f}°F)")
    return stats


def main():
    conn = sqlite3.connect(DB_PATH)
    ensure_schema(conn)

    print("=" * 60)
    print("STEP 1/4: Extend forecast history to", BACKFILL_START)
    print("=" * 60)
    for city_key in CITIES:
        extend_history(conn, city_key)

    print("\n" + "=" * 60)
    print("STEP 2/4: Backfill ICON forecasts")
    print("=" * 60)
    for city_key in CITIES:
        backfill_icon(conn, city_key)

    print("\n" + "=" * 60)
    print("STEP 3/4: Replace ERA5 actuals with settlement-station truth")
    print("=" * 60)
    for city_key in CITIES:
        fix_actuals_to_station(conn, city_key)

    print("\n" + "=" * 60)
    print("STEP 4/4: Retrain models on corrected data")
    print("=" * 60)
    from train_model import train_and_evaluate
    for city_key in CITIES:
        try:
            train_and_evaluate(city_key, conn)
        except Exception as e:
            print(f"  [{city_key}] training failed: {e}")

    conn.close()
    print("\nDone. New models saved as forecast_model_<city>.pkl")
    print("The next scheduled run picks them up automatically.")


if __name__ == "__main__":
    main()
