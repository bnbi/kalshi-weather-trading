"""
Daily Self-Training Pipeline
Each run, this module:
    1. Re-verifies provisional actuals (NWS obs feed / ERA5) against the
       official NOAA GHCND record as it arrives (1-3 day lag)
    2. Verifies recent unverified days (official if available, else the
       provisional feed max, clearly labelled as such)
    3. Syncs official, lead-1 days into the training set
    4. Retrains the per-city models on the expanded dataset
    5. Records what each source (and the model) predicts for today/tomorrow

Ground truth policy (the part that used to be wrong): only a GHCND value is
the settlement number. The NWS obs feed max differed from it on ~40% of days
and by up to -1.4°F on average per city, so feed values are stored with
actual_source='feed', kept OUT of training / live calibration / the sniper
gate, and replaced once GHCND publishes.

Usage:
    python daily_learner.py                # run full daily cycle (all cities)
    python daily_learner.py --city chicago # single city
    python daily_learner.py record         # just record today's forecasts
    python daily_learner.py learn          # just verify/re-verify and retrain
    python daily_learner.py stats          # show learning stats
"""

from __future__ import annotations

import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from weather import CITIES
from historical_data import init_historical_tables, fetch_actual_temps
from train_model import train_and_evaluate, get_model_path
from weather_ensemble import fetch_open_meteo_forecast, ml_predict_for
from db_migrations import migrate_db, sync_training_row, lead_days_for
from station_obs import (fetch_station_daily_high_with_source,
                         fetch_ghcnd_daily_highs, SOURCE_OFFICIAL, SOURCE_FEED)
import weatherkit

BOT_DIR = Path(__file__).parent
DB_PATH = BOT_DIR / "kalshi_data.db"

# How many recent unverified days each learning cycle tries to catch up on.
VERIFY_LOOKBACK_DAYS = 7

# Provisional (feed/ERA5/unknown-source) rows are re-checked against GHCND
# for this long. GHCND normally lands within 3 days; the long window also
# lets the first run after this change correct months of legacy rows.
REVERIFY_LOOKBACK_DAYS = 400


def _local_today(city_key: str) -> str:
    return datetime.now(ZoneInfo(CITIES[city_key].timezone)).strftime("%Y-%m-%d")


# ── Database setup ─────────────────────────────────────────────────

def init_prediction_log(conn: sqlite3.Connection) -> None:
    """Create table for daily prediction logging."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS daily_predictions (
            date TEXT NOT NULL,
            city TEXT NOT NULL,
            gfs_forecast_f REAL,
            ecmwf_forecast_f REAL,
            blend_forecast_f REAL,
            model_prediction_f REAL,
            actual_high_f REAL,
            model_error REAL,
            gfs_error REAL,
            ecmwf_error REAL,
            blend_error REAL,
            recorded_at TEXT,
            verified_at TEXT,
            PRIMARY KEY (date, city)
        );
    """)
    conn.commit()

    # Extra columns added after the table was first created.
    for col in ["icon_forecast_f REAL", "icon_error REAL",
                "wk_forecast_f REAL", "wk_error REAL",
                # Where actual_high_f came from: 'station' (GHCND, the
                # settlement number), 'feed' (NWS obs max, provisional) or
                # 'era5' (reanalysis fallback). Only 'station' rows may
                # feed training, live calibration or the sniper gate.
                "actual_source TEXT",
                # Days between the recording date and the target date.
                "lead_days INTEGER"]:
        try:
            conn.execute(f"ALTER TABLE daily_predictions ADD COLUMN {col}")
            conn.commit()
        except sqlite3.OperationalError:
            pass  # column already exists


# ── Step 1: Record today's forecasts ───────────────────────────────

def record_forecasts(conn: sqlite3.Connection, city_key: str,
                     target_date: str = None):
    """
    Record what each model predicts for a given date.
    Call this each morning to log today's or tomorrow's forecast.
    """
    city = CITIES[city_key]

    if target_date is None:
        target_date = _local_today(city_key)

    # Check if already recorded
    existing = conn.execute(
        "SELECT 1 FROM daily_predictions WHERE date = ? AND city = ?",
        (target_date, city_key)
    ).fetchone()

    if existing:
        print(f"  [{city_key}] Already recorded forecast for {target_date}")
        return

    print(f"  [{city_key}] Recording forecasts for {target_date}...")

    # Fetch from each model
    gfs = ecmwf = blend = icon = wk = None

    try:
        forecasts = fetch_open_meteo_forecast(city, "/v1/gfs", "gfs")
        gfs = forecasts.get(target_date)
    except Exception as e:
        print(f"    GFS fetch failed: {e}")

    time.sleep(0.3)

    try:
        forecasts = fetch_open_meteo_forecast(city, "/v1/ecmwf", "ecmwf")
        ecmwf = forecasts.get(target_date)
    except Exception as e:
        print(f"    ECMWF fetch failed: {e}")

    time.sleep(0.3)

    try:
        forecasts = fetch_open_meteo_forecast(city, "/v1/forecast", "best_match")
        blend = forecasts.get(target_date)
    except Exception as e:
        print(f"    Blend fetch failed: {e}")

    time.sleep(0.3)

    try:
        forecasts = fetch_open_meteo_forecast(city, "/v1/dwd-icon", "icon")
        icon = forecasts.get(target_date)
    except Exception as e:
        print(f"    ICON fetch failed: {e}")

    # Apple WeatherKit — independent of the NWP models above, so it is the
    # most informative addition to the ensemble. Silently skipped when
    # credentials are absent.
    if weatherkit.is_enabled():
        try:
            wk = weatherkit.fetch_daily_highs(city).get(target_date)
        except Exception as e:
            print(f"    WeatherKit fetch failed: {e}")

    if gfs is None and ecmwf is None and blend is None:
        print(f"    No forecasts available — skipping")
        return

    # Trained-model prediction on the SAME inputs and features the live
    # pipeline uses (exclusions, imputation, weather + trend features).
    model_pred = None
    try:
        result = ml_predict_for(city_key, target_date, gfs, ecmwf, blend,
                                icon=icon, wk=wk)
        if result is not None:
            model_pred = result["predicted_high"]
    except Exception as e:
        print(f"    Trained model prediction failed: {e}")

    now = datetime.now(timezone.utc).isoformat()
    lead = lead_days_for(target_date, now, city.timezone)
    conn.execute("""
        INSERT OR IGNORE INTO daily_predictions (
            date, city, gfs_forecast_f, ecmwf_forecast_f, blend_forecast_f,
            icon_forecast_f, wk_forecast_f, model_prediction_f, recorded_at,
            lead_days
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (target_date, city_key, gfs, ecmwf, blend, icon, wk, model_pred, now, lead))
    conn.commit()

    forecasts_str = f"GFS={gfs}, ECMWF={ecmwf}, Blend={blend}, ICON={icon}"
    if wk is not None:
        forecasts_str += f", Apple={wk}"
    model_str = f", Model={model_pred}" if model_pred else ""
    print(f"    Recorded (lead {lead}d): {forecasts_str}{model_str}")


# ── Step 2: Verify past predictions ────────────────────────────────

def _write_verification(conn: sqlite3.Connection, city_key: str, check_date: str,
                        actual: float, source: str) -> None:
    row = conn.execute("""
        SELECT gfs_forecast_f, ecmwf_forecast_f, blend_forecast_f,
               model_prediction_f, icon_forecast_f, wk_forecast_f
        FROM daily_predictions WHERE date = ? AND city = ?
    """, (check_date, city_key)).fetchone()
    if row is None:
        return
    gfs, ecmwf, blend, model_pred, icon, wk = row
    err = lambda v: (v - actual) if v is not None else None
    now = datetime.now(timezone.utc).isoformat()
    conn.execute("""
        UPDATE daily_predictions SET
            actual_high_f = ?, actual_source = ?,
            model_error = ?, gfs_error = ?, ecmwf_error = ?, blend_error = ?,
            icon_error = ?, wk_error = ?, verified_at = ?
        WHERE date = ? AND city = ?
    """, (actual, source, err(model_pred), err(gfs), err(ecmwf), err(blend),
          err(icon), err(wk), now, check_date, city_key))
    conn.commit()


def verify_yesterday(conn: sqlite3.Connection, city_key: str,
                     check_date: str = None) -> dict | None:
    """
    Fetch the actual high for a past date and compare to predictions.
    Returns the verification result or None if not available.
    """
    if check_date is None:
        check_date = (datetime.now(ZoneInfo(CITIES[city_key].timezone))
                      - timedelta(days=1)).strftime("%Y-%m-%d")

    # Check if we have a prediction to verify
    row = conn.execute("""
        SELECT gfs_forecast_f, ecmwf_forecast_f, blend_forecast_f,
               model_prediction_f, actual_high_f,
               icon_forecast_f, wk_forecast_f, actual_source
        FROM daily_predictions
        WHERE date = ? AND city = ?
    """, (check_date, city_key)).fetchone()

    if row is None:
        print(f"  [{city_key}] No prediction recorded for {check_date}")
        return None

    if row[4] is not None:
        print(f"  [{city_key}] Already verified {check_date} "
              f"(actual={row[4]}°F, source={row[7] or 'unknown'})")
        return {"actual": row[4], "already_done": True, "source": row[7]}

    gfs, ecmwf, blend, model_pred = row[0], row[1], row[2], row[3]
    icon, wk = row[5], row[6]

    # Fetch actual temperature — official settlement-station truth first,
    # then the provisional obs feed, ERA5 reanalysis only as a last resort.
    print(f"  [{city_key}] Fetching actual high for {check_date}...")
    actual, actual_source = None, None
    try:
        actual, actual_source = fetch_station_daily_high_with_source(city_key, check_date)
        if actual is not None:
            label = ("official station reading" if actual_source == SOURCE_OFFICIAL
                     else "PROVISIONAL obs-feed max — will be re-verified against GHCND")
            print(f"    ({label})")
    except Exception as e:
        print(f"    Station obs failed: {e}")

    if actual is None:
        try:
            city = CITIES[city_key]
            actuals = fetch_actual_temps(city, check_date, check_date)
            actual = actuals.get(check_date)
            if actual is not None:
                actual_source = "era5"
                print(f"    WARNING: using ERA5 fallback — may differ "
                      f"from settlement station; will be re-verified")
        except Exception as e:
            print(f"    Could not fetch actual temp: {e}")
            return None

    if actual is None:
        print(f"    Actual temp not yet available for {check_date}")
        return None

    _write_verification(conn, city_key, check_date, actual, actual_source)

    print(f"    Actual: {actual}°F [{actual_source}]")
    for name, val in (("GFS", gfs), ("ECMWF", ecmwf), ("Blend", blend),
                      ("ICON", icon), ("Apple", wk), ("Model", model_pred)):
        if val is not None:
            print(f"    {name:<6} {val}°F (error: {val - actual:+.1f}°F)")

    return {
        "date": check_date,
        "city": city_key,
        "actual": actual,
        "source": actual_source,
        "model_error": (model_pred - actual) if model_pred is not None else None,
        "gfs_error": (gfs - actual) if gfs is not None else None,
    }


def reverify_provisional(conn: sqlite3.Connection, city_key: str,
                         lookback_days: int = REVERIFY_LOOKBACK_DAYS) -> int:
    """
    Replace provisional actuals (feed / ERA5 / unknown-source legacy rows)
    with the official GHCND value wherever it is now available, recompute
    the error columns, and re-sync the training row. One NCEI call per city.
    Returns the number of rows corrected.
    """
    today = _local_today(city_key)
    floor = (datetime.strptime(today, "%Y-%m-%d")
             - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    rows = conn.execute("""
        SELECT date, actual_high_f FROM daily_predictions
        WHERE city = ? AND actual_high_f IS NOT NULL
          AND (actual_source IS NULL OR actual_source != 'station')
          AND date >= ? AND date < ?
        ORDER BY date
    """, (city_key, floor, today)).fetchall()
    if not rows:
        return 0

    start, end = rows[0][0], rows[-1][0]
    try:
        official = fetch_ghcnd_daily_highs(city_key, start, end)
    except Exception as e:
        print(f"  [{city_key}] GHCND re-verification fetch failed: {e}")
        return 0

    fixed = 0
    changed = []
    for date, provisional in rows:
        truth = official.get(date)
        if truth is None:
            continue  # GHCND hasn't published this day yet — try next run
        _write_verification(conn, city_key, date, truth, SOURCE_OFFICIAL)
        sync_training_row(conn, city_key, date)
        fixed += 1
        if provisional is not None and round(provisional) != round(truth):
            changed.append((date, provisional, truth))

    if fixed:
        print(f"  [{city_key}] Re-verified {fixed} provisional day(s) against GHCND"
              + (f"; settlement value changed on {len(changed)}: "
                 + ", ".join(f"{d} {p:.1f}->{t:.0f}" for d, p, t in changed[:6])
                 + (" ..." if len(changed) > 6 else "") if changed else ""))
    return fixed


# ── Step 3: Add to training set ────────────────────────────────────

def add_to_training_data(conn: sqlite3.Connection, city_key: str,
                          check_date: str = None) -> bool:
    """
    Sync a verified day into historical_forecasts for future training.
    Only OFFICIAL (station) truth at lead >= 1 is admitted — see
    db_migrations.sync_training_row. Weather features (wind, humidity,
    cloud) are fetched from the archive API on first insert.
    """
    if check_date is None:
        check_date = (datetime.now(ZoneInfo(CITIES[city_key].timezone))
                      - timedelta(days=1)).strftime("%Y-%m-%d")

    row = conn.execute("""
        SELECT actual_source, lead_days FROM daily_predictions
        WHERE date = ? AND city = ? AND actual_high_f IS NOT NULL
    """, (check_date, city_key)).fetchone()
    if row is None or row[0] != SOURCE_OFFICIAL:
        return False
    if row[1] is not None and row[1] < 1:
        print(f"  [{city_key}] {check_date} was recorded on the day itself "
              f"(lead 0) — kept out of the training set")
        return False

    existing = conn.execute(
        "SELECT 1 FROM historical_forecasts WHERE date = ? AND city = ?",
        (check_date, city_key)).fetchone()

    weather = None
    if not existing:
        try:
            city = CITIES[city_key]
            wx_all = fetch_actual_temps(city, check_date, check_date, include_weather=True)
            wx = wx_all.get(check_date, {})
            if isinstance(wx, dict):
                weather = {"wind": wx.get("wind"), "humidity": wx.get("humidity"),
                           "cloud": wx.get("cloud")}
                print(f"    Weather: wind={weather['wind']}, "
                      f"humidity={weather['humidity']}, cloud={weather['cloud']}")
        except Exception as e:
            print(f"    Could not fetch weather features: {e}")

    init_historical_tables(conn)
    written = sync_training_row(conn, city_key, check_date, weather=weather)
    if written and not existing:
        print(f"  [{city_key}] Added {check_date} to training data")
    return written and not existing


# ── Step 4: Retrain ────────────────────────────────────────────────

def retrain_model(conn: sqlite3.Connection, city_key: str) -> bool:
    """Retrain the model with the expanded dataset."""
    count = conn.execute(
        "SELECT COUNT(*) FROM historical_forecasts WHERE city = ? "
        "AND lead_ok = 1 AND actual_source = 'station'",
        (city_key,)
    ).fetchone()[0]

    print(f"  [{city_key}] Retraining on {count} clean days of data...")

    try:
        model_data = train_and_evaluate(city_key, conn)
        print(f"  [{city_key}] New MAE: {model_data['train_mae']:.2f}°F "
              f"(σ={model_data['residual_std']:.2f}°F, "
              f"{model_data['model_name']}, "
              f"day-σ {'on' if model_data['sigma_model'] is not None else 'off'})")
        return True
    except Exception as e:
        print(f"  [{city_key}] Retraining failed: {e}")
        return False


# ── Daily cycle pieces ─────────────────────────────────────────────

def learn(conn: sqlite3.Connection, cities: list[str], retrain: bool = True) -> bool:
    """
    Verification + training half of the cycle. Returns True if the
    training set gained or corrected rows (and models were retrained).
    """
    any_new_data = False

    print(f"\n--- Re-verifying provisional actuals against GHCND ---")
    for city in cities:
        try:
            if reverify_provisional(conn, city):
                any_new_data = True
        except Exception as e:
            print(f"  [{city}] re-verification failed: {e}")

    # Verify recent unverified days (not just yesterday — if the machine was
    # off for a day, that date previously stayed unverified forever,
    # silently starving live calibration and the sniper gate).
    print(f"\n--- Verifying recent unverified predictions ---")
    for city in cities:
        today = _local_today(city)
        pending = conn.execute("""
            SELECT date FROM daily_predictions
            WHERE city = ? AND actual_high_f IS NULL AND date < ?
            ORDER BY date DESC LIMIT ?
        """, (city, today, VERIFY_LOOKBACK_DAYS)).fetchall()
        for (check_date,) in pending:
            result = verify_yesterday(conn, city, check_date)
            if result and not result.get("already_done"):
                if add_to_training_data(conn, city, check_date):
                    any_new_data = True

    if retrain:
        if any_new_data:
            print(f"\n--- Retraining models with new data ---")
            for city in cities:
                retrain_model(conn, city)
        else:
            print(f"\n  No new official data to retrain on.")
    return any_new_data


def record(conn: sqlite3.Connection, cities: list[str]) -> None:
    """Record today's and tomorrow's forecasts for later verification."""
    print(f"\n--- Recording today's forecasts ---")
    for city in cities:
        record_forecasts(conn, city, _local_today(city))

    print(f"\n--- Recording tomorrow's forecasts ---")
    for city in cities:
        tomorrow = (datetime.now(ZoneInfo(CITIES[city].timezone))
                    + timedelta(days=1)).strftime("%Y-%m-%d")
        record_forecasts(conn, city, tomorrow)


def run_daily_learning(cities: list[str] = None, retrain: bool = True,
                       do_record: bool = True) -> None:
    """
    Full daily learning cycle:
    1. Re-verify provisional actuals against GHCND; verify recent days
    2. Sync official data into the training set and retrain
    3. Record today's/tomorrow's forecasts for later verification
    """
    if cities is None:
        cities = list(CITIES.keys())

    conn = sqlite3.connect(str(DB_PATH))
    init_prediction_log(conn)
    init_historical_tables(conn)
    migrate_db(conn, verbose=True)

    today = datetime.now().strftime("%Y-%m-%d")
    print(f"\n{'=' * 60}")
    print(f"  DAILY LEARNING CYCLE — {today}")
    print(f"{'=' * 60}")

    learn(conn, cities, retrain=retrain)
    if do_record:
        record(conn, cities)

    conn.close()
    print(f"\n  Daily learning cycle complete.")


# ── Stats ──────────────────────────────────────────────────────────

def print_learning_stats(conn: sqlite3.Connection) -> None:
    """Print statistics about the daily learning process."""
    init_prediction_log(conn)
    migrate_db(conn)

    print(f"\n{'=' * 60}")
    print(f"  DAILY LEARNING STATS")
    print(f"{'=' * 60}")

    for city_key in CITIES:
        row = conn.execute("""
            SELECT COUNT(*),
                   SUM(CASE WHEN actual_high_f IS NOT NULL THEN 1 ELSE 0 END),
                   AVG(ABS(model_error)),
                   AVG(ABS(gfs_error)),
                   AVG(ABS(ecmwf_error)),
                   AVG(ABS(blend_error)),
                   MIN(date), MAX(date),
                   AVG(ABS(icon_error)),
                   AVG(ABS(wk_error)),
                   SUM(CASE WHEN wk_error IS NOT NULL THEN 1 ELSE 0 END),
                   SUM(CASE WHEN actual_source = 'station' THEN 1 ELSE 0 END),
                   SUM(CASE WHEN actual_source = 'feed' THEN 1 ELSE 0 END),
                   AVG(CASE WHEN actual_source = 'station' THEN model_error END)
            FROM daily_predictions
            WHERE city = ?
        """, (city_key,)).fetchone()

        total, verified = row[0], row[1] or 0
        if total == 0:
            print(f"\n  {city_key}: no predictions recorded yet")
            continue

        print(f"\n  {city_key.upper()}")
        print(f"  {'─' * 40}")
        print(f"  Recorded:   {total} days ({row[6]} to {row[7]})")
        print(f"  Verified:   {verified} days  (official GHCND: {row[11] or 0}, "
              f"provisional feed: {row[12] or 0}, other/legacy: "
              f"{verified - (row[11] or 0) - (row[12] or 0)})")
        print(f"  Pending:    {total - verified} days")

        if verified > 0:
            model_mae = row[2]
            print(f"\n  Daily tracking MAE (all verified days):")
            if model_mae is not None:
                print(f"    Trained model: {model_mae:.2f}°F"
                      + (f"  (bias vs official {row[13]:+.2f}°F)" if row[13] is not None else ""))
            for label, val in (("GFS", row[3]), ("ECMWF", row[4]), ("Blend", row[5]),
                               ("ICON", row[8])):
                if val is not None:
                    print(f"    {label:<14} {val:.2f}°F")
            if row[9] is not None:
                print(f"    Apple:         {row[9]:.2f}°F  ({row[10]} days)")
                _print_weatherkit_status(city_key)

        _print_live_calibration(city_key)

    # Training data growth
    print(f"\n  {'─' * 40}")
    print(f"  Training data size (clean lead-1, official-truth rows):")
    for city_key in CITIES:
        count = conn.execute(
            "SELECT COUNT(*) FROM historical_forecasts WHERE city = ? "
            "AND lead_ok = 1 AND actual_source = 'station'",
            (city_key,)
        ).fetchone()[0]
        total = conn.execute(
            "SELECT COUNT(*) FROM historical_forecasts WHERE city = ?",
            (city_key,)).fetchone()[0]
        print(f"    {city_key}: {count} days (of {total} stored)")


def _print_live_calibration(city_key: str) -> None:
    """Current-model errors over recent official days, plus σ coverage."""
    try:
        from weather_ensemble import live_calibration_errors
        from train_model import load_model
        errs = live_calibration_errors(city_key)
        if len(errs) < 3:
            print(f"    Live calibration: only {len(errs)} official day(s) re-scored")
            return
        mean = sum(errs) / len(errs)
        sd = (sum((e - mean) ** 2 for e in errs) / max(len(errs) - 1, 1)) ** 0.5
        md = load_model(get_model_path(city_key))
        const = md.get("residual_std")
        print(f"    Live calibration (current model, {len(errs)} official days): "
              f"bias {mean:+.2f}°F, σ {sd:.2f}°F  [model residual σ {const:.2f}]")
    except Exception as e:
        print(f"    Live calibration unavailable: {e}")


def _print_weatherkit_status(city_key: str) -> None:
    """Show whether Apple's forecast has earned a blend weight yet."""
    from weather_ensemble import EnsembleForecast, _wk_config
    from weather import CITIES as _CITIES

    # Fair comparison: Apple vs the model on the SAME verified days only.
    # Per-column AVG(ABS(...)) elsewhere averages each source over its own
    # coverage window — the model's window includes months Apple wasn't
    # collected, so those numbers are NOT comparable across seasons.
    try:
        conn = sqlite3.connect(str(DB_PATH))
        n, wk_mae, model_mae = conn.execute("""
            SELECT COUNT(*), AVG(ABS(wk_error)), AVG(ABS(model_error))
            FROM daily_predictions
            WHERE city = ? AND wk_error IS NOT NULL
              AND model_error IS NOT NULL AND actual_source = 'station'
        """, (city_key,)).fetchone()
        conn.close()
        if n:
            print(f"    → Same-day comparison ({n} official day(s)): "
                  f"Apple MAE {wk_mae:.2f}°F vs model {model_mae:.2f}°F")
    except Exception:
        pass

    ens = EnsembleForecast(city=_CITIES[city_key].name, date="")
    weight = ens._get_weatherkit_weight()
    mode = _wk_config("WEATHERKIT_MODE", "shadow")
    min_days = _wk_config("WEATHERKIT_MIN_VERIFIED_DAYS",
                          EnsembleForecast.WK_MIN_VERIFIED_DAYS)

    if weight <= 0:
        print(f"    → Apple blend weight: 0.00 "
              f"(needs {min_days} verified days, or Apple adds no signal)")
    elif mode != "blend":
        print(f"    → Apple blend weight: {weight:.2f} would apply, but "
              f"WEATHERKIT_MODE is \"{mode}\" — set it to \"blend\" to use it")
    else:
        print(f"    → Apple blend weight: {weight:.2f} (active)")


# ── CLI ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Daily self-training pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  (default)    Full daily cycle: re-verify → verify → learn → retrain → record
  record       Just record today's/tomorrow's forecasts
  learn        Just re-verify/verify and retrain
  stats        Show learning statistics
        """
    )
    parser.add_argument("command", nargs="?", default="full",
                        choices=["full", "record", "learn", "stats"],
                        help="What to do (default: full cycle)")
    parser.add_argument("--city", type=str, choices=list(CITIES.keys()),
                        help="Single city only")
    args = parser.parse_args()

    cities = [args.city] if args.city else None

    if args.command == "full":
        run_daily_learning(cities)

    elif args.command == "record":
        conn = sqlite3.connect(str(DB_PATH))
        init_prediction_log(conn)
        migrate_db(conn)
        record(conn, cities or list(CITIES.keys()))
        conn.close()

    elif args.command == "learn":
        conn = sqlite3.connect(str(DB_PATH))
        init_prediction_log(conn)
        init_historical_tables(conn)
        migrate_db(conn, verbose=True)
        learn(conn, cities or list(CITIES.keys()))
        conn.close()

    elif args.command == "stats":
        conn = sqlite3.connect(str(DB_PATH))
        print_learning_stats(conn)
        conn.close()
