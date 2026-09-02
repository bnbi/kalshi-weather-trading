"""
Daily Self-Training Pipeline
Each morning, this module:
    1. Records what each model predicted for today (saved for tomorrow's comparison)
    2. Fetches yesterday's actual high temperature
    3. Compares yesterday's predictions to reality
    4. Appends the new data point to the training set
    5. Retrains the model with the expanded dataset

The model continuously improves as it accumulates more data and adapts to
seasonal patterns in real time.

Usage:
    python daily_learner.py                # run full daily cycle (all cities)
    python daily_learner.py --city chicago # single city
    python daily_learner.py record         # just record today's forecasts
    python daily_learner.py learn          # just learn from yesterday
    python daily_learner.py stats          # show learning stats
"""

from __future__ import annotations

import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from weather import CITIES
from historical_data import init_historical_tables, fetch_actual_temps
from train_model import train_and_evaluate, get_model_path
from weather_ensemble import fetch_open_meteo_forecast
import weatherkit

BOT_DIR = Path(__file__).parent
DB_PATH = BOT_DIR / "kalshi_data.db"

# How many recent unverified days each learning cycle tries to catch up on.
VERIFY_LOOKBACK_DAYS = 7


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

    # Extra forecast sources added after the table was first created.
    # ICON was previously only backfilled into historical_forecasts, so
    # live-collected rows lost it; both are now logged here every run.
    for col in ["icon_forecast_f REAL", "icon_error REAL",
                "wk_forecast_f REAL", "wk_error REAL",
                # Where actual_high_f came from: 'station' (settlement truth)
                # or 'era5' (reanalysis fallback, can differ by 1-3°F).
                # The sniper's validation gate only trusts station rows.
                "actual_source TEXT"]:
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
        target_date = datetime.now().strftime("%Y-%m-%d")

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

    # Get trained model prediction if available
    model_pred = None
    if all(v is not None for v in [gfs, ecmwf, blend]):
        try:
            from train_model import predict_with_trained_model
            model_path = str(BOT_DIR / get_model_path(city_key))
            dt = datetime.strptime(target_date, "%Y-%m-%d")
            result = predict_with_trained_model(
                gfs=gfs, ecmwf=ecmwf, blend=blend, icon=icon,
                weatherkit=wk,   # keep this recorded prediction on the same
                                 # inputs the live pipeline uses
                month=dt.month,
                day_of_year=dt.timetuple().tm_yday,
                model_path=model_path,
            )
            model_pred = result["predicted_high"]
        except Exception as e:
            print(f"    Trained model prediction failed: {e}")

    now = datetime.now(timezone.utc).isoformat()
    conn.execute("""
        INSERT OR IGNORE INTO daily_predictions (
            date, city, gfs_forecast_f, ecmwf_forecast_f, blend_forecast_f,
            icon_forecast_f, wk_forecast_f, model_prediction_f, recorded_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (target_date, city_key, gfs, ecmwf, blend, icon, wk, model_pred, now))
    conn.commit()

    forecasts_str = f"GFS={gfs}, ECMWF={ecmwf}, Blend={blend}, ICON={icon}"
    if wk is not None:
        forecasts_str += f", Apple={wk}"
    model_str = f", Model={model_pred}" if model_pred else ""
    print(f"    Recorded: {forecasts_str}{model_str}")


# ── Step 2: Verify yesterday's predictions ─────────────────────────

def verify_yesterday(conn: sqlite3.Connection, city_key: str,
                     check_date: str = None) -> dict | None:
    """
    Fetch the actual high for a past date and compare to predictions.
    Returns the verification result or None if not available.
    """
    city = CITIES[city_key]

    if check_date is None:
        check_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    # Check if we have a prediction to verify
    row = conn.execute("""
        SELECT gfs_forecast_f, ecmwf_forecast_f, blend_forecast_f,
               model_prediction_f, actual_high_f,
               icon_forecast_f, wk_forecast_f
        FROM daily_predictions
        WHERE date = ? AND city = ?
    """, (check_date, city_key)).fetchone()

    if row is None:
        print(f"  [{city_key}] No prediction recorded for {check_date}")
        return None

    if row[4] is not None:
        print(f"  [{city_key}] Already verified {check_date} (actual={row[4]}°F)")
        return {"actual": row[4], "already_done": True}

    gfs, ecmwf, blend, model_pred = row[0], row[1], row[2], row[3]
    icon, wk = row[5], row[6]

    # Fetch actual temperature — settlement-station truth first (this is
    # the number Kalshi settles on), ERA5 reanalysis only as a fallback.
    print(f"  [{city_key}] Fetching actual high for {check_date}...")
    actual = None
    actual_source = None
    try:
        from station_obs import fetch_station_daily_high
        actual = fetch_station_daily_high(city_key, check_date)
        if actual is not None:
            actual_source = "station"
            print(f"    (official station reading)")
    except Exception as e:
        print(f"    Station obs failed: {e}")

    if actual is None:
        try:
            actuals = fetch_actual_temps(city, check_date, check_date)
            actual = actuals.get(check_date)
            if actual is not None:
                actual_source = "era5"
                print(f"    WARNING: using ERA5 fallback — may differ "
                      f"from settlement station")
        except Exception as e:
            print(f"    Could not fetch actual temp: {e}")
            return None

    if actual is None:
        print(f"    Actual temp not yet available for {check_date}")
        return None

    # Compute errors
    gfs_err = (gfs - actual) if gfs is not None else None
    ecmwf_err = (ecmwf - actual) if ecmwf is not None else None
    blend_err = (blend - actual) if blend is not None else None
    model_err = (model_pred - actual) if model_pred is not None else None
    icon_err = (icon - actual) if icon is not None else None
    wk_err = (wk - actual) if wk is not None else None

    now = datetime.now(timezone.utc).isoformat()
    conn.execute("""
        UPDATE daily_predictions SET
            actual_high_f = ?,
            actual_source = ?,
            model_error = ?,
            gfs_error = ?,
            ecmwf_error = ?,
            blend_error = ?,
            icon_error = ?,
            wk_error = ?,
            verified_at = ?
        WHERE date = ? AND city = ?
    """, (actual, actual_source, model_err, gfs_err, ecmwf_err, blend_err,
          icon_err, wk_err, now, check_date, city_key))
    conn.commit()

    print(f"    Actual: {actual}°F")
    if gfs is not None:
        print(f"    GFS:    {gfs}°F (error: {gfs_err:+.1f}°F)")
    if ecmwf is not None:
        print(f"    ECMWF:  {ecmwf}°F (error: {ecmwf_err:+.1f}°F)")
    if blend is not None:
        print(f"    Blend:  {blend}°F (error: {blend_err:+.1f}°F)")
    if icon is not None:
        print(f"    ICON:   {icon}°F (error: {icon_err:+.1f}°F)")
    if wk is not None:
        print(f"    Apple:  {wk}°F (error: {wk_err:+.1f}°F)")
    if model_pred is not None:
        print(f"    Model:  {model_pred}°F (error: {model_err:+.1f}°F)")

    return {
        "date": check_date,
        "city": city_key,
        "actual": actual,
        "model_error": model_err,
        "gfs_error": gfs_err,
    }


# ── Step 3: Add to training set ────────────────────────────────────

def add_to_training_data(conn: sqlite3.Connection, city_key: str,
                          check_date: str = None):
    """
    Add a verified prediction to the historical_forecasts table
    so it's included in future model training.
    Also fetches weather features (wind, humidity, cloud cover) for the date.
    """
    if check_date is None:
        check_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    # Get the verified prediction
    row = conn.execute("""
        SELECT gfs_forecast_f, ecmwf_forecast_f, blend_forecast_f, actual_high_f,
               icon_forecast_f, wk_forecast_f
        FROM daily_predictions
        WHERE date = ? AND city = ? AND actual_high_f IS NOT NULL
    """, (check_date, city_key)).fetchone()

    if row is None:
        return False

    gfs, ecmwf, blend, actual, icon, wk = row

    # Check if already in historical_forecasts
    existing = conn.execute(
        "SELECT 1 FROM historical_forecasts WHERE date = ? AND city = ?",
        (check_date, city_key)
    ).fetchone()

    if existing:
        return False

    dt = datetime.strptime(check_date, "%Y-%m-%d")
    available = [v for v in [gfs, ecmwf, blend] if v is not None]
    spread = (max(available) - min(available)) if len(available) > 1 else 0

    # Fetch weather features for this date
    wind, humidity, cloud = None, None, None
    try:
        city = CITIES[city_key]
        weather = fetch_actual_temps(city, check_date, check_date, include_weather=True)
        wx = weather.get(check_date, {})
        if isinstance(wx, dict):
            wind = wx.get("wind")
            humidity = wx.get("humidity")
            cloud = wx.get("cloud")
            if any(v is not None for v in [wind, humidity, cloud]):
                print(f"    Weather: wind={wind}, humidity={humidity}, cloud={cloud}")
    except Exception as e:
        print(f"    Could not fetch weather features: {e}")

    init_historical_tables(conn)
    conn.execute("""
        INSERT OR REPLACE INTO historical_forecasts (
            date, city, actual_high_f,
            gfs_forecast_f, ecmwf_forecast_f, blend_forecast_f,
            gfs_error, ecmwf_error, blend_error,
            month, day_of_year, model_spread,
            wind_speed_max, humidity_mean, cloud_cover_mean,
            icon_forecast_f, icon_error, wk_forecast_f, wk_error
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        check_date, city_key, actual,
        gfs, ecmwf, blend,
        (gfs - actual) if gfs is not None else None,
        (ecmwf - actual) if ecmwf is not None else None,
        (blend - actual) if blend is not None else None,
        dt.month,
        dt.timetuple().tm_yday,
        spread,
        wind, humidity, cloud,
        icon, (icon - actual) if icon is not None else None,
        wk, (wk - actual) if wk is not None else None,
    ))
    conn.commit()

    print(f"  [{city_key}] Added {check_date} to training data")
    return True


# ── Step 4: Retrain ────────────────────────────────────────────────

def retrain_model(conn: sqlite3.Connection, city_key: str) -> bool:
    """Retrain the model with the expanded dataset."""
    count = conn.execute(
        "SELECT COUNT(*) FROM historical_forecasts WHERE city = ?",
        (city_key,)
    ).fetchone()[0]

    print(f"  [{city_key}] Retraining on {count} days of data...")

    try:
        model_data = train_and_evaluate(city_key, conn)
        print(f"  [{city_key}] New MAE: {model_data['train_mae']:.2f}°F "
              f"(σ={model_data['residual_std']:.2f}°F)")
        return True
    except Exception as e:
        print(f"  [{city_key}] Retraining failed: {e}")
        return False


# ── Full daily cycle ───────────────────────────────────────────────

def run_daily_learning(cities: list[str] = None) -> None:
    """
    Full daily learning cycle:
    1. Verify yesterday's predictions against actuals
    2. Add verified data to training set
    3. Retrain models
    4. Record today's forecasts for tomorrow's verification
    """
    if cities is None:
        cities = list(CITIES.keys())

    conn = sqlite3.connect(str(DB_PATH))
    init_prediction_log(conn)
    init_historical_tables(conn)

    today = datetime.now().strftime("%Y-%m-%d")

    print(f"\n{'=' * 60}")
    print(f"  DAILY LEARNING CYCLE — {today}")
    print(f"{'=' * 60}")

    # Step 1: Verify recent unverified days (not just yesterday — if the
    # machine was off for a day, that date previously stayed unverified
    # forever, silently starving live calibration and the sniper gate).
    print(f"\n--- Verifying recent unverified predictions ---")
    any_new_data = False
    for city in cities:
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

    # Step 2: Retrain if we got new data
    if any_new_data:
        print(f"\n--- Retraining models with new data ---")
        for city in cities:
            retrain_model(conn, city)
    else:
        print(f"\n  No new data to retrain on.")

    # Step 3: Record today's forecasts
    print(f"\n--- Recording today's forecasts ({today}) ---")
    for city in cities:
        record_forecasts(conn, city, today)

    # Also record tomorrow's if available
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"\n--- Recording tomorrow's forecasts ({tomorrow}) ---")
    for city in cities:
        record_forecasts(conn, city, tomorrow)

    conn.close()

    print(f"\n  Daily learning cycle complete.")


# ── Stats ──────────────────────────────────────────────────────────

def print_learning_stats(conn: sqlite3.Connection) -> None:
    """Print statistics about the daily learning process."""
    init_prediction_log(conn)

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
                   SUM(CASE WHEN wk_error IS NOT NULL THEN 1 ELSE 0 END)
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
        print(f"  Verified:   {verified} days")
        print(f"  Pending:    {total - verified} days")

        if verified > 0:
            model_mae = row[2]
            gfs_mae = row[3]
            ecmwf_mae = row[4]
            blend_mae = row[5]

            print(f"\n  Daily tracking MAE:")
            if model_mae is not None:
                print(f"    Trained model: {model_mae:.2f}°F")
            if gfs_mae is not None:
                print(f"    GFS:           {gfs_mae:.2f}°F")
            if ecmwf_mae is not None:
                print(f"    ECMWF:         {ecmwf_mae:.2f}°F")
            if blend_mae is not None:
                print(f"    Blend:         {blend_mae:.2f}°F")
            if row[8] is not None:
                print(f"    ICON:          {row[8]:.2f}°F")
            if row[9] is not None:
                print(f"    Apple:         {row[9]:.2f}°F  ({row[10]} days)")
                _print_weatherkit_status(city_key)

    # Training data growth
    print(f"\n  {'─' * 40}")
    print(f"  Training data size:")
    for city_key in CITIES:
        count = conn.execute(
            "SELECT COUNT(*) FROM historical_forecasts WHERE city = ?",
            (city_key,)
        ).fetchone()[0]
        print(f"    {city_key}: {count} days")


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
              AND model_error IS NOT NULL
        """, (city_key,)).fetchone()
        conn.close()
        if n:
            print(f"    → Same-day comparison ({n} day(s)): "
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
  (default)    Full daily cycle: verify → learn → retrain → record
  record       Just record today's forecasts
  learn        Just verify yesterday and retrain
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
        today = datetime.now().strftime("%Y-%m-%d")
        for city in (cities or list(CITIES.keys())):
            record_forecasts(conn, city, today)
        conn.close()

    elif args.command == "learn":
        conn = sqlite3.connect(str(DB_PATH))
        init_prediction_log(conn)
        init_historical_tables(conn)
        today = datetime.now().strftime("%Y-%m-%d")
        any_new = False
        for city in (cities or list(CITIES.keys())):
            pending = conn.execute("""
                SELECT date FROM daily_predictions
                WHERE city = ? AND actual_high_f IS NULL AND date < ?
                ORDER BY date DESC LIMIT ?
            """, (city, today, VERIFY_LOOKBACK_DAYS)).fetchall()
            for (check_date,) in pending:
                result = verify_yesterday(conn, city, check_date)
                if result and not result.get("already_done"):
                    if add_to_training_data(conn, city, check_date):
                        any_new = True
        if any_new:
            for city in (cities or list(CITIES.keys())):
                retrain_model(conn, city)
        conn.close()

    elif args.command == "stats":
        conn = sqlite3.connect(str(DB_PATH))
        print_learning_stats(conn)
        conn.close()
