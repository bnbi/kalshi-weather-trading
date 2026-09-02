"""
Historical Data Pipeline
Pulls historical actual temperatures and what each weather model predicted
from Open-Meteo's free APIs. This data is used to train a forecast error model.

APIs used:
    1. Historical Weather API (/v1/archive) — actual observed temperatures
    2. Historical Forecast API — what GFS/ECMWF/blend predicted for each day

The output is a SQLite database with a clean table:
    date | actual_high | nws_forecast | gfs_forecast | ecmwf_forecast | blend_forecast | ...

NOTE: the historical-forecast API serves SAME-DAY (lead-0) forecasts and the
archive API serves ERA5 reanalysis "actuals". Rows written here are tagged
source='archive', lead_ok=0, actual_source='era5' and are NOT used for
training until backfill_lead1.py (lead-1 forecasts) and backfill_history.py
(official GHCND actuals) have upgraded them. See db_migrations.py.
"""

import sqlite3
import time
from datetime import datetime, timedelta
from weather import CITIES, City
from http_util import get_with_retry

OPEN_METEO_HEADERS = {
    "User-Agent": "(kalshi-weather-bot, github.com/bnbi/kalshi-weather-trading)",
}

import os
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kalshi_data.db")


def init_historical_tables(conn: sqlite3.Connection) -> None:
    """Create tables for historical forecast vs actual data."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS historical_forecasts (
            date TEXT NOT NULL,
            city TEXT NOT NULL,
            actual_high_f REAL,
            gfs_forecast_f REAL,
            ecmwf_forecast_f REAL,
            blend_forecast_f REAL,
            gfs_error REAL,
            ecmwf_error REAL,
            blend_error REAL,
            month INTEGER,
            day_of_year INTEGER,
            model_spread REAL,
            PRIMARY KEY (date, city)
        );
    """)
    conn.commit()

    # Add weather feature and extra-source columns if they don't exist yet.
    # icon_* is backfilled by backfill_history.py; wk_* (Apple WeatherKit)
    # accumulates live only — Apple publishes no forecast archive.
    for col in ["wind_speed_max REAL", "humidity_mean REAL", "cloud_cover_mean REAL",
                "icon_forecast_f REAL", "icon_error REAL",
                "wk_forecast_f REAL", "wk_error REAL",
                # Provenance (see db_migrations.py): which pipeline wrote the
                # row, its lead time, whether every source is lead-1, and
                # whether the actual is the official GHCND value.
                "source TEXT", "lead_days INTEGER", "lead_ok INTEGER",
                "actual_source TEXT"]:
        try:
            conn.execute(f"ALTER TABLE historical_forecasts ADD COLUMN {col}")
            conn.commit()
        except Exception:
            pass  # column already exists


def fetch_actual_temps(city: City, start_date: str, end_date: str,
                      include_weather: bool = False) -> dict:
    """
    Fetch actual observed daily high temperatures from Open-Meteo Historical API.

    If include_weather=True, also fetches wind speed, humidity, and cloud cover.
    Returns: {date_str: high_temp_f} or {date_str: {high, wind, humidity, cloud}}
    """
    daily_vars = "temperature_2m_max"
    if include_weather:
        daily_vars = "temperature_2m_max,wind_speed_10m_max,relative_humidity_2m_mean,cloud_cover_mean"

    resp = get_with_retry("https://archive-api.open-meteo.com/v1/archive", params={
        "latitude": city.lat,
        "longitude": city.lon,
        "daily": daily_vars,
        "temperature_unit": "fahrenheit",
        "start_date": start_date,
        "end_date": end_date,
        "timezone": city.timezone,
    }, headers=OPEN_METEO_HEADERS, timeout=30)
    data = resp.json()

    daily = data.get("daily", {})
    dates = daily.get("time", [])
    highs = daily.get("temperature_2m_max", [])

    if not include_weather:
        return {d: h for d, h in zip(dates, highs) if h is not None}

    # Return full weather data
    winds = daily.get("wind_speed_10m_max", [None] * len(dates))
    humids = daily.get("relative_humidity_2m_mean", [None] * len(dates))
    clouds = daily.get("cloud_cover_mean", [None] * len(dates))

    result = {}
    for i, d in enumerate(dates):
        if i < len(highs) and highs[i] is not None:
            result[d] = {
                "high": highs[i],
                "wind": winds[i] if i < len(winds) else None,
                "humidity": humids[i] if i < len(humids) else None,
                "cloud": clouds[i] if i < len(clouds) else None,
            }
    return result


def fetch_historical_forecasts(city: City, start_date: str, end_date: str,
                                model_name: str = None) -> dict:
    """
    Fetch what a model predicted for past dates.
    Uses Open-Meteo's historical forecast API.

    model_name: None for best_match, or 'gfs_seamless', 'ecmwf_ifs025', etc.
    Returns: {date_str: forecasted_high_f}
    """
    url = "https://historical-forecast-api.open-meteo.com/v1/forecast"

    params = {
        "latitude": city.lat,
        "longitude": city.lon,
        "daily": "temperature_2m_max",
        "temperature_unit": "fahrenheit",
        "start_date": start_date,
        "end_date": end_date,
        "timezone": city.timezone,
    }
    if model_name:
        params["models"] = model_name

    resp = get_with_retry(url, params=params, headers=OPEN_METEO_HEADERS, timeout=30)
    data = resp.json()

    daily = data.get("daily", {})
    dates = daily.get("time", [])
    highs = daily.get("temperature_2m_max", [])

    return {d: round(h, 1) for d, h in zip(dates, highs) if h is not None}


def fetch_all_historical(city_key: str, start_date: str, end_date: str,
                          conn: sqlite3.Connection):
    """
    Fetch actuals + all model forecasts for a date range and store in DB.
    Fetches in 3-month chunks to stay within API limits.
    """
    city = CITIES[city_key]
    init_historical_tables(conn)

    print(f"\nFetching historical data for {city.name}: {start_date} to {end_date}")

    # Parse dates
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    # Process in 90-day chunks (API may have limits on range)
    chunk_start = start
    total_rows = 0

    while chunk_start < end:
        chunk_end = min(chunk_start + timedelta(days=89), end)
        cs = chunk_start.strftime("%Y-%m-%d")
        ce = chunk_end.strftime("%Y-%m-%d")

        print(f"  Chunk: {cs} to {ce}...")

        # Fetch actual temps + weather features
        try:
            actuals_raw = fetch_actual_temps(city, cs, ce, include_weather=True)
            actuals = {d: v["high"] for d, v in actuals_raw.items()}
            weather_data = actuals_raw  # {date: {high, wind, humidity, cloud}}
            print(f"    Actuals: {len(actuals)} days (with weather features)")
        except Exception as e:
            print(f"    Error fetching actuals: {e}")
            actuals = {}
            weather_data = {}

        time.sleep(0.5)

        # Fetch model forecasts
        # model_name param: None = best_match, or Open-Meteo model identifiers
        models = {
            "gfs": "gfs_seamless",
            "ecmwf": "ecmwf_ifs025",
            "blend": None,  # best_match (default)
        }

        model_forecasts = {}
        for label, model_param in models.items():
            try:
                forecasts = fetch_historical_forecasts(city, cs, ce, model_param)
                model_forecasts[label] = forecasts
                print(f"    {label}: {len(forecasts)} days")
            except Exception as e:
                print(f"    Error fetching {label}: {e}")
                model_forecasts[label] = {}
            time.sleep(0.5)

        # Combine and store — require actual + at least one model
        all_dates = set(actuals.keys())

        for date_str in sorted(all_dates):
            actual = actuals.get(date_str)
            if actual is None:
                continue

            gfs = model_forecasts.get("gfs", {}).get(date_str)
            ecmwf = model_forecasts.get("ecmwf", {}).get(date_str)
            blend = model_forecasts.get("blend", {}).get(date_str)

            # Need at least one forecast
            available = [v for v in [gfs, ecmwf, blend] if v is not None]
            if not available:
                continue

            dt = datetime.strptime(date_str, "%Y-%m-%d")
            spread = (max(available) - min(available)) if len(available) > 1 else 0

            # Weather features for this date
            wx = weather_data.get(date_str, {})
            wind = wx.get("wind") if isinstance(wx, dict) else None
            humidity = wx.get("humidity") if isinstance(wx, dict) else None
            cloud = wx.get("cloud") if isinstance(wx, dict) else None

            # Never clobber a live-collected row: the live feed is the
            # serving distribution and is the training anchor for its dates.
            live = conn.execute(
                "SELECT 1 FROM historical_forecasts WHERE date = ? AND city = ? "
                "AND source = 'live'", (date_str, city_key)).fetchone()
            if live:
                continue

            conn.execute("""
                INSERT OR REPLACE INTO historical_forecasts (
                    date, city, actual_high_f,
                    gfs_forecast_f, ecmwf_forecast_f, blend_forecast_f,
                    gfs_error, ecmwf_error, blend_error,
                    month, day_of_year, model_spread,
                    wind_speed_max, humidity_mean, cloud_cover_mean,
                    source, lead_ok, actual_source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'archive', 0, 'era5')
            """, (
                date_str,
                city_key,
                actual,
                gfs, ecmwf, blend,
                (gfs - actual) if gfs is not None else None,
                (ecmwf - actual) if ecmwf is not None else None,
                (blend - actual) if blend is not None else None,
                dt.month,
                dt.timetuple().tm_yday,
                spread,
                wind, humidity, cloud,
            ))
            total_rows += 1

        conn.commit()
        chunk_start = chunk_end + timedelta(days=1)

    print(f"\n  Total: {total_rows} complete rows stored for {city.name}")
    return total_rows


def print_data_summary(conn: sqlite3.Connection, city_key: str) -> None:
    """Print summary statistics of the historical data."""
    rows = conn.execute("""
        SELECT COUNT(*),
               MIN(date), MAX(date),
               AVG(gfs_error), AVG(ecmwf_error), AVG(blend_error),
               AVG(ABS(gfs_error)), AVG(ABS(ecmwf_error)), AVG(ABS(blend_error))
        FROM historical_forecasts
        WHERE city = ?
    """, (city_key,)).fetchone()

    if not rows or rows[0] == 0:
        print(f"\n  No data for {city_key}")
        return

    count, min_date, max_date = rows[0], rows[1], rows[2]
    gfs_bias, ecmwf_bias, blend_bias = rows[3], rows[4], rows[5]
    gfs_mae, ecmwf_mae, blend_mae = rows[6], rows[7], rows[8]

    print(f"\n{'=' * 60}")
    print(f"  HISTORICAL DATA SUMMARY — {city_key}")
    print(f"{'=' * 60}")
    print(f"  Date range:    {min_date} to {max_date} ({count} days)")
    print(f"\n  {'Model':<15} {'Bias (°F)':<12} {'MAE (°F)':<12}")
    print(f"  {'-' * 38}")
    print(f"  {'GFS':<15} {gfs_bias:>+6.2f}      {gfs_mae:>6.2f}")
    print(f"  {'ECMWF':<15} {ecmwf_bias:>+6.2f}      {ecmwf_mae:>6.2f}")
    print(f"  {'Blend':<15} {blend_bias:>+6.2f}      {blend_mae:>6.2f}")

    # Monthly breakdown
    print(f"\n  Monthly MAE:")
    print(f"  {'Month':<10} {'GFS':<10} {'ECMWF':<10} {'Blend':<10}")
    print(f"  {'-' * 38}")
    monthly = conn.execute("""
        SELECT month,
               AVG(ABS(gfs_error)), AVG(ABS(ecmwf_error)), AVG(ABS(blend_error))
        FROM historical_forecasts
        WHERE city = ?
        GROUP BY month
        ORDER BY month
    """, (city_key,)).fetchall()

    month_names = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    for row in monthly:
        m, g, e, b = row
        print(f"  {month_names[m]:<10} {g:>6.2f}    {e:>6.2f}    {b:>6.2f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch historical forecast data")
    parser.add_argument("city", choices=list(CITIES.keys()), help="City to fetch")
    parser.add_argument("--start", type=str, default="2024-06-01",
                        help="Start date (YYYY-MM-DD). Default: 2024-06-01")
    parser.add_argument("--end", type=str, default=None,
                        help="End date (YYYY-MM-DD). Default: yesterday")
    parser.add_argument("--db", type=str, default=DB_PATH, help="Database path")
    parser.add_argument("--summary", action="store_true", help="Just print summary stats")
    args = parser.parse_args()

    if args.end is None:
        yesterday = datetime.now() - timedelta(days=1)
        args.end = yesterday.strftime("%Y-%m-%d")

    conn = sqlite3.connect(args.db)

    if args.summary:
        print_data_summary(conn, args.city)
    else:
        fetch_all_historical(args.city, args.start, args.end, conn)
        print_data_summary(conn, args.city)

    conn.close()
