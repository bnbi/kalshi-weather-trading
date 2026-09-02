"""
Weather Data Module
Fetches temperature forecasts from the National Weather Service (NWS) API.
No API key needed — completely free.

The key insight: NWS gives point forecasts, but their forecasts have known
error distributions. By modeling that error, we can generate probability
distributions for daily high temperatures that are likely better-calibrated
than what retail bettors use.
"""

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional

from http_util import get_with_retry


# ── City configurations ─────────────────────────────────────────────

@dataclass
class City:
    name: str
    kalshi_series: str
    lat: float
    lon: float
    timezone: str = "America/Chicago"
    nws_grid_url: Optional[str] = None  # cached after first lookup
    # Forecast sources that must never feed this city's model or spread.
    # Mirrored in training (the column is imputed from the other sources)
    # so live and training see the same inputs.
    excluded_sources: tuple = ()


CITIES = {
    "chicago": City(
        name="Chicago",
        kalshi_series="KXHIGHCHI",
        # Kalshi settles on NWS reading at Chicago Midway Airport (KMDW)
        lat=41.7868,
        lon=-87.7522,
        timezone="America/Chicago",
    ),
    "nyc": City(
        name="New York City",
        kalshi_series="KXHIGHNY",
        # Kalshi settles on NWS reading at Central Park (KNYC)
        lat=40.7789,
        lon=-73.9692,
        timezone="America/New_York",
    ),
    "miami": City(
        name="Miami",
        kalshi_series="KXHIGHMIA",
        # Kalshi settles on NWS reading at Miami Intl Airport (KMIA)
        lat=25.7959,
        lon=-80.2870,
        timezone="America/New_York",
    ),
    "denver": City(
        name="Denver",
        kalshi_series="KXHIGHDEN",
        # Kalshi settles on NWS reading at Denver Intl Airport (KDEN)
        lat=39.8561,
        lon=-104.6737,
        timezone="America/Denver",
    ),
    "austin": City(
        name="Austin",
        kalshi_series="KXHIGHAUS",
        # Kalshi settles on the NWS CLI report issued by AUS =
        # Austin-Bergstrom Intl (KAUS) — NOT Camp Mabry, which is the
        # other Austin climate station and reads differently.
        # Verified against the Kalshi series API settlement_sources.
        lat=30.1945,
        lon=-97.6699,
        timezone="America/Chicago",
    ),
    "la": City(
        name="Los Angeles",
        kalshi_series="KXHIGHLAX",
        # Kalshi settles on NWS reading at LAX (KLAX)
        lat=33.9425,
        lon=-118.4081,
        timezone="America/Los_Angeles",
        # Open-Meteo's ECMWF (0.25° IFS) cell for LAX sits inland of the
        # sea-breeze boundary and reads +6°F (annual) to +11°F (summer)
        # warm vs the KLAX sensor. It poisoned the spread (11°F -> +2.8°F
        # sigma inflation) and the LA model. Excluded everywhere.
        excluded_sources=("ecmwf",),
    ),
    "philly": City(
        name="Philadelphia",
        kalshi_series="KXHIGHPHIL",
        # Kalshi settles on NWS reading at Philadelphia Intl (KPHL)
        lat=39.8683,
        lon=-75.2311,
        timezone="America/New_York",
    ),
}


# ── NWS API interaction ─────────────────────────────────────────────

NWS_BASE = "https://api.weather.gov"
# NWS asks for identifying contact info in the User-Agent and throttles
# anonymous placeholders. The public repo is the contact point.
NWS_HEADERS = {
    "User-Agent": "(kalshi-weather-bot, github.com/bnbi/kalshi-weather-trading)",
    "Accept": "application/geo+json",
}

# Grid lookups are static per coordinate — cache them so every forecast
# call doesn't spend an extra round-trip on /points.
_GRID_CACHE: dict = {}


def get_grid_urls(city: City) -> dict:
    """
    Look up the NWS grid point for a city (cached per coordinates).
    Returns URLs for forecast endpoints.
    """
    key = (city.lat, city.lon)
    if key in _GRID_CACHE:
        return _GRID_CACHE[key]

    resp = get_with_retry(
        f"{NWS_BASE}/points/{city.lat},{city.lon}",
        headers=NWS_HEADERS,
        timeout=15,
    )
    props = resp.json()["properties"]

    grid = {
        "forecast": props["forecast"],
        "forecast_hourly": props["forecastHourly"],
        "forecast_grid_data": props["forecastGridData"],
        "grid_id": props["gridId"],
        "grid_x": props["gridX"],
        "grid_y": props["gridY"],
    }
    _GRID_CACHE[key] = grid
    return grid


def get_hourly_forecast(city: City) -> list[dict]:
    """
    Get hourly temperature forecast for a city.
    Returns list of {time, temperature, unit} dicts.
    """
    grid = get_grid_urls(city)
    resp = get_with_retry(grid["forecast_hourly"], headers=NWS_HEADERS, timeout=15)

    periods = resp.json()["properties"]["periods"]
    return [
        {
            "time": p["startTime"],
            "temperature": p["temperature"],
            "unit": p["temperatureUnit"],
            "short_forecast": p["shortForecast"],
        }
        for p in periods
    ]


def get_gridpoint_data(city: City) -> dict:
    """
    Get raw gridpoint forecast data — includes min/max temps
    and quantitative precipitation forecasts.
    This is more detailed than the public forecast.
    """
    grid = get_grid_urls(city)
    resp = get_with_retry(grid["forecast_grid_data"], headers=NWS_HEADERS, timeout=15)
    return resp.json()["properties"]


def get_daily_high_forecast(city: City, target_date: str = None) -> dict:
    """
    Get the forecasted daily high temperature for a specific date.

    target_date: 'YYYY-MM-DD' format. Defaults to tomorrow.

    Returns:
        {
            'city': str,
            'date': str,
            'forecast_high_f': int,
            'hourly_temps': list[int],
            'hourly_max': int,
            'source': 'NWS',
        }
    """
    if target_date is None:
        tomorrow = datetime.now(timezone.utc) + timedelta(days=1)
        target_date = tomorrow.strftime("%Y-%m-%d")

    # Get hourly forecast
    hourly = get_hourly_forecast(city)

    # Filter to target date and daytime hours
    target_temps = []
    for h in hourly:
        # Parse the ISO time string
        hour_time = datetime.fromisoformat(h["time"])
        hour_date = hour_time.strftime("%Y-%m-%d")

        if hour_date == target_date:
            target_temps.append(h["temperature"])

    # NOTE: no fallback to another date. Substituting today's temps when the
    # requested date has no hourly coverage (dates >~7 days out) silently
    # priced far-out contracts off today's weather. A missing date now
    # returns forecast_high_f=None and callers skip the contract.
    forecast_high = max(target_temps) if target_temps else None

    # Gridpoint maxTemperature is only a FALLBACK for when the hourly
    # forecast has no coverage for the date. It used to be fetched on every
    # call (one extra NWS request per forecast, never used when hourly data
    # existed), which added to the 429/5xx churn in the logs.
    official_high = None
    if forecast_high is None:
        try:
            grid_data = get_gridpoint_data(city)
            max_temps = grid_data.get("maxTemperature", {}).get("values", [])
            for entry in max_temps:
                # entries have 'validTime' like '2026-05-06T...' and 'value' in Celsius
                valid_time = entry.get("validTime", "")
                if target_date in valid_time and entry.get("value") is not None:
                    celsius = entry["value"]
                    official_high = round(celsius * 9 / 5 + 32)  # convert to F
                    break
        except Exception:
            pass  # gridpoint data sometimes fails

    # Use hourly max as primary source — it's more reliable.
    # The gridpoint maxTemperature sometimes returns stale or incorrect values.
    # Only use official_high if hourly data is unavailable. (is-None check:
    # a legitimate 0°F winter high is falsy and must not fall through.)
    best_forecast = forecast_high if forecast_high is not None else official_high

    # Assess how confident we are that the observed max is the final high.
    # Key factors:
    #   - Are there future hours with temps close to the current max? (second peak possible)
    #   - How far below the max are the remaining forecast hours?
    #   - How many hours remain in the day?
    peak_confidence = "low"  # 'low', 'medium', 'high'

    if target_temps and len(target_temps) >= 6:
        max_temp = max(target_temps)
        max_idx = target_temps.index(max_temp)
        remaining_temps = target_temps[max_idx + 1:]

        if remaining_temps:
            # How close do future temps get to the current max?
            future_max = max(remaining_temps)
            gap_from_peak = max_temp - future_max

            # How many hours remain after the peak?
            hours_remaining = len(remaining_temps)

            if gap_from_peak >= 8 and hours_remaining <= 6:
                # Temps are far below max and day is almost over
                peak_confidence = "high"
            elif gap_from_peak >= 5 and hours_remaining <= 10:
                # Reasonable gap, limited time for recovery
                peak_confidence = "medium"
            else:
                # Future temps are close to max — second peak possible
                peak_confidence = "low"
        else:
            # No remaining hours — we're at the last reading
            peak_confidence = "high"

    return {
        "city": city.name,
        "date": target_date,
        "forecast_high_f": best_forecast,
        "official_high_f": official_high,
        "hourly_high_f": forecast_high,
        "hourly_temps": target_temps,
        "peak_confidence": peak_confidence,
        "source": "NWS",
    }


# ── Historical forecast error modeling ──────────────────────────────

# NWS forecast errors for daily high temps (based on published verification data).
# These represent the standard deviation of (actual - forecast) in degrees F,
# as a function of forecast lead time in days.
# Source: NWS forecast verification statistics
# You should refine these by collecting your own data over time.

NWS_FORECAST_ERROR_STDDEV = {
    0: 2.0,   # same-day forecast: ~2°F std dev
    1: 3.0,   # 1-day out: ~3°F
    2: 4.0,   # 2-day out: ~4°F
    3: 5.0,   # 3-day out: ~5°F
    4: 5.5,
    5: 6.0,
    6: 6.5,
    7: 7.0,
}

# NWS has a slight warm bias in summer, cool bias in winter (varies by region).
# Start with 0 and refine as you collect data.
NWS_FORECAST_BIAS = {
    "chicago": 0.0,
    "nyc": 0.0,
    "miami": 0.0,
    "denver": 0.0,
    "austin": 0.0,
    "la": 0.0,
    "philly": 0.0,
}


def get_forecast_lead_days(target_date: str, tz_name: str = None) -> int:
    """
    Calculate how many days out the forecast is.

    tz_name: the city's IANA timezone. Pass it whenever the date refers to a
    local market day — "today" in UTC rolls over at ~5-8pm US local time, so
    the UTC default misclassifies evening runs (tomorrow looks like today).
    """
    from zoneinfo import ZoneInfo
    target = datetime.strptime(target_date, "%Y-%m-%d").date()
    now = datetime.now(ZoneInfo(tz_name)) if tz_name else datetime.now(timezone.utc)
    return (target - now.date()).days


def get_forecast_error_std(lead_days: int) -> float:
    """Get the expected standard deviation of forecast error."""
    if lead_days in NWS_FORECAST_ERROR_STDDEV:
        return NWS_FORECAST_ERROR_STDDEV[lead_days]
    # Extrapolate for longer lead times
    return min(7.0 + (lead_days - 7) * 0.5, 12.0)


# ── Convenience functions ───────────────────────────────────────────

def print_forecast_summary(city_key: str, target_date: str = None) -> None:
    """Print a quick forecast summary for a city."""
    city = CITIES[city_key]
    forecast = get_daily_high_forecast(city, target_date)

    lead_days = get_forecast_lead_days(forecast["date"], tz_name=city.timezone)
    error_std = get_forecast_error_std(lead_days)

    print(f"\n{'=' * 50}")
    print(f"  {city.name} — {forecast['date']}")
    print(f"{'=' * 50}")
    print(f"  Forecast high:    {forecast['forecast_high_f']}°F")
    print(f"  Hourly max:       {forecast['hourly_high_f']}°F")
    print(f"  Lead time:        {lead_days} day(s)")
    print(f"  Forecast error σ: ±{error_std}°F")
    print(f"  68% range:        {forecast['forecast_high_f'] - error_std:.0f}–{forecast['forecast_high_f'] + error_std:.0f}°F")
    print(f"  95% range:        {forecast['forecast_high_f'] - 2*error_std:.0f}–{forecast['forecast_high_f'] + 2*error_std:.0f}°F")

    if forecast["hourly_temps"]:
        print(f"\n  Hourly temps: {forecast['hourly_temps']}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch weather forecasts")
    parser.add_argument("city", choices=list(CITIES.keys()), help="City to forecast")
    parser.add_argument("--date", type=str, help="Target date (YYYY-MM-DD), default tomorrow")
    args = parser.parse_args()

    print_forecast_summary(args.city, args.date)
