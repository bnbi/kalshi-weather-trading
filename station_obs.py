"""
Settlement-Station Observations
Fetches the OFFICIAL daily high temperature at the exact stations Kalshi
settles on — not ERA5 reanalysis, which is a model grid average that can
differ from the station sensor by 1-3°F systematically.

Kalshi settlement stations:
    chicago -> Chicago Midway Airport   (NWS: KMDW, GHCND: USW00014819)
    nyc     -> Central Park             (NWS: KNYC, GHCND: USW00094728)
    miami   -> Miami Intl Airport       (NWS: KMIA, GHCND: USW00012839)

Two sources, in order of preference:
    1. NOAA NCEI GHCND daily summaries — the official climate record
       (same numbers as the NWS climate report Kalshi settles on).
       Lags 1-3 days behind real time.
    2. NWS observations API — near-real-time station obs; we take the max
       over the local calendar day. Used for recent dates GHCND doesn't
       have yet.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import requests

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None

from weather import CITIES

NCEI_URL = "https://www.ncei.noaa.gov/access/services/data/v1"
NWS_BASE = "https://api.weather.gov"
HEADERS = {"User-Agent": "(kalshi-weather-bot, contact@example.com)"}

# Station identifiers for each city's Kalshi settlement location
STATIONS = {
    "chicago": {"nws": "KMDW", "ghcnd": "USW00014819"},  # Midway
    "nyc": {"nws": "KNYC", "ghcnd": "USW00094728"},      # Central Park
    "miami": {"nws": "KMIA", "ghcnd": "USW00012839"},    # Miami Intl
    "denver": {"nws": "KDEN", "ghcnd": "USW00003017"},   # Denver Intl
    "austin": {"nws": "KAUS", "ghcnd": "USW00013904"},   # Austin-Bergstrom
    "la": {"nws": "KLAX", "ghcnd": "USW00023174"},       # LAX
    "philly": {"nws": "KPHL", "ghcnd": "USW00013739"},   # Philadelphia Intl
}


def fetch_ghcnd_daily_highs(city_key: str, start_date: str,
                            end_date: str) -> dict:
    """
    Fetch official daily TMAX from NOAA NCEI GHCND daily summaries.

    Returns {date_str: tmax_f}. Dates with no record are simply absent
    (GHCND typically lags 1-3 days behind real time).
    """
    station = STATIONS[city_key]["ghcnd"]
    resp = requests.get(NCEI_URL, params={
        "dataset": "daily-summaries",
        "stations": station,
        "startDate": start_date,
        "endDate": end_date,
        "dataTypes": "TMAX",
        "format": "json",
        "units": "standard",  # Fahrenheit
    }, headers=HEADERS, timeout=60)
    resp.raise_for_status()

    highs = {}
    for row in resp.json():
        date = row.get("DATE")
        tmax = row.get("TMAX")
        if date and tmax not in (None, ""):
            try:
                highs[date] = float(tmax)
            except (TypeError, ValueError):
                continue
    return highs


def fetch_nws_station_high(city_key: str, date: str) -> float | None:
    """
    Compute the daily high from NWS station observations for a local
    calendar day. Near-real-time; available within hours of midnight.

    Returns the max observed temperature in °F, or None if there are
    too few observations to trust (< 12 hourly obs).
    """
    station = STATIONS[city_key]["nws"]
    city = CITIES[city_key]

    if ZoneInfo is not None:
        tz = ZoneInfo(city.timezone)
        day_start = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=tz)
    else:  # pragma: no cover
        day_start = datetime.strptime(date, "%Y-%m-%d")
    day_end = day_start + timedelta(days=1)

    resp = requests.get(
        f"{NWS_BASE}/stations/{station}/observations",
        params={
            "start": day_start.isoformat(),
            "end": day_end.isoformat(),
            "limit": 500,
        },
        headers={**HEADERS, "Accept": "application/geo+json"},
        timeout=30,
    )
    resp.raise_for_status()

    temps_c = []
    for feature in resp.json().get("features", []):
        val = (feature.get("properties", {})
                      .get("temperature", {}) or {}).get("value")
        if val is not None:
            temps_c.append(float(val))

    # A full day has ~24+ obs; with fewer than 12 the max is unreliable
    # (could miss the afternoon peak entirely).
    if len(temps_c) < 12:
        return None

    return round(max(temps_c) * 9 / 5 + 32, 1)


def fetch_station_daily_high(city_key: str, date: str) -> float | None:
    """
    Get the settlement-station daily high for one date.
    Tries the official GHCND record first, then near-real-time NWS obs.
    """
    try:
        highs = fetch_ghcnd_daily_highs(city_key, date, date)
        if date in highs:
            return highs[date]
    except Exception as e:
        print(f"    GHCND lookup failed for {city_key} {date}: {e}")

    try:
        return fetch_nws_station_high(city_key, date)
    except Exception as e:
        print(f"    NWS obs lookup failed for {city_key} {date}: {e}")
        return None
