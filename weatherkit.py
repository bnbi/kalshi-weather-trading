"""
Apple WeatherKit REST API Client

Apple's forecast is genuinely independent of the NWP models already in the
ensemble (GFS/ECMWF/ICON are all raw physics models; Open-Meteo's best_match
is a blend of them). Apple runs its own post-processing on top of a model mix
it inherited from Dark Sky, so its errors are only partially correlated with
the rest — which is exactly what makes an ensemble member useful.

Auth is an ES256 JWT signed with the .p8 key from your Apple Developer
account. The Apple-specific quirk is the `id` field in the JWT *header*:
it must be "<TeamID>.<ServiceID>", not just the key id.

Setup (config.py):
    WEATHERKIT_TEAM_ID          = "ABCDE12345"        # Apple Developer Team ID
    WEATHERKIT_SERVICE_ID       = "com.example.weather"  # Services ID identifier
    WEATHERKIT_KEY_ID           = "XYZ9876543"        # Key ID of the .p8
    WEATHERKIT_PRIVATE_KEY_PATH = "AuthKey_XYZ9876543.p8"

Self-test:
    python weatherkit.py            # all cities, next 7 days
    python weatherkit.py chicago    # one city
"""

from __future__ import annotations

import base64
import json
import os
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric import utils as asym_utils

from weather import CITIES, City

BASE_URL = "https://weatherkit.apple.com/api/v1"
TOKEN_TTL_SECONDS = 3600          # Apple caps JWT lifetime at 1 hour
TOKEN_REFRESH_MARGIN = 300        # refresh 5 min early

# Module-level token cache: (token, expires_at_epoch)
_TOKEN_CACHE: tuple[str, float] | None = None


class WeatherKitNotConfigured(RuntimeError):
    """Raised when config.py is missing WeatherKit credentials."""


# ── Configuration ───────────────────────────────────────────────────

def _load_config() -> dict:
    """Read WeatherKit credentials from config.py. Raises if incomplete."""
    try:
        import config
    except ImportError as e:
        raise WeatherKitNotConfigured(f"config.py not importable: {e}")

    fields = {
        "team_id": "WEATHERKIT_TEAM_ID",
        "service_id": "WEATHERKIT_SERVICE_ID",
        "key_id": "WEATHERKIT_KEY_ID",
        "key_path": "WEATHERKIT_PRIVATE_KEY_PATH",
    }
    out = {}
    missing = []
    for name, attr in fields.items():
        val = getattr(config, attr, None)
        if not val or str(val).startswith("your-"):
            missing.append(attr)
        out[name] = val
    if missing:
        raise WeatherKitNotConfigured(
            "missing in config.py: " + ", ".join(missing))

    # Resolve a relative key path against the bot directory, matching how
    # KALSHI_PRIVATE_KEY_PATH is used elsewhere.
    if not os.path.isabs(out["key_path"]):
        out["key_path"] = os.path.join(os.path.dirname(__file__), out["key_path"])
    if not os.path.exists(out["key_path"]):
        raise WeatherKitNotConfigured(f"key file not found: {out['key_path']}")
    return out


def is_configured() -> bool:
    """True if WeatherKit credentials are present and usable."""
    try:
        _load_config()
        return True
    except WeatherKitNotConfigured:
        return False


def is_enabled() -> bool:
    """
    True if WeatherKit should contribute to live forecasts.

    Gated by config.WEATHERKIT_ENABLED (default True when credentials
    exist) so it can be switched off without deleting credentials.
    """
    if not is_configured():
        return False
    try:
        import config
        return bool(getattr(config, "WEATHERKIT_ENABLED", True))
    except ImportError:
        return False


# ── JWT signing (ES256) ─────────────────────────────────────────────

def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _build_token(cfg: dict) -> tuple[str, float]:
    """Mint a fresh ES256 JWT. Returns (token, expiry_epoch)."""
    with open(cfg["key_path"], "rb") as f:
        key = serialization.load_pem_private_key(f.read(), password=None)

    if not isinstance(key, ec.EllipticCurvePrivateKey):
        raise WeatherKitNotConfigured(
            f"{cfg['key_path']} is not an EC private key — WeatherKit .p8 keys "
            "are P-256. Did you point this at the Kalshi RSA key by mistake?")

    issued = int(time.time())
    expires = issued + TOKEN_TTL_SECONDS

    header = {
        "alg": "ES256",
        "kid": cfg["key_id"],
        "id": f"{cfg['team_id']}.{cfg['service_id']}",
        "typ": "JWT",
    }
    payload = {
        "iss": cfg["team_id"],
        "sub": cfg["service_id"],
        "iat": issued,
        "exp": expires,
    }

    signing_input = ".".join([
        _b64url(json.dumps(header, separators=(",", ":")).encode()),
        _b64url(json.dumps(payload, separators=(",", ":")).encode()),
    ]).encode("ascii")

    # cryptography emits DER; JWS requires the raw r||s (P1363) form.
    der = key.sign(signing_input, ec.ECDSA(hashes.SHA256()))
    r, s = asym_utils.decode_dss_signature(der)
    raw_sig = r.to_bytes(32, "big") + s.to_bytes(32, "big")

    token = signing_input.decode("ascii") + "." + _b64url(raw_sig)
    return token, expires


def get_token() -> str:
    """Return a cached JWT, minting a new one when it is close to expiry."""
    global _TOKEN_CACHE
    now = time.time()
    if _TOKEN_CACHE and _TOKEN_CACHE[1] - TOKEN_REFRESH_MARGIN > now:
        return _TOKEN_CACHE[0]
    token, expires = _build_token(_load_config())
    _TOKEN_CACHE = (token, expires)
    return token


# ── Forecast fetch ──────────────────────────────────────────────────

def _c_to_f(celsius: float) -> float:
    return celsius * 9.0 / 5.0 + 32.0


def fetch_daily_highs(city: City, days: int = 7, timeout: int = 15) -> dict:
    """
    Fetch Apple's daily high forecast for a city.

    Returns {"YYYY-MM-DD": high_f} keyed by the city's LOCAL date, matching
    how Open-Meteo forecasts are keyed (and how Kalshi settles).

    WeatherKit reports temperatures in Celsius and stamps each day with a
    UTC `forecastStart`; passing `timezone` makes those stamps local
    midnight, so converting back to the city timezone recovers the date.
    """
    url = f"{BASE_URL}/weather/en_US/{city.lat}/{city.lon}"
    params = {
        "dataSets": "forecastDaily",
        "timezone": city.timezone,
        "countryCode": "US",
    }
    headers = {"Authorization": f"Bearer {get_token()}"}

    resp = requests.get(url, params=params, headers=headers, timeout=timeout)
    if resp.status_code == 401:
        raise RuntimeError(
            "WeatherKit rejected the token (401). Check that WEATHERKIT_TEAM_ID, "
            "WEATHERKIT_SERVICE_ID and WEATHERKIT_KEY_ID match the .p8, and that "
            "the Services ID is enabled for WeatherKit in the developer portal.")
    resp.raise_for_status()

    tz = ZoneInfo(city.timezone)
    forecasts = {}
    for day in resp.json().get("forecastDaily", {}).get("days", []):
        high_c = day.get("temperatureMax")
        start = day.get("forecastStart")
        if high_c is None or not start:
            continue
        # forecastStart is ISO8601 UTC, e.g. "2026-08-26T05:00:00Z"
        dt_utc = datetime.fromisoformat(start.replace("Z", "+00:00"))
        local_date = dt_utc.astimezone(tz).date().isoformat()
        forecasts[local_date] = round(_c_to_f(high_c), 1)

    return dict(sorted(forecasts.items())[:days])


def fetch_high_for_date(city: City, target_date: str) -> float | None:
    """Convenience wrapper: Apple's high for one date, or None."""
    try:
        return fetch_daily_highs(city).get(target_date)
    except Exception as e:
        print(f"  Warning: WeatherKit fetch failed for {city.name}: {e}")
        return None


# ── Self-test ───────────────────────────────────────────────────────

def _selftest(city_keys: list) -> int:
    try:
        cfg = _load_config()
    except WeatherKitNotConfigured as e:
        print(f"NOT CONFIGURED: {e}")
        print("\nAdd the four WEATHERKIT_* settings to config.py "
              "(see config.example.py).")
        return 1

    print(f"Team {cfg['team_id']} / Service {cfg['service_id']} / "
          f"Key {cfg['key_id']}")
    print(f"Key file: {cfg['key_path']}\n")

    failures = 0
    for key in city_keys:
        city = CITIES[key]
        try:
            highs = fetch_daily_highs(city)
        except Exception as e:
            print(f"  {key:<10} FAILED: {e}")
            failures += 1
            continue
        preview = "  ".join(f"{d[5:]}={v:.0f}F" for d, v in list(highs.items())[:5])
        print(f"  {key:<10} {len(highs)} days   {preview}")

    print("\nOK" if not failures else f"\n{failures} city(s) failed")
    return 1 if failures else 0


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Apple WeatherKit self-test")
    parser.add_argument("city", nargs="?", choices=list(CITIES.keys()),
                        help="City to test (default: all)")
    args = parser.parse_args()
    sys.exit(_selftest([args.city] if args.city else list(CITIES.keys())))
