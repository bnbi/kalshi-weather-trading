"""
Multi-Source Weather Ensemble
Pulls forecasts from multiple weather models via Open-Meteo (free, no API key)
and combines them with NWS data to produce a better forecast.

Sources:
    1. NWS (existing weather.py) — official US forecast
    2. Open-Meteo / GFS — NOAA's Global Forecast System
    3. Open-Meteo / ECMWF — European model (often the best globally)
    4. Open-Meteo / Best Match — Open-Meteo's own blended forecast
    5. Open-Meteo / ICON — German DWD model
    6. Apple WeatherKit — Apple's post-processed forecast (optional; needs
       credentials in config.py). Not a raw NWP model, so its errors are
       the least correlated with the rest of the ensemble.

The ensemble forecast is more accurate than any single source because:
- Individual model errors are partially independent
- Averaging reduces variance (basic statistics)
- The spread between models gives us a better uncertainty estimate
"""

from __future__ import annotations

import os
import statistics
import requests
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta

from weather import CITIES, City, get_daily_high_forecast
import weatherkit


def _wk_config(name: str, default):
    """Read a WEATHERKIT_* setting from config.py, falling back to default."""
    try:
        import config
        return getattr(config, name, default)
    except ImportError:
        return default


OPEN_METEO_HEADERS = {
    "User-Agent": "(kalshi-weather-bot, github.com/bnbi/kalshi-weather-trading)",
}


@dataclass
class ForecastSource:
    """A single forecast from one source."""
    source: str
    high_f: float  # forecasted high in Fahrenheit


@dataclass
class EnsembleForecast:
    """Combined forecast from multiple sources."""
    city: str
    date: str
    sources: list = field(default_factory=list)
    ensemble_high_f: float = None
    ensemble_std: float = None
    model_spread: float = None
    used_trained_model: bool = False

    # Minimum uncertainty floor when we have NO live verification data.
    # The in-sample residual std is NOT the true forecast uncertainty.
    MIN_UNCERTAINTY_STD = 2.5

    # Floor when live verified errors ARE available. Live next-day error std
    # measured over May-Jun 2026 was 1.8-2.1°F per city, so we allow σ to
    # tighten below 2.5 — but never below this.
    MIN_LIVE_STD = 1.8

    # Live bias correction is clipped to this range (°F) to prevent a few
    # bad verification days from swinging predictions.
    MAX_BIAS_CORRECTION = 1.5

    # Apple WeatherKit has no forecast archive, so it cannot be backfilled
    # into the training set the way ICON was. Instead it is applied as a
    # post-hoc shrinkage toward Apple's number, with a single weight fitted
    # on the live verified rows. Defaults are overridable from config.py.
    WK_MIN_VERIFIED_DAYS = 45
    WK_MAX_WEIGHT = 0.35

    def compute(self):
        """
        Calculate ensemble statistics from individual sources.
        Uses the trained ML model if available, otherwise falls back to naive average.
        """
        if not self.sources:
            return

        # ── Sanity-check inputs: filter out obviously bad values ──
        # API failures sometimes return 0.0°F or other nonsense values.
        # If a source is wildly different from the others, drop it.
        highs = [s.high_f for s in self.sources]
        if len(highs) >= 2:
            median_high = sorted(highs)[len(highs) // 2]
            valid_sources = []
            for s in self.sources:
                # Drop any source that's more than 20°F from the median
                # (e.g., 0.0°F when others are 85°F is clearly an API failure)
                if abs(s.high_f - median_high) <= 20:
                    valid_sources.append(s)
                else:
                    print(f"  Warning: dropping {s.source} forecast ({s.high_f}°F) — "
                          f"too far from consensus ({median_high}°F), likely API failure")
            if valid_sources:
                self.sources = valid_sources
            # Recalculate after filtering
            highs = [s.high_f for s in self.sources]

        # Model spread: how much the models disagree
        if len(highs) >= 2:
            self.model_spread = max(highs) - min(highs)
        else:
            self.model_spread = 0

        # Try to use the trained ML model for a better prediction
        ml_result = self._predict_with_trained_model()
        if ml_result is not None:
            self.ensemble_high_f = ml_result["predicted_high"]

            # ── Live calibration (bias + σ) from verified predictions ──
            # The CV residual std from training is in-sample; the live
            # verified errors in daily_predictions are the ground truth for
            # how good the model actually is right now. Use them when we
            # have enough (≥10 verified days).
            live = self._get_live_calibration()
            if live is not None:
                bias, live_std = live
                # model_error is stored as (predicted - actual), so a negative
                # mean error means we under-predict: subtract it to correct.
                correction = max(-self.MAX_BIAS_CORRECTION,
                                 min(self.MAX_BIAS_CORRECTION, bias))
                self.ensemble_high_f = round(self.ensemble_high_f - correction, 1)
                # Take the LARGER of live-verified σ and the model's
                # (possibly day-specific) σ — conservative for Kelly sizing.
                base_std = max(live_std, ml_result["uncertainty_std"],
                               self.MIN_LIVE_STD)
            else:
                # No live data — fall back to the conservative floor.
                base_std = max(ml_result["uncertainty_std"], self.MIN_UNCERTAINTY_STD)

            # ── Apple WeatherKit shrinkage ──────────────────────────
            # Nudge the ML point forecast toward Apple's, by a weight
            # learned from how much Apple's disagreement has historically
            # predicted the model's own error. No-ops until there is
            # enough verified history — and is SKIPPED once the trained
            # model consumes Apple as a feature, so Apple never counts
            # twice in the same prediction.
            if not ml_result.get("used_weatherkit_feature"):
                self._apply_weatherkit_blend()

            spread_inflation = self.model_spread / 4.0  # more disagreement → more uncertainty
            self.ensemble_std = base_std + spread_inflation
            self.used_trained_model = True
            return

        # Fallback: naive average + spread-based uncertainty
        self.ensemble_high_f = statistics.mean(highs)
        base_uncertainty = 3.0
        spread_uncertainty = self.model_spread / 2.0
        self.ensemble_std = max(base_uncertainty, spread_uncertainty + 2.0)

    def _predict_with_trained_model(self) -> dict | None:
        """
        Try to use the trained Ridge/GB model from train_model.py.
        Returns prediction dict or None if model not available.
        """
        # Find the correct per-city model file
        bot_dir = os.path.dirname(__file__)
        city_key = self._get_city_key()
        model_path = os.path.join(bot_dir, f"forecast_model_{city_key}.pkl") if city_key else None

        if not model_path or not os.path.exists(model_path):
            return None

        # Extract GFS, ECMWF, blend (+ optional ICON) forecasts from sources
        source_map = {}
        for s in self.sources:
            name = s.source.lower()
            if "gfs" in name:
                source_map["gfs"] = s.high_f
            elif "ecmwf" in name:
                source_map["ecmwf"] = s.high_f
            elif "icon" in name:
                source_map["icon"] = s.high_f
            elif "weatherkit" in name or "apple" in name:
                source_map["weatherkit"] = s.high_f
            elif "best_match" in name or "blend" in name:
                source_map["blend"] = s.high_f

        # Need all three model forecasts for the trained model
        if not all(k in source_map for k in ("gfs", "ecmwf", "blend")):
            return None

        # Sanity-check: if any model forecast is unreasonably low or high,
        # it's likely an API failure. Skip the trained model to avoid garbage.
        values = list(source_map.values())
        if min(values) < 10 or max(values) > 140:
            print(f"  Warning: suspect forecast values {source_map}, skipping trained model")
            return None
        # If models disagree by more than 25°F, data is probably corrupted
        if max(values) - min(values) > 25:
            print(f"  Warning: extreme model disagreement {source_map}, skipping trained model")
            return None

        try:
            from train_model import predict_with_trained_model
            dt = datetime.strptime(self.date, "%Y-%m-%d")

            # Try to fetch weather features and error trends
            weather_kwargs = self._get_weather_features(city_key)

            result = predict_with_trained_model(
                gfs=source_map["gfs"],
                ecmwf=source_map["ecmwf"],
                blend=source_map["blend"],
                icon=source_map.get("icon"),  # optional 4th source
                weatherkit=source_map.get("weatherkit"),  # optional 5th source
                month=dt.month,
                day_of_year=dt.timetuple().tm_yday,
                model_path=model_path,
                **weather_kwargs,
            )
            return result
        except Exception as e:
            print(f"  Warning: trained model failed, using naive average: {e}")
            return None

    def _weatherkit_high(self) -> float | None:
        """Apple's forecast from this ensemble's sources, if present."""
        for s in self.sources:
            name = s.source.lower()
            if "weatherkit" in name or "apple" in name:
                return s.high_f
        return None

    def _apply_weatherkit_blend(self) -> None:
        """
        Shrink the ML prediction toward Apple's forecast by a learned weight.

        The weight w solves min_w Σ (actual - [pred + w·(apple - pred)])²
        over verified live rows — i.e. an OLS through the origin of the
        model's residual on Apple's disagreement. w = 0 means Apple adds
        nothing beyond the ML model; w > 0 means Apple systematically
        points in the direction the model was wrong.
        """
        wk = self._weatherkit_high()
        if wk is None or self.ensemble_high_f is None:
            return
        if _wk_config("WEATHERKIT_MODE", "shadow") != "blend":
            return

        weight = self._get_weatherkit_weight()
        if not weight:
            return

        self.ensemble_high_f = round(
            self.ensemble_high_f + weight * (wk - self.ensemble_high_f), 1)

    def _get_weatherkit_weight(self) -> float:
        """
        Fit the WeatherKit shrinkage weight from verified live predictions.
        Returns 0.0 when there is not yet enough data to estimate it.
        """
        city_key = self._get_city_key()
        if not city_key:
            return 0.0

        min_days = _wk_config("WEATHERKIT_MIN_VERIFIED_DAYS",
                              self.WK_MIN_VERIFIED_DAYS)
        max_weight = _wk_config("WEATHERKIT_MAX_WEIGHT", self.WK_MAX_WEIGHT)

        try:
            import sqlite3
            from pathlib import Path
            db_path = Path(os.path.dirname(__file__)) / "kalshi_data.db"
            conn = sqlite3.connect(str(db_path))
            rows = conn.execute("""
                SELECT model_prediction_f, wk_forecast_f, actual_high_f
                FROM daily_predictions
                WHERE city = ?
                  AND model_prediction_f IS NOT NULL
                  AND wk_forecast_f IS NOT NULL
                  AND actual_high_f IS NOT NULL
                ORDER BY date DESC LIMIT 120
            """, (city_key,)).fetchall()
            conn.close()
        except Exception:
            return 0.0  # column may not exist yet on an un-migrated DB

        if len(rows) < min_days:
            return 0.0

        # x = how far Apple sits from the model; y = the model's actual miss
        sxx = sxy = 0.0
        for pred, apple, actual in rows:
            x = apple - pred
            y = actual - pred
            sxx += x * x
            sxy += x * y

        if sxx < 1e-6:
            return 0.0  # Apple never disagrees — nothing to learn from

        return max(0.0, min(max_weight, sxy / sxx))

    def _get_live_calibration(self) -> tuple | None:
        """
        Compute (mean_bias, error_std) from the last 30 verified live
        predictions for this city. Returns None if fewer than 10 are
        available — in that case the caller uses conservative defaults.
        """
        city_key = self._get_city_key()
        if not city_key:
            return None
        try:
            import sqlite3
            from pathlib import Path
            db_path = Path(os.path.dirname(__file__)) / "kalshi_data.db"
            conn = sqlite3.connect(str(db_path))
            rows = conn.execute("""
                SELECT model_error FROM daily_predictions
                WHERE city = ? AND model_error IS NOT NULL
                ORDER BY date DESC LIMIT 30
            """, (city_key,)).fetchall()
            conn.close()
        except Exception:
            return None

        errors = [r[0] for r in rows if r[0] is not None]
        if len(errors) < 10:
            return None

        mean_bias = sum(errors) / len(errors)
        variance = sum((e - mean_bias) ** 2 for e in errors) / (len(errors) - 1)
        return mean_bias, variance ** 0.5

    def _get_weather_features(self, city_key: str) -> dict:
        """Fetch current weather features and recent error trends for prediction."""
        kwargs = {}

        # Try to get weather forecast (wind, humidity, cloud) for the target date
        try:
            city = CITIES[city_key]
            resp = requests.get("https://api.open-meteo.com/v1/forecast", params={
                "latitude": city.lat,
                "longitude": city.lon,
                "daily": "wind_speed_10m_max,relative_humidity_2m_mean,cloud_cover_mean",
                "forecast_days": 7,
                "timezone": city.timezone,
            }, headers=OPEN_METEO_HEADERS, timeout=15)
            if resp.ok:
                daily = resp.json().get("daily", {})
                dates = daily.get("time", [])
                if self.date in dates:
                    idx = dates.index(self.date)
                    winds = daily.get("wind_speed_10m_max", [])
                    humids = daily.get("relative_humidity_2m_mean", [])
                    clouds = daily.get("cloud_cover_mean", [])
                    if idx < len(winds) and winds[idx] is not None:
                        kwargs["wind_speed"] = winds[idx]
                    if idx < len(humids) and humids[idx] is not None:
                        kwargs["humidity"] = humids[idx]
                    if idx < len(clouds) and clouds[idx] is not None:
                        kwargs["cloud_cover"] = clouds[idx]
        except Exception:
            pass

        # Try to get recent error trends from daily_predictions table
        try:
            import sqlite3
            from pathlib import Path
            db_path = Path(os.path.dirname(__file__)) / "kalshi_data.db"
            conn = sqlite3.connect(str(db_path))
            rows = conn.execute("""
                SELECT gfs_error, ecmwf_error, blend_error
                FROM daily_predictions
                WHERE city = ? AND actual_high_f IS NOT NULL
                ORDER BY date DESC LIMIT 3
            """, (city_key,)).fetchall()
            conn.close()

            if rows:
                gfs_errs = [r[0] for r in rows if r[0] is not None]
                ecmwf_errs = [r[1] for r in rows if r[1] is not None]
                blend_errs = [r[2] for r in rows if r[2] is not None]
                if gfs_errs:
                    kwargs["gfs_error_trend"] = sum(gfs_errs) / len(gfs_errs)
                if ecmwf_errs:
                    kwargs["ecmwf_error_trend"] = sum(ecmwf_errs) / len(ecmwf_errs)
                if blend_errs:
                    kwargs["blend_error_trend"] = sum(blend_errs) / len(blend_errs)
        except Exception:
            pass

        return kwargs

    def _get_city_key(self) -> str | None:
        """Reverse-lookup city key from city name."""
        for key, city in CITIES.items():
            if city.name == self.city:
                return key
        return None


# ── Open-Meteo API calls ───────────────────────────────────────────

def fetch_open_meteo_forecast(city: City, endpoint: str = "/v1/forecast",
                               model_name: str = "best_match") -> dict:
    """
    Fetch daily max temperature from Open-Meteo.

    endpoint options:
        /v1/forecast  — Open-Meteo's best blend of models
        /v1/gfs       — NOAA GFS model
        /v1/ecmwf     — ECMWF IFS model
    """
    base_url = f"https://api.open-meteo.com{endpoint}"
    params = {
        "latitude": city.lat,
        "longitude": city.lon,
        "daily": "temperature_2m_max",
        "temperature_unit": "fahrenheit",
        "forecast_days": 7,
        "timezone": city.timezone,  # use city's local timezone for correct date alignment
    }

    resp = requests.get(base_url, params=params, headers=OPEN_METEO_HEADERS, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    # Parse the daily data into {date: high_temp} dict
    daily = data.get("daily", {})
    dates = daily.get("time", [])
    highs = daily.get("temperature_2m_max", [])

    forecasts = {}
    for date, high in zip(dates, highs):
        if high is not None:
            forecasts[date] = round(high, 1)

    return forecasts


def fetch_all_open_meteo(city: City) -> dict:
    """
    Fetch forecasts from all three Open-Meteo endpoints.
    Returns: {date: {model_name: high_temp_f}}
    """
    models = {
        "best_match": "/v1/forecast",
        "gfs": "/v1/gfs",
        "ecmwf": "/v1/ecmwf",
        "icon": "/v1/dwd-icon",
    }

    all_forecasts = {}
    for model_name, endpoint in models.items():
        try:
            forecasts = fetch_open_meteo_forecast(city, endpoint, model_name)
            for date, high in forecasts.items():
                if date not in all_forecasts:
                    all_forecasts[date] = {}
                all_forecasts[date][model_name] = high
        except Exception as e:
            print(f"  Warning: failed to fetch {model_name}: {e}")

    return all_forecasts


# ── Ensemble construction ───────────────────────────────────────────

def build_ensemble(city_key: str, target_date: str = None) -> EnsembleForecast:
    """
    Build an ensemble forecast by combining NWS + Open-Meteo models.

    This is the main function to call from the model.
    """
    city = CITIES[city_key]

    if target_date is None:
        tomorrow = datetime.now(timezone.utc) + timedelta(days=1)
        target_date = tomorrow.strftime("%Y-%m-%d")

    ensemble = EnsembleForecast(city=city.name, date=target_date)

    # Source 1: NWS
    try:
        nws = get_daily_high_forecast(city, target_date)
        if nws["forecast_high_f"] is not None:
            ensemble.sources.append(ForecastSource(
                source="NWS",
                high_f=nws["forecast_high_f"],
            ))
    except Exception as e:
        print(f"  Warning: NWS forecast failed: {e}")

    # Sources 2-4: Open-Meteo (GFS, ECMWF, best_match)
    try:
        open_meteo = fetch_all_open_meteo(city)
        date_forecasts = open_meteo.get(target_date, {})

        for model_name, high in date_forecasts.items():
            ensemble.sources.append(ForecastSource(
                source=f"OpenMeteo/{model_name}",
                high_f=high,
            ))
    except Exception as e:
        print(f"  Warning: Open-Meteo forecast failed: {e}")

    # Source 6: Apple WeatherKit (optional — needs credentials).
    # ONLY joins the ensemble in explicit "blend" mode. In "shadow" mode it
    # must stay out of `sources`: model_spread = max-min across sources
    # feeds sigma (spread/4 inflation), so a shadow source would silently
    # widen uncertainty and suppress trades before earning any trust.
    # Shadow-mode data still accumulates via daily_learner (wk_* columns),
    # which is what the 45-day blend gate is fitted on.
    if weatherkit.is_enabled() and _wk_config("WEATHERKIT_MODE", "shadow") == "blend":
        try:
            high = weatherkit.fetch_daily_highs(city).get(target_date)
            if high is not None:
                ensemble.sources.append(ForecastSource(
                    source="AppleWeatherKit",
                    high_f=high,
                ))
        except Exception as e:
            print(f"  Warning: WeatherKit forecast failed: {e}")

    # Compute ensemble statistics
    ensemble.compute()

    return ensemble


def print_ensemble(ensemble: EnsembleForecast) -> None:
    """Pretty-print an ensemble forecast."""
    print(f"\n{'=' * 60}")
    print(f"  ENSEMBLE FORECAST — {ensemble.city} — {ensemble.date}")
    print(f"{'=' * 60}")

    print(f"\n  Individual forecasts:")
    for s in ensemble.sources:
        print(f"    {s.source:<25} {s.high_f:.0f}°F")

    if ensemble.ensemble_high_f is not None:
        method = "ML model (Ridge)" if ensemble.used_trained_model else "naive average"
        print(f"\n  Method:              {method}")
        print(f"  Ensemble high:       {ensemble.ensemble_high_f:.1f}°F")
        print(f"  Model spread:        {ensemble.model_spread:.1f}°F")
        print(f"  Ensemble σ:          ±{ensemble.ensemble_std:.1f}°F")
        print(f"  68% range:           "
              f"{ensemble.ensemble_high_f - ensemble.ensemble_std:.0f}–"
              f"{ensemble.ensemble_high_f + ensemble.ensemble_std:.0f}°F")
        print(f"  95% range:           "
              f"{ensemble.ensemble_high_f - 2*ensemble.ensemble_std:.0f}–"
              f"{ensemble.ensemble_high_f + 2*ensemble.ensemble_std:.0f}°F")
    else:
        print("\n  No forecast data available!")


# ── Main ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Multi-source ensemble forecast")
    parser.add_argument("city", choices=list(CITIES.keys()), help="City to forecast")
    parser.add_argument("--date", type=str, help="Target date (YYYY-MM-DD)")
    args = parser.parse_args()

    ensemble = build_ensemble(args.city, args.date)
    print_ensemble(ensemble)
