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

Live recalibration (bias and sigma) is computed by RE-SCORING the current
model on the stored decision-time features of the last 30 days that have an
OFFICIAL (GHCND) actual — so a retrain never inherits the previous model's
bias, and a provisional feed value never masquerades as the truth.
"""

from __future__ import annotations

import json
import os
import sqlite3
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path

from weather import CITIES, City, get_daily_high_forecast, get_forecast_lead_days
from http_util import get_with_retry
import weatherkit


def _wk_config(name: str, default):
    """Read a setting from config.py, falling back to default."""
    try:
        import config
        return getattr(config, name, default)
    except ImportError:
        return default


OPEN_METEO_HEADERS = {
    "User-Agent": "(kalshi-weather-bot, github.com/bnbi/kalshi-weather-trading)",
}

BOT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = Path(BOT_DIR) / "kalshi_data.db"


def _ro_conn() -> sqlite3.Connection:
    """Read-only connection that never creates an empty DB as a side effect."""
    return sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)

# A forecast source whose LIVE mean error over recent official days exceeds
# this is dropped from the ensemble and imputed (safety net for the kind of
# grid-cell problem LA's ECMWF had). Static exclusions live in weather.CITIES.
LIVE_BIAS_GATE_F = 5.0
LIVE_BIAS_GATE_MIN_DAYS = 20

CORE_SOURCES = ("gfs", "ecmwf", "blend")


def excluded_sources(city_key: str) -> set:
    """Static (config) + dynamic (live-bias gate) exclusions for a city."""
    excl = set(CITIES[city_key].excluded_sources) if city_key in CITIES else set()
    excl |= _dynamic_exclusions(city_key)
    return excl


def _dynamic_exclusions(city_key: str) -> set:
    """
    Sources whose mean live error against official actuals is implausibly
    large. Computed from daily_predictions; empty on any error.
    """
    out = set()
    try:
        conn = _ro_conn()
        rows = conn.execute("""
            SELECT gfs_error, ecmwf_error, blend_error, icon_error
            FROM daily_predictions
            WHERE city = ? AND actual_source = 'station'
            ORDER BY date DESC LIMIT 60
        """, (city_key,)).fetchall()
        conn.close()
    except Exception:
        return out
    for idx, name in enumerate(("gfs", "ecmwf", "blend", "icon")):
        errs = [r[idx] for r in rows if r[idx] is not None]
        if len(errs) >= LIVE_BIAS_GATE_MIN_DAYS:
            bias = sum(errs) / len(errs)
            if abs(bias) > LIVE_BIAS_GATE_F:
                print(f"  Warning: [{city_key}] {name} live bias {bias:+.1f}°F "
                      f"over {len(errs)} days — excluded from ensemble")
                out.add(name)
    return out


def impute_core_sources(gfs, ecmwf, blend) -> tuple:
    """
    Fill a missing/excluded core source with the mean of the available ones —
    the same row-mean imputation train_model.load_training_data applies, so
    live features never drift from training features. Returns None-triple
    when fewer than two core sources are usable.
    """
    vals = {"gfs": gfs, "ecmwf": ecmwf, "blend": blend}
    avail = [v for v in vals.values() if v is not None]
    if len(avail) < 2:
        return None, None, None
    mean = sum(avail) / len(avail)
    return tuple(vals[k] if vals[k] is not None else mean for k in CORE_SOURCES)


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

    # Decision-time record (written to decision_log by the trading pipeline)
    raw_pred_f: float = None          # trained model output before corrections
    bias_applied_f: float = 0.0       # live bias subtracted from raw_pred_f
    ml_features: dict = None          # full feature dict used for the prediction
    model_trained_date: str = None    # pickle version that produced raw_pred_f
    sigma_source: str = None          # how ensemble_std was chosen
    live_n: int = 0                   # official days behind the live calibration
    excluded: list = field(default_factory=list)

    # Minimum uncertainty floor when we have NO live verification data.
    # The in-sample residual std is NOT the true forecast uncertainty.
    MIN_UNCERTAINTY_STD = 2.5

    # Floor when live verified errors ARE available. Live next-day error std
    # measured over May-Sep 2026 was 1.9-2.6°F per city, so we allow σ to
    # tighten below 2.5 — but never below this.
    MIN_LIVE_STD = 1.8

    # Live bias correction is clipped to this range (°F). Now that the live
    # window uses OFFICIAL actuals and the current model, a wider clip is
    # safe; measured live biases have reached ±2.5°F. Override with
    # config.MAX_BIAS_CORRECTION.
    MAX_BIAS_CORRECTION = 3.0

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
            self.raw_pred_f = ml_result["predicted_high"]
            self.ml_features = ml_result.get("features")
            self.model_trained_date = ml_result.get("trained_date")
            self.ensemble_high_f = self.raw_pred_f

            # ── Live calibration (bias + σ) from verified predictions ──
            # Errors of the CURRENT model on the last 30 days with official
            # actuals (re-scored from stored decision-time features).
            live = self._get_live_calibration()
            max_bias = float(_wk_config("MAX_BIAS_CORRECTION", self.MAX_BIAS_CORRECTION))
            if live is not None:
                bias, live_std, n_live = live
                self.live_n = n_live
                # error is (predicted - actual): a positive mean means we
                # run warm, so subtract it.
                correction = max(-max_bias, min(max_bias, bias))
                self.bias_applied_f = round(correction, 2)
                self.ensemble_high_f = round(self.raw_pred_f - correction, 1)
                base_std = max(live_std, ml_result["uncertainty_std"], self.MIN_LIVE_STD)
                self.sigma_source = (f"max(live σ {live_std:.2f} over {n_live}d, "
                                     f"model σ {ml_result['uncertainty_std']:.2f})")
            else:
                # No live data — fall back to the conservative floor.
                base_std = max(ml_result["uncertainty_std"], self.MIN_UNCERTAINTY_STD)
                self.sigma_source = (f"max(model σ {ml_result['uncertainty_std']:.2f}, "
                                     f"floor {self.MIN_UNCERTAINTY_STD})")

            # ── Apple WeatherKit shrinkage ──────────────────────────
            # Nudge the ML point forecast toward Apple's, by a weight
            # learned from how much Apple's disagreement has historically
            # predicted the model's own error. No-ops until there is
            # enough verified history — and is SKIPPED once the trained
            # model consumes Apple as a feature, so Apple never counts
            # twice in the same prediction.
            if not ml_result.get("used_weatherkit_feature"):
                self._apply_weatherkit_blend()

            # Spread inflation is only for CONSTANT-σ models. A day-specific
            # σ model already takes model_spread as an input, and the live σ
            # already reflects high-spread days; stacking spread/4 on top
            # double-counted disagreement (LA got +2.8°F/day from one bad
            # source) and manufactured fat-tail YES edge.
            if not ml_result.get("has_sigma_model"):
                base_std += self.model_spread / 4.0
                self.sigma_source += " + spread/4 (constant-σ model)"
            self.ensemble_std = round(base_std, 3)
            self.used_trained_model = True
            return

        # Fallback: naive average + spread-based uncertainty
        self.ensemble_high_f = statistics.mean(highs)
        base_uncertainty = 3.0
        spread_uncertainty = self.model_spread / 2.0
        self.ensemble_std = max(base_uncertainty, spread_uncertainty + 2.0)
        self.sigma_source = "naive average (no trained model)"

    def _source_map(self) -> dict:
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
        return source_map

    def _predict_with_trained_model(self) -> dict | None:
        """
        Try to use the trained Ridge/GB model from train_model.py.
        Returns prediction dict or None if model not available.
        """
        city_key = self._get_city_key()
        if not city_key:
            return None
        model_path = os.path.join(BOT_DIR, f"forecast_model_{city_key}.pkl")
        if not os.path.exists(model_path):
            return None

        source_map = self._source_map()
        gfs, ecmwf, blend = impute_core_sources(
            source_map.get("gfs"), source_map.get("ecmwf"), source_map.get("blend"))
        if gfs is None:
            return None  # fewer than two core sources — nothing to impute from

        # Sanity-check: if any model forecast is unreasonably low or high,
        # it's likely an API failure. Skip the trained model to avoid garbage.
        values = [v for v in (gfs, ecmwf, blend, source_map.get("icon"),
                              source_map.get("weatherkit")) if v is not None]
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
            weather_kwargs = self._get_weather_features(city_key)
            result = predict_with_trained_model(
                gfs=gfs, ecmwf=ecmwf, blend=blend,
                icon=source_map.get("icon"),          # optional 4th source
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
        Only official (station) actuals are used.
        """
        city_key = self._get_city_key()
        if not city_key:
            return 0.0

        min_days = _wk_config("WEATHERKIT_MIN_VERIFIED_DAYS",
                              self.WK_MIN_VERIFIED_DAYS)
        max_weight = _wk_config("WEATHERKIT_MAX_WEIGHT", self.WK_MAX_WEIGHT)

        try:
            conn = _ro_conn()
            rows = conn.execute("""
                SELECT model_prediction_f, wk_forecast_f, actual_high_f
                FROM daily_predictions
                WHERE city = ?
                  AND model_prediction_f IS NOT NULL
                  AND wk_forecast_f IS NOT NULL
                  AND actual_high_f IS NOT NULL
                  AND actual_source = 'station'
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
        (mean_bias, error_std, n) of the CURRENT model over the last 30 days
        that have an official actual, or None if fewer than 10 such days.
        See live_calibration_errors() for how errors are re-scored.
        """
        city_key = self._get_city_key()
        if not city_key:
            return None
        try:
            errors = live_calibration_errors(city_key)
        except Exception as e:
            print(f"  Warning: live calibration unavailable for {city_key}: {e}")
            return None
        if len(errors) < 10:
            return None
        mean_bias = sum(errors) / len(errors)
        variance = sum((e - mean_bias) ** 2 for e in errors) / (len(errors) - 1)
        return mean_bias, variance ** 0.5, len(errors)

    def _get_weather_features(self, city_key: str) -> dict:
        """Fetch current weather features and recent error trends for prediction."""
        return weather_feature_kwargs(city_key, self.date)

    def _get_city_key(self) -> str | None:
        """Reverse-lookup city key from city name."""
        for key, city in CITIES.items():
            if city.name == self.city:
                return key
        return None


# ── Feature helpers shared with daily_learner ─────────────────────

def weather_feature_kwargs(city_key: str, date: str) -> dict:
    """
    Weather features (wind, humidity, cloud) for `date` from Open-Meteo, plus
    per-source 3-day error trends from the most recent verified days.

    Trend alignment: train_model uses shift(2) — the trend for target T is
    the mean error of T-2..T-4, the freshest days that can be verified by
    the time a lead-1 forecast is issued (verification runs before trading).
    Live: the latest verified dates strictly before today.
    """
    kwargs = {}

    # Try to get weather forecast (wind, humidity, cloud) for the target date
    try:
        city = CITIES[city_key]
        resp = get_with_retry("https://api.open-meteo.com/v1/forecast", params={
            "latitude": city.lat,
            "longitude": city.lon,
            "daily": "wind_speed_10m_max,relative_humidity_2m_mean,cloud_cover_mean",
            "forecast_days": 7,
            "timezone": city.timezone,
        }, headers=OPEN_METEO_HEADERS, timeout=15, retries=2)
        daily = resp.json().get("daily", {})
        dates = daily.get("time", [])
        if date in dates:
            idx = dates.index(date)
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

    # Recent error trends from daily_predictions (verified days before today,
    # local to the city). ERA5-verified rows are excluded; provisional feed
    # rows are accepted here — the trend is a weak, short-lived feature and
    # waiting 1-3 days for GHCND would make it stale.
    try:
        from zoneinfo import ZoneInfo
        today_local = datetime.now(ZoneInfo(CITIES[city_key].timezone)).strftime("%Y-%m-%d")
        conn = _ro_conn()
        rows = conn.execute("""
            SELECT gfs_error, ecmwf_error, blend_error
            FROM daily_predictions
            WHERE city = ? AND actual_high_f IS NOT NULL AND date < ?
              AND (actual_source IS NULL OR actual_source != 'era5')
            ORDER BY date DESC LIMIT 3
        """, (city_key, today_local)).fetchall()
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


def ml_predict_for(city_key: str, date: str, gfs, ecmwf, blend,
                   icon=None, wk=None, fetch_weather: bool = True) -> dict | None:
    """
    The trained model's prediction for one (city, date) from raw source
    values — with the same exclusion/imputation and (optionally) the same
    weather/trend features the live pipeline uses. Used by daily_learner so
    the recorded model_prediction_f is comparable to what actually trades.
    Returns train_model.predict_with_trained_model's dict, or None.
    """
    model_path = os.path.join(BOT_DIR, f"forecast_model_{city_key}.pkl")
    if not os.path.exists(model_path):
        return None
    excl = excluded_sources(city_key)
    if "gfs" in excl: gfs = None
    if "ecmwf" in excl: ecmwf = None
    if "blend" in excl: blend = None
    if "icon" in excl: icon = None
    gfs, ecmwf, blend = impute_core_sources(gfs, ecmwf, blend)
    if gfs is None:
        return None
    from train_model import predict_with_trained_model
    dt = datetime.strptime(date, "%Y-%m-%d")
    kwargs = weather_feature_kwargs(city_key, date) if fetch_weather else {}
    return predict_with_trained_model(
        gfs=gfs, ecmwf=ecmwf, blend=blend, icon=icon, weatherkit=wk,
        month=dt.month, day_of_year=dt.timetuple().tm_yday,
        model_path=model_path, **kwargs)


def live_calibration_errors(city_key: str, limit: int = 30,
                            conn: sqlite3.Connection | None = None) -> list[float]:
    """
    Errors (predicted - official actual) of the CURRENT model over the most
    recent `limit` days with a GHCND actual, re-scored from:
      1. decision_log features recorded by the trading pipeline (exact), or
      2. the forecast inputs stored in daily_predictions with default
         weather/trend features (approximate) — for days before the
         decision log existed.
    Feed/ERA5-verified days are skipped: they are not the settlement number.
    """
    from train_model import load_model, predict_from_features, build_feature_dict

    model_path = os.path.join(BOT_DIR, f"forecast_model_{city_key}.pkl")
    if not os.path.exists(model_path):
        return []
    model_data = load_model(model_path)

    close = conn is None
    if conn is None:
        conn = _ro_conn()
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(daily_predictions)")}
        if "actual_source" not in cols:
            return []
        have_log = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='decision_log'"
        ).fetchone() is not None

        rows = conn.execute("""
            SELECT date, actual_high_f, gfs_forecast_f, ecmwf_forecast_f,
                   blend_forecast_f, icon_forecast_f, wk_forecast_f, lead_days
            FROM daily_predictions
            WHERE city = ? AND actual_high_f IS NOT NULL
              AND actual_source = 'station'
            ORDER BY date DESC LIMIT ?
        """, (city_key, limit)).fetchall()

        excl = excluded_sources(city_key) if rows else set()
        errors = []
        for (date, actual, gfs, ecmwf, blend, icon, wk, lead) in rows:
            features = None
            if have_log:
                r = conn.execute("""
                    SELECT features_json FROM decision_log
                    WHERE city = ? AND date = ? AND lead_days >= 1
                      AND features_json IS NOT NULL
                    ORDER BY CASE WHEN purpose = 'trade' THEN 0 ELSE 1 END,
                             issued_at DESC LIMIT 1
                """, (city_key, date)).fetchone()
                if r and r[0]:
                    try:
                        features = json.loads(r[0])
                    except ValueError:
                        features = None
            if features is None:
                if lead is not None and lead < 1:
                    continue  # a same-day recording is not a lead-1 error
                if "gfs" in excl: gfs = None
                if "ecmwf" in excl: ecmwf = None
                if "blend" in excl: blend = None
                if "icon" in excl: icon = None
                g, e, b = impute_core_sources(gfs, ecmwf, blend)
                if g is None:
                    continue
                dt = datetime.strptime(date, "%Y-%m-%d")
                features = build_feature_dict(g, e, b, dt.month,
                                              dt.timetuple().tm_yday,
                                              icon=icon, weatherkit=wk)
            try:
                pred, _ = predict_from_features(model_data, features)
            except Exception:
                continue
            errors.append(float(pred) - float(actual))
        return errors
    finally:
        if close:
            conn.close()


def record_decision(ensemble: EnsembleForecast, city_key: str,
                    purpose: str = "trade",
                    conn: sqlite3.Connection | None = None) -> None:
    """
    Persist what the pipeline actually used for (city, date) at this moment.
    This is the record the live calibration re-scores against, and the
    honest basis for any later "what did the model claim" analysis.
    """
    close = conn is None
    try:
        if conn is None:
            conn = sqlite3.connect(str(DB_PATH))
        from db_migrations import ensure_decision_log
        ensure_decision_log(conn)
        tz = CITIES[city_key].timezone
        lead = get_forecast_lead_days(ensemble.date, tz_name=tz)
        conn.execute("""
            INSERT INTO decision_log (
                issued_at, date, city, lead_days, model_trained_date,
                raw_pred_f, bias_applied_f, final_pred_f, sigma_f,
                model_spread_f, n_sources, features_json, used_trained_model,
                purpose
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now(timezone.utc).isoformat(), ensemble.date, city_key, lead,
            ensemble.model_trained_date, ensemble.raw_pred_f,
            ensemble.bias_applied_f, ensemble.ensemble_high_f, ensemble.ensemble_std,
            ensemble.model_spread, len(ensemble.sources),
            json.dumps(ensemble.ml_features) if ensemble.ml_features else None,
            1 if ensemble.used_trained_model else 0, purpose,
        ))
        conn.commit()
    except Exception as e:
        print(f"  Warning: could not record decision for {city_key} {ensemble.date}: {e}")
    finally:
        if close and conn is not None:
            conn.close()


# ── Open-Meteo API calls ───────────────────────────────────────────

def fetch_open_meteo_forecast(city: City, endpoint: str = "/v1/forecast",
                               model_name: str = "best_match") -> dict:
    """
    Fetch daily max temperature from Open-Meteo.

    endpoint options:
        /v1/forecast  — Open-Meteo's best blend of models
        /v1/gfs       — NOAA GFS model
        /v1/ecmwf     — ECMWF IFS model
        /v1/dwd-icon  — DWD ICON model
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

    resp = get_with_retry(base_url, params=params, headers=OPEN_METEO_HEADERS, timeout=15)
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


def fetch_all_open_meteo(city: City, skip: set = frozenset()) -> dict:
    """
    Fetch forecasts from all Open-Meteo endpoints (minus `skip`).
    Returns: {date: {model_name: high_temp_f}}
    """
    models = {
        "best_match": "/v1/forecast",
        "gfs": "/v1/gfs",
        "ecmwf": "/v1/ecmwf",
        "icon": "/v1/dwd-icon",
    }
    aliases = {"best_match": "blend"}

    all_forecasts = {}
    for model_name, endpoint in models.items():
        if model_name in skip or aliases.get(model_name) in skip:
            continue
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
        from zoneinfo import ZoneInfo
        tomorrow = datetime.now(ZoneInfo(city.timezone)) + timedelta(days=1)
        target_date = tomorrow.strftime("%Y-%m-%d")

    ensemble = EnsembleForecast(city=city.name, date=target_date)
    excl = excluded_sources(city_key)
    ensemble.excluded = sorted(excl)

    # Source 1: NWS
    if "nws" not in excl:
        try:
            nws = get_daily_high_forecast(city, target_date)
            if nws["forecast_high_f"] is not None:
                ensemble.sources.append(ForecastSource(
                    source="NWS",
                    high_f=nws["forecast_high_f"],
                ))
        except Exception as e:
            print(f"  Warning: NWS forecast failed: {e}")

    # Sources 2-5: Open-Meteo (GFS, ECMWF, best_match, ICON)
    try:
        open_meteo = fetch_all_open_meteo(city, skip=excl)
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
    # feeds sigma, so a shadow source would silently widen uncertainty and
    # suppress trades before earning any trust. Shadow-mode data still
    # accumulates via daily_learner (wk_* columns), which is what the
    # 45-day blend gate is fitted on.
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
    if ensemble.excluded:
        print(f"    (excluded sources: {', '.join(ensemble.excluded)})")

    if ensemble.ensemble_high_f is not None:
        method = "ML model" if ensemble.used_trained_model else "naive average"
        print(f"\n  Method:              {method}")
        if ensemble.raw_pred_f is not None:
            print(f"  Raw model output:    {ensemble.raw_pred_f:.1f}°F "
                  f"(live bias applied {ensemble.bias_applied_f:+.2f}°F over "
                  f"{ensemble.live_n} official days)")
        print(f"  Ensemble high:       {ensemble.ensemble_high_f:.1f}°F")
        print(f"  Model spread:        {ensemble.model_spread:.1f}°F")
        print(f"  Ensemble σ:          ±{ensemble.ensemble_std:.1f}°F  [{ensemble.sigma_source}]")
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
