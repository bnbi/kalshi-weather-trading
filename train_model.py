"""
Train Forecast Model
Learns optimal weights and corrections for each forecast source
using historical actual vs. predicted data.

The model learns:
    - Which forecast source is most accurate (and when)
    - Systematic biases (e.g., GFS runs warm in spring)
    - How model disagreement (spread) correlates with actual uncertainty
    - Seasonal patterns in forecast error

Training rows (see db_migrations.py for the provenance columns):
    - lead_ok = 1 only: every forecast column is a genuine lead-1 value.
      The 2021-2023 archive is mixed-lead (ICON lead-0, ECMWF missing) and
      made the models lean on ICON far beyond its live skill — excluded.
    - actual_source = 'station' only: NOAA GHCND, the number Kalshi settles
      on. Feed/ERA5 "actuals" are provisional and never training targets.
    - Live-collected rows carry the LIVE forecasts (the serving distribution).
    - Recency weighting (1-year half-life): source biases drift between
      years as NWP models are upgraded (2026 GFS/ECMWF ran ~1.5-2°F warmer
      than 2024-25 in Chicago/NYC), so recent rows count more.

Output: a trained model saved to forecast_model_<city>.pkl that the
prediction pipeline can load and use instead of the naive average.
"""

from __future__ import annotations

import os
import sqlite3
import pickle
import warnings
import numpy as np
import pandas as pd
from datetime import datetime

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

# Anchor to the module's directory — CWD-relative paths trained against a
# fresh empty DB (or saved models where the loaders never look) whenever a
# script ran from anywhere but the bot directory.
BOT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BOT_DIR, "kalshi_data.db")
MODEL_PATH = os.path.join(BOT_DIR, "forecast_model_chicago.pkl")

# Apple WeatherKit becomes a trained feature only once it covers enough of
# the training set. Expect roughly 6 months of live collection to clear this.
WK_MIN_TRAIN_ROWS = 180
WK_MIN_TRAIN_COVERAGE = 0.50

# Recency weighting of training rows (exponential, in days).
RECENCY_HALF_LIFE_DAYS = 365.0
RECENCY_MIN_WEIGHT = 0.05

# Error-trend alignment: the trend feature for target T is the mean error of
# T-2..T-4. A lead-1 forecast is issued on T-1; the freshest day whose high is
# complete AND verified by then is T-2 (verification runs before trading).
# shift(1) used T-1 — tomorrow's own eve, unknowable at issue time.
TREND_SHIFT_DAYS = 2

CORE_COLS = ["gfs_forecast_f", "ecmwf_forecast_f", "blend_forecast_f"]
SOURCE_COL = {"gfs": "gfs_forecast_f", "ecmwf": "ecmwf_forecast_f",
              "blend": "blend_forecast_f", "icon": "icon_forecast_f",
              "weatherkit": "wk_forecast_f"}

# Feature defaults when a live value is unavailable (also used when
# re-scoring stored inputs that predate the decision log).
DEFAULT_WIND, DEFAULT_HUMIDITY, DEFAULT_CLOUD = 10.0, 50.0, 50.0


def _wk_mode() -> str:
    """Current WeatherKit mode from config ('shadow' unless set)."""
    try:
        import config
        return getattr(config, "WEATHERKIT_MODE", "shadow")
    except ImportError:
        return "shadow"


def get_model_path(city: str) -> str:
    """Absolute path of the model file for a specific city."""
    return os.path.join(BOT_DIR, f"forecast_model_{city}.pkl")


def _static_exclusions(city: str) -> set:
    try:
        from weather import CITIES
        return set(CITIES[city].excluded_sources) if city in CITIES else set()
    except Exception:
        return set()


def load_training_data(conn: sqlite3.Connection, city: str) -> pd.DataFrame:
    """Load historical forecast data into a DataFrame (training-clean rows only)."""
    existing_cols = {r[1] for r in
                     conn.execute("PRAGMA table_info(historical_forecasts)")}
    icon_select = ", icon_forecast_f" if "icon_forecast_f" in existing_cols else ""
    wk_select = ", wk_forecast_f" if "wk_forecast_f" in existing_cols else ""

    # Provenance filters exist only after db_migrations.migrate_db has run.
    # Without them we cannot tell clean rows apart, so refuse rather than
    # silently train on mixed-lead / provisional data.
    if "lead_ok" not in existing_cols or "actual_source" not in existing_cols:
        raise ValueError("historical_forecasts lacks provenance columns — "
                         "run db_migrations.migrate_db first")

    df = pd.read_sql(f"""
        SELECT date, actual_high_f,
               gfs_forecast_f, ecmwf_forecast_f, blend_forecast_f,
               gfs_error, ecmwf_error, blend_error,
               month, day_of_year, model_spread,
               wind_speed_max, humidity_mean, cloud_cover_mean{icon_select}{wk_select}
        FROM historical_forecasts
        WHERE city = ?
          AND actual_high_f IS NOT NULL
          AND lead_ok = 1
          AND actual_source = 'station'
        ORDER BY date
    """, conn, params=(city,))

    df["date"] = pd.to_datetime(df["date"])

    # Ensure optional columns exist even on older DBs
    for col in ["wind_speed_max", "humidity_mean", "cloud_cover_mean",
                "icon_forecast_f", "wk_forecast_f"]:
        if col not in df.columns:
            df[col] = np.nan
    for col in ["gfs_forecast_f", "ecmwf_forecast_f", "blend_forecast_f",
                "icon_forecast_f", "wk_forecast_f"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── Excluded sources (mirror of the live pipeline) ──────────────
    # A source the live ensemble never uses for this city (e.g. LA's
    # inland-cell ECMWF) must not be a training input either; it is
    # imputed exactly like a missing source so the feature distributions
    # match between training and serving.
    for src in _static_exclusions(city):
        col = SOURCE_COL.get(src)
        if col in df.columns:
            df[col] = np.nan

    # ── NaN handling ────────────────────────────────────────────────
    # Live-collected rows sometimes have a missing forecast source (API
    # failure that morning). Impute a missing source with the row-mean of
    # the others — the live pipeline does the same (impute_core_sources).
    df = df[df["actual_high_f"].notna()].copy()

    row_mean = df[CORE_COLS].mean(axis=1)  # mean of available sources
    n_avail = df[CORE_COLS].notna().sum(axis=1)
    n_imputed = int(((n_avail < 3) & (n_avail >= 2)).sum())

    # Spread of the AVAILABLE core sources (recomputed — the stored column
    # may include a since-excluded source). Identical to the live
    # computation over imputed values, since an imputed value sits at the
    # mean and never widens the range.
    df["model_spread"] = df[CORE_COLS].max(axis=1) - df[CORE_COLS].min(axis=1)

    for col in CORE_COLS:
        df[col] = df[col].fillna(row_mean)

    # ICON is optional (4th source). Impute missing ICON with the mean of
    # the other sources so the feature is usable where the archive has gaps.
    df["icon_forecast_f"] = df["icon_forecast_f"].fillna(row_mean)

    # WeatherKit: flag real coverage BEFORE imputing, so build_features can
    # decide whether the column carries enough signal to be worth a feature.
    df["wk_present"] = df["wk_forecast_f"].notna()
    df["wk_forecast_f"] = df["wk_forecast_f"].fillna(row_mean)

    # Drop rows with fewer than two core sources (nothing to impute from)
    df = df[n_avail.loc[df.index] >= 2].copy()
    if n_imputed:
        print(f"  Note: imputed a core source on {n_imputed} row(s) "
              f"(missing or excluded for this city)")

    # Recompute per-source errors against the (possibly imputed) inputs
    # (predicted - actual)
    for fc_col, err_col in zip(CORE_COLS, ["gfs_error", "ecmwf_error", "blend_error"]):
        df[err_col] = df[fc_col] - df["actual_high_f"]

    df["model_spread"] = df["model_spread"].fillna(0)

    return df


def recency_weights(dates: pd.Series,
                    half_life_days: float = RECENCY_HALF_LIFE_DAYS) -> np.ndarray:
    """Exponential recency weights, 1.0 for the newest row."""
    age = (dates.max() - dates).dt.days.values.astype(float)
    w = 0.5 ** (age / half_life_days)
    return np.maximum(w, RECENCY_MIN_WEIGHT)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features for the model.

    Features:
        - Each model's raw forecast
        - Mean of all forecasts
        - Spread between models (disagreement signal)
        - Month (seasonal patterns)
        - Sine/cosine encoding of day-of-year (smooth seasonality)
        - Difference between each pair of models
        - Weather features (wind, humidity, cloud cover)
        - Recent error trends (rolling 3-day bias, lagged TREND_SHIFT_DAYS)
    """
    features = pd.DataFrame(index=df.index)

    # Raw forecasts
    features["gfs"] = df["gfs_forecast_f"]
    features["ecmwf"] = df["ecmwf_forecast_f"]
    features["blend"] = df["blend_forecast_f"]

    # Ensemble mean
    features["mean_forecast"] = df[["gfs_forecast_f", "ecmwf_forecast_f",
                                     "blend_forecast_f"]].mean(axis=1)

    # Model spread (how much they disagree)
    features["model_spread"] = df["model_spread"]

    # ICON (4th source) — only used when the column carries real signal
    if "icon_forecast_f" in df.columns and df["icon_forecast_f"].std() > 0.01:
        features["icon"] = df["icon_forecast_f"]
        features["icon_minus_ecmwf"] = (df["icon_forecast_f"]
                                        - df["ecmwf_forecast_f"])

    # Apple WeatherKit (5th source) — admitted as a feature only once it
    # covers a real share of the training rows. Below that, the column is
    # mostly row-mean imputation and would add noise rather than signal;
    # WeatherKit still contributes live through the ensemble's post-hoc
    # shrinkage in weather_ensemble._apply_weatherkit_blend().
    # HARD GATE: the wk feature additionally requires explicit blend mode.
    # In shadow mode Apple must not influence predictions ANYWHERE — and
    # a shadow-trained wk feature would also create training/serving skew:
    # the model would train on real Apple values while live prediction
    # (which excludes Apple from sources in shadow mode) served mean-
    # imputed placeholders in their place.
    wk_rows = int(df["wk_present"].sum()) if "wk_present" in df.columns else 0
    if (_wk_mode() == "blend"
            and wk_rows >= WK_MIN_TRAIN_ROWS
            and wk_rows / max(len(df), 1) >= WK_MIN_TRAIN_COVERAGE
            and df["wk_forecast_f"].std() > 0.01):
        features["wk"] = df["wk_forecast_f"]
        features["wk_minus_ecmwf"] = (df["wk_forecast_f"]
                                      - df["ecmwf_forecast_f"])

    # Pairwise differences (which model is warmer/cooler)
    features["gfs_minus_ecmwf"] = df["gfs_forecast_f"] - df["ecmwf_forecast_f"]
    features["ecmwf_minus_blend"] = df["ecmwf_forecast_f"] - df["blend_forecast_f"]

    # Only include gfs_minus_blend if it has variance (GFS and blend are often identical)
    gfs_blend_diff = df["gfs_forecast_f"] - df["blend_forecast_f"]
    if gfs_blend_diff.std() > 0.01:
        features["gfs_minus_blend"] = gfs_blend_diff

    # Seasonal features
    features["month"] = df["month"]
    day_frac = df["day_of_year"] / 365.25
    features["day_sin"] = np.sin(2 * np.pi * day_frac)
    features["day_cos"] = np.cos(2 * np.pi * day_frac)

    # Weather features (fill NaN with median so older data still works)
    if "wind_speed_max" in df.columns:
        wind = df["wind_speed_max"].copy()
        wind = wind.fillna(wind.median() if wind.notna().any() else DEFAULT_WIND)
        features["wind_speed"] = wind

    if "humidity_mean" in df.columns:
        hum = df["humidity_mean"].copy()
        hum = hum.fillna(hum.median() if hum.notna().any() else DEFAULT_HUMIDITY)
        features["humidity"] = hum

    if "cloud_cover_mean" in df.columns:
        cloud = df["cloud_cover_mean"].copy()
        cloud = cloud.fillna(cloud.median() if cloud.notna().any() else DEFAULT_CLOUD)
        features["cloud_cover"] = cloud

    # Recent error trends: rolling 3-day mean error for each model, lagged
    # so only errors verifiable at issue time are used (see TREND_SHIFT_DAYS).
    # The blend trend is omitted: Open-Meteo's best_match equals GFS on 96%
    # of US rows, and the near-duplicate pair let Ridge fit large offsetting
    # coefficients (+0.77 / -0.84 in Miami) that amplified the rare days
    # they differ.
    for src in ("gfs", "ecmwf"):
        err_col = f"{src}_error"
        if err_col in df.columns:
            err = df[err_col].fillna(0)
            features[f"{src}_error_trend_3d"] = (
                err.shift(TREND_SHIFT_DAYS).rolling(3, min_periods=1).mean().fillna(0))

    return features


def _sigma_cv_coverage(X_oof: np.ndarray, abs_resid: np.ndarray,
                       resid: np.ndarray, w: np.ndarray,
                       floor: float, cap: float) -> tuple:
    """
    Out-of-sample calibration check for the per-day σ model: fit on the
    earlier folds, score coverage on the later ones (time-ordered). The old
    check scored the σ model on the rows it was fitted on, which passes
    almost by construction.
    """
    tscv = TimeSeriesSplit(n_splits=3)
    within1, within2, n = 0, 0, 0
    for tr, te in tscv.split(X_oof):
        m = GradientBoostingRegressor(
            n_estimators=60, max_depth=2, learning_rate=0.05,
            min_samples_leaf=25, random_state=42)
        m.fit(X_oof[tr], abs_resid[tr], sample_weight=w[tr])
        sig = np.clip(1.2533 * m.predict(X_oof[te]), floor, cap)
        within1 += int(np.sum(np.abs(resid[te]) <= sig))
        within2 += int(np.sum(np.abs(resid[te]) <= 2 * sig))
        n += len(te)
    return (within1 / n, within2 / n, n) if n else (float("nan"), float("nan"), 0)


def train_and_evaluate(city: str, conn: sqlite3.Connection) -> dict:
    """
    Train the forecast model and evaluate on held-out data.

    Uses time-series cross-validation to avoid look-ahead bias.
    Returns the trained model and evaluation metrics.
    """
    df = load_training_data(conn, city)

    if len(df) < 30:
        raise ValueError(f"Not enough data for {city}: {len(df)} rows (need ≥30)")

    print(f"\nTraining on {len(df)} days of data for {city}")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")

    features = build_features(df)
    target = df["actual_high_f"]

    feature_names = list(features.columns)
    X = features.values
    y = target.values
    w = recency_weights(df["date"])

    # ── Compare models ──────────────────────────────────────────────

    # Baseline: simple average of all models
    baseline_pred = df[["gfs_forecast_f", "ecmwf_forecast_f", "blend_forecast_f"]].mean(axis=1).values
    baseline_mae = mean_absolute_error(y, baseline_pred)
    print(f"\n  Baseline (simple average) MAE: {baseline_mae:.2f}°F")

    # Individual model MAEs
    for model_name in ["gfs_forecast_f", "ecmwf_forecast_f", "blend_forecast_f"]:
        mae = mean_absolute_error(y, df[model_name].values)
        print(f"  {model_name:<25} MAE: {mae:.2f}°F")

    # ── Time-series cross-validation ────────────────────────────────

    tscv = TimeSeriesSplit(n_splits=5)

    models_to_try = {
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.1, max_iter=5000),
        "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            min_samples_leaf=10,
            random_state=42,
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            min_samples_leaf=10,
            random_state=42,
        ),
    }

    best_model_name = None
    best_mae = float("inf")
    best_residuals = None
    best_oof_mask = None

    for name, model in models_to_try.items():
        fold_maes = []
        all_preds = np.zeros(len(y))
        all_mask = np.zeros(len(y), dtype=bool)

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model.fit(X_train, y_train, sample_weight=w[train_idx])
            preds = model.predict(X_test)
            fold_mae = mean_absolute_error(y_test, preds)
            fold_maes.append(fold_mae)

            all_preds[test_idx] = preds
            all_mask[test_idx] = True

        avg_mae = np.mean(fold_maes)
        print(f"\n  {name}:")
        print(f"    CV MAE: {avg_mae:.2f}°F (folds: {[f'{m:.2f}' for m in fold_maes]})")

        if avg_mae < best_mae:
            best_mae = avg_mae
            best_model_name = name
            best_residuals = y[all_mask] - all_preds[all_mask]
            best_oof_mask = all_mask.copy()

    print(f"\n  Best model: {best_model_name} (MAE: {best_mae:.2f}°F)")
    print(f"  Improvement over baseline: {baseline_mae - best_mae:.2f}°F")

    # ── Train final model on all data ───────────────────────────────

    final_model = models_to_try[best_model_name]
    final_model.fit(X, y, sample_weight=w)

    # ── Calibrated uncertainty from residuals ──────────────────────
    # Use the CV residuals to compute a well-calibrated std
    # Also check if residuals are roughly normal (affects probability quality)
    residual_std = float(np.std(best_residuals)) if best_residuals is not None else 3.0
    residual_mean = float(np.mean(best_residuals)) if best_residuals is not None else 0.0

    # Calibration check: what fraction of actuals fall within 1σ and 2σ?
    if best_residuals is not None and len(best_residuals) > 10:
        within_1sigma = float(np.mean(np.abs(best_residuals) <= residual_std))
        within_2sigma = float(np.mean(np.abs(best_residuals) <= 2 * residual_std))
        print(f"\n  Calibration check:")
        print(f"    Residual mean: {residual_mean:+.2f}°F (actual - predicted, should be ~0)")
        print(f"    Residual std:  {residual_std:.2f}°F")
        print(f"    Within 1σ: {within_1sigma:.1%} (ideal: 68.3%)")
        print(f"    Within 2σ: {within_2sigma:.1%} (ideal: 95.4%)")

        # If residuals are fatter-tailed than normal, inflate std for safety
        if within_1sigma < 0.60:
            adj_factor = 0.683 / max(within_1sigma, 0.40)
            old_std = residual_std
            residual_std *= adj_factor
            print(f"    Adjusted σ: {old_std:.2f} → {residual_std:.2f}°F "
                  f"(fat tails detected, inflated for safer Kelly sizing)")

    # ── Per-day uncertainty model ───────────────────────────────────
    # A constant σ makes the model equally confident on a calm day and
    # ahead of a volatile front — which is exactly how phantom edge gets
    # created. Fit a second regressor on out-of-fold |residual| so σ can
    # vary with the situation (spread, season, wind, cloud...).
    # For a normal distribution E|X| = σ·sqrt(2/π), so σ ≈ 1.2533·E|X|.
    SIGMA_FLOOR, SIGMA_CAP = 1.0, 6.0
    sigma_model = None
    sigma_cv = None
    if best_residuals is not None and len(best_residuals) >= 100:
        X_oof = X[best_oof_mask]
        w_oof = w[best_oof_mask]
        abs_resid = np.abs(best_residuals)

        # Out-of-sample calibration gate (time-ordered folds)
        within_1s, within_2s, n_cv = _sigma_cv_coverage(
            X_oof, abs_resid, best_residuals, w_oof, SIGMA_FLOOR, SIGMA_CAP)
        sigma_cv = {"within_1s": within_1s, "within_2s": within_2s, "n": n_cv}
        print(f"\n  Per-day σ model (out-of-sample check on {n_cv} days):")
        print(f"    Within 1σ: {within_1s:.1%} (ideal: 68.3%)")
        print(f"    Within 2σ: {within_2s:.1%} (ideal: 95.4%)")
        if 0.60 <= within_1s <= 0.78:
            sigma_model = GradientBoostingRegressor(
                n_estimators=60, max_depth=2, learning_rate=0.05,
                min_samples_leaf=25, random_state=42,
            )
            sigma_model.fit(X_oof, abs_resid, sample_weight=w_oof)
            day_sigma = np.clip(1.2533 * sigma_model.predict(X_oof),
                                SIGMA_FLOOR, SIGMA_CAP)
            print(f"    Kept (in-sample range {day_sigma.min():.1f}–"
                  f"{day_sigma.max():.1f}°F)")
        else:
            print("    Poorly calibrated out of sample — discarding per-day σ, "
                  "keeping constant σ")

    # Feature importances (for Gradient Boosting / Random Forest)
    if hasattr(final_model, "feature_importances_"):
        print(f"\n  Feature importances:")
        importances = sorted(zip(feature_names, final_model.feature_importances_),
                             key=lambda x: x[1], reverse=True)
        for fname, imp in importances:
            bar = "█" * int(imp * 50)
            print(f"    {fname:<22} {imp:.3f} {bar}")

    # For linear models, show coefficients
    if hasattr(final_model, "coef_"):
        print(f"\n  {best_model_name} coefficients:")
        for fname, coef in sorted(zip(feature_names, final_model.coef_),
                                   key=lambda x: abs(x[1]), reverse=True):
            print(f"    {fname:<26} {coef:>+8.4f}")

    # ── Save model ──────────────────────────────────────────────────

    model_data = {
        "model": final_model,
        "model_name": best_model_name,
        "feature_names": feature_names,
        "city": city,
        "residual_std": residual_std,
        "residual_mean": residual_mean,
        "sigma_model": sigma_model,          # per-day σ (None if uncalibrated)
        "sigma_cv": sigma_cv,
        "sigma_floor": SIGMA_FLOOR,
        "sigma_cap": SIGMA_CAP,
        "train_mae": best_mae,
        "baseline_mae": baseline_mae,
        "n_training_days": len(df),
        "train_start": str(df["date"].min().date()),
        "train_end": str(df["date"].max().date()),
        "excluded_sources": sorted(_static_exclusions(city)),
        "trend_shift_days": TREND_SHIFT_DAYS,
        "trained_date": datetime.now().isoformat(),
    }

    save_path = get_model_path(city)
    tmp_path = save_path + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(model_data, f)
    os.replace(tmp_path, save_path)  # atomic: a reader never sees a partial file

    print(f"\n  Model saved to {save_path}")
    print(f"  Calibrated σ: {residual_std:.2f}°F")

    return model_data


# ── Prediction ──────────────────────────────────────────────────────

_MODEL_CACHE: dict = {}


def load_model(model_path: str) -> dict:
    """Load a model pickle, cached by (path, mtime)."""
    mtime = os.path.getmtime(model_path)
    cached = _MODEL_CACHE.get(model_path)
    if cached and cached[0] == mtime:
        return cached[1]
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
    _MODEL_CACHE[model_path] = (mtime, model_data)
    return model_data


def build_feature_dict(gfs: float, ecmwf: float, blend: float,
                       month: int, day_of_year: int,
                       icon: float = None, weatherkit: float = None,
                       wind_speed: float = None, humidity: float = None,
                       cloud_cover: float = None,
                       gfs_error_trend: float = 0.0,
                       ecmwf_error_trend: float = 0.0,
                       blend_error_trend: float = 0.0) -> dict:
    """Every feature build_features can produce, from raw inputs."""
    # ICON is optional — mirror training's imputation (mean of the others)
    icon_val = icon if icon is not None else (gfs + ecmwf + blend) / 3
    wk_val = weatherkit if weatherkit is not None else (gfs + ecmwf + blend) / 3
    mean_forecast = (gfs + ecmwf + blend) / 3
    model_spread = max(gfs, ecmwf, blend) - min(gfs, ecmwf, blend)
    day_frac = day_of_year / 365.25
    return {
        "gfs": gfs,
        "ecmwf": ecmwf,
        "blend": blend,
        "icon": icon_val,
        "mean_forecast": mean_forecast,
        "model_spread": model_spread,
        "gfs_minus_ecmwf": gfs - ecmwf,
        "gfs_minus_blend": gfs - blend,
        "ecmwf_minus_blend": ecmwf - blend,
        "icon_minus_ecmwf": icon_val - ecmwf,
        "wk": wk_val,
        "wk_minus_ecmwf": wk_val - ecmwf,
        "month": month,
        "day_sin": float(np.sin(2 * np.pi * day_frac)),
        "day_cos": float(np.cos(2 * np.pi * day_frac)),
        "wind_speed": wind_speed if wind_speed is not None else DEFAULT_WIND,
        "humidity": humidity if humidity is not None else DEFAULT_HUMIDITY,
        "cloud_cover": cloud_cover if cloud_cover is not None else DEFAULT_CLOUD,
        "gfs_error_trend_3d": gfs_error_trend,
        "ecmwf_error_trend_3d": ecmwf_error_trend,
        "blend_error_trend_3d": blend_error_trend,
    }


def predict_from_features(model_data: dict, features: dict) -> tuple:
    """(predicted_high, uncertainty_std) for a feature dict."""
    model = model_data["model"]
    feature_names = model_data["feature_names"]
    X = np.array([[features[name] for name in feature_names]])
    predicted = float(model.predict(X)[0])

    uncertainty = model_data["residual_std"]
    sigma_model = model_data.get("sigma_model")
    if sigma_model is not None:
        floor = model_data.get("sigma_floor", 1.5)
        cap = model_data.get("sigma_cap", 6.0)
        day_sigma = 1.2533 * float(sigma_model.predict(X)[0])
        uncertainty = float(np.clip(day_sigma, floor, cap))
    return predicted, uncertainty


def predict_with_trained_model(gfs: float, ecmwf: float, blend: float,
                                month: int, day_of_year: int,
                                model_path: str = None,
                                icon: float = None,
                                weatherkit: float = None,
                                wind_speed: float = None,
                                humidity: float = None,
                                cloud_cover: float = None,
                                gfs_error_trend: float = 0.0,
                                ecmwf_error_trend: float = 0.0,
                                blend_error_trend: float = 0.0) -> dict:
    """
    Use the trained model to predict the actual high temperature.

    Returns:
        {
            'predicted_high': float,
            'uncertainty_std': float,
            'model_name': str,
            'features': dict,          # exactly what went into the model
            'trained_date': str,       # model version
            'has_sigma_model': bool,
        }
    """
    if model_path is None:
        model_path = MODEL_PATH

    model_data = load_model(model_path)
    features = build_feature_dict(
        gfs, ecmwf, blend, month, day_of_year, icon=icon, weatherkit=weatherkit,
        wind_speed=wind_speed, humidity=humidity, cloud_cover=cloud_cover,
        gfs_error_trend=gfs_error_trend, ecmwf_error_trend=ecmwf_error_trend,
        blend_error_trend=blend_error_trend)
    predicted, uncertainty = predict_from_features(model_data, features)

    return {
        "predicted_high": round(predicted, 1),
        "uncertainty_std": uncertainty,
        "constant_std": model_data["residual_std"],
        "model_name": model_data["model_name"],
        "train_mae": model_data["train_mae"],
        "features": features,
        "trained_date": model_data.get("trained_date"),
        "has_sigma_model": model_data.get("sigma_model") is not None,
        # True when Apple's forecast is already a trained input — the
        # ensemble's post-hoc WeatherKit shrinkage must then be skipped
        # or Apple would be counted twice.
        "used_weatherkit_feature": "wk" in model_data["feature_names"],
    }


if __name__ == "__main__":
    import argparse

    from weather import CITIES

    parser = argparse.ArgumentParser(description="Train forecast error model")
    parser.add_argument("city", choices=list(CITIES.keys()), help="City to train on")
    parser.add_argument("--db", type=str, default=DB_PATH, help="Database path")
    parser.add_argument("--predict", nargs=3, type=float, metavar=("GFS", "ECMWF", "BLEND"),
                        help="Make a prediction given three forecasts")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)

    if args.predict:
        gfs, ecmwf, blend = args.predict
        today = datetime.now()
        result = predict_with_trained_model(
            gfs, ecmwf, blend,
            month=today.month,
            day_of_year=today.timetuple().tm_yday,
            model_path=get_model_path(args.city),
        )
        print(f"\n  Trained model prediction: {result['predicted_high']}°F")
        print(f"  Uncertainty: ±{result['uncertainty_std']:.1f}°F")
        print(f"  Model: {result['model_name']} (train MAE: {result['train_mae']:.2f}°F)")
    else:
        from db_migrations import migrate_db
        migrate_db(conn, verbose=True)
        train_and_evaluate(args.city, conn)

    conn.close()
