"""
Tests for the model upgrade: settlement-station observations, ICON as a
4th source, per-day sigma model, and the backfill's actual-replacement
logic. All network calls are mocked.
"""

import json
import math
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import station_obs
import find_edge
from backfill_history import ensure_schema, fix_actuals_to_station


# ── station_obs ─────────────────────────────────────────────────────

def _mock_response(payload):
    resp = MagicMock()
    resp.json.return_value = payload
    resp.raise_for_status.return_value = None
    return resp


def test_ghcnd_parsing():
    payload = [
        {"DATE": "2026-07-01", "STATION": "USW00014819", "TMAX": "88.0"},
        {"DATE": "2026-07-02", "STATION": "USW00014819", "TMAX": "91.0"},
        {"DATE": "2026-07-03", "STATION": "USW00014819", "TMAX": ""},  # missing
    ]
    with patch.object(station_obs.requests, "get",
                      return_value=_mock_response(payload)) as mock_get:
        highs = station_obs.fetch_ghcnd_daily_highs(
            "chicago", "2026-07-01", "2026-07-03")

    assert highs == {"2026-07-01": 88.0, "2026-07-02": 91.0}
    params = mock_get.call_args.kwargs["params"]
    assert params["stations"] == "USW00014819"  # Midway, the settlement station
    assert params["units"] == "standard"        # Fahrenheit


def test_nws_station_high_computes_daily_max():
    # 24 hourly obs in °C peaking at 33.3°C (~92°F)
    temps_c = [22, 21, 21, 20, 20, 21, 22, 24, 26, 28, 30, 31,
               32, 33, 33.3, 33, 32, 30, 28, 27, 26, 25, 24, 23]
    payload = {"features": [
        {"properties": {"temperature": {"value": t}}} for t in temps_c
    ]}
    with patch.object(station_obs.requests, "get",
                      return_value=_mock_response(payload)):
        high = station_obs.fetch_nws_station_high("chicago", "2026-07-01")

    assert high == pytest.approx(33.3 * 9 / 5 + 32, abs=0.1)


def test_nws_station_high_rejects_sparse_data():
    # Only 5 observations — could easily miss the afternoon peak
    payload = {"features": [
        {"properties": {"temperature": {"value": t}}} for t in [20, 21, 22, 21, 20]
    ]}
    with patch.object(station_obs.requests, "get",
                      return_value=_mock_response(payload)):
        assert station_obs.fetch_nws_station_high("chicago", "2026-07-01") is None


def test_station_daily_high_prefers_ghcnd():
    with patch.object(station_obs, "fetch_ghcnd_daily_highs",
                      return_value={"2026-07-01": 90.0}), \
         patch.object(station_obs, "fetch_nws_station_high",
                      return_value=85.0):
        assert station_obs.fetch_station_daily_high("chicago", "2026-07-01") == 90.0


def test_station_daily_high_falls_back_to_nws():
    with patch.object(station_obs, "fetch_ghcnd_daily_highs",
                      return_value={}), \
         patch.object(station_obs, "fetch_nws_station_high",
                      return_value=85.0):
        assert station_obs.fetch_station_daily_high("chicago", "2026-07-01") == 85.0


# ── backfill: replacing ERA5 actuals with station truth ────────────

def _make_db_with_rows(tmp_path, n=5):
    conn = sqlite3.connect(tmp_path / "test.db")
    conn.execute("""
        CREATE TABLE historical_forecasts (
            date TEXT NOT NULL, city TEXT NOT NULL,
            actual_high_f REAL, gfs_forecast_f REAL, ecmwf_forecast_f REAL,
            blend_forecast_f REAL, gfs_error REAL, ecmwf_error REAL,
            blend_error REAL, month INTEGER, day_of_year INTEGER,
            model_spread REAL, wind_speed_max REAL, humidity_mean REAL,
            cloud_cover_mean REAL,
            PRIMARY KEY (date, city))
    """)
    base = datetime(2026, 6, 1)
    for i in range(n):
        d = (base + timedelta(days=i)).strftime("%Y-%m-%d")
        era5_actual = 80.0 + i          # ERA5 says 80,81,82...
        gfs = 81.0 + i
        conn.execute("""
            INSERT INTO historical_forecasts
            (date, city, actual_high_f, gfs_forecast_f, ecmwf_forecast_f,
             blend_forecast_f, gfs_error, ecmwf_error, blend_error,
             month, day_of_year, model_spread)
            VALUES (?, 'chicago', ?, ?, ?, ?, ?, ?, ?, 6, 152, 1.0)
        """, (d, era5_actual, gfs, gfs - 1, gfs, gfs - era5_actual,
              gfs - 1 - era5_actual, gfs - era5_actual))
    conn.commit()
    return conn


def test_fix_actuals_replaces_and_preserves_era5(tmp_path):
    conn = _make_db_with_rows(tmp_path)
    ensure_schema(conn)

    # Station read 2°F warmer than ERA5 on every day
    station = {(datetime(2026, 6, 1) + timedelta(days=i)).strftime("%Y-%m-%d"):
               82.0 + i for i in range(5)}
    with patch("backfill_history.fetch_ghcnd_daily_highs",
               return_value=station):
        stats = fix_actuals_to_station(conn, "chicago")

    assert stats["replaced"] == 5
    assert stats["era5_vs_station_bias"] == pytest.approx(-2.0)

    row = conn.execute("""
        SELECT actual_high_f, era5_high_f, gfs_error
        FROM historical_forecasts WHERE date = '2026-06-01'
    """).fetchone()
    assert row[0] == 82.0                      # station truth now the target
    assert row[1] == 80.0                      # ERA5 preserved
    assert row[2] == pytest.approx(81.0 - 82.0)  # error recomputed


def test_fix_actuals_is_idempotent(tmp_path):
    conn = _make_db_with_rows(tmp_path)
    ensure_schema(conn)
    station = {(datetime(2026, 6, 1) + timedelta(days=i)).strftime("%Y-%m-%d"):
               82.0 + i for i in range(5)}
    with patch("backfill_history.fetch_ghcnd_daily_highs",
               return_value=station):
        fix_actuals_to_station(conn, "chicago")
        fix_actuals_to_station(conn, "chicago")  # second run

    row = conn.execute("""
        SELECT actual_high_f, era5_high_f FROM historical_forecasts
        WHERE date = '2026-06-01'
    """).fetchone()
    # era5_high_f must NOT get overwritten by the new actual on re-run
    assert row == (82.0, 80.0)


# ── training: sigma model + ICON feature ────────────────────────────

def _make_training_db(tmp_path, n_days=400):
    """Synthetic but realistic training data with seasonal cycle + noise."""
    rng = np.random.default_rng(42)
    conn = sqlite3.connect(tmp_path / "train.db")
    conn.execute("""
        CREATE TABLE historical_forecasts (
            date TEXT NOT NULL, city TEXT NOT NULL,
            actual_high_f REAL, gfs_forecast_f REAL, ecmwf_forecast_f REAL,
            blend_forecast_f REAL, icon_forecast_f REAL,
            gfs_error REAL, ecmwf_error REAL, blend_error REAL,
            month INTEGER, day_of_year INTEGER, model_spread REAL,
            wind_speed_max REAL, humidity_mean REAL, cloud_cover_mean REAL,
            era5_high_f REAL, icon_error REAL,
            PRIMARY KEY (date, city))
    """)
    base = datetime(2025, 1, 1)
    for i in range(n_days):
        dt = base + timedelta(days=i)
        doy = dt.timetuple().tm_yday
        true_high = 55 + 30 * math.sin(2 * math.pi * (doy - 100) / 365.25)
        # Noisier days have higher spread — lets the sigma model learn
        day_noise = rng.choice([1.0, 3.0])
        actual = true_high + rng.normal(0, day_noise)
        gfs = true_high + rng.normal(0.5, day_noise)
        ecmwf = true_high + rng.normal(-0.3, day_noise * 0.8)
        icon = true_high + rng.normal(0.1, day_noise)
        blend = (gfs + ecmwf) / 2
        spread = max(gfs, ecmwf, blend, icon) - min(gfs, ecmwf, blend, icon)
        conn.execute("""
            INSERT INTO historical_forecasts
            (date, city, actual_high_f, gfs_forecast_f, ecmwf_forecast_f,
             blend_forecast_f, icon_forecast_f, gfs_error, ecmwf_error,
             blend_error, month, day_of_year, model_spread,
             wind_speed_max, humidity_mean, cloud_cover_mean)
            VALUES (?, 'chicago', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (dt.strftime("%Y-%m-%d"), actual, gfs, ecmwf, blend, icon,
              gfs - actual, ecmwf - actual, blend - actual,
              dt.month, doy, spread,
              float(rng.uniform(5, 25)), float(rng.uniform(30, 90)),
              float(rng.uniform(0, 100))))
    conn.commit()
    return conn


def test_training_produces_sigma_model_and_icon_feature(tmp_path, monkeypatch):
    import train_model
    conn = _make_training_db(tmp_path)
    monkeypatch.chdir(tmp_path)  # pkl gets written here

    model_data = train_model.train_and_evaluate("chicago", conn)

    assert "sigma_model" in model_data
    assert "icon" in model_data["feature_names"]

    result = train_model.predict_with_trained_model(
        gfs=85.0, ecmwf=84.0, blend=84.5, icon=85.5,
        month=7, day_of_year=197,
        model_path=str(tmp_path / "forecast_model_chicago.pkl"))
    assert 70 < result["predicted_high"] < 100
    assert 1.5 <= result["uncertainty_std"] <= 6.0

    # Without ICON the prediction must still work (imputation path)
    result2 = train_model.predict_with_trained_model(
        gfs=85.0, ecmwf=84.0, blend=84.5,
        month=7, day_of_year=197,
        model_path=str(tmp_path / "forecast_model_chicago.pkl"))
    assert 70 < result2["predicted_high"] < 100


def test_sigma_varies_by_day(tmp_path, monkeypatch):
    """If the sigma model survived calibration, high-disagreement days
    must get a higher sigma than calm days."""
    import train_model
    conn = _make_training_db(tmp_path)
    monkeypatch.chdir(tmp_path)
    model_data = train_model.train_and_evaluate("chicago", conn)

    if model_data["sigma_model"] is None:
        pytest.skip("sigma model discarded by calibration gate")

    path = str(tmp_path / "forecast_model_chicago.pkl")
    calm = train_model.predict_with_trained_model(
        gfs=85.0, ecmwf=85.0, blend=85.0, icon=85.0,
        month=7, day_of_year=197, model_path=path)
    volatile = train_model.predict_with_trained_model(
        gfs=89.0, ecmwf=82.0, blend=85.5, icon=88.0,
        month=7, day_of_year=197, model_path=path)
    assert volatile["uncertainty_std"] >= calm["uncertainty_std"]


def test_old_pkl_without_sigma_model_still_works(tmp_path):
    """Backward compat: pkls trained before this upgrade lack sigma keys."""
    import pickle
    import train_model
    from sklearn.linear_model import Ridge

    feature_names = ["gfs", "ecmwf", "blend", "mean_forecast"]
    model = Ridge().fit([[80, 80, 80, 80], [90, 90, 90, 90]], [80, 90])
    old_pkl = {
        "model": model, "model_name": "Ridge",
        "feature_names": feature_names, "city": "chicago",
        "residual_std": 2.5, "residual_mean": 0.0,
        "train_mae": 1.5, "baseline_mae": 2.0,
        "n_training_days": 100, "trained_date": "2026-01-01",
    }
    path = tmp_path / "old_model.pkl"
    with open(path, "wb") as f:
        pickle.dump(old_pkl, f)

    result = train_model.predict_with_trained_model(
        gfs=85.0, ecmwf=84.0, blend=84.5, month=7, day_of_year=197,
        model_path=str(path))
    assert result["uncertainty_std"] == 2.5  # falls back to constant sigma


# ── sniper v4 signal classes ────────────────────────────────────────

def test_v4_pass_through_requires_afternoon():
    from sniper import v4_signal_class
    # Classic pass-through: obs 75, bracket 76-77, forecast 80
    assert v4_signal_class(75.0, 80.0, 8.0, 76.0, 77.0) == "pass-through"
    # Same shape at 10h remaining (1pm) — v3's repeated loss — must not fire
    assert v4_signal_class(75.0, 80.0, 10.0, 76.0, 77.0) is None


def test_v4_pass_through_requires_overshoot():
    from sniper import v4_signal_class
    # Forecast only reaches bracket_high + 1 — could stall inside; no trade
    assert v4_signal_class(75.0, 78.0, 8.0, 76.0, 77.0) is None
    # Forecast clears by 2+ — qualifies
    assert v4_signal_class(75.0, 79.0, 8.0, 76.0, 77.0) == "pass-through"


def test_v4_already_passed_needs_sensor_margin():
    from sniper import v4_signal_class
    # The real 2026-07-08 loss: obs feed 89.6, bracket 88-89, official
    # settled 89 — a 0.6 margin is NOT enough
    assert v4_signal_class(89.6, 89.0, 7.0, 88.0, 89.0) is None
    # 1.5°F above the top survives feed-vs-official disagreement
    assert v4_signal_class(90.5, 90.0, 7.0, 88.0, 89.0) == "passed"


def test_v4_no_signal_when_obs_inside_bracket():
    from sniper import v4_signal_class
    # Obs already inside the bracket: NO is a live coin-flip, never fire
    assert v4_signal_class(76.5, 80.0, 8.0, 76.0, 77.0) is None


def test_v4_prob_cap():
    from sniper import V4_PROB_CAP
    assert V4_PROB_CAP <= 0.95  # v3 claimed 97-99% and won 36% — never again


# ── city expansion consistency ──────────────────────────────────────

def test_all_cities_have_station_ids():
    """Every configured city must have settlement-station IDs everywhere
    they're needed — a missing entry would crash verification or the sniper."""
    from weather import CITIES, NWS_FORECAST_BIAS
    import sniper
    for city_key in CITIES:
        assert city_key in station_obs.STATIONS, f"station_obs missing {city_key}"
        assert station_obs.STATIONS[city_key]["ghcnd"].startswith("USW"), city_key
        assert city_key in sniper.STATIONS, f"sniper missing {city_key}"
        assert city_key in NWS_FORECAST_BIAS, f"bias table missing {city_key}"


def test_scheduler_only_trades_cities_with_models(tmp_path, monkeypatch):
    """A configured-but-untrained city must never enter the trading rotation."""
    import scheduler
    monkeypatch.setattr(scheduler, "BOT_DIR", tmp_path)
    (tmp_path / "forecast_model_chicago.pkl").write_bytes(b"x")
    (tmp_path / "forecast_model_denver.pkl").write_bytes(b"x")
    assert scheduler.tradeable_cities() == ["chicago", "denver"]


# ── Kalshi CreateOrder V2 migration ─────────────────────────────────

def _client_with_mock_session():
    from kalshi_client import KalshiClient
    client = KalshiClient(api_key_id="test", private_key_path="")
    client._sign_request = lambda m, p: {}  # skip RSA for tests
    resp = MagicMock()
    resp.status_code = 201
    resp.json.return_value = {"order_id": "ord-1", "client_order_id": "c-1",
                              "fill_count": "1.00", "remaining_count": "0.00",
                              "ts_ms": 1}
    resp.raise_for_status.return_value = None
    client.session = MagicMock()
    client.session.request.return_value = resp
    return client


def test_buy_yes_maps_to_bid():
    client = _client_with_mock_session()
    client.create_order(ticker="KXHIGHCHI-26AUG17-B80.5", side="yes",
                        action="buy", count=2, yes_price=35,
                        client_order_id="c-1")
    body = client.session.request.call_args.kwargs["json"]
    assert body["side"] == "bid"
    assert body["price"] == "0.35"
    assert body["count"] == "2"
    assert body["time_in_force"] == "good_till_canceled"
    url = client.session.request.call_args.kwargs["url"]
    assert url.endswith("/portfolio/events/orders")


def test_buy_no_maps_to_ask_at_one_minus_price():
    client = _client_with_mock_session()
    client.create_order(ticker="KXHIGHCHI-26AUG17-B80.5", side="no",
                        action="buy", count=1, no_price=70)
    body = client.session.request.call_args.kwargs["json"]
    # Buying NO at 70c == selling (ask) YES at 30c
    assert body["side"] == "ask"
    assert body["price"] == "0.30"


def test_v2_response_normalized_for_legacy_callers():
    client = _client_with_mock_session()
    resp = client.create_order(ticker="T", side="yes", action="buy",
                               count=1, yes_price=50)
    # trader.py reads response["order"]["order_id"]
    assert resp["order"]["order_id"] == "ord-1"


def test_sell_orders_rejected():
    client = _client_with_mock_session()
    with pytest.raises(ValueError):
        client.create_order(ticker="T", side="yes", action="sell",
                            count=1, yes_price=50)


# ── percentage-based sizing (no dollar caps) ────────────────────────

def test_pct_sizing_full_pipeline_dry_run(monkeypatch):
    """End-to-end through run_trading_pipeline: % caps must be computed
    from the bankroll AFTER it's known, and sizing must scale with it."""
    import trader

    sig = find_edge.TradeSignal(
        ticker="KXHIGHCHI-99DEC31-B85.5", side="no", action="buy",
        model_prob=0.78, market_price=0.70, edge=0.06,
        expected_value=0.06, description="test")

    monkeypatch.setattr(trader, "get_market_prices", lambda s: [{"ticker": "x"}])
    monkeypatch.setattr(trader, "predict_all_for_city", lambda c, m: [object()])
    monkeypatch.setattr(trader, "calculate_edge", lambda p, m, min_edge: [sig])
    monkeypatch.setattr(trader, "filter_tomorrow_only", lambda s: s)

    results = trader.run_trading_pipeline("chicago", dry_run=True)

    assert len(results) == 1
    order = results[0].order
    # $100 dry-run bankroll, quarter-Kelly f=0.067 -> $6.68 -> 9 contracts,
    # clipped by the 5-contract liquidity guard -> 5 x 70c = $3.50
    assert order.contracts == 5
    assert order.cost_dollars == pytest.approx(3.50)
    # position must respect the 8% ceiling of the dry-run bankroll
    assert order.cost_dollars <= 100.0 * 0.08


def test_pct_sizing_scales_with_bankroll():
    """Same signal, 10x bankroll -> larger allocation (capped by contracts),
    proving no fixed dollar cap survives anywhere in sizing."""
    import trader
    sig = find_edge.TradeSignal(
        ticker="T", side="no", action="buy", model_prob=0.78,
        market_price=0.70, edge=0.06, expected_value=0.06, description="t")

    small = trader.size_orders([sig], bankroll=50, kelly_fraction=0.25,
                               max_position_dollars=50 * 0.08, max_contracts=100,
                               max_total_dollars=50 * 0.25, max_positions=6)
    large = trader.size_orders([sig], bankroll=500, kelly_fraction=0.25,
                               max_position_dollars=500 * 0.08, max_contracts=100,
                               max_total_dollars=500 * 0.25, max_positions=6)
    # The invariant: allocation as a FRACTION of bankroll is the same at
    # any scale (~quarter-Kelly f=6.7% here), modulo integer-contract
    # rounding, which bites harder at small bankrolls.
    assert 0.04 <= small[0].cost_dollars / 50 <= 0.08
    assert 0.04 <= large[0].cost_dollars / 500 <= 0.08
    # and the per-position % ceiling binds identically at both scales
    assert small[0].cost_dollars <= 50 * 0.08 + 0.70
    assert large[0].cost_dollars <= 500 * 0.08 + 0.70
