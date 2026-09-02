"""
Unit tests for the Apple WeatherKit integration.

Covers the pure logic only — JWT construction, Celsius/date parsing of a
canned WeatherKit payload, and the learned shrinkage weight. No network
calls and no Apple credentials are required.

Run:  pytest -q
"""
from __future__ import annotations

import base64
import json
import sqlite3
from datetime import datetime, timezone

import pytest

import weatherkit
from weather import CITIES
from weather_ensemble import EnsembleForecast, ForecastSource


# ── JWT signing ─────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def ec_key_file(tmp_path_factory):
    """A throwaway P-256 key standing in for the Apple .p8."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    key = ec.generate_private_key(ec.SECP256R1())
    path = tmp_path_factory.mktemp("wk") / "AuthKey_TEST.p8"
    path.write_bytes(key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ))
    return str(path)


def _decode_segment(seg: str) -> dict:
    return json.loads(base64.urlsafe_b64decode(seg + "=" * (-len(seg) % 4)))


def test_token_has_apple_specific_header_and_claims(ec_key_file):
    cfg = {"team_id": "TEAM123456", "service_id": "com.example.wx",
           "key_id": "KEY9876543", "key_path": ec_key_file}
    token, expires = weatherkit._build_token(cfg)

    header_seg, payload_seg, sig_seg = token.split(".")
    header = _decode_segment(header_seg)
    payload = _decode_segment(payload_seg)

    assert header["alg"] == "ES256"
    assert header["kid"] == "KEY9876543"
    # The Apple quirk: header carries "<TeamID>.<ServiceID>"
    assert header["id"] == "TEAM123456.com.example.wx"

    assert payload["iss"] == "TEAM123456"
    assert payload["sub"] == "com.example.wx"
    assert payload["exp"] - payload["iat"] == weatherkit.TOKEN_TTL_SECONDS
    assert expires == payload["exp"]

    # JWS needs the raw 64-byte r||s form, not DER
    sig = base64.urlsafe_b64decode(sig_seg + "=" * (-len(sig_seg) % 4))
    assert len(sig) == 64


def test_token_is_verifiable_by_the_public_key(ec_key_file):
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec, utils

    cfg = {"team_id": "T", "service_id": "S", "key_id": "K",
           "key_path": ec_key_file}
    token, _ = weatherkit._build_token(cfg)
    signing_input, sig_seg = token.rsplit(".", 1)

    with open(ec_key_file, "rb") as f:
        pub = serialization.load_pem_private_key(f.read(), password=None).public_key()

    raw = base64.urlsafe_b64decode(sig_seg + "=" * (-len(sig_seg) % 4))
    r = int.from_bytes(raw[:32], "big")
    s = int.from_bytes(raw[32:], "big")
    pub.verify(utils.encode_dss_signature(r, s),
               signing_input.encode("ascii"),
               ec.ECDSA(hashes.SHA256()))


def test_rsa_key_is_rejected_with_a_useful_message(tmp_path):
    """Pointing WEATHERKIT_PRIVATE_KEY_PATH at the Kalshi RSA key is an
    easy mistake; it must fail loudly rather than produce a bad token."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    path = tmp_path / "rsa.pem"
    path.write_bytes(key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ))

    with pytest.raises(weatherkit.WeatherKitNotConfigured, match="not an EC"):
        weatherkit._build_token({"team_id": "T", "service_id": "S",
                                 "key_id": "K", "key_path": str(path)})


# ── Response parsing ────────────────────────────────────────────────

class _FakeResponse:
    status_code = 200

    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_daily_highs_convert_celsius_and_key_by_local_date(monkeypatch):
    # Chicago is UTC-5 in August: local midnight is 05:00Z, so the day
    # stamped 2026-08-27T05:00:00Z is the LOCAL 27th, not the 27th UTC.
    payload = {"forecastDaily": {"days": [
        {"forecastStart": "2026-08-27T05:00:00Z", "temperatureMax": 30.0},
        {"forecastStart": "2026-08-28T05:00:00Z", "temperatureMax": 26.6667},
        {"forecastStart": "2026-08-29T05:00:00Z", "temperatureMax": None},
    ]}}
    monkeypatch.setattr(weatherkit, "get_token", lambda: "fake")
    monkeypatch.setattr(weatherkit.requests, "get",
                        lambda *a, **k: _FakeResponse(payload))

    highs = weatherkit.fetch_daily_highs(CITIES["chicago"])

    assert highs == {"2026-08-27": 86.0, "2026-08-28": 80.0}
    assert "2026-08-29" not in highs  # null max is dropped, not zeroed


def test_unauthorized_raises_actionable_error(monkeypatch):
    class _Unauthorized(_FakeResponse):
        status_code = 401

    monkeypatch.setattr(weatherkit, "get_token", lambda: "fake")
    monkeypatch.setattr(weatherkit.requests, "get",
                        lambda *a, **k: _Unauthorized({}))

    with pytest.raises(RuntimeError, match="401"):
        weatherkit.fetch_daily_highs(CITIES["chicago"])


# ── Learned shrinkage weight ────────────────────────────────────────

def _seed_predictions(rows):
    """In-memory daily_predictions with (pred, apple, actual) triples."""
    conn = sqlite3.connect(":memory:")
    conn.execute("""
        CREATE TABLE daily_predictions (
            date TEXT, city TEXT, model_prediction_f REAL,
            wk_forecast_f REAL, actual_high_f REAL, actual_source TEXT)
    """)
    conn.executemany(
        "INSERT INTO daily_predictions VALUES (?,?,?,?,?,'station')",
        [(f"2026-01-{i % 28 + 1:02d}", "chicago", p, a, act)
         for i, (p, a, act) in enumerate(rows)])
    conn.commit()
    return conn


@pytest.fixture
def ensemble_with_apple(monkeypatch):
    def _make(rows, apple_high=85.0, ml_high=80.0, mode="blend"):
        conn = _seed_predictions(rows)
        monkeypatch.setattr("sqlite3.connect", lambda *a, **k: conn)
        monkeypatch.setattr("weather_ensemble._wk_config",
                            lambda name, default:
                            mode if name == "WEATHERKIT_MODE" else default)
        ens = EnsembleForecast(city="Chicago", date="2026-08-27")
        ens.sources = [ForecastSource("AppleWeatherKit", apple_high)]
        ens.ensemble_high_f = ml_high
        return ens
    return _make


def test_weight_is_zero_below_the_minimum_sample(ensemble_with_apple):
    # Apple perfectly predicts the miss, but only 10 days of it
    rows = [(80.0, 82.0, 82.0)] * 10
    ens = ensemble_with_apple(rows)
    assert ens._get_weatherkit_weight() == 0.0


def test_weight_recovers_a_known_relationship(ensemble_with_apple):
    # actual = pred + 0.5*(apple - pred): the true weight is 0.5,
    # which the MAX_WEIGHT cap then clips to 0.35.
    rows = [(80.0, 80.0 + d, 80.0 + 0.5 * d)
            for d in [-4, -3, -2, -1, 1, 2, 3, 4] * 8]
    ens = ensemble_with_apple(rows)
    assert ens._get_weatherkit_weight() == pytest.approx(
        EnsembleForecast.WK_MAX_WEIGHT)


def test_weight_is_zero_when_apple_adds_nothing(ensemble_with_apple):
    # Apple's disagreement is pure noise w.r.t. the model's error
    rows = [(80.0, 80.0 + d, 80.0) for d in [-3, -1, 1, 3] * 16]
    ens = ensemble_with_apple(rows)
    assert ens._get_weatherkit_weight() == pytest.approx(0.0)


def test_weight_never_goes_negative(ensemble_with_apple):
    # Apple leans the wrong way — shrink toward it by 0, never away from it
    rows = [(80.0, 80.0 + d, 80.0 - 0.5 * d)
            for d in [-4, -2, 2, 4] * 16]
    ens = ensemble_with_apple(rows)
    assert ens._get_weatherkit_weight() == 0.0


def test_shadow_mode_never_moves_the_point_forecast(ensemble_with_apple):
    rows = [(80.0, 80.0 + d, 80.0 + 0.5 * d)
            for d in [-4, -3, -2, -1, 1, 2, 3, 4] * 8]
    ens = ensemble_with_apple(rows, mode="shadow")
    ens._apply_weatherkit_blend()
    assert ens.ensemble_high_f == 80.0


def test_blend_mode_moves_the_forecast_toward_apple(ensemble_with_apple):
    rows = [(80.0, 80.0 + d, 80.0 + 0.5 * d)
            for d in [-4, -3, -2, -1, 1, 2, 3, 4] * 8]
    ens = ensemble_with_apple(rows, apple_high=90.0, ml_high=80.0, mode="blend")
    ens._apply_weatherkit_blend()
    # 80 + 0.35*(90-80) = 83.5
    assert ens.ensemble_high_f == pytest.approx(83.5)


def test_blend_is_a_no_op_without_an_apple_source(ensemble_with_apple):
    rows = [(80.0, 80.0 + d, 80.0 + 0.5 * d)
            for d in [-4, -3, -2, -1, 1, 2, 3, 4] * 8]
    ens = ensemble_with_apple(rows, mode="blend")
    ens.sources = [ForecastSource("OpenMeteo/gfs", 79.0)]
    ens._apply_weatherkit_blend()
    assert ens.ensemble_high_f == 80.0


def test_missing_wk_column_degrades_to_zero_weight(monkeypatch):
    """An un-migrated database must not crash the prediction path."""
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE daily_predictions (date TEXT, city TEXT)")
    monkeypatch.setattr("sqlite3.connect", lambda *a, **k: conn)
    ens = EnsembleForecast(city="Chicago", date="2026-08-27")
    assert ens._get_weatherkit_weight() == 0.0


def test_shadow_mode_never_joins_ensemble(monkeypatch):
    """In shadow mode WeatherKit must NOT enter ensemble.sources — spread
    feeds sigma, so a shadow source would alter live trade sizing."""
    import weather_ensemble as we

    monkeypatch.setattr(we.weatherkit, "is_enabled", lambda: True)
    monkeypatch.setattr(we.weatherkit, "fetch_daily_highs",
                        lambda city: {"2099-01-01": 84.0})
    monkeypatch.setattr(we, "_wk_config",
                        lambda name, default: "shadow" if name == "WEATHERKIT_MODE" else default)
    # Silence all other sources
    monkeypatch.setattr(we, "get_daily_high_forecast",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("skip")))
    monkeypatch.setattr(we, "fetch_open_meteo_forecast",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("skip")))

    ens = we.build_ensemble("chicago", target_date="2099-01-01")
    assert all("apple" not in s.source.lower() for s in ens.sources)


def test_blend_mode_does_join_ensemble(monkeypatch):
    import weather_ensemble as we

    monkeypatch.setattr(we.weatherkit, "is_enabled", lambda: True)
    monkeypatch.setattr(we.weatherkit, "fetch_daily_highs",
                        lambda city: {"2099-01-01": 84.0})
    monkeypatch.setattr(we, "_wk_config",
                        lambda name, default: "blend" if name == "WEATHERKIT_MODE" else default)
    monkeypatch.setattr(we, "get_daily_high_forecast",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("skip")))
    monkeypatch.setattr(we, "fetch_open_meteo_forecast",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("skip")))

    ens = we.build_ensemble("chicago", target_date="2099-01-01")
    assert any("apple" in s.source.lower() for s in ens.sources)


def test_wk_training_feature_requires_blend_mode(monkeypatch):
    """Even with perfect Apple coverage, shadow mode must keep 'wk' out of
    the trained feature set — otherwise the nightly retrain would create a
    model whose live inputs (Apple excluded from sources in shadow mode)
    can't match its training inputs."""
    import pandas as pd
    import numpy as np
    import train_model

    n = 400
    df = pd.DataFrame({
        "gfs_forecast_f": np.linspace(60, 90, n),
        "ecmwf_forecast_f": np.linspace(61, 89, n),
        "blend_forecast_f": np.linspace(60.5, 89.5, n),
        "icon_forecast_f": np.linspace(60, 90, n),
        "wk_forecast_f": np.linspace(59, 91, n),   # full coverage, has variance
        "wk_present": [True] * n,
        "month": [6] * n,
        "day_of_year": list(range(1, n + 1)),
        "model_spread": [1.0] * n,
        "gfs_error": [0.5] * n, "ecmwf_error": [-0.5] * n, "blend_error": [0.1] * n,
        "wind_speed_max": [10.0] * n, "humidity_mean": [50.0] * n,
        "cloud_cover_mean": [50.0] * n,
    })

    monkeypatch.setattr(train_model, "_wk_mode", lambda: "shadow")
    shadow_features = train_model.build_features(df)
    assert "wk" not in shadow_features.columns
    assert "wk_minus_ecmwf" not in shadow_features.columns

    monkeypatch.setattr(train_model, "_wk_mode", lambda: "blend")
    blend_features = train_model.build_features(df)
    assert "wk" in blend_features.columns
