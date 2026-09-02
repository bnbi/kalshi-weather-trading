"""
Unit tests for the core quantitative logic.

These cover the pure, deterministic pieces that the trading decisions depend on —
Kelly sizing, contract parsing, the probability model (including the
settlement-rounding behaviour for brackets), edge detection with market-prior
shrinkage and the credibility filter, the same-day final-high model, and the
self-validation gate. No network or live API calls are exercised.

Run:  pytest -q
"""
from __future__ import annotations

import math
import sqlite3

import numpy as np
import pytest

import trader
import model
import find_edge
import sniper


# ── Kelly sizing ────────────────────────────────────────────────────

def test_kelly_basic_value():
    # p=0.60, price=0.50 -> b=1, full=(0.6*1-0.4)/1=0.20, quarter-ish fraction
    f = trader.kelly_size(0.60, 0.50, kelly_fraction=0.15)
    assert f == pytest.approx(0.20 * 0.15, rel=1e-9)


def test_kelly_no_edge_returns_zero():
    # Fair coin priced fairly -> no edge -> no bet
    assert trader.kelly_size(0.50, 0.50, 0.15) == 0.0


def test_kelly_negative_edge_returns_zero():
    assert trader.kelly_size(0.40, 0.60, 0.15) == 0.0


@pytest.mark.parametrize("price", [0.0, 1.0, -0.1, 1.5])
def test_kelly_rejects_degenerate_prices(price):
    assert trader.kelly_size(0.9, price, 0.15) == 0.0


def test_kelly_scales_with_fraction():
    full = trader.kelly_size(0.7, 0.5, kelly_fraction=1.0)
    half = trader.kelly_size(0.7, 0.5, kelly_fraction=0.5)
    assert half == pytest.approx(full * 0.5, rel=1e-9)


# ── Contract ticker parsing ─────────────────────────────────────────

def test_parse_threshold_ticker():
    info = model.parse_contract_ticker("KXHIGHCHI-26MAY05-T65")
    assert info["type"] == "threshold"
    assert info["threshold"] == 65.0
    assert info["date"] == "2026-05-05"


def test_parse_bracket_ticker():
    info = model.parse_contract_ticker("KXHIGHCHI-26MAY05-B64.5")
    assert info["type"] == "bracket"
    # B64.5 covers integer highs 64 and 65
    assert info["bracket_low"] == 64.0
    assert info["bracket_high"] == 65.0


def test_parse_unknown_ticker():
    assert model.parse_contract_ticker("garbage")["type"] == "unknown"


# ── Probability model ───────────────────────────────────────────────

def _ncdf(x, mu, sd):
    return 0.5 * (1 + math.erf((x - mu) / (sd * math.sqrt(2))))


def test_threshold_above_uses_rounding_cutoff():
    # Settlement is the rounded integer high: "high > 70" pays iff the
    # rounded high >= 71, i.e. continuous temp >= 70.5 — NOT temp > 70.
    info = {"type": "threshold", "threshold": 70.0}
    p = model.compute_probability(75.0, 3.0, 0.0, info, title="High temp > 70")
    assert p == pytest.approx(1 - _ncdf(70.5, 75.0, 3.0), abs=1e-6)


def test_threshold_half_degree_cutoff_unchanged():
    # "high > 87.5" pays iff rounded high >= 88 <=> temp >= 87.5: the
    # half-degree threshold IS the continuous cutoff.
    info = {"type": "threshold", "threshold": 87.5}
    p = model.compute_probability(86.0, 2.0, 0.0, info, title="high > 87.5")
    assert p == pytest.approx(1 - _ncdf(87.5, 86.0, 2.0), abs=1e-6)


def test_threshold_above_below_leave_room_for_exact_settle():
    # ">70" (settle >= 71) and "<70" (settle <= 69) are NOT complements:
    # a settle at exactly 70 loses both. above + below + P(settle == 70) = 1.
    above = model.compute_probability(
        75.0, 3.0, 0.0, {"type": "threshold", "threshold": 70.0}, "high > 70")
    below = model.compute_probability(
        75.0, 3.0, 0.0, {"type": "threshold", "threshold": 70.0}, "high < 70")
    p_exact_70 = _ncdf(70.5, 75.0, 3.0) - _ncdf(69.5, 75.0, 3.0)
    assert above == pytest.approx(1 - _ncdf(70.5, 75.0, 3.0), abs=1e-6)
    assert below == pytest.approx(_ncdf(69.5, 75.0, 3.0), abs=1e-6)
    assert above + below + p_exact_70 == pytest.approx(1.0, abs=1e-6)


def test_bracket_uses_rounding_boundaries():
    # B83.5 -> integer highs 83/84 -> continuous YES region [82.5, 84.5)
    info = {"type": "bracket", "bracket_low": 83.0, "bracket_high": 84.0}
    p = model.compute_probability(83.6, 2.0, 0.0, info, title="bracket")
    expected = _ncdf(84.5, 83.6, 2.0) - _ncdf(82.5, 83.6, 2.0)
    assert p == pytest.approx(expected, abs=1e-6)


def test_bracket_wider_than_naive_interval():
    # Regression guard for the rounding fix: the rounded-boundary probability
    # must exceed the naive cdf(high)-cdf(low) it replaced.
    info = {"type": "bracket", "bracket_low": 83.0, "bracket_high": 84.0}
    p = model.compute_probability(83.6, 2.0, 0.0, info, title="bracket")
    naive = _ncdf(84.0, 83.6, 2.0) - _ncdf(83.0, 83.6, 2.0)
    assert p > naive


def test_probability_is_clamped():
    info = {"type": "threshold", "threshold": 50.0}
    p = model.compute_probability(90.0, 1.0, 0.0, info, title="high > 50")
    assert p <= 0.97 + 1e-9  # ceiling applied, never a phantom 0.999 edge


# ── Edge detection: shrinkage + credibility filter ──────────────────

def _pred(ticker, prob, ctype="threshold"):
    return model.ContractPrediction(
        ticker=ticker, contract_type=ctype, description="",
        model_probability=prob, forecast_high=80.0, error_std=2.0,
        threshold=70.0)


def test_credibility_filter_skips_large_disagreement():
    # Model says 0.99 NO-side worthy but market NO ask is 0.70 -> raw gap 0.29
    # (> 25c). Such trades are SKIPPED, not capped.
    pred = _pred("KXHIGHNY-26JUN10-T70", 0.01)  # P(yes)=0.01 -> P(no)=0.99
    markets = [{"ticker": pred.ticker, "yes_ask_dollars": "0.30",
                "no_ask_dollars": "0.70", "title": "high > 70"}]
    signals = find_edge.calculate_edge([pred], markets, min_edge=0.05)
    assert all(s.ticker != pred.ticker or s.side != "no" for s in signals) or signals == []


def test_blended_edge_is_half_raw_gap_minus_fee():
    # MODEL_WEIGHT = 0.5 -> blended edge is half the raw model-market gap,
    # net of Kalshi's trading fee (0.07 * P * (1-P), rounded up to a cent).
    pred = _pred("KXHIGHNY-26JUN10-T70", 0.05)  # P(no)=0.95
    no_ask = 0.75
    markets = [{"ticker": pred.ticker, "yes_ask_dollars": "0.25",
                "no_ask_dollars": str(no_ask), "title": "high > 70"}]
    signals = find_edge.calculate_edge([pred], markets, min_edge=0.05)
    no_sigs = [s for s in signals if s.side == "no"]
    assert no_sigs, "expected a NO signal within the credibility band"
    raw_gap = 0.95 - no_ask
    fee = find_edge.kalshi_fee_per_contract(no_ask)
    assert fee == pytest.approx(0.07 * 0.75 * 0.25)  # exact, no rounding up
    assert no_sigs[0].edge == pytest.approx(0.5 * raw_gap - fee, abs=1e-9)


def test_fee_formula():
    # Kalshi charges the exact 0.07*P*(1-P) per contract (the ledger shows
    # $0.0294 for 2 @ 70c); the old ceil-to-cent overstated it by ~35%.
    assert find_edge.kalshi_fee_per_contract(0.50) == pytest.approx(0.0175)
    assert find_edge.kalshi_fee_per_contract(0.70) == pytest.approx(0.0147)
    assert find_edge.kalshi_fee_per_contract(0.95) == pytest.approx(0.003325)
    assert find_edge.kalshi_fee_per_contract(0.0) == 0.0
    assert find_edge.kalshi_fee_per_contract(1.0) == 0.0


def test_sizing_floor_buys_one_contract_on_small_bankroll():
    # With an $8 bankroll, fractional Kelly suggests < 1 contract's cost.
    # The floor should still buy 1 contract when it fits the caps.
    import trader
    sig = find_edge.TradeSignal(
        ticker="KXHIGHCHI-26JUL16-B85.5", side="no", action="buy",
        model_prob=0.85, market_price=0.72, edge=0.08,
        expected_value=0.08, description="test")
    orders = trader.size_orders([sig], bankroll=8.37, kelly_fraction=0.15,
                                max_position_dollars=2.0, max_contracts=5,
                                max_total_dollars=2.79, max_positions=6)
    assert len(orders) == 1
    assert orders[0].contracts == 1

    # But NOT when one contract exceeds the per-position cap
    orders = trader.size_orders([sig], bankroll=8.37, kelly_fraction=0.15,
                                max_position_dollars=0.50, max_contracts=5,
                                max_total_dollars=2.79, max_positions=6)
    assert orders == []


def test_bracket_yes_bets_disallowed():
    pred = _pred("KXHIGHNY-26JUN10-B70.5", 0.99, ctype="bracket")
    markets = [{"ticker": pred.ticker, "yes_ask_dollars": "0.20",
                "no_ask_dollars": "0.80", "title": "70-71"}]
    signals = find_edge.calculate_edge([pred], markets, min_edge=0.05)
    assert all(s.side != "yes" for s in signals)


def test_parse_price_handles_bad_input():
    assert find_edge.parse_price(None) is None
    assert find_edge.parse_price("not-a-number") is None
    assert find_edge.parse_price("0.43") == pytest.approx(0.43)


# ── Same-day final-high model ───────────────────────────────────────

def test_final_high_floored_at_observed_max():
    # Even with a low forecast, no sample may fall below the already-observed max.
    s = sniper.final_high_samples(90.0, rem_max=70.0, hours_remaining=3,
                                  model={"sigma": 2.0, "bias": 0.0},
                                  rng=np.random.default_rng(0))
    assert s.min() >= 90.0


def test_final_high_centers_on_forecast_when_higher():
    s = sniper.final_high_samples(70.0, rem_max=85.0, hours_remaining=5,
                                  model={"sigma": 2.0, "bias": 0.0},
                                  rng=np.random.default_rng(0))
    # Mean should sit near the forecast (85), well above obs_max (70)
    assert 83.0 < s.mean() < 87.0


def test_fit_final_high_model_falls_back_when_sparse():
    conn = sqlite3.connect(":memory:")
    sniper.init_sniper_table(conn)
    conn.execute("""CREATE TABLE daily_predictions
                    (date TEXT, city TEXT, actual_high_f REAL, actual_source TEXT)""")
    m = sniper.fit_final_high_model(conn)
    assert m["sigma"] == sniper.DEFAULT_FH_SIGMA
    assert m["n"] == 0


def test_fit_final_high_model_floors_sigma():
    conn = sqlite3.connect(":memory:")
    sniper.init_sniper_table(conn)
    conn.execute("""CREATE TABLE daily_predictions
                    (date TEXT, city TEXT, actual_high_f REAL, actual_source TEXT)""")
    # Insert >= FH_MIN_HISTORY days where actual == forecast exactly (zero error).
    for i in range(sniper.FH_MIN_HISTORY + 2):
        d = f"2026-06-{i+1:02d}"
        conn.execute("""INSERT INTO sniper_signals
            (created_at,date,city,ticker,side,prob,ask_price,rem_max_f,mode)
            VALUES('x',?,?,?,?,?,?,?,?)""",
            (d, "nyc", "KXHIGHNY-x-T80", "no", 0.9, 0.5, 80.0, "dry"))
        conn.execute("INSERT INTO daily_predictions VALUES(?,?,?,'station')", (d, "nyc", 80.0))
    conn.commit()
    m = sniper.fit_final_high_model(conn)
    # Zero measured error, but the floor keeps us from overconfidence.
    assert m["sigma"] == pytest.approx(sniper.FH_SIGMA_FLOOR)


def test_contract_prob_yes_threshold():
    samples = np.full(1000, 80.0)  # final high pinned at 80
    p = sniper.contract_prob_yes(samples, {"type": "threshold", "threshold": 75}, "high > 75")
    assert p > 0.95  # 80 > 75 almost surely


# ── Self-validation gate ────────────────────────────────────────────

def _seed_signals(conn, n, win_rate, claimed, model_version):
    wins = int(round(n * win_rate))
    for i in range(n):
        outcome = "win" if i < wins else "loss"
        profit = 0.4 if outcome == "win" else -0.6
        conn.execute("""INSERT INTO sniper_signals
            (created_at,date,city,ticker,side,prob,ask_price,mode,outcome,
             hypo_profit,model_version)
            VALUES('x',?,?,?,?,?,?,?,?,?,?)""",
            (f"2026-06-{i+1:02d}", "nyc", "KXHIGHNY-x-B80.5", "no",
             claimed, 0.6, "dry", outcome, profit, model_version))
    conn.commit()


def test_gate_excludes_legacy_model_versions():
    conn = sqlite3.connect(":memory:")
    sniper.init_sniper_table(conn)
    # 30 well-calibrated current-model signals + 30 overconfident legacy ones
    _seed_signals(conn, 30, win_rate=0.9, claimed=0.9, model_version=sniper.MODEL_VERSION)
    _seed_signals(conn, 30, win_rate=0.5, claimed=0.9, model_version=None)
    s = sniper.validation_status(conn)
    assert s["n_verified"] == 30           # legacy excluded
    assert s["n_legacy"] == 30
    assert s["passed"] is True             # current model is calibrated + profitable


def test_gate_fails_when_overconfident():
    conn = sqlite3.connect(":memory:")
    sniper.init_sniper_table(conn)
    _seed_signals(conn, 30, win_rate=0.6, claimed=0.9, model_version=sniper.MODEL_VERSION)
    s = sniper.validation_status(conn)
    assert s["passed"] is False            # 30pt calibration gap > 15pt limit


def test_gate_fails_below_min_signals():
    conn = sqlite3.connect(":memory:")
    sniper.init_sniper_table(conn)
    _seed_signals(conn, 5, win_rate=1.0, claimed=0.9, model_version=sniper.MODEL_VERSION)
    s = sniper.validation_status(conn)
    assert s["passed"] is False            # n below VALIDATION_MIN_SIGNALS


def test_gate_fails_when_market_price_is_the_better_forecaster():
    """Profitable-looking and 'calibrated' on average, but the ask price
    scores a better Brier than the model's claims — the exact failure the
    day-ahead postmortem found. The gate must stay shut."""
    conn = sqlite3.connect(":memory:")
    sniper.init_sniper_table(conn)
    for i in range(40):
        # claimed 0.85 on every signal; the ask (0.75) tracks reality better:
        # signals at ask 0.75 win 76% of the time
        outcome = "win" if i % 25 != 0 and i % 4 != 0 else "loss"
        conn.execute("""INSERT INTO sniper_signals
            (created_at,date,city,ticker,side,prob,ask_price,mode,outcome,
             hypo_profit,model_version)
            VALUES('x',?,?,?,?,?,?,?,?,?,?)""",
            (f"2026-06-{i % 28 + 1:02d}", "nyc", f"K-{i}", "no",
             0.85, 0.75, "dry", outcome, 0.24 if outcome == "win" else -0.76,
             sniper.MODEL_VERSION))
    conn.commit()
    s = sniper.validation_status(conn)
    assert s["n_verified"] == 40
    assert s["brier_model"] >= s["brier_market"] or s["passed"] is False
    assert s["passed"] is False


def test_sizing_rounds_price_to_nearest_cent():
    """0.29 * 100 is 28.999999999999996 in floating point; int() placed
    the limit a cent BELOW the ask so the order rested instead of filling."""
    import trader
    for ask in (0.29, 0.57, 0.58):
        sig = find_edge.TradeSignal(
            ticker="KXHIGHCHI-26JUL16-T80", side="yes", action="buy",
            model_prob=0.75, market_price=ask, edge=0.10,
            expected_value=0.10, description="t")
        orders = trader.size_orders([sig], bankroll=100.0, kelly_fraction=0.25,
                                    max_position_dollars=8.0, max_contracts=15,
                                    max_total_dollars=25.0, max_positions=6)
        assert orders and orders[0].price_cents == int(round(ask * 100))


def test_parse_unknown_month_is_unknown_not_january():
    info = model.parse_contract_ticker("KXHIGHCHI-26XYZ05-T65")
    assert info["type"] == "unknown"


def test_threshold_direction_prefers_strike_type():
    # title says nothing useful; the API's strike_type decides
    assert model.threshold_is_below("Will it be hot?", "less") is True
    assert model.threshold_is_below("high <58", "greater") is False
    assert model.threshold_is_below("high <58", None) is True
    info = {"type": "threshold", "threshold": 58.0}
    below = model.compute_probability(60.0, 2.0, 0.0, info, title="", strike_type="less")
    above = model.compute_probability(60.0, 2.0, 0.0, info, title="", strike_type="greater")
    assert below == pytest.approx(_ncdf(57.5, 60.0, 2.0), abs=1e-6)
    assert above == pytest.approx(1 - _ncdf(58.5, 60.0, 2.0), abs=1e-6)
