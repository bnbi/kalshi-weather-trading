"""
Same-Day Settlement Sniper

The next-day forecasting strategy competes against market makers who see the
same model runs we do — live results showed ~zero edge there. This module
attacks a different niche: SAME-DAY contracts in the afternoon, where the
daily high is often already locked in by real-time observations before thin,
inattentive books reprice.

Edge source: attention/latency, not modeling.
    1. Kalshi settles on the NWS climate report max at a specific station
       (KMDW / KNYC / KMIA).
    2. We poll real-time observations from that exact station. By early-to-mid
       afternoon the running max plus the remaining-hours forecast pins the
       final high tightly.
    3. Stale quotes in thin books still price yesterday's uncertainty. We buy
       the near-certain side when it's offered too cheap.

Safety: this strategy SELF-VALIDATES before risking money. Every run logs its
signals; outcomes are verified against actual highs. In --auto mode it trades
live only after ≥15 verified signals show positive hypothetical ROI and honest
calibration. Until then it runs dry.

Usage:
    python sniper.py run               # dry run, all cities
    python sniper.py run --auto        # live IF validation passed, else dry
    python sniper.py run --live        # force live (not recommended until validated)
    python sniper.py verify            # score past signals against actual highs
    python sniper.py report            # show signal accuracy + validation status
"""

from __future__ import annotations

# Silence the cosmetic LibreSSL warning that urllib3 prints on macOS —
# it fires once per run into stderr and looks like an error while meaning
# nothing. Filter by MESSAGE, before any urllib3 import: importing
# urllib3.exceptions to get the class itself triggered the warning it was
# meant to silence (the warning fires during urllib3's own import).
import warnings as _warnings
_warnings.filterwarnings("ignore", message=".*OpenSSL 1\\.1\\.1.*")


import sqlite3
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from weather import CITIES, get_hourly_forecast
from model import parse_contract_ticker, threshold_is_below
from find_edge import (get_market_prices, market_price, TradeSignal,
                       kalshi_fee_per_contract)
from trader import load_risk_config, _size_and_execute
from kalshi_client import create_client_from_config
from db_migrations import migrate_db
import station_obs
from station_obs import get_observed_max

BOT_DIR = Path(__file__).parent
DB_PATH = BOT_DIR / "kalshi_data.db"
KILL_SWITCH_FILE = BOT_DIR / "KILL_SWITCH"

# Settlement stations (single source of truth lives in station_obs.STATIONS)
STATIONS = {k: v["nws"] for k, v in station_obs.STATIONS.items()}

# ── Strategy parameters ─────────────────────────────────────────────
MIN_LOCAL_HOUR = 13     # don't snipe before 1pm local — high not locked yet
MAX_LOCAL_HOUR = 21     # markets near close; settlement imminent
MIN_PROB = 0.80         # only bet sides we believe ≥80% — certainty is the edge
MIN_EDGE = 0.10         # require 10¢ of mispricing (thin books, real spread cost)
MAX_TRADES_PER_CITY = 2
DEFAULT_BUDGET = None   # None = MAX_RUN_EXPOSURE_PCT of bankroll (config)
MC_SAMPLES = 20000

# Validation gate for --auto mode. 15 signals at a capped 93% claim could
# pass on luck alone; 30 plus a proper-scoring-rule test against the ask
# price (the market's own forecast) is the minimum for an honest verdict.
VALIDATION_MIN_SIGNALS = 30
VALIDATION_MAX_CALIB_GAP = 0.15   # claimed prob may exceed realized win rate by ≤15pts

# Probability-model version. Bump this whenever the final-high model changes so
# the validation gate only counts signals scored by the CURRENT model. Signals
# logged before this tag existed have model_version = NULL (the old, badly
# overconfident max(obs+spike, forecast) model) and are excluded from the gate.
MODEL_VERSION = "fh-passthrough-v4"

# ── v4 signal classes (from 50 verified signals through 2026-08-02) ──
# Verified history splits bracket-NO bets into two shapes:
#   PASS-THROUGH: obs still below the bracket, remaining forecast well above
#     it — the temp must climb THROUGH the bracket; NO wins unless it stalls
#     inside. Afternoon signals (≤9h to close) with ≥2°F forecast overshoot:
#     10W/0L, +238% ROI. The SAME shape fired at 10h (1pm, still forecast-
#     dependent): repeatedly lost. Morning pass-throughs are re-forecasting
#     the day — the game the day-ahead results proved we lose.
#   ALREADY-PASSED: obs feed above the bracket top. Sounds free, but the
#     official CLI sensor can read ~0.5-1°F below our obs feed (a 89.6°F obs
#     settled 89°F and lost). Require a 1.5°F margin so feed-vs-official
#     disagreement cannot flip settlement.
V4_MAX_HOURS_REMAINING = 9     # no pass-through signals before ~2pm local
V4_MIN_OVERSHOOT_F = 2.0       # rem forecast must clear bracket top by this
V4_OBS_PASSED_MARGIN_F = 1.5   # obs margin over bracket top for "already passed"
V4_PROB_CAP = 0.93             # never claim more than 93% — hubris tax


# ── Database ────────────────────────────────────────────────────────

def init_sniper_table(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sniper_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            date TEXT NOT NULL,
            city TEXT NOT NULL,
            ticker TEXT NOT NULL,
            side TEXT NOT NULL,
            prob REAL NOT NULL,
            ask_price REAL NOT NULL,
            obs_max_f REAL,
            rem_max_f REAL,
            hours_remaining REAL,
            mode TEXT NOT NULL,            -- 'dry' or 'live'
            traded INTEGER DEFAULT 0,
            outcome TEXT,                  -- 'win'/'loss' once verified
            hypo_profit REAL,              -- profit of 1 contract at ask
            model_version TEXT             -- which prob model scored this signal
        );
    """)
    # Migrations: model_version (NULL = legacy model); strike_type (so a
    # threshold signal can be graded in the right direction); truth_source
    # (what graded the outcome: 'kalshi' = the exchange's own settlement,
    # 'station' = official GHCND, 'legacy' = the old obs-feed/ERA5 proxy).
    cols = {r[1] for r in conn.execute("PRAGMA table_info(sniper_signals)")}
    for col in ("model_version TEXT", "strike_type TEXT", "truth_source TEXT",
                "legacy_outcome TEXT", "last_ask_price REAL", "times_seen INTEGER"):
        if col.split()[0] not in cols:
            conn.execute(f"ALTER TABLE sniper_signals ADD COLUMN {col}")
    if "truth_source" not in cols:
        # Outcomes graded before this change used the obs-feed / ERA5
        # proxy, which differs from the settlement value on ~40% of days.
        # Keep them for reference and re-grade against real truth.
        conn.execute("""UPDATE sniper_signals
                        SET legacy_outcome = outcome, truth_source = 'legacy'
                        WHERE outcome IS NOT NULL""")
        conn.execute("""UPDATE sniper_signals SET outcome = NULL, hypo_profit = NULL
                        WHERE truth_source = 'legacy'""")
    conn.commit()


# ── Real-time observations from the settlement station ─────────────
# get_observed_max moved to station_obs.py (imported above) so model.py's
# same-day logic can use it too without a circular import.


def get_remaining_forecast(city_key: str, local_date: str) -> dict:
    """Max forecast temp for the REMAINING hours of today (local)."""
    tz = ZoneInfo(CITIES[city_key].timezone)
    now_local = datetime.now(tz)

    rem_temps = []
    try:
        hourly = get_hourly_forecast(CITIES[city_key])
        for h in hourly:
            ht = datetime.fromisoformat(h["time"])
            if ht.astimezone(tz).strftime("%Y-%m-%d") == local_date and ht > now_local:
                rem_temps.append(h["temperature"])
    except Exception as e:
        print(f"  [{city_key}] hourly forecast failed: {e}")

    return {
        "rem_max_f": max(rem_temps) if rem_temps else None,
        "hours_remaining": len(rem_temps),
    }


# ── Final-high model (live-recalibrated) ────────────────────────────
#
# The original model — final = max(obs_max + [0,1,2]°F spike, Normal(rem_max,
# 0.6+0.18·hours)) — was badly overconfident: on 38 verified signals it
# claimed 89% and realized 61% (log loss 1.06). Reconstructing those signals
# showed the remaining-hours forecast max is itself an ~unbiased predictor of
# the settled high (error mean +0.15°F, σ≈1.4°F live), and that a single
# Normal centered there, floored at the already-observed max, calibrates to
# within ~3 points (claimed 64% / realized 61%, log loss 0.23) and is robust
# across σ ∈ [1.4, 4.0]. So we drop the two-component max() hack and model:
#
#     final_high = max(obs_max, Normal(center + bias, σ))
#     center = max(obs_max, rem_max)            # forecast is the best center
#     bias, σ = live-recalibrated from (settled_high − rem_max) errors
#
# obs_max is a hard physical floor — the day's high can't be below what's
# already been recorded.

DEFAULT_FH_SIGMA = 2.5     # used until enough verified history exists (wide on purpose)
FH_SIGMA_FLOOR = 2.0       # never trust the forecast tighter than this
FH_SIGMA_CAP = 4.0
FH_BIAS_CLIP = 2.0
FH_MIN_HISTORY = 6         # days needed before trusting fitted bias/σ


def fit_final_high_model(conn: sqlite3.Connection) -> dict:
    """
    Calibrate the final-high distribution from verified history.

    For each past day we have the remaining-hours forecast max logged at signal
    time (rem_max_f, averaged over that day's signals so a day counts once
    regardless of how many hourly runs fired) and the OFFICIAL settled high
    (daily_predictions.actual_high_f, GHCND only). The settlement error
    (actual − rem_max) gives a single bias and σ — hours_remaining is not a
    σ input; the floor/cap and the v4 structural rules carry that load.
    Below FH_MIN_HISTORY days we fall back to a deliberately wide default.
    """
    rows = conn.execute(f"""
        SELECT AVG(s.rem_max_f), p.actual_high_f
        FROM sniper_signals s
        JOIN daily_predictions p ON p.date = s.date AND p.city = s.city
        WHERE s.rem_max_f IS NOT NULL AND p.actual_high_f IS NOT NULL
        {_station_truth_filter(conn)}
        GROUP BY s.date, s.city
    """).fetchall()
    errs = [act - rem for rem, act in rows if rem is not None and act is not None]

    if len(errs) < FH_MIN_HISTORY:
        return {"bias": 0.0, "sigma": DEFAULT_FH_SIGMA, "n": len(errs)}

    mean = sum(errs) / len(errs)
    var = sum((e - mean) ** 2 for e in errs) / len(errs)
    sigma = min(max(var ** 0.5, FH_SIGMA_FLOOR), FH_SIGMA_CAP)
    bias = max(-FH_BIAS_CLIP, min(FH_BIAS_CLIP, mean))
    return {"bias": bias, "sigma": sigma, "n": len(errs)}


def final_high_samples(obs_max: float, rem_max: float | None,
                       hours_remaining: int, model: dict | None = None,
                       rng=None) -> np.ndarray:
    """
    Monte Carlo distribution of the settlement (CLI) daily max.

    Center on the remaining-hours forecast max (best single predictor), spread
    by the live-recalibrated settlement error, and floor at obs_max — the high
    cannot be below what's already been observed. When no remaining forecast is
    available, fall back to obs_max plus a small CLI-vs-METAR spike.
    """
    rng = rng or np.random.default_rng(42)
    sigma = (model or {}).get("sigma", DEFAULT_FH_SIGMA)
    bias = (model or {}).get("bias", 0.0)

    center = max(obs_max, rem_max) if rem_max is not None else obs_max + 1.0
    samples = rng.normal(center + bias, sigma, size=MC_SAMPLES)
    return np.maximum(obs_max, samples)   # physical floor: high >= observed max


def v4_signal_class(obs_max: float, rem_max: float | None,
                    hours_remaining: float,
                    bracket_low: float, bracket_high: float) -> str | None:
    """
    Classify a potential bracket-NO bet under the v4 rules.

    Returns 'passed', 'pass-through', or None (no trade). See the v4
    parameter block for the evidence behind each rule.
    """
    # Already passed: obs comfortably above the bracket top, with margin
    # for the official sensor reading lower than our obs feed.
    if obs_max >= bracket_high + V4_OBS_PASSED_MARGIN_F:
        return "passed"

    # Pass-through: temp must climb through the bracket, afternoon only,
    # with the remaining forecast well clear of the bracket top.
    if (rem_max is not None
            and obs_max < bracket_low
            and rem_max >= bracket_high + V4_MIN_OVERSHOOT_F
            and hours_remaining <= V4_MAX_HOURS_REMAINING):
        return "pass-through"

    return None


def contract_prob_yes(samples: np.ndarray, info: dict, title: str,
                      strike_type: str = None) -> float | None:
    """P(contract settles YES) from final-high samples. Settlement is integer °F."""
    settled = np.round(samples)
    if info["type"] == "threshold":
        th = info["threshold"]
        if threshold_is_below(title, strike_type):
            p = float(np.mean(settled < th))
        else:
            p = float(np.mean(settled > th))
    elif info["type"] == "bracket":
        p = float(np.mean((settled >= info["bracket_low"]) & (settled <= info["bracket_high"])))
    else:
        return None
    return min(0.99, max(0.01, p))


# ── Signal generation ───────────────────────────────────────────────

def find_sniper_signals(city_key: str, local_date: str,
                        fh_model: dict | None = None) -> tuple[list[TradeSignal], dict]:
    """Generate same-day signals for one city. Returns (signals, context)."""
    city = CITIES[city_key]
    ctx = {"obs": None, "rem": None}

    obs = get_observed_max(city_key, local_date)
    if obs is None:
        print(f"  [{city_key}] no usable observations — skipping")
        return [], ctx
    rem = get_remaining_forecast(city_key, local_date)
    ctx["obs"], ctx["rem"] = obs, rem

    print(f"  [{city_key}] obs max {obs['obs_max_f']:.1f}°F ({obs['n_obs']} obs) | "
          f"remaining fcst max {rem['rem_max_f']}°F over {rem['hours_remaining']}h")

    samples = final_high_samples(obs["obs_max_f"], rem["rem_max_f"],
                                 rem["hours_remaining"], model=fh_model)

    # Today's Kalshi date code, e.g. 2026-06-11 -> 26JUN11
    dt = datetime.strptime(local_date, "%Y-%m-%d")
    months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
              "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
    kalshi_date = f"{dt.year % 100:02d}{months[dt.month - 1]}{dt.day:02d}"

    signals = []
    ctx["strike_types"] = {}
    for m in get_market_prices(city.kalshi_series):
        ticker = m.get("ticker", "")
        if f"-{kalshi_date}-" not in ticker:
            continue
        info = parse_contract_ticker(ticker)
        ctx["strike_types"][ticker] = m.get("strike_type")
        # v4: bracket NO bets only. Thresholds went 1W/14L (7%) in
        # validation; YES bracket bets mean picking the exact 1°F stop —
        # the same bet type the day-ahead post-mortem showed is -80% ROI.
        if info.get("type") != "bracket":
            continue

        # v4 structural gate: the shape of the situation must qualify,
        # regardless of what the Monte Carlo probability claims.
        sig_class = v4_signal_class(
            obs_max=obs["obs_max_f"],
            rem_max=rem["rem_max_f"],
            hours_remaining=rem["hours_remaining"],
            bracket_low=info["bracket_low"],
            bracket_high=info["bracket_high"],
        )
        if sig_class is None:
            continue

        p_yes = contract_prob_yes(samples, info, m.get("title", ""),
                                  strike_type=m.get("strike_type"))
        if p_yes is None:
            continue

        no_ask = market_price(m, "no_ask")
        if no_ask is None or no_ask <= 0.02 or no_ask >= 0.97:
            continue

        # Cap claimed probability — v3's 97-99% claims won 36% of the time.
        p_no = min(V4_PROB_CAP, 1 - p_yes)
        # Net of the exchange fee, like the day-ahead pipeline — a 10c gross
        # edge at an 80c ask is really ~8.8c after the fee.
        edge = p_no - no_ask - kalshi_fee_per_contract(no_ask)
        if p_no >= MIN_PROB and edge >= MIN_EDGE:
            signals.append(TradeSignal(
                ticker=ticker, side="no", action="buy",
                model_prob=p_no, market_price=no_ask,
                edge=edge, expected_value=edge,
                description=f"SNIPE NO @ ${no_ask:.2f} [{sig_class}] | "
                            f"P={p_no:.1%} | obs_max={obs['obs_max_f']:.1f}°F",
            ))

    signals.sort(key=lambda s: s.edge, reverse=True)
    return signals[:MAX_TRADES_PER_CITY], ctx


# ── Validation gate ─────────────────────────────────────────────────

def _station_truth_filter(conn: sqlite3.Connection) -> str:
    """
    SQL clause restricting daily_predictions rows to OFFICIAL settlement-
    station truth (NOAA GHCND). Feed-max and ERA5 values differ from the
    settlement number on ~40% of days — enough to flip a bracket outcome —
    and must never feed the live-money validation gate or the final-high
    calibration. Legacy rows are re-labelled by daily_learner's
    re-verification, so unknown-source rows are excluded, not trusted.
    Returns a clause that matches nothing when the column doesn't exist.
    """
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(daily_predictions)")}
    except sqlite3.Error:
        return " AND 0"
    if "actual_source" not in cols:
        return " AND 0"
    return " AND p.actual_source = 'station'"


def _record_outcome(conn: sqlite3.Connection, sid: int, side: str, ask: float,
                    yes_settled: bool, truth_source: str) -> None:
    won = yes_settled if side == "yes" else not yes_settled
    # Net of the exchange fee, so the validation gate's "positive
    # hypothetical P&L" bar matches what live trading would earn.
    fee = kalshi_fee_per_contract(ask)
    profit = (1 - ask - fee) if won else (-ask - fee)
    conn.execute("""UPDATE sniper_signals
                    SET outcome = ?, hypo_profit = ?, truth_source = ?
                    WHERE id = ?""",
                 ("win" if won else "loss", profit, truth_source, sid))


def verify_signals(conn: sqlite3.Connection, client=None) -> int:
    """
    Score unverified signals against REAL settlement:
      1. Kalshi's own market result (the exchange's settlement — exact),
         while the market is still fetchable (~2 weeks);
      2. otherwise the official GHCND daily high from daily_predictions
         (never the provisional obs-feed max), with threshold direction
         taken from the stored strike_type.
    Signals that can be graded neither way are marked 'void' after 120 days.
    """
    n = 0
    rows = conn.execute("""
        SELECT id, ticker, side, ask_price, strike_type, date FROM sniper_signals
        WHERE outcome IS NULL ORDER BY date
    """).fetchall()
    if not rows:
        return 0

    # 1. Exchange settlement
    if client is None:
        try:
            client = create_client_from_config()
        except Exception:
            client = None
    graded = set()
    if client is not None:
        results_by_ticker = {}
        # Kalshi archives markets ~2 weeks after settlement; older signals go
        # straight to station truth instead of a guaranteed 404 per run.
        fetch_floor = (datetime.now(timezone.utc) - timedelta(days=21)).strftime("%Y-%m-%d")
        for sid, ticker, side, ask, strike_type, date in rows:
            if date < fetch_floor:
                continue
            if ticker in results_by_ticker:
                result = results_by_ticker[ticker]
            else:
                try:
                    market = client.get_market(ticker).get("market", {})
                    ok = market.get("status") in ("settled", "finalized")
                    result = market.get("result") if ok else None
                except Exception:
                    result = None
                results_by_ticker[ticker] = result
                time.sleep(0.1)
            if result in ("yes", "no"):
                _record_outcome(conn, sid, side, ask, result == "yes", "kalshi")
                graded.add(sid)
                n += 1

    # 2. Official station truth for what's left
    rows2 = conn.execute(f"""
        SELECT s.id, s.ticker, s.side, s.ask_price, s.strike_type, p.actual_high_f
        FROM sniper_signals s
        JOIN daily_predictions p ON p.date = s.date AND p.city = s.city
        WHERE s.outcome IS NULL AND p.actual_high_f IS NOT NULL
        {_station_truth_filter(conn)}
    """).fetchall()
    for sid, ticker, side, ask, strike_type, actual in rows2:
        if sid in graded:
            continue
        info = parse_contract_ticker(ticker)
        high = round(actual)
        if info["type"] == "threshold":
            if not strike_type:
                continue  # direction unknown — cannot grade honestly
            if threshold_is_below("", strike_type):
                yes_settled = high < info["threshold"]
            else:
                yes_settled = high > info["threshold"]
        elif info["type"] == "bracket":
            yes_settled = info["bracket_low"] <= high <= info["bracket_high"]
        else:
            continue
        _record_outcome(conn, sid, side, ask, yes_settled, "station")
        n += 1

    # 3. Give up on the ungradeable (GHCND is certainly published by then)
    conn.execute("""
        UPDATE sniper_signals SET outcome = 'void'
        WHERE outcome IS NULL AND date < date('now', '-120 days')
    """)
    conn.commit()
    return n


def validation_status(conn: sqlite3.Connection,
                      model_version: str = MODEL_VERSION) -> dict:
    """
    Validation metrics for the CURRENT probability model only. Signals scored by
    older models (model_version != current, including legacy NULL rows) are
    excluded so the gate reflects how the model trading today actually performs.
    """
    rows = conn.execute("""
        SELECT prob, ask_price, outcome, hypo_profit
        FROM sniper_signals
        WHERE outcome IN ('win', 'loss') AND model_version = ?
    """, (model_version,)).fetchall()
    n = len(rows)
    wins = sum(1 for r in rows if r[2] == "win")
    win_rate = wins / n if n else None
    avg_prob = sum(r[0] for r in rows) / n if n else None
    profit = sum(r[3] or 0 for r in rows) if n else None
    staked = sum(r[1] for r in rows) if n else None
    # Proper scoring: our claimed P(NO) vs the market's implied P(NO) (the
    # ask) on the same outcomes. A gate that only checks "profit > 0" and a
    # 15-point gap can be passed on luck; the model must also FORECAST
    # better than the price it is trading against.
    brier_model = brier_market = None
    if n:
        brier_model = sum((r[0] - (1.0 if r[2] == "win" else 0.0)) ** 2 for r in rows) / n
        brier_market = sum((r[1] - (1.0 if r[2] == "win" else 0.0)) ** 2 for r in rows) / n
    # Count legacy verified signals (old model) for reporting context only
    legacy = conn.execute("""
        SELECT COUNT(*) FROM sniper_signals
        WHERE outcome IN ('win', 'loss')
          AND (model_version IS NULL OR model_version != ?)
    """, (model_version,)).fetchone()[0]
    passed = bool(
        n >= VALIDATION_MIN_SIGNALS
        and (profit or 0) > 0
        and (avg_prob or 1) - (win_rate or 0) <= VALIDATION_MAX_CALIB_GAP
        and brier_model is not None and brier_model < brier_market
    )
    return {"n_verified": n, "win_rate": win_rate, "avg_claimed_prob": avg_prob,
            "hypo_profit": profit, "hypo_staked": staked, "passed": passed,
            "n_legacy": legacy, "model_version": model_version,
            "brier_model": brier_model, "brier_market": brier_market}


# ── Main run ────────────────────────────────────────────────────────

def run(cities: list[str] = None, mode: str = "dry", budget: float = DEFAULT_BUDGET):
    if KILL_SWITCH_FILE.exists():
        print("KILL SWITCH ACTIVE — sniper aborting.")
        return

    conn = sqlite3.connect(str(DB_PATH))
    init_sniper_table(conn)
    migrate_db(conn)

    # Exchange access is optional in dry mode (paper deployments have no
    # credentials); with it we grade signals on real settlement and true up
    # the day-ahead run's fills every hour — the stale-order backstop used
    # to run only once a day.
    client = None
    try:
        client = create_client_from_config()
    except Exception as e:
        print(f"  (no Kalshi credentials: {e}; grading on station truth only)")
    if client is not None:
        try:
            from pnl_tracker import init_pnl_tables, reconcile_fills
            init_pnl_tables(conn)
            reconcile_fills(conn, client)
        except Exception as e:
            print(f"  Warning: fill reconciliation failed: {e}")

    verify_signals(conn, client)  # opportunistically score old signals

    # Calibrate the final-high model from verified history (once per run)
    fh_model = fit_final_high_model(conn)
    print(f"Final-high model: σ={fh_model['sigma']:.2f}°F bias={fh_model['bias']:+.2f}°F "
          f"(from {fh_model['n']} verified day(s)"
          f"{'; using wide defaults' if fh_model['n'] < FH_MIN_HISTORY else ''})")

    if mode == "auto":
        status = validation_status(conn)
        mode = "live" if status["passed"] else "dry"
        bm = status.get("brier_model"); bk = status.get("brier_market")
        print(f"AUTO mode → {mode.upper()} "
              f"(verified={status['n_verified']}, "
              f"hypo P&L={status['hypo_profit'] or 0:+.2f}, "
              f"win rate={(status['win_rate'] or 0):.0%} vs "
              f"claimed {(status['avg_claimed_prob'] or 0):.0%}"
              + (f", Brier model {bm:.3f} vs market {bk:.3f}" if bm is not None else "")
              + ")")

    cities = cities or list(CITIES.keys())
    all_signals: list[tuple[str, TradeSignal, dict]] = []

    for ck in cities:
        tz = ZoneInfo(CITIES[ck].timezone)
        now_local = datetime.now(tz)
        if not (MIN_LOCAL_HOUR <= now_local.hour <= MAX_LOCAL_HOUR):
            print(f"  [{ck}] local time {now_local:%H:%M} outside snipe window — skipping")
            continue
        local_date = now_local.strftime("%Y-%m-%d")
        try:
            signals, ctx = find_sniper_signals(ck, local_date, fh_model=fh_model)
        except Exception as e:
            print(f"  [{ck}] ERROR: {e}")
            continue
        for s in signals:
            all_signals.append((ck, s, ctx))
        time.sleep(0.3)

    if not all_signals:
        print("No snipe opportunities found.")
        conn.close()
        return

    # Log every signal (dry or live) for validation.
    # Deduplicate: skip if we already logged this (date, ticker, side) today —
    # multiple runs per day would otherwise inflate signal count and make
    # validation stats noisy with correlated duplicates. Repeat sightings
    # update last_ask_price / times_seen so persistence is still visible.
    now_iso = datetime.now(timezone.utc).isoformat()
    logged = 0
    for ck, s, ctx in all_signals:
        tz = ZoneInfo(CITIES[ck].timezone)
        local_date = datetime.now(tz).strftime("%Y-%m-%d")
        already = conn.execute("""
            SELECT id FROM sniper_signals
            WHERE date = ? AND ticker = ? AND side = ? LIMIT 1
        """, (local_date, s.ticker, s.side)).fetchone()
        if already:
            conn.execute("""UPDATE sniper_signals
                            SET last_ask_price = ?, times_seen = COALESCE(times_seen, 1) + 1
                            WHERE id = ?""", (s.market_price, already[0]))
            continue
        conn.execute("""
            INSERT INTO sniper_signals
            (created_at, date, city, ticker, side, prob, ask_price,
             obs_max_f, rem_max_f, hours_remaining, mode, model_version,
             strike_type, last_ask_price, times_seen)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
        """, (now_iso, local_date, ck, s.ticker, s.side,
              s.model_prob, s.market_price,
              ctx["obs"]["obs_max_f"] if ctx["obs"] else None,
              ctx["rem"]["rem_max_f"] if ctx["rem"] else None,
              ctx["rem"]["hours_remaining"] if ctx["rem"] else None,
              mode, MODEL_VERSION,
              (ctx.get("strike_types") or {}).get(s.ticker), s.market_price))
        logged += 1
    conn.commit()
    if logged < len(all_signals):
        print(f"  ({len(all_signals) - logged} repeat signal(s) — already logged today)")

    print(f"\n  {len(all_signals)} signal(s) [{mode.upper()}]:")
    for ck, s, _ in all_signals:
        print(f"    {s.ticker} {s.side.upper()} edge={s.edge:+.1%} — {s.description}")

    if mode != "live":
        print("\n  Dry run — signals logged for validation. "
              "Run `python sniper.py report` to track accuracy.")
        conn.close()
        return

    # ── Live execution path ─────────────────────────────────────────
    if client is None:
        print("  Live mode requested but no Kalshi credentials — aborting.")
        conn.close()
        return
    cfg = load_risk_config()
    signals_only = [s for _, s, _ in all_signals]
    results = _size_and_execute(client, signals_only, dry_run=False, cfg=cfg,
                                max_spend=budget, label=" (sniper)")
    ok = [r for r in results if r.success and (r.filled_contracts is None
                                               or r.filled_contracts > 0)]
    if ok:
        # Match on (date, ticker, side) — NOT created_at: a signal first
        # logged by an earlier (possibly dry) run today keeps that run's
        # created_at. The sniper trades same-day contracts, so the signal's
        # date column equals the date embedded in the ticker. Upgrade mode
        # to 'live' too so the record shows what actually happened.
        for r in ok:
            sig_date = parse_contract_ticker(r.order.ticker).get("date")
            conn.execute("""UPDATE sniper_signals
                            SET traded = 1, mode = 'live'
                            WHERE date = ? AND ticker = ? AND side = ?""",
                         (sig_date, r.order.ticker, r.order.side))
        conn.commit()
    conn.close()


def report():
    conn = sqlite3.connect(str(DB_PATH))
    init_sniper_table(conn)
    migrate_db(conn)
    n = verify_signals(conn)
    if n:
        print(f"Verified {n} new signal(s).")
    s = validation_status(conn)
    print(f"\n  SNIPER VALIDATION STATUS")
    print(f"  Model version:      {s['model_version']}")
    print(f"  Verified signals:   {s['n_verified']} (need ≥{VALIDATION_MIN_SIGNALS}, current model only)")
    if s["n_legacy"]:
        print(f"  Legacy signals:     {s['n_legacy']} excluded (scored by an older model)")
    if s["n_verified"]:
        print(f"  Win rate:           {s['win_rate']:.0%} (claimed {s['avg_claimed_prob']:.0%}, "
              f"gap {s['avg_claimed_prob'] - s['win_rate']:+.0%})")
        print(f"  Hypothetical P&L:   ${s['hypo_profit']:+.2f} on ${s['hypo_staked']:.2f} staked")
        print(f"  Brier (lower=better): model {s['brier_model']:.3f} vs market ask "
              f"{s['brier_market']:.3f} — "
              f"{'model beats the price' if s['brier_model'] < s['brier_market'] else 'the PRICE is the better forecaster'}")
    else:
        print(f"  (no verified signals under the current model yet — keep running dry)")
    print(f"  LIVE TRADING GATE:  {'PASSED — --auto will trade live' if s['passed'] else 'not yet passed — --auto stays dry'}")
    print(f"  (gate: ≥{VALIDATION_MIN_SIGNALS} graded signals, positive hypo P&L, "
          f"≤{VALIDATION_MAX_CALIB_GAP:.0%} calibration gap, Brier below the ask)")

    print(f"\n  Recent signals (outcome / truth source):")
    for r in conn.execute("""SELECT date, city, ticker, side, ROUND(prob,2), ask_price,
                                    mode, COALESCE(outcome,'?'), COALESCE(truth_source,'-'),
                                    COALESCE(model_version,'legacy')
                             FROM sniper_signals ORDER BY id DESC LIMIT 15"""):
        print(f"    {r}")
    conn.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Same-day settlement sniper")
    sub = parser.add_subparsers(dest="cmd")

    p_run = sub.add_parser("run", help="Scan and (maybe) trade")
    p_run.add_argument("--city", choices=list(CITIES.keys()), default=None)
    p_run.add_argument("--live", action="store_true", help="Force live trading")
    p_run.add_argument("--auto", action="store_true",
                       help="Live only if validation gate has passed")
    p_run.add_argument("--budget", type=float, default=None,
                       help="Optional hard dollar ceiling (default: "
                            "MAX_RUN_EXPOSURE_PCT of bankroll)")

    sub.add_parser("verify", help="Score past signals")
    sub.add_parser("report", help="Show validation status")

    args = parser.parse_args()

    if args.cmd == "run":
        mode = "live" if args.live else ("auto" if args.auto else "dry")
        run(cities=[args.city] if args.city else None, mode=mode, budget=args.budget)
    elif args.cmd == "verify":
        conn = sqlite3.connect(str(DB_PATH))
        init_sniper_table(conn)
        migrate_db(conn)
        print(f"Verified {verify_signals(conn)} signal(s).")
        conn.close()
    else:
        report()
