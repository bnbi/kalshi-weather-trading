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
from model import parse_contract_ticker
from find_edge import (get_market_prices, parse_price, TradeSignal,
                       kalshi_fee_per_contract)
from trader import (size_orders, execute_orders, optimize_with_orderbook,
                    check_existing_positions)
from pnl_tracker import log_trade_results
from kalshi_client import create_client_from_config
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

# Validation gate for --auto mode
VALIDATION_MIN_SIGNALS = 15
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
    # Migration: add model_version to pre-existing tables (NULL = legacy model)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(sniper_signals)")}
    if "model_version" not in cols:
        conn.execute("ALTER TABLE sniper_signals ADD COLUMN model_version TEXT")
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
    time (rem_max_f) and the settled high (daily_predictions.actual_high_f).
    The settlement error (actual − rem_max) gives bias and σ. A conservative
    σ floor keeps us from ever getting overconfident again; below FH_MIN_HISTORY
    days we fall back to a deliberately wide default.
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


def contract_prob_yes(samples: np.ndarray, info: dict, title: str) -> float | None:
    """P(contract settles YES) from final-high samples. Settlement is integer °F."""
    settled = np.round(samples)
    if info["type"] == "threshold":
        th = info["threshold"]
        title_l = title.lower()
        if "<" in title or "below" in title_l:
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
    for m in get_market_prices(city.kalshi_series):
        ticker = m.get("ticker", "")
        if kalshi_date not in ticker:
            continue
        info = parse_contract_ticker(ticker)
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

        p_yes = contract_prob_yes(samples, info, m.get("title", ""))
        if p_yes is None:
            continue

        no_ask = parse_price(m.get("no_ask_dollars", m.get("no_ask")))
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
    SQL clause restricting daily_predictions rows to settlement-station
    truth. Rows verified from the ERA5 fallback can differ from the station
    by 1-3°F — enough to flip a bracket outcome — and must never feed the
    live-money validation gate or the final-high calibration. NULL is
    accepted for legacy rows written before the source was recorded.
    Returns '' when the column doesn't exist yet (un-migrated DB).
    """
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(daily_predictions)")}
    except sqlite3.Error:
        return ""
    if "actual_source" not in cols:
        return ""
    return " AND (p.actual_source IS NULL OR p.actual_source = 'station')"


def verify_signals(conn: sqlite3.Connection) -> int:
    """Score past signals against actual highs from daily_predictions."""
    rows = conn.execute(f"""
        SELECT s.id, s.ticker, s.side, s.ask_price, p.actual_high_f
        FROM sniper_signals s
        JOIN daily_predictions p ON p.date = s.date AND p.city = s.city
        WHERE s.outcome IS NULL AND p.actual_high_f IS NOT NULL
        {_station_truth_filter(conn)}
    """).fetchall()

    n = 0
    for sid, ticker, side, ask, actual in rows:
        info = parse_contract_ticker(ticker)
        high = round(actual)
        if info["type"] == "threshold":
            yes_settled = high > info["threshold"]
        elif info["type"] == "bracket":
            yes_settled = info["bracket_low"] <= high <= info["bracket_high"]
        else:
            continue
        won = yes_settled if side == "yes" else not yes_settled
        # Net of the exchange fee, so the validation gate's "positive
        # hypothetical P&L" bar matches what live trading would earn.
        fee = kalshi_fee_per_contract(ask)
        profit = (1 - ask - fee) if won else (-ask - fee)
        conn.execute("UPDATE sniper_signals SET outcome = ?, hypo_profit = ? WHERE id = ?",
                     ("win" if won else "loss", profit, sid))
        n += 1
    conn.commit()
    return n


def validation_status(conn: sqlite3.Connection,
                      model_version: str = MODEL_VERSION) -> dict:
    """
    Validation metrics for the CURRENT probability model only. Signals scored by
    older models (model_version != current, including legacy NULL rows) are
    excluded so the gate reflects how the model trading today actually performs.
    """
    row = conn.execute("""
        SELECT COUNT(*), AVG(CASE WHEN outcome='win' THEN 1.0 ELSE 0.0 END),
               AVG(prob), SUM(hypo_profit), SUM(ask_price)
        FROM sniper_signals
        WHERE outcome IS NOT NULL AND model_version = ?
    """, (model_version,)).fetchone()
    n, win_rate, avg_prob, profit, staked = row
    n = n or 0
    # Count legacy verified signals (old model) for reporting context only
    legacy = conn.execute("""
        SELECT COUNT(*) FROM sniper_signals
        WHERE outcome IS NOT NULL
          AND (model_version IS NULL OR model_version != ?)
    """, (model_version,)).fetchone()[0]
    passed = bool(
        n >= VALIDATION_MIN_SIGNALS
        and (profit or 0) > 0
        and (avg_prob or 1) - (win_rate or 0) <= VALIDATION_MAX_CALIB_GAP
    )
    return {"n_verified": n, "win_rate": win_rate, "avg_claimed_prob": avg_prob,
            "hypo_profit": profit, "hypo_staked": staked, "passed": passed,
            "n_legacy": legacy, "model_version": model_version}


# ── Main run ────────────────────────────────────────────────────────

def run(cities: list[str] = None, mode: str = "dry", budget: float = DEFAULT_BUDGET):
    if KILL_SWITCH_FILE.exists():
        print("KILL SWITCH ACTIVE — sniper aborting.")
        return

    conn = sqlite3.connect(str(DB_PATH))
    init_sniper_table(conn)
    verify_signals(conn)  # opportunistically score old signals

    # Calibrate the final-high model from verified history (once per run)
    fh_model = fit_final_high_model(conn)
    print(f"Final-high model: σ={fh_model['sigma']:.2f}°F bias={fh_model['bias']:+.2f}°F "
          f"(from {fh_model['n']} verified day(s)"
          f"{'; using wide defaults' if fh_model['n'] < FH_MIN_HISTORY else ''})")

    if mode == "auto":
        status = validation_status(conn)
        mode = "live" if status["passed"] else "dry"
        print(f"AUTO mode → {mode.upper()} "
              f"(verified={status['n_verified']}, "
              f"hypo P&L={status['hypo_profit'] or 0:+.2f}, "
              f"win rate={(status['win_rate'] or 0):.0%} vs "
              f"claimed {(status['avg_claimed_prob'] or 0):.0%})")

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
    # validation stats noisy with correlated duplicates.
    now_iso = datetime.now(timezone.utc).isoformat()
    logged = 0
    for ck, s, ctx in all_signals:
        tz = ZoneInfo(CITIES[ck].timezone)
        local_date = datetime.now(tz).strftime("%Y-%m-%d")
        already = conn.execute("""
            SELECT 1 FROM sniper_signals
            WHERE date = ? AND ticker = ? AND side = ? LIMIT 1
        """, (local_date, s.ticker, s.side)).fetchone()
        if already:
            continue
        conn.execute("""
            INSERT INTO sniper_signals
            (created_at, date, city, ticker, side, prob, ask_price,
             obs_max_f, rem_max_f, hours_remaining, mode, model_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (now_iso, local_date, ck, s.ticker, s.side,
              s.model_prob, s.market_price,
              ctx["obs"]["obs_max_f"] if ctx["obs"] else None,
              ctx["rem"]["rem_max_f"] if ctx["rem"] else None,
              ctx["rem"]["hours_remaining"] if ctx["rem"] else None,
              mode, MODEL_VERSION))
        logged += 1
    conn.commit()
    if logged < len(all_signals):
        print(f"  ({len(all_signals) - logged} duplicate signal(s) skipped — already logged today)")

    print(f"\n  {len(all_signals)} signal(s) [{mode.upper()}]:")
    for ck, s, _ in all_signals:
        print(f"    {s.ticker} {s.side.upper()} edge={s.edge:+.1%} — {s.description}")

    if mode != "live":
        print("\n  Dry run — signals logged for validation. "
              "Run `python sniper.py report` to track accuracy.")
        conn.close()
        return

    # ── Live execution path ─────────────────────────────────────────
    client = create_client_from_config()

    # True-up fills and cancel stale resting orders BEFORE committing new
    # capital — resting maker orders fill adversely (the price only reaches
    # them when the market has moved against the thesis).
    try:
        from pnl_tracker import init_pnl_tables, reconcile_fills
        init_pnl_tables(conn)
        reconcile_fills(conn, client)
    except Exception as e:
        print(f"  Warning: fill reconciliation failed: {e}")

    balance = client.get_balance().get("balance", 0) / 100
    # Percentage-based budget, same scheme as the day-ahead trader;
    # an explicit --budget still acts as a hard dollar ceiling.
    try:
        from config import MAX_RUN_EXPOSURE_PCT as _pct
    except ImportError:
        _pct = 0.25
    pct_budget = balance * _pct
    budget = min(budget, pct_budget) if budget is not None else pct_budget
    budget = min(budget, balance)
    if budget <= 0.5:
        print("  Insufficient balance.")
        conn.close()
        return

    signals_only = [s for _, s, _ in all_signals]
    existing = check_existing_positions(client, [s.ticker for s in signals_only])
    if existing:
        print(f"  Skipping existing positions: {list(existing.keys())}")
        signals_only = [s for s in signals_only if s.ticker not in existing]

    # Percentage-based sizing, same scheme as the day-ahead trader
    try:
        import config
        max_position = balance * getattr(config, "MAX_POSITION_PCT", 0.08)
        kelly = getattr(config, "KELLY_FRACTION", 0.25)
        max_contracts = getattr(config, "MAX_CONTRACTS_PER_ORDER", 15)
        max_open = getattr(config, "MAX_OPEN_POSITIONS", 6)
    except ImportError:
        max_position, kelly, max_contracts, max_open = balance * 0.08, 0.25, 15, 6

    # MAX_OPEN_POSITIONS counts TOTAL simultaneous positions across both
    # strategies, so open slots come off this run's allowance.
    from trader import count_open_positions
    n_open = count_open_positions(client)
    if n_open:
        max_open = max(0, max_open - n_open)
        if max_open == 0:
            print("  Position limit reached — no new positions.")
            conn.close()
            return

    orders = size_orders(signals_only, bankroll=balance, kelly_fraction=kelly,
                         max_position_dollars=max_position,
                         max_contracts=max_contracts,
                         max_total_dollars=budget, max_positions=max_open)
    if not orders:
        print("  Nothing sized to trade.")
        conn.close()
        return

    orders = optimize_with_orderbook(client, orders)
    if not orders:
        print("  All orders filtered by orderbook (thin/wide).")
        conn.close()
        return

    results = execute_orders(client, orders, dry_run=False)
    ok = [r for r in results if r.success]
    if ok:
        log_trade_results(ok)
        # Match on (date, ticker, side) — NOT created_at: a signal first
        # logged by an earlier (possibly dry) run today keeps that run's
        # created_at, and matching on it silently left live trades
        # unmarked. The sniper trades same-day contracts, so the signal's
        # date column equals the date embedded in the ticker. Upgrade mode
        # to 'live' too so the record shows what actually happened.
        for r in ok:
            sig_date = parse_contract_ticker(r.order.ticker).get("date")
            conn.execute("""UPDATE sniper_signals
                            SET traded = 1, mode = 'live'
                            WHERE date = ? AND ticker = ? AND side = ?""",
                         (sig_date, r.order.ticker, r.order.side))
        conn.commit()
    for r in results:
        tag = "OK" if r.success else f"FAILED: {r.error}"
        print(f"    [{tag}] {r.order.ticker} {r.order.side} x{r.order.contracts} "
              f"@ {r.order.price_cents}¢")
    conn.close()


def report():
    conn = sqlite3.connect(str(DB_PATH))
    init_sniper_table(conn)
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
    else:
        print(f"  (no verified signals under the current model yet — keep running dry)")
    print(f"  LIVE TRADING GATE:  {'PASSED — --auto will trade live' if s['passed'] else 'not yet passed — --auto stays dry'}")

    print(f"\n  Recent signals:")
    for r in conn.execute("""SELECT date, city, ticker, side, ROUND(prob,2), ask_price,
                                    mode, COALESCE(outcome,'?'),
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
        print(f"Verified {verify_signals(conn)} signal(s).")
        conn.close()
    else:
        report()
