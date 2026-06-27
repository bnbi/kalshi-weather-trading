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

import sqlite3
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import requests

from weather import CITIES, NWS_HEADERS, get_hourly_forecast
from model import parse_contract_ticker
from find_edge import get_market_prices, parse_price, TradeSignal
from trader import (size_orders, execute_orders, optimize_with_orderbook,
                    check_existing_positions)
from pnl_tracker import log_trade_results
from kalshi_client import create_client_from_config

BOT_DIR = Path(__file__).parent
DB_PATH = BOT_DIR / "kalshi_data.db"
KILL_SWITCH_FILE = BOT_DIR / "KILL_SWITCH"

# Settlement stations (must match what Kalshi settles on)
STATIONS = {"chicago": "KMDW", "nyc": "KNYC", "miami": "KMIA"}

# ── Strategy parameters ─────────────────────────────────────────────
MIN_LOCAL_HOUR = 13     # don't snipe before 1pm local — high not locked yet
MAX_LOCAL_HOUR = 21     # markets near close; settlement imminent
MIN_PROB = 0.80         # only bet sides we believe ≥80% — certainty is the edge
MIN_EDGE = 0.10         # require 10¢ of mispricing (thin books, real spread cost)
MAX_TRADES_PER_CITY = 2
DEFAULT_BUDGET = 6.0    # max $ per run across all cities
MC_SAMPLES = 20000

# Validation gate for --auto mode
VALIDATION_MIN_SIGNALS = 15
VALIDATION_MAX_CALIB_GAP = 0.15   # claimed prob may exceed realized win rate by ≤15pts


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
            hypo_profit REAL               -- profit of 1 contract at ask
        );
    """)
    conn.commit()


# ── Real-time observations from the settlement station ─────────────

def get_observed_max(city_key: str, local_date: str) -> dict | None:
    """
    Running max temperature today at the settlement station.
    Returns {'obs_max_f', 'n_obs', 'last_obs'} or None if unavailable.
    """
    station = STATIONS[city_key]
    tz = ZoneInfo(CITIES[city_key].timezone)
    midnight_local = datetime.strptime(local_date, "%Y-%m-%d").replace(tzinfo=tz)
    start = midnight_local.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    try:
        resp = requests.get(
            f"https://api.weather.gov/stations/{station}/observations",
            params={"start": start, "limit": 200},
            headers=NWS_HEADERS, timeout=15)
        resp.raise_for_status()
        features = resp.json().get("features", [])
    except Exception as e:
        print(f"  [{city_key}] obs fetch failed: {e}")
        return None

    temps_f, last_obs = [], None
    for f in features:
        props = f.get("properties", {})
        t = (props.get("temperature") or {}).get("value")
        ts = props.get("timestamp")
        if t is None or ts is None:
            continue
        # Only count observations whose LOCAL date matches the target date
        obs_local = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(tz)
        if obs_local.strftime("%Y-%m-%d") != local_date:
            continue
        temps_f.append(t * 9 / 5 + 32)
        if last_obs is None or ts > last_obs:
            last_obs = ts

    if len(temps_f) < 3:
        return None

    return {"obs_max_f": max(temps_f), "n_obs": len(temps_f), "last_obs": last_obs}


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


# ── Final-high distribution ─────────────────────────────────────────

def final_high_samples(obs_max: float, rem_max: float | None,
                       hours_remaining: int, rng=None) -> np.ndarray:
    """
    Monte Carlo distribution of the settlement (CLI) daily max.

        final = max(obs_max + spike, future_max)

    spike: the CLI max comes from continuous sensor data and typically reads
    0-2°F above the max of hourly METAR observations.
    future_max: Normal around the remaining-hours forecast max; σ grows with
    how many hours are left for the forecast to be wrong.
    """
    rng = rng or np.random.default_rng(42)
    spike = rng.choice([0.0, 1.0, 2.0], size=MC_SAMPLES, p=[0.50, 0.40, 0.10])
    locked = obs_max + spike

    if rem_max is None or hours_remaining == 0:
        return locked

    sigma = min(0.6 + 0.18 * hours_remaining, 3.0)
    future = rng.normal(rem_max, sigma, size=MC_SAMPLES)
    return np.maximum(locked, future)


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

def find_sniper_signals(city_key: str, local_date: str) -> tuple[list[TradeSignal], dict]:
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

    samples = final_high_samples(obs["obs_max_f"], rem["rem_max_f"], rem["hours_remaining"])

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
        p_yes = contract_prob_yes(samples, info, m.get("title", ""))
        if p_yes is None:
            continue

        yes_ask = parse_price(m.get("yes_ask_dollars", m.get("yes_ask")))
        no_ask = parse_price(m.get("no_ask_dollars", m.get("no_ask")))

        for side, p_side, ask in (("yes", p_yes, yes_ask), ("no", 1 - p_yes, no_ask)):
            if ask is None or ask <= 0.02 or ask >= 0.97:
                continue
            edge = p_side - ask
            if p_side >= MIN_PROB and edge >= MIN_EDGE:
                signals.append(TradeSignal(
                    ticker=ticker, side=side, action="buy",
                    model_prob=p_side, market_price=ask,
                    edge=edge, expected_value=edge,
                    description=f"SNIPE {side.upper()} @ ${ask:.2f} | "
                                f"P={p_side:.1%} | obs_max={obs['obs_max_f']:.1f}°F",
                ))

    signals.sort(key=lambda s: s.edge, reverse=True)
    return signals[:MAX_TRADES_PER_CITY], ctx


# ── Validation gate ─────────────────────────────────────────────────

def verify_signals(conn: sqlite3.Connection) -> int:
    """Score past signals against actual highs from daily_predictions."""
    rows = conn.execute("""
        SELECT s.id, s.ticker, s.side, s.ask_price, p.actual_high_f
        FROM sniper_signals s
        JOIN daily_predictions p ON p.date = s.date AND p.city = s.city
        WHERE s.outcome IS NULL AND p.actual_high_f IS NOT NULL
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
        profit = (1 - ask) if won else -ask
        conn.execute("UPDATE sniper_signals SET outcome = ?, hypo_profit = ? WHERE id = ?",
                     ("win" if won else "loss", profit, sid))
        n += 1
    conn.commit()
    return n


def validation_status(conn: sqlite3.Connection) -> dict:
    row = conn.execute("""
        SELECT COUNT(*), AVG(CASE WHEN outcome='win' THEN 1.0 ELSE 0.0 END),
               AVG(prob), SUM(hypo_profit), SUM(ask_price)
        FROM sniper_signals WHERE outcome IS NOT NULL
    """).fetchone()
    n, win_rate, avg_prob, profit, staked = row
    n = n or 0
    passed = bool(
        n >= VALIDATION_MIN_SIGNALS
        and (profit or 0) > 0
        and (avg_prob or 1) - (win_rate or 0) <= VALIDATION_MAX_CALIB_GAP
    )
    return {"n_verified": n, "win_rate": win_rate, "avg_claimed_prob": avg_prob,
            "hypo_profit": profit, "hypo_staked": staked, "passed": passed}


# ── Main run ────────────────────────────────────────────────────────

def run(cities: list[str] = None, mode: str = "dry", budget: float = DEFAULT_BUDGET):
    if KILL_SWITCH_FILE.exists():
        print("KILL SWITCH ACTIVE — sniper aborting.")
        return

    conn = sqlite3.connect(str(DB_PATH))
    init_sniper_table(conn)
    verify_signals(conn)  # opportunistically score old signals

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
            signals, ctx = find_sniper_signals(ck, local_date)
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

    # Log every signal (dry or live) for validation
    now_iso = datetime.now(timezone.utc).isoformat()
    for ck, s, ctx in all_signals:
        tz = ZoneInfo(CITIES[ck].timezone)
        conn.execute("""
            INSERT INTO sniper_signals
            (created_at, date, city, ticker, side, prob, ask_price,
             obs_max_f, rem_max_f, hours_remaining, mode)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (now_iso, datetime.now(tz).strftime("%Y-%m-%d"), ck, s.ticker, s.side,
              s.model_prob, s.market_price,
              ctx["obs"]["obs_max_f"] if ctx["obs"] else None,
              ctx["rem"]["rem_max_f"] if ctx["rem"] else None,
              ctx["rem"]["hours_remaining"] if ctx["rem"] else None,
              mode))
    conn.commit()

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
    balance = client.get_balance().get("balance", 0) / 100
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

    try:
        import config
        max_position = getattr(config, "MAX_POSITION_SIZE_CENTS", 200) / 100
        kelly = getattr(config, "KELLY_FRACTION", 0.15)
    except ImportError:
        max_position, kelly = 2.0, 0.15

    orders = size_orders(signals_only, bankroll=balance, kelly_fraction=kelly,
                         max_position_dollars=max_position,
                         max_total_dollars=budget)
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
        conn.execute("""UPDATE sniper_signals SET traded = 1
                        WHERE created_at = ? AND ticker IN ({})""".format(
            ",".join("?" * len(ok))), (now_iso, *[r.order.ticker for r in ok]))
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
    print(f"  Verified signals:   {s['n_verified']} (need ≥{VALIDATION_MIN_SIGNALS})")
    if s["n_verified"]:
        print(f"  Win rate:           {s['win_rate']:.0%} (claimed {s['avg_claimed_prob']:.0%})")
        print(f"  Hypothetical P&L:   ${s['hypo_profit']:+.2f} on ${s['hypo_staked']:.2f} staked")
    print(f"  LIVE TRADING GATE:  {'PASSED — --auto will trade live' if s['passed'] else 'not yet passed — --auto stays dry'}")

    print(f"\n  Recent signals:")
    for r in conn.execute("""SELECT date, city, ticker, side, ROUND(prob,2), ask_price,
                                    mode, COALESCE(outcome,'?') FROM sniper_signals
                             ORDER BY id DESC LIMIT 15"""):
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
    p_run.add_argument("--budget", type=float, default=DEFAULT_BUDGET)

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
