"""
Walk-Forward Backtester (live code path)

Simulates the day-ahead strategy over the clean training history using the
SAME functions the live pipeline uses — model.compute_probability (integer-
settlement rounding cutoffs), find_edge.calculate_edge (market-prior
shrinkage, credibility filter, price floors, fee-net edges) and
trader.size_orders (fee-aware fractional Kelly with the percentage caps).
The previous version predated every one of those fixes (no rounding, no
fees, no shrinkage, half-open bracket settlement) and no longer described
the system.

How it works:
    1. Load the training-clean rows (lead-1 forecasts, official GHCND truth)
    2. For each day (after a minimum training window):
        a. Fit a Ridge model on ALL prior data (recency-weighted, no look-ahead)
        b. Predict the day's high; σ = std of the last 60 walk-forward residuals
        c. Build Kalshi-style contracts (2°F brackets + two tails) around the
           naive ensemble mean and price them as a naive market would
           (Normal at the naive mean, σ from model spread, ±3¢ half-spread)
        d. Run the live edge/sizing code and settle on the rounded actual
    3. Track cumulative P&L, win rate, max drawdown

What this can and cannot tell you:
    - It measures whether the corrected forecast + probability layer beats a
      NAIVE-ENSEMBLE market. Real Kalshi makers are far better than that, so
      dollar results are an upper bound on skill, not a P&L forecast.
    - It does validate the plumbing: calibration of the probability layer
      (reliability of the claimed probabilities), sizing behaviour, and the
      effect of the guardrails, on official settlement truth.

Usage:
    python backtest.py chicago               # backtest chicago
    python backtest.py chicago --chart        # with P&L chart
    python backtest.py --all                  # all cities
    python backtest.py chicago --start 2025-01-01  # custom start
"""

from __future__ import annotations

import json
import math
import sqlite3
import warnings
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")

from db_migrations import migrate_db
from find_edge import calculate_edge, kalshi_fee_per_contract
from model import ContractPrediction, compute_probability
from trader import size_orders
from train_model import load_training_data, build_features, recency_weights
from weather import CITIES

BOT_DIR = Path(__file__).parent
DB_PATH = BOT_DIR / "kalshi_data.db"

# Backtest parameters (mirror config.py's live defaults)
MIN_TRAINING_DAYS = 120    # minimum days before we start trading
KELLY_FRACTION = 0.25      # quarter-Kelly
MIN_EDGE = 0.05            # 5¢ minimum blended, fee-net edge (config.MIN_EDGE_CENTS)
MAX_POSITION_PCT = 0.08
MAX_RUN_EXPOSURE_PCT = 0.25
MAX_POSITIONS = 6
STARTING_BANKROLL = 100.0  # simulated starting capital
SIGMA_LOOKBACK = 60        # walk-forward residuals used for σ
SIGMA_FLOOR = 1.8          # matches EnsembleForecast.MIN_LIVE_STD
HALF_SPREAD = 0.03         # naive market quotes mid ± 3¢


@dataclass
class BacktestTrade:
    """A single simulated trade."""
    date: str
    contract: str           # ticker of the simulated contract
    side: str               # 'yes' or 'no'
    model_prob: float       # blended probability of the traded side
    market_price: float     # ask paid
    edge: float
    contracts: int
    cost: float
    actual_high: float
    won: bool
    profit: float


@dataclass
class BacktestResult:
    """Full backtest results."""
    city: str
    trades: list[BacktestTrade] = field(default_factory=list)
    daily_pnl: list[dict] = field(default_factory=list)

    # Summary stats
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    total_invested: float = 0.0
    roi: float = 0.0
    max_drawdown: float = 0.0
    brier_model: float = float("nan")
    brier_market: float = float("nan")
    avg_edge: float = 0.0
    days_tested: int = 0
    training_days: int = 0


def simulate_contracts(series: str, date_code: str, center: float) -> list[dict]:
    """
    Kalshi-style ladder for one day: a "< low" tail, 2°F brackets, and a
    "> high" tail, laid out around the naive forecast the way the exchange
    lists them. Returns parse-able tickers plus the fields calculate_edge
    and compute_probability read.
    """
    base = int(math.floor(center)) - 5   # lowest bracket bottom (odd offset ok)
    contracts = []
    contracts.append({"ticker": f"{series}-{date_code}-T{base}",
                      "info": {"type": "threshold", "threshold": float(base)},
                      "title": f"<{base}°", "strike_type": "less"})
    for lo in range(base, base + 10, 2):
        hi = lo + 1
        contracts.append({"ticker": f"{series}-{date_code}-B{lo + 0.5}",
                          "info": {"type": "bracket", "bracket_low": float(lo),
                                   "bracket_high": float(hi)},
                          "title": f"{lo}-{hi}°", "strike_type": "between"})
    top = base + 9
    contracts.append({"ticker": f"{series}-{date_code}-T{top}",
                      "info": {"type": "threshold", "threshold": float(top)},
                      "title": f">{top}°", "strike_type": "greater"})
    return contracts


def settles_yes(actual_high: float, contract: dict) -> bool:
    """Integer (rounded) settlement, exactly as Kalshi grades the CLI value."""
    high = round(actual_high)
    info = contract["info"]
    if info["type"] == "bracket":
        return info["bracket_low"] <= high <= info["bracket_high"]
    if contract["strike_type"] == "less":
        return high < info["threshold"]
    return high > info["threshold"]


def run_backtest(city: str, start_date: str = None,
                 min_training: int = MIN_TRAINING_DAYS) -> BacktestResult:
    conn = sqlite3.connect(str(DB_PATH))
    migrate_db(conn)
    df = load_training_data(conn, city)
    conn.close()

    if len(df) < min_training + 10:
        raise ValueError(f"Not enough data for {city}: {len(df)} days "
                         f"(need {min_training + 10})")

    if start_date:
        mask = df["date"] >= pd.Timestamp(start_date)
        first_test_idx = int(np.argmax(mask.values)) if mask.any() else len(df)
        actual_start = max(min_training, first_test_idx)
    else:
        actual_start = min_training

    result = BacktestResult(city=city, training_days=actual_start)
    bankroll = STARTING_BANKROLL
    peak_bankroll = bankroll
    max_dd = 0.0
    series = CITIES[city].kalshi_series
    months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
              "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

    features_all = build_features(df)
    X_all = features_all.values
    y_all = df["actual_high_f"].values
    dates = df["date"].values

    print(f"\n{'=' * 60}")
    print(f"  BACKTESTING — {city.upper()}  (live edge/sizing code, naive market)")
    print(f"{'=' * 60}")
    print(f"  Data: {len(df)} clean days ({df['date'].iloc[0].date()} to {df['date'].iloc[-1].date()})")
    print(f"  Training window: first {actual_start} days")
    print(f"  Test period: {len(df) - actual_start} days")
    print(f"  Starting bankroll: ${STARTING_BANKROLL:.2f}")

    residuals: list[float] = []
    probs_model: list[float] = []
    probs_market: list[float] = []
    outcomes: list[int] = []

    for t in range(actual_start, len(df)):
        d = pd.Timestamp(dates[t])
        date_str = d.strftime("%Y-%m-%d")
        date_code = f"{d.year % 100:02d}{months[d.month - 1]}{d.day:02d}"
        actual_high = float(y_all[t])
        row = df.iloc[t]

        # Train on all prior data (recency weighted, like the live trainer)
        X_train, y_train = X_all[:t], y_all[:t]
        col_std = np.std(X_train, axis=0)
        good_cols = col_std > 0.001
        model = Ridge(alpha=1.0)
        model.fit(X_train[:, good_cols], y_train,
                  sample_weight=recency_weights(df["date"].iloc[:t]))
        pred = float(model.predict(X_all[t:t + 1][:, good_cols])[0])
        if not np.isfinite(pred):
            continue

        # Walk-forward σ from recent out-of-sample residuals
        if len(residuals) >= 10:
            sigma = max(float(np.std(residuals[-SIGMA_LOOKBACK:])), SIGMA_FLOOR)
        else:
            sigma = 2.5
        residuals.append(actual_high - pred)

        # Naive market: mean of the raw sources, σ from their spread
        naive_mean = float(row[["gfs_forecast_f", "ecmwf_forecast_f",
                                "blend_forecast_f"]].mean())
        naive_sigma = max(float(row["model_spread"]) / 2 + 1.5, 2.0)

        contracts = simulate_contracts(series, date_code, naive_mean)
        predictions, markets = [], []
        for c in contracts:
            p_model = compute_probability(pred, sigma, 0.0, c["info"], c["title"],
                                          strike_type=c["strike_type"])
            p_mkt = compute_probability(naive_mean, naive_sigma, 0.0, c["info"],
                                        c["title"], strike_type=c["strike_type"])
            yes_ask = min(0.99, p_mkt + HALF_SPREAD)
            yes_bid = max(0.01, p_mkt - HALF_SPREAD)
            markets.append({
                "ticker": c["ticker"], "title": c["title"],
                "strike_type": c["strike_type"],
                "yes_ask_dollars": f"{yes_ask:.4f}",
                "yes_bid_dollars": f"{yes_bid:.4f}",
                "no_ask_dollars": f"{1 - yes_bid:.4f}",
            })
            predictions.append(ContractPrediction(
                ticker=c["ticker"], contract_type=c["info"]["type"],
                description=c["title"], model_probability=p_model,
                forecast_high=pred, error_std=sigma,
                threshold=c["info"].get("threshold"),
                bracket_low=c["info"].get("bracket_low"),
                bracket_high=c["info"].get("bracket_high")))
            # Reliability bookkeeping on the YES side of every contract
            probs_model.append(p_model)
            probs_market.append(p_mkt)
            outcomes.append(1 if settles_yes(actual_high, c) else 0)

        signals = calculate_edge(predictions, markets, min_edge=MIN_EDGE)
        orders = size_orders(signals, bankroll=bankroll,
                             kelly_fraction=KELLY_FRACTION,
                             max_position_dollars=bankroll * MAX_POSITION_PCT,
                             max_total_dollars=bankroll * MAX_RUN_EXPOSURE_PCT,
                             max_positions=MAX_POSITIONS)

        day_trades, day_cost = [], 0.0
        by_ticker = {c["ticker"]: c for c in contracts}
        for o in orders:
            c = by_ticker[o.ticker]
            yes_settled = settles_yes(actual_high, c)
            won = yes_settled if o.side == "yes" else not yes_settled
            price = o.price_cents / 100
            fee = kalshi_fee_per_contract(price) * o.contracts
            cost = o.contracts * price
            profit = (o.contracts * 1.0 - cost - fee) if won else (-cost - fee)
            day_trades.append(BacktestTrade(
                date=date_str, contract=o.ticker, side=o.side,
                model_prob=o.signal.model_prob, market_price=price,
                edge=o.edge, contracts=o.contracts, cost=cost,
                actual_high=actual_high, won=won, profit=profit))
            day_cost += cost

        day_pnl = sum(tr.profit for tr in day_trades)
        bankroll += day_pnl
        peak_bankroll = max(peak_bankroll, bankroll)
        drawdown = (peak_bankroll - bankroll) / peak_bankroll if peak_bankroll > 0 else 0
        max_dd = max(max_dd, drawdown)

        result.trades.extend(day_trades)
        if day_trades:
            result.daily_pnl.append({
                "date": date_str, "trades": len(day_trades), "invested": day_cost,
                "pnl": day_pnl, "bankroll": bankroll,
                "wins": sum(1 for tr in day_trades if tr.won),
                "losses": sum(1 for tr in day_trades if not tr.won),
            })

    # Summary statistics
    result.total_trades = len(result.trades)
    result.wins = sum(1 for tr in result.trades if tr.won)
    result.losses = result.total_trades - result.wins
    result.win_rate = result.wins / result.total_trades if result.total_trades else 0
    result.total_invested = sum(tr.cost for tr in result.trades)
    result.total_pnl = sum(tr.profit for tr in result.trades)
    result.roi = (result.total_pnl / result.total_invested * 100) if result.total_invested else 0
    result.max_drawdown = max_dd
    result.avg_edge = float(np.mean([tr.edge for tr in result.trades])) if result.trades else 0
    result.days_tested = len(df) - actual_start
    if outcomes:
        o = np.array(outcomes, dtype=float)
        result.brier_model = float(np.mean((np.array(probs_model) - o) ** 2))
        result.brier_market = float(np.mean((np.array(probs_market) - o) ** 2))
    return result


def print_backtest_results(result: BacktestResult) -> None:
    print(f"\n{'=' * 60}")
    print(f"  BACKTEST RESULTS — {result.city.upper()}")
    print(f"{'=' * 60}")
    print(f"\n  Period:           {result.days_tested} days tested "
          f"({result.training_days} training)")
    print(f"  Total trades:     {result.total_trades}")
    print(f"  Win rate:         {result.win_rate:.1%}")
    print(f"\n  {'─' * 40}")
    print(f"  Starting capital: ${STARTING_BANKROLL:.2f}")
    print(f"  Final capital:    ${STARTING_BANKROLL + result.total_pnl:.2f}")
    print(f"  Total P&L:        ${result.total_pnl:+.2f}  (vs a NAIVE market — upper bound)")
    print(f"  ROI on stakes:    {result.roi:+.1f}%")
    print(f"  Max drawdown:     {result.max_drawdown:.1%}")
    print(f"  Avg blended edge: {result.avg_edge:.1%}")
    print(f"\n  Probability layer (all simulated contracts, YES side):")
    print(f"    Brier model {result.brier_model:.4f} vs naive market {result.brier_market:.4f}"
          f"  → skill {1 - result.brier_model / result.brier_market:+.3f}")

    if result.daily_pnl:
        print(f"\n  {'─' * 40}")
        print(f"  Monthly breakdown:")
        print(f"  {'Month':<10} {'Trades':<8} {'Win%':<7} {'P&L':<10} {'Cumul.'}")
        print(f"  {'─' * 45}")
        monthly = {}
        for d in result.daily_pnl:
            m = monthly.setdefault(d["date"][:7], {"trades": 0, "wins": 0, "pnl": 0.0})
            m["trades"] += d["trades"]; m["wins"] += d["wins"]; m["pnl"] += d["pnl"]
        cumul = 0.0
        for month in sorted(monthly):
            m = monthly[month]
            cumul += m["pnl"]
            wr = m["wins"] / m["trades"] if m["trades"] else 0
            print(f"  {month:<10} {m['trades']:<8} {wr:<7.0%} ${m['pnl']:<+8.2f} ${cumul:+.2f}")


def generate_backtest_chart(results: list[BacktestResult]) -> Path:
    """Write a self-contained HTML chart of cumulative P&L per city."""
    chart_data = {}
    for r in results:
        cumul, series = 0.0, []
        for d in r.daily_pnl:
            cumul += d["pnl"]
            series.append({"date": d["date"], "pnl": round(cumul, 2)})
        chart_data[r.city] = series
    all_dates = sorted({d["date"] for s in chart_data.values() for d in s})
    aligned = {}
    for city, series in chart_data.items():
        lookup = {d["date"]: d["pnl"] for d in series}
        last, out = None, []
        for date in all_dates:
            if date in lookup:
                last = lookup[date]
            out.append(last)
        aligned[city] = out
    stats = {r.city: {"pnl": round(r.total_pnl, 2), "trades": r.total_trades,
                      "win_rate": round(r.win_rate, 3),
                      "brier_model": round(r.brier_model, 4),
                      "brier_market": round(r.brier_market, 4)} for r in results}
    html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8"><title>Backtest Results</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>body{{font-family:-apple-system,sans-serif;background:#f5f5f5;padding:24px;margin:0}}
.container{{max-width:900px;margin:0 auto}}.card{{background:#fff;border-radius:12px;border:1px solid #e5e5e5;padding:20px;margin-bottom:20px}}
table{{width:100%;border-collapse:collapse;font-size:14px}}td,th{{padding:6px;text-align:center}}</style></head><body>
<div class="container"><h1 style="font-size:20px;font-weight:500">Walk-forward backtest (live edge/sizing code vs a naive market)</h1>
<div class="card"><canvas id="chart" height="300"></canvas></div><div class="card" id="stats"></div></div>
<script>
const DATA={json.dumps(aligned)};const DATES={json.dumps(all_dates)};const STATS={json.dumps(stats)};
document.getElementById('stats').innerHTML='<table><tr><th>City</th><th>Trades</th><th>Win rate</th><th>P&L</th><th>Brier model</th><th>Brier market</th></tr>'+
Object.entries(STATS).map(([c,s])=>`<tr><td>${{c}}</td><td>${{s.trades}}</td><td>${{(s.win_rate*100).toFixed(1)}}%</td><td>$${{s.pnl.toFixed(2)}}</td><td>${{s.brier_model}}</td><td>${{s.brier_market}}</td></tr>`).join('')+'</table>';
new Chart(document.getElementById('chart').getContext('2d'),{{type:'line',data:{{labels:DATES,datasets:Object.entries(DATA).map(([c,s])=>({{label:c,data:s,fill:false,tension:0.3,pointRadius:0,borderWidth:2,spanGaps:true}}))}},
options:{{responsive:true,scales:{{y:{{title:{{display:true,text:'Cumulative P&L ($)'}}}}}}}}}});
</script></body></html>"""
    chart_path = BOT_DIR / "backtest_results.html"
    chart_path.write_text(html)
    return chart_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Walk-forward backtester")
    parser.add_argument("city", nargs="?", choices=list(CITIES.keys()),
                        help="City to backtest")
    parser.add_argument("--all", action="store_true", help="Backtest all cities")
    parser.add_argument("--start", type=str, default=None,
                        help="Start date for test period (YYYY-MM-DD)")
    parser.add_argument("--chart", action="store_true",
                        help="Generate and open P&L chart")
    parser.add_argument("--min-training", type=int, default=MIN_TRAINING_DAYS,
                        help=f"Minimum training days (default: {MIN_TRAINING_DAYS})")
    args = parser.parse_args()

    if not args.city and not args.all:
        parser.error("Specify a city or use --all")

    cities = list(CITIES.keys()) if args.all else [args.city]
    results = []
    for city in cities:
        result = run_backtest(city, start_date=args.start,
                              min_training=args.min_training)
        print_backtest_results(result)
        results.append(result)

    if args.chart:
        chart_path = generate_backtest_chart(results)
        print(f"\n  Chart saved to: {chart_path}")
        webbrowser.open(f"file://{chart_path}")

    if len(results) > 1:
        total_pnl = sum(r.total_pnl for r in results)
        total_trades = sum(r.total_trades for r in results)
        total_wins = sum(r.wins for r in results)
        total_invested = sum(r.total_invested for r in results)
        print(f"\n{'=' * 60}")
        print(f"  COMBINED RESULTS")
        print(f"{'=' * 60}")
        print(f"  Total P&L:    ${total_pnl:+.2f}")
        print(f"  Total trades: {total_trades}")
        if total_trades:
            print(f"  Win rate:     {total_wins / total_trades:.1%}")
        if total_invested:
            print(f"  ROI:          {total_pnl / total_invested * 100:+.1f}%")
