"""
Walk-Forward Backtester
Simulates what the trading bot would have earned over the past year+.

How it works:
    1. Load all historical data (700+ days of forecasts vs actuals)
    2. For each day (starting after a minimum training window):
        a. Train the model on ALL prior data (no look-ahead)
        b. Predict today's high using the trained model
        c. Simulate Kalshi-style bracket contracts
        d. Estimate "market" prices using the naive ensemble average
        e. Find edges and size bets using Kelly criterion
        f. Settle based on the actual observed high
    3. Track cumulative P&L, win rate, Sharpe ratio, max drawdown

Why this works without real Kalshi prices:
    - We assume the market prices contracts at the naive ensemble probability
    - Our edge comes from the trained model being more accurate
    - This is conservative: real markets may be LESS efficient than our proxy
    - Standard approach in quant finance (alpha = model vs benchmark)

Usage:
    python backtest.py chicago               # backtest chicago
    python backtest.py chicago --chart        # with P&L chart
    python backtest.py --all                  # all cities
    python backtest.py chicago --start 2025-01-01  # custom start
"""

from __future__ import annotations

import sqlite3
import json
import warnings
import webbrowser
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
from scipy import stats
from sklearn.linear_model import Ridge

# Suppress sklearn numerical warnings (from near-singular feature matrices)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")

from train_model import load_training_data, build_features
from weather import CITIES

BOT_DIR = Path(__file__).parent
DB_PATH = BOT_DIR / "kalshi_data.db"

# Backtest parameters
MIN_TRAINING_DAYS = 90     # minimum days before we start trading
KELLY_FRACTION = 0.25      # quarter-Kelly
MIN_EDGE = 0.05            # 5% minimum edge
MAX_BET_DOLLARS = 5.0      # max per contract
STARTING_BANKROLL = 100.0  # simulated starting capital
BRACKET_WIDTH = 2          # bracket contracts span 2°F (e.g. 55-56)


@dataclass
class BacktestTrade:
    """A single simulated trade."""
    date: str
    contract: str           # e.g. "B55-56" or "T>60"
    side: str               # 'yes' or 'no'
    model_prob: float       # our probability
    market_prob: float      # simulated market probability
    edge: float
    contracts: int
    price_cents: int
    cost: float
    actual_high: float
    won: bool
    payout: float
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
    sharpe_ratio: float = 0.0
    avg_edge: float = 0.0
    days_tested: int = 0
    training_days: int = 0


def simulate_bracket_contracts(temp_center: float, bracket_width: int = BRACKET_WIDTH) -> list[dict]:
    """
    Generate simulated bracket contracts around a temperature.
    Creates brackets covering a reasonable range (±15°F from center).
    """
    contracts = []
    # Create brackets from center-15 to center+15
    low = int(temp_center - 15)
    high = int(temp_center + 15)

    for bracket_low in range(low, high, bracket_width):
        bracket_high = bracket_low + bracket_width
        contracts.append({
            "type": "bracket",
            "label": f"B{bracket_low}-{bracket_high}",
            "low": bracket_low,
            "high": bracket_high,
        })

    # Add some threshold contracts
    for threshold in range(low + 5, high - 5, 5):
        contracts.append({
            "type": "threshold_below",
            "label": f"T<{threshold}",
            "threshold": threshold,
        })

    return contracts


def compute_contract_prob(forecast: float, std: float, contract: dict) -> float:
    """Compute probability of a contract paying YES given forecast distribution."""
    dist = stats.norm(loc=forecast, scale=std)

    if contract["type"] == "bracket":
        return dist.cdf(contract["high"]) - dist.cdf(contract["low"])
    elif contract["type"] == "threshold_below":
        return dist.cdf(contract["threshold"])
    elif contract["type"] == "threshold_above":
        return 1 - dist.cdf(contract["threshold"])
    return 0.5


def did_contract_win(actual_high: float, contract: dict, side: str) -> bool:
    """Check if a contract bet won based on actual temperature."""
    if contract["type"] == "bracket":
        in_bracket = contract["low"] <= actual_high < contract["high"]
        return in_bracket if side == "yes" else not in_bracket
    elif contract["type"] == "threshold_below":
        below = actual_high < contract["threshold"]
        return below if side == "yes" else not below
    elif contract["type"] == "threshold_above":
        above = actual_high > contract["threshold"]
        return above if side == "yes" else not above
    return False


def run_backtest(city: str, start_date: str = None,
                 min_training: int = MIN_TRAINING_DAYS) -> BacktestResult:
    """
    Run a walk-forward backtest for a city.

    For each day t:
    1. Train Ridge model on days 0..t-1
    2. Predict day t using trained model
    3. Generate bracket contracts, compute probabilities
    4. Estimate market prices using naive ensemble
    5. Find edges, size with Kelly, simulate trades
    6. Settle based on actual high
    """
    conn = sqlite3.connect(str(DB_PATH))
    df = load_training_data(conn, city)
    conn.close()

    if len(df) < min_training + 10:
        raise ValueError(f"Not enough data for {city}: {len(df)} days "
                         f"(need {min_training + 10})")

    # Filter by start date
    if start_date:
        start_dt = pd.Timestamp(start_date)
        # Find the index where we'd start testing
        # We still need min_training days before this
        mask = df["date"] >= start_dt
        first_test_idx = mask.idxmax() if mask.any() else len(df)
        actual_start = max(min_training, first_test_idx)
    else:
        actual_start = min_training

    result = BacktestResult(city=city, training_days=actual_start)
    bankroll = STARTING_BANKROLL
    peak_bankroll = bankroll
    max_dd = 0.0

    features_all = build_features(df)
    X_all = features_all.values
    y_all = df["actual_high_f"].values

    print(f"\n{'=' * 60}")
    print(f"  BACKTESTING — {city.upper()}")
    print(f"{'=' * 60}")
    print(f"  Data: {len(df)} days ({df['date'].iloc[0].date()} to {df['date'].iloc[-1].date()})")
    print(f"  Training window: first {actual_start} days")
    print(f"  Test period: {len(df) - actual_start} days")
    print(f"  Starting bankroll: ${STARTING_BANKROLL:.2f}")

    # Walk forward through each day
    for t in range(actual_start, len(df)):
        date_str = str(df["date"].iloc[t].date())
        actual_high = y_all[t]

        # Skip if missing data
        row = df.iloc[t]
        if pd.isna(row["gfs_forecast_f"]) or pd.isna(row["ecmwf_forecast_f"]) or pd.isna(row["blend_forecast_f"]):
            continue

        # Train on all prior data
        X_train = X_all[:t]
        y_train = y_all[:t]

        # Remove zero-variance columns (e.g. gfs_minus_blend always 0)
        col_std = np.std(X_train, axis=0)
        good_cols = col_std > 0.001
        X_train_clean = X_train[:, good_cols]

        model = Ridge(alpha=1.0)
        model.fit(X_train_clean, y_train)

        # Predict today
        X_today = X_all[t:t+1][:, good_cols]
        model_prediction = model.predict(X_today)[0]

        # Skip if prediction is NaN (numerical issues)
        if np.isnan(model_prediction) or np.isinf(model_prediction):
            continue

        # Compute residual std from recent predictions (last 30 days)
        lookback = min(30, t - 1)
        recent_X = X_all[t-lookback:t][:, good_cols]
        recent_preds = model.predict(recent_X)
        recent_residuals = y_all[t-lookback:t] - recent_preds
        model_std = max(float(np.std(recent_residuals)), 1.0)

        # Naive ensemble (what the "market" would use)
        naive_mean = float(row[["gfs_forecast_f", "ecmwf_forecast_f", "blend_forecast_f"]].mean())
        naive_std = max(float(row["model_spread"]) / 2 + 1.5, 2.0)

        # Generate contracts
        contracts = simulate_bracket_contracts(naive_mean)

        # Find edges
        day_trades = []
        day_cost = 0.0

        for contract in contracts:
            model_prob_yes = compute_contract_prob(model_prediction, model_std, contract)
            market_prob_yes = compute_contract_prob(naive_mean, naive_std, contract)

            # Check YES edge
            yes_edge = model_prob_yes - market_prob_yes
            # Check NO edge
            no_edge = (1 - model_prob_yes) - (1 - market_prob_yes)

            # Trade YES if we have edge
            if yes_edge >= MIN_EDGE and market_prob_yes > 0.01 and market_prob_yes < 0.99:
                price = market_prob_yes
                kelly_f = _kelly(model_prob_yes, price)
                if kelly_f > 0:
                    bet = min(bankroll * kelly_f * KELLY_FRACTION, MAX_BET_DOLLARS)
                    if bet < 0.01 or day_cost + bet > MAX_BET_DOLLARS * 3:
                        continue
                    num_contracts = max(1, int(bet / price))
                    cost = num_contracts * price
                    won = did_contract_win(actual_high, contract, "yes")

                    trade = BacktestTrade(
                        date=date_str, contract=contract["label"],
                        side="yes", model_prob=model_prob_yes,
                        market_prob=market_prob_yes, edge=yes_edge,
                        contracts=num_contracts, price_cents=int(price * 100),
                        cost=cost, actual_high=actual_high, won=won,
                        payout=num_contracts * 1.0 if won else 0.0,
                        profit=(num_contracts * 1.0 - cost) if won else -cost,
                    )
                    day_trades.append(trade)
                    day_cost += cost

            # Trade NO if we have edge
            elif -no_edge >= MIN_EDGE and market_prob_yes > 0.01 and market_prob_yes < 0.99:
                # Buying NO at (1 - market_prob_yes)
                no_price = 1 - market_prob_yes
                no_model_prob = 1 - model_prob_yes
                no_edge_val = no_model_prob - no_price
                if no_edge_val < MIN_EDGE:
                    continue
                kelly_f = _kelly(no_model_prob, no_price)
                if kelly_f > 0:
                    bet = min(bankroll * kelly_f * KELLY_FRACTION, MAX_BET_DOLLARS)
                    if bet < 0.01 or day_cost + bet > MAX_BET_DOLLARS * 3:
                        continue
                    num_contracts = max(1, int(bet / no_price))
                    cost = num_contracts * no_price
                    won = did_contract_win(actual_high, contract, "no")

                    trade = BacktestTrade(
                        date=date_str, contract=contract["label"],
                        side="no", model_prob=no_model_prob,
                        market_prob=no_price, edge=no_edge_val,
                        contracts=num_contracts, price_cents=int(no_price * 100),
                        cost=cost, actual_high=actual_high, won=won,
                        payout=num_contracts * 1.0 if won else 0.0,
                        profit=(num_contracts * 1.0 - cost) if won else -cost,
                    )
                    day_trades.append(trade)
                    day_cost += cost

        # Update bankroll
        day_pnl = sum(t.profit for t in day_trades)
        bankroll += day_pnl
        peak_bankroll = max(peak_bankroll, bankroll)
        drawdown = (peak_bankroll - bankroll) / peak_bankroll if peak_bankroll > 0 else 0
        max_dd = max(max_dd, drawdown)

        result.trades.extend(day_trades)
        if day_trades:
            result.daily_pnl.append({
                "date": date_str,
                "trades": len(day_trades),
                "invested": day_cost,
                "pnl": day_pnl,
                "bankroll": bankroll,
                "wins": sum(1 for t in day_trades if t.won),
                "losses": sum(1 for t in day_trades if not t.won),
            })

    # Compute summary statistics
    result.total_trades = len(result.trades)
    result.wins = sum(1 for t in result.trades if t.won)
    result.losses = result.total_trades - result.wins
    result.win_rate = result.wins / result.total_trades if result.total_trades > 0 else 0
    result.total_invested = sum(t.cost for t in result.trades)
    result.total_pnl = sum(t.profit for t in result.trades)
    result.roi = (result.total_pnl / result.total_invested * 100) if result.total_invested > 0 else 0
    result.max_drawdown = max_dd
    result.avg_edge = float(np.mean([t.edge for t in result.trades])) if result.trades else 0
    result.days_tested = len(df) - actual_start

    # Sharpe ratio (annualized from daily returns)
    if result.daily_pnl:
        daily_returns = [d["pnl"] / max(d["invested"], 0.01) for d in result.daily_pnl]
        if len(daily_returns) > 1 and np.std(daily_returns) > 0:
            result.sharpe_ratio = float(np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252))
        else:
            result.sharpe_ratio = 0.0

    return result


def _kelly(prob: float, price: float) -> float:
    """Quick Kelly calculation."""
    if price <= 0 or price >= 1:
        return 0.0
    b = (1 - price) / price
    if b <= 0:
        return 0.0
    f = (prob * b - (1 - prob)) / b
    return max(0.0, f)


def print_backtest_results(result: BacktestResult) -> None:
    """Print backtest summary."""
    print(f"\n{'=' * 60}")
    print(f"  BACKTEST RESULTS — {result.city.upper()}")
    print(f"{'=' * 60}")

    print(f"\n  Period:           {result.days_tested} days tested "
          f"({result.training_days} training)")
    print(f"  Total trades:     {result.total_trades}")
    print(f"  Win rate:         {result.win_rate:.1%}")

    print(f"\n  {'─' * 40}")
    print(f"  Starting capital: ${STARTING_BANKROLL:.2f}")
    final = STARTING_BANKROLL + result.total_pnl
    print(f"  Final capital:    ${final:.2f}")
    print(f"  Total P&L:        ${result.total_pnl:+.2f}")
    print(f"  ROI:              {result.roi:+.1f}%")

    print(f"\n  {'─' * 40}")
    print(f"  Sharpe ratio:     {result.sharpe_ratio:.2f}")
    print(f"  Max drawdown:     {result.max_drawdown:.1%}")
    print(f"  Avg edge:         {result.avg_edge:.1%}")
    print(f"  Total invested:   ${result.total_invested:.2f}")

    # Monthly breakdown
    if result.daily_pnl:
        print(f"\n  {'─' * 40}")
        print(f"  Monthly breakdown:")
        print(f"  {'Month':<10} {'Trades':<8} {'Win%':<7} {'P&L':<10} {'Cumul.'}")
        print(f"  {'─' * 45}")

        monthly = {}
        for d in result.daily_pnl:
            month = d["date"][:7]
            if month not in monthly:
                monthly[month] = {"trades": 0, "wins": 0, "pnl": 0.0}
            monthly[month]["trades"] += d["trades"]
            monthly[month]["wins"] += d["wins"]
            monthly[month]["pnl"] += d["pnl"]

        cumul = 0.0
        for month in sorted(monthly.keys()):
            m = monthly[month]
            cumul += m["pnl"]
            wr = m["wins"] / m["trades"] if m["trades"] > 0 else 0
            print(f"  {month:<10} {m['trades']:<8} {wr:<7.0%} ${m['pnl']:<+8.2f} ${cumul:+.2f}")


def generate_backtest_chart(results: list[BacktestResult]) -> Path:
    """Generate an HTML chart of backtest results."""
    chart_data = {}
    for r in results:
        cumul = 0.0
        series = []
        for d in r.daily_pnl:
            cumul += d["pnl"]
            series.append({"date": d["date"], "pnl": round(cumul, 2), "bankroll": round(d["bankroll"], 2)})
        chart_data[r.city] = series

    # Build a unified date axis so all cities align properly
    all_dates = sorted(set(
        d["date"] for series in chart_data.values() for d in series
    ))

    # For each city, create a lookup and fill in the cumulative P&L for every date
    aligned = {}
    for city, series in chart_data.items():
        lookup = {d["date"]: d["pnl"] for d in series}
        aligned_series = []
        last_val = None
        for date in all_dates:
            if date in lookup:
                last_val = lookup[date]
            if last_val is not None:
                aligned_series.append({"date": date, "pnl": last_val})
            else:
                aligned_series.append({"date": date, "pnl": None})
        aligned[city] = aligned_series

    data_json = json.dumps(aligned)
    colors = {"chicago": "#2563eb", "nyc": "#16a34a", "miami": "#dc2626"}
    colors_json = json.dumps(colors)
    dates_json = json.dumps(all_dates)

    # Compute summary stats for metrics
    summary_stats = {}
    for r in results:
        summary_stats[r.city] = {
            "pnl": round(r.total_pnl, 2),
            "trades": r.total_trades,
            "win_rate": round(r.win_rate, 3),
            "sharpe": round(r.sharpe_ratio, 2),
        }
    stats_json = json.dumps(summary_stats)

    html = f"""<!DOCTYPE html>
<html><head>
<meta charset="UTF-8">
<title>Backtest Results</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
body {{ font-family: -apple-system, sans-serif; background: #f5f5f5; padding: 24px; margin: 0; }}
.container {{ max-width: 900px; margin: 0 auto; }}
h1 {{ font-size: 22px; font-weight: 500; margin-bottom: 20px; }}
.card {{ background: #fff; border-radius: 12px; border: 1px solid #e5e5e5; padding: 20px; margin-bottom: 20px; }}
.metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin-bottom: 20px; }}
.metric {{ background: #fff; border-radius: 8px; padding: 14px; border: 1px solid #e5e5e5; }}
.metric-label {{ font-size: 12px; color: #888; text-transform: uppercase; }}
.metric-value {{ font-size: 20px; font-weight: 500; margin-top: 4px; }}
.positive {{ color: #16a34a; }}
.negative {{ color: #dc2626; }}
</style>
</head><body>
<div class="container">
<h1>Walk-forward backtest results</h1>
<div class="metrics" id="metrics"></div>
<div class="card"><canvas id="chart" height="300"></canvas></div>
<div class="card" id="city-stats"></div>
</div>
<script>
const DATA = {data_json};
const COLORS = {colors_json};
const DATES = {dates_json};
const STATS = {stats_json};
const STARTING = {STARTING_BANKROLL};

// Metrics
const metricsEl = document.getElementById('metrics');
let totalPnl = 0, totalTrades = 0;
for (const [city, s] of Object.entries(STATS)) {{
    totalPnl += s.pnl;
    totalTrades += s.trades;
}}
const metrics = [
    ['Cities', Object.keys(STATS).length, ''],
    ['Total trades', totalTrades, ''],
    ['Total P&L', '$' + totalPnl.toFixed(2), totalPnl >= 0 ? 'positive' : 'negative'],
    ['Final capital', '$' + (STARTING + totalPnl).toFixed(2), totalPnl >= 0 ? 'positive' : 'negative'],
];
metricsEl.innerHTML = metrics.map(([l,v,c]) =>
    '<div class="metric"><div class="metric-label">'+l+'</div><div class="metric-value '+c+'">'+v+'</div></div>'
).join('');

// Per-city stats
const cityEl = document.getElementById('city-stats');
cityEl.innerHTML = '<h2 style="font-size:16px;font-weight:500;margin-bottom:12px">Per-city breakdown</h2>' +
    '<table style="width:100%;border-collapse:collapse;font-size:14px">' +
    '<tr style="border-bottom:1px solid #e5e5e5"><th style="text-align:left;padding:8px">City</th><th>Trades</th><th>Win rate</th><th>P&L</th><th>Sharpe</th></tr>' +
    Object.entries(STATS).map(([city, s]) =>
        '<tr style="border-bottom:1px solid #f0f0f0"><td style="padding:8px;text-transform:capitalize">'+city+'</td>' +
        '<td style="text-align:center">'+s.trades+'</td>' +
        '<td style="text-align:center">'+(s.win_rate*100).toFixed(1)+'%</td>' +
        '<td style="text-align:center;color:'+(s.pnl>=0?'#16a34a':'#dc2626')+'">$'+s.pnl.toFixed(2)+'</td>' +
        '<td style="text-align:center">'+s.sharpe+'</td></tr>'
    ).join('') + '</table>';

// Chart — unified date axis, each city as separate dataset
const ctx = document.getElementById('chart').getContext('2d');
const datasets = Object.entries(DATA).map(([city, series]) => ({{
    label: city.charAt(0).toUpperCase() + city.slice(1),
    data: series.map(s => s.pnl),
    borderColor: COLORS[city] || '#666',
    fill: false,
    tension: 0.3,
    pointRadius: 0,
    borderWidth: 2,
    spanGaps: true,
}}));
new Chart(ctx, {{
    type: 'line',
    data: {{ labels: DATES, datasets }},
    options: {{
        responsive: true,
        plugins: {{ legend: {{ position: 'top' }} }},
        scales: {{
            x: {{ ticks: {{ maxTicksLimit: 12, font: {{ size: 11 }} }} }},
            y: {{ title: {{ display: true, text: 'Cumulative P&L ($)' }}, grid: {{ color: '#f0f0f0' }} }}
        }}
    }}
}});
</script></body></html>"""

    chart_path = BOT_DIR / "backtest_results.html"
    chart_path.write_text(html)
    return chart_path


# ── CLI ────────────────────────────────────────────────────────────

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

    # Combined summary if multiple cities
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
        print(f"  Win rate:     {total_wins/total_trades:.1%}" if total_trades > 0 else "")
        print(f"  ROI:          {total_pnl/total_invested*100:+.1f}%" if total_invested > 0 else "")
