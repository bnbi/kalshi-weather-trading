"""
P&L Tracker
Logs every trade, checks settlement results, and reports cumulative performance.

Features:
    - Logs trades when placed (from trader.py)
    - Checks Kalshi API for settlement results
    - Computes: cumulative P&L, win rate, ROI, avg edge captured
    - Model calibration: are predicted probabilities accurate?
    - Daily/weekly performance breakdowns

Usage:
    python pnl_tracker.py summary              # overall stats
    python pnl_tracker.py trades               # list all trades
    python pnl_tracker.py update               # check for settlements
    python pnl_tracker.py daily                # daily P&L breakdown
    python pnl_tracker.py calibration          # model calibration check
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent / "kalshi_data.db"


# ── Database setup ─────────────────────────────────────────────────

def init_pnl_tables(conn: sqlite3.Connection) -> None:
    """Create P&L tracking tables."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            ticker TEXT NOT NULL,
            side TEXT NOT NULL,
            action TEXT NOT NULL,
            contracts INTEGER NOT NULL,
            price_cents INTEGER NOT NULL,
            cost_dollars REAL NOT NULL,
            model_prob REAL,
            edge REAL,
            kelly_fraction REAL,
            order_id TEXT,
            settled INTEGER DEFAULT 0,
            settlement_result TEXT,
            payout_dollars REAL,
            profit_dollars REAL,
            settled_at TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_trades_ticker
            ON trades(ticker);
        CREATE INDEX IF NOT EXISTS idx_trades_settled
            ON trades(settled);
        CREATE INDEX IF NOT EXISTS idx_trades_timestamp
            ON trades(timestamp);
    """)
    conn.commit()


# ── Trade logging ──────────────────────────────────────────────────

def log_trade(conn: sqlite3.Connection, ticker: str, side: str, action: str,
              contracts: int, price_cents: int, cost_dollars: float,
              model_prob: float = None, edge: float = None,
              kelly_fraction: float = None, order_id: str = None):
    """Log a trade when it's placed."""
    now = datetime.now(timezone.utc).isoformat()

    conn.execute("""
        INSERT INTO trades (
            timestamp, ticker, side, action, contracts, price_cents,
            cost_dollars, model_prob, edge, kelly_fraction, order_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (now, ticker, side, action, contracts, price_cents,
          cost_dollars, model_prob, edge, kelly_fraction, order_id))
    conn.commit()


# ── Settlement checking ────────────────────────────────────────────

def check_settlements(conn: sqlite3.Connection) -> int:
    """
    Check Kalshi API for settled markets and update trade records.
    Returns number of newly settled trades.
    """
    from kalshi_client import create_client_from_config

    # Get unsettled trades
    unsettled = conn.execute("""
        SELECT id, ticker, side, contracts, price_cents, cost_dollars
        FROM trades WHERE settled = 0
    """).fetchall()

    if not unsettled:
        print("  No unsettled trades to check.")
        return 0

    print(f"  Checking {len(unsettled)} unsettled trade(s)...")

    client = create_client_from_config()
    settled_count = 0

    # Group by ticker to minimize API calls
    tickers = set(row[1] for row in unsettled)

    for ticker in tickers:
        try:
            market_resp = client.get_market(ticker)
            market = market_resp.get("market", {})
            status = market.get("status", "")
            result = market.get("result", "")  # "yes" or "no" or ""

            if status not in ("settled", "finalized") or not result:
                continue

            # Update all trades for this ticker
            ticker_trades = [r for r in unsettled if r[1] == ticker]
            now = datetime.now(timezone.utc).isoformat()

            for trade_id, _, side, contracts, price_cents, cost in ticker_trades:
                # Did we win?
                won = (side == result)

                if won:
                    # Payout = contracts * $1 (each contract pays $1 if correct)
                    payout = contracts * 1.0
                    profit = payout - cost
                else:
                    payout = 0.0
                    profit = -cost

                conn.execute("""
                    UPDATE trades SET
                        settled = 1,
                        settlement_result = ?,
                        payout_dollars = ?,
                        profit_dollars = ?,
                        settled_at = ?
                    WHERE id = ?
                """, (result, payout, profit, now, trade_id))

                result_str = "WIN" if won else "LOSS"
                print(f"    [{result_str}] {ticker} — {side.upper()} x{contracts} — "
                      f"P&L: ${profit:+.2f}")
                settled_count += 1

        except Exception as e:
            print(f"    Error checking {ticker}: {e}")

    conn.commit()
    return settled_count


# ── Performance reporting ──────────────────────────────────────────

def get_summary(conn: sqlite3.Connection) -> dict:
    """Compute overall performance statistics."""
    stats = {}

    # Total trades
    row = conn.execute("SELECT COUNT(*) FROM trades").fetchone()
    stats["total_trades"] = row[0]

    # Settled trades
    row = conn.execute("SELECT COUNT(*) FROM trades WHERE settled = 1").fetchone()
    stats["settled_trades"] = row[0]

    # Unsettled (pending)
    stats["pending_trades"] = stats["total_trades"] - stats["settled_trades"]

    if stats["settled_trades"] == 0:
        return stats

    # Win/loss
    row = conn.execute("""
        SELECT COUNT(*) FROM trades WHERE settled = 1 AND profit_dollars > 0
    """).fetchone()
    stats["wins"] = row[0]
    stats["losses"] = stats["settled_trades"] - stats["wins"]
    stats["win_rate"] = stats["wins"] / stats["settled_trades"]

    # P&L
    row = conn.execute("""
        SELECT SUM(profit_dollars), SUM(cost_dollars), SUM(payout_dollars)
        FROM trades WHERE settled = 1
    """).fetchone()
    stats["total_pnl"] = row[0] or 0
    stats["total_invested"] = row[1] or 0
    stats["total_payout"] = row[2] or 0
    stats["roi"] = (stats["total_pnl"] / stats["total_invested"] * 100) if stats["total_invested"] > 0 else 0

    # Average edge and profit per trade
    row = conn.execute("""
        SELECT AVG(edge), AVG(profit_dollars), AVG(cost_dollars)
        FROM trades WHERE settled = 1
    """).fetchone()
    stats["avg_edge"] = row[0] or 0
    stats["avg_profit"] = row[1] or 0
    stats["avg_cost"] = row[2] or 0

    # Best and worst trades
    row = conn.execute("""
        SELECT ticker, profit_dollars FROM trades WHERE settled = 1
        ORDER BY profit_dollars DESC LIMIT 1
    """).fetchone()
    stats["best_trade"] = {"ticker": row[0], "profit": row[1]} if row else None

    row = conn.execute("""
        SELECT ticker, profit_dollars FROM trades WHERE settled = 1
        ORDER BY profit_dollars ASC LIMIT 1
    """).fetchone()
    stats["worst_trade"] = {"ticker": row[0], "profit": row[1]} if row else None

    # Current streak
    recent = conn.execute("""
        SELECT profit_dollars FROM trades WHERE settled = 1
        ORDER BY settled_at DESC
    """).fetchall()

    streak = 0
    if recent:
        direction = recent[0][0] > 0
        for row in recent:
            if (row[0] > 0) == direction:
                streak += 1
            else:
                break
        stats["streak"] = streak if direction else -streak
    else:
        stats["streak"] = 0

    return stats


def print_summary(conn: sqlite3.Connection) -> None:
    """Print overall performance summary."""
    stats = get_summary(conn)

    print(f"\n{'=' * 60}")
    print(f"  P&L SUMMARY")
    print(f"{'=' * 60}")

    print(f"\n  Total trades:     {stats['total_trades']}")
    print(f"  Settled:          {stats['settled_trades']}")
    print(f"  Pending:          {stats['pending_trades']}")

    if stats["settled_trades"] == 0:
        print(f"\n  No settled trades yet. Run 'python pnl_tracker.py update' after markets settle.")
        return

    print(f"\n  {'─' * 40}")
    print(f"  Wins:             {stats['wins']}")
    print(f"  Losses:           {stats['losses']}")
    print(f"  Win rate:         {stats['win_rate']:.1%}")

    streak = stats["streak"]
    streak_str = f"{abs(streak)}W" if streak > 0 else f"{abs(streak)}L"
    print(f"  Current streak:   {streak_str}")

    print(f"\n  {'─' * 40}")
    print(f"  Total invested:   ${stats['total_invested']:.2f}")
    print(f"  Total payout:     ${stats['total_payout']:.2f}")
    print(f"  Total P&L:        ${stats['total_pnl']:+.2f}")
    print(f"  ROI:              {stats['roi']:+.1f}%")

    print(f"\n  Avg edge (model): {stats['avg_edge']:.1%}")
    print(f"  Avg profit/trade: ${stats['avg_profit']:+.2f}")
    print(f"  Avg cost/trade:   ${stats['avg_cost']:.2f}")

    if stats["best_trade"]:
        print(f"\n  Best trade:       {stats['best_trade']['ticker']} (${stats['best_trade']['profit']:+.2f})")
    if stats["worst_trade"]:
        print(f"  Worst trade:      {stats['worst_trade']['ticker']} (${stats['worst_trade']['profit']:+.2f})")


def print_trades(conn: sqlite3.Connection, limit: int = 20) -> None:
    """Print recent trades."""
    trades = conn.execute("""
        SELECT timestamp, ticker, side, contracts, price_cents,
               cost_dollars, model_prob, edge, settled,
               settlement_result, profit_dollars
        FROM trades
        ORDER BY timestamp DESC
        LIMIT ?
    """, (limit,)).fetchall()

    if not trades:
        print("\n  No trades recorded yet.")
        return

    print(f"\n{'=' * 90}")
    print(f"  RECENT TRADES (last {limit})")
    print(f"{'=' * 90}")
    print(f"\n  {'Date':<12} {'Ticker':<35} {'Side':<4} {'Qty':<4} {'Cost':<7} {'Edge':<6} {'Result':<8} {'P&L'}")
    print(f"  {'─' * 85}")

    for t in trades:
        ts, ticker, side, contracts, price, cost, prob, edge, settled, result, profit = t
        date_str = ts[:10] if ts else "?"
        edge_str = f"{edge:.0%}" if edge else "?"

        if settled:
            result_str = "WIN" if profit > 0 else "LOSS"
            pnl_str = f"${profit:+.2f}"
        else:
            result_str = "pending"
            pnl_str = "—"

        print(f"  {date_str:<12} {ticker:<35} {side:<4} {contracts:<4} "
              f"${cost:<5.2f} {edge_str:<6} {result_str:<8} {pnl_str}")


def print_daily(conn: sqlite3.Connection) -> None:
    """Print daily P&L breakdown."""
    rows = conn.execute("""
        SELECT DATE(timestamp) as day,
               COUNT(*) as trades,
               SUM(cost_dollars) as invested,
               SUM(CASE WHEN settled = 1 THEN profit_dollars ELSE 0 END) as pnl,
               SUM(CASE WHEN settled = 1 AND profit_dollars > 0 THEN 1 ELSE 0 END) as wins,
               SUM(CASE WHEN settled = 1 THEN 1 ELSE 0 END) as settled
        FROM trades
        GROUP BY DATE(timestamp)
        ORDER BY day DESC
        LIMIT 30
    """).fetchall()

    if not rows:
        print("\n  No trades recorded yet.")
        return

    print(f"\n{'=' * 70}")
    print(f"  DAILY P&L")
    print(f"{'=' * 70}")
    print(f"\n  {'Date':<12} {'Trades':<8} {'Invested':<10} {'Settled':<9} {'Win%':<7} {'P&L':<10} {'Cumul.'}")
    print(f"  {'─' * 65}")

    cumulative = 0.0
    # Reverse to show chronological for cumulative calc
    for row in reversed(rows):
        day, trades, invested, pnl, wins, settled = row
        pnl = pnl or 0
        cumulative += pnl
        win_rate = f"{wins/settled:.0%}" if settled > 0 else "—"

        print(f"  {day:<12} {trades:<8} ${invested:<8.2f} {settled:<9} "
              f"{win_rate:<7} ${pnl:<+8.2f} ${cumulative:+.2f}")


def print_calibration(conn: sqlite3.Connection) -> None:
    """
    Check model calibration: do predicted probabilities match actual outcomes?
    Groups trades into probability buckets and compares predicted vs actual win rate.
    """
    rows = conn.execute("""
        SELECT model_prob, profit_dollars, price_cents / 100.0
        FROM trades
        WHERE settled = 1 AND model_prob IS NOT NULL
    """).fetchall()

    if len(rows) < 5:
        print("\n  Not enough settled trades for calibration analysis (need ≥5).")
        return

    # Bucket by model probability
    buckets = {}
    for prob, profit, _price in rows:
        # Round to nearest 10%
        bucket = round(prob * 10) / 10
        bucket = max(0.0, min(1.0, bucket))
        if bucket not in buckets:
            buckets[bucket] = {"count": 0, "wins": 0, "total_prob": 0}
        buckets[bucket]["count"] += 1
        buckets[bucket]["wins"] += 1 if profit > 0 else 0
        buckets[bucket]["total_prob"] += prob

    print(f"\n{'=' * 60}")
    print(f"  MODEL CALIBRATION")
    print(f"{'=' * 60}")
    print(f"\n  A well-calibrated model's predicted probability should")
    print(f"  match the actual win rate in each bucket.")
    print(f"\n  {'Predicted':<12} {'Actual':<10} {'Count':<8} {'Calibration'}")
    print(f"  {'─' * 45}")

    for bucket in sorted(buckets.keys()):
        b = buckets[bucket]
        actual = b["wins"] / b["count"]
        avg_pred = b["total_prob"] / b["count"]
        diff = actual - avg_pred

        # Visual indicator
        if abs(diff) < 0.05:
            indicator = "good"
        elif abs(diff) < 0.15:
            indicator = "ok"
        else:
            indicator = "off"

        print(f"  {avg_pred:<12.0%} {actual:<10.0%} {b['count']:<8} {indicator} ({diff:+.0%})")

    # ── Proper scoring rules, benchmarked against the market ───────
    # The market price of the traded side is itself a probability forecast.
    # A model only adds value if it beats that baseline on the same trades.
    import math

    def _scores(probs_outcomes):
        brier = sum((p - y) ** 2 for p, y in probs_outcomes) / len(probs_outcomes)
        eps = 1e-6
        logloss = -sum(y * math.log(max(p, eps)) + (1 - y) * math.log(max(1 - p, eps))
                       for p, y in probs_outcomes) / len(probs_outcomes)
        return brier, logloss

    outcomes = [1 if profit > 0 else 0 for _, profit, _ in rows]
    model_pairs = [(prob, y) for (prob, _, _), y in zip(rows, outcomes)]
    market_pairs = [(price, y) for (_, _, price), y in zip(rows, outcomes)]

    brier_m, ll_m = _scores(model_pairs)
    brier_mkt, ll_mkt = _scores(market_pairs)
    # Murphy skill score vs the market baseline (>0 means model beats market)
    skill = 1 - brier_m / brier_mkt if brier_mkt > 0 else float("nan")

    print(f"\n  Proper scoring rules (n={len(rows)} settled trades):")
    print(f"  {'':<18} {'Brier':<10} {'Log loss'}")
    print(f"  {'Model':<18} {brier_m:<10.4f} {ll_m:.4f}")
    print(f"  {'Market (price)':<18} {brier_mkt:<10.4f} {ll_mkt:.4f}")
    print(f"  Brier skill score vs market: {skill:+.3f} "
          f"({'model adds information' if skill > 0 else 'market is the better forecaster'})")
    print(f"  (0.25 Brier = uninformed coin flip; selection bias caveat: only traded")
    print(f"   contracts are scored, which conditions on model-market disagreement)")


# ── Integration with trader.py ─────────────────────────────────────

def log_trade_results(results: list, conn: sqlite3.Connection = None) -> None:
    """
    Log trade results from trader.py's execute_orders().
    Call this after orders are placed.

    results: list of TradeResult from trader.py
    """
    if conn is None:
        conn = sqlite3.connect(str(DB_PATH))
        init_pnl_tables(conn)
        close_after = True
    else:
        init_pnl_tables(conn)
        close_after = False

    for r in results:
        if not r.success:
            continue

        log_trade(
            conn=conn,
            ticker=r.order.ticker,
            side=r.order.side,
            action=r.order.action,
            contracts=r.order.contracts,
            price_cents=r.order.price_cents,
            cost_dollars=r.order.cost_dollars,
            model_prob=r.order.signal.model_prob if r.order.signal else None,
            edge=r.order.edge,
            kelly_fraction=r.order.kelly_fraction,
            order_id=r.order_id,
        )

    if close_after:
        conn.close()


# ── CLI ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="P&L Tracker for Kalshi trading bot")
    parser.add_argument("command", nargs="?", default="summary",
                        choices=["summary", "trades", "update", "daily", "calibration"],
                        help="What to show (default: summary)")
    parser.add_argument("--limit", type=int, default=20,
                        help="Max trades to show (for 'trades' command)")
    args = parser.parse_args()

    conn = sqlite3.connect(str(DB_PATH))
    init_pnl_tables(conn)

    if args.command == "summary":
        print_summary(conn)

    elif args.command == "trades":
        print_trades(conn, limit=args.limit)

    elif args.command == "update":
        print("Checking for settled markets...")
        count = check_settlements(conn)
        print(f"\n  Settled {count} trade(s).")
        if count > 0:
            print_summary(conn)

    elif args.command == "daily":
        print_daily(conn)

    elif args.command == "calibration":
        print_calibration(conn)

    conn.close()
