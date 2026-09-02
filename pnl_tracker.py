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

# Bankroll at the start of the current model era (2026-08-18), taken from
# the trading log's balance check. Used to compute wealth growth %.
ERA_START_BANKROLL = 47.57


def estimate_fee(contracts: int, price_cents: int) -> float:
    """Kalshi trading fee: ceil_to_cent(0.07 * C * P * (1-P)), in dollars."""
    import math
    p = price_cents / 100.0
    if p <= 0 or p >= 1:
        return 0.0
    return math.ceil(0.07 * contracts * p * (1 - p) * 100) / 100


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

    # Migration: exchange fees (actual from V2 responses, else estimated)
    try:
        conn.execute("ALTER TABLE trades ADD COLUMN fee_dollars REAL")
        conn.commit()
    except sqlite3.OperationalError:
        pass  # column already exists

    # Migration: reconciled fill count (NULL = not yet reconciled)
    try:
        conn.execute("ALTER TABLE trades ADD COLUMN filled_contracts REAL")
        conn.commit()
    except sqlite3.OperationalError:
        pass

    # Counterfactual log: signals the guardrails rejected, verified at
    # settlement so the filters themselves stay falsifiable.
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS skipped_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            ticker TEXT NOT NULL,
            side TEXT NOT NULL,
            reason TEXT NOT NULL,          -- credibility / price_floor / bracket_yes
            model_prob REAL,               -- blended probability at skip time
            ask_price REAL,
            raw_gap REAL,                  -- raw model-market disagreement
            edge REAL,                     -- blended, fee-adjusted edge
            outcome TEXT,                  -- win/loss once verified
            hypo_profit REAL,              -- 1 contract at ask, net est. fee
            UNIQUE(ticker, side, reason)
        );
    """)
    conn.commit()


# ── Trade logging ──────────────────────────────────────────────────

def log_trade(conn: sqlite3.Connection, ticker: str, side: str, action: str,
              contracts: int, price_cents: int, cost_dollars: float,
              model_prob: float = None, edge: float = None,
              kelly_fraction: float = None, order_id: str = None,
              fee_dollars: float = None):
    """Log a trade when it's placed. fee_dollars: actual fee if known."""
    now = datetime.now(timezone.utc).isoformat()

    conn.execute("""
        INSERT INTO trades (
            timestamp, ticker, side, action, contracts, price_cents,
            cost_dollars, model_prob, edge, kelly_fraction, order_id,
            fee_dollars
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (now, ticker, side, action, contracts, price_cents,
          cost_dollars, model_prob, edge, kelly_fraction, order_id,
          fee_dollars))
    conn.commit()


# ── Fill reconciliation ────────────────────────────────────────────

# Orders still resting on the book after this many hours are canceled.
# A resting maker order is adversely selected: the price only reaches it
# when the market has moved AGAINST the model since the order was placed,
# so the stale limit fills exactly when the thesis has gone bad. The saved
# spread is captured on quick fills; anything older is pulled.
STALE_ORDER_HOURS = 4

# Order states that mean the order is off the book (nothing to cancel).
TERMINAL_ORDER_STATUSES = {"canceled", "cancelled", "executed", "expired"}


def _order_age_hours(timestamp: str) -> float:
    """Hours since a trade row's ISO timestamp (UTC). inf if unparseable."""
    try:
        placed = datetime.fromisoformat(timestamp)
        if placed.tzinfo is None:
            placed = placed.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - placed).total_seconds() / 3600
    except (TypeError, ValueError):
        return float("inf")


def reconcile_fills(conn: sqlite3.Connection, client=None) -> int:
    """
    True-up unsettled trades against Kalshi's actual order state.

    The optimizer sometimes posts maker orders inside the spread, which can
    rest, fill partially, or get canceled at market close. This replaces the
    optimistic assume-full-fill records with reality:
      - 0 filled + canceled  -> trade row deleted (it never happened)
      - 0 filled + resting   -> marked filled_contracts=0 (excluded from
                                settlement scoring until it fills or dies);
                                canceled outright once STALE_ORDER_HOURS old
      - partial/full fill    -> contracts, cost, and fees set to the actual
                                values from the exchange (incl. maker/taker
                                fee split, replacing estimates); any stale
                                unfilled remainder is canceled
    Returns number of trades adjusted.
    """
    rows = conn.execute("""
        SELECT id, order_id, contracts, cost_dollars, timestamp FROM trades
        WHERE settled = 0 AND order_id IS NOT NULL
          AND order_id NOT LIKE 'DRY-%'
    """).fetchall()
    if not rows:
        return 0

    if client is None:
        from kalshi_client import create_client_from_config
        client = create_client_from_config()

    adjusted = 0
    for tid, order_id, logged_contracts, logged_cost, placed_at in rows:
        try:
            o = client.get_order(order_id).get("order", {})
        except Exception as e:
            print(f"    fill-reconcile failed for {order_id}: {e}")
            continue
        if not o:
            continue
        try:
            filled = float(o.get("fill_count_fp") or o.get("fill_count") or 0)
            status = o.get("status", "")
            cost = (float(o.get("taker_fill_cost_dollars") or 0)
                    + float(o.get("maker_fill_cost_dollars") or 0))
            fees = (float(o.get("taker_fees_dollars") or 0)
                    + float(o.get("maker_fees_dollars") or 0))
        except (TypeError, ValueError):
            continue

        # Stale-order guard: pull anything still on the book after the
        # cutoff, then re-read its final state so the fills recorded below
        # are the true ones (a fill can race the cancel).
        still_open = status not in TERMINAL_ORDER_STATUSES
        if still_open and _order_age_hours(placed_at) >= STALE_ORDER_HOURS:
            try:
                client.cancel_order(order_id)
                print(f"    [STALE] canceled resting order {order_id} "
                      f"(> {STALE_ORDER_HOURS}h on the book)")
            except Exception as e:
                print(f"    stale-cancel failed for {order_id}: {e}")
            try:
                o2 = client.get_order(order_id).get("order", {})
                filled = float(o2.get("fill_count_fp")
                               or o2.get("fill_count") or filled)
                status = o2.get("status", status)
                cost = (float(o2.get("taker_fill_cost_dollars") or 0)
                        + float(o2.get("maker_fill_cost_dollars") or 0)) or cost
                fees = (float(o2.get("taker_fees_dollars") or 0)
                        + float(o2.get("maker_fees_dollars") or 0)) or fees
            except Exception:
                pass
            if filled == 0:
                # Nothing ever filled — remove the phantom row now rather
                # than waiting for the next pass to see the canceled state.
                conn.execute("DELETE FROM trades WHERE id = ?", (tid,))
                adjusted += 1
                continue

        if filled == 0:
            if status in TERMINAL_ORDER_STATUSES:
                conn.execute("DELETE FROM trades WHERE id = ?", (tid,))
                print(f"    [UNFILLED] order {order_id} {status} with no "
                      f"fill — phantom trade removed")
                adjusted += 1
            else:  # still resting — exclude from scoring for now
                conn.execute("""UPDATE trades SET filled_contracts = 0
                                WHERE id = ?""", (tid,))
            continue

        # Partial or full fill: record reality
        if (abs(filled - logged_contracts) > 0.001
                or abs(cost - logged_cost) > 0.005):
            print(f"    [RECONCILED] {order_id}: {logged_contracts} -> "
                  f"{filled:g} contracts, cost ${logged_cost:.2f} -> "
                  f"${cost:.2f}")
            adjusted += 1
        conn.execute("""
            UPDATE trades SET contracts = ?, cost_dollars = ?,
                              fee_dollars = ?, filled_contracts = ?
            WHERE id = ?
        """, (int(round(filled)), cost, fees, filled, tid))

    conn.commit()
    return adjusted


# ── Settlement checking ────────────────────────────────────────────

def check_settlements(conn: sqlite3.Connection) -> int:
    """
    Check Kalshi API for settled markets and update trade records.
    Returns number of newly settled trades.

    Always reconciles fills FIRST: scoring an assumed-full-fill record is
    permanent (settled=1 excludes it from reconciliation forever), so a
    partially-filled resting order settled before reconciliation would lock
    wrong contracts/cost/P&L into the books.
    """
    from kalshi_client import create_client_from_config

    client = create_client_from_config()

    try:
        reconcile_fills(conn, client)
    except Exception as e:
        print(f"  Warning: fill reconciliation failed: {e}")

    # Get unsettled trades (after reconciliation may have pruned/fixed rows)
    unsettled = conn.execute("""
        SELECT id, ticker, side, contracts, price_cents, cost_dollars,
               fee_dollars
        FROM trades
        WHERE settled = 0
          AND (filled_contracts IS NULL OR filled_contracts > 0)
    """).fetchall()

    if not unsettled:
        print("  No unsettled trades to check.")
        return 0

    print(f"  Checking {len(unsettled)} unsettled trade(s)...")

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

            for trade_id, _, side, contracts, price_cents, cost, fee in ticker_trades:
                # Did we win?
                won = (side == result)

                # Exchange fee: actual (recorded at fill) or estimated.
                # Fees are paid at execution regardless of outcome.
                if fee is None:
                    fee = estimate_fee(contracts, price_cents)

                if won:
                    # Payout = contracts * $1 (each contract pays $1 if correct)
                    payout = contracts * 1.0
                    profit = payout - cost - fee
                else:
                    payout = 0.0
                    profit = -cost - fee

                conn.execute("""
                    UPDATE trades SET
                        settled = 1,
                        settlement_result = ?,
                        payout_dollars = ?,
                        profit_dollars = ?,
                        fee_dollars = ?,
                        settled_at = ?
                    WHERE id = ?
                """, (result, payout, profit, fee, now, trade_id))

                result_str = "WIN" if won else "LOSS"
                print(f"    [{result_str}] {ticker} — {side.upper()} x{contracts} — "
                      f"P&L: ${profit:+.2f}")
                settled_count += 1

        except Exception as e:
            print(f"    Error checking {ticker}: {e}")

    conn.commit()
    return settled_count


# ── Performance reporting ──────────────────────────────────────────

def get_summary(conn: sqlite3.Connection, since: str = None) -> dict:
    """
    Compute overall performance statistics.

    since: only count trades on/after this date (e.g. '2026-08-18' scopes
    the stats to the current model era). Default: all trades.
    """
    stats = {}
    w = "WHERE timestamp >= ?" if since else "WHERE 1=1"
    p = (since,) if since else ()

    # Total trades
    row = conn.execute(f"SELECT COUNT(*) FROM trades {w}", p).fetchone()
    stats["total_trades"] = row[0]

    # Settled trades
    row = conn.execute(
        f"SELECT COUNT(*) FROM trades {w} AND settled = 1", p).fetchone()
    stats["settled_trades"] = row[0]

    # Unsettled (pending)
    stats["pending_trades"] = stats["total_trades"] - stats["settled_trades"]

    if stats["settled_trades"] == 0:
        return stats

    # Win/loss
    row = conn.execute("""
        SELECT COUNT(*) FROM trades {w} AND settled = 1 AND profit_dollars > 0
    """.format(w=w), p).fetchone()
    stats["wins"] = row[0]
    stats["losses"] = stats["settled_trades"] - stats["wins"]
    stats["win_rate"] = stats["wins"] / stats["settled_trades"]

    # P&L
    row = conn.execute("""
        SELECT SUM(profit_dollars), SUM(cost_dollars), SUM(payout_dollars)
        FROM trades {w} AND settled = 1
    """.format(w=w), p).fetchone()
    stats["total_pnl"] = row[0] or 0
    stats["total_invested"] = row[1] or 0
    stats["total_payout"] = row[2] or 0
    stats["roi"] = (stats["total_pnl"] / stats["total_invested"] * 100) if stats["total_invested"] > 0 else 0

    # Average edge and profit per trade
    row = conn.execute("""
        SELECT AVG(edge), AVG(profit_dollars), AVG(cost_dollars)
        FROM trades {w} AND settled = 1
    """.format(w=w), p).fetchone()
    stats["avg_edge"] = row[0] or 0
    stats["avg_profit"] = row[1] or 0
    stats["avg_cost"] = row[2] or 0

    # Total exchange fees paid (actual where known, else estimated)
    row = conn.execute("""
        SELECT SUM(COALESCE(fee_dollars, 0)) FROM trades {w} AND settled = 1
    """.format(w=w), p).fetchone()
    stats["total_fees"] = row[0] or 0

    # Wealth growth: net P&L relative to the era's starting bankroll,
    # overall and as a geometric daily rate. Only meaningful when the
    # window starts at the era boundary (deposits before that are unknown).
    if since:
        row = conn.execute(
            f"SELECT MIN(DATE(timestamp)) FROM trades {w}", p).fetchone()
        first_day = row[0]
        if first_day and ERA_START_BANKROLL > 0:
            days = max((datetime.now(timezone.utc).date()
                        - datetime.strptime(first_day, "%Y-%m-%d").date()).days, 1)
            growth = stats["total_pnl"] / ERA_START_BANKROLL
            stats["wealth_growth_pct"] = growth * 100
            stats["daily_growth_pct"] = ((1 + growth) ** (1 / days) - 1) * 100
            stats["growth_days"] = days

            # Latest settlement day: P&L that day as a % of the bankroll
            # going INTO that day (not an average — the actual day's move).
            row = conn.execute(f"""
                SELECT MAX(DATE(settled_at)) FROM trades {w} AND settled = 1
            """, p).fetchone()
            latest = row[0]
            if latest:
                day_pnl = conn.execute(f"""
                    SELECT COALESCE(SUM(profit_dollars), 0) FROM trades
                    {w} AND settled = 1 AND DATE(settled_at) = ?
                """, p + (latest,)).fetchone()[0]
                pnl_before = conn.execute(f"""
                    SELECT COALESCE(SUM(profit_dollars), 0) FROM trades
                    {w} AND settled = 1 AND DATE(settled_at) < ?
                """, p + (latest,)).fetchone()[0]
                bankroll_before = ERA_START_BANKROLL + pnl_before
                if bankroll_before > 0:
                    stats["latest_day"] = latest
                    stats["latest_day_pnl"] = day_pnl
                    stats["latest_day_growth_pct"] = day_pnl / bankroll_before * 100

    # Best and worst trades
    row = conn.execute("""
        SELECT ticker, profit_dollars FROM trades {w} AND settled = 1
        ORDER BY profit_dollars DESC LIMIT 1
    """.format(w=w), p).fetchone()
    stats["best_trade"] = {"ticker": row[0], "profit": row[1]} if row else None

    row = conn.execute("""
        SELECT ticker, profit_dollars FROM trades {w} AND settled = 1
        ORDER BY profit_dollars ASC LIMIT 1
    """.format(w=w), p).fetchone()
    stats["worst_trade"] = {"ticker": row[0], "profit": row[1]} if row else None

    # Current streak
    recent = conn.execute("""
        SELECT profit_dollars FROM trades {w} AND settled = 1
        ORDER BY settled_at DESC
    """.format(w=w), p).fetchall()

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


def print_calibration(conn: sqlite3.Connection, since: str = None) -> None:
    """
    Check model calibration: do predicted probabilities match actual outcomes?
    Groups trades into probability buckets and compares predicted vs actual win rate.

    since: only score trades on/after this date (e.g. '2026-08-18' for the
    current model era). Default: all settled trades — which mixes model
    regimes, so pass --since when judging the CURRENT model.
    """
    where = "WHERE settled = 1 AND model_prob IS NOT NULL"
    params = ()
    if since:
        where += " AND timestamp >= ?"
        params = (since,)
        print(f"\n  (scoring trades since {since} only)")

    rows = conn.execute(f"""
        SELECT model_prob, profit_dollars, price_cents / 100.0
        FROM trades
        {where}
    """, params).fetchall()

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
            fee_dollars=getattr(r, "fee_dollars", None),
        )

    if close_after:
        conn.close()


# ── Counterfactual tracking of filtered signals ────────────────────

def log_skipped_signals(skips: list, conn: sqlite3.Connection = None) -> int:
    """Record guardrail-rejected signals (from find_edge) for later scoring.
    Deduplicated per (ticker, side, reason)."""
    if not skips:
        return 0
    close_after = conn is None
    if conn is None:
        conn = sqlite3.connect(str(DB_PATH))
    init_pnl_tables(conn)
    now = datetime.now(timezone.utc).isoformat()
    n = 0
    for s in skips:
        cur = conn.execute("""
            INSERT OR IGNORE INTO skipped_signals
            (created_at, ticker, side, reason, model_prob, ask_price,
             raw_gap, edge)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (now, s["ticker"], s["side"], s["reason"], s["model_prob"],
              s["ask_price"], s["raw_gap"], s["edge"]))
        n += cur.rowcount
    conn.commit()
    if close_after:
        conn.close()
    return n


def verify_skipped_signals(conn: sqlite3.Connection) -> int:
    """Score unverified skipped signals against Kalshi's settled results."""
    from kalshi_client import create_client_from_config

    # Markets older than ~2 weeks get archived by Kalshi and can no longer
    # be fetched via get_market — mark those rows void instead of retrying
    # them on every run forever.
    conn.execute("""
        UPDATE skipped_signals SET outcome = 'void'
        WHERE outcome IS NULL
          AND created_at < datetime('now', '-14 days')
    """)
    conn.commit()

    rows = conn.execute("""
        SELECT id, ticker, side, ask_price FROM skipped_signals
        WHERE outcome IS NULL
    """).fetchall()
    if not rows:
        return 0

    client = create_client_from_config()
    verified = 0
    for sid, ticker, side, ask in rows:
        try:
            market = client.get_market(ticker).get("market", {})
            if market.get("status") not in ("settled", "finalized"):
                continue
            result = market.get("result", "")
            if result not in ("yes", "no"):
                continue
            won = (side == result)
            fee = estimate_fee(1, int(round(ask * 100)))
            profit = (1 - ask - fee) if won else (-ask - fee)
            conn.execute("""
                UPDATE skipped_signals SET outcome = ?, hypo_profit = ?
                WHERE id = ?
            """, ("win" if won else "loss", profit, sid))
            verified += 1
        except Exception as e:
            print(f"    skipped-signal verify failed for {ticker}: {e}")
    conn.commit()
    if verified:
        print(f"  Verified {verified} skipped signal(s) (counterfactuals)")
    return verified


def print_skipped_report(conn: sqlite3.Connection) -> None:
    """Would-have-been record of everything the guardrails rejected."""
    print(f"\n{'=' * 62}")
    print("  FILTERED-SIGNAL COUNTERFACTUALS (would-have-been record)")
    print(f"{'=' * 62}")
    rows = conn.execute("""
        SELECT reason, COUNT(*),
               SUM(outcome IN ('win','loss')),
               SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END),
               AVG(CASE WHEN outcome IN ('win','loss') THEN model_prob END),
               SUM(hypo_profit),
               SUM(CASE WHEN outcome IN ('win','loss') THEN ask_price ELSE 0 END)
        FROM skipped_signals GROUP BY reason ORDER BY 2 DESC
    """).fetchall()
    if not rows:
        print("  Nothing logged yet — accumulates from the next run onward.")
        return
    for reason, total, ver, wins, avg_p, pnl, staked in rows:
        ver = ver or 0
        print(f"\n  {reason}: {total} logged, {ver} verified")
        if ver:
            wr = (wins or 0) / ver
            roi = (pnl or 0) / staked * 100 if staked else 0
            print(f"    would-have-been: {wins}/{ver} wins ({wr:.0%}) vs "
                  f"claimed {avg_p:.0%} | hypo P&L ${pnl or 0:+.2f} "
                  f"({roi:+.0f}% ROI at 1 contract each)")
            if reason == "credibility" and ver >= 20:
                verdict = ("FILTER JUSTIFIED — skips would have lost"
                           if (pnl or 0) < 0 else
                           "FILTER MAY BE TOO STRICT — skips would have won; "
                           "consider raising MAX_CREDIBLE_EDGE")
                print(f"    → {verdict}")


# ── CLI ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="P&L Tracker for Kalshi trading bot")
    parser.add_argument("command", nargs="?", default="summary",
                        choices=["summary", "trades", "update", "daily",
                                 "calibration", "skipped"],
                        help="What to show (default: summary)")
    parser.add_argument("--since", type=str, default=None,
                        help="Only score trades on/after this date "
                             "(e.g. 2026-08-18 = current model era)")
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
        print_calibration(conn, since=args.since)

    elif args.command == "skipped":
        print_skipped_report(conn)

    conn.close()
