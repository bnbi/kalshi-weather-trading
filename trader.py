"""
Auto-Trader
Takes trade signals from the edge calculator and executes them on Kalshi.

Key safety features:
    - Fractional Kelly sizing (never risk full Kelly — too aggressive)
    - Maximum position size per contract
    - Maximum total exposure across all positions
    - Maximum number of open positions
    - Dry-run mode (prints what it WOULD do without placing orders)
    - Only trades tomorrow's contracts (today's are too close to settlement)

Usage:
    python trader.py chicago                    # dry run (default)
    python trader.py chicago --live             # actually place orders
    python trader.py chicago --live --max-spend 10  # cap at $10 total
"""

from __future__ import annotations

import uuid
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta

from kalshi_client import KalshiClient, create_client_from_config
from find_edge import get_market_prices, calculate_edge, TradeSignal
from model import predict_all_for_city
from weather import CITIES
from pnl_tracker import log_trade_results
from orderbook import analyze_orderbook


# ── Configuration defaults (overridden by config.py if present) ────

DEFAULT_KELLY_FRACTION = 0.15       # 15% Kelly (was 25% — too aggressive)
DEFAULT_MAX_CONTRACTS = 5           # max contracts per order (was 50!)
DEFAULT_MAX_POSITION_DOLLARS = 2.0  # max cost per single trade (was 5)
DEFAULT_MAX_EXPOSURE_DOLLARS = 15.0 # max total money at risk (was 50)
DEFAULT_MAX_POSITIONS = 6           # max simultaneous open trades (was 10)
DEFAULT_MIN_EDGE = 0.07             # 7% minimum edge to trade (was 5%)


@dataclass
class TradeOrder:
    """A sized order ready to be placed."""
    ticker: str
    side: str           # 'yes' or 'no'
    action: str         # 'buy'
    contracts: int      # number of contracts to buy
    price_cents: int    # limit price in cents (1-99)
    cost_dollars: float # total cost = contracts * price / 100
    edge: float         # expected edge
    kelly_fraction: float  # what fraction of bankroll Kelly suggests
    signal: TradeSignal    # the original signal


@dataclass
class TradeResult:
    """Result of attempting to place an order."""
    order: TradeOrder
    success: bool
    order_id: str = None
    error: str = None


# ── Kelly criterion ────────────────────────────────────────────────

def kelly_size(model_prob: float, market_price: float,
               kelly_fraction: float = DEFAULT_KELLY_FRACTION) -> float:
    """
    Compute fractional Kelly bet size.

    Full Kelly for a binary bet:
        f* = (p * b - q) / b
    where:
        p = probability of winning
        b = net odds (payout / cost - 1) = (1 - price) / price for a $1 contract
        q = 1 - p

    We then multiply by kelly_fraction (e.g. 0.25) to be conservative.
    Returns fraction of bankroll to wager (0 to 1).
    """
    if market_price <= 0 or market_price >= 1:
        return 0.0

    p = model_prob
    q = 1 - p
    b = (1 - market_price) / market_price  # net odds

    if b <= 0:
        return 0.0

    full_kelly = (p * b - q) / b
    if full_kelly <= 0:
        return 0.0

    return full_kelly * kelly_fraction


def size_orders(signals: list[TradeSignal], bankroll: float,
                kelly_fraction: float = DEFAULT_KELLY_FRACTION,
                max_position_dollars: float = DEFAULT_MAX_POSITION_DOLLARS,
                max_contracts: int = DEFAULT_MAX_CONTRACTS,
                max_total_dollars: float = DEFAULT_MAX_EXPOSURE_DOLLARS,
                max_positions: int = DEFAULT_MAX_POSITIONS) -> list[TradeOrder]:
    """
    Convert trade signals into sized orders using Kelly criterion.

    Returns orders sorted by edge (best first), respecting all limits.
    """
    orders = []
    total_cost = 0.0

    for signal in signals:
        if len(orders) >= max_positions:
            break

        # Kelly sizing
        kf = kelly_size(signal.model_prob, signal.market_price, kelly_fraction)
        if kf <= 0:
            continue

        # Dollar amount to risk
        dollars_to_risk = bankroll * kf

        # Apply position size cap
        dollars_to_risk = min(dollars_to_risk, max_position_dollars)

        # Apply total exposure cap
        remaining_budget = max_total_dollars - total_cost
        if remaining_budget <= 0:
            break
        dollars_to_risk = min(dollars_to_risk, remaining_budget)

        # Convert to contracts
        price_cents = int(signal.market_price * 100)
        if price_cents <= 0 or price_cents >= 100:
            continue

        contracts = int(dollars_to_risk / signal.market_price)
        contracts = min(contracts, max_contracts)

        if contracts <= 0:
            continue

        actual_cost = contracts * signal.market_price

        orders.append(TradeOrder(
            ticker=signal.ticker,
            side=signal.side,
            action="buy",
            contracts=contracts,
            price_cents=price_cents,
            cost_dollars=actual_cost,
            edge=signal.edge,
            kelly_fraction=kf,
            signal=signal,
        ))

        total_cost += actual_cost

    return orders


# ── Order execution ────────────────────────────────────────────────

def execute_orders(client: KalshiClient, orders: list[TradeOrder],
                   dry_run: bool = True) -> list[TradeResult]:
    """
    Place orders on Kalshi (or simulate in dry-run mode).

    Returns a list of TradeResults showing what happened.
    """
    results = []

    for order in orders:
        if dry_run:
            results.append(TradeResult(
                order=order,
                success=True,
                order_id=f"DRY-{uuid.uuid4().hex[:8]}",
            ))
            continue

        # Live order
        try:
            # Build order params
            order_params = {
                "ticker": order.ticker,
                "side": order.side,
                "action": order.action,
                "count": order.contracts,
                "type": "limit",
                "client_order_id": str(uuid.uuid4()),
            }

            # Set price on the correct side
            if order.side == "yes":
                order_params["yes_price"] = order.price_cents
            else:
                order_params["no_price"] = order.price_cents

            response = client.create_order(**order_params)
            order_data = response.get("order", {})
            order_id = order_data.get("order_id", "unknown")

            results.append(TradeResult(
                order=order,
                success=True,
                order_id=order_id,
            ))

            # Be polite to the API
            time.sleep(0.2)

        except Exception as e:
            results.append(TradeResult(
                order=order,
                success=False,
                error=str(e),
            ))

    return results


# ── Orderbook optimization ─────────────────────────────────────

def optimize_with_orderbook(client: KalshiClient, orders: list[TradeOrder],
                            max_spread_cents: int = 10,
                            min_depth: int = 3) -> list[TradeOrder]:
    """
    Check orderbook for each order and:
    1. Skip markets that are too thin or have wide spreads
    2. Optimize limit price to get inside the spread
    3. Cap order size to available depth
    """
    print(f"\n  Analyzing orderbooks for {len(orders)} order(s)...")
    optimized = []

    for order in orders:
        analysis = analyze_orderbook(client, order.ticker, max_spread_cents, min_depth)
        time.sleep(0.15)

        # Check if market is tradeable
        if not analysis.is_tradeable:
            reasons = []
            if not analysis.is_liquid:
                reasons.append("thin")
            if not analysis.is_tight:
                spread = analysis.yes_spread_cents or analysis.no_spread_cents
                reasons.append(f"wide spread ({spread}¢)")
            print(f"    SKIP {order.ticker}: {', '.join(reasons)}")
            continue

        # Optimize price — try to get a better fill
        if order.side == "yes" and analysis.optimal_yes_price is not None:
            old_price = order.price_cents
            new_price = analysis.optimal_yes_price

            # Only use the optimal price if it's better (lower) than our current limit
            # and still maintains positive edge
            if new_price < old_price:
                # Recalculate edge with better price
                new_edge = order.signal.model_prob - (new_price / 100)
                if new_edge > 0.03:  # keep at least 3% edge
                    order.price_cents = new_price
                    order.cost_dollars = order.contracts * new_price / 100
                    order.edge = new_edge
                    print(f"    {order.ticker} YES: improved price {old_price}¢ → {new_price}¢ "
                          f"(saved {old_price - new_price}¢/contract)")

        elif order.side == "no" and analysis.optimal_no_price is not None:
            old_price = order.price_cents
            new_price = analysis.optimal_no_price

            if new_price < old_price:
                # signal.model_prob for NO signals is already P(NO)
                new_edge = order.signal.model_prob - (new_price / 100)
                if new_edge > 0.03:
                    order.price_cents = new_price
                    order.cost_dollars = order.contracts * new_price / 100
                    order.edge = new_edge
                    print(f"    {order.ticker} NO: improved price {old_price}¢ → {new_price}¢ "
                          f"(saved {old_price - new_price}¢/contract)")

        # Cap order size to available depth
        depth = analysis.yes_ask_depth if order.side == "yes" else analysis.no_ask_depth
        if depth > 0 and order.contracts > depth:
            old_qty = order.contracts
            order.contracts = max(1, depth)
            order.cost_dollars = order.contracts * order.price_cents / 100
            print(f"    {order.ticker}: capped {old_qty} → {order.contracts} contracts (book depth)")

        # Warn about slippage for larger orders
        if order.contracts >= 10 and analysis.slippage_10 > 2:
            print(f"    WARNING: {order.ticker} — {analysis.slippage_10:.1f}¢ slippage for 10 contracts")

        optimized.append(order)

    kept = len(optimized)
    skipped = len(orders) - kept
    if skipped > 0:
        print(f"\n  Orderbook filter: {kept} kept, {skipped} skipped")

    return optimized


# ── Pre-trade checks ───────────────────────────────────────────────

def check_existing_positions(client: KalshiClient, tickers: list[str]) -> dict:
    """
    Check if we already have positions in any of these markets.
    Returns: {ticker: position_count}
    """
    try:
        positions = client.get_positions(limit=200)
        pos_list = positions.get("market_positions", [])

        existing = {}
        for pos in pos_list:
            t = pos.get("ticker", "")
            if t in tickers:
                yes_count = pos.get("yes_count", 0) or 0
                no_count = pos.get("no_count", 0) or 0
                existing[t] = yes_count + no_count

        return existing
    except Exception as e:
        print(f"  Warning: could not check positions: {e}")
        return {}


def filter_tomorrow_only(signals: list[TradeSignal]) -> list[TradeSignal]:
    """
    Only keep signals for tomorrow's contracts.
    Today's contracts are too close to settlement — less edge, more risk.
    """
    tomorrow = (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d")

    # Convert tomorrow to Kalshi date format (e.g. 2026-05-06 -> 26MAY06)
    dt = datetime.strptime(tomorrow, "%Y-%m-%d")
    months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
              "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
    kalshi_date = f"{dt.year % 100:02d}{months[dt.month - 1]}{dt.day:02d}"

    filtered = [s for s in signals if kalshi_date in s.ticker]

    if len(filtered) < len(signals):
        skipped = len(signals) - len(filtered)
        print(f"  Filtered to tomorrow ({tomorrow}): {len(filtered)} signals "
              f"({skipped} skipped — today/other dates)")

    return filtered


# ── Pretty printing ────────────────────────────────────────────────

def print_trade_plan(orders: list[TradeOrder], bankroll: float, dry_run: bool) -> None:
    """Print the planned trades before execution."""
    mode = "DRY RUN" if dry_run else "LIVE"
    print(f"\n{'=' * 70}")
    print(f"  TRADE PLAN — {mode}")
    print(f"{'=' * 70}")
    print(f"  Bankroll: ${bankroll:.2f}")
    print(f"  Orders: {len(orders)}")
    total_cost = sum(o.cost_dollars for o in orders)
    print(f"  Total cost: ${total_cost:.2f}")
    print(f"  Weighted avg edge: {sum(o.edge * o.cost_dollars for o in orders) / total_cost:.1%}"
          if total_cost > 0 else "")

    print(f"\n  {'Ticker':<40} {'Side':<5} {'Qty':<5} {'Price':<7} {'Cost':<8} {'Edge':<7} {'Kelly'}")
    print(f"  {'-' * 85}")

    for o in orders:
        print(f"  {o.ticker:<40} {o.side:<5} {o.contracts:<5} "
              f"${o.price_cents / 100:.2f}   ${o.cost_dollars:<6.2f}  "
              f"{o.edge:>+5.1%}  {o.kelly_fraction:.1%}")


def print_results(results: list[TradeResult], dry_run: bool) -> None:
    """Print execution results."""
    mode = "DRY RUN" if dry_run else "LIVE"
    print(f"\n{'=' * 70}")
    print(f"  EXECUTION RESULTS — {mode}")
    print(f"{'=' * 70}")

    successes = [r for r in results if r.success]
    failures = [r for r in results if not r.success]

    print(f"  Placed: {len(successes)}  |  Failed: {len(failures)}")

    if successes:
        total = sum(r.order.cost_dollars for r in successes)
        print(f"  Total invested: ${total:.2f}")

    for r in successes:
        status = "OK" if not dry_run else "SIMULATED"
        print(f"    [{status}] {r.order.ticker} — {r.order.side.upper()} x{r.order.contracts} "
              f"@ ${r.order.price_cents}¢ — ID: {r.order_id}")

    for r in failures:
        print(f"    [FAILED] {r.order.ticker} — {r.error}")


# ── Main pipeline ──────────────────────────────────────────────────

def run_trading_pipeline(city_key: str, dry_run: bool = True,
                         max_spend: float = None,
                         min_edge: float = DEFAULT_MIN_EDGE,
                         kelly_fraction: float = DEFAULT_KELLY_FRACTION,
                         tomorrow_only: bool = True) -> list[TradeResult]:
    """
    Full trading pipeline: find edges → size → execute.

    1. Fetch markets and generate predictions
    2. Find edges above threshold
    3. Size positions with Kelly criterion
    4. Execute orders (or simulate in dry-run)
    """
    city = CITIES[city_key]

    # Load risk config from config.py if available
    try:
        import config
        kelly_fraction = getattr(config, "KELLY_FRACTION", kelly_fraction)
        max_position = getattr(config, "MAX_POSITION_SIZE_CENTS", 500) / 100
        max_exposure = getattr(config, "MAX_TOTAL_EXPOSURE_CENTS", 5000) / 100
        max_positions = getattr(config, "MAX_OPEN_POSITIONS", DEFAULT_MAX_POSITIONS)
    except ImportError:
        max_position = DEFAULT_MAX_POSITION_DOLLARS
        max_exposure = DEFAULT_MAX_EXPOSURE_DOLLARS
        max_positions = DEFAULT_MAX_POSITIONS

    if max_spend is not None:
        max_exposure = min(max_exposure, max_spend)

    # Step 1: Get markets and predictions
    print(f"Fetching {city.name} weather markets...")
    markets = get_market_prices(city.kalshi_series)
    print(f"  Found {len(markets)} open markets")

    print(f"\nGenerating predictions...")
    predictions = predict_all_for_city(city_key, markets)
    print(f"  Generated {len(predictions)} predictions")

    # Step 2: Find edges
    signals = calculate_edge(predictions, markets, min_edge=min_edge)
    print(f"\n  Found {len(signals)} signals with edge > {min_edge:.0%}")

    if not signals:
        print("  No profitable trades found. Market is efficient today.")
        return []

    # Step 3: Filter to tomorrow only (today's are too close to settlement)
    if tomorrow_only:
        signals = filter_tomorrow_only(signals)
        if not signals:
            print("  No tomorrow signals. Try --include-today for same-day contracts.")
            return []

    # Step 4: Get bankroll
    if dry_run:
        # Use a simulated bankroll in dry-run mode
        bankroll = 100.0
        print(f"\n  [DRY RUN] Using simulated bankroll: ${bankroll:.2f}")
    else:
        client = create_client_from_config()
        balance = client.get_balance()
        bankroll = balance.get("balance", 0) / 100  # API returns cents
        print(f"\n  Account balance: ${bankroll:.2f}")

        if bankroll <= 0:
            print("  ERROR: No funds available. Deposit money first.")
            return []

        # CRITICAL: Never try to spend more than available balance
        # This prevents overdraft when config allows higher exposure
        if max_exposure > bankroll:
            print(f"  Capping max exposure ${max_exposure:.2f} → ${bankroll:.2f} (available balance)")
            max_exposure = bankroll
        if max_position > bankroll:
            max_position = min(max_position, bankroll)

        # Check existing positions
        tickers = [s.ticker for s in signals]
        existing = check_existing_positions(client, tickers)
        if existing:
            print(f"  Already have positions in: {list(existing.keys())}")
            signals = [s for s in signals if s.ticker not in existing]

    # Step 5: Size positions
    orders = size_orders(
        signals,
        bankroll=bankroll,
        kelly_fraction=kelly_fraction,
        max_position_dollars=max_position,
        max_contracts=DEFAULT_MAX_CONTRACTS,
        max_total_dollars=max_exposure,
        max_positions=max_positions,
    )

    if not orders:
        print("  Positions too small to trade (Kelly suggests zero allocation).")
        return []

    # Step 6: Orderbook analysis — optimize prices, skip thin markets
    if not dry_run:
        orders = optimize_with_orderbook(client, orders)
        if not orders:
            print("  All orders filtered out by orderbook analysis (too thin/wide).")
            return []

    # Step 7: Show plan
    print_trade_plan(orders, bankroll, dry_run)

    # Step 8: Execute
    if dry_run:
        results = execute_orders(None, orders, dry_run=True)
    else:
        results = execute_orders(client, orders, dry_run=False)

        # Log live trades to P&L tracker
        successes = [r for r in results if r.success]
        if successes:
            log_trade_results(successes)
            print(f"\n  Logged {len(successes)} trade(s) to P&L tracker.")

    print_results(results, dry_run)

    return results


# ── CLI ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Auto-trade Kalshi weather contracts based on model edge"
    )
    parser.add_argument("city", choices=list(CITIES.keys()), help="City to trade")
    parser.add_argument("--live", action="store_true",
                        help="Actually place orders (default: dry run)")
    parser.add_argument("--max-spend", type=float, default=None,
                        help="Maximum total dollars to spend this run")
    parser.add_argument("--min-edge", type=float, default=DEFAULT_MIN_EDGE,
                        help=f"Minimum edge to trade (default: {DEFAULT_MIN_EDGE})")
    parser.add_argument("--kelly", type=float, default=None,
                        help="Kelly fraction override (default: from config or 0.25)")
    parser.add_argument("--include-today", action="store_true",
                        help="Also trade same-day contracts (risky)")
    args = parser.parse_args()

    kelly = args.kelly if args.kelly else DEFAULT_KELLY_FRACTION

    if args.live:
        print("\n  *** LIVE TRADING MODE ***")
        print("  Real orders will be placed on Kalshi.")
        print("  Press Ctrl+C within 5 seconds to cancel...\n")
        try:
            time.sleep(5)
        except KeyboardInterrupt:
            print("\n  Cancelled.")
            exit(0)

    run_trading_pipeline(
        city_key=args.city,
        dry_run=not args.live,
        max_spend=args.max_spend,
        min_edge=args.min_edge,
        kelly_fraction=kelly,
        tomorrow_only=not args.include_today,
    )
