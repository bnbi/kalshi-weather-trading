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
from find_edge import (get_market_prices, calculate_edge, TradeSignal,
                       kalshi_fee_per_contract)
from model import predict_all_for_city
from weather import CITIES
from pnl_tracker import log_trade_results
from orderbook import analyze_orderbook


# ── Configuration defaults (overridden by config.py if present) ────

DEFAULT_KELLY_FRACTION = 0.15       # fallback ONLY when config.py is absent.
                                    # With config.py present, KELLY_FRACTION
                                    # there is what actually trades (0.25 as
                                    # of 2026-09) — change it THERE, not here.
                                    # An explicit --kelly beats both.
DEFAULT_MAX_CONTRACTS = 15          # per-order cap; real limits are the 8%
                                    # position cap and orderbook depth cap
                                    # (was 5, which began binding at ~$60
                                    # bankroll; fills are reconciled so
                                    # over-posting is accounting-safe)
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
    fee_dollars: float = None   # actual exchange fee from the V2 response
                                # (None when unknown, e.g. resting orders —
                                # estimated at settlement instead)


# ── Kelly criterion ────────────────────────────────────────────────

def kelly_size(model_prob: float, market_price: float,
               kelly_fraction: float = DEFAULT_KELLY_FRACTION,
               fee: float = 0.0) -> float:
    """
    Compute fractional Kelly bet size.

    Full Kelly for a binary bet:
        f* = (p * b - q) / b
    where:
        p = probability of winning
        b = net odds = (payout - cost) / cost per contract
        q = 1 - p

    fee: exchange fee per contract in dollars. It raises the cost basis and
    lowers the win payout, so ignoring it oversizes every bet by a few
    percent (and more near 50c where the fee peaks).

    We then multiply by kelly_fraction (e.g. 0.25) to be conservative.
    Returns fraction of bankroll to wager (0 to 1).
    """
    if market_price <= 0 or market_price >= 1:
        return 0.0

    cost = market_price + fee
    win = 1 - market_price - fee
    if cost <= 0 or win <= 0:
        return 0.0

    p = model_prob
    q = 1 - p
    b = win / cost  # net odds

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
    # ── Pass 1: what does Kelly want for each signal? ──────────────
    # Collect unconstrained-by-budget demands first so that, when the run
    # cap binds, we can scale ALL positions proportionally instead of
    # greedily filling early ones and truncating the last. Proportional
    # scaling preserves Kelly's relative allocation — including the balance
    # of hedged same-city bracket pairs, which greedy truncation distorted.
    candidates = []
    for signal in signals:
        if len(candidates) >= max_positions:
            break
        fee = kalshi_fee_per_contract(signal.market_price)
        kf = kelly_size(signal.model_prob, signal.market_price,
                        kelly_fraction, fee=fee)
        if kf <= 0:
            continue
        price_cents = int(signal.market_price * 100)
        if price_cents <= 0 or price_cents >= 100:
            continue
        want = min(bankroll * kf, max_position_dollars)
        candidates.append((signal, kf, want, price_cents))

    total_want = sum(w for _, _, w, _ in candidates)
    scale = 1.0
    if total_want > max_total_dollars > 0:
        scale = max_total_dollars / total_want

    # ── Pass 2: allocate scaled amounts (budget guard for rounding) ─
    orders = []
    total_cost = 0.0

    for signal, kf, want, price_cents in candidates:
        remaining_budget = max_total_dollars - total_cost
        if remaining_budget <= 0:
            break
        dollars_to_risk = min(want * scale, remaining_budget)

        # Size on the all-in cost per contract (price + fee) so the wagered
        # fraction of bankroll matches what Kelly computed.
        per_contract = signal.market_price + kalshi_fee_per_contract(signal.market_price)
        contracts = int(dollars_to_risk / per_contract)
        contracts = min(contracts, max_contracts)

        # Floor: with a small bankroll, fractional Kelly often suggests less
        # than one contract's cost. Trade minimum size rather than sitting
        # out — but only when Kelly wanted at least HALF a contract (round
        # to nearest, capped at 2x the Kelly stake). An unconditional floor
        # overbet the weakest admitted signals by 5-10x at small bankrolls.
        if contracts == 0:
            one_contract_cost = signal.market_price
            if (dollars_to_risk >= 0.5 * per_contract
                    and one_contract_cost <= max_position_dollars
                    and one_contract_cost <= remaining_budget):
                contracts = 1

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

            # V2 reports the actual fee for immediately-filled contracts.
            # Field names vary across API generations — try the current
            # *_dollars totals first, then the legacy per-contract average.
            # None is fine: reconcile_fills trues fees up from GetOrder.
            fee = None
            try:
                filled = float(order_data.get("fill_count_fp")
                               or order_data.get("fill_count") or 0)
                total_fees = (float(order_data.get("taker_fees_dollars") or 0)
                              + float(order_data.get("maker_fees_dollars") or 0))
                avg_fee = float(order_data.get("average_fee_paid") or 0)
                if total_fees > 0:
                    fee = round(total_fees, 4)
                elif filled > 0 and avg_fee > 0:
                    fee = round(avg_fee * filled, 4)
            except (TypeError, ValueError):
                pass

            results.append(TradeResult(
                order=order,
                success=True,
                order_id=order_id,
                fee_dollars=fee,
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
                # Recalculate edge with better price — net of the exchange
                # fee, same basis as the edge that admitted the signal.
                new_edge = (order.signal.model_prob - (new_price / 100)
                            - kalshi_fee_per_contract(new_price / 100))
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
                new_edge = (order.signal.model_prob - (new_price / 100)
                            - kalshi_fee_per_contract(new_price / 100))
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

def position_size(pos: dict) -> float:
    """
    Contracts held in a position row, handling API schema generations.
    Current payloads use position_fp (signed fixed-point string: + = YES,
    - = NO); older ones used position, or yes_count/no_count.
    """
    for key in ("position_fp", "position"):
        val = pos.get(key)
        if val is not None:
            try:
                return abs(float(val))
            except (TypeError, ValueError):
                pass
    return (pos.get("yes_count") or 0) + (pos.get("no_count") or 0)


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
                count = position_size(pos)
                if count > 0:
                    existing[t] = count

        return existing
    except Exception as e:
        print(f"  Warning: could not check positions: {e}")
        return {}


def count_open_positions(client: KalshiClient) -> int | None:
    """
    Number of markets we currently hold ANY position in, or None if the
    API call fails. Used to make MAX_OPEN_POSITIONS mean what it says:
    a cap on total simultaneous positions, not just orders per run.
    """
    try:
        positions = client.get_positions(limit=200)
        return sum(1 for p in positions.get("market_positions", [])
                   if position_size(p) > 0)
    except Exception as e:
        print(f"  Warning: could not count open positions: {e}")
        return None


def filter_tomorrow_only(signals: list[TradeSignal],
                         tz_name: str = None) -> list[TradeSignal]:
    """
    Only keep signals for tomorrow's contracts.
    Today's contracts are too close to settlement — less edge, more risk.

    tz_name: the city's IANA timezone. Without it, "tomorrow" is computed in
    UTC, which after ~5-8pm US local time points at the local day-after-
    tomorrow — dropping every genuine tomorrow signal in evening runs.
    """
    from zoneinfo import ZoneInfo
    now = datetime.now(ZoneInfo(tz_name)) if tz_name else datetime.now(timezone.utc)
    tomorrow = (now + timedelta(days=1)).strftime("%Y-%m-%d")

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
    if total_cost > 0:
        avg_edge = sum(o.edge * o.cost_dollars for o in orders) / total_cost
        print(f"  Weighted avg edge: {avg_edge:.1%}")

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
              f"@ {r.order.price_cents}¢ — ID: {r.order_id}")

    for r in failures:
        print(f"    [FAILED] {r.order.ticker} — {r.error}")


# ── Main pipeline ──────────────────────────────────────────────────

def _gather_signals(city_key: str, min_edge: float,
                    tomorrow_only: bool = True) -> list[TradeSignal]:
    """
    Steps 1-3 of the pipeline for one city: fetch markets, predict, find
    edges, log counterfactuals, filter to tomorrow. Returns trade signals.
    """
    city = CITIES[city_key]

    print(f"Fetching {city.name} weather markets...")
    markets = get_market_prices(city.kalshi_series)
    print(f"  Found {len(markets)} open markets")

    print(f"\nGenerating predictions...")
    predictions = predict_all_for_city(city_key, markets)
    print(f"  Generated {len(predictions)} predictions")

    signals = calculate_edge(predictions, markets, min_edge=min_edge)
    print(f"\n  Found {len(signals)} signals with edge > {min_edge:.0%}")

    skips = getattr(calculate_edge, "last_skipped", [])
    if skips:
        try:
            from pnl_tracker import log_skipped_signals
            n_new = log_skipped_signals(skips)
            print(f"  Logged {n_new} filtered signal(s) for counterfactual "
                  f"tracking ({len(skips)} seen)")
        except Exception as e:
            print(f"  Warning: could not log skipped signals: {e}")

    if tomorrow_only and signals:
        pre_filter = list(signals)
        signals = filter_tomorrow_only(signals, city.timezone)
        dropped = [s for s in pre_filter if s not in signals]
        if dropped:
            try:
                from pnl_tracker import log_skipped_signals
                log_skipped_signals([{
                    "ticker": s.ticker, "side": s.side,
                    "model_prob": s.model_prob, "ask_price": s.market_price,
                    "raw_gap": None, "edge": s.edge, "reason": "same_day",
                } for s in dropped])
            except Exception as e:
                print(f"  Warning: could not log same-day skips: {e}")

    return signals


def run_global_pipeline(city_keys: list, dry_run: bool = True,
                        min_edge: float = DEFAULT_MIN_EDGE,
                        tomorrow_only: bool = True,
                        max_spend: float = None) -> list[TradeResult]:
    """
    Cross-city pipeline: gather signals from EVERY city first, then size
    them globally, best edge first, under ONE run budget.

    This replaces the per-city budget split, which fragmented the run
    budget 7 ways before knowing where the signals were — on days when
    only one city had edges, just 1/7th of the intended capital deployed.
    """
    try:
        import config
        kelly_fraction = getattr(config, "KELLY_FRACTION", DEFAULT_KELLY_FRACTION)
        max_position_pct = getattr(config, "MAX_POSITION_PCT", 0.08)
        max_exposure_pct = getattr(config, "MAX_RUN_EXPOSURE_PCT", 0.25)
        max_positions = getattr(config, "MAX_OPEN_POSITIONS", DEFAULT_MAX_POSITIONS)
        max_contracts = getattr(config, "MAX_CONTRACTS_PER_ORDER", DEFAULT_MAX_CONTRACTS)
    except ImportError:
        kelly_fraction = DEFAULT_KELLY_FRACTION
        max_position_pct, max_exposure_pct = 0.08, 0.25
        max_positions = DEFAULT_MAX_POSITIONS
        max_contracts = DEFAULT_MAX_CONTRACTS

    # Gather signals across all cities
    all_signals: list[TradeSignal] = []
    for ck in city_keys:
        print(f"\n--- {CITIES[ck].name} ---")
        try:
            all_signals.extend(_gather_signals(ck, min_edge, tomorrow_only))
        except Exception as e:
            print(f"  ERROR gathering {ck}: {e}")

    if not all_signals:
        print("\nNo profitable trades found in any city today.")
        return []

    # Best edges get budget first, regardless of which city they're in
    all_signals.sort(key=lambda s: s.edge, reverse=True)
    print(f"\n{len(all_signals)} signal(s) across all cities")

    # Bankroll and percentage caps
    if dry_run:
        bankroll = 100.0
        client = None
        print(f"\n  [DRY RUN] Using simulated bankroll: ${bankroll:.2f}")
    else:
        client = create_client_from_config()
        bankroll = client.get_balance().get("balance", 0) / 100
        print(f"\n  Account balance: ${bankroll:.2f}")
        if bankroll <= 0:
            print("  ERROR: No funds available.")
            return []

    max_position = bankroll * max_position_pct
    max_exposure = min(bankroll * max_exposure_pct, bankroll)
    if max_spend is not None:
        max_exposure = min(max_exposure, max_spend)
    print(f"  Sizing: {kelly_fraction:.0%} Kelly, position cap "
          f"${max_position:.2f} ({max_position_pct:.0%}), run cap "
          f"${max_exposure:.2f} (global, not split per city)")

    if not dry_run:
        existing = check_existing_positions(client,
                                            [s.ticker for s in all_signals])
        if existing:
            print(f"  Already have positions in: {list(existing.keys())}")
            all_signals = [s for s in all_signals
                           if s.ticker not in existing]
        # MAX_OPEN_POSITIONS caps TOTAL simultaneous positions, so slots
        # already occupied by open positions come off this run's budget.
        n_open = count_open_positions(client)
        if n_open:
            max_positions = max(0, max_positions - n_open)
            print(f"  Open positions: {n_open} — up to {max_positions} "
                  f"new position(s) this run")
            if max_positions == 0:
                print("  Position limit reached — no new positions.")
                return []

    orders = size_orders(
        all_signals, bankroll=bankroll, kelly_fraction=kelly_fraction,
        max_position_dollars=max_position,
        max_contracts=max_contracts,
        max_total_dollars=max_exposure, max_positions=max_positions,
    )
    if not orders:
        print("  Positions too small to trade.")
        return []

    if not dry_run:
        orders = optimize_with_orderbook(client, orders)
        if not orders:
            print("  All orders filtered out by orderbook analysis.")
            return []

    print_trade_plan(orders, bankroll, dry_run)
    results = execute_orders(client, orders, dry_run=dry_run)

    if not dry_run:
        successes = [r for r in results if r.success]
        if successes:
            log_trade_results(successes)
            print(f"\n  Logged {len(successes)} trade(s) to P&L tracker.")

    print_results(results, dry_run)
    return results


def run_trading_pipeline(city_key: str, dry_run: bool = True,
                         max_spend: float = None,
                         min_edge: float = DEFAULT_MIN_EDGE,
                         kelly_fraction: float = None,
                         tomorrow_only: bool = True) -> list[TradeResult]:
    """
    Full trading pipeline: find edges → size → execute.

    1. Fetch markets and generate predictions
    2. Find edges above threshold
    3. Size positions with Kelly criterion
    4. Execute orders (or simulate in dry-run)

    kelly_fraction: None (default) uses config.KELLY_FRACTION; an explicit
    value (e.g. the CLI's --kelly) OVERRIDES config — previously config
    silently won and the flag did nothing.
    """
    city = CITIES[city_key]

    # Load risk config from config.py if available.
    # Sizing is percentage-of-bankroll: the dollar limits are computed
    # AFTER the bankroll is known (below), so they scale automatically.
    try:
        import config
        if kelly_fraction is None:
            kelly_fraction = getattr(config, "KELLY_FRACTION", DEFAULT_KELLY_FRACTION)
        max_position_pct = getattr(config, "MAX_POSITION_PCT", 0.08)
        max_exposure_pct = getattr(config, "MAX_RUN_EXPOSURE_PCT", 0.25)
        max_positions = getattr(config, "MAX_OPEN_POSITIONS", DEFAULT_MAX_POSITIONS)
        max_contracts = getattr(config, "MAX_CONTRACTS_PER_ORDER", DEFAULT_MAX_CONTRACTS)
    except ImportError:
        if kelly_fraction is None:
            kelly_fraction = DEFAULT_KELLY_FRACTION
        max_position_pct = 0.08
        max_exposure_pct = 0.25
        max_positions = DEFAULT_MAX_POSITIONS
        max_contracts = DEFAULT_MAX_CONTRACTS

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

    # Counterfactual log: record what the guardrails rejected so their
    # would-have-been outcomes get verified at settlement.
    skips = getattr(calculate_edge, "last_skipped", [])
    if skips:
        try:
            from pnl_tracker import log_skipped_signals
            n_new = log_skipped_signals(skips)
            print(f"  Logged {n_new} filtered signal(s) for counterfactual "
                  f"tracking ({len(skips)} seen)")
        except Exception as e:
            print(f"  Warning: could not log skipped signals: {e}")

    if not signals:
        print("  No profitable trades found. Market is efficient today.")
        return []

    # Step 3: Filter to tomorrow only (today's are too close to settlement)
    if tomorrow_only:
        pre_filter = list(signals)
        signals = filter_tomorrow_only(signals, city.timezone)
        dropped = [s for s in pre_filter if s not in signals]
        if dropped:
            # Same-day signals are a policy exclusion, not a math one —
            # track their would-have-been outcomes like other skips.
            try:
                from pnl_tracker import log_skipped_signals
                log_skipped_signals([{
                    "ticker": s.ticker, "side": s.side,
                    "model_prob": s.model_prob, "ask_price": s.market_price,
                    "raw_gap": None, "edge": s.edge, "reason": "same_day",
                } for s in dropped])
            except Exception as e:
                print(f"  Warning: could not log same-day skips: {e}")
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

    # Percentage caps -> dollar limits for THIS bankroll (scale-invariant).
    # max_spend (per-city budget from the scheduler) still applies on top.
    max_position = bankroll * max_position_pct
    max_exposure = bankroll * max_exposure_pct
    if max_spend is not None:
        max_exposure = min(max_exposure, max_spend)
    max_exposure = min(max_exposure, bankroll)  # never overdraft
    print(f"  Sizing: {kelly_fraction:.0%} Kelly, position cap "
          f"${max_position:.2f} ({max_position_pct:.0%}), run cap "
          f"${max_exposure:.2f}")

    if not dry_run:
        # Check existing positions
        tickers = [s.ticker for s in signals]
        existing = check_existing_positions(client, tickers)
        if existing:
            print(f"  Already have positions in: {list(existing.keys())}")
            signals = [s for s in signals if s.ticker not in existing]
        n_open = count_open_positions(client)
        if n_open:
            max_positions = max(0, max_positions - n_open)
            if max_positions == 0:
                print("  Position limit reached — no new positions.")
                return []

    # Step 5: Size positions
    orders = size_orders(
        signals,
        bankroll=bankroll,
        kelly_fraction=kelly_fraction,
        max_position_dollars=max_position,
        max_contracts=max_contracts,
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
                        help="Kelly fraction override (default: config.KELLY_FRACTION)")
    parser.add_argument("--include-today", action="store_true",
                        help="Also trade same-day contracts (risky)")
    args = parser.parse_args()

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
        kelly_fraction=args.kelly,
        tomorrow_only=not args.include_today,
    )
