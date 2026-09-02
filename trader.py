"""
Auto-Trader
Takes trade signals from the edge calculator and executes them on Kalshi.

Key safety features:
    - Fractional Kelly sizing (never risk full Kelly — too aggressive)
    - Percentage-of-bankroll caps per position, per run, and in TOTAL
      across everything still open (both strategies share the ledger)
    - Maximum number of open positions (resting orders count)
    - Orders are sized to what can fill AT THE LIMIT PRICE, and anything
      still unfilled after a short wait is canceled — no order rests all
      day waiting to be filled only when the market moves against it
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
from pnl_tracker import log_trade_results, open_exposure_dollars
from orderbook import analyze_orderbook, fillable_size


# ── Configuration defaults (overridden by config.py if present) ────

DEFAULT_KELLY_FRACTION = 0.15       # fallback ONLY when config.py is absent.
                                    # With config.py present, KELLY_FRACTION
                                    # there is what actually trades (0.25 as
                                    # of 2026-09) — change it THERE, not here.
                                    # An explicit --kelly beats both.
DEFAULT_MAX_CONTRACTS = 15          # per-order cap; the binding limits are the
                                    # % position cap and fillable book size
DEFAULT_MAX_POSITIONS = 6           # max simultaneous open positions (incl.
                                    # resting orders), both strategies
DEFAULT_MIN_EDGE = 0.07             # 7% minimum edge to trade (CLI default;
                                    # scheduler uses config.MIN_EDGE_CENTS)
DEFAULT_MAX_POSITION_PCT = 0.08     # max 8% of bankroll on one position
DEFAULT_MAX_RUN_EXPOSURE_PCT = 0.25 # max 25% of bankroll deployed per run
DEFAULT_MAX_TOTAL_EXPOSURE_PCT = 0.40  # max 40% of bankroll open at once,
                                    # across all runs of both strategies
DEFAULT_FILL_WAIT_SECONDS = 20      # how long an order may rest before the
                                    # unfilled remainder is canceled
DEFAULT_IMPROVE_PRICES = False      # post inside the spread instead of
                                    # taking the ask (see optimize_with_orderbook)


def load_risk_config(kelly_override: float = None) -> dict:
    """Risk knobs from config.py with safe defaults."""
    try:
        import config
    except ImportError:
        config = None
    g = lambda name, default: getattr(config, name, default) if config else default
    return {
        "kelly_fraction": (kelly_override if kelly_override is not None
                           else g("KELLY_FRACTION", DEFAULT_KELLY_FRACTION)),
        "max_position_pct": g("MAX_POSITION_PCT", DEFAULT_MAX_POSITION_PCT),
        "max_run_exposure_pct": g("MAX_RUN_EXPOSURE_PCT", DEFAULT_MAX_RUN_EXPOSURE_PCT),
        "max_total_exposure_pct": g("MAX_TOTAL_EXPOSURE_PCT", DEFAULT_MAX_TOTAL_EXPOSURE_PCT),
        "max_positions": g("MAX_OPEN_POSITIONS", DEFAULT_MAX_POSITIONS),
        "max_contracts": g("MAX_CONTRACTS_PER_ORDER", DEFAULT_MAX_CONTRACTS),
        "fill_wait_seconds": g("FILL_WAIT_SECONDS", DEFAULT_FILL_WAIT_SECONDS),
        "improve_prices": bool(g("IMPROVE_PRICES", DEFAULT_IMPROVE_PRICES)),
    }


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
    fee_dollars: float = None       # actual exchange fee for the filled part
    filled_contracts: float = None  # contracts actually filled (None = unknown)
    fill_cost_dollars: float = None # actual cost of the filled part
    canceled_remainder: bool = False  # unfilled remainder was canceled
    note: str = None


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
                max_position_dollars: float = None,
                max_contracts: int = DEFAULT_MAX_CONTRACTS,
                max_total_dollars: float = None,
                max_positions: int = DEFAULT_MAX_POSITIONS) -> list[TradeOrder]:
    """
    Convert trade signals into sized orders using Kelly criterion.

    max_position_dollars / max_total_dollars: dollar caps for THIS run
    (None = uncapped). The pipelines derive them from the percentage
    settings and the live bankroll.

    Returns orders sorted by edge (best first), respecting all limits.
    """
    pos_cap = max_position_dollars if max_position_dollars is not None else float("inf")
    total_cap = max_total_dollars if max_total_dollars is not None else float("inf")

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
        # round(), not int(): 0.29*100 is 28.999999999999996 in floating
        # point, and int() placed those orders a cent BELOW the ask.
        price_cents = int(round(signal.market_price * 100))
        if price_cents <= 0 or price_cents >= 100:
            continue
        want = min(bankroll * kf, pos_cap)
        candidates.append((signal, kf, want, price_cents))

    total_want = sum(w for _, _, w, _ in candidates)
    scale = 1.0
    if total_want > total_cap > 0:
        scale = total_cap / total_want

    # ── Pass 2: allocate scaled amounts (budget guard for rounding) ─
    orders = []
    total_cost = 0.0

    for signal, kf, want, price_cents in candidates:
        remaining_budget = total_cap - total_cost
        if remaining_budget <= 0:
            break
        dollars_to_risk = min(want * scale, remaining_budget)

        # Size on the all-in cost per contract (price + fee) so the wagered
        # fraction of bankroll matches what Kelly computed.
        per_contract = price_cents / 100 + kalshi_fee_per_contract(price_cents / 100)
        contracts = int(dollars_to_risk / per_contract)
        contracts = min(contracts, max_contracts)

        # Floor: with a small bankroll, fractional Kelly often suggests less
        # than one contract's cost. Trade minimum size rather than sitting
        # out — but only when Kelly wanted at least HALF a contract (round
        # to nearest, capped at 2x the Kelly stake). An unconditional floor
        # overbet the weakest admitted signals by 5-10x at small bankrolls.
        if contracts == 0:
            one_contract_cost = price_cents / 100
            if (dollars_to_risk >= 0.5 * per_contract
                    and one_contract_cost <= pos_cap
                    and one_contract_cost <= remaining_budget):
                contracts = 1

        if contracts <= 0:
            continue

        actual_cost = contracts * price_cents / 100

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

def _f(x, default=0.0) -> float:
    try:
        return float(x) if x is not None and x != "" else default
    except (TypeError, ValueError):
        return default


def order_fill_state(order_data: dict) -> dict:
    """
    Normalize an order payload (CreateOrder response or GetOrder) into
    {filled, remaining, status, cost, fees}. Field names vary across API
    generations — try the fixed-point/_dollars names first, then legacy.
    """
    filled = _f(order_data.get("fill_count_fp") or order_data.get("fill_count"))
    remaining = order_data.get("remaining_count_fp")
    if remaining is None:
        remaining = order_data.get("remaining_count")
    remaining = _f(remaining, default=float("nan"))
    cost = (_f(order_data.get("taker_fill_cost_dollars"))
            + _f(order_data.get("maker_fill_cost_dollars")))
    if cost == 0 and order_data.get("taker_fill_cost") is not None:
        cost = (_f(order_data.get("taker_fill_cost"))
                + _f(order_data.get("maker_fill_cost"))) / 100.0
    fees = (_f(order_data.get("taker_fees_dollars"))
            + _f(order_data.get("maker_fees_dollars")))
    if fees == 0 and order_data.get("taker_fees") is not None:
        fees = (_f(order_data.get("taker_fees")) + _f(order_data.get("maker_fees"))) / 100.0
    if fees == 0 and filled > 0 and order_data.get("average_fee_paid"):
        fees = _f(order_data.get("average_fee_paid")) * filled
    return {
        "filled": filled,
        "remaining": remaining,
        "status": (order_data.get("status") or "").lower(),
        "cost": cost,
        "fees": fees,
    }


TERMINAL_ORDER_STATUSES = {"canceled", "cancelled", "executed", "expired"}


def settle_unfilled(client: KalshiClient, results: list[TradeResult],
                    fill_wait_seconds: float) -> None:
    """
    After placing orders: wait briefly, then cancel whatever has not filled
    and record the true fill for each result.

    A limit order that rests is adversely selected — it fills exactly when
    the market has moved against the model since it was placed. The old
    pipeline let orders rest until the NEXT day's run (the 4h "stale"
    cancel only ran once a day), so every partially-filled or
    price-improved order sat on the book all afternoon.
    """
    pending = [r for r in results if r.success and r.order_id
               and not str(r.order_id).startswith("DRY-")]
    if not pending:
        return
    if fill_wait_seconds > 0:
        time.sleep(fill_wait_seconds)

    for r in pending:
        try:
            state = order_fill_state(client.get_order(r.order_id).get("order", {}))
        except Exception as e:
            r.note = f"fill check failed: {e}"
            continue

        still_open = state["status"] not in TERMINAL_ORDER_STATUSES
        remaining = state["remaining"]
        unfilled = (remaining > 0) if remaining == remaining else \
            (state["filled"] < r.order.contracts)  # NaN remaining → infer

        if still_open and unfilled:
            try:
                client.cancel_order(r.order_id)
                r.canceled_remainder = True
                time.sleep(0.5)
                state = order_fill_state(client.get_order(r.order_id).get("order", {}))
            except Exception as e:
                r.note = f"cancel of unfilled remainder failed: {e}"
                # Leave filled_contracts unknown: reconcile_fills will
                # true it up next run and cancel if still resting.
                continue

        r.filled_contracts = state["filled"]
        r.fill_cost_dollars = state["cost"] if state["filled"] > 0 else 0.0
        if state["fees"] > 0:
            r.fee_dollars = state["fees"]
        if r.filled_contracts == 0:
            r.note = "no fill — canceled" if r.canceled_remainder else "no fill"
        elif r.canceled_remainder:
            r.note = (f"partial fill {state['filled']:g}/{r.order.contracts}, "
                      f"remainder canceled")


def execute_orders(client: KalshiClient, orders: list[TradeOrder],
                   dry_run: bool = True,
                   fill_wait_seconds: float = DEFAULT_FILL_WAIT_SECONDS,
                   cancel_unfilled: bool = True) -> list[TradeResult]:
    """
    Place orders on Kalshi (or simulate in dry-run mode).

    Returns a list of TradeResults showing what happened, including the
    true filled size once unfilled remainders have been canceled.
    """
    results = []

    for order in orders:
        if dry_run:
            results.append(TradeResult(
                order=order,
                success=True,
                order_id=f"DRY-{uuid.uuid4().hex[:8]}",
                filled_contracts=order.contracts,
                fill_cost_dollars=order.cost_dollars,
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
            state = order_fill_state(order_data)
            results.append(TradeResult(
                order=order,
                success=True,
                order_id=order_id,
                fee_dollars=state["fees"] if state["fees"] > 0 else None,
            ))

            # Be polite to the API
            time.sleep(0.2)

        except Exception as e:
            results.append(TradeResult(
                order=order,
                success=False,
                error=str(e),
            ))

    if not dry_run and cancel_unfilled:
        settle_unfilled(client, results, fill_wait_seconds)

    return results


# ── Orderbook optimization ─────────────────────────────────────

def optimize_with_orderbook(client: KalshiClient, orders: list[TradeOrder],
                            max_spread_cents: int = 10,
                            min_depth: int = 3,
                            improve_prices: bool = DEFAULT_IMPROVE_PRICES) -> list[TradeOrder]:
    """
    Check orderbook for each order and:
    1. Skip markets that are too thin or have wide spreads
    2. Optionally post inside the spread (improve_prices) — OFF by default:
       the fee-net edge already assumes paying the ask, and an improved
       order only fills when the market comes to it, i.e. against us.
    3. Cap order size to what can fill at OUR limit price
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

        # Optional price improvement — try to get a better fill
        if improve_prices:
            optimal = (analysis.optimal_yes_price if order.side == "yes"
                       else analysis.optimal_no_price)
            if optimal is not None and optimal < order.price_cents:
                old_price = order.price_cents
                # Recalculate edge with better price — net of the exchange
                # fee, same basis as the edge that admitted the signal.
                # signal.model_prob for NO signals is already P(NO).
                new_edge = (order.signal.model_prob - (optimal / 100)
                            - kalshi_fee_per_contract(optimal / 100))
                if new_edge > 0.03:  # keep at least 3% edge
                    order.price_cents = optimal
                    order.cost_dollars = order.contracts * optimal / 100
                    order.edge = new_edge
                    print(f"    {order.ticker} {order.side.upper()}: improved price "
                          f"{old_price}¢ → {optimal}¢ (saved {old_price - optimal}¢/contract)")

        # Cap order size to what can fill at our limit (0 = nothing resting
        # at or better than our price — the order would only rest).
        fillable = fillable_size(analysis, order.side, order.price_cents)
        if fillable <= 0 and not improve_prices:
            print(f"    SKIP {order.ticker}: no size at {order.price_cents}¢ "
                  f"(book moved)")
            continue
        if 0 < fillable < order.contracts:
            old_qty = order.contracts
            order.contracts = fillable
            order.cost_dollars = order.contracts * order.price_cents / 100
            print(f"    {order.ticker}: capped {old_qty} → {order.contracts} contracts "
                  f"(size at {order.price_cents}¢)")

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


def resting_order_tickers(client: KalshiClient) -> set:
    """Tickers with an order of ours still resting on the book."""
    try:
        resp = client.get_orders(status="resting", limit=200)
        return {o.get("ticker") for o in resp.get("orders", []) if o.get("ticker")}
    except Exception as e:
        print(f"  Warning: could not list resting orders: {e}")
        return set()


def check_existing_positions(client: KalshiClient, tickers: list[str]) -> dict:
    """
    Markets we already have exposure in: a filled position OR a resting
    order (the old check saw only positions, so a resting order and a new
    order on the same ticker could stack).
    Returns: {ticker: position_count}  (resting orders reported as 0.5)
    """
    existing = {}
    try:
        positions = client.get_positions(limit=200)
        for pos in positions.get("market_positions", []):
            t = pos.get("ticker", "")
            if t in tickers:
                count = position_size(pos)
                if count > 0:
                    existing[t] = count
    except Exception as e:
        print(f"  Warning: could not check positions: {e}")

    for t in resting_order_tickers(client):
        if t in tickers and t not in existing:
            existing[t] = 0.5
    return existing


def count_open_positions(client: KalshiClient) -> int | None:
    """
    Number of markets we currently hold ANY position or resting order in,
    or None if the API call fails. Used to make MAX_OPEN_POSITIONS mean what
    it says: a cap on total simultaneous positions, not just orders per run.
    """
    try:
        positions = client.get_positions(limit=200)
        held = {p.get("ticker") for p in positions.get("market_positions", [])
                if position_size(p) > 0}
    except Exception as e:
        print(f"  Warning: could not count open positions: {e}")
        return None
    return len(held | resting_order_tickers(client))


def exposure_limits(cash_balance: float, cfg: dict, max_spend: float = None) -> dict:
    """
    Dollar caps for this run from the percentage settings.

    bankroll = cash + cost of everything still open (positions valued at
    cost, from our own trade ledger). Kelly should see total wealth, not
    just the cash left after yesterday's orders.
    run cap   = MAX_RUN_EXPOSURE_PCT of bankroll, never more than cash
    total cap = MAX_TOTAL_EXPOSURE_PCT of bankroll minus what is already
                open — the sniper's hourly runs and the day-ahead run can
                no longer each spend 25% until half the bankroll is at risk
    """
    open_cost = open_exposure_dollars()
    bankroll = cash_balance + open_cost
    max_position = bankroll * cfg["max_position_pct"]
    run_cap = bankroll * cfg["max_run_exposure_pct"]
    total_room = bankroll * cfg["max_total_exposure_pct"] - open_cost
    run_cap = min(run_cap, max(total_room, 0.0), cash_balance)
    if max_spend is not None:
        run_cap = min(run_cap, max_spend)
    return {"bankroll": bankroll, "open_cost": open_cost,
            "max_position": max_position, "run_cap": run_cap,
            "total_room": total_room}


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

    filtered = [s for s in signals if f"-{kalshi_date}-" in s.ticker]

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
        total = sum(filled_cost(r) for r in successes)
        print(f"  Total invested: ${total:.2f}")

    for r in successes:
        status = "OK" if not dry_run else "SIMULATED"
        filled = r.filled_contracts
        qty = (f"{filled:g}/{r.order.contracts}" if filled is not None
               and filled != r.order.contracts else f"{r.order.contracts}")
        if filled == 0:
            status = "NO FILL"
        note = f" — {r.note}" if r.note else ""
        print(f"    [{status}] {r.order.ticker} — {r.order.side.upper()} x{qty} "
              f"@ {r.order.price_cents}¢ — ID: {r.order_id}{note}")

    for r in failures:
        print(f"    [FAILED] {r.order.ticker} — {r.error}")


def filled_cost(r: TradeResult) -> float:
    """Dollars actually committed by a result (planned cost if fill unknown)."""
    if r.filled_contracts is None:
        return r.order.cost_dollars
    if r.filled_contracts == 0:
        return 0.0
    if r.fill_cost_dollars:
        return r.fill_cost_dollars
    return r.filled_contracts * r.order.price_cents / 100


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
    predictions = predict_all_for_city(city_key, markets, log_decisions="trade")
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


def _size_and_execute(client, signals: list[TradeSignal], dry_run: bool,
                      cfg: dict, max_spend: float, label: str) -> list[TradeResult]:
    """Shared tail of both pipelines: bankroll → caps → size → book → execute."""
    if dry_run:
        bankroll = 100.0
        limits = {"bankroll": bankroll, "open_cost": 0.0,
                  "max_position": bankroll * cfg["max_position_pct"],
                  "run_cap": min(bankroll * cfg["max_run_exposure_pct"],
                                 max_spend if max_spend is not None else float("inf")),
                  "total_room": bankroll * cfg["max_total_exposure_pct"]}
        print(f"\n  [DRY RUN] Using simulated bankroll: ${bankroll:.2f}")
    else:
        cash = client.get_balance().get("balance", 0) / 100  # API returns cents
        print(f"\n  Account balance: ${cash:.2f}")
        if cash <= 0:
            print("  ERROR: No funds available.")
            return []
        limits = exposure_limits(cash, cfg, max_spend)
        bankroll = limits["bankroll"]
        print(f"  Bankroll (cash + ${limits['open_cost']:.2f} open at cost): "
              f"${bankroll:.2f}")

    max_positions = cfg["max_positions"]
    print(f"  Sizing: {cfg['kelly_fraction']:.0%} Kelly, position cap "
          f"${limits['max_position']:.2f} ({cfg['max_position_pct']:.0%}), run cap "
          f"${limits['run_cap']:.2f}{label}, total-open room ${limits['total_room']:.2f}")
    if limits["run_cap"] < 0.50:
        print("  Total-exposure cap reached — nothing to deploy this run.")
        return []

    if not dry_run:
        existing = check_existing_positions(client, [s.ticker for s in signals])
        if existing:
            print(f"  Already have positions/orders in: {list(existing.keys())}")
            signals = [s for s in signals if s.ticker not in existing]
        # MAX_OPEN_POSITIONS caps TOTAL simultaneous positions, so slots
        # already occupied by open positions come off this run's budget.
        n_open = count_open_positions(client)
        if n_open:
            max_positions = max(0, max_positions - n_open)
            print(f"  Open positions/orders: {n_open} — up to {max_positions} "
                  f"new position(s) this run")
            if max_positions == 0:
                print("  Position limit reached — no new positions.")
                return []

    orders = size_orders(
        signals, bankroll=bankroll, kelly_fraction=cfg["kelly_fraction"],
        max_position_dollars=limits["max_position"],
        max_contracts=cfg["max_contracts"],
        max_total_dollars=limits["run_cap"], max_positions=max_positions,
    )
    if not orders:
        print("  Positions too small to trade.")
        return []

    if not dry_run:
        orders = optimize_with_orderbook(client, orders,
                                         improve_prices=cfg["improve_prices"])
        if not orders:
            print("  All orders filtered out by orderbook analysis.")
            return []

    print_trade_plan(orders, bankroll, dry_run)
    results = execute_orders(client, orders, dry_run=dry_run,
                             fill_wait_seconds=cfg["fill_wait_seconds"])

    if not dry_run:
        successes = [r for r in results if r.success]
        if successes:
            n = log_trade_results(successes)
            print(f"\n  Logged {n} trade(s) to P&L tracker.")

    print_results(results, dry_run)
    return results


def run_global_pipeline(city_keys: list, dry_run: bool = True,
                        min_edge: float = DEFAULT_MIN_EDGE,
                        tomorrow_only: bool = True,
                        max_spend: float = None,
                        kelly_fraction: float = None) -> list[TradeResult]:
    """
    Cross-city pipeline: gather signals from EVERY city first, then size
    them globally, best edge first, under ONE run budget.

    This replaces the per-city budget split, which fragmented the run
    budget 7 ways before knowing where the signals were — on days when
    only one city had edges, just 1/7th of the intended capital deployed.
    """
    cfg = load_risk_config(kelly_fraction)

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

    client = None if dry_run else create_client_from_config()
    return _size_and_execute(client, all_signals, dry_run, cfg, max_spend,
                             label=" (global, not split per city)")


def run_trading_pipeline(city_key: str, dry_run: bool = True,
                         max_spend: float = None,
                         min_edge: float = DEFAULT_MIN_EDGE,
                         kelly_fraction: float = None,
                         tomorrow_only: bool = True) -> list[TradeResult]:
    """
    Full trading pipeline for ONE city: find edges → size → execute.

    kelly_fraction: None (default) uses config.KELLY_FRACTION; an explicit
    value (e.g. the CLI's --kelly) OVERRIDES config.
    """
    cfg = load_risk_config(kelly_fraction)

    try:
        signals = _gather_signals(city_key, min_edge, tomorrow_only)
    except Exception as e:
        print(f"  ERROR gathering {city_key}: {e}")
        return []
    if not signals:
        print("  No profitable trades found. Market is efficient today."
              if tomorrow_only else "  No profitable trades found.")
        return []

    client = None if dry_run else create_client_from_config()
    return _size_and_execute(client, signals, dry_run, cfg, max_spend, label="")


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
