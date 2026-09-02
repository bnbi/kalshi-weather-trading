"""
Orderbook Analysis
Analyzes market depth to find optimal entry prices and avoid thin markets.

Key concepts:
    - Spread: gap between best bid and best ask (wide = illiquid)
    - Depth: total contracts available at reasonable prices
    - Slippage: how much worse your fill price gets for larger orders
    - Fair value: midpoint between bid and ask, weighted by size

This module helps the trader:
    1. Skip markets where the spread eats the edge
    2. Find the right limit price (don't overpay)
    3. Size orders to what the book can actually fill
    4. Detect stale/empty books where nobody is trading
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from kalshi_client import KalshiClient


@dataclass
class OrderbookAnalysis:
    """Analysis of a market's orderbook."""
    ticker: str

    # Best prices
    best_yes_bid: int = None    # highest price someone will buy YES at (cents)
    best_yes_ask: int = None    # lowest price someone will sell YES at (cents)
    best_no_bid: int = None     # highest price someone will buy NO at (cents)
    best_no_ask: int = None     # lowest price someone will sell NO at (cents)

    # Spread
    yes_spread_cents: int = None  # ask - bid for YES side
    no_spread_cents: int = None

    # Depth (total contracts across the best 5 price levels)
    yes_bid_depth: int = 0
    yes_ask_depth: int = 0
    no_bid_depth: int = 0
    no_ask_depth: int = 0

    # Midpoint (fair value estimate)
    yes_midpoint: float = None
    no_midpoint: float = None

    # Quality flags
    is_liquid: bool = False       # enough depth to trade
    is_tight: bool = False        # spread narrow enough
    is_tradeable: bool = False    # both liquid and tight

    # Optimal prices
    optimal_yes_price: int = None  # suggested limit price for buying YES
    optimal_no_price: int = None   # suggested limit price for buying NO

    # Slippage for various sizes
    slippage_5: float = 0.0   # extra cost for 5 contracts
    slippage_10: float = 0.0  # extra cost for 10 contracts
    slippage_25: float = 0.0  # extra cost for 25 contracts

    # Raw book
    yes_levels: list = None
    no_levels: list = None


def parse_book_levels(raw_levels: list) -> list[tuple[int, int]]:
    """
    Parse orderbook levels from Kalshi API format.
    Input: [["0.4500", "128.36"], ...] — [price_dollars_str, qty_str]
    Output: [(45, 128), ...] — [(price_cents, qty_int)]
    """
    parsed = []
    for entry in raw_levels:
        price_dollars = float(entry[0])
        qty = float(entry[1])
        price_cents = int(round(price_dollars * 100))
        qty_int = int(qty)
        if qty_int > 0:
            parsed.append((price_cents, qty_int))
    return parsed


def analyze_orderbook(client: KalshiClient, ticker: str,
                      max_spread_cents: int = 10,
                      min_depth: int = 5) -> OrderbookAnalysis:
    """
    Fetch and analyze the orderbook for a market.

    max_spread_cents: maximum acceptable bid-ask spread
    min_depth: minimum contracts at best price to consider liquid
    """
    result = OrderbookAnalysis(ticker=ticker)

    try:
        raw = client.get_orderbook(ticker)
        # API returns under "orderbook_fp" with "yes_dollars" and "no_dollars"
        ob = raw.get("orderbook_fp", raw.get("orderbook", {}))
    except Exception as e:
        print(f"    Orderbook fetch failed for {ticker}: {e}")
        return result

    # Parse levels: lists of [price_str, qty_str]
    yes_raw = ob.get("yes_dollars", ob.get("yes", []))
    no_raw = ob.get("no_dollars", ob.get("no", []))

    # Convert to [(price_cents, qty_int)] sorted by price descending (best bids first)
    yes_bids = sorted(parse_book_levels(yes_raw), key=lambda x: x[0], reverse=True)
    no_bids = sorted(parse_book_levels(no_raw), key=lambda x: x[0], reverse=True)

    result.yes_levels = yes_bids
    result.no_levels = no_bids

    # YES bids = people willing to buy YES at that price
    # NO bids = people willing to buy NO at that price
    #
    # To BUY YES: you match against the cheapest ask. On Kalshi, someone
    # bidding NO at X¢ is equivalent to offering YES at (100-X)¢.
    # So: YES ask levels = NO bids inverted.
    # Best YES ask = 100 - (highest NO bid)
    #
    # To BUY NO: similarly, someone bidding YES at X¢ = offering NO at (100-X)¢.
    # Best NO ask = 100 - (highest YES bid)

    if yes_bids:
        result.best_yes_bid = yes_bids[0][0]
        result.yes_bid_depth = sum(qty for _, qty in yes_bids[:5])
        # NO ask = 100 - best YES bid
        result.best_no_ask = 100 - yes_bids[0][0]
        result.no_ask_depth = sum(qty for _, qty in yes_bids[:5])

    if no_bids:
        result.best_no_bid = no_bids[0][0]
        result.no_bid_depth = sum(qty for _, qty in no_bids[:5])
        # YES ask = 100 - best NO bid
        result.best_yes_ask = 100 - no_bids[0][0]
        result.yes_ask_depth = sum(qty for _, qty in no_bids[:5])

    # Spreads
    if result.best_yes_bid is not None and result.best_yes_ask is not None:
        result.yes_spread_cents = result.best_yes_ask - result.best_yes_bid
        result.yes_midpoint = (result.best_yes_bid + result.best_yes_ask) / 2.0

    if result.best_no_bid is not None and result.best_no_ask is not None:
        result.no_spread_cents = result.best_no_ask - result.best_no_bid
        result.no_midpoint = (result.best_no_bid + result.best_no_ask) / 2.0

    # Quality assessment
    yes_depth = max(result.yes_bid_depth, result.yes_ask_depth)
    no_depth = max(result.no_bid_depth, result.no_ask_depth)
    total_depth = yes_depth + no_depth

    result.is_liquid = total_depth >= min_depth
    result.is_tight = (
        (result.yes_spread_cents is not None and result.yes_spread_cents <= max_spread_cents) or
        (result.no_spread_cents is not None and result.no_spread_cents <= max_spread_cents)
    )
    result.is_tradeable = result.is_liquid and result.is_tight

    # Optimal limit prices — try to get inside the spread
    if result.best_yes_bid is not None and result.best_yes_ask is not None:
        # Place limit 1 cent above best bid (improve the bid to get priority)
        result.optimal_yes_price = result.best_yes_bid + 1
        # But don't exceed the ask
        if result.optimal_yes_price >= result.best_yes_ask:
            result.optimal_yes_price = result.best_yes_ask

    if result.best_no_bid is not None and result.best_no_ask is not None:
        result.optimal_no_price = result.best_no_bid + 1
        if result.optimal_no_price >= result.best_no_ask:
            result.optimal_no_price = result.best_no_ask

    # Slippage estimation
    # Buying YES = matching against NO bids (which form YES asks when inverted)
    # We want to walk through NO bids from highest to lowest (they become cheapest YES asks)
    # Buying NO = matching against YES bids (inverted to NO asks)
    for size, attr in [(5, "slippage_5"), (10, "slippage_10"), (25, "slippage_25")]:
        yes_slip = _estimate_slippage(no_bids, size)
        no_slip = _estimate_slippage(yes_bids, size)
        setattr(result, attr, max(yes_slip, no_slip))

    return result


def fillable_size(analysis: OrderbookAnalysis, side: str, limit_cents: int) -> int:
    """
    Contracts that can fill IMMEDIATELY at `limit_cents` for a buy on `side`.

    Buying YES at L matches resting NO bids at 100-L or better (a NO bid at
    X¢ is a YES offer at 100-X¢); buying NO at L matches YES bids at
    100-L or better. The old size cap summed the best FIVE levels of the
    opposite book — depth at prices you were NOT paying — so orders
    regularly exceeded what could fill and the remainder rested.
    """
    levels = analysis.no_levels if side == "yes" else analysis.yes_levels
    if not levels:
        return 0
    floor_price = 100 - limit_cents
    return int(sum(qty for price, qty in levels if price >= floor_price))


def _estimate_slippage(levels: list, order_size: int) -> float:
    """
    Estimate slippage in cents for a given order size.
    Returns the average price worsening vs best price.
    """
    if not levels or order_size <= 0:
        return 0.0

    best_price = levels[0][0]
    filled = 0
    total_cost = 0

    for price, qty in levels:
        can_fill = min(qty, order_size - filled)
        total_cost += can_fill * price
        filled += can_fill
        if filled >= order_size:
            break

    if filled == 0:
        return 0.0

    avg_price = total_cost / filled
    # Slippage is how much worse the average fill is vs best price
    return abs(avg_price - best_price)


def analyze_markets(client: KalshiClient, tickers: list[str],
                    max_spread_cents: int = 10,
                    min_depth: int = 5) -> dict[str, OrderbookAnalysis]:
    """
    Analyze orderbooks for multiple markets.
    Returns: {ticker: OrderbookAnalysis}
    """
    analyses = {}

    for ticker in tickers:
        analyses[ticker] = analyze_orderbook(
            client, ticker, max_spread_cents, min_depth
        )
        time.sleep(0.15)  # rate limiting

    return analyses


def print_orderbook_analysis(analysis: OrderbookAnalysis) -> None:
    """Pretty-print orderbook analysis."""
    a = analysis
    print(f"\n  {a.ticker}")
    print(f"  {'─' * 50}")

    # YES side
    if a.best_yes_bid is not None:
        print(f"    YES: bid {a.best_yes_bid}¢ / ask {a.best_yes_ask}¢ "
              f"(spread: {a.yes_spread_cents}¢) depth: {a.yes_bid_depth}")
    else:
        print(f"    YES: no bids")

    # NO side
    if a.best_no_bid is not None:
        print(f"    NO:  bid {a.best_no_bid}¢ / ask {a.best_no_ask}¢ "
              f"(spread: {a.no_spread_cents}¢) depth: {a.no_bid_depth}")
    else:
        print(f"    NO:  no bids")

    # Quality
    flags = []
    if a.is_liquid:
        flags.append("liquid")
    else:
        flags.append("THIN")
    if a.is_tight:
        flags.append("tight")
    else:
        flags.append("WIDE")
    print(f"    Status: {', '.join(flags)} → {'TRADEABLE' if a.is_tradeable else 'AVOID'}")

    # Optimal prices
    if a.optimal_yes_price:
        print(f"    Optimal YES limit: {a.optimal_yes_price}¢")
    if a.optimal_no_price:
        print(f"    Optimal NO limit:  {a.optimal_no_price}¢")

    # Slippage
    if a.slippage_5 > 0:
        print(f"    Slippage: 5ct→{a.slippage_5:.1f}¢  "
              f"10ct→{a.slippage_10:.1f}¢  "
              f"25ct→{a.slippage_25:.1f}¢")


# ── CLI ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from weather import CITIES
    from find_edge import get_market_prices

    parser = argparse.ArgumentParser(description="Analyze orderbook depth")
    parser.add_argument("city", choices=list(CITIES.keys()), help="City to analyze")
    parser.add_argument("--max-spread", type=int, default=10,
                        help="Max spread in cents (default: 10)")
    parser.add_argument("--min-depth", type=int, default=5,
                        help="Min depth to consider liquid (default: 5)")
    args = parser.parse_args()

    city = CITIES[args.city]
    client = KalshiClient(api_key_id="", private_key_path="",
                          base_url="https://api.elections.kalshi.com/trade-api/v2")
    client.private_key = None

    print(f"Fetching {city.name} markets...")
    markets = get_market_prices(city.kalshi_series)
    from find_edge import market_price
    tickers = [m["ticker"] for m in markets
               if not ((market_price(m, "yes_ask") or 1.0) <= 0.02 or
                       (market_price(m, "no_ask") or 1.0) <= 0.02)]

    print(f"Analyzing {len(tickers)} orderbooks...")
    analyses = analyze_markets(client, tickers, args.max_spread, args.min_depth)

    print(f"\n{'=' * 60}")
    print(f"  ORDERBOOK ANALYSIS — {city.name}")
    print(f"{'=' * 60}")

    tradeable = 0
    for ticker in sorted(analyses.keys()):
        a = analyses[ticker]
        print_orderbook_analysis(a)
        if a.is_tradeable:
            tradeable += 1

    print(f"\n  Summary: {tradeable}/{len(analyses)} markets are tradeable")
