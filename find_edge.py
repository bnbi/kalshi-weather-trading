"""
Edge Calculator
Compares model probabilities to Kalshi market prices to find profitable trades.

A trade has positive expected value when:
    model_probability - market_price > spread_cost + min_edge

This script ties everything together:
1. Fetches current Kalshi markets for a city
2. Gets the NWS forecast
3. Runs the probability model
4. Compares to market prices
5. Outputs recommended trades
"""

from __future__ import annotations

import requests
from dataclasses import dataclass

from weather import CITIES
from model import predict_all_for_city, ContractPrediction


BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"


@dataclass
class TradeSignal:
    """A potential trade identified by the edge calculator."""
    ticker: str
    side: str              # 'yes' or 'no'
    action: str            # always 'buy' for now
    model_prob: float      # our probability of YES
    market_price: float    # best price to buy (in dollars)
    edge: float            # model_prob - market_price (if buying yes)
    expected_value: float  # edge * notional
    description: str


def kalshi_fee_per_contract(price: float) -> float:
    """
    Kalshi trading fee per contract, in dollars.

    Fee formula: 0.07 * price * (1 - price), rounded up to the next cent.
    Worst case is 1.75c at a 50c price; ~1.5c at 70c. Edges must clear
    this to be profitable, so we subtract it before comparing to min_edge.
    """
    import math
    if price <= 0 or price >= 1:
        return 0.0
    return math.ceil(0.07 * price * (1 - price) * 100) / 100


def get_market_prices(series_ticker: str) -> list[dict]:
    """Fetch current markets and prices for a series."""
    resp = requests.get(f"{BASE_URL}/markets", params={
        "limit": 200,
        "status": "open",
        "series_ticker": series_ticker,
    }, timeout=15)
    resp.raise_for_status()
    return resp.json().get("markets", [])


def calculate_edge(predictions: list[ContractPrediction],
                   markets: list[dict],
                   min_edge: float = 0.05) -> list[TradeSignal]:
    """
    Find trades where our model disagrees with the market by more than min_edge.

    min_edge: minimum edge (in dollars) required to signal a trade.
              0.05 = 5 cents = 5% edge minimum.
    """
    # Build a lookup of market prices by ticker
    market_lookup = {}
    for m in markets:
        ticker = m.get("ticker", "")
        market_lookup[ticker] = {
            "yes_ask": parse_price(m.get("yes_ask_dollars", m.get("yes_ask"))),
            "no_ask": parse_price(m.get("no_ask_dollars", m.get("no_ask"))),
            "last_price": parse_price(m.get("last_price_dollars", m.get("last_price"))),
            "title": m.get("title", ""),
            "volume": m.get("volume_fp", m.get("volume", "0")),
        }

    # Maximum credible disagreement — if our model disagrees with the market
    # by more than this, the market almost certainly knows something we don't
    # (intraday obs, newer model runs). These trades are SKIPPED, not capped.
    #
    # Evidence (89 settled trades through 2026-06-11): trades with raw
    # disagreement >25¢ lost $31.54 in aggregate; trades within the band
    # made +$8.68. Capping the edge but still trading was the single
    # largest source of losses.
    MAX_CREDIBLE_EDGE = 0.25

    # Market-prior shrinkage: the market price is treated as a Bayesian
    # prior and our model only moves us partway off it.
    #     blended_prob = MODEL_WEIGHT * model_prob + (1 - MODEL_WEIGHT) * price
    # Live calibration showed the model claiming ~95% on bets that won 75%,
    # while the market price was nearly unbiased. Blending halves phantom
    # edge and shrinks Kelly stakes toward sanity. With MODEL_WEIGHT = 0.5,
    # clearing a 7¢ min edge requires a raw model-market gap of 14¢.
    MODEL_WEIGHT = 0.5

    # ── Trade selection rules (based on 75-trade P&L analysis) ─────
    #
    # Our model's strength is identifying outcomes that WON'T happen.
    # Buying NO on high-confidence outcomes: 92% win rate, +17% ROI.
    # Buying YES on speculative outcomes:    8% win rate, -75% ROI.
    #
    # Rules:
    # 1. NO bets: only take when market price ≥ 68 cents (high-confidence)
    # 2. YES bets on brackets: NEVER — model can't pick specific 1°F ranges
    # 3. YES bets on thresholds: only when market price ≥ 13 cents
    #    (avoid cheap lottery tickets that almost never pay off)
    MIN_NO_PRICE = 0.68     # only buy NO at 68c or above
    MIN_YES_PRICE = 0.13    # only buy YES at 13c or above
    ALLOW_BRACKET_YES = False  # bracket YES bets are -80% ROI — never take them

    signals = []

    for pred in predictions:
        market = market_lookup.get(pred.ticker)
        if not market:
            continue

        yes_ask = market["yes_ask"]
        no_ask = market["no_ask"]

        if yes_ask is None or no_ask is None:
            continue

        # Skip already-settled contracts (both sides at extreme prices)
        if (yes_ask <= 0.02 and no_ask >= 0.98) or (yes_ask >= 0.98 and no_ask <= 0.02):
            continue

        # Raw disagreement with the market, per side
        raw_yes = pred.model_probability - yes_ask
        raw_no = (1 - pred.model_probability) - no_ask

        # Blend model with market prior. Since blended = w*model + (1-w)*price,
        # the blended edge is simply w * (model - price).
        yes_blend_prob = MODEL_WEIGHT * pred.model_probability + (1 - MODEL_WEIGHT) * yes_ask
        no_blend_prob = MODEL_WEIGHT * (1 - pred.model_probability) + (1 - MODEL_WEIGHT) * no_ask
        # Edges are net of Kalshi's trading fee — a thin gross edge that
        # doesn't cover the fee is a losing trade, not a marginal one.
        yes_edge = yes_blend_prob - yes_ask - kalshi_fee_per_contract(yes_ask)
        no_edge = no_blend_prob - no_ask - kalshi_fee_per_contract(no_ask)

        # Distrust filter: SKIP (don't cap) implausible disagreements
        yes_credible = raw_yes <= MAX_CREDIBLE_EDGE
        no_credible = raw_no <= MAX_CREDIBLE_EDGE

        # Signal YES trade (with strict filters)
        if yes_credible and yes_edge >= min_edge and yes_ask >= MIN_YES_PRICE:
            # Block bracket YES bets entirely — model can't pick 1°F ranges
            is_bracket = pred.contract_type == "bracket"
            if is_bracket and not ALLOW_BRACKET_YES:
                pass  # skip
            else:
                signals.append(TradeSignal(
                    ticker=pred.ticker,
                    side="yes",
                    action="buy",
                    model_prob=yes_blend_prob,
                    market_price=yes_ask,
                    edge=yes_edge,
                    expected_value=yes_edge,
                    description=f"BUY YES @ ${yes_ask:.2f} | Model: {pred.model_probability:.1%} "
                                f"(blended {yes_blend_prob:.1%}) | {pred.description}",
                ))

        # Signal NO trade
        if no_credible and no_edge >= min_edge and no_ask >= MIN_NO_PRICE:
            signals.append(TradeSignal(
                ticker=pred.ticker,
                side="no",
                action="buy",
                model_prob=no_blend_prob,
                market_price=no_ask,
                edge=no_edge,
                expected_value=no_edge,
                description=f"BUY NO @ ${no_ask:.2f} | Model: {1 - pred.model_probability:.1%} "
                            f"(blended {no_blend_prob:.1%}) | {pred.description}",
            ))

    # Sort by edge (best opportunities first)
    signals.sort(key=lambda s: s.edge, reverse=True)
    return signals


def parse_price(price_str: object) -> float | None:
    """Parse a price string like '0.4300' into a float."""
    if price_str is None:
        return None
    try:
        return float(price_str)
    except (ValueError, TypeError):
        return None


def print_signals(signals: list[TradeSignal], city_name: str) -> None:
    """Pretty-print trade signals."""
    print(f"\n{'=' * 80}")
    print(f"  TRADE SIGNALS — {city_name}")
    print(f"{'=' * 80}")

    if not signals:
        print("  No trades with sufficient edge found.")
        print("  (This is normal — the market is often efficient.)")
        return

    print(f"\n  Found {len(signals)} potential trade(s):\n")
    print(f"  {'Ticker':<45} {'Side':<5} {'Edge':<8} {'EV/contract'}")
    print(f"  {'-' * 75}")

    for s in signals:
        print(f"  {s.ticker:<45} {s.side:<5} {s.edge:>+5.1%}  ${s.expected_value:.2f}")
        print(f"    {s.description}")
        print()


def print_full_comparison(predictions: list[ContractPrediction], markets: list[dict]) -> None:
    """Print a full comparison table of model vs market."""
    market_lookup = {m["ticker"]: m for m in markets}

    print(f"\n  {'Ticker':<40} {'Model':<8} {'Mkt Yes':<9} {'Mkt No':<9} {'Edge(Y)':<9} {'Edge(N)'}")
    print(f"  {'-' * 90}")

    for pred in sorted(predictions, key=lambda p: p.ticker):
        m = market_lookup.get(pred.ticker, {})
        yes_ask = parse_price(m.get("yes_ask_dollars", m.get("yes_ask")))
        no_ask = parse_price(m.get("no_ask_dollars", m.get("no_ask")))

        yes_edge = (pred.model_probability - yes_ask) if yes_ask else None
        no_edge = ((1 - pred.model_probability) - no_ask) if no_ask else None

        yes_str = f"${yes_ask:.2f}" if yes_ask else "?"
        no_str = f"${no_ask:.2f}" if no_ask else "?"
        ye_str = f"{yes_edge:>+5.1%}" if yes_edge else "  ?"
        ne_str = f"{no_edge:>+5.1%}" if no_edge else "  ?"

        # Highlight positive edges
        marker = " <--" if (yes_edge and yes_edge > 0.05) or (no_edge and no_edge > 0.05) else ""

        print(f"  {pred.ticker:<40} {pred.model_probability:>5.1%}   {yes_str:<8} {no_str:<8} "
              f"{ye_str:<8} {ne_str}{marker}")


# ── Main ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Find trading edges in weather contracts")
    parser.add_argument("city", choices=list(CITIES.keys()), help="City to analyze")
    parser.add_argument("--min-edge", type=float, default=0.05,
                        help="Minimum edge to signal (default 0.05 = 5 cents)")
    parser.add_argument("--show-all", action="store_true",
                        help="Show full model vs market comparison")
    args = parser.parse_args()

    city = CITIES[args.city]

    # Step 1: Fetch markets
    print(f"Fetching {city.name} weather markets...")
    markets = get_market_prices(city.kalshi_series)
    print(f"  Found {len(markets)} open markets")

    # Step 2: Generate predictions
    print(f"\nFetching NWS forecast and computing probabilities...")
    predictions = predict_all_for_city(args.city, markets)
    print(f"  Generated {len(predictions)} predictions")

    # Step 3: Find edges
    signals = calculate_edge(predictions, markets, min_edge=args.min_edge)
    print_signals(signals, city.name)

    # Step 4: Optionally show full comparison
    if args.show_all:
        print(f"\n\n  FULL MODEL vs MARKET COMPARISON")
        print_full_comparison(predictions, markets)
