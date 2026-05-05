"""
Market Explorer
Quick script to browse available Kalshi markets and inspect order books.
No authentication needed — all public data.

Usage:
    python explore_markets.py                          # list first page of markets
    python explore_markets.py --search MLB             # search in first page
    python explore_markets.py --series KXHIGHNY        # fetch by series (weather)
    python explore_markets.py --event KXHIGHNY-26MAY06 # fetch by event
    python explore_markets.py --ticker KXHIGHNY-26MAY06-T80  # inspect one market
    python explore_markets.py --events                 # list events instead of markets
"""

import argparse
import json
import requests
import time


BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"


def fetch_markets(series_ticker: str = None, event_ticker: str = None,
                  max_pages: int = 1) -> list:
    """Fetch markets, optionally filtered by series or event ticker."""
    all_markets = []
    cursor = None

    for _ in range(max_pages):
        params = {"limit": 200, "status": "open"}
        if series_ticker:
            params["series_ticker"] = series_ticker
        if event_ticker:
            params["event_ticker"] = event_ticker
        if cursor:
            params["cursor"] = cursor

        resp = requests.get(f"{BASE_URL}/markets", params=params)
        resp.raise_for_status()
        data = resp.json()
        markets = data.get("markets", [])
        if not markets:
            break
        all_markets.extend(markets)
        cursor = data.get("cursor")
        if not cursor:
            break
        time.sleep(0.1)

    return all_markets


def list_events(search: str = None, limit: int = 20):
    """List events — a higher-level view than individual markets."""
    params = {"limit": 200, "status": "open"}
    resp = requests.get(f"{BASE_URL}/events", params=params)
    resp.raise_for_status()
    events = resp.json().get("events", [])

    if search:
        search_lower = search.lower()
        events = [
            e for e in events
            if search_lower in e.get("title", "").lower()
            or search_lower in e.get("ticker", "").lower()
            or search_lower in e.get("series_ticker", "").lower()
            or search_lower in e.get("category", "").lower()
            or search_lower in e.get("sub_title", "").lower()
        ]

    print(f"\nFound {len(events)} events" + (f" matching '{search}'" if search else ""))
    print("-" * 90)

    for e in events[:limit]:
        print(f"  {e.get('ticker', '?'):<40} series: {e.get('series_ticker', '?')}")
        print(f"    {e.get('title', '')}")
        print(f"    Category: {e.get('category', '?')} | Markets: {e.get('mutually_exclusive', '?')}")
        print()

    if len(events) > limit:
        print(f"  ... and {len(events) - limit} more. Use --limit to see more.")


def list_markets(search: str = None, limit: int = 20,
                 series_ticker: str = None, event_ticker: str = None,
                 max_pages: int = 1):
    """List markets with optional filters."""
    pages_label = f" ({max_pages} page{'s' if max_pages > 1 else ''})"
    print(f"Fetching markets{pages_label}...")
    markets = fetch_markets(
        series_ticker=series_ticker,
        event_ticker=event_ticker,
        max_pages=max_pages,
    )

    if search:
        search_lower = search.lower()
        markets = [
            m for m in markets
            if search_lower in m.get("title", "").lower()
            or search_lower in m.get("ticker", "").lower()
            or search_lower in m.get("event_ticker", "").lower()
            or search_lower in m.get("no_sub_title", "").lower()
        ]

    print(f"Found {len(markets)} markets" +
          (f" matching '{search}'" if search else "") +
          (f" in series '{series_ticker}'" if series_ticker else "") +
          (f" in event '{event_ticker}'" if event_ticker else ""))
    print("-" * 90)

    for m in markets[:limit]:
        # Use the actual field names from the API
        yes_ask = m.get("yes_ask_dollars", m.get("yes_ask", "?"))
        no_ask = m.get("no_ask_dollars", m.get("no_ask", "?"))
        last_price = m.get("last_price_dollars", "?")
        volume = m.get("volume_fp", m.get("volume", "0"))

        print(f"  {m['ticker']}")
        print(f"    {m.get('title', '(no title)')}")
        print(f"    Last: ${last_price}  |  Yes ask: ${yes_ask}  |  No ask: ${no_ask}  |  Vol: {volume}")
        print()

    if len(markets) > limit:
        print(f"  ... and {len(markets) - limit} more. Use --limit to see more.")


def inspect_market(ticker: str):
    """Show detailed info and order book for a specific market."""
    resp = requests.get(f"{BASE_URL}/markets/{ticker}")
    resp.raise_for_status()
    market = resp.json().get("market", {})

    print(f"\n{'=' * 60}")
    print(f"  {market.get('title', 'Unknown')}")
    print(f"  {market.get('no_sub_title', '')}")
    print(f"{'=' * 60}")
    print(f"  Ticker:         {market.get('ticker')}")
    print(f"  Event:          {market.get('event_ticker')}")
    print(f"  Status:         {market.get('status')}")
    print(f"  Last price:     ${market.get('last_price_dollars', '?')}")
    print(f"  Yes ask:        ${market.get('yes_ask_dollars', '?')}")
    print(f"  No ask:         ${market.get('no_ask_dollars', '?')}")
    print(f"  Volume:         {market.get('volume_fp', '?')}")
    print(f"  Open interest:  {market.get('open_interest_fp', '?')}")
    print(f"  Closes:         {market.get('close_time')}")
    print(f"  Expires:        {market.get('expiration_time')}")
    print(f"  Market type:    {market.get('market_type')}")

    # Order book
    resp = requests.get(f"{BASE_URL}/markets/{ticker}/orderbook")
    resp.raise_for_status()
    ob = resp.json().get("orderbook", {})

    print(f"\n  Order Book:")
    print(f"  {'YES side':<40} {'NO side':<40}")
    print(f"  {'-' * 38}   {'-' * 38}")

    yes_data = ob.get("yes", [])
    no_data = ob.get("no", [])
    max_rows = max(len(yes_data), len(no_data), 1)

    for i in range(min(max_rows, 10)):
        yes_str = json.dumps(yes_data[i]) if i < len(yes_data) else ""
        no_str = json.dumps(no_data[i]) if i < len(no_data) else ""
        print(f"  {yes_str:<40} {no_str:<40}")

    # Also dump a few raw fields for debugging
    print(f"\n  Raw fields sample:")
    skip = {"rules_primary", "rules_secondary", "custom_strike", "mve_selected_legs", "price_ranges"}
    for k, v in list(market.items())[:25]:
        if k not in skip:
            print(f"    {k}: {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Explore Kalshi markets")
    parser.add_argument("--search", "-s", type=str, help="Search markets by keyword")
    parser.add_argument("--series", type=str, help="Filter by series ticker (e.g. KXHIGHNY)")
    parser.add_argument("--event", type=str, help="Filter by event ticker")
    parser.add_argument("--ticker", "-t", type=str, help="Inspect a specific market")
    parser.add_argument("--events", action="store_true", help="List events instead of markets")
    parser.add_argument("--limit", "-l", type=int, default=20, help="Max results to show")
    parser.add_argument("--pages", "-p", type=int, default=1, help="Number of pages to fetch (200 per page)")
    args = parser.parse_args()

    if args.ticker:
        inspect_market(args.ticker)
    elif args.events:
        list_events(search=args.search, limit=args.limit)
    else:
        list_markets(
            search=args.search,
            limit=args.limit,
            series_ticker=args.series,
            event_ticker=args.event,
            max_pages=args.pages,
        )
