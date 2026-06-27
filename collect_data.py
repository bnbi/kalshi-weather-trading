"""
Data Collection Script
Pulls market snapshots from Kalshi and stores them in a local SQLite database.
Run this periodically (e.g. every hour via cron) to build a historical dataset.
"""

from __future__ import annotations

import sqlite3
import json
import time
from datetime import datetime, timezone
from kalshi_client import KalshiClient


# ── Database setup ──────────────────────────────────────────────────

DB_PATH = "kalshi_data.db"


def init_db(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Create tables if they don't exist."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS markets (
            ticker TEXT PRIMARY KEY,
            event_ticker TEXT,
            series_ticker TEXT,
            title TEXT,
            subtitle TEXT,
            status TEXT,
            open_time TEXT,
            close_time TEXT,
            expiration_time TEXT,
            yes_sub_title TEXT,
            no_sub_title TEXT,
            first_seen TEXT,
            last_updated TEXT,
            raw_json TEXT
        );

        CREATE TABLE IF NOT EXISTS snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            yes_price INTEGER,
            no_price INTEGER,
            yes_bid INTEGER,
            yes_ask INTEGER,
            no_bid INTEGER,
            no_ask INTEGER,
            volume INTEGER,
            open_interest INTEGER,
            raw_json TEXT,
            FOREIGN KEY (ticker) REFERENCES markets(ticker)
        );

        CREATE INDEX IF NOT EXISTS idx_snapshots_ticker
            ON snapshots(ticker);
        CREATE INDEX IF NOT EXISTS idx_snapshots_timestamp
            ON snapshots(timestamp);
        CREATE INDEX IF NOT EXISTS idx_snapshots_ticker_time
            ON snapshots(ticker, timestamp);

        CREATE TABLE IF NOT EXISTS orderbook_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            yes_bids TEXT,
            no_bids TEXT,
            FOREIGN KEY (ticker) REFERENCES markets(ticker)
        );
    """)

    conn.commit()
    return conn


# ── Data collection ─────────────────────────────────────────────────

def collect_markets(client: KalshiClient, conn: sqlite3.Connection,
                    series_ticker: str = None, event_ticker: str = None) -> int:
    """Fetch all open markets and upsert into the database."""
    now = datetime.now(timezone.utc).isoformat()
    cursor = None
    total = 0

    while True:
        result = client.get_markets(
            limit=100,
            cursor=cursor,
            status="open",
            series_ticker=series_ticker,
            event_ticker=event_ticker,
        )

        markets = result.get("markets", [])
        if not markets:
            break

        for m in markets:
            conn.execute("""
                INSERT INTO markets (
                    ticker, event_ticker, series_ticker, title, subtitle,
                    status, open_time, close_time, expiration_time,
                    yes_sub_title, no_sub_title, first_seen, last_updated, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticker) DO UPDATE SET
                    status = excluded.status,
                    last_updated = excluded.last_updated,
                    raw_json = excluded.raw_json
            """, (
                m.get("ticker"),
                m.get("event_ticker"),
                m.get("series_ticker"),
                m.get("title"),
                m.get("subtitle"),
                m.get("status"),
                m.get("open_time"),
                m.get("close_time"),
                m.get("expiration_time"),
                m.get("yes_sub_title"),
                m.get("no_sub_title"),
                now,
                now,
                json.dumps(m),
            ))
            total += 1

        cursor = result.get("cursor")
        if not cursor:
            break

    conn.commit()
    print(f"[{now}] Upserted {total} markets")
    return total


def collect_snapshots(client: KalshiClient, conn: sqlite3.Connection,
                      tickers: list[str] = None) -> int:
    """
    Take price/volume snapshots for specified markets.
    If tickers is None, snapshot all open markets in the database.
    """
    now = datetime.now(timezone.utc).isoformat()

    if tickers is None:
        rows = conn.execute(
            "SELECT ticker FROM markets WHERE status IN ('open', 'active')"
        ).fetchall()
        tickers = [r[0] for r in rows]

    count = 0
    for ticker in tickers:
        try:
            market = client.get_market(ticker).get("market", {})
            conn.execute("""
                INSERT INTO snapshots (
                    ticker, timestamp, yes_price, no_price,
                    yes_bid, yes_ask, no_bid, no_ask,
                    volume, open_interest, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                ticker,
                now,
                market.get("last_price_dollars", market.get("yes_price")),
                market.get("no_ask_dollars", market.get("no_price")),
                market.get("previous_yes_bid_dollars", market.get("yes_bid")),
                market.get("yes_ask_dollars", market.get("yes_ask")),
                market.get("no_bid_dollars", market.get("no_bid")),
                market.get("no_ask_dollars", market.get("no_ask")),
                market.get("volume_fp", market.get("volume")),
                market.get("open_interest_fp", market.get("open_interest")),
                json.dumps(market),
            ))
            count += 1

            # Be polite to the API
            time.sleep(0.1)

        except Exception as e:
            print(f"  Error snapshotting {ticker}: {e}")

    conn.commit()
    print(f"[{now}] Collected {count} snapshots")
    return count


def collect_orderbooks(client: KalshiClient, conn: sqlite3.Connection,
                       tickers: list[str]) -> int:
    """Take order book snapshots for specific markets."""
    now = datetime.now(timezone.utc).isoformat()
    count = 0

    for ticker in tickers:
        try:
            ob = client.get_orderbook(ticker).get("orderbook", {})
            conn.execute("""
                INSERT INTO orderbook_snapshots (
                    ticker, timestamp, yes_bids, no_bids
                ) VALUES (?, ?, ?, ?)
            """, (
                ticker,
                now,
                json.dumps(ob.get("yes", [])),
                json.dumps(ob.get("no", [])),
            ))
            count += 1
            time.sleep(0.1)

        except Exception as e:
            print(f"  Error getting orderbook for {ticker}: {e}")

    conn.commit()
    print(f"[{now}] Collected {count} orderbook snapshots")
    return count


# ── Main ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Collect Kalshi market data")
    parser.add_argument("--series", type=str, help="Filter by series ticker")
    parser.add_argument("--event", type=str, help="Filter by event ticker")
    parser.add_argument(
        "--orderbooks", nargs="*",
        help="Also collect orderbooks for these tickers (or all if no args)"
    )
    parser.add_argument("--db", type=str, default=DB_PATH, help="Database path")
    args = parser.parse_args()

    # Public data doesn't require auth — use a minimal client
    client = KalshiClient(
        api_key_id="",
        private_key_path="",  # won't be used for public endpoints
        base_url="https://api.elections.kalshi.com/trade-api/v2",
    )
    # Override to skip loading private key for public-only usage
    client.private_key = None

    conn = init_db(args.db)

    print("=== Collecting markets ===")
    collect_markets(client, conn, series_ticker=args.series, event_ticker=args.event)

    print("\n=== Collecting price snapshots ===")
    collect_snapshots(client, conn)

    if args.orderbooks is not None:
        tickers = args.orderbooks if args.orderbooks else None
        if tickers is None:
            rows = conn.execute(
                "SELECT ticker FROM markets WHERE status = 'open'"
            ).fetchall()
            tickers = [r[0] for r in rows]
        print("\n=== Collecting orderbooks ===")
        collect_orderbooks(client, conn, tickers)

    conn.close()
    print("\nDone!")
