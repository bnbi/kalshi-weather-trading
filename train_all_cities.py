"""
Train models for all cities.
Pulls historical data (if needed) and trains a forecast model for each city.

Usage:
    python train_all_cities.py              # train all cities
    python train_all_cities.py --fetch      # fetch fresh historical data first
    python train_all_cities.py --city nyc   # train just one city
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import sqlite3
from datetime import datetime, timedelta

from weather import CITIES
from historical_data import fetch_all_historical, print_data_summary, init_historical_tables
from train_model import train_and_evaluate, get_model_path

from pathlib import Path
DB_PATH = str(Path(__file__).parent / "kalshi_data.db")


def check_data_available(conn: sqlite3.Connection, city: str) -> int:
    """Check how many days of historical data we have for a city."""
    init_historical_tables(conn)
    row = conn.execute(
        "SELECT COUNT(*) FROM historical_forecasts WHERE city = ?", (city,)
    ).fetchone()
    return row[0] if row else 0


def fetch_data_for_city(conn: sqlite3.Connection, city: str,
                         start_date: str = "2024-06-01"):
    """Fetch historical data for a city if we don't have enough."""
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"\n{'=' * 60}")
    print(f"  FETCHING DATA: {city.upper()}")
    print(f"{'=' * 60}")
    fetch_all_historical(city, start_date, yesterday, conn)


def train_city(conn: sqlite3.Connection, city: str):
    """Train a model for a single city."""
    print(f"\n{'=' * 60}")
    print(f"  TRAINING MODEL: {city.upper()}")
    print(f"{'=' * 60}")

    days = check_data_available(conn, city)
    if days < 30:
        print(f"  ERROR: Only {days} days of data for {city}. Need at least 30.")
        print(f"  Run with --fetch to pull historical data first.")
        return False

    print(f"  Data available: {days} days")
    try:
        model_data = train_and_evaluate(city, conn)
        print(f"\n  SUCCESS: Model saved to {get_model_path(city)}")
        print(f"  MAE: {model_data['train_mae']:.2f}°F")
        return True
    except Exception as e:
        print(f"  ERROR training {city}: {e}")
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train models for all cities")
    parser.add_argument("--fetch", action="store_true",
                        help="Fetch historical data before training")
    parser.add_argument("--city", type=str, choices=list(CITIES.keys()),
                        help="Train only this city")
    parser.add_argument("--start", type=str, default="2024-06-01",
                        help="Start date for historical data (default: 2024-06-01)")
    parser.add_argument("--db", type=str, default=DB_PATH,
                        help="Database path")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    cities = [args.city] if args.city else list(CITIES.keys())

    print(f"\nCities to process: {', '.join(cities)}")

    # Step 1: Check/fetch data
    for city in cities:
        days = check_data_available(conn, city)
        print(f"  {city}: {days} days of historical data")

        if args.fetch or days < 30:
            fetch_data_for_city(conn, city, args.start)

    # Step 2: Train models
    print(f"\n\n{'#' * 60}")
    print(f"  TRAINING MODELS")
    print(f"{'#' * 60}")

    results = {}
    for city in cities:
        success = train_city(conn, city)
        results[city] = success

    # Summary
    print(f"\n\n{'=' * 60}")
    print(f"  TRAINING SUMMARY")
    print(f"{'=' * 60}")
    for city, success in results.items():
        status = "OK" if success else "FAILED"
        print(f"  {city:<12} [{status}]")

    conn.close()


if __name__ == "__main__":
    main()
