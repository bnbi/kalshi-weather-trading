"""
Scheduled Auto-Trader
Runs the trading pipeline automatically each morning.

Kill switch:
    To STOP all trading:   python scheduler.py --kill
    To RESUME trading:     python scheduler.py --resume
    Check status:          python scheduler.py --status

The kill switch creates/removes a file called KILL_SWITCH in the bot directory.
When present, no trades will be placed — the scheduler checks this BEFORE
doing anything with real money.

Usage:
    Run once (e.g. from cron):     python scheduler.py run
    Run continuously (daemon):      python scheduler.py daemon
    Kill switch on:                 python scheduler.py --kill
    Kill switch off:                python scheduler.py --resume
    Check status:                   python scheduler.py --status
"""

from __future__ import annotations

# Silence the cosmetic LibreSSL warning that urllib3 prints on macOS —
# it fires once per run into stderr and looks like an error while meaning
# nothing. Must run before the first `import requests`.
import warnings as _warnings
try:
    from urllib3.exceptions import NotOpenSSLWarning as _NotOpenSSL
    _warnings.filterwarnings("ignore", category=_NotOpenSSL)
except Exception:
    pass


import os
import sys
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────

BOT_DIR = Path(__file__).parent
KILL_SWITCH_FILE = BOT_DIR / "KILL_SWITCH"
LOG_DIR = BOT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# All configured cities (import kept lazy-safe: weather has no side effects)
from weather import CITIES as _CITIES
CITIES_ALL = list(_CITIES.keys())


def tradeable_cities() -> list:
    """Cities that have a trained forecast model on disk."""
    return [c for c in CITIES_ALL
            if (BOT_DIR / f"forecast_model_{c}.pkl").exists()]


# ── Logging setup ──────────────────────────────────────────────────

def setup_logging() -> logging.Logger:
    """Configure logging to both file and console."""
    today = datetime.now().strftime("%Y-%m-%d")
    log_file = LOG_DIR / f"trader_{today}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file, mode="a"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger("scheduler")


# ── Kill switch ────────────────────────────────────────────────────

def kill_switch_active() -> bool:
    """Check if the kill switch is engaged."""
    return KILL_SWITCH_FILE.exists()


def engage_kill_switch(reason: str = "manual") -> None:
    """Create the kill switch file to halt all trading."""
    KILL_SWITCH_FILE.write_text(
        f"Kill switch engaged at {datetime.now().isoformat()}\n"
        f"Reason: {reason}\n"
        f"\nTo resume trading: python scheduler.py --resume\n"
    )
    print(f"\n  KILL SWITCH ENGAGED")
    print(f"  Reason: {reason}")
    print(f"  All automated trading is now STOPPED.")
    print(f"  File: {KILL_SWITCH_FILE}")
    print(f"\n  To resume: python scheduler.py --resume")


def disengage_kill_switch() -> None:
    """Remove the kill switch file to resume trading."""
    if KILL_SWITCH_FILE.exists():
        KILL_SWITCH_FILE.unlink()
        print(f"\n  KILL SWITCH DISENGAGED")
        print(f"  Automated trading will resume on next scheduled run.")
    else:
        print(f"\n  Kill switch was not active — trading is already enabled.")


def print_status() -> None:
    """Print current status of the scheduler."""
    print(f"\n{'=' * 50}")
    print(f"  SCHEDULER STATUS")
    print(f"{'=' * 50}")

    if kill_switch_active():
        content = KILL_SWITCH_FILE.read_text()
        print(f"\n  Status: STOPPED (kill switch active)")
        print(f"  {content.strip()}")
    else:
        print(f"\n  Status: ACTIVE (trading enabled)")

    # Show recent logs
    today = datetime.now().strftime("%Y-%m-%d")
    log_file = LOG_DIR / f"trader_{today}.log"
    if log_file.exists():
        lines = log_file.read_text().strip().split("\n")
        recent = lines[-10:] if len(lines) > 10 else lines
        print(f"\n  Recent log ({log_file.name}):")
        for line in recent:
            print(f"    {line}")
    else:
        print(f"\n  No log file for today yet.")

    # Show upcoming schedule
    now = datetime.now()
    run_hour = 7
    next_run = now.replace(hour=run_hour, minute=0, second=0, microsecond=0)
    if now.hour >= run_hour:
        next_run += timedelta(days=1)
    print(f"\n  Next scheduled run: {next_run.strftime('%Y-%m-%d %H:%M')} local time")
    print()


# ── Trading run ────────────────────────────────────────────────────

def run_once(cities: list[str] = None, max_spend: float = None,
             min_edge: float = 0.05):
    """
    Execute one trading run across specified cities.
    Checks kill switch first.
    """
    logger = setup_logging()

    logger.info("=" * 50)
    logger.info("SCHEDULED TRADING RUN STARTING")
    logger.info("=" * 50)

    # CHECK KILL SWITCH FIRST
    if kill_switch_active():
        logger.warning("KILL SWITCH IS ACTIVE — aborting run.")
        logger.warning(f"To resume: python scheduler.py --resume")
        return

    if cities is None:
        # Trade every configured city that has a trained model. New cities
        # enter the rotation automatically once backfill_history.py has
        # bootstrapped their data and trained a model — never before, so an
        # untrained naive-average forecast can never place real orders.
        # (NYC re-enabled 2026-08-02: retrained on station truth it scores
        # 1.10°F MAE vs 1.31 baseline — the earlier "no edge" verdict was
        # an artifact of grading against ERA5.)
        cities = tradeable_cities()
        skipped = [c for c in CITIES_ALL if c not in cities]
        if skipped:
            logger.info(f"  Cities awaiting trained models (run "
                        f"backfill_history.py): {skipped}")

    # Import here so kill switch check happens before any API calls
    from trader import run_trading_pipeline
    from pnl_tracker import check_settlements, init_pnl_tables, print_summary
    from daily_learner import run_daily_learning
    from kalshi_client import create_client_from_config

    # Step 1: Check settlements from yesterday's trades
    logger.info("Checking for settled markets...")
    try:
        import sqlite3
        pnl_conn = sqlite3.connect(str(Path(__file__).parent / "kalshi_data.db"))
        init_pnl_tables(pnl_conn)
        # True-up fills BEFORE scoring: resting maker orders may have
        # filled partially, fully, or not at all since the last run.
        from pnl_tracker import reconcile_fills
        reconcile_fills(pnl_conn)
        settled = check_settlements(pnl_conn)
        if settled > 0:
            logger.info(f"  Settled {settled} trade(s) from previous days")
            print_summary(pnl_conn)
        # Score guardrail-rejected signals against settled results so the
        # filters themselves stay falsifiable (see pnl_tracker skipped).
        from pnl_tracker import verify_skipped_signals
        verify_skipped_signals(pnl_conn)
        pnl_conn.close()
    except Exception as e:
        logger.warning(f"  Could not check settlements: {e}")

    # Step 2: Pre-flight balance check — cap total spend to available balance
    # (Trade FIRST, retrain AFTER — retraining takes ~45 min and would delay orders)
    try:
        client = create_client_from_config()
        balance = client.get_balance()
        available = balance.get("balance", 0) / 100  # API returns cents
        logger.info(f"  Account balance: ${available:.2f}")

        # Budget is computed inside the global pipeline as a percentage
        # of the live bankroll; --max-spend still overrides as a ceiling.
    except Exception as e:
        logger.error(f"  Could not fetch balance: {e}")
        logger.error(f"  Aborting run for safety.")
        return

    total_results = _execute_trading(cities, max_spend, min_edge, logger)

    # If we ran before 6:30am and placed 0 orders, tomorrow's markets
    # probably aren't posted yet.  Wait until 7am and retry once.
    now = datetime.now()
    successes = [r for r in total_results if r.success]
    if len(successes) == 0 and now.hour < 7:
        target = now.replace(hour=7, minute=0, second=0, microsecond=0)
        wait_seconds = (target - now).total_seconds()
        logger.info(f"\n  No orders placed and it's before 7am.")
        logger.info(f"  Tomorrow's markets may not be posted yet.")
        logger.info(f"  Waiting {int(wait_seconds)}s until 7:00am to retry...")
        time.sleep(wait_seconds)

        logger.info("\n  RETRYING at 7:00am...")
        total_results = _execute_trading(cities, max_spend, min_edge, logger)
        successes = [r for r in total_results if r.success]

    logger.info(f"\nRUN COMPLETE: {len(successes)} orders placed across {len(cities)} cities")
    total_spent = sum(r.order.cost_dollars for r in total_results if r.success)
    if successes:
        logger.info(f"  Total spent: ${total_spent:.2f}")

    # Step 4: Daily learning — verify yesterday, retrain, record today
    # (Runs AFTER trading so orders aren't delayed by ~45 min of retraining)
    logger.info("Running daily learning cycle...")
    try:
        # Learn on ALL configured cities (including ones not yet trading)
        # so every city keeps accumulating verified data.
        run_daily_learning(CITIES_ALL)
    except Exception as e:
        logger.warning(f"  Daily learning failed: {e}")


class _LoggerWriter:
    """File-like object that forwards print() output line-by-line to a logger.

    trader.py reports skip reasons ('No profitable trades found', 'Positions
    too small to trade', ...) via print(). Without this, those reasons only
    land in launchd_stdout.log and the daily trader_*.log shows silent gaps.
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self._buf = ""

    def write(self, text: str):
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                self.logger.info(line)

    def flush(self):
        if self._buf.strip():
            self.logger.info(self._buf)
        self._buf = ""


def _execute_trading(cities: list[str], max_spend: float,
                     min_edge: float, logger: logging.Logger) -> list:
    """
    Execute one trading pass. SIZING_MODE in config.py selects:
      "global"   — pooled cross-city sizing (default)
      "per_city" — legacy even split, the proven-but-small Aug 18-31
                   behavior (the one-line revert switch)
    """
    from contextlib import redirect_stdout
    from trader import run_global_pipeline

    try:
        from config import SIZING_MODE
    except ImportError:
        SIZING_MODE = "global"
    if SIZING_MODE == "per_city":
        return _execute_trading_per_city(cities, max_spend, min_edge, logger)

    total_results = []
    logger.info(f"\n--- Trading {len(cities)} cities (global sizing) ---")
    try:
        with redirect_stdout(_LoggerWriter(logger)):
            total_results = run_global_pipeline(
                city_keys=cities,
                dry_run=False,
                min_edge=min_edge,
                tomorrow_only=True,
                max_spend=max_spend,
            )
        total_spent = sum(r.order.cost_dollars for r in total_results if r.success)
        if total_spent:
            logger.info(f"  Spent ${total_spent:.2f} this run")

        failures = [r for r in total_results if not r.success]
        if failures:
            logger.error(f"  {len(failures)} order(s) FAILED")
            for f in failures:
                logger.error(f"    {f.order.ticker}: {f.error}")
            if len(failures) >= 3:
                engage_kill_switch(f"Auto-kill: {len(failures)} order failures")
                logger.critical("AUTO KILL SWITCH engaged due to multiple failures")
    except Exception as e:
        logger.error(f"  ERROR in global trading pass: {e}")

    return total_results



def _execute_trading_per_city(cities: list[str], max_spend: float,
                              min_edge: float, logger: logging.Logger) -> list:
    """Legacy per-city budget split (pre-2026-09-01). Kept as the revert
    path: proven Aug 18-31 live record, smaller bets."""
    from contextlib import redirect_stdout
    from trader import run_trading_pipeline
    from kalshi_client import create_client_from_config

    try:
        from config import MAX_RUN_EXPOSURE_PCT as _pct
    except ImportError:
        _pct = 0.25
    try:
        client = create_client_from_config()
        available = client.get_balance().get("balance", 0) / 100
    except Exception as e:
        logger.error(f"  Could not fetch balance for per-city split: {e}")
        return []

    total_budget = available * _pct
    if max_spend is not None:
        total_budget = min(total_budget, max_spend)
    total_budget = min(total_budget, available)
    per_city_budget = total_budget / len(cities)
    logger.info(f"  [per_city mode] Budget: ${total_budget:.2f} total, "
                f"${per_city_budget:.2f} per city")

    total_results, total_spent = [], 0.0
    for city in cities:
        remaining = total_budget - total_spent
        city_budget = min(per_city_budget, remaining)
        if city_budget < 0.50:
            logger.warning(f"  Skipping {city}: only ${remaining:.2f} remaining")
            continue
        logger.info(f"\n--- Trading {city.upper()} (budget: ${city_budget:.2f}) ---")
        try:
            with redirect_stdout(_LoggerWriter(logger)):
                results = run_trading_pipeline(
                    city_key=city, dry_run=False, max_spend=city_budget,
                    min_edge=min_edge, tomorrow_only=True)
            total_results.extend(results)
            total_spent += sum(r.order.cost_dollars for r in results if r.success)
            failures = [r for r in results if not r.success]
            if failures:
                logger.error(f"  {len(failures)} order(s) FAILED for {city}")
                if len(failures) >= 3:
                    engage_kill_switch(f"Auto-kill: {len(failures)} failures in {city}")
                    logger.critical("AUTO KILL SWITCH engaged")
                    break
        except Exception as e:
            logger.error(f"  ERROR trading {city}: {e}")
    return total_results


def run_daemon(run_hour: int = 7, cities: list[str] = None,
               max_spend: float = None):
    """
    Run continuously, executing trades once per day at run_hour (local time).
    Useful if you don't want to set up cron.
    """
    logger = setup_logging()
    logger.info(f"Daemon started. Will trade daily at {run_hour}:00 local time.")
    logger.info(f"Cities: {cities or tradeable_cities()}")
    logger.info(f"Max spend per run: "
                f"{'$' + format(max_spend, '.2f') if max_spend is not None else 'percentage of bankroll (config)'}")
    logger.info(f"Kill switch file: {KILL_SWITCH_FILE}")
    logger.info(f"PID: {os.getpid()}")

    last_run_date = None

    while True:
        now = datetime.now()

        # Run once per day at the specified hour
        if now.hour == run_hour and now.strftime("%Y-%m-%d") != last_run_date:
            last_run_date = now.strftime("%Y-%m-%d")
            logger.info(f"Triggering daily run at {now.strftime('%H:%M:%S')}")
            run_once(cities=cities, max_spend=max_spend)

        # Sleep 60 seconds between checks
        time.sleep(60)


# ── CLI ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Scheduled auto-trader with kill switch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scheduler.py run                    # Run once now (live)
  python scheduler.py run --dry-run          # Run once (simulated)
  python scheduler.py daemon                 # Run daily at 7am
  python scheduler.py daemon --hour 6        # Run daily at 6am
  python scheduler.py --kill                 # STOP all trading
  python scheduler.py --resume               # Resume trading
  python scheduler.py --status               # Check status
        """
    )

    # Kill switch commands (can be used without subcommand)
    parser.add_argument("--kill", action="store_true",
                        help="Engage kill switch — stop all trading")
    parser.add_argument("--resume", action="store_true",
                        help="Disengage kill switch — resume trading")
    parser.add_argument("--status", action="store_true",
                        help="Show scheduler status")

    subparsers = parser.add_subparsers(dest="command")

    # 'run' subcommand
    run_parser = subparsers.add_parser("run", help="Execute one trading run")
    run_parser.add_argument("--cities", nargs="+", default=None,
                            choices=CITIES_ALL,
                            help="Cities to trade (default: every city "
                                 "with a trained model)")
    run_parser.add_argument("--max-spend", type=float, default=None,
                            help="Optional hard dollar ceiling (default: "
                                 "MAX_RUN_EXPOSURE_PCT of bankroll)")
    run_parser.add_argument("--min-edge", type=float, default=0.05,
                            help="Minimum edge to trade (default: 5%%)")
    run_parser.add_argument("--dry-run", action="store_true",
                            help="Simulate without placing orders")

    # 'daemon' subcommand
    daemon_parser = subparsers.add_parser("daemon", help="Run continuously")
    daemon_parser.add_argument("--hour", type=int, default=7,
                               help="Hour to run each day (default: 7 = 7am)")
    daemon_parser.add_argument("--cities", nargs="+", default=None,
                               choices=CITIES_ALL,
                               help="Cities to trade (default: every city "
                                    "with a trained model)")
    daemon_parser.add_argument("--max-spend", type=float, default=None,
                               help="Optional hard dollar ceiling (default: "
                                    "MAX_RUN_EXPOSURE_PCT of bankroll)")

    args = parser.parse_args()

    # Handle kill switch commands first
    if args.kill:
        engage_kill_switch("manual — user ran --kill")
        sys.exit(0)

    if args.resume:
        disengage_kill_switch()
        sys.exit(0)

    if args.status:
        print_status()
        sys.exit(0)

    # Handle subcommands
    if args.command == "run":
        if args.dry_run:
            # Dry run — same budget split as live so numbers are realistic
            from trader import run_trading_pipeline
            dry_cities = args.cities or tradeable_cities()
            # Dry run simulates a $100 bankroll (see trader.py), so the
            # default budget mirrors the live percentage sizing.
            try:
                from config import MAX_RUN_EXPOSURE_PCT as _pct
            except ImportError:
                _pct = 0.25
            dry_budget = args.max_spend if args.max_spend is not None else 100.0 * _pct
            per_city = dry_budget / len(dry_cities)
            total_spent = 0.0
            for city in dry_cities:
                remaining = dry_budget - total_spent
                city_budget = min(per_city, remaining)
                if city_budget < 0.50:
                    print(f"  Skipping {city}: only ${remaining:.2f} remaining")
                    continue
                results = run_trading_pipeline(
                    city_key=city,
                    dry_run=True,
                    max_spend=city_budget,
                    min_edge=args.min_edge,
                    tomorrow_only=True,
                )
                total_spent += sum(r.order.cost_dollars for r in results if r.success)
        else:
            run_once(cities=args.cities, max_spend=args.max_spend,
                     min_edge=args.min_edge)

    elif args.command == "daemon":
        run_daemon(run_hour=args.hour, cities=args.cities,
                   max_spend=args.max_spend)

    else:
        # No command given — show status
        print_status()
