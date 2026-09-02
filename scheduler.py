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
# nothing. Filter by MESSAGE, before any urllib3 import: importing
# urllib3.exceptions to get the class itself triggered the warning it was
# meant to silence (the warning fires during urllib3's own import).
import warnings as _warnings
_warnings.filterwarnings("ignore", message=".*OpenSSL 1\\.1\\.1.*")


import os
import sys
import time
import logging
from contextlib import redirect_stdout
from datetime import datetime, timedelta
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────

BOT_DIR = Path(__file__).parent
KILL_SWITCH_FILE = BOT_DIR / "KILL_SWITCH"
LOG_DIR = BOT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Hour of the daily launchd run (com.kalshi.weatherbot.plist). Set
# config.RUN_HOUR to match the installed plist — status displays and the
# daemon derive "next run" from this. (The installed job has been firing at
# 13:00 local while the repo plist said 10:00.)
try:
    from config import RUN_HOUR as _cfg_run_hour
    RUN_HOUR = int(_cfg_run_hour)
except Exception:
    RUN_HOUR = 13

# Per-day trader logs older than this are pruned on each run.
LOG_RETENTION_DAYS = 60

# All configured cities (import kept lazy-safe: weather has no side effects)
from weather import CITIES as _CITIES
CITIES_ALL = list(_CITIES.keys())


def tradeable_cities() -> list:
    """
    Cities whose trained forecast model is on disk AND loads. A pickle that
    exists but fails to unpickle (library mismatch, partial write) used to
    pass this check and silently trade the naive-average fallback.
    """
    out = []
    for c in CITIES_ALL:
        path = BOT_DIR / f"forecast_model_{c}.pkl"
        if not path.exists():
            continue
        try:
            from train_model import load_model
            md = load_model(str(path))
            if "model" in md and "feature_names" in md:
                out.append(c)
            else:
                print(f"  Warning: {path.name} is missing model data — city skipped")
        except Exception as e:
            print(f"  Warning: {path.name} failed to load ({e}) — city skipped")
    return out


# ── Logging setup ──────────────────────────────────────────────────

def setup_logging() -> logging.Logger:
    """Configure logging to both file and console."""
    today = datetime.now().strftime("%Y-%m-%d")
    log_file = LOG_DIR / f"trader_{today}.log"

    # force=True replaces any handlers from a previous day's setup, so a
    # long-lived daemon rolls to the new day's file instead of writing to
    # the file named for the day it started.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file, mode="a"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    return logging.getLogger("scheduler")


def prune_old_logs(days: int = LOG_RETENTION_DAYS) -> int:
    """Delete per-day trader logs older than `days`. Returns count removed."""
    import time as _time
    cutoff = _time.time() - days * 86400
    removed = 0
    try:
        for f in LOG_DIR.glob("trader_*.log"):
            if f.stat().st_mtime < cutoff:
                f.unlink()
                removed += 1
    except OSError:
        pass
    return removed


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
    next_run = now.replace(hour=RUN_HOUR, minute=0, second=0, microsecond=0)
    if now.hour >= RUN_HOUR:
        next_run += timedelta(days=1)
    print(f"\n  Next scheduled run: {next_run.strftime('%Y-%m-%d %H:%M')} local time")
    print()


# ── Trading run ────────────────────────────────────────────────────

def default_min_edge() -> float:
    """
    The live edge threshold: config.MIN_EDGE_CENTS (in cents) when set,
    else 5 cents. Previously the config knob was documentation-only and the
    hardcoded 5c always won, whatever config said.
    """
    try:
        from config import MIN_EDGE_CENTS
        return MIN_EDGE_CENTS / 100.0
    except ImportError:
        return 0.05


def run_once(cities: list[str] = None, max_spend: float = None,
             min_edge: float = None, dry_run: bool = False):
    """
    Execute one trading run across specified cities.
    Checks kill switch first.

    Order of operations (changed 2026-09):
      settlements → verify/re-verify + retrain → TRADE → record forecasts
    Learning used to run after trading, so every decision used a model and
    a calibration window that were one day staler than necessary; the
    whole learning step takes well under a minute now.

    dry_run: paper mode. Still runs settlement checks (if credentials
    exist) and the full learning cycle — a paper deployment must keep
    learning, or its models and calibration silently freeze.
    """
    logger = setup_logging()
    if min_edge is None:
        min_edge = default_min_edge()

    pruned = prune_old_logs()
    if pruned:
        logger.info(f"Pruned {pruned} log file(s) older than "
                    f"{LOG_RETENTION_DAYS} days")

    logger.info("=" * 50)
    logger.info(f"SCHEDULED TRADING RUN STARTING{' (DRY RUN)' if dry_run else ''}")
    logger.info("=" * 50)

    # CHECK KILL SWITCH FIRST
    if kill_switch_active():
        logger.warning("KILL SWITCH IS ACTIVE — aborting run.")
        logger.warning(f"To resume: python scheduler.py --resume")
        return

    # Import here so kill switch check happens before any API calls
    from pnl_tracker import check_settlements, init_pnl_tables, print_summary
    from daily_learner import learn, record, init_prediction_log
    from db_migrations import migrate_db
    from historical_data import init_historical_tables
    import sqlite3

    # Step 1: Check settlements from yesterday's trades (needs credentials)
    logger.info("Checking for settled markets...")
    try:
        pnl_conn = sqlite3.connect(str(Path(__file__).parent / "kalshi_data.db"))
        init_pnl_tables(pnl_conn)
        migrate_db(pnl_conn, verbose=True)
        with redirect_stdout(_LoggerWriter(logger)):
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

    # Step 2: Learn FIRST — re-verify against official truth, verify recent
    # days, retrain — so today's decisions use today's best model.
    learn_cities = CITIES_ALL  # every city keeps accumulating data
    logger.info("Running learning cycle (verify → retrain) before trading...")
    try:
        conn = sqlite3.connect(str(Path(__file__).parent / "kalshi_data.db"))
        init_prediction_log(conn)
        init_historical_tables(conn)
        migrate_db(conn, verbose=True)
        with redirect_stdout(_LoggerWriter(logger)):
            learn(conn, learn_cities, retrain=True)
            # A model file that exists but will not load (library mismatch,
            # partial write) must be rebuilt here, on this machine, rather
            # than silently dropping the city from the rotation.
            from daily_learner import retrain_model
            from train_model import load_model
            for c in learn_cities:
                path = BOT_DIR / f"forecast_model_{c}.pkl"
                if not path.exists():
                    continue
                try:
                    load_model(str(path))
                except Exception as e:
                    print(f"  [{c}] model file unreadable ({e}) — retraining")
                    retrain_model(conn, c)
        conn.close()
    except Exception as e:
        logger.warning(f"  Learning cycle failed: {e}")

    if cities is None:
        # Trade every configured city that has a trained model. New cities
        # enter the rotation automatically once backfill_history.py has
        # bootstrapped their data and trained a model — never before, so an
        # untrained naive-average forecast can never place real orders.
        cities = tradeable_cities()
        skipped = [c for c in CITIES_ALL if c not in cities]
        if skipped:
            logger.info(f"  Cities awaiting trained models (run "
                        f"backfill_history.py): {skipped}")

    # Step 3: Pre-flight balance check (live only)
    if not dry_run:
        try:
            from kalshi_client import create_client_from_config
            client = create_client_from_config()
            balance = client.get_balance()
            available = balance.get("balance", 0) / 100  # API returns cents
            logger.info(f"  Account balance: ${available:.2f}")
        except Exception as e:
            logger.error(f"  Could not fetch balance: {e}")
            logger.error(f"  Aborting run for safety.")
            return

    # Step 4: Trade
    total_results = _execute_trading(cities, max_spend, min_edge, logger,
                                     dry_run=dry_run)

    successes = [r for r in total_results if r.success]
    logger.info(f"\nRUN COMPLETE: {len(successes)} orders placed across {len(cities)} cities")
    total_spent = sum(_filled_cost(r) for r in successes)
    if successes:
        logger.info(f"  Total spent: ${total_spent:.2f}")

    # Step 5: Record today's/tomorrow's forecasts for later verification
    logger.info("Recording forecasts for verification...")
    try:
        conn = sqlite3.connect(str(Path(__file__).parent / "kalshi_data.db"))
        init_prediction_log(conn)
        migrate_db(conn)
        with redirect_stdout(_LoggerWriter(logger)):
            record(conn, CITIES_ALL)
        conn.close()
    except Exception as e:
        logger.warning(f"  Recording forecasts failed: {e}")


def _filled_cost(r) -> float:
    try:
        from trader import filled_cost
        return filled_cost(r)
    except Exception:
        return r.order.cost_dollars


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
                     min_edge: float, logger: logging.Logger,
                     dry_run: bool = False) -> list:
    """
    Execute one trading pass. SIZING_MODE in config.py selects:
      "global"   — pooled cross-city sizing (default)
      "per_city" — legacy even split, the proven-but-small Aug 18-31
                   behavior (the one-line revert switch)
    """
    from trader import run_global_pipeline

    try:
        from config import SIZING_MODE
    except ImportError:
        SIZING_MODE = "global"
    if SIZING_MODE == "per_city":
        return _execute_trading_per_city(cities, max_spend, min_edge, logger,
                                         dry_run=dry_run)

    total_results = []
    logger.info(f"\n--- Trading {len(cities)} cities (global sizing"
                f"{', DRY RUN' if dry_run else ''}) ---")
    try:
        with redirect_stdout(_LoggerWriter(logger)):
            total_results = run_global_pipeline(
                city_keys=cities,
                dry_run=dry_run,
                min_edge=min_edge,
                tomorrow_only=True,
                max_spend=max_spend,
            )
        total_spent = sum(_filled_cost(r) for r in total_results if r.success)
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
                              min_edge: float, logger: logging.Logger,
                              dry_run: bool = False) -> list:
    """Legacy per-city budget split (pre-2026-09-01). Kept as the revert
    path: proven Aug 18-31 live record, smaller bets."""
    from trader import run_trading_pipeline
    from kalshi_client import create_client_from_config

    try:
        from config import MAX_RUN_EXPOSURE_PCT as _pct
    except ImportError:
        _pct = 0.25
    if dry_run:
        available = 100.0
    else:
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
                    city_key=city, dry_run=dry_run, max_spend=city_budget,
                    min_edge=min_edge, tomorrow_only=True)
            total_results.extend(results)
            total_spent += sum(_filled_cost(r) for r in results if r.success)
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


def run_daemon(run_hour: int = None, cities: list[str] = None,
               max_spend: float = None):
    """
    Run continuously, executing trades once per day at run_hour (local time).
    Useful if you don't want to set up cron.
    """
    if run_hour is None:
        run_hour = RUN_HOUR
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
  python scheduler.py daemon                 # Run daily at config.RUN_HOUR
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
    run_parser.add_argument("--min-edge", type=float, default=None,
                            help="Minimum edge to trade (default: "
                                 "config.MIN_EDGE_CENTS, else 5%%)")
    run_parser.add_argument("--dry-run", action="store_true",
                            help="Simulate without placing orders")

    # 'daemon' subcommand
    daemon_parser = subparsers.add_parser("daemon", help="Run continuously")
    daemon_parser.add_argument("--hour", type=int, default=None,
                               help=f"Hour to run each day (default: config.RUN_HOUR = {RUN_HOUR})")
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
        # Dry runs share the LIVE code path (settlement check if credentials
        # exist, learning cycle, pooled sizing) with orders simulated, so
        # paper numbers rehearse exactly the code that trades.
        run_once(cities=args.cities, max_spend=args.max_spend,
                 min_edge=args.min_edge, dry_run=args.dry_run)

    elif args.command == "daemon":
        run_daemon(run_hour=args.hour, cities=args.cities,
                   max_spend=args.max_spend)

    else:
        # No command given — show status
        print_status()
