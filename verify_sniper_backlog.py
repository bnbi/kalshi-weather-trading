"""
Sniper Signal Verification Backfill

Grades every unverified sniper signal with sniper.verify_signals — the
exchange's own settlement result where the market is still fetchable,
otherwise the official GHCND daily high (direction-aware for thresholds)
— then prints the validation-gate status. The grading logic used to be
duplicated here with a bug (every threshold graded as "above").

    python verify_sniper_backlog.py
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from sniper import (init_sniper_table, verify_signals, validation_status,
                    VALIDATION_MIN_SIGNALS)
from db_migrations import migrate_db

DB_PATH = str(Path(__file__).parent / "kalshi_data.db")


def main():
    conn = sqlite3.connect(DB_PATH)
    init_sniper_table(conn)
    migrate_db(conn)
    before = conn.execute(
        "SELECT COUNT(*) FROM sniper_signals WHERE outcome IS NULL").fetchone()[0]
    print(f"Unverified signals: {before}")
    verified = verify_signals(conn)
    print(f"\nVerified {verified} signal(s)")

    s = validation_status(conn)
    print(f"\nValidation gate (model {s['model_version']}):")
    print(f"  Verified signals: {s['n_verified']} (need >= {VALIDATION_MIN_SIGNALS})"
          f"  [+{s['n_legacy']} from older model versions, excluded]")
    if s["n_verified"]:
        print(f"  Win rate:         {s['win_rate']:.0%} "
              f"(claimed avg: {s['avg_claimed_prob']:.0%})")
        print(f"  Hypothetical P&L: ${s['hypo_profit'] or 0:+.2f} "
              f"on ${s['hypo_staked'] or 0:.2f} staked")
        print(f"  Brier: model {s['brier_model']:.3f} vs market {s['brier_market']:.3f}")
    gate_msg = ("PASSED — sniper will trade live in --auto mode"
                if s["passed"] else "not passed — sniper stays in dry-run")
    print(f"  GATE {gate_msg}")
    conn.close()


if __name__ == "__main__":
    main()
