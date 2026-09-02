"""
Export sanitized CSVs so the RESEARCH.md numbers are reproducible by a reviewer
without exposing the live SQLite DB or any account identifiers.

Drops order ids and timestamps-to-the-second; keeps everything needed to
recompute calibration, Brier scores, and P&L. Sniper rows carry both the
outcome graded on real settlement (`outcome`, `truth_source`) and the
pre-Sep-2026 outcome that was graded on reanalysis/obs-feed values
(`legacy_outcome`), so the re-grading is itself reproducible.

Usage:  python export_data.py     # writes data/live_trades.csv, data/sniper_signals.csv
"""
from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

DB = Path(__file__).parent / "kalshi_data.db"
OUT = Path(__file__).parent / "data"
OUT.mkdir(exist_ok=True)


def export(query: str, header: list[str], filename: str) -> int:
    c = sqlite3.connect(str(DB))
    rows = c.execute(query).fetchall()
    c.close()
    with open(OUT / filename, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return len(rows)


def main():
    n1 = export(
        """SELECT date(timestamp) AS date, ticker, side, model_prob, price_cents,
                  edge, kelly_fraction, settlement_result, profit_dollars
           FROM trades WHERE settled = 1 ORDER BY timestamp""",
        ["date", "ticker", "side", "model_prob", "price_cents", "edge",
         "kelly_fraction", "settlement_result", "profit_dollars"],
        "live_trades.csv",
    )
    n2 = export(
        """SELECT date, city, ticker, side, prob, ask_price, obs_max_f, rem_max_f,
                  hours_remaining, mode, outcome, truth_source, legacy_outcome,
                  hypo_profit, COALESCE(model_version, 'legacy') AS model_version
           FROM sniper_signals ORDER BY id""",
        ["date", "city", "ticker", "side", "prob", "ask_price", "obs_max_f",
         "rem_max_f", "hours_remaining", "mode", "outcome", "truth_source",
         "legacy_outcome", "hypo_profit", "model_version"],
        "sniper_signals.csv",
    )
    print(f"wrote data/live_trades.csv ({n1} rows), data/sniper_signals.csv ({n2} rows)")


if __name__ == "__main__":
    main()
