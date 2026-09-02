"""
Sniper Signal Verification Backfill

Scores every unverified sniper signal against the OFFICIAL settlement-station
daily high (NOAA GHCND / NWS obs), using the same settlement logic as
sniper.verify_signals. Unblocks the self-validation gate, which can only
accumulate evidence from verified signals.

    python verify_sniper_backlog.py
"""

from __future__ import annotations

import sqlite3
import time

from sniper import parse_contract_ticker, validation_status, VALIDATION_MIN_SIGNALS
from station_obs import fetch_station_daily_high
from find_edge import kalshi_fee_per_contract

from pathlib import Path
DB_PATH = str(Path(__file__).parent / "kalshi_data.db")


def main():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT id, date, city, ticker, side, ask_price
        FROM sniper_signals
        WHERE outcome IS NULL
        ORDER BY date
    """).fetchall()

    print(f"Unverified signals: {len(rows)}")

    # Fetch each needed (city, date) actual only once
    needed = sorted({(city, date) for _, date, city, _, _, _ in rows})
    actuals = {}
    for city, date in needed:
        high = fetch_station_daily_high(city, date)
        actuals[(city, date)] = high
        status = f"{high}°F" if high is not None else "unavailable"
        print(f"  {city} {date}: {status}")
        time.sleep(0.3)

    verified = 0
    for sid, date, city, ticker, side, ask in rows:
        actual = actuals.get((city, date))
        if actual is None:
            continue

        info = parse_contract_ticker(ticker)
        high = round(actual)
        if info["type"] == "threshold":
            yes_settled = high > info["threshold"]
        elif info["type"] == "bracket":
            yes_settled = info["bracket_low"] <= high <= info["bracket_high"]
        else:
            continue

        won = yes_settled if side == "yes" else not yes_settled
        # Net of the exchange fee, matching sniper.verify_signals — the
        # validation gate must be graded on live economics.
        fee = kalshi_fee_per_contract(ask)
        profit = (1 - ask - fee) if won else (-ask - fee)
        conn.execute("""
            UPDATE sniper_signals SET outcome = ?, hypo_profit = ?
            WHERE id = ?
        """, ("win" if won else "loss", profit, sid))
        verified += 1

    conn.commit()
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
    gate_msg = ("PASSED — sniper will trade live in --auto mode"
                if s["passed"] else "not passed — sniper stays in dry-run")
    print(f"  GATE {gate_msg}")
    conn.close()


if __name__ == "__main__":
    main()
