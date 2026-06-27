"""One-time script to log the first trade that was placed before the P&L tracker existed."""

import sqlite3
from pathlib import Path
from pnl_tracker import init_pnl_tables, log_trade

DB_PATH = Path(__file__).parent / "kalshi_data.db"

conn = sqlite3.connect(str(DB_PATH))
init_pnl_tables(conn)

# Log the trade from earlier today
log_trade(
    conn=conn,
    ticker="KXHIGHCHI-26MAY06-B57.5",
    side="no",
    action="buy",
    contracts=9,
    price_cents=55,
    cost_dollars=4.95,
    model_prob=0.876,  # 87.6% model probability of NO
    edge=0.336,        # 33.6% edge
    kelly_fraction=0.181,
    order_id="a92d2280-7eca-4c62-a459-6b4380cda588",
)

print("First trade logged successfully!")
print("Run 'python pnl_tracker.py summary' to see it.")

conn.close()
