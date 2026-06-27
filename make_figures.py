"""
Generate the postmortem figures from the live trade log.

Reproduces, straight from kalshi_data.db, the two charts that tell the story:
  1. Reliability diagram — claimed probability vs realized win rate on the 92
     settled live trades, with model and market-implied Brier scores. The model
     sits below the diagonal (overconfident); the market price is the better
     probability forecaster on the traded subset.
  2. Cumulative P&L — the -$23 journey over the trade sequence.

Usage:  python make_figures.py        # writes assets/calibration.png
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DB = Path(__file__).parent / "kalshi_data.db"
OUT = Path(__file__).parent / "assets"
OUT.mkdir(exist_ok=True)


def load_trades():
    c = sqlite3.connect(str(DB))
    rows = c.execute("""
        SELECT timestamp, side, model_prob, price_cents, settlement_result, profit_dollars
        FROM trades WHERE settled = 1 ORDER BY timestamp
    """).fetchall()
    c.close()
    out = []
    for ts, side, mp, pc, result, profit in rows:
        win = 1 if side == result else 0          # contract settled on our side?
        mkt = pc / 100.0                           # market-implied prob of traded side
        out.append((ts, float(mp), mkt, win, float(profit)))
    return out


def brier(probs, wins):
    p = np.array(probs); w = np.array(wins)
    return float(np.mean((p - w) ** 2))


def main():
    data = load_trades()
    mp = [d[1] for d in data]
    mkt = [d[2] for d in data]
    wins = [d[3] for d in data]
    profit = [d[4] for d in data]
    n = len(data)

    b_model = brier(mp, wins)
    b_market = brier(mkt, wins)
    skill = 1 - b_model / b_market
    total = sum(profit)

    plt.rcParams.update({"font.size": 11, "axes.edgecolor": "#555",
                         "axes.grid": True, "grid.color": "#eee"})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ── Reliability diagram ──────────────────────────────────────────
    edges = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.01])
    centers, realized, counts = [], [], []
    mp_arr, w_arr = np.array(mp), np.array(wins)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (mp_arr >= lo) & (mp_arr < hi)
        if m.sum() == 0:
            continue
        centers.append(mp_arr[m].mean())
        realized.append(w_arr[m].mean())
        counts.append(int(m.sum()))

    ax1.plot([0, 1], [0, 1], "--", color="#888", lw=1.5, label="perfect calibration")
    sizes = [40 + 12 * c for c in counts]
    ax1.scatter(centers, realized, s=sizes, color="#c0392b", zorder=3,
                edgecolor="white", linewidth=1.2, label="model (size ∝ #trades)")
    for x, y, c in zip(centers, realized, counts):
        ax1.annotate(f"n={c}", (x, y), textcoords="offset points",
                     xytext=(8, -4), fontsize=9, color="#444")
    ax1.fill_between([0, 1], [0, 1], 1, color="#c0392b", alpha=0.05)
    ax1.text(0.34, 0.60, "overconfident\n(claimed > realized)", color="#c0392b",
             fontsize=10, ha="center", rotation=34)
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
    ax1.set_xlabel("Claimed probability of the traded side")
    ax1.set_ylabel("Realized win rate")
    ax1.set_title("Reliability diagram — 92 live trades")
    ax1.legend(loc="upper left", frameon=False, fontsize=9)
    ax1.text(0.97, 0.04,
             f"Brier  model {b_model:.3f}  vs  market {b_market:.3f}\n"
             f"Brier skill vs market: {skill:+.2f}  (negative = worse)",
             ha="right", va="bottom", fontsize=9.5,
             bbox=dict(boxstyle="round", fc="#fbeaea", ec="#c0392b", alpha=0.9))

    # ── Cumulative P&L ───────────────────────────────────────────────
    cum = np.cumsum(profit)
    ax2.axhline(0, color="#888", lw=1)
    ax2.plot(range(1, n + 1), cum, color="#2c3e50", lw=2)
    ax2.fill_between(range(1, n + 1), cum, 0, where=(cum < 0),
                     color="#c0392b", alpha=0.12)
    ax2.scatter([n], [cum[-1]], color="#c0392b", zorder=3)
    ax2.annotate(f"${total:,.2f}", (n, cum[-1]), textcoords="offset points",
                 xytext=(-10, 8), ha="right", color="#c0392b", fontweight="bold")
    ax2.set_xlabel("Settled trade #")
    ax2.set_ylabel("Cumulative P&L ($)")
    ax2.set_title("Live P&L — the result that drove the postmortem")

    fig.suptitle("Kalshi weather strategy: good point forecasts, miscalibrated "
                 "probabilities on the traded subset", fontsize=12.5, y=1.00)
    fig.tight_layout()
    path = OUT / "calibration.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"wrote {path}")
    print(f"  n={n}  Brier model={b_model:.3f} market={b_market:.3f} "
          f"skill={skill:+.2f}  total P&L=${total:.2f}")


if __name__ == "__main__":
    main()
