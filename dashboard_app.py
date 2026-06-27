"""
Live Trading Dashboard (Flask)
Real-time web dashboard for monitoring the Kalshi weather trading bot.

Features:
    - Live positions and cumulative P&L chart
    - Model predictions vs market prices for all cities
    - Kill switch controls (engage/disengage from the browser)
    - Trigger dry runs or live runs
    - Model health: calibration curve, MAE tracking, feature importances

Usage:
    python dashboard_app.py                  # start on port 5050
    python dashboard_app.py --port 8080      # custom port
"""

from __future__ import annotations

import sqlite3
import pickle
import subprocess
import sys
import threading
from datetime import datetime, timedelta
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from weather import CITIES
from pnl_tracker import init_pnl_tables, get_summary
from scheduler import kill_switch_active, engage_kill_switch, disengage_kill_switch
from train_model import get_model_path

BOT_DIR = Path(__file__).parent
DB_PATH = BOT_DIR / "kalshi_data.db"
STATIC_DIR = BOT_DIR / "dashboard_static"

app = Flask(__name__, static_folder=str(STATIC_DIR))


# ── Helpers ───────────────────────────────────────────────────────

def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    init_pnl_tables(conn)
    return conn


# ── Page routes ───────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(str(BOT_DIR), "dashboard_live.html")


# ── API: Summary + P&L ───────────────────────────────────────────

@app.route("/api/summary")
def api_summary():
    conn = get_db()
    summary = get_summary(conn)
    conn.close()
    return jsonify(summary)


@app.route("/api/trades")
def api_trades():
    conn = get_db()
    rows = conn.execute("""
        SELECT timestamp, ticker, side, contracts, price_cents,
               cost_dollars, model_prob, edge, settled,
               settlement_result, profit_dollars, settled_at
        FROM trades ORDER BY timestamp DESC LIMIT 100
    """).fetchall()
    conn.close()

    trades = []
    for r in rows:
        trades.append({
            "timestamp": r["timestamp"],
            "ticker": r["ticker"],
            "side": r["side"],
            "contracts": r["contracts"],
            "price_cents": r["price_cents"],
            "cost": r["cost_dollars"],
            "model_prob": r["model_prob"],
            "edge": r["edge"],
            "settled": bool(r["settled"]),
            "result": r["settlement_result"],
            "profit": r["profit_dollars"],
            "settled_at": r["settled_at"],
        })
    return jsonify(trades)


@app.route("/api/daily_pnl")
def api_daily_pnl():
    conn = get_db()
    rows = conn.execute("""
        SELECT DATE(timestamp) as day,
               SUM(cost_dollars) as invested,
               SUM(CASE WHEN settled = 1 THEN profit_dollars ELSE 0 END) as pnl,
               COUNT(*) as trades,
               SUM(CASE WHEN settled = 1 AND profit_dollars > 0 THEN 1 ELSE 0 END) as wins,
               SUM(CASE WHEN settled = 1 THEN 1 ELSE 0 END) as settled
        FROM trades GROUP BY DATE(timestamp) ORDER BY day
    """).fetchall()
    conn.close()

    daily = []
    cumulative = 0
    for r in rows:
        pnl = r["pnl"] or 0
        cumulative += pnl
        daily.append({
            "date": r["day"],
            "invested": r["invested"] or 0,
            "pnl": round(pnl, 2),
            "cumulative": round(cumulative, 2),
            "trades": r["trades"],
            "wins": r["wins"] or 0,
            "settled": r["settled"] or 0,
        })
    return jsonify(daily)


@app.route("/api/positions")
def api_positions():
    try:
        from kalshi_client import create_client_from_config
        client = create_client_from_config()
        resp = client.get_positions(limit=200)
        positions = resp.get("market_positions", [])
        active = [p for p in positions
                  if (p.get("yes_count", 0) or 0) + (p.get("no_count", 0) or 0) > 0]
        return jsonify(active)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/balance")
def api_balance():
    try:
        from kalshi_client import create_client_from_config
        client = create_client_from_config()
        balance = client.get_balance()
        return jsonify(balance)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── API: Settlement check ────────────────────────────────

@app.route("/api/settle", methods=["POST"])
def api_settle():
    """Manually trigger settlement checking."""
    try:
        from pnl_tracker import check_settlements
        conn = get_db()
        settled = check_settlements(conn)
        conn.close()
        return jsonify({"settled": settled})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── API: Predictions ─────────────────────────────────────────────

@app.route("/api/predictions")
def api_predictions():
    """Get model predictions vs market prices. Cached for 5 minutes."""
    try:
        from find_edge import get_market_prices, parse_price
        from model import predict_all_for_city

        predictions_data = []
        for city_key in CITIES:
            city = CITIES[city_key]
            try:
                markets = get_market_prices(city.kalshi_series)
                predictions = predict_all_for_city(city_key, markets)
                market_lookup = {m["ticker"]: m for m in markets}

                for pred in predictions:
                    m = market_lookup.get(pred.ticker, {})
                    yes_ask = parse_price(m.get("yes_ask_dollars", m.get("yes_ask")))
                    no_ask = parse_price(m.get("no_ask_dollars", m.get("no_ask")))
                    if yes_ask is None or no_ask is None:
                        continue
                    if (yes_ask <= 0.02 and no_ask >= 0.98) or (yes_ask >= 0.98 and no_ask <= 0.02):
                        continue

                    predictions_data.append({
                        "ticker": pred.ticker,
                        "city": city_key,
                        "description": pred.description,
                        "model_prob": round(pred.model_probability, 4),
                        "yes_ask": yes_ask,
                        "no_ask": no_ask,
                        "yes_edge": round(pred.model_probability - yes_ask, 4),
                        "no_edge": round((1 - pred.model_probability) - no_ask, 4),
                        "forecast_high": pred.forecast_high,
                        "error_std": round(pred.error_std, 2),
                    })
            except Exception as e:
                predictions_data.append({"city": city_key, "error": str(e)})

        return jsonify(predictions_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── API: Kill switch + controls ──────────────────────────────────

@app.route("/api/status")
def api_status():
    """Get scheduler status."""
    kill_active = kill_switch_active()

    # Check recent log
    today = datetime.now().strftime("%Y-%m-%d")
    log_file = BOT_DIR / "logs" / f"trader_{today}.log"
    recent_logs = []
    if log_file.exists():
        lines = log_file.read_text().strip().split("\n")
        recent_logs = lines[-15:]

    # Next scheduled run
    now = datetime.now()
    next_run = now.replace(hour=7, minute=0, second=0, microsecond=0)
    if now.hour >= 7:
        next_run += timedelta(days=1)

    return jsonify({
        "kill_switch_active": kill_active,
        "next_run": next_run.strftime("%Y-%m-%d %H:%M"),
        "recent_logs": recent_logs,
        "log_file": str(log_file.name) if log_file.exists() else None,
    })


@app.route("/api/kill", methods=["POST"])
def api_kill():
    engage_kill_switch("dashboard — user clicked kill switch")
    return jsonify({"status": "killed", "active": True})


@app.route("/api/resume", methods=["POST"])
def api_resume():
    disengage_kill_switch()
    return jsonify({"status": "resumed", "active": False})


_run_lock = threading.Lock()
_run_active = False

@app.route("/api/run", methods=["POST"])
def api_run():
    """Trigger a trading run in the background."""
    global _run_active

    if _run_active:
        return jsonify({"status": "error", "message": "A run is already in progress"}), 409

    data = request.json or {}
    dry_run = data.get("dry_run", True)
    cities = data.get("cities", ["chicago", "nyc", "miami"])

    cmd = [sys.executable, str(BOT_DIR / "scheduler.py"), "run",
           "--cities"] + cities + ["--max-spend", "10"]
    if dry_run:
        cmd.append("--dry-run")

    def run_in_background():
        global _run_active
        _run_active = True
        try:
            subprocess.run(cmd, cwd=str(BOT_DIR))
        finally:
            _run_active = False

    thread = threading.Thread(target=run_in_background, daemon=True)
    thread.start()

    mode = "dry run" if dry_run else "live"
    return jsonify({"status": f"started ({mode})", "cities": cities})


# ── API: Model health ────────────────────────────────────────────

@app.route("/api/calibration")
def api_calibration():
    """Get calibration data for the calibration curve."""
    conn = get_db()
    rows = conn.execute("""
        SELECT model_prob, profit_dollars
        FROM trades WHERE settled = 1 AND model_prob IS NOT NULL
    """).fetchall()
    conn.close()

    if len(rows) < 3:
        return jsonify({"buckets": [], "brier": None, "n": len(rows)})

    buckets = {}
    for r in rows:
        prob, profit = r["model_prob"], r["profit_dollars"]
        bucket = round(prob * 10) / 10
        bucket = max(0.0, min(1.0, bucket))
        if bucket not in buckets:
            buckets[bucket] = {"count": 0, "wins": 0, "total_prob": 0}
        buckets[bucket]["count"] += 1
        buckets[bucket]["wins"] += 1 if profit > 0 else 0
        buckets[bucket]["total_prob"] += prob

    result = []
    for bucket in sorted(buckets.keys()):
        b = buckets[bucket]
        result.append({
            "predicted": round(b["total_prob"] / b["count"], 3),
            "actual": round(b["wins"] / b["count"], 3),
            "count": b["count"],
        })

    brier = sum((r["model_prob"] - (1 if r["profit_dollars"] > 0 else 0)) ** 2
                for r in rows) / len(rows)

    return jsonify({"buckets": result, "brier": round(brier, 4), "n": len(rows)})


@app.route("/api/model_health")
def api_model_health():
    """Get model info: MAE, features, training data size, daily tracking."""
    models = {}
    for city_key in CITIES:
        model_path = BOT_DIR / get_model_path(city_key)
        if model_path.exists():
            with open(model_path, "rb") as f:
                data = pickle.load(f)
            models[city_key] = {
                "model_name": data.get("model_name"),
                "train_mae": round(data.get("train_mae", 0), 2),
                "residual_std": round(data.get("residual_std", 0), 2),
                "n_training_days": data.get("n_training_days", 0),
                "trained_date": data.get("trained_date"),
                "feature_names": data.get("feature_names", []),
                "baseline_mae": round(data.get("baseline_mae", 0), 2),
            }

            # Feature importances (GradientBoosting) or coefficients (Ridge)
            model = data.get("model")
            if hasattr(model, "feature_importances_"):
                models[city_key]["importances"] = {
                    name: round(float(imp), 4)
                    for name, imp in zip(data["feature_names"], model.feature_importances_)
                }
            elif hasattr(model, "coef_"):
                models[city_key]["coefficients"] = {
                    name: round(float(coef), 4)
                    for name, coef in zip(data["feature_names"], model.coef_)
                }

    # Daily tracking MAE
    conn = get_db()
    tracking = {}
    for city_key in CITIES:
        rows = conn.execute("""
            SELECT date, model_error, gfs_error, ecmwf_error, blend_error
            FROM daily_predictions
            WHERE city = ? AND actual_high_f IS NOT NULL
            ORDER BY date
        """, (city_key,)).fetchall()

        if rows:
            tracking[city_key] = [{
                "date": r["date"],
                "model_error": r["model_error"],
                "gfs_error": r["gfs_error"],
                "ecmwf_error": r["ecmwf_error"],
                "blend_error": r["blend_error"],
            } for r in rows]

    # Training data size
    data_size = {}
    for city_key in CITIES:
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM historical_forecasts WHERE city = ?",
            (city_key,)
        ).fetchone()
        data_size[city_key] = row["cnt"] if row else 0

    conn.close()

    return jsonify({
        "models": models,
        "tracking": tracking,
        "data_size": data_size,
    })


# ── Main ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Live trading dashboard")
    parser.add_argument("--port", type=int, default=5050, help="Port (default: 5050)")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    args = parser.parse_args()

    print(f"\n  Kalshi Weather Bot — Dashboard")
    print(f"  http://localhost:{args.port}")
    print(f"  Press Ctrl+C to stop\n")

    app.run(host="0.0.0.0", port=args.port, debug=args.debug)
