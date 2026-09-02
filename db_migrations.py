"""
Idempotent schema + data migrations for kalshi_data.db.

Every entry point that touches the database (scheduler, daily_learner,
sniper, backfills, dashboard) calls migrate_db() first, so a fresh checkout
and a years-old DB converge on the same schema without manual steps.

What this adds (all safe to re-run):
  daily_predictions.lead_days        days between issue date and target date
                                     (0 = recorded on the target day itself)
  historical_forecasts.source        'live' (collected by daily_learner) or
                                     'archive' (Open-Meteo backfills)
  historical_forecasts.lead_days     copied from daily_predictions for live rows
  historical_forecasts.lead_ok       1 when every forecast column is a genuine
                                     lead-1 value (see backfill_lead1.py)
  historical_forecasts.actual_source 'station' (NOAA GHCND, the settlement
                                     number) / 'feed' (NWS obs feed max,
                                     provisional) / 'era5' (reanalysis)
  decision_log                       what the live pipeline actually used at
                                     decision time (features, raw prediction,
                                     sigma, model version) — the basis for
                                     honest live recalibration

Live rows in historical_forecasts get their forecast columns restored from
daily_predictions: backfill_lead1.py had overwritten them with Previous-Runs
archive values, but the live feed IS the serving distribution and must be the
training anchor for those dates.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from weather import CITIES


def _add_column(conn: sqlite3.Connection, table: str, col_def: str) -> bool:
    """ALTER TABLE ... ADD COLUMN, returning True if the column was added."""
    col_name = col_def.split()[0]
    existing = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
    if col_name in existing:
        return False
    conn.execute(f"ALTER TABLE {table} ADD COLUMN {col_def}")
    conn.commit()
    return True


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    return row is not None


def lead_days_for(target_date: str, recorded_at_iso: str | None,
                  tz_name: str) -> int | None:
    """
    Lead time in whole days: target local date minus the local date the
    forecast was recorded. None when recorded_at is missing/unparseable.
    """
    if not recorded_at_iso:
        return None
    try:
        rec = datetime.fromisoformat(recorded_at_iso.replace("Z", "+00:00"))
        if rec.tzinfo is None:
            rec = rec.replace(tzinfo=ZoneInfo("UTC"))
        rec_local = rec.astimezone(ZoneInfo(tz_name)).date()
        tgt = datetime.strptime(target_date, "%Y-%m-%d").date()
        return (tgt - rec_local).days
    except (TypeError, ValueError):
        return None


def ensure_decision_log(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS decision_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            issued_at TEXT NOT NULL,       -- UTC ISO timestamp of the decision
            date TEXT NOT NULL,            -- target (market) local date
            city TEXT NOT NULL,
            lead_days INTEGER,
            model_trained_date TEXT,       -- pickle's trained_date (model version)
            raw_pred_f REAL,               -- trained model output, no corrections
            bias_applied_f REAL,           -- live bias subtracted from raw_pred_f
            final_pred_f REAL,             -- the number the probability model used
            sigma_f REAL,                  -- the sigma the probability model used
            model_spread_f REAL,
            n_sources INTEGER,
            features_json TEXT,            -- full feature dict for re-scoring
            used_trained_model INTEGER,
            purpose TEXT                   -- 'trade' (pipeline) / 'view' (dashboard)
        );
        CREATE INDEX IF NOT EXISTS idx_decision_log_date_city
            ON decision_log(date, city);
    """)
    _add_column(conn, "decision_log", "purpose TEXT")
    conn.commit()


def migrate_db(conn: sqlite3.Connection, verbose: bool = False) -> dict:
    """
    Apply all schema and data migrations. Returns a dict of counts.
    """
    stats = {}

    # ── daily_predictions: lead_days ────────────────────────────────
    if _table_exists(conn, "daily_predictions"):
        _add_column(conn, "daily_predictions", "lead_days INTEGER")
        _add_column(conn, "daily_predictions", "actual_source TEXT")
        rows = conn.execute("""
            SELECT date, city, recorded_at FROM daily_predictions
            WHERE lead_days IS NULL AND recorded_at IS NOT NULL
        """).fetchall()
        n = 0
        for date, city, rec in rows:
            tz = CITIES[city].timezone if city in CITIES else "UTC"
            ld = lead_days_for(date, rec, tz)
            if ld is not None:
                conn.execute("""UPDATE daily_predictions SET lead_days = ?
                                WHERE date = ? AND city = ?""", (ld, date, city))
                n += 1
        conn.commit()
        stats["daily_predictions.lead_days"] = n

    # ── historical_forecasts: source / lead / truth provenance ─────
    if _table_exists(conn, "historical_forecasts"):
        for col in ("source TEXT", "lead_days INTEGER", "lead_ok INTEGER",
                    "actual_source TEXT", "wk_forecast_f REAL", "wk_error REAL",
                    "icon_forecast_f REAL", "icon_error REAL", "era5_high_f REAL",
                    "gfs_lead0_f REAL", "ecmwf_lead0_f REAL", "blend_lead0_f REAL",
                    "icon_lead0_f REAL"):
            _add_column(conn, "historical_forecasts", col)

        untagged = conn.execute(
            "SELECT COUNT(*) FROM historical_forecasts WHERE source IS NULL"
        ).fetchone()[0]
        if untagged and _table_exists(conn, "daily_predictions"):
            stats.update(_tag_and_restore_live_rows(conn))
        if untagged:
            # Everything not claimed by daily_predictions is archive data.
            conn.execute("""
                UPDATE historical_forecasts SET source = 'archive'
                WHERE source IS NULL
            """)
            # A row is lead-1-clean only if backfill_lead1.py replaced EVERY
            # source (it preserves the pre-existing value in *_lead0_f when
            # it does). ECMWF/ICON have no Previous-Runs coverage before
            # 2024, so 2021-2023 rows are mixed-lead and excluded.
            conn.execute("""
                UPDATE historical_forecasts SET lead_ok = CASE
                    WHEN source = 'live' THEN
                        CASE WHEN lead_days IS NOT NULL AND lead_days >= 1
                             THEN 1 ELSE 0 END
                    WHEN gfs_lead0_f IS NOT NULL AND ecmwf_lead0_f IS NOT NULL
                         AND blend_lead0_f IS NOT NULL AND icon_lead0_f IS NOT NULL
                        THEN 1
                    ELSE 0 END
                WHERE lead_ok IS NULL
            """)
            # Archive actuals were replaced with GHCND by backfill_history
            # (era5_high_f holds the pre-replacement value when that ran).
            conn.execute("""
                UPDATE historical_forecasts SET actual_source = 'station'
                WHERE actual_source IS NULL AND source = 'archive'
                  AND era5_high_f IS NOT NULL
            """)
            conn.commit()
            stats["historical_forecasts.tagged"] = untagged

    ensure_decision_log(conn)
    if verbose and stats:
        print(f"  DB migration: {stats}")
    return stats


def _tag_and_restore_live_rows(conn: sqlite3.Connection) -> dict:
    """
    Mark rows that came from live collection and put the LIVE forecasts
    back into their forecast columns (backfill_lead1 overwrote them with
    Previous-Runs values). Errors and spread are recomputed against the
    row's current actual. Truth provenance is copied from daily_predictions
    (NULL there means 'unknown, provisional' — re-verification will settle it).
    """
    rows = conn.execute("""
        SELECT p.date, p.city, p.gfs_forecast_f, p.ecmwf_forecast_f,
               p.blend_forecast_f, p.icon_forecast_f, p.wk_forecast_f,
               p.lead_days, p.actual_source, h.actual_high_f, h.era5_high_f,
               p.model_prediction_f
        FROM daily_predictions p
        JOIN historical_forecasts h ON h.date = p.date AND h.city = p.city
        WHERE h.source IS NULL
    """).fetchall()
    n = 0
    n_truth = 0
    for (date, city, gfs, ecmwf, blend, icon, wk, lead, src, actual,
         era5_saved, model_pred) in rows:
        if era5_saved is not None and actual is not None:
            # backfill_history.fix_actuals_to_station already replaced this
            # row's actual with the official GHCND value (preserving the
            # provisional one in era5_high_f). That official number is the
            # truth for daily_predictions too — copy it back so the live
            # calibration window and the sniper gate stop using the feed.
            src = "station"
            errf = lambda v: (v - actual) if v is not None else None
            conn.execute("""
                UPDATE daily_predictions SET
                    actual_high_f = ?, actual_source = 'station',
                    model_error = ?, gfs_error = ?, ecmwf_error = ?,
                    blend_error = ?, icon_error = ?, wk_error = ?
                WHERE date = ? AND city = ?
            """, (actual, errf(model_pred), errf(gfs), errf(ecmwf),
                  errf(blend), errf(icon), errf(wk), date, city))
            n_truth += 1
        if gfs is None and ecmwf is None and blend is None:
            # Nothing usable was recorded live; keep the archive values but
            # still tag provenance so the row is treated consistently.
            conn.execute("""UPDATE historical_forecasts
                            SET source = 'live', lead_days = ?, actual_source = ?
                            WHERE date = ? AND city = ?""",
                         (lead, src, date, city))
            continue
        core = [v for v in (gfs, ecmwf, blend) if v is not None]
        spread = (max(core) - min(core)) if len(core) > 1 else 0.0
        err = lambda v: (v - actual) if (v is not None and actual is not None) else None
        conn.execute("""
            UPDATE historical_forecasts SET
                source = 'live', lead_days = ?, actual_source = ?,
                gfs_forecast_f = ?, ecmwf_forecast_f = ?, blend_forecast_f = ?,
                icon_forecast_f = ?, wk_forecast_f = ?,
                gfs_error = ?, ecmwf_error = ?, blend_error = ?,
                icon_error = ?, wk_error = ?, model_spread = ?
            WHERE date = ? AND city = ?
        """, (lead, src, gfs, ecmwf, blend, icon, wk,
              err(gfs), err(ecmwf), err(blend), err(icon), err(wk), spread,
              date, city))
        n += 1
    conn.commit()
    return {"historical_forecasts.live_restored": n,
            "daily_predictions.truth_from_ghcnd": n_truth}


def sync_training_row(conn: sqlite3.Connection, city_key: str, date: str,
                      weather: dict | None = None) -> bool:
    """
    Upsert one live-collected day into historical_forecasts from
    daily_predictions. Only official (station) truth at lead >= 1 is
    admitted — a same-day recording or a provisional feed value would
    contaminate the training target. Existing rows are updated in place
    (weather features are kept unless new ones are supplied).
    Returns True if a row was written.
    """
    row = conn.execute("""
        SELECT gfs_forecast_f, ecmwf_forecast_f, blend_forecast_f,
               icon_forecast_f, wk_forecast_f, actual_high_f,
               actual_source, lead_days
        FROM daily_predictions WHERE date = ? AND city = ?
    """, (date, city_key)).fetchone()
    if row is None:
        return False
    gfs, ecmwf, blend, icon, wk, actual, src, lead = row
    if actual is None or src != "station":
        return False
    if lead is None or lead < 1:
        return False
    if gfs is None and ecmwf is None and blend is None:
        return False

    dt = datetime.strptime(date, "%Y-%m-%d")
    core = [v for v in (gfs, ecmwf, blend) if v is not None]
    spread = (max(core) - min(core)) if len(core) > 1 else 0.0
    err = lambda v: (v - actual) if v is not None else None
    wx = weather or {}

    existing = conn.execute(
        "SELECT wind_speed_max, humidity_mean, cloud_cover_mean "
        "FROM historical_forecasts WHERE date = ? AND city = ?",
        (date, city_key)).fetchone()
    if existing:
        wind = wx.get("wind", existing[0])
        hum = wx.get("humidity", existing[1])
        cloud = wx.get("cloud", existing[2])
        conn.execute("""
            UPDATE historical_forecasts SET
                actual_high_f = ?, actual_source = 'station',
                source = 'live', lead_days = ?, lead_ok = 1,
                gfs_forecast_f = ?, ecmwf_forecast_f = ?, blend_forecast_f = ?,
                icon_forecast_f = ?, wk_forecast_f = ?,
                gfs_error = ?, ecmwf_error = ?, blend_error = ?,
                icon_error = ?, wk_error = ?, model_spread = ?,
                month = ?, day_of_year = ?,
                wind_speed_max = ?, humidity_mean = ?, cloud_cover_mean = ?
            WHERE date = ? AND city = ?
        """, (actual, lead, gfs, ecmwf, blend, icon, wk,
              err(gfs), err(ecmwf), err(blend), err(icon), err(wk), spread,
              dt.month, dt.timetuple().tm_yday, wind, hum, cloud,
              date, city_key))
    else:
        conn.execute("""
            INSERT INTO historical_forecasts (
                date, city, actual_high_f, actual_source, source, lead_days,
                lead_ok, gfs_forecast_f, ecmwf_forecast_f, blend_forecast_f,
                icon_forecast_f, wk_forecast_f,
                gfs_error, ecmwf_error, blend_error, icon_error, wk_error,
                month, day_of_year, model_spread,
                wind_speed_max, humidity_mean, cloud_cover_mean
            ) VALUES (?, ?, ?, 'station', 'live', ?, 1, ?, ?, ?, ?, ?,
                      ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (date, city_key, actual, lead, gfs, ecmwf, blend, icon, wk,
              err(gfs), err(ecmwf), err(blend), err(icon), err(wk),
              dt.month, dt.timetuple().tm_yday, spread,
              wx.get("wind"), wx.get("humidity"), wx.get("cloud")))
    conn.commit()
    return True


if __name__ == "__main__":
    from pathlib import Path
    conn = sqlite3.connect(str(Path(__file__).parent / "kalshi_data.db"))
    print(migrate_db(conn, verbose=True) or "nothing to do")
    conn.close()
