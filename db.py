"""
Database layer — SQLite locally, Postgres on Railway.
Switch by setting DATABASE_URL env var (Railway sets this automatically).

Schema:
  scan_log — one row per stock per scan run
"""
import json
import sqlite3
import os
import threading
from datetime import datetime, timedelta
from pathlib import Path

# Railway sets DATABASE_URL to postgres://...; locally we use SQLite
DATABASE_URL = os.environ.get("DATABASE_URL", "")
_USE_POSTGRES = DATABASE_URL.startswith("postgres")

_lock = threading.Lock()

# ── SQLite path ──────────────────────────────────────────────
_DB_PATH = Path(__file__).parent / "signals.db"

# ── Schema ───────────────────────────────────────────────────
_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS scan_cache (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    saved_at  TEXT NOT NULL,
    scan_type TEXT NOT NULL DEFAULT 'swing',
    universe  TEXT NOT NULL,
    results   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_scan_cache_type ON scan_cache(scan_type, saved_at);

CREATE TABLE IF NOT EXISTS scan_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    scanned_at  TEXT    NOT NULL,
    universe    TEXT    NOT NULL DEFAULT 'nifty100',
    scan_type   TEXT    NOT NULL DEFAULT 'swing',
    symbol      TEXT    NOT NULL,
    score100    REAL,
    verdict     TEXT,
    close       REAL,
    change_pct  REAL,
    live_price  REAL,
    rsi         REAL,
    win_rate    REAL,
    rr_ratio    REAL,
    sl_buy      REAL,
    target1     REAL,
    ai_verdict      TEXT,
    ai_confidence   INTEGER,
    ai_reasoning    TEXT,
    fund_score  INTEGER,
    pe          REAL,
    sector      TEXT,
    signals     TEXT,
    data_source TEXT
);

CREATE INDEX IF NOT EXISTS ix_scan_log_symbol     ON scan_log(symbol);
CREATE INDEX IF NOT EXISTS ix_scan_log_scanned_at ON scan_log(scanned_at);
"""


# ── Connection helpers ────────────────────────────────────────
def _get_conn():
    if _USE_POSTGRES:
        import psycopg2
        return psycopg2.connect(DATABASE_URL)
    return sqlite3.connect(str(_DB_PATH))


def init_db():
    """Create tables if they don't exist. Call once at startup."""
    with _lock:
        conn = _get_conn()
        try:
            cur = conn.cursor()
            for stmt in _CREATE_SQL.strip().split(";"):
                stmt = stmt.strip()
                if stmt:
                    cur.execute(stmt)
            # Preserve old rows as unversioned; identify new calculations.
            if not _USE_POSTGRES:
                columns = {r[1] for r in cur.execute("PRAGMA table_info(scan_log)")}
                for column in ("score_version", "bar_date"):
                    if column not in columns:
                        cur.execute(f"ALTER TABLE scan_log ADD COLUMN {column} TEXT")
            else:
                for column in ("score_version", "bar_date"):
                    cur.execute(f"ALTER TABLE scan_log ADD COLUMN IF NOT EXISTS {column} TEXT")
            conn.commit()
        finally:
            conn.close()


# ── Write ─────────────────────────────────────────────────────
def save_scan_results(results: list[dict], universe: str = "nifty100", scan_type: str = "swing"):
    """Persist a full scan result list to the database."""
    if not results:
        return
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    for r in results:
        rows.append((
            now,
            universe,
            scan_type,
            r.get("symbol", ""),
            r.get("score100"),
            r.get("verdict"),
            r.get("close"),
            r.get("change_pct"),
            r.get("live_price"),
            r.get("rsi"),
            r.get("win_rate"),
            r.get("rr_ratio"),
            r.get("sl_buy"),
            r.get("target1"),
            r.get("ai_verdict"),
            r.get("ai_confidence"),
            r.get("ai_reasoning"),
            r.get("fund_score"),
            r.get("pe"),
            r.get("sector"),
            json.dumps(r.get("signals", [])[:10]),  # cap at 10 signals
            r.get("data_source", "yfinance"),
            r.get("score_version"),
            r.get("bar_date"),
        ))

    placeholder = "?" if not _USE_POSTGRES else "%s"
    sql = f"""
        INSERT INTO scan_log
            (scanned_at, universe, scan_type, symbol, score100, verdict,
             close, change_pct, live_price, rsi, win_rate, rr_ratio, sl_buy, target1,
             ai_verdict, ai_confidence, ai_reasoning, fund_score, pe, sector, signals, data_source,
             score_version, bar_date)
        VALUES ({','.join([placeholder]*24)})
    """
    with _lock:
        conn = _get_conn()
        try:
            conn.executemany(sql, rows)
            conn.commit()
            print(f"[DB] Saved {len(rows)} rows ({scan_type}/{universe})")
        except Exception as e:
            print(f"[DB] Save failed: {e}")
        finally:
            conn.close()


# ── Read ──────────────────────────────────────────────────────
def get_history(limit: int = 200, symbol: str = "", scan_type: str = "swing", days: int = 7) -> list[dict]:
    """Fetch recent scan history. Returns list of dicts, newest first."""
    params = []
    where  = ["scan_type = ?"]
    params.append(scan_type)

    if symbol:
        where.append("symbol = ?")
        params.append(symbol.upper())

    where.append("scanned_at >= datetime('now', ?)")
    params.append(f"-{days} days")

    sql = f"""
        SELECT id, scanned_at, universe, symbol, score100, verdict,
               close, change_pct, live_price, rsi, win_rate, rr_ratio,
               sl_buy, target1, ai_verdict, ai_confidence, ai_reasoning,
               fund_score, pe, sector, signals, data_source
        FROM scan_log
        WHERE {' AND '.join(where)}
        ORDER BY scanned_at DESC, score100 DESC
        LIMIT ?
    """
    params.append(limit)

    if _USE_POSTGRES:
        sql = sql.replace("datetime('now', ?)", "NOW() - INTERVAL %s")
        sql = sql.replace("?", "%s")
        params[params.index(f"-{days} days")] = f"{days} days"

    rows = []
    with _lock:
        conn = _get_conn()
        try:
            conn.row_factory = sqlite3.Row if not _USE_POSTGRES else None
            cur = conn.cursor()
            cur.execute(sql, params)
            cols = [d[0] for d in cur.description]
            for row in cur.fetchall():
                d = dict(zip(cols, row))
                try:
                    d["signals"] = json.loads(d.get("signals") or "[]")
                except Exception:
                    d["signals"] = []
                rows.append(d)
        finally:
            conn.close()
    return rows


def save_full_scan(results: list[dict], universe: str = "nifty500", scan_type: str = "swing"):
    """
    Persist the full scan result list as a JSON blob so the dashboard can
    reload instantly on next startup without waiting for a fresh scan.
    Keeps only the 3 most recent full scans to cap disk use.
    """
    if not results:
        return
    import pandas as pd
    clean = []
    for r in results:
        row = {}
        for k, v in r.items():
            if k == "_df" or isinstance(v, (pd.DataFrame, pd.Series)):
                continue
            try:
                json.dumps(v)          # test serialisability
                row[k] = v
            except (TypeError, ValueError):
                row[k] = str(v)
        clean.append(row)

    blob = json.dumps(clean, default=str)
    now  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with _lock:
        conn = _get_conn()
        try:
            conn.execute(
                "INSERT INTO scan_cache (saved_at, scan_type, universe, results) VALUES (?,?,?,?)",
                (now, scan_type, universe, blob)
            )
            # keep only last 3 full scans
            conn.execute("""
                DELETE FROM scan_cache WHERE id NOT IN (
                    SELECT id FROM scan_cache WHERE scan_type=? ORDER BY id DESC LIMIT 3
                ) AND scan_type=?
            """, (scan_type, scan_type))
            conn.commit()
            print(f"[DB] Full scan cached ({len(clean)} stocks, {len(blob)//1024} KB)")
        except Exception as e:
            print(f"[DB] Full scan cache save failed: {e}")
        finally:
            conn.close()


# A scan is only allowed to replace a broader one if it is this much newer.
# Inside the window, breadth wins; past it, freshness does.
_CACHE_BREADTH_WINDOW_HRS = 12


def load_last_scan(scan_type: str = "swing") -> tuple[list[dict], str | None, str | None]:
    """
    Load the scan the dashboard should open on — the BROADEST recent one, not
    simply the newest.

    This used to be `ORDER BY id DESC LIMIT 1`, which quietly threw away the
    thing the user had waited for. `AUTO_SCAN_HOURS` fires a 50-stock nifty50
    scan at 09:20 / 11:00 / 13:00 / 15:00, and each one wrote a new cache row —
    so a 2,373-stock all_india scan that took four hours was buried by a 50-stock
    refresh ninety minutes later, and the dashboard showed 50 names with no
    indication that the other 2,323 existed. Nothing was lost; it was just
    invisible, which for this project amounts to the same thing.

    A narrow scan is an update to a slice, not a replacement of the picture. So
    within a 12-hour window the row covering the most symbols wins; beyond it,
    freshness takes over so a stale full scan cannot outlive its usefulness.
    Live LTP keeps prices current on whichever row is chosen.

    Returns (results, saved_at_str, universe_str) or ([], None, None) if none.
    """
    with _lock:
        conn = _get_conn()
        try:
            if not _USE_POSTGRES:
                conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute("""
                SELECT results, saved_at, universe FROM scan_cache
                WHERE scan_type=? ORDER BY id DESC LIMIT 12
            """, (scan_type,))
            rows = cur.fetchall()
            if not rows:
                return [], None, None

            def _fields(r):
                if _USE_POSTGRES:
                    return r[0], r[1], r[2]
                return r["results"], r["saved_at"], r["universe"]

            newest_raw, newest_at, newest_uni = _fields(rows[0])
            try:
                cutoff = (datetime.strptime(newest_at, "%Y-%m-%d %H:%M:%S")
                          - timedelta(hours=_CACHE_BREADTH_WINDOW_HRS))
            except (ValueError, TypeError):
                cutoff = None

            best = (json.loads(newest_raw), newest_at, newest_uni)
            if cutoff is not None:
                for r in rows:
                    raw, at, uni = _fields(r)
                    try:
                        if datetime.strptime(at, "%Y-%m-%d %H:%M:%S") < cutoff:
                            continue
                    except (ValueError, TypeError):
                        continue
                    parsed = json.loads(raw)
                    if len(parsed) > len(best[0]):
                        best = (parsed, at, uni)

            if best[1] != newest_at:
                print(f"[DB] Loading {best[2]} ({len(best[0]):,} stocks, {best[1]}) "
                      f"over the newer but narrower {newest_uni} "
                      f"({len(json.loads(newest_raw)):,} stocks, {newest_at})")
            return best
        except Exception as e:
            print(f"[DB] Full scan cache load failed: {e}")
            return [], None, None
        finally:
            conn.close()


def get_scan_summary(days: int = 7) -> dict:
    """Aggregate stats: how many scans, top symbols, verdict breakdown."""
    sql = """
        SELECT
            COUNT(DISTINCT scanned_at) as total_scans,
            COUNT(*) as total_rows,
            COUNT(DISTINCT symbol) as unique_symbols
        FROM scan_log
        WHERE scanned_at >= datetime('now', ?)
    """
    top_sql = """
        SELECT symbol, COUNT(*) as appearances, AVG(score100) as avg_score,
               MAX(scanned_at) as last_seen
        FROM scan_log
        WHERE scanned_at >= datetime('now', ?)
          AND score100 >= 60
        GROUP BY symbol
        ORDER BY appearances DESC, avg_score DESC
        LIMIT 10
    """
    verdict_sql = """
        SELECT verdict, COUNT(*) as cnt
        FROM scan_log
        WHERE scanned_at >= datetime('now', ?)
        GROUP BY verdict
    """
    param = (f"-{days} days",)
    result = {"total_scans": 0, "unique_symbols": 0, "top_symbols": [], "verdicts": {}}

    with _lock:
        conn = _get_conn()
        try:
            cur = conn.cursor()
            cur.execute(sql, param)
            row = cur.fetchone()
            if row:
                result["total_scans"]    = row[0]
                result["unique_symbols"] = row[2]

            cur.execute(top_sql, param)
            cols = [d[0] for d in cur.description]
            result["top_symbols"] = [dict(zip(cols, r)) for r in cur.fetchall()]

            cur.execute(verdict_sql, param)
            for sym, cnt in cur.fetchall():
                if sym:
                    result["verdicts"][sym] = cnt
        finally:
            conn.close()
    return result
