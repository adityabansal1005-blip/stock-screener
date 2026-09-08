"""
10-year OHLCV cache for the Strategy Research Engine.
Downloads daily data via yfinance, stores in SQLite for instant re-use.
Each symbol is refreshed at most once per day (staleness check on load).
"""
import sqlite3
import threading
import contextlib
import io
import hashlib
from datetime import datetime, date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

_DB_PATH  = Path(__file__).parent / "strategy_data.db"
_lock     = threading.Lock()
_YF_START = "2015-01-01"   # ~10 years back from 2025


# ── Schema ────────────────────────────────────────────────────
_DDL = """
CREATE TABLE IF NOT EXISTS ohlcv (
    symbol   TEXT    NOT NULL,
    date     TEXT    NOT NULL,
    open     REAL,
    high     REAL,
    low      REAL,
    close    REAL,
    volume   REAL,
    PRIMARY KEY (symbol, date)
);
CREATE INDEX IF NOT EXISTS ix_ohlcv_symbol ON ohlcv(symbol);

CREATE TABLE IF NOT EXISTS cache_meta (
    symbol      TEXT PRIMARY KEY,
    last_fetched TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS series_provenance (
    symbol TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    price_basis TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS series_review_queue (
    symbol TEXT PRIMARY KEY,
    attempted_date TEXT NOT NULL,
    reason TEXT NOT NULL,
    snapshot_path TEXT NOT NULL
);
"""


@contextlib.contextmanager
def _conn():
    c = sqlite3.connect(str(_DB_PATH))
    try:
        c.execute("PRAGMA journal_mode=WAL")
        with c:
            yield c
    finally:
        c.close()


def _init():
    with _lock, _conn() as c:
        for stmt in _DDL.strip().split(";"):
            s = stmt.strip()
            if s:
                c.execute(s)
        c.commit()


_init()


def _yf_fetch(symbol: str) -> pd.DataFrame | None:
    """Download full history from yfinance with stderr suppressed."""
    with contextlib.redirect_stderr(io.StringIO()):
        df = yf.Ticker(symbol).history(start=_YF_START, interval="1d", auto_adjust=True)
    if df is None or len(df) < 200:
        return None
    df = df.rename(columns=str.lower)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "date"
    return df[["open", "high", "low", "close", "volume"]].dropna()


def _groww_fetch(symbol: str) -> pd.DataFrame | None:
    """
    Full history via Groww, chunked at ~1000 days per request.

    Preferred over yfinance for two reasons that matter to this project:
      * yfinance returns nothing for delisted / renamed NSE tickers, which is
        precisely the population a survivorship-free study needs (see
        survivorship.py). Groww serves their truncated history fine.
      * yfinance rate-limits above ~100 symbols; Groww is throttled and stable.
    """
    try:
        import groww_client
        if not groww_client.is_connected() or groww_client.data_forbidden():
            return None
    except Exception:
        return None

    sym   = symbol.replace(".NS", "").replace(".BO", "").upper()
    start = pd.Timestamp(_YF_START).date()
    today = date.today()

    # Walk backwards in explicit 1000-day windows. This MUST use the date-range
    # call: get_daily_bars(days=N) counts back from today, so looping it just
    # re-fetches the same recent window and never reaches deep history.
    frames, cursor = [], today
    while cursor > start:
        win_start = max(start, cursor - timedelta(days=1000))
        try:
            part = groww_client.get_daily_bars_range(sym, win_start, cursor)
        except Exception:
            part = None
        if part is not None and not part.empty:
            frames.append(part)
        if win_start <= start:
            break
        cursor = win_start - timedelta(days=1)

    if not frames:
        return None

    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df[df.index >= pd.Timestamp(start)]
    if len(df) < 200:
        return None

    df = df.rename(columns=str.lower)
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    return df[["open", "high", "low", "close", "volume"]].dropna()


def _nselib_fetch(symbol: str) -> pd.DataFrame | None:
    """
    NSE direct, via the scanner's parser. Sits between Groww and yfinance
    because it resolves symbols the others cannot — notably renamed tickers
    (ZOMATO -> ETERNAL), where Groww returns nothing but NSE still serves the
    old name. Without this rung a retrievable symbol fell through to yfinance,
    which rate-limits and takes the whole job down with it.
    Only ~18 months deep, so it is a gap-filler, not an archive source.
    """
    try:
        import scanner
        d = scanner._nselib_ohlcv(symbol.replace(".NS", "").replace(".BO", ""))
        if d is None or len(d) < 200:
            return None
        d = d.rename(columns=str.lower)
        d.index = pd.to_datetime(d.index)
        d.index.name = "date"
        return d[["open", "high", "low", "close", "volume"]].dropna()
    except Exception:
        return None


def _fetch(symbol: str) -> pd.DataFrame | None:
    """Groww -> nselib -> yfinance. yfinance is last: it rate-limits hard."""
    for src in (_groww_fetch, _nselib_fetch, _yf_fetch):
        try:
            df = src(symbol)
        except Exception as e:
            # A rate-limited yfinance must not abort a 100-symbol training run.
            print(f"[DataMgr] {symbol}: {src.__name__} failed ({str(e)[:60]})")
            continue
        if df is not None and len(df) >= 200:
            df.attrs['provider'] = src.__name__
            df.attrs['price_basis'] = 'total_return_adjusted' if src is _yf_fetch else 'unknown'
            return df
    return None


def _store(symbol: str, df: pd.DataFrame):
    # Retain each provider's incoming series before deciding whether it can update
    # legacy data. Unknown provenance is never silently mixed with a new scale.
    from fund_engine.data import Provenance, save_snapshot
    provider = df.attrs.get('provider', 'unknown')
    basis = df.attrs.get('price_basis', 'unknown')
    content_hash = hashlib.sha256(df.to_csv().encode()).hexdigest()
    symbol_id = hashlib.sha256(symbol.encode()).hexdigest()[:16]
    snapshot_id = hashlib.sha256((provider+'|'+basis+'|'+content_hash).encode()).hexdigest()
    series_dir = _DB_PATH.parent / 'data' / 'source_snapshots' / symbol_id / snapshot_id
    if not series_dir.exists():
        save_snapshot(df, series_dir, Provenance(provider, basis, 'unknown'))
    rows = [(symbol, d.strftime("%Y-%m-%d"), r.open, r.high, r.low, r.close, r.volume)
            for d, r in df.iterrows()]
    with _lock, _conn() as c:
        existing = c.execute('SELECT count(*) FROM ohlcv WHERE symbol=?', (symbol,)).fetchone()[0]
        provenance = c.execute('SELECT source,price_basis FROM series_provenance WHERE symbol=?', (symbol,)).fetchone()
        reason = None
        if existing and (provenance is None or tuple(provenance) != (provider,basis)):
            reason = 'unknown_or_changed_provider_basis'
        elif existing:
            # Same provider can revise an adjusted series after a corporate action.
            # A partial replacement must not leave older bars on another scale.
            old_dates = {r[0] for r in c.execute('SELECT date FROM ohlcv WHERE symbol=?',(symbol,))}
            if not old_dates.issubset({r[1] for r in rows}):
                reason = 'partial_history_requires_review'
        if reason:
            c.execute('INSERT OR REPLACE INTO series_review_queue VALUES(?,?,?,?)',
                      (symbol,date.today().isoformat(),reason,str(series_dir)))
            c.commit()
            print(f'[DataMgr] {symbol}: refresh staged for provenance review; existing history preserved')
            return False
        c.executemany(
            "INSERT OR REPLACE INTO ohlcv(symbol,date,open,high,low,close,volume) VALUES(?,?,?,?,?,?,?)",
            rows
        )
        c.execute(
            "INSERT OR REPLACE INTO cache_meta(symbol, last_fetched) VALUES(?,?)",
            (symbol, datetime.now().strftime("%Y-%m-%d"))
        )
        c.execute('INSERT OR REPLACE INTO series_provenance VALUES(?,?,?)',(symbol,provider,basis))
        c.execute('DELETE FROM series_review_queue WHERE symbol=?',(symbol,))
        c.commit()
    return True


def _is_stale(symbol: str) -> bool:
    """Returns True if data is missing or was last fetched before today."""
    today = date.today().isoformat()
    with _lock, _conn() as c:
        pending = c.execute('SELECT attempted_date FROM series_review_queue WHERE symbol=?',(symbol,)).fetchone()
        if pending and pending[0] == today:
            return False  # Avoid repeatedly downloading a quarantined series today.
        row = c.execute(
            "SELECT last_fetched FROM cache_meta WHERE symbol=?", (symbol,)
        ).fetchone()
    return row is None or row[0] < today


def load(symbol: str, force_refresh: bool = False) -> pd.DataFrame | None:
    """
    Load OHLCV for a symbol.  Fetches via Groww (falling back to yfinance)
    if stale or missing.
    Returns DataFrame with DatetimeIndex and columns: open, high, low, close, volume.
    """
    if force_refresh or _is_stale(symbol):
        print(f"[DataMgr] Fetching {symbol}...")
        df = _fetch(symbol)
        if df is not None:
            _store(symbol, df)
        else:
            print(f"[DataMgr] {symbol}: no data from any source")
            return None

    with _lock, _conn() as c:
        rows = c.execute(
            "SELECT date,open,high,low,close,volume FROM ohlcv WHERE symbol=? ORDER BY date",
            (symbol,)
        ).fetchall()

    if not rows:
        return None

    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    return df


def load_many(symbols: list[str], progress: bool = True) -> dict[str, pd.DataFrame]:
    """Load multiple symbols. Returns {symbol: df} for those that succeed."""
    out = {}
    for i, sym in enumerate(symbols):
        df = load(sym)
        if df is not None:
            out[sym] = df
        if progress and (i + 1) % 10 == 0:
            print(f"[DataMgr] {i+1}/{len(symbols)} loaded")
    return out


def list_cached() -> list[str]:
    with _lock, _conn() as c:
        return [r[0] for r in c.execute("SELECT symbol FROM cache_meta ORDER BY symbol").fetchall()]


def cache_stats() -> dict:
    with _lock, _conn() as c:
        total_rows = c.execute("SELECT COUNT(*) FROM ohlcv").fetchone()[0]
        total_syms = c.execute("SELECT COUNT(*) FROM cache_meta").fetchone()[0]
        oldest     = c.execute("SELECT MIN(date) FROM ohlcv").fetchone()[0]
        newest     = c.execute("SELECT MAX(date) FROM ohlcv").fetchone()[0]
    return {
        "cached_symbols": total_syms,
        "total_rows":     total_rows,
        "date_range":     f"{oldest} to {newest}",
    }
