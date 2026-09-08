"""
Market context: Nifty price/change/VIX from NSE live API (no rate limits),
EMA trend + raw DataFrames from yfinance cached 24 hours.

Other modules (market_regime, scanner) use get_nifty_daily() / get_nifty_weekly()
so there is exactly ONE yfinance fetch for Nifty data per day, shared by all.
"""
import time
import threading
import requests
import numpy as np
import pandas as pd
from pathlib import Path

# ── NSE live session ──────────────────────────────────────────────────────────
_NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.nseindia.com/",
}
_nse_sess: requests.Session | None = None
_nse_sess_lock = threading.Lock()


def _get_nse_session() -> requests.Session:
    global _nse_sess
    with _nse_sess_lock:
        if _nse_sess is None:
            _nse_sess = requests.Session()
            _nse_sess.headers.update(_NSE_HEADERS)
            try:
                _nse_sess.get("https://www.nseindia.com", timeout=8)
            except Exception:
                pass
        return _nse_sess


# ── Live cache (Nifty price + VIX) — 5 min TTL ───────────────────────────────
_live_cache: dict = {}
_live_ts: float = 0.0
_LIVE_TTL = 300


def _fetch_live() -> dict:
    """NSE allIndices API — current price, change, VIX. No yfinance."""
    result = {}
    try:
        sess = _get_nse_session()
        r = sess.get("https://www.nseindia.com/api/allIndices", timeout=10)
        r.raise_for_status()
        for idx in r.json().get("data", []):
            name = idx.get("index", "")
            if name == "NIFTY 50":
                result["nifty_price"]  = round(float(idx["last"]), 2)
                result["nifty_change"] = round(float(idx["percentChange"]), 2)
            elif name == "INDIA VIX":
                vix = round(float(idx["last"]), 2)
                result["vix"]      = vix
                result["vix_high"] = vix > 20
    except Exception as e:
        print(f"[MarketContext] NSE live fetch failed: {e}")
    return result


# ── Historical Nifty DataFrames — 24h TTL, shared by market_regime + scanner ─
_nifty_daily_df:  pd.DataFrame | None = None
_nifty_weekly_df: pd.DataFrame | None = None
_trend_cache: dict = {}
_hist_ts: float = 0.0
_HIST_TTL = 86400   # 24 hours
_hist_lock = threading.Lock()


_NIFTY_CACHE_FILE = Path(__file__).parent / "nifty_history_cache.pkl"


def _load_disk_cache() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Load Nifty daily/weekly DataFrames from disk. Returns (None, None) if stale/missing."""
    try:
        if not _NIFTY_CACHE_FILE.exists():
            return None, None
        age = time.time() - _NIFTY_CACHE_FILE.stat().st_mtime
        if age > _HIST_TTL:        # older than 24h → stale
            return None, None
        import pickle
        with open(_NIFTY_CACHE_FILE, "rb") as f:
            data = pickle.load(f)
        daily  = data.get("daily")
        weekly = data.get("weekly")
        if daily is not None and len(daily) >= 50:
            print(f"[MarketContext] Loaded Nifty history from disk cache ({len(daily)} days)")
            return daily, weekly
    except Exception:
        pass
    return None, None


def _save_disk_cache(daily: pd.DataFrame, weekly: pd.DataFrame) -> None:
    try:
        import pickle
        with open(_NIFTY_CACHE_FILE, "wb") as f:
            pickle.dump({"daily": daily, "weekly": weekly}, f)
    except Exception:
        pass


def _fetch_nifty_via_nselib(days: int = 730) -> pd.DataFrame | None:
    """
    Fetch Nifty 50 daily OHLCV via nselib in monthly chunks (NSE index_data caps ~70 rows/call).
    Concatenates chunks to build the full history needed for EMA 200.
    """
    try:
        from nselib import capital_market
        from datetime import datetime, timedelta
        end = datetime.now()
        # NSE index_data returns ~70 rows per call; fetch in 90-day chunks
        all_chunks = []
        chunk_end = end
        for _ in range(9):          # up to 9 chunks × ~70 rows ≈ 630 rows (>2 years)
            chunk_start = chunk_end - timedelta(days=90)
            if chunk_start < end - timedelta(days=days):
                chunk_start = end - timedelta(days=days)
            df_chunk = capital_market.index_data(
                "NIFTY 50",
                from_date=chunk_start.strftime("%d-%m-%Y"),
                to_date=chunk_end.strftime("%d-%m-%Y"),
            )
            if df_chunk is not None and not df_chunk.empty:
                all_chunks.append(df_chunk)
            chunk_end = chunk_start
            if chunk_start <= end - timedelta(days=days):
                break

        if not all_chunks:
            return None
        df = pd.concat(all_chunks, ignore_index=True)
        # Rename columns
        df = df.rename(columns={
            "OPEN_INDEX_VAL":  "Open",
            "HIGH_INDEX_VAL":  "High",
            "LOW_INDEX_VAL":   "Low",
            "CLOSE_INDEX_VAL": "Close",
            "TRADED_QTY":      "Volume",
        })
        if "Close" not in df.columns:
            return None
        # Parse date (format: "11-APR-2025")
        if "TIMESTAMP" in df.columns:
            df["Date"] = pd.to_datetime(df["TIMESTAMP"], format="%d-%b-%Y", errors="coerce")
            df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
        df = df[~df.index.duplicated(keep="last")]   # deduplicate overlap
        df = df.sort_index()
        print(f"[MarketContext] Nifty history loaded via nselib ({len(df)} days)")
        return df if len(df) >= 50 else None
    except Exception as e:
        print(f"[MarketContext] nselib Nifty fetch failed: {e}")
        return None


def _fetch_history() -> None:
    """
    Load Nifty historical data. Priority:
      1. Disk cache (if < 24h old) — instant, no network
      2. nselib (NSE direct API)   — free, reliable, no rate limits
      3. yfinance                  — last resort, rate-limited by Yahoo
    """
    global _nifty_daily_df, _nifty_weekly_df, _trend_cache, _hist_ts

    # ── 1. Disk cache ─────────────────────────────────────────
    daily, weekly = _load_disk_cache()
    if daily is not None:
        _nifty_daily_df  = daily
        _nifty_weekly_df = weekly
        _hist_ts = time.time()
        _compute_trend(daily)
        return

    # ── 2. nselib (NSE direct, reliable, no rate limits) ──────
    daily = _fetch_nifty_via_nselib(days=730)
    weekly = None
    if daily is not None and not daily.empty:
        try:
            weekly = daily.resample("W").agg({
                "Open": "first", "High": "max", "Low": "min",
                "Close": "last", "Volume": "sum"
            }).dropna(subset=["Close"])
        except Exception:
            weekly = None

    # ── 3. yfinance last resort ───────────────────────────────
    if daily is None:
        import yfinance as yf, contextlib, io, requests as _req
        class _TS(_req.Session):
            def request(self, *a, **kw):
                kw.setdefault("timeout", (8, 15)); return super().request(*a, **kw)
        _sess = _TS()
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                daily = yf.Ticker("^NSEI", session=_sess).history(
                    period="2y", interval="1d", auto_adjust=True)
            if daily is not None and daily.empty:
                daily = None
        except Exception as e:
            print(f"[MarketContext] yfinance also failed: {e}")
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                weekly = yf.Ticker("^NSEI", session=_sess).history(
                    period="2y", interval="1wk", auto_adjust=True)
            if weekly is not None and weekly.empty:
                weekly = None
        except Exception:
            weekly = None

    with _hist_lock:
        if daily is not None and len(daily) >= 50:
            _nifty_daily_df = daily
            _compute_trend(daily)
            if weekly is not None and len(weekly) >= 15:
                _nifty_weekly_df = weekly
            _save_disk_cache(daily, weekly if weekly is not None else daily)   # persist so next startup is instant
        _hist_ts = time.time()


def _compute_trend(daily: pd.DataFrame) -> None:
    """Recompute EMA trend labels from a daily DataFrame. Updates _trend_cache in-place."""
    global _trend_cache
    try:
        close  = daily["Close"]
        last   = float(close.iloc[-1])
        ema50  = close.ewm(span=50,  adjust=False).mean()
        ema200 = close.ewm(span=200, adjust=False).mean()
        e50    = float(ema50.iloc[-1])
        e200   = float(ema200.iloc[-1])
        above_200 = not np.isnan(e200) and last > e200
        above_50  = last > e50
        trend = ("UPTREND" if above_50 else "PULLBACK IN UPTREND") if above_200 \
                else ("DOWNTREND" if not above_50 else "RECOVERY IN DOWNTREND")
        _trend_cache = {
            "nifty_trend":        trend,
            "nifty_above_200ema": above_200,
            "nifty_above_50ema":  above_50,
            "nifty_ema200":       round(e200, 2),
            "nifty_ema50":        round(e50, 2),
        }
    except Exception:
        pass


def _ensure_history() -> None:
    """Trigger a history refresh if cache is stale. Thread-safe."""
    if time.time() - _hist_ts > _HIST_TTL:
        _fetch_history()


def get_nifty_daily() -> pd.DataFrame | None:
    """Raw 2y daily Nifty DataFrame. Used by market_regime for ADX/EMA and scanner for RS."""
    _ensure_history()
    return _nifty_daily_df


def get_nifty_weekly() -> pd.DataFrame | None:
    """Raw 2y weekly Nifty DataFrame. Used by market_regime for weekly trend."""
    _ensure_history()
    return _nifty_weekly_df


# ── Public API ────────────────────────────────────────────────────────────────

def get_market_context() -> dict:
    global _live_cache, _live_ts

    now = time.time()

    # Live data (price + VIX) — NSE API, refresh every 5 min
    if now - _live_ts > _LIVE_TTL:
        fresh = _fetch_live()
        if fresh.get("nifty_price"):
            _live_cache = fresh
            _live_ts    = now
        elif not _live_cache:
            _live_cache = fresh

    # Historical trend — yfinance, once per day
    _ensure_history()

    ctx = {
        "nifty_price":        None,
        "nifty_change":       None,
        "nifty_trend":        "UNKNOWN",
        "nifty_above_200ema": None,
        "nifty_above_50ema":  None,
        "vix":                None,
        "vix_high":           False,
    }
    ctx.update(_trend_cache)
    ctx.update(_live_cache)
    return ctx
