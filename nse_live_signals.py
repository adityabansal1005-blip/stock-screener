"""
NSE Live Signals — real-time market intelligence from NSE India APIs.

Provides:
  - OI spurts: stocks with unusual open interest buildup (smart money positioning)
  - Most active: top volume/value stocks (where capital is flowing today)
  - Top gainers / losers: momentum movers
  - Sector performance: which sectors are leading/lagging
  - F&O ban list: stocks in futures ban (avoid new positions)
  - Index snapshots: Nifty, BankNifty, VIX, Nifty500 levels

All endpoints require a proper NSE browser session (homepage cookie).
Cache TTL: 15 minutes (market data changes fast but API is rate-limited).
"""
import time
import logging
import threading
import requests

log = logging.getLogger(__name__)

_CACHE_TTL = 900        # 15 minutes
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
    "Connection": "keep-alive",
    "DNT": "1",
}

_session_lock = threading.Lock()
_session: requests.Session | None = None
_session_ts: float = 0
_SESSION_TTL = 300   # refresh session cookie every 5 minutes

_caches: dict = {}


def _get_session() -> requests.Session:
    global _session, _session_ts
    now = time.time()
    with _session_lock:
        if _session is None or now - _session_ts > _SESSION_TTL:
            s = requests.Session()
            s.headers.update(_HEADERS)
            try:
                s.get("https://www.nseindia.com", timeout=10)
                time.sleep(0.4)
                # Also hit market page to get additional cookies
                s.get("https://www.nseindia.com/market-data/live-equity-market", timeout=8)
                time.sleep(0.2)
            except Exception:
                pass
            _session = s
            _session_ts = now
    return _session


def _fetch(endpoint: str, cache_key: str) -> dict | list | None:
    """Generic NSE API fetch with caching."""
    now = time.time()
    cached = _caches.get(cache_key)
    if cached and now - cached["ts"] < _CACHE_TTL:
        return cached["data"]

    try:
        sess = _get_session()
        url = f"https://www.nseindia.com/api/{endpoint}"
        r = sess.get(url, timeout=15)
        if r.status_code != 200:
            log.debug(f"[NSE] {endpoint} → {r.status_code}")
            return None
        data = r.json()
        _caches[cache_key] = {"data": data, "ts": now}
        return data
    except Exception as e:
        log.debug(f"[NSE] {endpoint} failed: {e}")
        return None


def _safe_float(v, default=0.0) -> float:
    try:
        return float(str(v).replace(",", "").replace("%", "").strip())
    except Exception:
        return default


# ── OI Spurts ────────────────────────────────────────────────────────────────

def get_oi_spurts() -> list[dict]:
    """
    Stocks with unusual open interest buildup vs average.
    High OI change relative to average = smart money building positions.
    Returns list of {symbol, oi_change_pct, avg_oi_ratio, latest_oi, volume}.
    """
    data = _fetch("live-analysis-oi-spurts-underlyings", "oi_spurts")
    if not data or not isinstance(data, dict):
        return []

    rows = data.get("data", data) if isinstance(data, dict) else data
    if not isinstance(rows, list):
        return []

    result = []
    for row in rows:
        sym = row.get("symbol") or row.get("underlying", "")
        if not sym:
            continue

        latest_oi = _safe_float(row.get("latestOI") or row.get("latestoi", 0))
        prev_oi   = _safe_float(row.get("prevOI") or row.get("prevoi", 0))
        avg_oi    = _safe_float(row.get("avgInOI") or row.get("avginoi") or row.get("averageOI", 1))
        vol       = _safe_float(row.get("volume", 0))
        fut_val   = _safe_float(row.get("futValue") or row.get("futvalue", 0))

        oi_change_pct = ((latest_oi - prev_oi) / prev_oi * 100) if prev_oi > 0 else 0
        avg_oi_ratio  = (latest_oi / avg_oi) if avg_oi > 0 else 1.0

        result.append({
            "symbol":        sym.upper(),
            "oi_change_pct": round(oi_change_pct, 1),
            "avg_oi_ratio":  round(avg_oi_ratio, 2),
            "latest_oi":     int(latest_oi),
            "volume":        int(vol),
            "fut_value_cr":  round(fut_val / 1e7, 1) if fut_val > 1e5 else round(fut_val, 1),
        })

    result.sort(key=lambda x: x.get("avg_oi_ratio", 0), reverse=True)
    return result


def get_oi_spurt_symbols() -> set[str]:
    """Fast set of symbols currently showing unusual OI buildup."""
    return {r["symbol"] for r in get_oi_spurts() if r.get("avg_oi_ratio", 0) >= 1.5}


# ── Most Active ───────────────────────────────────────────────────────────────

def get_most_active(by: str = "volume", limit: int = 30) -> list[dict]:
    """
    Top stocks by volume or value traded today.
    by: 'volume' or 'value'
    Returns {symbol, ltp, change_pct, volume, value_cr, year_high, year_low}.
    """
    endpoint = f"live-analysis-most-active-securities?index={by}&limit={limit}"
    data = _fetch(endpoint, f"most_active_{by}")
    if not data:
        return []

    rows = data.get("data", data) if isinstance(data, dict) else data
    if not isinstance(rows, list):
        return []

    result = []
    for row in rows:
        sym = row.get("symbol", "")
        if not sym:
            continue
        result.append({
            "symbol":     sym.upper(),
            "ltp":        _safe_float(row.get("lastPrice") or row.get("ltp") or row.get("close", 0)),
            "change_pct": _safe_float(row.get("pChange") or row.get("perChange", 0)),
            "volume":     int(_safe_float(row.get("totalTradedVolume") or row.get("volume", 0))),
            "value_cr":   round(_safe_float(row.get("totalTradedValue") or row.get("value", 0)) / 1e7, 1),
            "year_high":  _safe_float(row.get("yearHigh") or row.get("52wHigh", 0)),
            "year_low":   _safe_float(row.get("yearLow") or row.get("52wLow", 0)),
        })
    return result


def get_most_active_symbols(limit: int = 30) -> set[str]:
    """Set of top volume symbols today."""
    return {r["symbol"] for r in get_most_active("volume", limit)}


# ── Gainers / Losers ──────────────────────────────────────────────────────────

def get_gainers(index: str = "gainers", limit: int = 20) -> list[dict]:
    """
    Top gainers (or losers if index='loosers') for the day.
    Returns {symbol, ltp, change_pct, volume, prev_close}.
    """
    data = _fetch(f"live-analysis-variations?index={index}", f"movers_{index}")
    if not data:
        return []

    rows = data.get("data", data) if isinstance(data, dict) else data
    if not isinstance(rows, list):
        rows = data.get("NIFTY500", data.get("NIFTY 500", []))

    result = []
    for row in rows[:limit]:
        sym = row.get("symbol", "")
        if not sym:
            continue
        result.append({
            "symbol":     sym.upper(),
            "ltp":        _safe_float(row.get("ltp") or row.get("lastPrice", 0)),
            "change_pct": _safe_float(row.get("perChange") or row.get("pChange", 0)),
            "volume":     int(_safe_float(row.get("tradedQuantity") or row.get("volume", 0))),
            "prev_close": _safe_float(row.get("previousPrice") or row.get("prevClose", 0)),
        })
    return result


def get_losers(limit: int = 20) -> list[dict]:
    return get_gainers("loosers", limit)


def get_gainer_symbols(limit: int = 20) -> set[str]:
    return {r["symbol"] for r in get_gainers("gainers", limit)}


# ── F&O Ban List ──────────────────────────────────────────────────────────────

_ban_cache: dict = {"symbols": set(), "ts": 0}

def get_fno_ban_list() -> set[str]:
    """
    Symbols currently in F&O ban period (position limit breached).
    In ban: no new F&O positions allowed. Avoid swing trades too — volatility risk.
    """
    now = time.time()
    if now - _ban_cache["ts"] < _CACHE_TTL:
        return _ban_cache["symbols"]

    try:
        sess = _get_session()
        r = sess.get("https://www.nseindia.com/api/equity-stockIndices?index=SECURITIES%20IN%20F%26O%20BAN%20PERIOD", timeout=15)
        if r.status_code == 200:
            data = r.json()
            rows = data.get("data", [])
            symbols = {row.get("symbol", "").upper() for row in rows if row.get("symbol")}
            _ban_cache["symbols"] = symbols
            _ban_cache["ts"] = now
            return symbols
    except Exception as e:
        log.debug(f"[NSE] Ban list failed: {e}")

    # Fallback: NSE ban list as text
    try:
        sess = _get_session()
        r = sess.get("https://nsearchives.nseindia.com/content/fo/fo_secban.csv", timeout=15)
        if r.status_code == 200:
            lines = r.text.strip().splitlines()
            symbols = {line.strip().upper() for line in lines if line.strip() and not line.startswith("Security")}
            _ban_cache["symbols"] = symbols
            _ban_cache["ts"] = now
            return symbols
    except Exception as e:
        log.debug(f"[NSE] Ban list CSV failed: {e}")

    return set()


# ── Index Snapshots ────────────────────────────────────────────────────────────

def get_index_snapshot() -> dict:
    """
    Current levels for major indices: Nifty50, BankNifty, Nifty500, MidCap150, VIX.
    Useful for quick market context without yfinance.
    """
    data = _fetch("allIndices", "all_indices")
    if not data:
        return {}

    rows = data.get("data", [])
    result = {}
    target_names = {
        "NIFTY 50": "nifty50",
        "NIFTY BANK": "banknifty",
        "NIFTY 500": "nifty500",
        "NIFTY MIDCAP 150": "midcap150",
        "NIFTY SMALLCAP 100": "smallcap100",
        "INDIA VIX": "vix",
        "NIFTY IT": "nifty_it",
        "NIFTY PHARMA": "nifty_pharma",
        "NIFTY AUTO": "nifty_auto",
        "NIFTY FMCG": "nifty_fmcg",
        "NIFTY METAL": "nifty_metal",
        "NIFTY REALTY": "nifty_realty",
        "NIFTY FINANCIAL SERVICES": "nifty_fin",
        "NIFTY ENERGY": "nifty_energy",
    }
    for row in rows:
        name = row.get("index", row.get("indexSymbol", ""))
        key = target_names.get(name)
        if key:
            result[key] = {
                "last":       _safe_float(row.get("last") or row.get("lastPrice", 0)),
                "change_pct": _safe_float(row.get("percentChange") or row.get("pChange", 0)),
                "high":       _safe_float(row.get("high", 0)),
                "low":        _safe_float(row.get("low", 0)),
                "year_high":  _safe_float(row.get("yearHigh", 0)),
                "year_low":   _safe_float(row.get("yearLow", 0)),
            }
    return result


def get_sector_performance() -> dict[str, float]:
    """
    Returns {sector_key: change_pct_today} for major NSE sectoral indices.
    Use to identify which sectors are leading.
    """
    snapshot = get_index_snapshot()
    sectors = {}
    sector_keys = ["nifty_it", "nifty_pharma", "nifty_auto", "nifty_fmcg",
                   "nifty_metal", "nifty_realty", "nifty_fin", "nifty_energy"]
    for k in sector_keys:
        if k in snapshot:
            sectors[k] = snapshot[k].get("change_pct", 0.0)
    return sectors


# ── Market Intelligence Summary ────────────────────────────────────────────────

def get_market_intelligence() -> dict:
    """
    Aggregated market intelligence for today.
    Returns everything in one call — used to enrich scan results.

    {
      oi_spurt_symbols: set of str,
      most_active_symbols: set of str,
      gainer_symbols: set of str,
      fno_ban_symbols: set of str,
      sector_performance: dict,
      index_snapshot: dict,
    }
    """
    # Fetch in parallel using threads
    results = {}
    errors = {}

    def _run(key, fn):
        try:
            results[key] = fn()
        except Exception as e:
            errors[key] = str(e)
            results[key] = None

    tasks = [
        ("oi_spurts",     get_oi_spurt_symbols),
        ("most_active",   get_most_active_symbols),
        ("gainers",       get_gainer_symbols),
        ("fno_ban",       get_fno_ban_list),
        ("sectors",       get_sector_performance),
        ("indices",       get_index_snapshot),
    ]

    threads = [threading.Thread(target=_run, args=(k, fn), daemon=True) for k, fn in tasks]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20)

    return {
        "oi_spurt_symbols":  results.get("oi_spurts") or set(),
        "most_active_symbols": results.get("most_active") or set(),
        "gainer_symbols":    results.get("gainers") or set(),
        "fno_ban_symbols":   results.get("fno_ban") or set(),
        "sector_performance": results.get("sectors") or {},
        "index_snapshot":    results.get("indices") or {},
    }
