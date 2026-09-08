"""
Bulk fundamentals via TradingView Screener.
One HTTP call fetches PE, ROE, D/E, margins, sector for all ~2900 NSE stocks.
Result is cached for 6 hours and shared across all per-stock lookups.

Falls back to an empty dict if TradingView is unreachable, so yfinance
per-stock calls in fundamentals.py still work as a backup.
"""
import time
import threading
import math

_CACHE_TTL = 6 * 3600
_cache: dict = {}          # {symbol_clean: fundamentals_dict}
_cache_ts: float = 0.0
_lock = threading.Lock()


def _pct(val) -> float | None:
    try:
        f = float(val)
        return None if math.isnan(f) else round(f, 1)
    except (TypeError, ValueError):
        return None


def _f(val, d=2) -> float | None:
    try:
        f = float(val)
        return None if math.isnan(f) else round(f, d)
    except (TypeError, ValueError):
        return None


def _fund_score(roe, de, net_margin) -> int:
    score = 50
    if roe is not None:
        if roe >= 20:    score += 15
        elif roe >= 15:  score += 10
        elif roe >= 10:  score += 5
        elif roe < 0:    score -= 20
        elif roe < 5:    score -= 10
    if de is not None:
        if de <= 0.5:    score += 10
        elif de <= 1.0:  score += 5
        elif de <= 2.0:  pass
        elif de <= 4.0:  score -= 5
        else:            score -= 15
    if net_margin is not None:
        if net_margin >= 20:   score += 5
        elif net_margin >= 10: score += 3
        elif net_margin < 0:   score -= 10
    return max(0, min(100, score))


def _fetch_all() -> dict:
    """Fetch all NSE fundamentals in one bulk call. Returns {SYMBOL: dict}."""
    try:
        from tradingview_screener import Query, Column
        _, df = (
            Query()
            .select(
                'name', 'close', 'market_cap_basic',
                'price_earnings_ttm', 'price_book_ratio',
                'return_on_equity', 'debt_to_equity',
                'net_margin', 'gross_margin',
                'dividends_yield', 'sector', 'industry',
            )
            .where(Column('exchange').isin(['NSE']))
            .set_markets('india')
            .limit(5000)
            .get_scanner_data()
        )

        result = {}
        for row in df.itertuples(index=False):
            sym = str(row.name).strip().upper()
            roe        = _pct(row.return_on_equity)
            de         = _f(row.debt_to_equity)
            net_margin = _pct(row.net_margin)
            pe         = _f(row.price_earnings_ttm, 1)
            pb         = _f(row.price_book_ratio)
            mcap       = row.market_cap_basic
            mcap_cr    = round(float(mcap) / 1e7, 0) if mcap and not math.isnan(float(mcap)) else None

            result[sym] = {
                "pe":              pe,
                "pb":              pb,
                "roe":             roe,
                "revenue_growth":  None,   # not available from TV screener
                "earnings_growth": None,
                "debt_equity":     de,
                "debt_equity_unit": "ratio",
                "net_margin":      net_margin,
                "div_yield":       _pct(row.dividends_yield),
                "market_cap_cr":   mcap_cr,
                "sector":          str(row.sector) if row.sector else "",
                "industry":        str(row.industry) if row.industry else "",
                "fund_score":      _fund_score(roe, de, net_margin),
                "52w_high":        None,
                "52w_low":         None,
                "near_high_pct":   None,
            }
        print(f"[TV] Bulk fundamentals loaded — {len(result)} NSE symbols")
        return result
    except Exception as e:
        print(f"[TV] Bulk fetch failed: {e} — will fall back to yfinance per-stock")
        return {}


def warm_cache() -> None:
    """Pre-fetch all fundamentals into the in-memory cache. Call once before scan."""
    global _cache, _cache_ts
    now = time.time()
    with _lock:
        if _cache and now - _cache_ts < _CACHE_TTL:
            return
        data = _fetch_all()
        if data:
            _cache = data
            _cache_ts = now


def get(symbol_ns: str) -> dict | None:
    """
    Look up pre-fetched fundamentals for a symbol (e.g. 'RELIANCE.NS').
    Returns None if not in cache (caller falls back to yfinance).
    """
    sym = symbol_ns.upper().replace(".NS", "").strip()
    return _cache.get(sym)
