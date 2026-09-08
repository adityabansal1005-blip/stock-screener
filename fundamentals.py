"""
Fetches fundamental data (PE, PB, ROE, D/E, revenue growth, net margin, etc.)
via yfinance .info. Cached 6 hours per symbol — fundamentals change quarterly.
Hard 12s thread timeout prevents the scan from hanging on slow tickers.
"""
import time
import threading
import yfinance as yf

_TIMEOUT   = 12           # seconds before giving up on a .info call
_CACHE_TTL = 6 * 3600     # 6 hours
_cache: dict = {}
_sem = threading.Semaphore(5)  # max 5 concurrent yfinance .info calls


def _pct(val) -> float | None:
    """Convert decimal ratio (e.g. 0.15) → percentage (15.0). Returns None on bad input."""
    try:
        f = float(val)
        return None if f != f else round(f * 100, 1)   # NaN guard
    except (TypeError, ValueError):
        return None


def _f(val, d=2) -> float | None:
    try:
        f = float(val)
        return None if f != f else round(f, d)
    except (TypeError, ValueError):
        return None


def _fund_score(roe, rev_growth, de, net_margin) -> int:
    """
    0–100 fundamental quality score.
    High ROE + revenue growth + low debt + good margins = high score.
    Heavily penalises negative ROE or extreme leverage.
    """
    score = 50

    if roe is not None:
        if roe >= 20:    score += 15
        elif roe >= 15:  score += 10
        elif roe >= 10:  score += 5
        elif roe < 0:    score -= 20
        elif roe < 5:    score -= 10

    if rev_growth is not None:
        if rev_growth >= 20:   score += 10
        elif rev_growth >= 10: score += 5
        elif rev_growth >= 0:  score += 2
        elif rev_growth < -10: score -= 15
        elif rev_growth < 0:   score -= 7

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


def get_fundamentals(symbol_ns: str, *, bulk_only: bool = False) -> dict:
    """
    symbol_ns must include the exchange suffix (e.g. 'RELIANCE.NS').
    Returns a dict with: pe, pb, roe, revenue_growth, earnings_growth,
    debt_equity, net_margin, div_yield, market_cap_cr, sector, industry,
    fund_score, 52w_high, 52w_low, near_high_pct, plus Screener.in data:
    roce, promoter_holding, promoter_pledged, fii_change, dii_change,
    quality_score, red_flags, green_flags.
    Returns {} on timeout or any error.

    Checks TradingView bulk cache first (populated by tv_fundamentals.warm_cache()
    at scan start). Falls back to yfinance per-stock only if TV cache misses.
    Then enriches with Screener.in data (24h cached).
    """
    # Fast path: TradingView bulk cache (no HTTP call needed)
    try:
        import tv_fundamentals
        tv = tv_fundamentals.get(symbol_ns)
        if tv is not None:
            result = dict(tv)
            if not bulk_only:
                _enrich_screener(symbol_ns, result)
            result['fundamentals_source'] = 'TradingView bulk'
            return result
    except Exception:
        pass

    if bulk_only:
        # Full-universe refresh must not launch thousands of optional web
        # scrapes or per-symbol Yahoo requests. Missing is not a zero value.
        return {}

    now = time.time()
    if symbol_ns in _cache:
        data, ts = _cache[symbol_ns]
        if now - ts < _CACHE_TTL:
            return data

    result: dict = {}
    exc: list    = []

    def _fetch():
        with _sem:
            try:
                info = yf.Ticker(symbol_ns).info
                if not info or not info.get("symbol"):
                    return

                roe        = _pct(info.get("returnOnEquity"))
                rev_growth = _pct(info.get("revenueGrowth"))
                net_margin = _pct(info.get("profitMargins"))
                de_raw     = _f(info.get("debtToEquity"), 6)
                de         = de_raw / 100.0 if de_raw is not None else None

                w52_high = _f(info.get("fiftyTwoWeekHigh"))
                w52_low  = _f(info.get("fiftyTwoWeekLow"))
                price    = _f(info.get("currentPrice") or info.get("regularMarketPrice"))
                near_high_pct = (
                    round((price - w52_high) / w52_high * 100, 1)
                    if w52_high and price else None
                )
                mcap = info.get("marketCap")

                result.update({
                    "pe":              _f(info.get("trailingPE"), 1),
                    "pb":              _f(info.get("priceToBook")),
                    "roe":             roe,
                    "revenue_growth":  rev_growth,
                    "earnings_growth": _pct(info.get("earningsGrowth")),
                    "debt_equity":     de,
                    "debt_equity_unit": "ratio",
                    "net_margin":      net_margin,
                    "div_yield":       _pct(info.get("dividendYield")),
                    "market_cap_cr":   round(mcap / 1e7, 0) if mcap else None,
                    "sector":          info.get("sector", ""),
                    "industry":        info.get("industry", ""),
                    "fund_score":      _fund_score(roe, rev_growth, de, net_margin),
                    "52w_high":        w52_high,
                    "52w_low":         w52_low,
                    "near_high_pct":   near_high_pct,
                })
            except Exception as e:
                exc.append(str(e))

    t = threading.Thread(target=_fetch, daemon=True)
    t.start()
    t.join(timeout=_TIMEOUT)
    if t.is_alive():
        print(f"[Fundamentals] Timeout for {symbol_ns} — skipping")
    elif exc:
        print(f"[Fundamentals] Error for {symbol_ns}: {exc[0]}")

    _cache[symbol_ns] = (result, now)
    _enrich_screener(symbol_ns, result)
    return result


def _enrich_screener(symbol_ns: str, result: dict) -> None:
    """
    Merge Screener.in data (ROCE, pledging, FII/DII trend) into result in-place.
    Non-blocking — if screener fails, result is unchanged.
    Only runs when result already has some data (avoids wasted HTTP calls on empty yf results).
    """
    try:
        import screener_client
        sym = symbol_ns.replace(".NS", "").replace(".BO", "").upper()
        sc = screener_client.get(sym)
        if not sc:
            return

        # Merge Screener-specific fields (don't overwrite existing good values)
        for key in ["roce", "promoter_holding", "promoter_change",
                    "promoter_pledged", "fii_holding", "fii_change",
                    "dii_holding", "dii_change", "quality_score",
                    "red_flags", "green_flags", "screener_warnings"]:
            if key in sc and key not in result:
                result[key] = sc[key]

        # Screener PE/ROE may be more accurate (trailing vs yf's forward)
        if sc.get("roce") is not None:
            result["roce"] = sc["roce"]
        if sc.get("roe_screener") is not None and result.get("roe") is None:
            result["roe"] = sc["roe_screener"]

        # Recompute fund_score with ROCE if available
        if sc.get("roce") is not None:
            roce = sc["roce"]
            existing_score = result.get("fund_score", 50)
            if roce > 20:
                result["fund_score"] = min(100, existing_score + 8)
            elif roce > 15:
                result["fund_score"] = min(100, existing_score + 5)
            elif roce > 10:
                result["fund_score"] = min(100, existing_score + 2)
            elif roce < 5:
                result["fund_score"] = max(0, existing_score - 5)

        # Pledging penalty on fund_score
        pledged = sc.get("promoter_pledged", 0) or 0
        if pledged > 50:
            result["fund_score"] = max(0, result.get("fund_score", 50) - 20)
        elif pledged > 20:
            result["fund_score"] = max(0, result.get("fund_score", 50) - 10)

    except Exception:
        pass
