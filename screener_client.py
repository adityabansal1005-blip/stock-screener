"""
Screener.in client — scrapes fundamental + shareholding data.
Provides: promoter pledging %, FII/DII holding trend, ROCE, revenue growth,
          debt levels, Piotroski-like quality score, insider warnings.

Cache TTL: 24h (fundamentals don't change intraday).
"""
import time
import threading
import logging
import requests
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)

_cache: dict = {}
_CACHE_TTL = 86400   # 24h
_lock = threading.Lock()
_sem  = threading.Semaphore(3)   # max 3 concurrent screener requests

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.screener.in/",
}


def _safe_float(s: str) -> float | None:
    if not s:
        return None
    try:
        return float(s.replace(",", "").replace("%", "").replace("₹", "").strip())
    except Exception:
        return None


def _fetch_screener(symbol: str) -> dict:
    """Fetch and parse Screener.in consolidated page for a symbol."""
    with _sem:
        try:
            url = f"https://www.screener.in/company/{symbol}/consolidated/"
            r = requests.get(url, headers=_HEADERS, timeout=20)
            if r.status_code == 404:
                # Try standalone
                url = f"https://www.screener.in/company/{symbol}/"
                r = requests.get(url, headers=_HEADERS, timeout=20)
            if r.status_code != 200:
                return {}

            soup = BeautifulSoup(r.text, "html.parser")
            result: dict = {"symbol": symbol}

            # ── Top ratios ───────────────────────────────────────
            for li in soup.select("#top-ratios li"):
                name_el = li.select_one(".name")
                val_el  = li.select_one(".number, .nowrap")
                if not name_el or not val_el:
                    continue
                name = name_el.text.strip().lower()
                val  = val_el.text.strip()
                if "p/e" in name or "stock p/e" in name:
                    result["pe"] = _safe_float(val)
                elif "book value" in name:
                    result["book_value"] = _safe_float(val)
                elif "roce" in name:
                    result["roce"] = _safe_float(val)
                elif "roe" in name:
                    result["roe_screener"] = _safe_float(val)
                elif "debt" in name:
                    result["debt_to_equity"] = _safe_float(val)
                elif "dividend yield" in name:
                    result["div_yield"] = _safe_float(val)
                elif "market cap" in name:
                    result["market_cap_text"] = val

            # ── Shareholding trend ────────────────────────────────
            sh_tables = soup.select("#shareholding table")
            if sh_tables:
                rows = sh_tables[0].select("tr")
                headers = [th.text.strip() for th in rows[0].select("td,th")] if rows else []
                # Get latest quarter (last column)
                for row in rows[1:]:
                    cols = [c.text.strip() for c in row.select("td")]
                    if not cols:
                        continue
                    label = cols[0].lower().replace("\xa0", " ").replace("+", "").strip()
                    # Latest = last non-empty col
                    vals = [_safe_float(c) for c in cols[1:] if c]
                    if not vals:
                        continue
                    latest = vals[-1]
                    prev   = vals[-2] if len(vals) >= 2 else latest

                    if "promoter" in label:
                        result["promoter_holding"] = latest
                        result["promoter_change"]  = round(latest - prev, 2) if prev else None
                    elif "fii" in label or "fpi" in label:
                        result["fii_holding"]  = latest
                        result["fii_change"]   = round(latest - prev, 2) if prev else None
                    elif "dii" in label:
                        result["dii_holding"]  = latest
                        result["dii_change"]   = round(latest - prev, 2) if prev else None

            # ── Warnings / Insights ──────────────────────────────
            warnings = []
            for li in soup.select(".sub-list li, #analysis li"):
                text = li.text.strip()
                if text and len(text) > 10:
                    warnings.append(text[:200])
            if warnings:
                result["screener_warnings"] = warnings[:8]

            # ── Pledged shares (from shareholding detail) ────────
            # Look for pledge rows in the expanded table
            for row in soup.select("#shareholding tr"):
                text = row.text.lower()
                if "pledge" in text:
                    cols = [c.text.strip() for c in row.select("td")]
                    vals = [_safe_float(c) for c in cols[1:] if c]
                    if vals:
                        result["promoter_pledged"] = vals[-1]

            # ── Quick quality signals ────────────────────────────
            roce  = result.get("roce")
            roe   = result.get("roe_screener")
            de    = result.get("debt_to_equity")
            fii_c = result.get("fii_change")
            dii_c = result.get("dii_change")
            pro_c = result.get("promoter_change")
            pledg = result.get("promoter_pledged", 0) or 0

            # Quality score (0-10)
            q = 0
            if roce and roce > 15:  q += 2
            elif roce and roce > 10: q += 1
            if roe and roe > 15:    q += 2
            elif roe and roe > 10:  q += 1
            if de is not None and de < 0.5:  q += 2
            elif de is not None and de < 1:  q += 1
            if fii_c and fii_c > 0: q += 1
            if dii_c and dii_c > 0: q += 1
            if pledg < 10:          q += 1
            result["quality_score"] = q

            # Red flags
            red_flags = []
            if pledg > 50:    red_flags.append(f"HIGH PLEDGE: {pledg}%")
            if pledg > 20:    red_flags.append(f"Pledge: {pledg}%")
            if pro_c and pro_c < -2: red_flags.append(f"Promoter sold {abs(pro_c):.1f}%")
            if fii_c and fii_c < -3: red_flags.append(f"FII sold {abs(fii_c):.1f}%")
            if de and de > 3: red_flags.append(f"High D/E: {de}")
            if result.get("screener_warnings"):
                for w in result["screener_warnings"]:
                    if any(x in w.lower() for x in ["decrease", "low roe", "high debt", "negative"]):
                        red_flags.append(w[:60])
            result["red_flags"] = red_flags[:5]

            # Green flags
            green_flags = []
            if fii_c and fii_c > 1:  green_flags.append(f"FII buying: +{fii_c:.1f}%")
            if dii_c and dii_c > 1:  green_flags.append(f"DII buying: +{dii_c:.1f}%")
            if pro_c and pro_c > 0.5: green_flags.append(f"Promoter bought: +{pro_c:.1f}%")
            if roce and roce > 20:   green_flags.append(f"ROCE {roce}%")
            if pledg == 0:           green_flags.append("Zero pledge")
            result["green_flags"] = green_flags[:5]

            return result

        except Exception as e:
            log.debug(f"[Screener] {symbol}: {e}")
            return {}


def get(symbol: str) -> dict:
    """Get screener data for a symbol (cached 24h)."""
    symbol = symbol.upper().replace(".NS", "").replace(".BO", "")
    now = time.time()
    with _lock:
        entry = _cache.get(symbol)
        if entry and now - entry["ts"] < _CACHE_TTL:
            return entry["data"]

    data = _fetch_screener(symbol)

    with _lock:
        _cache[symbol] = {"data": data, "ts": now}
    return data


def get_red_flags(symbol: str) -> list[str]:
    """Quick check — return list of red flags for a symbol."""
    return get(symbol).get("red_flags", [])
