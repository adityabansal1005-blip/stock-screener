"""
Sector Intelligence — identifies hot sectors from news, ranks companies
by technical + fundamental composite score, finds the "king" of each sector.

Sources:
  - Economic Times RSS + MoneyControl for news
  - yfinance for fundamentals (PE, ROE, revenue growth, etc.)
  - nselib for recent OHLCV (tech score proxy)
  - Screener.in for order book data (web scrape)

Usage:
  from sector_intel import get_sector_report
  report = get_sector_report(force=False)   # returns cached if <6h old
"""

import time
import threading
import json
import re
import datetime
import requests
import xml.etree.ElementTree as ET

_cache = {"data": None, "ts": 0, "loading": False}
_TTL = 6 * 3600  # 6 hours


# ── Sector Universe ──────────────────────────────────────────────────────────
SECTORS = {
    "Cables & Wires": {
        "theme": "Power infra capex + EV charging grid expansion",
        "tailwinds": ["INR 11L Cr power sector spend", "EV infra roll-out", "Solar transmission"],
        "stocks": ["POLYCAB", "KEI", "HAVELLS", "FINCABLES", "RRKABEL", "APARINDS"],
    },
    "Defence & Aerospace": {
        "theme": "Indigenisation push — 68% of defence budget for domestic procurement",
        "tailwinds": ["Record ₹6.2L Cr defence budget FY26", "Export ambition ($5B by 2025)", "HAL pipeline"],
        "stocks": ["HAL", "BDL", "BEML", "MIDHANI", "MTARTECH", "PARAS", "COCHINSHIP", "GRSE", "MAZDOCK"],
    },
    "Railways & Metro": {
        "theme": "Indian Railways ₹2.5L Cr capex + 50-city metro expansion",
        "tailwinds": ["Vande Bharat fleet order", "Station redevelopment", "Dedicated freight corridor"],
        "stocks": ["RVNL", "IRFC", "RAILTEL", "RITES", "IRCON", "TEXRAIL", "KPIL", "NBCC"],
    },
    "Optical Fibre & Telecom Infra": {
        "theme": "BharatNet 2.0 + 5G densification drives fibre demand",
        "tailwinds": ["BharatNet ₹1.4L Cr Phase 3", "5G small-cell roll-out", "Data-centre connectivity"],
        "stocks": ["STLTECH", "HFCL", "OPTIEMUS", "VIMTALABS", "TEJASNET", "TATACOMM"],
    },
    "Electronics Manufacturing (EMS)": {
        "theme": "China+1 policy — India as global electronics hub",
        "tailwinds": ["PLI scheme ₹76K Cr", "Apple supply-chain localisation", "Defence electronics"],
        "stocks": ["DIXON", "AMBER", "KAYNES", "SYRMA", "PGEL", "AVALON", "ELIN"],
    },
    "Solar & Renewable Energy": {
        "theme": "500 GW by 2030 — fastest growing energy sector globally",
        "tailwinds": ["PM-KUSUM", "Solar park tenders", "Green hydrogen policy"],
        "stocks": ["ADANIGREEN", "TATAPOWER", "SJVN", "NTPC", "INDIGRID", "BORORENEW", "WAAREEENER"],
    },
    "Capital Goods & Engineering": {
        "theme": "Domestic capex cycle led by infra + manufacturing revival",
        "tailwinds": ["PLI-driven factory orders", "Infra pipeline ₹143L Cr", "Export order books"],
        "stocks": ["LT", "BHEL", "SIEMENS", "ABB", "THERMAX", "CUMMINSIND", "KEC", "KPIL"],
    },
    "Speciality Chemicals": {
        "theme": "China supply-chain shift + domestic pharma/agri API demand",
        "tailwinds": ["CRAMS demand from global pharma", "Fluorochemicals scarcity", "EV battery materials"],
        "stocks": ["AARTIIND", "DEEPAKNTR", "NAVINFLUOR", "SRF", "GALAXYSURF", "TATACHEM", "VINATIORGA"],
    },
    "Bearings & Precision Components": {
        "theme": "EV + auto capex + import substitution (60% still imported)",
        "tailwinds": ["India bearing market 10% CAGR", "EV drivetrain demand", "Make-in-India push"],
        "stocks": ["SKFINDIA", "SCHAEFFLER", "NRBBEARING", "TIMKEN", "GRINDWELL", "SUPRAJIT"],
    },
    "Hospitals & Healthcare": {
        "theme": "Premiumisation of healthcare + medical tourism boom",
        "tailwinds": ["Ayushman Bharat expansion", "NABH accreditation drive", "Insurance penetration"],
        "stocks": ["APOLLOHOSP", "MAXHEALTH", "FORTIS", "NH", "KIMS", "YATHARTH"],
    },
    "IT & AI Services": {
        "theme": "Gen-AI adoption + cloud migration cycle — next upcycle",
        "tailwinds": ["BFSI modernisation", "Global capability centres", "AI/ML deal flow"],
        "stocks": ["TCS", "INFY", "HCLTECH", "WIPRO", "TECHM", "PERSISTENT", "MPHASIS", "COFORGE"],
    },
    "Banks": {
        "theme": "Credit growth + improving asset quality after a decade of clean-up",
        "tailwinds": ["Credit growth outpacing deposits", "GNPA at multi-year lows",
                      "Retail lending expansion", "RBI rate cycle turning"],
        "stocks": ["HDFCBANK", "ICICIBANK", "KOTAKBANK", "AXISBANK", "SBIN",
                   "INDUSINDBK", "FEDERALBNK", "IDFCFIRSTB", "BANKBARODA",
                   "PNB", "CANBK", "AUBANK"],
    },
    "NBFC & Financial Services": {
        "theme": "Formalisation of credit — NBFCs reaching where banks do not",
        "tailwinds": ["Gold loan demand", "Vehicle & MSME financing",
                      "Housing finance growth", "Co-lending with banks"],
        "stocks": ["BAJFINANCE", "BAJAJFINSV", "CHOLAFIN", "SHRIRAMFIN",
                   "MUTHOOTFIN", "MANAPPURAM", "LTF", "PFC", "RECLTD",
                   "SBICARD", "LICHSGFIN", "M&MFIN", "POONAWALLA", "SUNDARMFIN"],
    },
    "Logistics & Warehousing": {
        "theme": "E-commerce growth + GST efficiency + PM Gati Shakti NMP",
        "tailwinds": ["Multimodal logistics infra", "Cold chain expansion", "Express parcel market"],
        "stocks": ["DELHIVERY", "BLUEDART", "MAHLOG", "GESHIP", "TCI", "ALLCARGO"],
    },
}


# ── News scraping ────────────────────────────────────────────────────────────
_ET_RSS = "https://economictimes.indiatimes.com/markets/stocks/rss.cms"
_MC_RSS = "https://www.moneycontrol.com/rss/marketreports.xml"

def _fetch_news(keywords: list[str], max_articles: int = 5) -> list[dict]:
    """Pull headline + summary from ET + MC RSS, filter by keywords."""
    results = []
    feeds = [_ET_RSS, _MC_RSS]

    for url in feeds:
        try:
            resp = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
            if resp.status_code != 200:
                continue
            root = ET.fromstring(resp.content)
            for item in root.iter("item"):
                title = (item.findtext("title") or "").strip()
                desc  = (item.findtext("description") or "").strip()
                link  = (item.findtext("link") or "").strip()
                pub   = (item.findtext("pubDate") or "").strip()
                text  = (title + " " + desc).lower()

                if any(kw.lower() in text for kw in keywords):
                    results.append({
                        "title": title,
                        "desc":  desc[:200] if desc else "",
                        "link":  link,
                        "pub":   pub,
                        "source": "ET" if "economictimes" in url else "MC",
                    })
                    if len(results) >= max_articles:
                        return results
        except Exception:
            continue

    return results


def _news_relevance_score(sector_name: str, tailwinds: list[str]) -> dict:
    """Return list of matching headlines + a relevance count."""
    kw_base  = sector_name.lower().split()
    kw_extra = []
    for t in tailwinds:
        kw_extra += [w for w in t.lower().split() if len(w) > 4]

    keywords = kw_base + kw_extra[:6]
    articles = _fetch_news(keywords)
    return {"count": len(articles), "articles": articles}


# ── Fundamental data ─────────────────────────────────────────────────────────

def _get_fundamentals_yf(symbol: str) -> dict:
    """Fetch key fundamentals from yfinance."""
    try:
        import yfinance as yf
        t = yf.Ticker(f"{symbol}.NS")
        info = t.info or {}
        return {
            "pe":              info.get("trailingPE"),
            "pb":              info.get("priceToBook"),
            "roe":             round(info.get("returnOnEquity", 0) * 100, 1) if info.get("returnOnEquity") else None,
            "revenue_growth":  round(info.get("revenueGrowth", 0) * 100, 1) if info.get("revenueGrowth") else None,
            "earnings_growth": round(info.get("earningsGrowth", 0) * 100, 1) if info.get("earningsGrowth") else None,
            "debt_equity":     info.get("debtToEquity"),
            "net_margin":      round(info.get("profitMargins", 0) * 100, 1) if info.get("profitMargins") else None,
            "market_cap":      info.get("marketCap"),
            "sector":          info.get("sector"),
            "short_name":      info.get("shortName", symbol),
        }
    except Exception:
        return {}


_FIN_SECTORS = ("financial", "bank", "insurance", "capital markets", "credit")


def _is_financial(f: dict) -> bool:
    """Banks/NBFCs carry leverage as their business model, not as a risk flag."""
    s = (f.get("sector") or "").lower()
    i = (f.get("industry") or "").lower()
    return any(k in s or k in i for k in _FIN_SECTORS)


def _fund_score(f: dict) -> int:
    """0-100 composite score from fundamentals."""
    score = 50  # neutral start
    financial = _is_financial(f)

    roe = f.get("roe")
    if roe is not None:
        if roe > 20: score += 15
        elif roe > 12: score += 8
        elif roe < 5: score -= 10

    rev_g = f.get("revenue_growth")
    if rev_g is not None:
        if rev_g > 20: score += 15
        elif rev_g > 10: score += 8
        elif rev_g < 0: score -= 12

    earn_g = f.get("earnings_growth")
    if earn_g is not None:
        if earn_g > 25: score += 10
        elif earn_g > 10: score += 5
        elif earn_g < -10: score -= 8

    # Debt/equity is meaningless for lenders — deposits and borrowings ARE the
    # raw material. A bank routinely runs 6-10x leverage by design, so the
    # blanket ">2.0 = -10" rule would penalise every bank and NBFC in the
    # universe for existing. Skip the leverage test for financials.
    de = f.get("debt_equity")
    if de is not None and not financial:
        if de < 0.3: score += 8
        elif de > 2.0: score -= 10

    # Lenders report far higher net margins than manufacturers (no COGS), so
    # the same thresholds would hand them free points. Use a stricter bar.
    nm = f.get("net_margin")
    if nm is not None:
        if financial:
            if nm > 25: score += 7
            elif nm < 10: score -= 8
        else:
            if nm > 15: score += 7
            elif nm < 3: score -= 8

    return max(0, min(100, score))


# ── Order book / Screener.in ─────────────────────────────────────────────────

def _get_order_book(symbol: str) -> dict:
    """
    Scrape order book metrics from Screener.in.
    Returns: {'order_book': str, 'revenue_ttm': str, 'pat': str}
    """
    try:
        url = f"https://www.screener.in/company/{symbol}/consolidated/"
        resp = requests.get(url, timeout=10, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        if resp.status_code != 200:
            url2 = f"https://www.screener.in/company/{symbol}/"
            resp = requests.get(url2, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code != 200:
            return {}

        html = resp.text
        result = {}

        # Order book — often in a "Key Metrics" or "Concall" table
        ob_match = re.search(r'[Oo]rder [Bb]ook.*?([\d,]+\.?\d*\s*(?:Cr|cr|lakh|Lakh))', html)
        if ob_match:
            result["order_book"] = ob_match.group(1).strip()

        # Revenue TTM
        rev_match = re.search(r'Sales.*?(\d[\d,]+)', html[:5000])
        if rev_match:
            result["revenue_ttm"] = rev_match.group(1).replace(",", "")

        # PAT
        pat_match = re.search(r'Net Profit.*?(\d[\d,]+)', html[:5000])
        if pat_match:
            result["pat"] = pat_match.group(1).replace(",", "")

        return result
    except Exception:
        return {}


# ── Technical score (lightweight via nselib) ─────────────────────────────────

def _get_tech_score(symbol: str) -> int | None:
    """
    Quick 3-month RSI + price vs 50-day EMA score (0-100).
    Uses nselib for OHLCV, returns None on failure.
    """
    try:
        from nselib import capital_market
        import pandas as pd
        import numpy as np

        end   = datetime.date.today()
        start = end - datetime.timedelta(days=120)

        raw = capital_market.price_volume_and_deliverable_position_data(
            symbol=symbol,
            from_date=start.strftime("%d-%m-%Y"),
            to_date=end.strftime("%d-%m-%Y"),
        )
        if raw is None or (hasattr(raw, '__len__') and len(raw) < 20):
            return None

        df = raw.copy()
        # Strip BOM and parse columns
        df.columns = [c.strip().lstrip("﻿").strip('"') for c in df.columns]
        close_col = next((c for c in df.columns if "Close" in c or "close" in c), None)
        if close_col is None:
            return None

        df[close_col] = pd.to_numeric(df[close_col].astype(str).str.replace(",", ""), errors="coerce")
        df = df.dropna(subset=[close_col])
        closes = df[close_col].values.astype(float)

        if len(closes) < 20:
            return None

        # RSI(14)
        delta = np.diff(closes)
        gain  = np.where(delta > 0, delta, 0)
        loss  = np.where(delta < 0, -delta, 0)
        avg_g = np.mean(gain[-14:]) if len(gain) >= 14 else np.mean(gain)
        avg_l = np.mean(loss[-14:]) if len(loss) >= 14 else np.mean(loss)
        rsi   = 100 - 100 / (1 + avg_g / avg_l) if avg_l > 0 else 100

        # Price vs EMA50
        ema50 = float(pd.Series(closes).ewm(span=50, adjust=False).mean().iloc[-1])
        price = closes[-1]
        above_ema = price > ema50

        # Simple score
        score = 50
        if 40 < rsi < 70: score += 20    # healthy momentum
        elif rsi >= 70:   score += 10    # overbought — slight bonus but not ideal
        elif rsi < 30:    score -= 10    # oversold / weakness
        if above_ema:     score += 20
        else:             score -= 20

        # 1M return
        ret_1m = (closes[-1] / closes[max(0, len(closes)-22)] - 1) * 100 if len(closes) >= 22 else 0
        if ret_1m > 5:    score += 10
        elif ret_1m < -5: score -= 10

        return max(0, min(100, int(score)))
    except Exception:
        return None


# ── AI sector narrative ───────────────────────────────────────────────────────

def _ai_sector_summary(sector_name: str, theme: str, tailwinds: list,
                        news_articles: list, king: dict) -> str:
    """Call Claude to write a crisp sector intelligence brief."""
    try:
        import config
        import anthropic

        key = getattr(config, "ANTHROPIC_API_KEY", "")
        if not key or key == "YOUR_ANTHROPIC_API_KEY":
            return ""

        client = anthropic.Anthropic(api_key=key)
        news_text = "\n".join(f"  • {a['title']}" for a in news_articles[:3]) or "  • No recent news found"
        king_text = (
            f"{king['symbol']} — composite score {king['composite']}/100, "
            f"fund score {king.get('fund_score','?')}, tech score {king.get('tech_score','?')}"
            if king else "No king identified"
        )

        prompt = f"""You are a senior Indian equity sector analyst. Write a crisp 3-sentence sector brief for BEGINNER investors.

Sector: {sector_name}
Theme: {theme}
Key Tailwinds: {', '.join(tailwinds)}
Recent News Headlines:
{news_text}
Top Ranked Company: {king_text}

Rules:
- Plain English, no jargon
- Sentence 1: What is driving this sector right now?
- Sentence 2: Why is it worth watching for the next 6-12 months?
- Sentence 3: What is the main risk that could derail it?
- Output ONLY the 3-sentence paragraph. No headers, no bullet points."""

        with client.messages.stream(
            model="claude-sonnet-4-6",
            max_tokens=200,
            timeout=20,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            msg = stream.get_final_message()

        return msg.content[0].text.strip()
    except Exception:
        return ""


# ── Per-stock scoring ─────────────────────────────────────────────────────────

def _score_stock(symbol: str) -> dict:
    """Full composite score for one stock."""
    fund = _get_fundamentals_yf(symbol)
    fs   = _fund_score(fund)
    ts   = _get_tech_score(symbol) or 50
    ob   = _get_order_book(symbol)

    # Growth bonus — extra weight if strong earnings + revenue growth
    rev_g  = fund.get("revenue_growth") or 0
    earn_g = fund.get("earnings_growth") or 0
    growth_bonus = min(30, max(0, (rev_g * 0.5 + earn_g * 0.5)))

    composite = int(ts * 0.40 + fs * 0.35 + growth_bonus * 0.25)
    composite = max(0, min(100, composite))

    return {
        "symbol":       symbol,
        "composite":    composite,
        "tech_score":   ts,
        "fund_score":   fs,
        "growth_bonus": round(growth_bonus, 1),
        "pe":           fund.get("pe"),
        "pb":           fund.get("pb"),
        "roe":          fund.get("roe"),
        "revenue_growth":  fund.get("revenue_growth"),
        "earnings_growth": fund.get("earnings_growth"),
        "debt_equity":  fund.get("debt_equity"),
        "net_margin":   fund.get("net_margin"),
        "market_cap":   fund.get("market_cap"),
        "short_name":   fund.get("short_name", symbol),
        "order_book":   ob.get("order_book"),
    }


# ── Sector scan ───────────────────────────────────────────────────────────────

def scan_sector(sector_name: str, sector_data: dict) -> dict:
    """Score all stocks in a sector and return ranked list + king + news."""
    stocks    = sector_data["stocks"]
    tailwinds = sector_data["tailwinds"]
    theme     = sector_data["theme"]

    print(f"[SectorIntel] Scanning {sector_name} ({len(stocks)} stocks)...")

    # Score stocks in parallel
    scored = []
    lock   = threading.Lock()

    def _worker(sym):
        try:
            result = _score_stock(sym)
            with lock:
                scored.append(result)
        except Exception as e:
            print(f"[SectorIntel] {sym} failed: {e}")

    threads = [threading.Thread(target=_worker, args=(s,), daemon=True) for s in stocks]
    for t in threads: t.start()
    for t in threads: t.join(timeout=30)

    scored.sort(key=lambda x: x["composite"], reverse=True)
    king = scored[0] if scored else None

    # News
    news_data = _news_relevance_score(sector_name, tailwinds)

    # AI brief
    ai_brief = _ai_sector_summary(sector_name, theme, tailwinds,
                                   news_data["articles"], king)

    return {
        "sector":      sector_name,
        "theme":       theme,
        "tailwinds":   tailwinds,
        "king":        king,
        "stocks":      scored,
        "news_count":  news_data["count"],
        "news":        news_data["articles"],
        "ai_brief":    ai_brief,
        "scanned_at":  datetime.datetime.now().strftime("%d %b %Y %I:%M %p"),
    }


# ── Public API ────────────────────────────────────────────────────────────────

def get_sector_report(force: bool = False) -> dict:
    """
    Returns cached sector report if < TTL, otherwise rebuilds.
    Thread-safe. Returns immediately with loading=True if scan is in progress.
    """
    now = time.time()
    if not force and _cache["data"] and (now - _cache["ts"]) < _TTL:
        return {"status": "ok", "data": _cache["data"], "cached": True}

    if _cache["loading"]:
        return {"status": "loading", "data": _cache["data"] or [], "cached": False}

    _cache["loading"] = True

    def _run():
        try:
            results = []
            for sector_name, sector_data in SECTORS.items():
                try:
                    r = scan_sector(sector_name, sector_data)
                    results.append(r)
                except Exception as e:
                    print(f"[SectorIntel] Sector {sector_name} failed: {e}")

            # Sort sectors by news relevance + king composite
            results.sort(
                key=lambda x: (x["news_count"] * 10 + (x["king"]["composite"] if x["king"] else 0)),
                reverse=True,
            )

            _cache["data"] = results
            _cache["ts"]   = time.time()
            print(f"[SectorIntel] Done. {len(results)} sectors scanned.")
        except Exception as e:
            print(f"[SectorIntel] Full scan failed: {e}")
        finally:
            _cache["loading"] = False

    threading.Thread(target=_run, daemon=True).start()
    return {"status": "scanning", "data": _cache["data"] or [], "cached": False}


def get_sector_status() -> dict:
    return {
        "loading":    _cache["loading"],
        "sectors":    len(_cache["data"]) if _cache["data"] else 0,
        "last_scan":  datetime.datetime.fromtimestamp(_cache["ts"]).strftime("%d %b %Y %I:%M %p") if _cache["ts"] else None,
    }
