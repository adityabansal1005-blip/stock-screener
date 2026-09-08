NIFTY_50 = [
    "RELIANCE", "TCS", "HDFCBANK", "BHARTIARTL", "ICICIBANK",
    "INFY", "SBIN", "HINDUNILVR", "ITC", "LT",
    "KOTAKBANK", "AXISBANK", "BAJFINANCE", "MARUTI", "ASIANPAINT",
    "HCLTECH", "WIPRO", "ULTRACEMCO", "TITAN", "NTPC",
    "POWERGRID", "SUNPHARMA", "ONGC", "TATAPOWER", "TATASTEEL",
    "ADANIENT", "ADANIPORTS", "COALINDIA", "BAJAJFINSV", "TECHM",
    "NESTLEIND", "JSWSTEEL", "INDUSINDBK", "CIPLA", "DRREDDY",
    "EICHERMOT", "BPCL", "HEROMOTOCO", "BRITANNIA", "DIVISLAB",
    "GRASIM", "HINDALCO", "M&M", "APOLLOHOSP", "SBILIFE",
    "HDFCLIFE", "BAJAJ-AUTO", "TATACONSUM", "JIOFIN", "SHRIRAMFIN"
]

NIFTY_NEXT50 = [
    "ADANIGREEN", "AMBUJACEM", "AUBANK", "BEL", "BERGEPAINT",
    "BIOCON", "BOSCHLTD", "CANBK", "CHOLAFIN", "COFORGE",
    "COLPAL", "CONCOR", "DABUR", "DLF", "FEDERALBNK",
    "GAIL", "GODREJCP", "GODREJPROP", "HAVELLS", "HDFCAMC",
    "HINDPETRO", "ICICIGI", "ICICIPRULI", "IDFCFIRSTB", "IGL",
    "INDHOTEL", "INDUSTOWER", "IRCTC", "LICHSGFIN", "LUPIN",
    "MFSL", "MPHASIS", "MRF", "MUTHOOTFIN", "NAUKRI",
    "OFSS", "PAGEIND", "PERSISTENT", "PETRONET", "PFC",
    "PIDILITIND", "PNB", "POLYCAB", "POONAWALLA", "RECLTD",
    "SAIL", "SIEMENS", "SRF", "TVSMOTOR", "ZOMATO"
]

NIFTY_MIDCAP = [
    "ABCAPITAL", "ABFRL", "ALKEM", "AUROPHARMA", "BANDHANBNK",
    "BANKBARODA", "DEEPAKNTR", "DIXON", "ESCORTS", "EXIDEIND",
    "GMRINFRA", "HFCL", "JINDALSTEL", "JUBLFOOD", "KAJARIACER",
    "L&TFH", "MANAPPURAM", "MARICO", "NBCC", "NH",
    "NMDC", "OBEROIRLTY", "PAGEIND", "PIIND", "SUPREMEIND",
    "SUZLON", "TORNTPHARM", "TRENT", "UBL", "UNIONBANK",
    "VEDL", "VOLTAS", "WHIRLPOOL", "ZYDUSLIFE", "SUNDARMFIN",
    "SUNTV", "SHREECEM", "DALBHARAT", "CROMPTON", "CUMMINSIND"
]


import time
import requests
import io
import pandas as pd

_nse_cache: dict = {"data": {}, "ts": {}}
_CACHE_TTL = 86400  # 24 hours

_NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "*/*",
    "Referer": "https://www.nseindia.com/",
}

def _fetch_nifty500() -> list[str]:
    """Download Nifty 500 constituent symbols from NSE indices site."""
    now = time.time()
    if _nse_cache["ts"].get("nifty500") and now - _nse_cache["ts"]["nifty500"] < _CACHE_TTL:
        return _nse_cache["data"].get("nifty500", [])
    try:
        url = "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"
        resp = requests.get(url, headers=_NSE_HEADERS, timeout=15)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        # Column is usually "Symbol" or "SYMBOL"
        col = next((c for c in df.columns if c.strip().upper() == "SYMBOL"), None)
        if col:
            syms = [s.strip().upper() for s in df[col].dropna().tolist() if s.strip()]
            _nse_cache["data"]["nifty500"] = syms
            _nse_cache["ts"]["nifty500"]   = now
            print(f"[NSE] Nifty 500 loaded — {len(syms)} symbols")
            return syms
    except Exception as e:
        print(f"[NSE] Nifty 500 fetch failed: {e} — falling back to local list")
    # Fallback to our hardcoded list
    return NIFTY_50 + NIFTY_NEXT50 + NIFTY_MIDCAP


def _fetch_all_nse() -> list[str]:
    """Download ALL NSE-listed equity symbols (~1800+ stocks)."""
    now = time.time()
    if _nse_cache["ts"].get("all_nse") and now - _nse_cache["ts"]["all_nse"] < _CACHE_TTL:
        return _nse_cache["data"].get("all_nse", [])
    try:
        # NSE publishes a complete equity list as CSV
        sess = requests.Session()
        sess.headers.update(_NSE_HEADERS)
        sess.get("https://www.nseindia.com", timeout=10)
        time.sleep(0.3)
        resp = sess.get(
            "https://archives.nseindia.com/content/equities/EQUITY_L.csv",
            timeout=20,
        )
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        col = next((c for c in df.columns if c.strip().upper() == "SYMBOL"), None)
        if col:
            syms = [s.strip().upper() for s in df[col].dropna().tolist() if s.strip()]
            # Filter out bonds, preference shares etc (keep EQ series only if column exists)
            if "SERIES" in df.columns:
                eq_mask = df["SERIES"].str.strip().str.upper() == "EQ"
                syms = [s.strip().upper() for s in df.loc[eq_mask, col].dropna().tolist() if s.strip()]
            _nse_cache["data"]["all_nse"] = syms
            _nse_cache["ts"]["all_nse"]   = now
            print(f"[NSE] All-NSE list loaded — {len(syms)} symbols")
            return syms
    except Exception as e:
        print(f"[NSE] All-NSE fetch failed: {e} — falling back to Nifty 500")
    return _fetch_nifty500()


def get_all_symbols(universe: str = "nifty50") -> list[str]:
    if universe == "nifty50":
        tickers = NIFTY_50
    elif universe == "niftynext50":
        tickers = NIFTY_NEXT50
    elif universe == "nifty100":
        tickers = NIFTY_50 + NIFTY_NEXT50
    elif universe == "midcap":
        tickers = NIFTY_MIDCAP
    elif universe == "nifty150":
        tickers = NIFTY_50 + NIFTY_NEXT50 + NIFTY_MIDCAP
    elif universe == "nifty500":
        tickers = _fetch_nifty500()
    elif universe == "all_nse":
        tickers = _fetch_all_nse()
    elif universe in ("bse", "all_india"):
        # BSE-only names are emitted as scrip codes with a .BO suffix so
        # groww_client._route() sends them to the BSE endpoint. They are
        # liquidity-validated in bse_universe.py — an unvalidated BSE list is
        # mostly debt and shells that cannot be traded at any size.
        try:
            import bse_universe
            bse = bse_universe.symbols()
        except Exception:
            bse = []
        if universe == "bse":
            return bse
        return [f"{s}.NS" for s in _dedupe(_fetch_all_nse())] + bse
    else:  # "all" — legacy fallback
        tickers = NIFTY_50 + NIFTY_NEXT50 + NIFTY_MIDCAP

    return [f"{s}.NS" for s in _dedupe(tickers)]


def _dedupe(tickers):
    """Deduplicate while preserving order."""
    seen, unique = set(), []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique


SYMBOL_NAMES = {s: s for s in NIFTY_50 + NIFTY_NEXT50 + NIFTY_MIDCAP}


# ── Watchlist (persisted to watchlist.json) ───────────────────────────────────
import json as _json
from pathlib import Path as _Path

_WATCHLIST_FILE = _Path(__file__).parent / "watchlist.json"


def load_watchlist() -> list[str]:
    try:
        if _WATCHLIST_FILE.exists():
            data = _json.loads(_WATCHLIST_FILE.read_text(encoding="utf-8"))
            return [s.upper().replace(".NS", "") for s in data if s]
    except Exception:
        pass
    return []


def save_watchlist(symbols: list[str]) -> None:
    _WATCHLIST_FILE.write_text(
        _json.dumps(symbols, indent=2), encoding="utf-8"
    )


def add_to_watchlist(symbol: str) -> list[str]:
    s = symbol.upper().replace(".NS", "").strip()
    syms = load_watchlist()
    if s and s not in syms:
        syms.append(s)
        save_watchlist(syms)
    return syms


def remove_from_watchlist(symbol: str) -> list[str]:
    s = symbol.upper().replace(".NS", "").strip()
    syms = [x for x in load_watchlist() if x != s]
    save_watchlist(syms)
    return syms
