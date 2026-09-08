"""
DhanHQ v2 data client — daily + intraday OHLCV for NSE equities.

Why this exists: Groww returns 403 for market data (key not entitled) and the
TrueData trial has lapsed, leaving nselib as the only working source at ~4s per
stock. Dhan allows 5 requests/second, which takes a Nifty 500 scan from ~29
minutes to under 2, and is the only remaining source for intraday candles.

Two things about this API differ from every other source in the project:

  1. It addresses instruments by numeric `securityId`, not ticker. The mapping
     comes from a 24 MB scrip master CSV, cached to disk for 24h.
  2. Responses are COLUMNAR — {"open": [...], "high": [...], ...} — not a list
     of row objects.

Credentials (optional; the module stays inert without them):
    DHAN_CLIENT_ID, DHAN_ACCESS_TOKEN

Public API:
    is_configured()        -> bool
    check_entitlement()    -> dict    one-shot diagnostic
    get_daily_bars(sym, days=550)          -> DataFrame | None
    get_intraday_candles(sym, interval_minutes=15, days=5) -> DataFrame | None
"""

from __future__ import annotations

import csv
import datetime
import io
import os
import pickle
import threading
import time
from pathlib import Path

import pandas as pd
import requests

import config

# ── Endpoints ────────────────────────────────────────────────────────────────
_BASE          = "https://api.dhan.co/v2"
_URL_DAILY     = f"{_BASE}/charts/historical"
_URL_INTRADAY  = f"{_BASE}/charts/intraday"
_URL_SCRIP     = "https://images.dhan.co/api-data/api-scrip-master.csv"

# ── Request constants (per DhanHQ v2 spec) ───────────────────────────────────
_EXCH_SEGMENT   = "NSE_EQ"
_INSTRUMENT     = "EQUITY"
VALID_INTERVALS = (1, 5, 15, 25, 60)
_INTRADAY_MAX_DAYS = 90          # API refuses wider windows in one call
_HTTP_TIMEOUT      = 20

# ── Scrip master cache ───────────────────────────────────────────────────────
_SCRIP_CACHE = Path(__file__).parent / "dhan_scrip_cache.pkl"
_SCRIP_TTL   = 24 * 3600

_symbol_map: dict[str, str] | None = None
_map_lock  = threading.Lock()

# ── Rate limiter — Dhan allows 5 data requests/second ────────────────────────
_RATE_PER_SEC = 5
_rate_lock    = threading.Lock()
_call_times: list[float] = []

# ── Entitlement breaker (mirrors groww_client) ───────────────────────────────
_data_forbidden = False
_forbidden_lock = threading.Lock()


# ─────────────────────────────────────────────────────────────────────────────
# Credentials
# ─────────────────────────────────────────────────────────────────────────────

def _client_id() -> str:
    return (getattr(config, "DHAN_CLIENT_ID", "") or "").strip()


def _access_token() -> str:
    return (getattr(config, "DHAN_ACCESS_TOKEN", "") or "").strip()


def is_configured() -> bool:
    """True when both credentials are present and not placeholders."""
    cid, tok = _client_id(), _access_token()
    if not cid or not tok:
        return False
    return not (cid.startswith("YOUR_") or tok.startswith("YOUR_"))


def _headers() -> dict:
    return {
        "Content-Type": "application/json",
        "Accept":       "application/json",
        "access-token": _access_token(),
        "client-id":    _client_id(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Entitlement breaker
# ─────────────────────────────────────────────────────────────────────────────

def _is_forbidden(status: int, body: str) -> bool:
    if status in (401, 403):
        return True
    b = (body or "").lower()
    return "forbidden" in b or "not subscribed" in b or "unauthor" in b


def _trip_forbidden(detail: str) -> None:
    """Latch off after a permissions failure — retrying every symbol can't fix it."""
    global _data_forbidden
    with _forbidden_lock:
        if _data_forbidden:
            return
        _data_forbidden = True
    print(
        f"[Dhan] Data request refused ({detail}). The token is valid but not "
        f"entitled to market data — activate the DhanHQ Data API subscription. "
        f"Skipping further Dhan calls this session."
    )


def data_forbidden() -> bool:
    return _data_forbidden


def reset_breaker() -> None:
    """Clear the latch, e.g. after refreshing the token."""
    global _data_forbidden
    with _forbidden_lock:
        _data_forbidden = False


# ─────────────────────────────────────────────────────────────────────────────
# Rate limiting
# ─────────────────────────────────────────────────────────────────────────────

def _throttle() -> None:
    """Block until issuing a request stays within 5/second, across all threads."""
    while True:
        with _rate_lock:
            now = time.time()
            # Drop timestamps older than the 1-second window
            while _call_times and now - _call_times[0] >= 1.0:
                _call_times.pop(0)
            if len(_call_times) < _RATE_PER_SEC:
                _call_times.append(now)
                return
            sleep_for = 1.0 - (now - _call_times[0])
        time.sleep(max(sleep_for, 0.01))


# ─────────────────────────────────────────────────────────────────────────────
# Scrip master → {SYMBOL: securityId}
# ─────────────────────────────────────────────────────────────────────────────

def _load_cached_map() -> dict[str, str] | None:
    try:
        if not _SCRIP_CACHE.exists():
            return None
        if time.time() - _SCRIP_CACHE.stat().st_mtime > _SCRIP_TTL:
            return None
        with open(_SCRIP_CACHE, "rb") as fh:
            data = pickle.load(fh)
        return data if isinstance(data, dict) and data else None
    except Exception:
        return None


def _download_map() -> dict[str, str]:
    """
    Pull the scrip master and keep only NSE cash-segment equities.

    Ground-truth columns (verified against the live file — note the published
    docs describe these incorrectly):
        SEM_SMST_SECURITY_ID   numeric id used by the charts API
        SEM_TRADING_SYMBOL     NSE ticker, e.g. RELIANCE
        SEM_EXM_EXCH_ID        'NSE'
        SEM_SEGMENT            'E' for cash
        SEM_INSTRUMENT_NAME    'EQUITY'
    """
    resp = requests.get(_URL_SCRIP, timeout=120)
    resp.raise_for_status()

    mapping: dict[str, str] = {}
    for row in csv.DictReader(io.StringIO(resp.text)):
        if (row.get("SEM_EXM_EXCH_ID") == "NSE"
                and row.get("SEM_SEGMENT") == "E"
                and row.get("SEM_INSTRUMENT_NAME") == "EQUITY"):
            sym = (row.get("SEM_TRADING_SYMBOL") or "").strip().upper()
            sid = (row.get("SEM_SMST_SECURITY_ID") or "").strip()
            if sym and sid:
                mapping[sym] = sid
    return mapping


def get_symbol_map(force: bool = False) -> dict[str, str]:
    """Cached {TICKER: securityId} for NSE equities. Downloads at most once daily."""
    global _symbol_map
    with _map_lock:
        if _symbol_map and not force:
            return _symbol_map

        if not force:
            cached = _load_cached_map()
            if cached:
                _symbol_map = cached
                return _symbol_map

        try:
            print("[Dhan] Downloading scrip master (~24 MB, first run only)...")
            mapping = _download_map()
            if not mapping:
                raise ValueError("scrip master produced no NSE equity rows")
            try:
                with open(_SCRIP_CACHE, "wb") as fh:
                    pickle.dump(mapping, fh)
            except Exception:
                pass    # cache write is best-effort
            _symbol_map = mapping
            print(f"[Dhan] Mapped {len(mapping):,} NSE equity instruments.")
        except Exception as e:
            print(f"[Dhan] Scrip master download failed: {e}")
            _symbol_map = _load_cached_map() or {}   # stale beats empty

        return _symbol_map


def get_security_id(symbol: str) -> str | None:
    sym = symbol.replace(".NS", "").replace(".BO", "").strip().upper()
    return get_symbol_map().get(sym)


# ─────────────────────────────────────────────────────────────────────────────
# Response parsing
# ─────────────────────────────────────────────────────────────────────────────

def _parse_columnar(payload: dict, intraday: bool) -> pd.DataFrame | None:
    """
    Convert Dhan's columnar response into an OHLCV frame matching the shape the
    rest of the project expects: Open/High/Low/Close/Volume on a DatetimeIndex.

    Timestamps arrive as Unix epoch seconds and are rendered in IST.
    """
    if not isinstance(payload, dict):
        return None

    ts = payload.get("timestamp")
    if not ts:
        return None

    try:
        df = pd.DataFrame({
            "Open":   payload.get("open",   []),
            "High":   payload.get("high",   []),
            "Low":    payload.get("low",    []),
            "Close":  payload.get("close",  []),
            "Volume": payload.get("volume", []),
        })
        if df.empty or len(df) != len(ts):
            return None

        idx = pd.to_datetime(pd.Series(ts), unit="s", utc=True)
        idx = idx.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
        if not intraday:
            idx = idx.dt.normalize()

        df.index = pd.DatetimeIndex(idx, name="Date")
        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.dropna(subset=["Close"]).sort_index()
        df = df[~df.index.duplicated(keep="last")]
        return df if not df.empty else None
    except Exception as e:
        print(f"[Dhan] Response parse failed: {e}")
        return None


def _post(url: str, body: dict) -> dict | None:
    """One rate-limited POST. Trips the breaker on a permissions failure."""
    if _data_forbidden or not is_configured():
        return None

    _throttle()
    try:
        r = requests.post(url, json=body, headers=_headers(), timeout=_HTTP_TIMEOUT)
    except Exception as e:
        print(f"[Dhan] Request error: {e}")
        return None

    if _is_forbidden(r.status_code, r.text):
        _trip_forbidden(f"HTTP {r.status_code}")
        return None

    if r.status_code != 200:
        print(f"[Dhan] HTTP {r.status_code}: {r.text[:180]}")
        return None

    try:
        return r.json()
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Public data calls
# ─────────────────────────────────────────────────────────────────────────────

def get_daily_bars(symbol: str, days: int = 550) -> pd.DataFrame | None:
    """Daily OHLCV for the trailing `days` calendar days. None if unavailable."""
    if _data_forbidden or not is_configured():
        return None            # checked before mapping — avoids a pointless 24 MB fetch
    sid = get_security_id(symbol)
    if not sid:
        return None

    today = datetime.date.today()
    body = {
        "securityId":      sid,
        "exchangeSegment": _EXCH_SEGMENT,
        "instrument":      _INSTRUMENT,
        "fromDate":        (today - datetime.timedelta(days=days)).strftime("%Y-%m-%d"),
        "toDate":          today.strftime("%Y-%m-%d"),
    }
    return _parse_columnar(_post(_URL_DAILY, body), intraday=False)


def get_intraday_candles(symbol: str,
                         interval_minutes: int = 15,
                         days: int = 5) -> pd.DataFrame | None:
    """
    Intraday candles for the trailing `days`. Valid intervals: 1, 5, 15, 25, 60.
    Windows wider than 90 days are split across calls and concatenated.
    """
    if interval_minutes not in VALID_INTERVALS:
        raise ValueError(
            f"interval_minutes must be one of {VALID_INTERVALS}, got {interval_minutes}"
        )

    if _data_forbidden or not is_configured():
        return None            # checked before mapping — avoids a pointless 24 MB fetch
    sid = get_security_id(symbol)
    if not sid:
        return None

    end   = datetime.datetime.now()
    start = end - datetime.timedelta(days=days)

    frames = []
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + datetime.timedelta(days=_INTRADAY_MAX_DAYS), end)
        body = {
            "securityId":      sid,
            "exchangeSegment": _EXCH_SEGMENT,
            "instrument":      _INSTRUMENT,
            "interval":        interval_minutes,
            "fromDate":        cursor.strftime("%Y-%m-%d %H:%M:%S"),
            "toDate":          chunk_end.strftime("%Y-%m-%d %H:%M:%S"),
        }
        part = _parse_columnar(_post(_URL_INTRADAY, body), intraday=True)
        if part is not None:
            frames.append(part)
        if _data_forbidden:
            break
        cursor = chunk_end

    if not frames:
        return None

    out = pd.concat(frames).sort_index()
    return out[~out.index.duplicated(keep="last")]


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def check_entitlement(symbol: str = "RELIANCE") -> dict:
    """
    One-shot check of credentials, symbol mapping and data entitlement.
    Safe to run repeatedly; never prints credential values.
    """
    out = {
        "configured":    is_configured(),
        "client_id_set": bool(_client_id()),
        "token_set":     bool(_access_token()),
        "scrip_master":  0,
        "security_id":   None,
        "daily_ok":      False,
        "intraday_ok":   False,
        "error":         None,
    }

    if not out["configured"]:
        out["error"] = "DHAN_CLIENT_ID / DHAN_ACCESS_TOKEN not set in .env"
        return out

    reset_breaker()

    try:
        out["scrip_master"] = len(get_symbol_map())
        sid = get_security_id(symbol)
        out["security_id"] = sid
        if not sid:
            out["error"] = f"{symbol} not found in scrip master"
            return out

        d = get_daily_bars(symbol, days=30)
        out["daily_ok"] = d is not None and not d.empty

        i = get_intraday_candles(symbol, interval_minutes=15, days=2)
        out["intraday_ok"] = i is not None and not i.empty

        if _data_forbidden:
            out["error"] = "Token valid but Data API not entitled (403/401)"
        elif not out["daily_ok"]:
            out["error"] = "Daily request returned no data"
    except Exception as e:
        out["error"] = str(e)

    return out


if __name__ == "__main__":
    import json
    print(json.dumps(check_entitlement(), indent=2, default=str))
