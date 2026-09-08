"""
TrueData client — two layers:

1. REST API (no WebSocket needed) — fast, reliable, used for historical OHLCV
   Endpoints: https://history.truedata.in/
   Auth:      https://auth.truedata.in/token (Bearer token, refreshed hourly)

2. WebSocket / Velocity 2.0 — live LTP and intraday candles
   Falls back gracefully when WebSocket is not connected.

Usage:
    import truedata_client as td
    td.init()                                          # connect WebSocket (optional)
    df = td.get_daily_bars_rest('RELIANCE', days=550)  # REST — always available
    prices = td.get_ltp(['RELIANCE', 'HDFC'])          # WebSocket LTP
    df = td.get_intraday_candles('RELIANCE', 15)       # WebSocket intraday
"""
import time
import logging
import threading
import requests
import io
import pandas as pd
import config

# Pre-emptively silence truedata-ws before it's even imported
logging.getLogger("truedata_ws").setLevel(logging.CRITICAL)
logging.getLogger("truedata_ws").propagate = False

import sys as _sys

class _TrueDataFilter:
    """Drop stderr lines from truedata-ws internal retry spam."""
    _KEYWORDS = (b"Invalid User Credentials", b"access_token", b"Failed to connect",
                 b"request encountered an error", b"user name or password", b"subscription expired",
                 b"truedata_ws", b"TD.py", b"websocket.TD")

    def __init__(self, wrapped):
        self._w = wrapped

    def write(self, data):
        try:
            data_b = data.encode("utf-8", errors="replace") if isinstance(data, str) else data
            if any(k in data_b for k in self._KEYWORDS):
                return
        except Exception:
            pass
        try:
            self._w.write(data)
        except Exception:
            pass

    def flush(self):
        try:
            self._w.flush()
        except Exception:
            pass

    def fileno(self):
        return self._w.fileno()

    def __getattr__(self, name):
        return getattr(self._w, name)

_sys.stderr = _TrueDataFilter(_sys.stderr)

_td   = None
_lock = threading.Lock()
_connected = False

# TrueData uses "SYMBOL-EQ" format for NSE equities
def _td_symbol(symbol: str) -> str:
    s = symbol.replace(".NS", "").replace(".BO", "").upper()
    return s if s.endswith("-EQ") else f"{s}-EQ"


# ── TrueData REST API (historical OHLCV — no WebSocket needed) ───────────────
_REST_AUTH_URL    = "https://auth.truedata.in/token"
_REST_HISTORY_URL = "https://history.truedata.in"

_rest_token:    str   = ""
_rest_token_ts: float = 0.0
_rest_auth_dead: bool = False   # latched on a hard 4xx — stops per-stock retry storms
_rest_lock = threading.Lock()
_REST_TOKEN_TTL = 3500   # refresh 100s before 1-hour expiry


def _get_rest_token() -> str:
    """
    Authenticate via TrueData REST and return a Bearer token. Cached for ~1 hour.

    A 4xx here means the subscription has lapsed or the credentials are rejected —
    retrying cannot fix that. Previously the failure path re-attempted the network
    call for EVERY stock (the `if not _rest_token` guard is always true on failure),
    so a 500-stock scan fired 500 doomed auth requests and printed 500 lines.
    A hard auth failure now latches the client off for the session.
    """
    global _rest_token, _rest_token_ts, _rest_auth_dead
    with _rest_lock:
        if _rest_auth_dead:
            return ""
        if _rest_token and time.time() - _rest_token_ts < _REST_TOKEN_TTL:
            return _rest_token
        user = getattr(config, "TRUEDATA_USER", "")
        pwd  = getattr(config, "TRUEDATA_PASSWORD", "")
        if not user or not pwd:
            _rest_auth_dead = True
            return ""
        try:
            r = requests.post(
                _REST_AUTH_URL,
                data={"username": user, "password": pwd, "grant_type": "password"},
                timeout=10,
            )
            r.raise_for_status()
            token = r.json().get("access_token", "")
            if token:
                _rest_token    = token
                _rest_token_ts = time.time()
                return token
            _rest_auth_dead = True
            print("[TrueData REST] Auth returned no token — disabling for this session.")
        except Exception as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status is not None and 400 <= status < 500:
                _rest_auth_dead = True
                print(
                    f"[TrueData REST] Auth rejected (HTTP {status}) — subscription "
                    f"lapsed or credentials invalid. Disabling for this session."
                )
            else:
                # Transient (network/5xx): allow a retry on the next call
                print(f"[TrueData REST] Auth unavailable: {e}")
        return ""


def rest_auth_dead() -> bool:
    """True once TrueData auth has hard-failed and been latched off."""
    return _rest_auth_dead


def get_daily_bars_rest(symbol: str, days: int = 550) -> pd.DataFrame | None:
    """
    Fetch daily (EOD) OHLCV via TrueData REST API.
    Uses multiple getlastnbars calls (documented, max 200 bars each) to cover the
    requested history. For 550 days (~385 trading days), 2 calls of 200 bars suffice.
    Returns DataFrame[Open, High, Low, Close, Volume] indexed by Date, or None.
    """
    token = _get_rest_token()
    if not token:
        return None
    try:
        sym_clean = symbol.replace(".NS", "").replace(".BO", "").upper()
        headers   = {"Authorization": f"Bearer {token}"}

        # Step 1: Try getbars?interval=eod (undocumented but faster — one call for full range)
        # Falls through silently if it returns nothing.
        from datetime import datetime, timedelta
        end   = datetime.now()
        start = end - timedelta(days=days)
        r = requests.get(
            f"{_REST_HISTORY_URL}/getbars",
            params={
                "symbol":   sym_clean,
                "from":     start.strftime("%y%m%dT00:00:00"),
                "to":       end.strftime("%y%m%dT23:59:59"),
                "interval": "eod",
                "response": "csv",
            },
            headers=headers,
            timeout=15,
        )
        if r.status_code == 200 and len(r.text.strip()) > 100:
            df = _parse_td_csv(r.text)
            if df is not None and len(df) >= 50:
                return df

        # Step 2: Multiple getlastnbars calls (documented, max 200 bars each).
        # Two calls of 200 bars (most recent + earlier) covers ~400 trading days ≈ 20 months.
        trading_days_needed = int(days * 5 / 7)   # ~5/7 of calendar days are trading days
        chunks: list[pd.DataFrame] = []

        # Call the endpoint with a before_timestamp to page backwards
        before_ts: str | None = None
        for _ in range(4):   # up to 4 calls × 200 bars = 800 trading days (~3.2 years)
            params: dict = {
                "symbol":   sym_clean,
                "nbars":    200,
                "interval": "eod",
                "bidask":   0,
                "response": "csv",
            }
            if before_ts:
                params["to"] = before_ts   # some APIs support a "to" parameter
            r2 = requests.get(
                f"{_REST_HISTORY_URL}/getlastnbars",
                params=params,
                headers=headers,
                timeout=15,
            )
            if r2.status_code != 200 or not r2.text.strip():
                break
            chunk = _parse_td_csv(r2.text)
            if chunk is None or chunk.empty:
                break
            chunks.append(chunk)
            # Stop if we have enough history
            total_rows = sum(len(c) for c in chunks)
            if total_rows >= trading_days_needed:
                break
            # Set before_ts to day before oldest bar in this chunk
            oldest = chunk.index.min()
            before_ts = (oldest - timedelta(days=1)).strftime("%y%m%dT23:59:59")

        if not chunks:
            return None
        combined = pd.concat(chunks).sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]
        return combined if len(combined) >= 50 else None

    except Exception:
        return None


def get_ltp_rest(symbols: list[str]) -> dict[str, float]:
    """
    Get live LTP for a list of symbols via TrueData REST API.
    Uses getlastnticks?nticks=1 (documented alternative to WebSocket LTP).
    Returns {symbol: ltp} dict. Empty dict if auth fails or no data.
    """
    token = _get_rest_token()
    if not token:
        return {}
    prices: dict[str, float] = {}
    headers = {"Authorization": f"Bearer {token}"}
    for sym in symbols:
        sym_clean = sym.replace(".NS", "").replace(".BO", "").upper()
        try:
            r = requests.get(
                f"{_REST_HISTORY_URL}/getlastnticks",
                params={
                    "symbol":   sym_clean,
                    "nticks":   1,
                    "bidask":   0,
                    "interval": "tick",
                    "response": "csv",
                },
                headers=headers,
                timeout=8,
            )
            if r.status_code == 200 and r.text.strip():
                lines = r.text.strip().splitlines()
                if len(lines) >= 2:   # header + data row
                    cols = [c.lower().strip() for c in lines[0].split(",")]
                    vals = lines[1].split(",")
                    row  = dict(zip(cols, vals))
                    ltp_val = row.get("ltp") or row.get("close") or row.get("price")
                    if ltp_val:
                        prices[sym] = float(ltp_val)
        except Exception:
            pass
    return prices


def _parse_td_csv(text: str) -> pd.DataFrame | None:
    """Parse TrueData CSV response into a standardised OHLCV DataFrame."""
    try:
        df = pd.read_csv(io.StringIO(text))
        # Normalise column names: 'timestamp'→index, OHLCV
        df.columns = [c.strip().lower() for c in df.columns]
        if "timestamp" not in df.columns:
            return None
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
        # Map to standard names
        col_map = {
            "open":   "Open",  "o": "Open",
            "high":   "High",  "h": "High",
            "low":    "Low",   "l": "Low",
            "close":  "Close", "c": "Close",
            "volume": "Volume","v": "Volume",
        }
        df = df.rename(columns=col_map)
        need = ["Open", "High", "Low", "Close", "Volume"]
        present = [c for c in need if c in df.columns]
        if "Close" not in present:
            return None
        # Ensure Volume exists (some feeds omit it)
        if "Volume" not in df.columns:
            df["Volume"] = 0
        df = df[["Open", "High", "Low", "Close", "Volume"]].apply(
            pd.to_numeric, errors="coerce"
        ).dropna(subset=["Close"])
        return df if len(df) >= 50 else None
    except Exception:
        return None


def get_bhavcopy(date_str: str | None = None) -> pd.DataFrame | None:
    """
    Fetch NSE EQ bhavcopy (all stocks' EOD OHLCV) for a given date.
    date_str: 'YYYY-MM-DD'. Defaults to today.
    Returns DataFrame with symbol as index + OHLCV columns, or None.
    """
    token = _get_rest_token()
    if not token:
        return None
    try:
        from datetime import datetime
        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")
        r = requests.get(
            f"{_REST_HISTORY_URL}/getbhavcopy",
            params={"segment": "eq", "date": date_str, "response": "csv"},
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
        )
        if r.status_code != 200 or not r.text.strip():
            return None
        df = pd.read_csv(io.StringIO(r.text))
        df.columns = [c.strip().lower() for c in df.columns]
        # Typical bhavcopy columns include: symbol, open, high, low, close, volume
        return df
    except Exception as e:
        print(f"[TrueData REST] Bhavcopy fetch failed: {e}")
        return None


def init() -> bool:
    """Connect to TrueData. Safe to call multiple times — only connects once."""
    global _td, _connected
    user = getattr(config, "TRUEDATA_USER", "")
    pwd  = getattr(config, "TRUEDATA_PASSWORD", "")
    if not user or not pwd:
        return False

    # Silence ALL truedata-ws loggers before doing anything
    for name in list(logging.Logger.manager.loggerDict.keys()):
        if "truedata" in name.lower():
            logging.getLogger(name).setLevel(logging.CRITICAL)
            logging.getLogger(name).propagate = False

    with _lock:
        if _connected and _td is not None:
            return True
        try:
            from truedata_ws.websocket.TD import TD
            _td = TD(
                login_id=user,
                password=pwd,
                live_port=8084,   # production port per TrueData v2.6 docs
                historical_api=True,
                log_level=logging.CRITICAL,
            )
            # Wait for handshake (max 5s — TD() is non-blocking, token arrives async)
            for _ in range(10):
                time.sleep(0.5)
                if hasattr(_td, 'access_token') and _td.access_token:
                    break

            # Re-silence any new loggers the library registered during init
            for name in list(logging.Logger.manager.loggerDict.keys()):
                if "truedata" in name.lower():
                    logging.getLogger(name).setLevel(logging.CRITICAL)
                    logging.getLogger(name).propagate = False

            if not hasattr(_td, 'access_token') or not _td.access_token:
                # Stop the library's internal retry threads before giving up
                try:
                    _td.disconnect()
                except Exception:
                    pass
                _td = None
                print("[TrueData] Credentials invalid — waiting for Market Data API approval")
                _connected = False
                return False

            _connected = True
            print(f"[TrueData] Connected as {user}")
            return True
        except Exception as e:
            try:
                if _td is not None:
                    _td.disconnect()
            except Exception:
                pass
            _td = None
            _connected = False
            if "credentials" not in str(e).lower() and "password" not in str(e).lower():
                print(f"[TrueData] Connection failed — skipping ({e})")
            return False


def is_connected() -> bool:
    return _connected and _td is not None


def get_ltp(symbols: list[str]) -> dict[str, float]:
    """
    Returns {symbol: ltp} for a list of NSE symbols (no suffix needed).
    Subscribes to live feed, waits briefly for first tick, then unsubscribes.
    """
    if not is_connected():
        return {}
    try:
        td_syms = [_td_symbol(s) for s in symbols]
        req_ids = _td.start_live_data(td_syms)
        time.sleep(1.5)  # wait for first tick

        prices = {}
        for sym, req_id in zip(symbols, req_ids):
            try:
                td_sym = _td_symbol(sym)
                data   = _td.live_data.get(td_sym) or _td.live_data.get(req_id)
                if data and hasattr(data, "ltp") and data.ltp:
                    prices[sym] = float(data.ltp)
            except Exception:
                pass

        _td.stop_live_data(req_ids)
        return prices
    except Exception as e:
        print(f"[TrueData] LTP fetch failed: {e}")
        return {}


def get_intraday_candles(symbol: str, interval_minutes: int = 15) -> pd.DataFrame | None:
    """
    Fetch today's intraday OHLCV candles.
    Returns DataFrame with columns: open, high, low, close, volume
    indexed by datetime. Returns None on failure.
    """
    if not is_connected():
        return None
    try:
        from datetime import datetime, timedelta
        td_sym = _td_symbol(symbol)
        bar_size = f"{interval_minutes} min"

        now   = datetime.now()
        start = now.replace(hour=9, minute=15, second=0, microsecond=0)

        bars = _td.get_historic_data(
            contract   = td_sym,
            start_time = start,
            end_time   = now,
            bar_size   = bar_size,
        )

        if bars is None or bars.empty:
            return None

        # Normalise column names to lowercase
        bars.columns = [c.lower() for c in bars.columns]

        required = {"open", "high", "low", "close", "volume"}
        if not required.issubset(set(bars.columns)):
            return None

        bars = bars[["open", "high", "low", "close", "volume"]].astype(float)
        return bars

    except Exception as e:
        print(f"[TrueData] Intraday candles failed for {symbol}: {e}")
        return None


def get_daily_bars(symbol: str, days: int = 365) -> pd.DataFrame | None:
    """
    Fetch daily OHLCV bars — can replace yfinance for swing scan.
    days=365 covers 1 year, needed for IBD RS Rank 12M calculation.
    """
    if not is_connected():
        return None
    try:
        from datetime import datetime, timedelta
        td_sym = _td_symbol(symbol)
        end    = datetime.now()
        start  = end - timedelta(days=days)

        bars = _td.get_historic_data(
            contract   = td_sym,
            start_time = start,
            end_time   = end,
            bar_size   = "1 day",
        )

        if bars is None or bars.empty:
            return None

        bars.columns = [c.lower() for c in bars.columns]
        required = {"open", "high", "low", "close", "volume"}
        if not required.issubset(set(bars.columns)):
            return None

        return bars[["open", "high", "low", "close", "volume"]].astype(float)

    except Exception as e:
        print(f"[TrueData] Daily bars failed for {symbol}: {e}")
        return None
