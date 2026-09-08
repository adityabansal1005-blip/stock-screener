"""
Groww API client — initialized once and shared across the app.
Token resets daily at 6 AM IST (Groww's reset window), so we auto-refresh it.
Falls back gracefully if keys are not configured.
"""
from growwapi import GrowwAPI
import datetime
import threading
import warnings
import pandas as pd

# We knowingly use the deprecated V1 candle endpoint — the V2 replacement
# (`get_historical_candles`) returns null for OPEN on every row, which breaks
# candlestick pattern detection. Silence the per-call warning so a 500-stock
# scan doesn't emit 500 identical blocks; the rationale lives in _candles_to_df.
warnings.filterwarnings(
    "ignore",
    message=r".*get_historical_candle_data.*deprecated.*",
    category=DeprecationWarning,
)

_groww: GrowwAPI | None = None
_api_key: str = ""
_api_secret: str = ""
_token_ts: datetime.datetime | None = None
_lock = threading.Lock()

# Cooldown after DNS/connection failures — stops retry spam
_api_last_fail: float = 0.0
_API_COOLDOWN_SECS = 120   # 2 minutes between retries after failure


# ── Rate limiting ────────────────────────────────────────────────────────────
# Groww caps the "Live Data" category (candles AND ltp share one budget) at
# 10 req/sec and 300 req/min. The per-MINUTE ceiling is the binding one — it
# works out to 5/sec sustained, so a burst-only limiter is not enough.
# Both windows are enforced with margin; scans run 8-10 threads concurrently.
_RATE_PER_SEC = 8
_RATE_PER_MIN = 280
_rate_lock = threading.Lock()
_hits_sec: list[float] = []
_hits_min: list[float] = []


def _throttle() -> None:
    """Block until a request fits inside both the 1s and 60s budgets."""
    import time
    while True:
        with _rate_lock:
            now = time.time()
            while _hits_sec and now - _hits_sec[0] >= 1.0:
                _hits_sec.pop(0)
            while _hits_min and now - _hits_min[0] >= 60.0:
                _hits_min.pop(0)

            if len(_hits_sec) < _RATE_PER_SEC and len(_hits_min) < _RATE_PER_MIN:
                _hits_sec.append(now)
                _hits_min.append(now)
                return

            waits = []
            if len(_hits_sec) >= _RATE_PER_SEC:
                waits.append(1.0 - (now - _hits_sec[0]))
            if len(_hits_min) >= _RATE_PER_MIN:
                waits.append(60.0 - (now - _hits_min[0]))
            sleep_for = max(min(waits), 0.01)
        time.sleep(sleep_for)


def _is_rate_limited(exc: Exception) -> bool:
    m = str(exc).lower()
    return "rate limit" in m or "too many request" in m or "429" in m


def _api_reachable() -> bool:
    """Returns False for 2 minutes after a DNS/connection failure."""
    import time
    return (time.time() - _api_last_fail) > _API_COOLDOWN_SECS


def _mark_api_fail():
    global _api_last_fail
    import time
    _api_last_fail = time.time()


# Market-data entitlement breaker. A 403 means the API key authenticates but is
# not scoped for live/historical data — retrying every symbol cannot fix that,
# it just burns thousands of calls per scan. Trip once, log once, skip the rest.
_data_forbidden: bool = False


def _is_forbidden(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "forbidden" in msg or "required permissions" in msg or "403" in msg


def _trip_forbidden(what: str) -> None:
    global _data_forbidden
    if not _data_forbidden:
        _data_forbidden = True
        print(
            f"[Groww] {what} refused (403). The API key authenticates but is not "
            f"entitled to market data — enable the Live/Historical Data API on the "
            f"Groww account. Skipping further Groww data calls this session."
        )


def data_forbidden() -> bool:
    """True once Groww has refused a market-data call for lack of permissions."""
    return _data_forbidden


_RESET = datetime.time(6, 0, 30)   # Groww invalidates tokens at 06:00 IST


def _last_reset(now: datetime.datetime) -> datetime.datetime:
    """The most recent 06:00 boundary at or before `now`."""
    boundary = datetime.datetime.combine(now.date(), _RESET)
    return boundary if now >= boundary else boundary - datetime.timedelta(days=1)


def _needs_refresh() -> bool:
    """
    A token is dead once a 06:00 reset has passed since it was minted.

    This used to compare CALENDAR DATES, which silently failed for exactly the
    case that matters: a token minted at 03:55 has today's date, so after the
    06:00 reset the check still said "same day, still good" and every request
    from 06:00 onward went out with a dead token. An overnight scan started
    before 6 AM would degrade to the nselib fallback (~10x slower) two hours in,
    with no error naming the cause. Compare against the reset boundary instead.
    """
    if _token_ts is None:
        return True
    return _token_ts < _last_reset(datetime.datetime.now())


def _connect() -> bool:
    global _groww, _token_ts
    try:
        token = GrowwAPI.get_access_token(api_key=_api_key, secret=_api_secret)
        _groww = GrowwAPI(token)
        _token_ts = datetime.datetime.now()
        print(f"[Groww] Token refreshed at {datetime.datetime.now().strftime('%H:%M:%S')}")
        return True
    except Exception as e:
        print(f"[Groww] Connection failed: {e}")
        _groww = None
        return False


def init_groww(api_key: str, api_secret: str) -> bool:
    global _api_key, _api_secret
    if api_key == "YOUR_GROWW_API_KEY" or not api_key:
        return False
    _api_key = api_key
    _api_secret = api_secret
    return _connect()


def ensure_connected() -> bool:
    """Call before every API request — refreshes token if needed."""
    with _lock:
        if _needs_refresh() and _api_key:
            _connect()
        return _groww is not None


def is_connected() -> bool:
    ensure_connected()
    return _groww is not None


def get_live_ltp(symbols: list[str]) -> dict[str, float]:
    """Returns {symbol: ltp} for a list of NSE symbols (no .NS suffix)."""
    if _data_forbidden or not ensure_connected():
        return {}
    try:
        _throttle()   # shares the Live Data budget with candle requests
        # Format required by Groww API: "NSE_SYMBOL" (underscore separator)
        groww_symbols = tuple(f"NSE_{s}" for s in symbols)
        result = _groww.get_ltp(
            exchange_trading_symbols=groww_symbols,
            segment=GrowwAPI.SEGMENT_CASH,
            timeout=8,
        )
        prices = {}
        if isinstance(result, dict):
            for key, val in result.items():
                # key may be "NSE_RELIANCE" → strip prefix
                sym = key.split("_", 1)[-1] if "_" in key else key
                if isinstance(val, dict):
                    prices[sym] = val.get("ltp", 0)
                elif isinstance(val, (int, float)):
                    prices[sym] = float(val)
        return prices
    except Exception as e:
        if _is_forbidden(e):
            _trip_forbidden("Live LTP")
        else:
            print(f"[Groww] LTP fetch failed: {e}")
        return {}


def get_holdings() -> list[dict]:
    """Returns list of held stocks with qty, avg price, current value."""
    if not ensure_connected() or not _api_reachable():
        return []
    try:
        data = _groww.get_holdings_for_user(timeout=10)
        raw  = data.get("holdingData") or data.get("holdings") or data.get("data") or []
        if isinstance(raw, dict):
            raw = list(raw.values())
        holdings = []
        for h in raw:
            if not isinstance(h, dict):
                continue
            sym = (h.get("tradingSymbol") or h.get("trading_symbol") or h.get("symbol") or "").upper()
            if not sym:
                continue
            holdings.append({
                "symbol":        sym,
                "quantity":      float(h.get("quantity") or h.get("qty") or 0),
                "avg_price":     float(h.get("averagePrice") or h.get("average_price") or h.get("avg_price") or 0),
                "ltp":           float(h.get("ltp") or h.get("lastPrice") or h.get("last_price") or 0),
                "current_value": float(h.get("currentValue") or h.get("current_value") or 0),
                "pnl":           float(h.get("pnl") or h.get("unrealised_pnl") or 0),
                "isin":          h.get("isin") or "",
                "exchange":      (h.get("exchange") or "NSE").upper(),
            })
        return holdings
    except Exception as e:
        if "getaddrinfo" in str(e) or "NameResolution" in str(e) or "Failed to resolve" in str(e):
            _mark_api_fail()
        else:
            print(f"[Groww] Holdings fetch failed: {e}")
        return []


def get_positions() -> list[dict]:
    """Returns today's intraday positions."""
    if not ensure_connected() or not _api_reachable():
        return []
    try:
        data = _groww.get_positions_for_user(
            segment="CASH",
            timeout=10,
        )
        raw = data.get("positionData") or data.get("positions") or data.get("data") or []
        if isinstance(raw, dict):
            raw = list(raw.values())
        positions = []
        for p in raw:
            if not isinstance(p, dict):
                continue
            sym = (p.get("tradingSymbol") or p.get("trading_symbol") or p.get("symbol") or "").upper()
            qty = float(p.get("quantity") or p.get("qty") or 0)
            product = (p.get("productType") or p.get("product") or "MIS").upper()
            if not sym or qty == 0:
                continue
            # Skip CNC (delivery) positions — those are holdings, not intraday
            if product == "CNC":
                continue
            avg = float(
                p.get("averagePrice") or p.get("average_price") or
                p.get("buyAvgPrice") or p.get("buy_avg_price") or
                p.get("avgPrice") or 0
            )
            positions.append({
                "symbol":    sym,
                "quantity":  qty,
                "avg_price": avg,
                "ltp":       float(p.get("ltp") or 0),
                "pnl":       float(p.get("pnl") or p.get("unrealised_pnl") or 0),
                "product":   product,
            })
        return positions
    except Exception as e:
        if "getaddrinfo" in str(e) or "NameResolution" in str(e) or "Failed to resolve" in str(e):
            _mark_api_fail()
        else:
            print(f"[Groww] Positions fetch failed: {e}")
        return []


def get_funds() -> dict:
    """Returns available cash balance."""
    if not ensure_connected() or not _api_reachable():
        return {}
    try:
        data = _groww.get_available_margin_details(timeout=10)
        eq   = data.get("equity") or data.get("EQUITY") or data
        return {
            "available": float(eq.get("availableMargin") or eq.get("available_margin") or eq.get("available") or 0),
            "used":      float(eq.get("utilisedMargin")  or eq.get("utilised_margin")  or eq.get("used") or 0),
            "total":     float(eq.get("totalMargin")     or eq.get("total_margin")      or eq.get("total") or 0),
        }
    except Exception as e:
        if "getaddrinfo" in str(e) or "NameResolution" in str(e) or "Failed to resolve" in str(e):
            _mark_api_fail()
        else:
            print(f"[Groww] Funds fetch failed: {e}")
        return {}


def get_user_profile() -> dict:
    """Returns basic user info — name, email."""
    if not ensure_connected():
        return {}
    try:
        data = _groww.get_user_profile(timeout=10)
        return {
            "name":  data.get("name") or data.get("userName") or data.get("user_name") or "Trader",
            "email": data.get("email") or "",
        }
    except Exception as e:
        return {}


def _candles_to_df(candles: list, intraday: bool) -> pd.DataFrame | None:
    """
    Convert Groww V1 candle rows into the Open/High/Low/Close/Volume frame the
    rest of the project expects, indexed by datetime (IST).

    Rows are [epoch_seconds, open, high, low, close, volume].

    We deliberately use the deprecated `get_historical_candle_data` (V1) rather
    than `get_historical_candles` (V2): V2 returns null for OPEN on every row,
    which breaks candlestick pattern detection. Revisit if Groww fixes V2.
    """
    if not candles:
        return None
    try:
        df = pd.DataFrame(candles).iloc[:, :6]
        df.columns = ["ts", "Open", "High", "Low", "Close", "Volume"]

        idx = pd.to_datetime(df["ts"], unit="s", utc=True)
        idx = idx.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
        if not intraday:
            idx = idx.dt.normalize()

        df = df.drop(columns=["ts"])
        df.index = pd.DatetimeIndex(idx, name="Date")
        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.dropna(subset=["Open", "High", "Low", "Close"]).sort_index()
        df = df[~df.index.duplicated(keep="last")]
        return df if not df.empty else None
    except Exception as e:
        print(f"[Groww] Candle parse failed: {e}")
        return None


def _route(symbol: str) -> tuple[str, str]:
    """
    Resolve a symbol to (trading_symbol, exchange).

    Groww serves BSE as well as NSE, but BSE wants the numeric SCRIP CODE as
    the trading symbol — 'HINDINSUL' fails where '539984' works. An all-digit
    symbol is therefore treated as a BSE scrip code; anything else is an NSE
    ticker. This is what makes BSE-only companies reachable at all: roughly
    2,500 genuine equities that were previously invisible to the scanner.
    """
    s = str(symbol).replace(".NS", "").replace(".BO", "").strip().upper()
    if s.isdigit():
        return s, GrowwAPI.EXCHANGE_BSE
    return s, GrowwAPI.EXCHANGE_NSE


def _fetch_candles(symbol: str, start_str: str, end_str: str, interval: int,
                   intraday: bool, label: str, retries: int = 3) -> pd.DataFrame | None:
    """
    Rate-limited candle fetch with backoff on rate-limit responses.

    Groww's limit is shared across the whole Live Data category, so a scan
    running 10 threads can breach it even when each thread looks slow. The
    throttle is global; a breach that still slips through is retried with
    exponential backoff rather than dropping the symbol.
    """
    import time
    tsym, exch = _route(symbol)
    for attempt in range(retries):
        _throttle()
        try:
            data = _groww.get_historical_candle_data(
                trading_symbol=tsym,
                exchange=exch,
                segment=GrowwAPI.SEGMENT_CASH,
                start_time=start_str,
                end_time=end_str,
                interval_in_minutes=interval,
                timeout=15,
            )
            return _candles_to_df((data or {}).get("candles"), intraday=intraday)
        except Exception as e:
            if _is_forbidden(e):
                _trip_forbidden(label)
                return None
            if _is_rate_limited(e) and attempt < retries - 1:
                time.sleep(2 ** attempt)      # 1s, 2s
                continue
            if not _is_rate_limited(e):
                print(f"[Groww] {label} failed for {symbol}: {e}")
            return None
    return None


def get_daily_bars(symbol: str, days: int = 550) -> pd.DataFrame | None:
    """
    Daily OHLCV for the trailing `days` calendar days.
    Groww allows ~1080 days per request with full history, so no chunking needed.
    """
    if _data_forbidden or not ensure_connected():
        return None
    end   = datetime.datetime.now()
    start = end - datetime.timedelta(days=days)
    return _fetch_candles(
        symbol,
        start.strftime("%Y-%m-%d 09:15:00"),
        end.strftime("%Y-%m-%d 15:30:00"),
        1440,
        intraday=False,
        label="Daily bars",
    )


def get_daily_bars_range(symbol: str, start: datetime.date,
                         end: datetime.date) -> pd.DataFrame | None:
    """
    Daily OHLCV between two explicit dates.

    `get_daily_bars(days=N)` always measures N days back from TODAY, so it
    cannot be used to walk backwards through history — every call returns the
    same recent window. Deep-history fetches need this variant instead.
    Groww caps a single request near 1080 days; callers should chunk.
    """
    if _data_forbidden or not ensure_connected():
        return None
    return _fetch_candles(
        symbol,
        start.strftime("%Y-%m-%d 09:15:00"),
        end.strftime("%Y-%m-%d 15:30:00"),
        1440,
        intraday=False,
        label="Daily bars",
    )


def get_intraday_candles(symbol: str, interval_minutes: int = 15,
                         days: int = 5) -> pd.DataFrame | None:
    """
    Intraday candles for the trailing `days` sessions.

    Defaults to 5 days rather than today alone for two reasons: a single
    session yields only ~25 bars at 15-min, far short of the ~50 the indicator
    suite needs, and a today-only window returns nothing on weekends and
    holidays. Extra sessions cost nothing and make the scan robust.

    (The previous implementation also emitted lowercase column names the
    scanner could not read, and parsed epoch seconds as nanoseconds — dating
    every candle to 1970. Both are fixed by routing through _candles_to_df.)
    """
    if _data_forbidden or not ensure_connected():
        return None
    end   = datetime.datetime.now()
    start = end - datetime.timedelta(days=days)
    return _fetch_candles(
        symbol,
        start.strftime("%Y-%m-%d 09:15:00"),
        end.strftime("%Y-%m-%d %H:%M:%S"),
        interval_minutes,
        intraday=True,
        label="Intraday candles",
    )
