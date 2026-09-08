import time
import threading
import contextlib
import io
import warnings

# Suppress pandas FutureWarning from strategy_library (.fillna boolean downcasting)
warnings.filterwarnings("ignore", category=FutureWarning, module="strategy_library")
import yfinance as yf
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from signals import compute_signals
from technical_enhanced import compute_enhanced
from market_context import get_market_context
from backtester import backtest_win_rate
from fundamentals import get_fundamentals
import ai_analyst
import groww_client
import truedata_client
import config
from stocks_nse import get_all_symbols, load_watchlist

_STOCK_TIMEOUT = 25   # seconds before skipping a hung stock
SCORE_VERSION = "measurement-v2-no-rr"

# ── Per-stock OHLCV cache (5-minute TTL) ────────────────────────────────────────────
# Eliminates repeated yfinance calls when multiple scans run in the same session.
# Key: symbol string.  Value: {"daily": df, "weekly": df, "ts": timestamp}
_ohlcv_cache: dict = {}
_OHLCV_TTL = 5 * 60   # current daily bars can change during a session
_ohlcv_lock = threading.Lock()


def _yf_history(symbol: str, **kwargs):
    kwargs.setdefault("timeout", 15)
    # yfinance owns its curl_cffi session; a requests.Session is incompatible.
    return yf.Ticker(symbol).history(**kwargs)


def _nselib_ohlcv(sym_clean: str) -> pd.DataFrame | None:
    """
    Fetch daily OHLCV for an NSE equity via nselib (NSE direct API, no rate limits).
    Returns a DataFrame with columns [Open, High, Low, Close, Volume] indexed by Date,
    or None on failure.
    """
    try:
        from nselib import capital_market
        from datetime import datetime as _dt, timedelta as _td
        _end   = _dt.now()
        _start = _end - _td(days=550)   # ~18 months for indicators + RS
        df = capital_market.price_volume_and_deliverable_position_data(
            symbol=sym_clean,
            from_date=_start.strftime("%d-%m-%Y"),
            to_date=_end.strftime("%d-%m-%Y"),
        )
        if df is None or df.empty:
            return None
        # Strip BOM / quotes from column names
        df.columns = [c.strip().lstrip("﻿").strip('"') for c in df.columns]
        # Parse numeric fields (they come with comma-thousands: "1,412.10")
        for col in ["OpenPrice", "HighPrice", "LowPrice", "ClosePrice", "TotalTradedQuantity"]:
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace(",", "", regex=False), errors="coerce"
                )
        # Parse date (format: "29-May-2025")
        if "Date" not in df.columns:
            return None
        df["Date"] = pd.to_datetime(df["Date"], format="%d-%b-%Y", errors="coerce")
        df = df.dropna(subset=["Date", "ClosePrice"])

        # NSE returns one row per SERIES, so a symbol can yield two rows for the
        # same date and left unhandled those duplicates double-count every
        # rolling indicator.
        #
        # The first fix for that filtered to EQ only, which introduced a worse
        # bug: a stock moved to trade-to-trade/surveillance has NO further EQ
        # rows, so its history was silently truncated at the transition date and
        # the last available bar was treated as today's. RELINFRA moved EQ -> BE
        # on 04 Jun 2025; fifteen months later the scanner still scored it on
        # that day's Rs 380.60 close while it traded at Rs 59.4.
        #
        # BE and BZ are surveillance series but they are real continuous trading
        # in the same equity, so they belong in the history. BL (block deals) and
        # anything else are negotiated prints, not the continuous series, and are
        # excluded. Duplicates are then resolved in favour of EQ rather than by
        # dropping the other series wholesale.
        if "Series" in df.columns:
            ser = df["Series"].astype(str).str.strip().str.upper()
            df = df[ser.isin(("EQ", "BE", "BZ"))].copy()
            if df.empty:
                return None
            df["_pref"] = (ser.loc[df.index] != "EQ").astype(int)  # EQ sorts first
            df = df.sort_values(["Date", "_pref"]).drop(columns="_pref")
            df = df[~df["Date"].duplicated(keep="first")]

        df = df.set_index("Date").sort_index()
        df = df[~df.index.duplicated(keep="last")]
        daily = df.rename(columns={
            "OpenPrice":  "Open",
            "HighPrice":  "High",
            "LowPrice":   "Low",
            "ClosePrice": "Close",
            "TotalTradedQuantity": "Volume",
        })[["Open", "High", "Low", "Close", "Volume"]]
        daily = daily.dropna(subset=["Close"])
        return daily if len(daily) >= 50 else None
    except Exception:
        return None


def _get_ohlcv(symbol: str) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """
    Return (daily_df, weekly_df) from cache or fetch fresh data.
    Priority:
      1. Groww     — ~0.4s/stock, full OHLCV, active subscription  ← primary
      2. Dhan v2   — ~0.2s/stock; only if DHAN_* credentials are set
      3. nselib    — NSE direct, ~4s/stock, free, no rate limits   ← safety net
      4. yfinance  — last resort, rate-limited by Yahoo
    """
    now = time.time()
    with _ohlcv_lock:
        entry = _ohlcv_cache.get(symbol)
        if entry and entry.get("daily") is not None and now - entry["ts"] < _OHLCV_TTL:
            return entry["daily"], entry["weekly"]

    sym_clean = symbol.replace(".NS", "").replace(".BO", "").upper()
    daily, weekly = None, None

    def _make_weekly(d: pd.DataFrame) -> pd.DataFrame | None:
        try:
            w = d.resample("W").agg({
                "Open": "first", "High": "max", "Low": "min",
                "Close": "last", "Volume": "sum"
            }).dropna(subset=["Close"])
            return w if len(w) >= 15 else None
        except Exception:
            return None

    src = "yfinance"  # track which source succeeded

    def _persist_read(sym: str):
        """
        Today's frame from the shared SQLite archive, if already fetched today.

        The in-memory cache is lost on every restart, so the first scan after
        starting the server refetched all 500 symbols (~8 min instead of ~2).
        The archive that data_manager maintains already holds this data — the
        scan path simply never read it.
        """
        try:
            import data_manager
            key = symbol
            if data_manager._is_stale(key):
                return None                       # never triggers a refetch here
            d = data_manager.load(key)
            if d is None or len(d) < 120:
                return None
            d = d.tail(600).copy()
            d.columns = [c.title() for c in d.columns]
            return d[["Open", "High", "Low", "Close", "Volume"]]
        except Exception:
            return None

    def _persist_write(sym: str, d):
        """Store a freshly fetched frame so the next restart starts warm."""
        try:
            import data_manager
            w = d.copy()
            w.columns = [c.lower() for c in w.columns]
            w.index.name = "date"
            data_manager._store(symbol, w)
        except Exception:
            pass

    def _accept(df, source: str):
        """Validate before trusting a source. Bad data falls through, loudly."""
        if df is None:
            return None

        try:
            from data_quality import validate_ohlcv
            ok, issues = validate_ohlcv(df, sym_clean, source)
            if not ok:
                print(f"[DataQuality] {source} rejected for {sym_clean}: {issues[0]}")
                return None
        except Exception as exc:
            print(f"[DataQuality] {source} rejected for {sym_clean}: {type(exc).__name__}")
            return None
        out = df.apply(pd.to_numeric, errors="raise").copy()
        if out.index.tz is not None:
            out.index = out.index.tz_convert("Asia/Kolkata").tz_localize(None)
        return out

    def _top_up_today(df):
        """
        Extend an archive frame with any session it is missing.

        The archive holds completed daily bars, so a scan run today reads a
        frame ending at the previous close and scores it as the present.
        ADANIENT, 31 Aug 2026: the 10:57 scan re-derived every indicator from
        the 28 Aug bar, reprinted the same score of 60, and reported
        change_pct -0.02% while the stock was down 2.9% on its way to -9.7%.
        The scan added no information and did not say so.

        Topping up costs one Groww call on a cache hit and makes `close`,
        `change_pct` and every indicator describe the current session.
        """
        if df is None or df.empty:
            return df
        try:
            import datetime as _d
            if not groww_client.is_connected() or groww_client.data_forbidden():
                return df
            fresh = groww_client.get_daily_bars(sym_clean, days=7)
            if fresh is None or fresh.empty:
                return df
            fresh = fresh.copy()
            if fresh.index.tz is not None:
                fresh.index = fresh.index.tz_convert("Asia/Kolkata").tz_localize(None)
            # Never splice visibly different adjustment scales.
            overlap = df.index.intersection(fresh.index)
            completed = overlap[overlap < pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None).normalize()]
            if len(completed):
                delta = (fresh.loc[completed, "Close"] / df.loc[completed, "Close"] - 1).abs()
                if (delta > 0.01).any():
                    print(f"[DataQuality] refusing mixed price scales for {sym_clean}")
                    return None
            # Replace the mutable latest bar as well as append new sessions.
            add = fresh[fresh.index >= df.index[-1]]
            if add.empty:
                return df
            out = pd.concat([df, add[df.columns]])
            return out[~out.index.duplicated(keep="last")].sort_index()
        except Exception:
            return df

    # ── 0. Persistent archive — survives restarts, no network ────────────────
    archived = entry.get("daily") if entry else None
    if archived is None:
        archived = _persist_read(sym_clean)
    daily = _accept(_top_up_today(archived), "cache")
    if daily is not None:
        weekly = _make_weekly(daily)
        src = "cache"

    # ── 1. Groww — primary source (subscribed, ~0.4s/stock) ───────────────────
    if daily is None:
        try:
            if groww_client.is_connected() and not groww_client.data_forbidden():
                daily = _accept(groww_client.get_daily_bars(sym_clean, days=550), "groww")
                if daily is not None:
                    weekly = _make_weekly(daily)
                    src = "groww"
                    _persist_write(sym_clean, daily)
        except Exception:
            pass

    # ── 2. Dhan v2 — only if DHAN_* credentials are configured ────────────────
    if daily is None:
        try:
            import dhan_client
            if dhan_client.is_configured() and not dhan_client.data_forbidden():
                daily = _accept(dhan_client.get_daily_bars(sym_clean, days=550), "dhan")
                if daily is not None:
                    weekly = _make_weekly(daily)
                    src = "dhan"
        except Exception:
            pass

    # ── 3. nselib — NSE direct, reliable, no Yahoo rate limits ───────────────
    if daily is None:
        daily = _accept(_nselib_ohlcv(sym_clean), "nselib")
        if daily is not None:
            weekly = _make_weekly(daily)
            src = "nselib"

    # ── 4. yfinance fallback (only if all above missed) ───────────────────────
    if daily is None:
        try:
            d = _yf_history(symbol, period="2y", interval="1d", auto_adjust=True)
            if d is not None:
                d = d.dropna(subset=["Close"])
                if len(d) >= 50:
                    daily = _accept(d, "yfinance")
        except Exception:
            pass
        try:
            if daily is not None:
                weekly = _make_weekly(daily)
        except Exception:
            pass

    with _ohlcv_lock:
        _ohlcv_cache[symbol] = {"daily": daily, "weekly": weekly, "ts": now, "src": src}
    return daily, weekly


def fetch_swing(symbol: str,
                nifty_close: pd.Series | None = None, *, bulk_fundamentals: bool = False) -> dict | None:
    """Fast path: daily + weekly OHLCV (cached 6h) + full indicator suite. No backtest."""
    try:
        df, df_weekly = _get_ohlcv(symbol)
        if df is None:
            return None

        # Keep mutable bars in the data cache but exclude them from swing scores.
        from screening_policy import completed_history
        df, df_weekly = completed_history(df)
        if df.empty:
            return None

        result = compute_signals(df)
        if result is None:
            return None
        result = compute_enhanced(df, result, is_intraday=False,
                                  weekly_df=df_weekly, nifty_close=nifty_close)
        result["_df"] = df
        name = symbol.replace(".NS", "").replace(".BO", "")
        result["symbol"]      = name
        result["full_symbol"] = symbol
        result["score_version"] = SCORE_VERSION
        # Detect actual source from cache entry
        with _ohlcv_lock:
            _src = _ohlcv_cache.get(symbol, {}).get("src", "nselib")
        result["data_source"] = _src
        result.setdefault("patterns", [])

        # Which session this score actually describes, and how far behind it is.
        # Every price on the row -- close, change_pct, stops, targets -- belongs
        # to this date. Without it a score from a previous session renders
        # identically to one computed minutes ago, which is how a 28 Aug reading
        # of ADANIENT was shown as "GOOD SETUP / ENTRY, -0.02%" while the stock
        # was down 2.9% intraday, and how RELINFRA's June-2025 close sat next to
        # a live price fifteen months newer.
        try:
            import datetime as _d
            _bar = df.index[-1].date()
            result["bar_date"] = str(_bar)
            result["bar_age_days"] = (_d.date.today() - _bar).days
        except Exception:
            pass

        # Liquidity profile — attached to EVERY result, not just AI-analysed
        # ones. A high score on a stock you cannot exit is worse than no
        # signal at all, and nothing else in the pipeline checks this.
        try:
            import liquidity
            liq = liquidity.assess(df)
            result["liq_turnover"]        = round(liq["median_turnover"]) if liq["median_turnover"] else None
            result["liq_trading_days_pct"]= round(liq["trading_days_pct"]) if liq["trading_days_pct"] else None
            result["liq_max_position"]    = liq["max_position_value"]
            result["liq_tradeable"]       = liq["tradeable"]
            result["liq_reasons"]         = liq["reasons"]
        except Exception:
            result["liq_tradeable"] = None

        # Attach fundamentals (cached 6h — runs in same thread, no extra latency)
        fund = get_fundamentals(symbol, bulk_only=True) if bulk_fundamentals else get_fundamentals(symbol)
        result['fundamentals_mode'] = 'bulk_only' if bulk_fundamentals else 'full_enrichment'
        result['fundamentals_available'] = bool(fund)
        if fund:
            result["fundamentals"] = fund
            # Hoist key fields to top level for easy table access
            result["fund_score"]      = fund.get("fund_score")
            result["pe"]              = fund.get("pe")
            result["pb"]              = fund.get("pb")
            result["roe"]             = fund.get("roe")
            result["debt_equity"]     = fund.get("debt_equity")
            result["revenue_growth"]  = fund.get("revenue_growth")
            result["sector"]          = fund.get("sector")
            result["industry"]        = fund.get("industry")
            result["net_margin"]      = fund.get("net_margin")
            result["earnings_growth"] = fund.get("earnings_growth")
            result["debt_equity_unit"] = "ratio"

        # ML signal probability (non-blocking — returns None if model not trained)
        try:
            import ml_signal
            # Rename columns for ml_signal (expects lowercase)
            df_ml = df.rename(columns={"Open":"open","High":"high","Low":"low",
                                        "Close":"close","Volume":"volume"})
            ml_prob = ml_signal.score_signal(df_ml)
            if ml_prob is not None:
                result["ml_prob"] = ml_prob
                result["ml_verdict"] = (
                    "ML: Strong"  if ml_prob >= 0.65 else
                    "ML: Likely"  if ml_prob >= 0.55 else
                    "ML: Neutral" if ml_prob >= 0.45 else
                    "ML: Weak"
                )
        except Exception:
            pass

        return result
    except Exception:
        return None


def fetch_intraday(symbol: str) -> dict | None:
    name = symbol.replace(".NS", "")

    # Groww first — the only subscribed intraday source. Then Dhan, then yfinance.
    if groww_client.is_connected() and not groww_client.data_forbidden():
        df = groww_client.get_intraday_candles(name, interval_minutes=15)
        if df is not None and len(df) >= 10:
            result = compute_signals(df)
            if result:
                result = compute_enhanced(df, result, is_intraday=True)
                result["symbol"]      = name
                result["full_symbol"] = symbol
                result["data_source"] = "groww"
                result.setdefault("patterns", [])
                return result

    try:
        import dhan_client
        if dhan_client.is_configured() and not dhan_client.data_forbidden():
            df = dhan_client.get_intraday_candles(name, interval_minutes=15, days=5)
            if df is not None and len(df) >= 10:
                result = compute_signals(df)
                if result:
                    result = compute_enhanced(df, result, is_intraday=True)
                    result["symbol"]      = name
                    result["full_symbol"] = symbol
                    result["data_source"] = "dhan"
                    result.setdefault("patterns", [])
                    return result
    except Exception:
        pass

    if truedata_client.is_connected():
        df = truedata_client.get_intraday_candles(name, interval_minutes=15)
        if df is not None and len(df) >= 10:
            result = compute_signals(df)
            if result:
                result = compute_enhanced(df, result, is_intraday=True)
                result["symbol"]      = name
                result["full_symbol"] = symbol
                result["data_source"] = "truedata_live"
                result.setdefault("patterns", [])
                return result

    if groww_client.is_connected():
        df = groww_client.get_intraday_candles(name, interval_minutes=15)
        if df is not None and len(df) >= 10:
            result = compute_signals(df)
            if result:
                result = compute_enhanced(df, result, is_intraday=True)
                result["symbol"]      = name
                result["full_symbol"] = symbol
                result["data_source"] = "groww_live"
                result.setdefault("patterns", [])
                return result

    try:
        df = _yf_history(symbol, period="5d", interval="15m", auto_adjust=True)
        if df is None or len(df) < 10:
            return None
        result = compute_signals(df)
        if result is None:
            return None
        result = compute_enhanced(df, result, is_intraday=True)
        result["symbol"]      = name
        result["full_symbol"] = symbol
        result["data_source"] = "yfinance"
        result.setdefault("patterns", [])
        return result
    except Exception:
        return None


def _enrich_with_ai(stock: dict, market: dict) -> dict:
    fund = stock.get("fundamentals", {})
    ai = ai_analyst.analyze_stock(stock["symbol"], stock, fund, market)
    stock.update(ai)
    return stock


def _run_backtest_phase(results: list[dict]) -> list[dict]:
    """Phase 3: run backtester on each stock using the df carried from Phase 1."""
    def _bt(r):
        df = r.pop("_df", None)
        if df is not None:
            bt = backtest_win_rate(df)
            r.update(bt)
        return r

    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = {ex.submit(_bt, r): r for r in results}
        for f in as_completed(futures):
            try:
                f.result(timeout=15)
            except Exception:
                pass
    return results


def scan_universe(universe: str = "nifty50", max_workers: int = 10) -> list[dict]:
    """
    Phase 1: parallel technical scan with per-stock timeout.
    Fetches Nifty + NSE data once, passes to all stocks for RS and delivery %.
    Watchlist symbols are always included regardless of universe.
    """
    from market_regime import get_regime
    from nse_data import get_fii_dii, get_delivery_pct, get_participant_oi, get_bulk_block_deals

    market  = get_market_context()
    regime  = get_regime()
    symbols = list(get_all_symbols(universe))

    # Pre-fetch all fundamentals in one bulk call (TradingView screener).
    # This populates the cache so per-stock calls are instant lookups.
    print("[Scan] Pre-fetching fundamentals via TradingView screener...")
    try:
        import tv_fundamentals
        tv_fundamentals.warm_cache()
    except Exception as e:
        print(f"[Scan] TV fundamentals prefetch failed: {e} — using yfinance fallback")

    # Always inject watchlist symbols even if not in the chosen universe
    watchlist_raw = load_watchlist()
    watchlist_ns  = {f"{s}.NS" for s in watchlist_raw}
    for ws in watchlist_ns:
        if ws not in symbols:
            symbols.append(ws)

    # nselib (NSE direct API) is the primary data source — no Yahoo rate limits.
    # Use 8 workers for all universes; all_nse uses 10 to compensate for larger symbol count.
    if universe == "all_nse":
        max_workers = 10
    else:
        max_workers = 8
    results = []

    # Fetch Nifty close once — shared across all stocks for RS calculation
    print("[Scan] Fetching Nifty baseline + NSE data...")
    from market_context import get_nifty_daily
    _nifty_df   = get_nifty_daily()
    if _nifty_df is not None:
        from screening_policy import completed_history
        _nifty_df, _ = completed_history(_nifty_df)
    nifty_close = _nifty_df["Close"].tail(252) if _nifty_df is not None else None
    fii_dii        = get_fii_dii()
    delivery_data  = get_delivery_pct()
    participant_oi = get_participant_oi()
    bulk_deals     = get_bulk_block_deals()

    # NSE live market intelligence: OI spurts, F&O ban, sector performance
    print("[Scan] Fetching NSE live signals (OI spurts, F&O ban, sector data)...")
    nse_intel = {}
    try:
        import nse_live_signals
        nse_intel = nse_live_signals.get_market_intelligence()
        oi_syms   = nse_intel.get("oi_spurt_symbols", set())
        ban_syms  = nse_intel.get("fno_ban_symbols", set())
        sectors   = nse_intel.get("sector_performance", {})
        indices   = nse_intel.get("index_snapshot", {})
        if oi_syms:
            print(f"[Scan] OI spurts: {len(oi_syms)} stocks — {sorted(list(oi_syms))[:8]}")
        if ban_syms:
            print(f"[Scan] F&O ban: {len(ban_syms)} stocks")
    except Exception as e:
        print(f"[Scan] NSE live signals failed: {e}")
        oi_syms = ban_syms = set()
        sectors = indices = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_swing, sym, nifty_close, bulk_fundamentals=(universe == 'all_nse')): sym for sym in symbols}
        for future in as_completed(futures):
            try:
                r = future.result(timeout=_STOCK_TIMEOUT)
                if r:
                    sym_name = r.get("symbol", "")
                    # Attach delivery %
                    if sym_name in delivery_data:
                        r["delivery_pct"] = delivery_data[sym_name]
                    # OI spurt flag (unusual open interest buildup = smart money)
                    r["oi_spurt"] = sym_name.upper() in oi_syms
                    # F&O ban flag (informational — don't exclude, just flag)
                    r["fno_ban"] = sym_name.upper() in ban_syms
                    results.append(r)
            except Exception:
                sym = futures[future]
                print(f"[Scan] Skipping {sym} — timeout or error")

    # Live LTP — prefer TrueData, fall back to Groww
    for r in results:
        r["live_price"] = None
    live = {}
    if truedata_client.is_connected():
        def _fetch_ltp_td():
            names = [r["symbol"] for r in results]
            live.update(truedata_client.get_ltp(names))
        t = threading.Thread(target=_fetch_ltp_td, daemon=True)
        t.start()
        t.join(timeout=25)
    elif groww_client.is_connected():
        def _fetch_ltp_groww():
            names = [r["symbol"] for r in results]
            for i in range(0, len(names), 49):
                live.update(groww_client.get_live_ltp(names[i:i + 49]))
        t = threading.Thread(target=_fetch_ltp_groww, daemon=True)
        t.start()
        t.join(timeout=20)
    if live:
        for r in results:
            ltp = live.get(r["symbol"])
            if ltp and ltp > 0:
                r["live_price"] = round(float(ltp), 2)

    # Attach market + regime + NSE intel, then re-score
    from technical_enhanced import compute_score100, compute_verdict, compute_data_confidence
    for r in results:
        r["market"] = market
        r["regime"] = regime
        if fii_dii:
            r["fii_dii"] = fii_dii
        if sectors:
            r["sector_performance"] = sectors
        if indices:
            r["index_snapshot"] = indices
        if participant_oi:
            r["participant_oi"] = participant_oi
        sym_up = r.get("symbol", "").upper()
        if sym_up in bulk_deals:
            r["bulk_deals"] = bulk_deals[sym_up]
        r["score100"]        = compute_score100(r)
        r["verdict"]         = compute_verdict(r["score100"])
        r["data_confidence"] = compute_data_confidence(r)

    # Tag watchlist stocks so the UI can highlight them
    for r in results:
        sym_clean = r.get("symbol", "").replace(".NS", "").upper()
        r["on_watchlist"] = sym_clean in watchlist_raw

    from screening_policy import assess, completed_history, load_risks
    risks = load_risks()
    reference = None
    if _nifty_df is not None and not _nifty_df.empty:
        completed_nifty, _ = completed_history(_nifty_df)
        if not completed_nifty.empty:
            reference = str(completed_nifty.index[-1].date())
    for r in results:
        r.update(assess(r, reference, risks))
        # Research candidates are not trade instructions.
        r["setup_status"] = "WATCH" if r["screen_eligible"] else "AVOID"
    results.sort(key=lambda x: (not x["screen_eligible"], -x.get("score100", 0), x.get("symbol", "")))
    return results


def enrich_with_ai(results: list[dict]) -> list[dict]:
    """Phase 3: Claude AI on top-N stocks — sequential with per-stock timeout."""
    ai_enabled = bool(
        getattr(config, "ANTHROPIC_API_KEY", "") and
        config.ANTHROPIC_API_KEY != "YOUR_ANTHROPIC_API_KEY"
    )
    if not ai_enabled or not results:
        return results

    market = results[0].get("market", {})
    top_n  = min(getattr(config, "AI_ANALYSE_TOP_N", 3), len(results))

    # Run sequentially — parallel AI calls hit Anthropic rate limits and timeouts.
    # Each stock gets a 45s budget; skip and continue if it fails.
    for i in range(top_n):
        sym = results[i].get("symbol", "?")
        try:
            with ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(_enrich_with_ai, results[i], market)
                results[i] = fut.result(timeout=45)
        except Exception as e:
            print(f"[AI] Skipping {sym} — {e}")

    ai_done = [r for r in results if r.get("ai_verdict")]
    rest    = [r for r in results if not r.get("ai_verdict")]
    ai_done.sort(key=lambda x: (x.get("ai_confidence") or 0, x.get("score100", 0)), reverse=True)
    return ai_done + rest


def enrich_with_backtest(results: list[dict]) -> list[dict]:
    """Phase 3: parallel backtest win-rate on all stocks."""
    return _run_backtest_phase(results)


def scan_intraday(universe: str = "nifty50") -> list[dict]:
    market  = get_market_context()
    symbols = get_all_symbols(universe)
    results = []

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_intraday, sym): sym for sym in symbols}
        for future in as_completed(futures):
            try:
                r = future.result(timeout=_STOCK_TIMEOUT)
                if r:
                    r["market"] = market
                    results.append(r)
            except Exception:
                pass

    results.sort(key=lambda x: x.get("score100", x.get("score", 0)), reverse=True)
    return results
