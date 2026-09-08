from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pathlib import Path
import os
import sys
import uvicorn
import threading
import time
import datetime
import webbrowser

import math
import numpy as np
import config
import groww_client
import truedata_client
import db
from scanner import scan_universe, scan_intraday, enrich_with_ai, enrich_with_backtest
from notifier import send_telegram, format_alert
from market_context import get_market_context


def _clean(obj):
    """Recursively replace NaN/inf (numpy or plain float) with None for JSON safety."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, (np.floating, np.integer)):
        v = float(obj) if isinstance(obj, np.floating) else int(obj)
        return None if (isinstance(v, float) and (math.isnan(v) or math.isinf(v))) else v
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

app = FastAPI(title="NSE Stock Signal Dashboard")
from trading_research import router as trading_research_router
app.include_router(trading_research_router)

_cache = {
    "swing":    {"data": [], "last_updated": None, "loading": False, "_ts": 0},
    "intraday": {"data": [], "last_updated": None, "loading": False, "_ts": 0},
}
_groww_connected    = False
_last_alert_sent    = None
_scan_lock          = threading.Lock()   # prevents overlapping scan threads
_ml_train_status    = {"running": False, "error": None, "metrics": None}
_exp_lock           = threading.Lock()   # one expectancy build at a time


# ── Groww init ──────────────────────────────────────────────
def _init_groww():
    global _groww_connected
    _groww_connected = groww_client.init_groww(config.GROWW_API_KEY, config.GROWW_API_SECRET)


# ── Scanner workers ─────────────────────────────────────────
def _merge_into_cache(new_results: list[dict], universe: str) -> list[dict]:
    """
    Fold a narrow scan into the broader set already on screen.

    A nifty50 refresh means fifty rows are newer. It does not mean the other
    2,323 stocks ceased to exist — but `_cache["swing"]["data"] = results`
    replaced the lot, so the 11:00 auto-scan turned a 2,373-stock all_india
    view into 50 rows, of which 8 cleared 60. The dashboard showed eight
    stocks and gave no hint that anything had been dropped.

    Same fault as the one fixed in `db.load_last_scan`, in a second place:
    breadth was being discarded in favour of recency without anyone deciding
    that was the trade. Newer rows win per symbol; unrefreshed rows stay.
    """
    prev = _cache["swing"].get("data") or []
    if not prev or len(new_results) >= len(prev):
        return new_results

    # Never resurrect rows from a set old enough to be a different market.
    ts = _cache["swing"].get("_ts")
    if ts and (time.time() - ts) > 12 * 3600:
        return new_results

    merged = {r["symbol"]: r for r in prev if r.get("symbol")}
    for r in new_results:
        if r.get("symbol"):
            merged[r["symbol"]] = r
    out = sorted(merged.values(), key=lambda r: r.get("score100") or 0, reverse=True)
    print(f"[Scan] merged {len(new_results)} {universe} rows into "
          f"{len(prev)} held — {len(out)} on screen")
    return out


def _run_swing_scan(universe: str):
    if not _scan_lock.acquire(blocking=False):
        print("[Scan] Already running — skipping")
        return
    try:
        _cache["swing"]["loading"] = True
        print(f"[Scan] Phase 1 starting — {universe}...")
        results = scan_universe(universe)
        # `results` stays the scan's own output — crossings, backtest, AI and
        # scan_log must all describe what was actually scanned. `on_screen` is
        # the merged view, and only that goes to the cache and the dashboard.
        on_screen = _merge_into_cache(results, universe)
        _cache["swing"]["data"] = on_screen
        _cache["swing"]["last_updated"] = datetime.datetime.now().strftime("%I:%M %p")
        _cache["swing"]["loading"] = False
        _cache["swing"]["_ts"] = time.time()
        print(f"[Scan] Phase 1 done — {len(results)} stocks. Top: {results[0]['symbol'] if results else 'none'}")
        # Save the merged view so a restart reopens on everything, not just the
        # last narrow refresh. scan_log below still records only what was scanned.
        db.save_full_scan(on_screen, universe=universe, scan_type="swing")

        # Crossings run here, not after Phase 3. The alert is about a stock
        # entering the bar, and that fact is known the moment scoring ends —
        # waiting for backtest and AI would delay it by the length of both.
        try:
            import crossings
            cx = crossings.check_and_alert(results)
            _cache["crossings"] = cx
            print(f"[Crossings] {len(cx['crossed'])} crossed {cx['bar']:.0f}+, "
                  f"{cx['alerted']} newly alerted, {len(cx['new'])} first sightings")
        except Exception as e:
            print(f"[Crossings] failed: {e}")

        # Phases 2 and 3 must re-merge for the same reason Phase 1 does. Writing
        # the narrow `results` straight back would silently undo the merge two
        # lines after making it.
        print("[Scan] Phase 2 — backtest...")
        results = enrich_with_backtest(results)
        _cache["swing"]["data"] = _merge_into_cache(results, universe)
        _cache["swing"]["last_updated"] = datetime.datetime.now().strftime("%I:%M %p")
        print("[Scan] Phase 2 done.")

        print("[Scan] Phase 3 — AI analysis...")
        results = enrich_with_ai(results)
        _cache["swing"]["data"] = _merge_into_cache(results, universe)
        _cache["swing"]["last_updated"] = datetime.datetime.now().strftime("%I:%M %p")
        print("[Scan] Phase 3 done. Scan complete.")
        _cache["swing"]["_ts"] = time.time()   # timestamp for VT cache reuse
        db.save_scan_results(results, universe=universe, scan_type="swing")
    except Exception as e:
        print(f"[Scan] ERROR: {e}")
        import traceback; traceback.print_exc()
    finally:
        _cache["swing"]["loading"] = False
        _scan_lock.release()


def _run_intraday_scan(universe: str):
    _cache["intraday"]["loading"] = True
    try:
        results = scan_intraday(universe)
        _cache["intraday"]["data"] = results
        _cache["intraday"]["last_updated"] = datetime.datetime.now().strftime("%I:%M %p")
        db.save_scan_results(results, universe=universe, scan_type="intraday")
    finally:
        _cache["intraday"]["loading"] = False


# ── Live LTP update loop (every 30s) ────────────────────────
def _live_ltp_loop():
    """
    Continuously refresh live_price for every stock in the swing cache.
    Priority: TrueData WebSocket → TrueData REST → Groww → skip.
    Runs every 30 seconds.
    """
    while True:
        try:
            data = _cache["swing"]["data"]
            if data:
                names = [r["symbol"] for r in data if r.get("symbol")]
                live  = {}

                # 1. TrueData WebSocket (fastest — live tick stream)
                if truedata_client.is_connected():
                    for i in range(0, len(names), 50):
                        live.update(truedata_client.get_ltp(names[i:i + 50]))

                # 2. TrueData REST (no WebSocket needed — works with active subscription)
                if not live:
                    try:
                        td_rest = truedata_client.get_ltp_rest(names[:50])  # cap for rate limit
                        live.update(td_rest)
                    except Exception:
                        pass

                # 3. Groww fallback
                if not live and groww_client.is_connected():
                    for i in range(0, len(names), 49):
                        live.update(groww_client.get_live_ltp(names[i:i + 49]))

                if live:
                    for r in data:
                        ltp = live.get(r["symbol"])
                        if ltp and ltp > 0:
                            r["live_price"] = round(float(ltp), 2)
        except Exception:
            pass   # never crash — silent fail, retry in 30s
        time.sleep(30)


# ── Token refresh at 6 AM IST ───────────────────────────────
def _token_refresh_loop():
    import datetime
    while True:
        now = datetime.datetime.now()
        # Wake up at 06:01 AM to refresh (1 min after Groww's reset)
        target = now.replace(hour=6, minute=1, second=0, microsecond=0)
        if now >= target:
            target += datetime.timedelta(days=1)
        wait_secs = (target - now).total_seconds()
        time.sleep(wait_secs)
        print("[Groww] 6 AM reset — refreshing token...")
        groww_client.ensure_connected()


# ── Auto-scan + WhatsApp alert scheduler ────────────────────
def _full_scan_loop():
    """
    One full-universe scan per day, after the close.

    Without this the crossing detector is decorative. `AUTO_SCAN_HOURS` runs
    `SCAN_UNIVERSE` four times a day for a fast intraday refresh, but a narrow
    universe cannot produce a crossing in a stock it never scores — and measured
    31 Aug 2026, none of MOREPENLAB / SHILPAMED / SUDEEPPHRM / ATHERENERG is in
    nifty50, only ATHERENERG is in nifty500, and all four are in all_nse. Every
    winner the user has named lived outside the scheduled universe.

    Runs post-close so the day's bars are final and it competes with nothing.
    """
    if not getattr(config, "FULL_SCAN_ENABLED", True):
        print("[FullScan] disabled by config")
        return
    target = getattr(config, "FULL_SCAN_AT", "18:30")
    universe = getattr(config, "FULL_SCAN_UNIVERSE", "all_india")
    print(f"[FullScan] scheduled daily at {target} IST ({universe})")
    last_run_date = None
    while True:
        now = datetime.datetime.now()
        if now.strftime("%H:%M") == target and last_run_date != now.date():
            last_run_date = now.date()
            if _scan_lock.locked() or _cache["swing"]["loading"]:
                print("[FullScan] a scan is already running — skipping today")
            else:
                print(f"[FullScan] starting {universe}...")
                try:
                    _run_swing_scan(universe)
                except Exception as e:
                    print(f"[FullScan] failed: {e}")
        time.sleep(55)


def _auto_scan_loop():
    global _last_alert_sent
    while True:
        now = datetime.datetime.now()
        ist_time = now.strftime("%H:%M")

        if ist_time in config.AUTO_SCAN_HOURS:
            universe = config.SCAN_UNIVERSE
            if not _scan_lock.locked() and not _cache["swing"]["loading"]:
                t = threading.Thread(target=_run_swing_scan, args=(universe,), daemon=True)
                t.start()
                t.join()

            # Send WhatsApp alert for strong signals
            alert_stocks = [s for s in _cache["swing"]["data"]
                            if s.get("score", 0) >= config.ALERT_MIN_SCORE]
            if alert_stocks and _last_alert_sent != ist_time:
                msg = format_alert(alert_stocks)
                ok = send_telegram(config.TELEGRAM_BOT_TOKEN, config.TELEGRAM_CHAT_ID, msg)
                if ok:
                    _last_alert_sent = ist_time
                    print(f"[Alert] Telegram sent at {ist_time} for {len(alert_stocks)} stocks")

        time.sleep(55)  # check every ~minute


# ── API Routes ───────────────────────────────────────────────
@app.get("/api/scan/swing")
async def api_swing_scan(universe: str = "nifty50", force: str = "false"):
    no_data = not _cache["swing"]["data"]
    trigger = force.lower() == "true" or no_data
    if trigger and not _scan_lock.locked() and not _cache["swing"]["loading"]:
        t = threading.Thread(target=_run_swing_scan, args=(universe,), daemon=True)
        t.start()
    scanning = _scan_lock.locked() or _cache["swing"]["loading"]
    safe_data = _clean([{k: v for k, v in r.items() if k != "_df"} for r in _cache["swing"]["data"]])
    return JSONResponse({
        "status": "scanning" if scanning else "ok",
        "last_updated": _cache["swing"]["last_updated"],
        "data": safe_data,
        "groww_connected": _groww_connected,
    })


@app.get("/api/scan/intraday")
async def api_intraday_scan(universe: str = "nifty50"):
    if not _cache["intraday"]["loading"]:
        t = threading.Thread(target=_run_intraday_scan, args=(universe,), daemon=True)
        t.start()
    return JSONResponse({
        "status": "scanning" if _cache["intraday"]["loading"] else "ok",
        "last_updated": _cache["intraday"]["last_updated"],
        "data": _clean(_cache["intraday"]["data"]),
        "groww_connected": _groww_connected,
    })


@app.get("/api/status")
async def api_status():
    return {
        "groww_connected": _groww_connected,
        "truedata_connected": truedata_client.is_connected(),
        "whatsapp_configured": (
            config.TELEGRAM_BOT_TOKEN != "YOUR_BOT_TOKEN" and
            config.TELEGRAM_CHAT_ID != "YOUR_CHAT_ID"
        ),
        "swing_loading": _cache["swing"]["loading"],
        "intraday_loading": _cache["intraday"]["loading"],
        "swing_updated": _cache["swing"]["last_updated"],
        "intraday_updated": _cache["intraday"]["last_updated"],
        "swing_count": len(_cache["swing"]["data"]),
        "intraday_count": len(_cache["intraday"]["data"]),
        "auto_scan_times": config.AUTO_SCAN_HOURS,
        "last_alert_sent": _last_alert_sent,
        "ai_configured": (
            hasattr(config, "ANTHROPIC_API_KEY") and
            bool(config.ANTHROPIC_API_KEY) and
            config.ANTHROPIC_API_KEY != "YOUR_ANTHROPIC_API_KEY"
        ),
    }


@app.get("/api/market")
async def api_market():
    """Current Nifty 50 trend and India VIX."""
    return JSONResponse(get_market_context())


@app.post("/api/alert/test")
async def api_test_alert():
    """Send a test Telegram message to verify setup."""
    msg = "✅ *NSE Signal Dashboard connected\\!* You will receive stock alerts here\\."
    ok = send_telegram(config.TELEGRAM_BOT_TOKEN, config.TELEGRAM_CHAT_ID, msg)
    return {"success": ok, "message": "Sent! Check your Telegram bot." if ok else "Failed — check TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in config.py"}


@app.get("/api/portfolio")
async def api_portfolio():
    """Fetch Groww holdings + positions + funds in one call."""
    if not _groww_connected:
        return JSONResponse({"error": "Groww not connected", "holdings": [], "positions": [], "funds": {}})
    holdings  = groww_client.get_holdings()
    positions = groww_client.get_positions()
    funds     = groww_client.get_funds()
    profile   = groww_client.get_user_profile()

    # Holdings API doesn't return live LTP — fetch it separately
    if holdings:
        symbols = [h["symbol"] for h in holdings]
        live = {}
        if truedata_client.is_connected():
            live = truedata_client.get_ltp(symbols)
        if not live:
            # Fetch in batches of 49 (Groww limit)
            for i in range(0, len(symbols), 49):
                live.update(groww_client.get_live_ltp(symbols[i:i+49]))
        for h in holdings:
            ltp = live.get(h["symbol"])
            if ltp and ltp > 0:
                h["ltp"] = round(float(ltp), 2)

    # Also enrich intraday positions with live LTP
    if positions:
        pos_symbols = [p["symbol"] for p in positions]
        pos_live = groww_client.get_live_ltp(pos_symbols) if pos_symbols else {}
        for p in positions:
            ltp = pos_live.get(p["symbol"])
            if ltp and ltp > 0:
                p["ltp"] = round(float(ltp), 2)
                p["pnl"] = round((ltp - p["avg_price"]) * p["quantity"], 2)

    # Compute portfolio summary
    total_invested = sum(h["avg_price"] * h["quantity"] for h in holdings)
    total_current  = sum(h["ltp"] * h["quantity"] for h in holdings if h.get("ltp", 0) > 0)
    total_pnl      = total_current - total_invested if total_current else 0

    return JSONResponse({
        "holdings":       holdings,
        "positions":      positions,
        "funds":          funds,
        "profile":        profile,
        "summary": {
            "total_holdings":  len(holdings),
            "total_invested":  round(total_invested, 2),
            "total_current":   round(total_current,  2),
            "total_pnl":       round(total_pnl,       2),
            "total_pnl_pct":   round((total_pnl / total_invested * 100) if total_invested and total_current else 0, 2),
        },
    })


@app.post("/api/portfolio/analyse")
async def api_portfolio_analyse():
    """Run signal scan on all currently held stocks."""
    if not _groww_connected:
        return JSONResponse({"error": "Groww not connected", "data": []})
    holdings = groww_client.get_holdings()
    if not holdings:
        return JSONResponse({"data": []})

    symbols = [f"{h['symbol']}.NS" for h in holdings]
    nifty_close = None
    try:
        from market_context import get_nifty_daily
        _nd = get_nifty_daily()
        if _nd is not None:
            nifty_close = _nd["Close"].tail(126)
    except Exception:
        pass

    from concurrent.futures import ThreadPoolExecutor, as_completed
    from scanner import fetch_swing
    results = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(fetch_swing, sym, nifty_close): sym for sym in symbols}
        for f in as_completed(futures):
            try:
                r = f.result(timeout=25)
                if r:
                    r.pop("_df", None)
                    # Attach holding info (avg price, qty, P&L)
                    h = next((x for x in holdings if x["symbol"] == r["symbol"]), {})
                    if h:
                        r["holding_qty"]   = h["quantity"]
                        r["holding_avg"]   = h["avg_price"]
                        r["holding_ltp"]   = h["ltp"] or r.get("close")
                        ltp = r["holding_ltp"] or 0
                        invested = h["avg_price"] * h["quantity"]
                        current  = ltp * h["quantity"]
                        r["holding_pnl"]     = round(current - invested, 2)
                        r["holding_pnl_pct"] = round(((ltp - h["avg_price"]) / h["avg_price"] * 100) if h["avg_price"] else 0, 2)
                    results.append(r)
            except Exception:
                pass

    results.sort(key=lambda x: x.get("score100", 50), reverse=True)

    # Run AI on top 5 holdings (they're stocks the user owns — all worth analysing)
    ai_enabled = bool(
        getattr(config, "ANTHROPIC_API_KEY", "") and
        config.ANTHROPIC_API_KEY != "YOUR_ANTHROPIC_API_KEY"
    )
    if ai_enabled and results:
        from scanner import _enrich_with_ai
        market = get_market_context()
        top_n  = min(10, len(results))
        for i in range(top_n):
            try:
                results[i] = _enrich_with_ai(results[i], market)
                if i < top_n - 1:
                    time.sleep(1.2)
            except Exception:
                pass

    return JSONResponse({"data": _clean(results)})


@app.get("/api/history")
async def api_history(symbol: str = "", scan_type: str = "swing", days: int = 7, limit: int = 200):
    rows = db.get_history(limit=limit, symbol=symbol, scan_type=scan_type, days=days)
    summary = db.get_scan_summary(days=days)
    return JSONResponse({"data": rows, "summary": summary})


@app.get("/api/strategy/lab")
async def api_strategy_lab(universe: str = "nifty50", force: str = "false"):
    """
    Run the 8-strategy walk-forward backtest on the given universe.
    Uses cached 10-year OHLCV (downloads on first call, ~2-3 min).
    """
    import data_manager
    import strategy_lab
    from stocks_nse import get_all_symbols

    symbols = get_all_symbols(universe)
    yf_symbols = [s.replace(".NS", "").replace(".BO", "") for s in symbols]
    ns_symbols = [f"{s}.NS" for s in yf_symbols]

    # Refresh stale data only when force=true
    refresh = force.lower() == "true"

    data = {}
    for sym in ns_symbols:
        df = data_manager.load(sym, force_refresh=refresh)
        if df is not None:
            data[sym.replace(".NS", "")] = df

    if not data:
        return JSONResponse({"error": "No historical data available. Try again in a moment."})

    print(f"[StratLab] Running strategies on {len(data)} symbols...")
    results = strategy_lab.run_strategy_lab(list(data.keys()), data=data)
    return JSONResponse(_clean(results))


@app.get("/api/strategy/ml/status")
async def api_ml_status():
    """Check if ML model is trained and return its metrics."""
    import ml_signal
    meta = ml_signal.get_model_meta("default")
    if not meta:
        return JSONResponse({"trained": False})
    return JSONResponse({"trained": True, **_clean(meta)})


@app.post("/api/strategy/ml/train")
async def api_ml_train(universe: str = "nifty50"):
    """
    Train LightGBM on 10 years of NSE data with 84 features (24 technical + 60 strategies).
    Phase 0: Strategy researcher fetches new strategies from web (Investopedia etc.)
    Phase 1: Download 10-year OHLCV via yfinance → SQLite cache
    Phase 2: LightGBM training — target = T1+2% hit before SL-1% within 15 days
    Runs in background — poll /api/strategy/ml/status for completion.
    """
    import data_manager
    import ml_signal
    from stocks_nse import get_all_symbols

    _ml_train_status["running"] = True
    _ml_train_status["error"]   = None

    def _do_train():
        try:
            # Phase 0: Research new strategies from the web (skips if already done)
            print("[ML] Phase 0: Running strategy researcher...")
            try:
                import strategy_researcher
                strategy_researcher.run_research()
            except Exception as e:
                print(f"[ML] Researcher skipped: {e}")

            # Phase 1: Download / load 10-year OHLCV
            symbols = get_all_symbols(universe)
            ns_symbols = [f"{s.replace('.NS','').replace('.BO','')}.NS" for s in symbols]
            print(f"[ML] Phase 1: Loading 10-year OHLCV for {len(ns_symbols)} symbols...")
            data = {}
            for sym in ns_symbols:
                df = data_manager.load(sym)
                if df is not None:
                    data[sym.replace(".NS", "")] = df
            print(f"[ML] Phase 2: Training LightGBM on {len(data)} symbols (84 features)...")
            metrics = ml_signal.train(list(data.keys()), data=data)
            _ml_train_status["metrics"] = metrics
        except Exception as e:
            print(f"[ML] Training error: {e}")
            import traceback; traceback.print_exc()
            _ml_train_status["error"] = str(e)
        finally:
            _ml_train_status["running"] = False

    threading.Thread(target=_do_train, daemon=True).start()
    return JSONResponse({"status": "training_started",
                         "message": "ML training running in background. Check /api/strategy/ml/status."})


@app.get("/api/strategy/ml/train/status")
async def api_ml_train_status():
    return JSONResponse({
        "running": _ml_train_status.get("running", False),
        "error":   _ml_train_status.get("error"),
        "metrics": _clean(_ml_train_status.get("metrics")),
    })


@app.post("/api/strategy/research")
async def api_strategy_research(request: Request):
    """
    Claude AI analysis of strategy lab results.
    POST body: {"lab_results": {...}, "ml_metrics": {...}}
    """
    import research_agent
    body = await request.json()
    lab  = body.get("lab_results", {})
    ml   = body.get("ml_metrics")
    if not lab:
        return JSONResponse({"error": "lab_results required"})
    result = research_agent.analyse_strategies(lab, ml)
    return JSONResponse(_clean(result))


@app.get("/api/data/cache/stats")
async def api_cache_stats():
    import data_manager
    return JSONResponse(data_manager.cache_stats())


@app.get("/api/strategy/expectancy")
async def api_expectancy(universe: str = "nifty50", force: str = "false"):
    """
    Regime x score-bucket expectancy table.
    First call triggers background computation (~5-15 min).
    Returns cached results instantly on subsequent calls.
    Use force=true to recompute.
    """
    import data_manager
    import regime_expectancy
    from stocks_nse import get_all_symbols

    # Return status if currently running
    status = regime_expectancy.get_status()
    if status["running"]:
        return JSONResponse({
            "status": "running",
            "progress": status["progress"],
            "total": status["total"],
            "symbol": status["symbol"],
        })

    # Check cache first (unless force)
    if force.lower() != "true":
        cached = regime_expectancy._load_cache("default")
        if cached:
            return JSONResponse({"status": "ok", **_clean(cached)})

    # Nothing cached yet (or force=true) — start background build
    if not _exp_lock.acquire(blocking=False):
        return JSONResponse({"status": "running", "message": "Already building"})

    def _do_build():
        try:
            symbols = get_all_symbols(universe)
            ns_syms = [f"{s.replace('.NS','').replace('.BO','')}.NS" for s in symbols]
            data    = {}
            for sym in ns_syms:
                df = data_manager.load(sym)
                if df is not None:
                    data[sym.replace(".NS", "")] = df

            nifty_df = data_manager.load("^NSEI")
            if nifty_df is None:
                raise RuntimeError("Could not load Nifty history from data_manager")

            regime_expectancy.build_expectancy_table(
                list(data.keys()), data, nifty_df,
                cache_key="default",
                force=force.lower() == "true",
            )
        except Exception as e:
            print(f"[Expectancy] Build failed: {e}")
        finally:
            _exp_lock.release()

    threading.Thread(target=_do_build, daemon=True).start()
    return JSONResponse({
        "status":  "building",
        "message": "Expectancy table is being built. This takes 5–15 minutes. "
                   "Poll /api/strategy/expectancy/status for progress.",
    })


@app.get("/api/strategy/expectancy/status")
async def api_expectancy_status():
    """Real-time progress for the background expectancy build."""
    import regime_expectancy
    s = regime_expectancy.get_status()
    pct = round(s["progress"] / s["total"] * 100) if s.get("total") else 0
    return JSONResponse({
        "running":  s["running"],
        "progress": s["progress"],
        "total":    s["total"],
        "symbol":   s["symbol"],
        "pct":      pct,
        "error":    s["error"],
    })


@app.get("/api/watchlist")
async def api_get_watchlist():
    from stocks_nse import load_watchlist
    return JSONResponse({"watchlist": load_watchlist()})


@app.post("/api/watchlist/add")
async def api_add_watchlist(symbol: str):
    from stocks_nse import add_to_watchlist
    import yfinance as yf
    s = symbol.upper().replace(".NS", "").strip()
    if not s:
        return JSONResponse({"error": "Empty symbol"}, status_code=400)
    # Quick validation — check yfinance has data for this symbol
    try:
        info = yf.Ticker(f"{s}.NS").fast_info
        if not getattr(info, "last_price", None):
            return JSONResponse({"error": f"{s} not found on NSE"}, status_code=404)
    except Exception:
        return JSONResponse({"error": f"Could not validate {s}"}, status_code=400)
    syms = add_to_watchlist(s)
    return JSONResponse({"watchlist": syms})


@app.delete("/api/watchlist/remove")
async def api_remove_watchlist(symbol: str):
    from stocks_nse import remove_from_watchlist
    syms = remove_from_watchlist(symbol)
    return JSONResponse({"watchlist": syms})


@app.get("/api/ltp")
async def api_ltp():
    """Lightweight: current live_price for every stock in the swing cache.
    Frontend polls this every 30s to update prices without a full re-render."""
    return JSONResponse({
        r["symbol"]: r.get("live_price")
        for r in _cache["swing"]["data"]
        if r.get("symbol")
    })


@app.get("/api/virtual-portfolio")
async def api_vt_dashboard():
    import virtual_trader as vt
    data = vt.get_dashboard_data()
    return JSONResponse(_clean(data))


@app.get("/api/virtual-portfolio/status")
async def api_vt_status():
    import virtual_trader as vt
    return JSONResponse(_clean(vt.agent_status()))


@app.post("/api/virtual-portfolio/start")
async def api_vt_start(universe: str = "nifty500"):
    import virtual_trader as vt
    vt.start_agent(universe)
    return JSONResponse({"status": "started", "universe": universe})


@app.post("/api/virtual-portfolio/stop")
async def api_vt_stop():
    import virtual_trader as vt
    vt.stop_agent()
    return JSONResponse({"status": "stopped"})


@app.post("/api/virtual-portfolio/reset")
async def api_vt_reset():
    import virtual_trader as vt
    vt.stop_agent()
    time.sleep(1.5)
    vt.get_portfolio().reset()
    return JSONResponse({"status": "reset"})


@app.get("/api/sectors")
async def api_sectors():
    """Return cached sector intelligence report. Triggers background scan on first call."""
    import sector_intel
    result = sector_intel.get_sector_report(force=False)
    return JSONResponse(_clean(result))


@app.get("/api/sectors/scan")
async def api_sectors_scan():
    """Force a fresh sector intelligence scan."""
    import sector_intel
    result = sector_intel.get_sector_report(force=True)
    return JSONResponse(_clean(result))


@app.get("/api/sectors/status")
async def api_sectors_status():
    import sector_intel
    return JSONResponse(sector_intel.get_sector_status())


# ── Crossings + pullback quality ────────────────────────────
@app.get("/api/crossings")
async def api_crossings(bar: float = 60.0, lookback: int = 21, live: str = "true"):
    """
    Stocks that moved INTO the bar since the last time they were scored.

    `live=true` compares the in-memory scan against the logged history, so a
    scan still in Phase 2/3 already reports its crossings. `live=false` reads
    entirely from the DB — use it to inspect what a past scan produced.
    """
    import crossings
    results = _cache["swing"]["data"] if live.lower() == "true" else None
    return JSONResponse(_clean(
        crossings.detect(bar=bar, lookback_days=lookback, results=results or None)))


@app.get("/api/crossings/history")
async def api_crossings_history(days: int = 90, symbol: str = ""):
    import crossings
    return JSONResponse({"rows": _clean(crossings.history(days=days, symbol=symbol))})


@app.get("/api/pullback")
async def api_pullback(symbols: str = ""):
    """
    Pullback quality on positions already held. Defaults to the watchlist.

    Deliberately separate from the scan: the scan answers "should I enter",
    this answers "should I stay", and conflating them is what left every
    drawdown looking like the same red number.
    """
    import pullback
    if symbols.strip():
        syms = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    else:
        from stocks_nse import load_watchlist
        syms = load_watchlist()
    if not syms:
        return JSONResponse({"rows": [], "note": "watchlist is empty"})
    return JSONResponse({
        "rows": _clean(pullback.assess_many(syms[:40])),
        "note": "UNVALIDATED — describes the dip, does not predict the outcome",
    })


_scout_lock = threading.Lock()
_scout_status = {"running": False, "error": None}


@app.get("/api/scout")
async def api_scout():
    """Cached small-cap accumulation screen."""
    import smallcap_scout
    return JSONResponse({
        "rows": _clean(smallcap_scout.load()),
        "running": _scout_status["running"],
        "error": _scout_status["error"],
    })


@app.get("/api/scout/scan")
async def api_scout_scan(limit: int = 0):
    """Kick off a fresh screen in the background."""
    import smallcap_scout

    if not _scout_lock.acquire(blocking=False):
        return JSONResponse({"status": "running", "message": "Screen already in progress"})

    def _run():
        _scout_status.update({"running": True, "error": None})
        try:
            smallcap_scout.scan(limit=limit or None)
        except Exception as e:
            _scout_status["error"] = str(e)
            print(f"[Scout] failed: {e}")
        finally:
            _scout_status["running"] = False
            _scout_lock.release()

    threading.Thread(target=_run, daemon=True).start()
    return JSONResponse({"status": "started",
                         "message": "Screening ~2,700 names — 20-40 minutes."})


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "templates" / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


# ── Startup ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  NSE Stock Signal Dashboard")
    print("="*60)

    db.init_db()
    _init_groww()
    threading.Thread(target=truedata_client.init, daemon=True).start()

    # ── Load last scan from cache instantly ──────────────────
    cached_results, cached_at, cached_universe = db.load_last_scan("swing")
    if cached_results:
        _cache["swing"]["data"]         = cached_results
        _cache["swing"]["last_updated"] = cached_at
        _cache["swing"]["_ts"]          = 0   # mark stale so VT waits for fresh scan
        print(f"  Loaded {len(cached_results)} stocks from last scan ({cached_at})")
    else:
        print("  No cached scan found — fresh scan will start automatically")

    # ── Auto-start background scan if data is stale / missing ─
    def _startup_scan():
        # Wait a few seconds for Groww/TrueData to connect first
        time.sleep(5)
        import datetime as _dt
        now = _dt.datetime.now()
        # Only auto-scan if last cache is > 2h old or missing
        stale = True
        if cached_at:
            try:
                last = _dt.datetime.strptime(cached_at, "%Y-%m-%d %H:%M:%S")
                stale = (now - last).total_seconds() > 7200
            except Exception:
                pass
        if stale and not _scan_lock.locked():
            print(f"  [Scan] Cache stale — auto-starting background scan ({config.SCAN_UNIVERSE})...")
            _run_swing_scan(config.SCAN_UNIVERSE)

    if config.STARTUP_SCAN_ENABLED:
        threading.Thread(target=_startup_scan, daemon=True).start()

    # Start background threads
    threading.Thread(target=_live_ltp_loop,      daemon=True).start()
    threading.Thread(target=_token_refresh_loop, daemon=True).start()
    threading.Thread(target=_auto_scan_loop,     daemon=True).start()
    threading.Thread(target=_full_scan_loop,     daemon=True).start()
    print(f"  Auto-scan scheduled at: {', '.join(config.AUTO_SCAN_HOURS)} IST")
    print(f"  Groww API: {'Connected [OK]' if _groww_connected else 'Not connected (using nselib)'}")
    port = int(os.environ.get("PORT", sys.argv[1] if len(sys.argv) > 1 else "8000"))
    print(f"  Listening on port {port}")
    print("="*60 + "\n")

    # Only open browser when running locally (Railway has no display)
    if not os.environ.get("RAILWAY_ENVIRONMENT"):
        threading.Timer(1.5, lambda: webbrowser.open(f"http://localhost:{port}")).start()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
