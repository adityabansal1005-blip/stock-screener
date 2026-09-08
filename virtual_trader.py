"""
Autonomous Virtual Trading Agent — NSE Paper Trading

₹1,00,000 starting capital. Trades continuously during market hours.
Data: openchart 1-min candles → TrueData websocket when available.
Brain: score-based filter + Claude AI portfolio decisions.

Flow:
  9:15 AM  → Morning scan: build candidate list for the day
  Every 1m → Check exits (SL/T1 hit), update unrealised P&L
  Every 5m → Scan candidates for new entries on free cash
  Every 30m → AI portfolio review: hold / add / exit decisions
  3:30 PM  → EOD report, log daily snapshot

Exit rules:
  1. Price hits T1    → sell 50%, trail SL to breakeven on remaining
  2. Price hits SL    → full exit, log loss
  3. Score degrades   → score drops below 40 after entry → AI decides
  4. 10 days held     → forced review / exit
"""
import time
import json
import math
import logging
import sqlite3
import threading
from datetime import datetime, timedelta, date
from pathlib import Path

import pandas as pd

import config

log = logging.getLogger(__name__)

_DB_PATH = Path(__file__).parent / "strategy_data.db"
_CAPITAL  = 100_000.0
_RISK_PCT = 0.02          # fallback flat risk if Kelly can't compute
_KELLY_FRACTION = 0.25    # quarter Kelly — conservative, reduces variance
_DAILY_LOSS_LIMIT = -0.03 # halt new entries if day PnL < -3% of start capital
_MIN_POS  = 2_000.0       # minimum position ₹
_CASH_RESERVE = 5_000.0   # keep this much dry powder
_SCORE_BUY   = 65         # minimum score to enter
_RR_MIN      = 1.8        # minimum risk:reward
_POLL_SECS   = 60         # price poll interval
_SCAN_SECS   = 300        # candidate re-scan interval
_AI_SECS     = 1800       # AI review interval (30 min)
_MAX_HOLD_DAYS = 10       # force review after this many days

# Market hours IST
_MARKET_OPEN  = (9, 15)
_MARKET_CLOSE = (15, 30)


# ── Database setup ────────────────────────────────────────────────────────────

def _init_db():
    with sqlite3.connect(_DB_PATH) as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS vt_portfolio (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol      TEXT UNIQUE NOT NULL,
            qty         INTEGER NOT NULL,
            avg_price   REAL NOT NULL,
            entry_date  TEXT NOT NULL,
            sl          REAL,
            t1          REAL,
            t2          REAL,
            score100    INTEGER,
            stage       INTEGER,
            rr_ratio    REAL,
            ai_rationale TEXT
        );

        CREATE TABLE IF NOT EXISTS vt_trades (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol         TEXT NOT NULL,
            action         TEXT NOT NULL,
            qty            INTEGER NOT NULL,
            price          REAL NOT NULL,
            timestamp      TEXT NOT NULL,
            reason         TEXT,
            pnl            REAL,
            pnl_pct        REAL,
            failure_reason TEXT,
            kelly_fraction REAL
        );

        CREATE TABLE IF NOT EXISTS vt_snapshots (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT NOT NULL,
            cash            REAL,
            invested_value  REAL,
            total_value     REAL,
            unrealised_pnl  REAL,
            day_pnl         REAL,
            nifty_close     REAL
        );

        CREATE TABLE IF NOT EXISTS vt_state (
            key   TEXT PRIMARY KEY,
            value TEXT
        );
        """)
    # Safe migrations for existing databases
    with sqlite3.connect(_DB_PATH) as conn:
        existing = {r[1] for r in conn.execute("PRAGMA table_info(vt_trades)").fetchall()}
        if "failure_reason" not in existing:
            conn.execute("ALTER TABLE vt_trades ADD COLUMN failure_reason TEXT")
        if "kelly_fraction" not in existing:
            conn.execute("ALTER TABLE vt_trades ADD COLUMN kelly_fraction REAL")
    log.info("[VT] Database initialised")


# ── Portfolio State ────────────────────────────────────────────────────────────

class VirtualPortfolio:
    """Thread-safe paper portfolio backed by SQLite."""

    def __init__(self):
        _init_db()
        self._lock = threading.Lock()
        self._ensure_cash()

    def _ensure_cash(self):
        with sqlite3.connect(_DB_PATH) as conn:
            row = conn.execute("SELECT value FROM vt_state WHERE key='cash'").fetchone()
            if row is None:
                conn.execute("INSERT INTO vt_state VALUES ('cash', ?)", (str(_CAPITAL),))
                conn.execute("INSERT INTO vt_state VALUES ('start_date', ?)",
                             (datetime.now().isoformat(),))

    @property
    def cash(self) -> float:
        with sqlite3.connect(_DB_PATH) as conn:
            row = conn.execute("SELECT value FROM vt_state WHERE key='cash'").fetchone()
            return float(row[0]) if row else _CAPITAL

    def _set_cash(self, conn, amount: float):
        conn.execute("INSERT OR REPLACE INTO vt_state VALUES ('cash', ?)", (str(round(amount, 2)),))

    def positions(self) -> list[dict]:
        with sqlite3.connect(_DB_PATH) as conn:
            rows = conn.execute("SELECT * FROM vt_portfolio").fetchall()
            cols = [d[0] for d in conn.execute("SELECT * FROM vt_portfolio LIMIT 0").description
                    ] if rows else ["id","symbol","qty","avg_price","entry_date","sl","t1","t2","score100","stage","rr_ratio","ai_rationale"]
            # Re-fetch with proper column names
            conn.row_factory = sqlite3.Row
            return [dict(r) for r in conn.execute("SELECT * FROM vt_portfolio").fetchall()]

    def trade_history(self, limit: int = 50) -> list[dict]:
        with sqlite3.connect(_DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            return [dict(r) for r in conn.execute(
                "SELECT * FROM vt_trades ORDER BY id DESC LIMIT ?", (limit,)).fetchall()]

    def snapshots(self, limit: int = 14) -> list[dict]:
        with sqlite3.connect(_DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            return [dict(r) for r in conn.execute(
                "SELECT * FROM vt_snapshots ORDER BY id DESC LIMIT ?", (limit,)).fetchall()]

    def buy(self, symbol: str, qty: int, price: float,
            sl: float, t1: float, t2: float,
            score100: int, stage: int, rr_ratio: float,
            ai_rationale: str = "", kelly_fraction: float = 0.0) -> bool:
        cost = qty * price
        with self._lock:
            cash = self.cash
            if cost > cash - _CASH_RESERVE:
                log.warning(f"[VT] Not enough cash for {symbol}: need ₹{cost:.0f}, have ₹{cash:.0f}")
                return False
            with sqlite3.connect(_DB_PATH) as conn:
                # Upsert position (may already hold partial)
                existing = conn.execute(
                    "SELECT qty, avg_price FROM vt_portfolio WHERE symbol=?", (symbol,)).fetchone()
                if existing:
                    old_qty, old_avg = existing
                    new_qty = old_qty + qty
                    new_avg = (old_qty * old_avg + qty * price) / new_qty
                    conn.execute("""
                        UPDATE vt_portfolio SET qty=?, avg_price=?, sl=?, t1=?, t2=?,
                        score100=?, rr_ratio=?, ai_rationale=? WHERE symbol=?
                    """, (new_qty, new_avg, sl, t1, t2, score100, rr_ratio, ai_rationale, symbol))
                else:
                    conn.execute("""
                        INSERT INTO vt_portfolio
                        (symbol, qty, avg_price, entry_date, sl, t1, t2, score100, stage, rr_ratio, ai_rationale)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?)
                    """, (symbol, qty, price, datetime.now().isoformat(),
                          sl, t1, t2, score100, stage, rr_ratio, ai_rationale))
                # Deduct cash
                self._set_cash(conn, cash - cost)
                # Log trade
                conn.execute("""
                    INSERT INTO vt_trades
                    (symbol, action, qty, price, timestamp, reason, pnl, pnl_pct, kelly_fraction)
                    VALUES (?,?,?,?,?,?,?,?,?)
                """, (symbol, "BUY", qty, price, datetime.now().isoformat(),
                      ai_rationale, 0.0, 0.0, round(kelly_fraction, 4)))
        print(f"[VT] BUY  {qty:>4} × {symbol:<15} @ ₹{price:>8.2f}  cost ₹{cost:>9,.0f}  | {ai_rationale[:60]}")
        return True

    def sell(self, symbol: str, qty: int, price: float, reason: str = "") -> float:
        """Sell qty shares at price. Returns realised PnL."""
        with self._lock:
            with sqlite3.connect(_DB_PATH) as conn:
                pos = conn.execute(
                    "SELECT qty, avg_price FROM vt_portfolio WHERE symbol=?", (symbol,)).fetchone()
                if not pos:
                    return 0.0
                held_qty, avg_price = pos
                sell_qty = min(qty, held_qty)
                pnl     = (price - avg_price) * sell_qty
                pnl_pct = (price - avg_price) / avg_price * 100
                proceeds = price * sell_qty
                # Classify failure before deleting position
                fail_tag = _classify_failure(reason, pnl_pct) if pnl < 0 else None
                # Update position
                new_qty = held_qty - sell_qty
                if new_qty <= 0:
                    conn.execute("DELETE FROM vt_portfolio WHERE symbol=?", (symbol,))
                else:
                    conn.execute("UPDATE vt_portfolio SET qty=? WHERE symbol=?", (new_qty, symbol))
                # Return cash
                cash = self.cash
                self._set_cash(conn, cash + proceeds)
                # Log with failure classification
                conn.execute("""
                    INSERT INTO vt_trades
                    (symbol, action, qty, price, timestamp, reason, pnl, pnl_pct, failure_reason)
                    VALUES (?,?,?,?,?,?,?,?,?)
                """, (symbol, "SELL", sell_qty, price, datetime.now().isoformat(),
                      reason, round(pnl, 2), round(pnl_pct, 2), fail_tag))
                if fail_tag:
                    log.info(f"[VT] Failure classified: {symbol} → {fail_tag}")
        icon = "✓" if pnl >= 0 else "✗"
        print(f"[VT] SELL {sell_qty:>4} × {symbol:<15} @ ₹{price:>8.2f}  PnL ₹{pnl:>+8,.0f} ({pnl_pct:+.1f}%)  {icon} {reason}{' ['+fail_tag+']' if pnl < 0 and fail_tag else ''}")
        return pnl

    def snapshot(self, prices: dict[str, float], nifty_price: float = 0):
        """Log portfolio value snapshot."""
        positions = self.positions()
        invested = 0.0
        unreal   = 0.0
        for p in positions:
            ltp = prices.get(p["symbol"], p["avg_price"])
            invested += p["avg_price"] * p["qty"]
            unreal   += (ltp - p["avg_price"]) * p["qty"]
        total = self.cash + invested + unreal
        with sqlite3.connect(_DB_PATH) as conn:
            conn.execute("""
                INSERT INTO vt_snapshots
                (timestamp, cash, invested_value, total_value, unrealised_pnl, day_pnl, nifty_close)
                VALUES (?,?,?,?,?,?,?)
            """, (datetime.now().isoformat(),
                  round(self.cash, 2), round(invested, 2),
                  round(total, 2), round(unreal, 2), 0.0, nifty_price))

    def portfolio_value(self, prices: dict[str, float]) -> float:
        positions = self.positions()
        val = self.cash
        for p in positions:
            ltp = prices.get(p["symbol"], p["avg_price"])
            val += ltp * p["qty"]
        return val

    def reset(self):
        """Full reset — wipe all positions, trades, start fresh."""
        with self._lock:
            with sqlite3.connect(_DB_PATH) as conn:
                conn.executescript("""
                    DELETE FROM vt_portfolio;
                    DELETE FROM vt_trades;
                    DELETE FROM vt_snapshots;
                    DELETE FROM vt_state;
                """)
        self._ensure_cash()
        print(f"[VT] Portfolio reset — ₹{_CAPITAL:,.0f} cash")


# ── Price Feed ────────────────────────────────────────────────────────────────

def _get_ltp(symbols: list[str]) -> dict[str, float]:
    """
    Get current prices for a list of NSE symbols.
    Priority: TrueData (websocket) → yfinance snapshot
    """
    if not symbols:
        return {}

    # TrueData live (best latency)
    try:
        import truedata_client as td
        if td.is_connected():
            prices = td.get_ltp(symbols)
            if prices:
                return prices
    except Exception:
        pass

    # yfinance snapshot (fast for small batches)
    try:
        import yfinance as yf
        tickers = [f"{s}.NS" for s in symbols]
        data = yf.download(tickers, period="1d", interval="1m",
                           progress=False, auto_adjust=True)
        if data is None or data.empty:
            return {}
        prices = {}
        if len(tickers) == 1:
            if not data["Close"].empty:
                prices[symbols[0]] = float(data["Close"].dropna().iloc[-1])
        else:
            for sym, ticker in zip(symbols, tickers):
                col = ("Close", ticker)
                if col in data.columns:
                    s = data[col].dropna()
                    if not s.empty:
                        prices[sym] = float(s.iloc[-1])
        return prices
    except Exception as e:
        log.debug(f"[VT] LTP fetch failed: {e}")
        return {}


def _get_candles_1m(symbol: str, bars: int = 30) -> pd.DataFrame | None:
    """Fetch latest N 1-minute candles for a symbol."""
    # TrueData first
    try:
        import truedata_client as td
        if td.is_connected():
            df = td.get_intraday_candles(symbol, interval_minutes=1)
            if df is not None and len(df) >= 5:
                return df.tail(bars)
    except Exception:
        pass

    # openchart (free, no auth, 1-min NSE data)
    try:
        from openchart import NSEData
        nse = NSEData()
        end   = datetime.now()
        start = end - timedelta(minutes=bars + 10)
        df = nse.historical(symbol, segment="EQ", start=start, end=end, interval="1m")
        if df is not None and not df.empty and len(df) >= 5:
            df.columns = [c.lower() for c in df.columns]
            return df[["open","high","low","close","volume"]].tail(bars)
    except Exception:
        pass

    # yfinance 2-min fallback
    try:
        import yfinance as yf
        import contextlib, io
        with contextlib.redirect_stderr(io.StringIO()):
            df = yf.Ticker(f"{symbol}.NS").history(period="1d", interval="2m", auto_adjust=True)
        if df is not None and not df.empty:
            df.columns = [c.lower() for c in df.columns]
            return df[["open","high","low","close","volume"]].tail(bars)
    except Exception:
        pass

    return None


# ── Fractional Kelly Position Sizing ─────────────────────────────────────────

def _kelly_size(candidate: dict, portfolio_value: float) -> tuple[float, float]:
    """
    Compute position size (₹) using fractional Kelly Criterion.

    f* = (p × b − q) / b          ← full Kelly
    invest = f* × KELLY_FRACTION × portfolio_value   ← quarter Kelly

    p = win_rate from backtest (or 0.50 fallback)
    b = rr_ratio (avg_win / avg_loss, e.g. 2.0)
    q = 1 − p

    Returns (invest_amount_₹, kelly_f_raw) so caller can log the fraction used.
    Clamped: min _MIN_POS, max 15% of portfolio.
    """
    win_rate = float(candidate.get("win_rate") or 0)
    rr_ratio = float(candidate.get("rr_ratio") or 0)

    # Need at least meaningful inputs to use Kelly
    if win_rate <= 0 or rr_ratio <= 0:
        # Fall back to flat 2% risk
        invest = portfolio_value * _RISK_PCT
        return (max(_MIN_POS, min(invest, portfolio_value * 0.15)), 0.0)

    p = win_rate / 100.0 if win_rate > 1 else win_rate  # accept 0–1 or 0–100
    p = max(0.01, min(0.99, p))
    q = 1.0 - p
    b = rr_ratio

    kelly_f = (p * b - q) / b          # full Kelly fraction
    kelly_f = max(0.0, kelly_f)         # never negative
    frac_kelly = kelly_f * _KELLY_FRACTION

    invest = frac_kelly * portfolio_value
    invest = max(_MIN_POS, min(invest, portfolio_value * 0.15))  # 2K–15% cap
    return (round(invest, 2), round(kelly_f, 4))


# ── Failure Classification ─────────────────────────────────────────────────────

# Tags and what they mean — stored in vt_trades.failure_reason
# SL_HIT          — clean stop loss hit (price kept going)
# SL_TOO_TIGHT    — stopped out within 0.5% of SL, then direction proved correct
# MAX_HOLD_LOSS   — held max days, still negative
# AI_DIRECTED     — AI portfolio review forced exit at a loss
# FALSE_BREAKOUT  — entered on breakout signal, price reversed back below breakout
# SMALL_LOSS_TRIM — exited with <1% loss (normal noise / trim)
# SECTOR_DRAG     — tagged when F&O ban or OI spurt flagged at entry

_FAILURE_COOLDOWN_DAYS = 3   # don't re-enter a failed symbol for this many days

def _classify_failure(reason: str, pnl_pct: float) -> str:
    """Auto-classify why a losing trade failed based on exit reason and P&L size."""
    if pnl_pct >= 0:
        return ""
    if "SL_HIT" in reason:
        # Tight SL if loss was very small (price barely crossed SL)
        return "SL_TOO_TIGHT" if abs(pnl_pct) < 0.8 else "SL_HIT"
    if "MAX_HOLD" in reason:
        return "MAX_HOLD_LOSS"
    if reason.startswith("AI:"):
        return "AI_DIRECTED"
    if abs(pnl_pct) < 1.0:
        return "SMALL_LOSS_TRIM"
    return "FALSE_BREAKOUT"


def _recently_failed(symbol: str) -> bool:
    """
    Return True if this symbol has a logged failure within the cooldown window.
    Prevents re-entering a setup that just stopped us out.
    """
    cutoff = (datetime.now() - timedelta(days=_FAILURE_COOLDOWN_DAYS)).isoformat()
    try:
        with sqlite3.connect(_DB_PATH) as conn:
            row = conn.execute("""
                SELECT COUNT(*) FROM vt_trades
                WHERE symbol=? AND action='SELL' AND pnl < 0
                  AND failure_reason NOT IN ('SMALL_LOSS_TRIM','')
                  AND timestamp >= ?
            """, (symbol, cutoff)).fetchone()
            return (row[0] or 0) > 0
    except Exception:
        return False


# ── Daily Loss Limit ──────────────────────────────────────────────────────────

def _get_daily_pnl() -> float:
    """Sum of realised PnL from all SELL trades today (IST date)."""
    today = date.today().isoformat()
    try:
        with sqlite3.connect(_DB_PATH) as conn:
            row = conn.execute("""
                SELECT COALESCE(SUM(pnl), 0) FROM vt_trades
                WHERE action='SELL' AND date(timestamp)=?
            """, (today,)).fetchone()
            return float(row[0]) if row else 0.0
    except Exception:
        return 0.0


def _daily_limit_breached() -> bool:
    """True if today's realised losses exceed the daily loss limit."""
    daily_pnl = _get_daily_pnl()
    return daily_pnl < (_CAPITAL * _DAILY_LOSS_LIMIT)


# ── Signal check on live candles ──────────────────────────────────────────────

def _quick_signal(symbol: str) -> dict:
    """
    Fast signal check on latest 1-min candles.
    Returns {bullish: bool, score: int, close: float, above_vwap: bool}
    """
    df = _get_candles_1m(symbol, bars=50)
    if df is None or len(df) < 20:
        return {}
    try:
        from signals import compute_signals
        result = compute_signals(df)
        return result or {}
    except Exception:
        return {}


# ── Claude AI Portfolio Decision ──────────────────────────────────────────────

def _ai_portfolio_decision(portfolio: VirtualPortfolio,
                            candidates: list[dict],
                            prices: dict[str, float]) -> dict:
    """
    Ask Claude to decide: which candidates to buy, which positions to trim/exit.
    Returns {buys: [{symbol, qty, reason}], exits: [{symbol, reason}]}
    """
    client_key = getattr(config, "ANTHROPIC_API_KEY", "")
    if not client_key or client_key == "YOUR_ANTHROPIC_API_KEY":
        return {"buys": [], "exits": []}

    positions = portfolio.positions()
    cash      = portfolio.cash
    port_val  = portfolio.portfolio_value(prices)

    # Build context
    pos_summary = []
    for p in positions:
        ltp = prices.get(p["symbol"], p["avg_price"])
        pnl_pct = (ltp - p["avg_price"]) / p["avg_price"] * 100
        days_held = (datetime.now() - datetime.fromisoformat(p["entry_date"])).days
        pos_summary.append({
            "symbol":     p["symbol"],
            "qty":        p["qty"],
            "avg_price":  round(p["avg_price"], 2),
            "current":    round(ltp, 2),
            "pnl_pct":    round(pnl_pct, 1),
            "days_held":  days_held,
            "sl":         p.get("sl"),
            "t1":         p.get("t1"),
            "score100":   p.get("score100"),
        })

    cand_summary = [{
        "symbol":   c.get("symbol"),
        "score100": c.get("score100"),
        "stage":    c.get("stage"),
        "rr_ratio": c.get("rr_ratio"),
        "rs_vs_nifty": c.get("rs_vs_nifty"),
        "ibd_rs":   c.get("ibd_rs"),
        "ml_prob":  round(c.get("ml_prob", 0) or 0, 2),
        "close":    c.get("close"),
        "sl":       c.get("sl_buy"),
        "t1":       c.get("target1"),
        "verdict":  c.get("verdict"),
    } for c in candidates[:8]]

    prompt = f"""You are managing a ₹{port_val:,.0f} virtual NSE portfolio (paper trading).
Available cash: ₹{cash:,.0f}
Reserve: ₹{_CASH_RESERVE:,.0f} (do not go below this)

CURRENT POSITIONS:
{json.dumps(pos_summary, indent=2)}

CANDIDATE STOCKS TO BUY (already filtered: score≥65, Stage2, R:R≥{_RR_MIN}):
{json.dumps(cand_summary, indent=2)}

TASK: Make optimal portfolio decisions RIGHT NOW.

Rules:
- Risk max 2% of portfolio per new trade (use sl price to calculate position size)
- Prefer: highest IBD RS rank, strong Stage 2, ML prob ≥ 0.60
- Exit a position if: PnL ≤ -1.5% AND score degraded, OR days_held ≥ 8 with <0 PnL
- Don't over-concentrate: max 25% of portfolio in any single stock

Respond ONLY with this exact JSON (no other text):
{{
  "buys": [
    {{"symbol": "STOCKNAME", "invest_amount": 15000, "reason": "one sentence why"}}
  ],
  "exits": [
    {{"symbol": "STOCKNAME", "reason": "one sentence why"}}
  ],
  "commentary": "one sentence overall market read"
}}"""

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=client_key)
        with client.messages.stream(
            model="claude-haiku-4-5-20251001",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            text = stream.get_final_message().content[0].text.strip()

        # Extract JSON
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(text[start:end])
            print(f"[VT/AI] {data.get('commentary', '')}")
            return data
    except Exception as e:
        log.debug(f"[VT/AI] Decision failed: {e}")

    return {"buys": [], "exits": []}


# ── Morning Scan ──────────────────────────────────────────────────────────────

def _get_main_scan_results() -> list[dict]:
    """
    Return the main scanner's cached swing results.
    Blocks (30s sleep loop) until the main scan finishes — no timeout,
    because All NSE can take 30+ minutes.
    The VT NEVER calls scan_universe() itself; it always consumes the
    main scanner's output to avoid double-scanning.
    """
    try:
        import main as _main
    except Exception:
        return []

    while True:
        scanning  = _main._scan_lock.locked() or _main._cache["swing"].get("loading", False)
        has_data  = bool(_main._cache["swing"].get("data"))

        if has_data and not scanning:
            return _main._cache["swing"]["data"]

        if scanning:
            print("[VT] Main scan in progress — waiting 30s to piggyback...")
            _agent_status["last_action"] = "Waiting for main scan to finish..."
            time.sleep(30)
            continue

        # No data and not scanning — trigger the main scan now
        import config as _cfg
        universe = getattr(_cfg, "SCAN_UNIVERSE", "nifty500")
        print(f"[VT] No scan data yet — triggering main scan ({universe})...")
        _agent_status["last_action"] = f"Triggered main scan ({universe})..."
        threading.Thread(
            target=_main._run_swing_scan, args=(universe,), daemon=True
        ).start()
        time.sleep(30)


def _morning_scan(universe: str = "nifty500") -> list[dict]:
    """
    Build buy candidates by filtering the main scanner's results.
    Never calls scan_universe() directly — zero double-scanning.
    The universe parameter is kept for API compatibility but ignored;
    the VT inherits whatever universe the main scanner used.
    """
    print("[VT] Building candidate list from main scan cache...")
    results = _get_main_scan_results()

    candidates = [
        r for r in results
        if (r.get("score100", 0) >= _SCORE_BUY
            and r.get("stage", 0) == 2
            and (r.get("rr_ratio") or 0) >= _RR_MIN
            and not r.get("fno_ban", False)
            and (r.get("promoter_pledged") or 0) < 30)
    ]
    candidates.sort(key=lambda x: (
        x.get("ibd_rs") or 0,
        x.get("ml_prob") or 0,
        x.get("score100", 0)
    ), reverse=True)
    print(f"[VT] Candidates: {len(results)} in cache → {len(candidates)} qualify")
    return candidates


# ── Exit Checker ──────────────────────────────────────────────────────────────

def _check_exits(portfolio: VirtualPortfolio, prices: dict[str, float]):
    """Check all positions for SL / T1 hit. Execute exits immediately."""
    for pos in portfolio.positions():
        sym = pos["symbol"]
        ltp = prices.get(sym)
        if not ltp:
            continue

        sl = pos.get("sl")
        t1 = pos.get("t1")
        t2 = pos.get("t2")
        qty = pos["qty"]

        # SL hit
        if sl and ltp <= sl:
            portfolio.sell(sym, qty, ltp, reason="SL_HIT")
            continue

        # T1 hit — sell 50%, trail SL to breakeven on remainder
        if t1 and ltp >= t1:
            sell_qty = max(1, qty // 2)
            portfolio.sell(sym, sell_qty, ltp, reason="T1_HIT")
            # Update SL to breakeven for remaining
            if qty - sell_qty > 0:
                try:
                    with sqlite3.connect(_DB_PATH) as conn:
                        conn.execute(
                            "UPDATE vt_portfolio SET sl=avg_price WHERE symbol=?", (sym,))
                    print(f"[VT] {sym} SL trailed to breakeven after T1")
                except Exception:
                    pass
            # If T2 exists, keep remaining for T2
            continue

        # Max hold exceeded — force exit
        try:
            entry_date = datetime.fromisoformat(pos["entry_date"])
            days_held  = (datetime.now() - entry_date).days
            if days_held >= _MAX_HOLD_DAYS:
                portfolio.sell(sym, qty, ltp, reason=f"MAX_HOLD_{days_held}D")
        except Exception:
            pass


# ── Main Trading Loop ─────────────────────────────────────────────────────────

_agent_running = False
_agent_thread: threading.Thread | None = None
_agent_status = {"state": "stopped", "last_action": "", "last_tick": None}
_candidates: list[dict] = []
_last_scan_ts   = 0.0
_last_ai_ts     = 0.0


def _is_market_open() -> bool:
    now = datetime.now()
    if now.weekday() >= 5:  # Saturday/Sunday
        return False
    h, m = now.hour, now.minute
    open_mins  = _MARKET_OPEN[0]  * 60 + _MARKET_OPEN[1]
    close_mins = _MARKET_CLOSE[0] * 60 + _MARKET_CLOSE[1]
    cur_mins   = h * 60 + m
    return open_mins <= cur_mins <= close_mins


def _trading_loop(portfolio: VirtualPortfolio, universe: str):
    """Main agent loop — runs in background thread."""
    global _last_scan_ts, _last_ai_ts, _candidates, _agent_status

    print("[VT] Agent started")
    _agent_status["state"] = "running"

    while _agent_running:
        try:
            if not _is_market_open():
                _agent_status["state"] = "waiting_for_market"
                _agent_status["last_action"] = "Market closed — waiting"
                time.sleep(60)
                continue

            _agent_status["state"] = "active"
            now_ts = time.time()

            # ── Morning scan / periodic re-scan ──────────────────────
            if now_ts - _last_scan_ts > _SCAN_SECS or not _candidates:
                _candidates = _morning_scan(universe)
                _last_scan_ts = now_ts

            # ── Get all relevant prices ───────────────────────────────
            held_syms   = [p["symbol"] for p in portfolio.positions()]
            cand_syms   = [c["symbol"] for c in _candidates[:20]]
            all_syms    = list(set(held_syms + cand_syms))
            prices      = _get_ltp(all_syms)

            if not prices:
                time.sleep(_POLL_SECS)
                continue

            # ── Check exits ───────────────────────────────────────────
            _check_exits(portfolio, prices)

            # ── Snapshot ──────────────────────────────────────────────
            portfolio.snapshot(prices)

            # ── AI portfolio decision ─────────────────────────────────
            if now_ts - _last_ai_ts > _AI_SECS and _candidates:
                ai_result = _ai_portfolio_decision(portfolio, _candidates, prices)
                _last_ai_ts = now_ts

                # Execute AI-directed exits
                for exit_order in ai_result.get("exits", []):
                    sym = exit_order.get("symbol", "")
                    reason = exit_order.get("reason", "AI_EXIT")
                    pos_map = {p["symbol"]: p for p in portfolio.positions()}
                    if sym in pos_map:
                        ltp = prices.get(sym, pos_map[sym]["avg_price"])
                        portfolio.sell(sym, pos_map[sym]["qty"], ltp, reason=f"AI: {reason[:50]}")

                # Execute AI-directed buys
                pos_syms = {p["symbol"] for p in portfolio.positions()}

                # Daily loss limit — stop all new entries for today
                if _daily_limit_breached():
                    daily_pnl = _get_daily_pnl()
                    print(f"[VT] ⛔ Daily loss limit hit (₹{daily_pnl:+,.0f}) — no new entries today")
                    _agent_status["last_action"] = (
                        f"⛔ Daily loss limit reached (₹{daily_pnl:+,.0f}) — entries paused"
                    )
                else:
                    port_val = portfolio.portfolio_value(prices)
                    for buy_order in ai_result.get("buys", []):
                        sym    = buy_order.get("symbol", "")
                        reason = buy_order.get("reason", "")

                        if sym in pos_syms:
                            continue

                        # Skip symbols that recently failed — failure cooldown
                        if _recently_failed(sym):
                            print(f"[VT] ⏭ Skipping {sym} — recent failure (cooldown {_FAILURE_COOLDOWN_DAYS}d)")
                            continue

                        # Find candidate data for SL/T1
                        cand = next((c for c in _candidates if c.get("symbol") == sym), None)
                        if not cand:
                            continue

                        ltp = prices.get(sym, cand.get("close", 0))
                        if not ltp or ltp <= 0:
                            continue

                        sl  = cand.get("sl_buy") or ltp * 0.97
                        t1  = cand.get("target1") or ltp * 1.04
                        t2  = cand.get("target2") or ltp * 1.08

                        # Kelly position sizing
                        amount, kelly_f = _kelly_size(cand, port_val)
                        if amount < _MIN_POS:
                            continue

                        qty = max(1, int(amount / ltp))

                        portfolio.buy(
                            symbol=sym, qty=qty, price=ltp,
                            sl=sl, t1=t1, t2=t2,
                            score100=cand.get("score100", 0),
                            stage=cand.get("stage", 0),
                            rr_ratio=cand.get("rr_ratio", 0),
                            ai_rationale=reason,
                            kelly_fraction=kelly_f,
                        )
                        pos_syms.add(sym)

            port_val = portfolio.portfolio_value(prices)
            pnl      = port_val - _CAPITAL
            _agent_status["last_tick"] = datetime.now().isoformat()
            _agent_status["last_action"] = (
                f"Portfolio ₹{port_val:,.0f} | PnL {pnl:+,.0f} ({pnl/_CAPITAL*100:+.1f}%)"
            )

        except Exception as e:
            log.exception(f"[VT] Loop error: {e}")

        time.sleep(_POLL_SECS)

    _agent_status["state"] = "stopped"
    print("[VT] Agent stopped")


# ── Public API ────────────────────────────────────────────────────────────────

_portfolio: VirtualPortfolio | None = None


def get_portfolio() -> VirtualPortfolio:
    global _portfolio
    if _portfolio is None:
        _portfolio = VirtualPortfolio()
    return _portfolio


def start_agent(universe: str = "nifty500"):
    global _agent_running, _agent_thread
    if _agent_running:
        print("[VT] Agent already running")
        return
    _agent_running = True
    port = get_portfolio()
    _agent_thread = threading.Thread(
        target=_trading_loop, args=(port, universe), daemon=True, name="VT-Agent"
    )
    _agent_thread.start()


def stop_agent():
    global _agent_running
    _agent_running = False
    print("[VT] Stop signal sent")


def agent_status() -> dict:
    port   = get_portfolio()
    prices = _get_ltp([p["symbol"] for p in port.positions()])
    val    = port.portfolio_value(prices)
    daily_pnl     = _get_daily_pnl()
    limit_breached = _daily_limit_breached()
    return {
        **_agent_status,
        "portfolio_value":   round(val, 2),
        "cash":              round(port.cash, 2),
        "pnl":               round(val - _CAPITAL, 2),
        "pnl_pct":           round((val - _CAPITAL) / _CAPITAL * 100, 2),
        "n_positions":       len(port.positions()),
        "running":           _agent_running,
        "daily_pnl":         round(daily_pnl, 2),
        "daily_limit_hit":   limit_breached,
        "daily_limit_pct":   round(daily_pnl / _CAPITAL * 100, 2),
    }


def get_dashboard_data() -> dict:
    """All data for the virtual trader dashboard."""
    port   = get_portfolio()
    prices = _get_ltp([p["symbol"] for p in port.positions()])

    positions = port.positions()
    for p in positions:
        ltp = prices.get(p["symbol"], p["avg_price"])
        p["current_price"] = round(ltp, 2)
        p["unrealised_pnl"] = round((ltp - p["avg_price"]) * p["qty"], 2)
        p["unrealised_pct"] = round((ltp - p["avg_price"]) / p["avg_price"] * 100, 2)
        p["value"]          = round(ltp * p["qty"], 2)

    val = port.portfolio_value(prices)
    return {
        "status":          agent_status(),
        "positions":       positions,
        "trade_history":   port.trade_history(50),
        "snapshots":       port.snapshots(14),
        "candidates":      [{
            "symbol":   c.get("symbol"),
            "score100": c.get("score100"),
            "stage":    c.get("stage_label"),
            "rr_ratio": c.get("rr_ratio"),
            "ibd_rs":   c.get("ibd_rs"),
            "ml_prob":  round((c.get("ml_prob") or 0) * 100),
            "verdict":  c.get("verdict"),
            "close":    c.get("close"),
        } for c in _candidates[:10]],
        "total_value":       round(val, 2),
        "cash":              round(port.cash, 2),
        "invested":          round(val - port.cash, 2),
        "pnl":               round(val - _CAPITAL, 2),
        "pnl_pct":           round((val - _CAPITAL) / _CAPITAL * 100, 2),
        "start_capital":     _CAPITAL,
        "daily_pnl":         round(_get_daily_pnl(), 2),
        "daily_limit_hit":   _daily_limit_breached(),
        "daily_loss_limit":  round(_CAPITAL * _DAILY_LOSS_LIMIT, 2),
    }
