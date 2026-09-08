"""
Regime x score-bucket expectancy table.

Answers the question the entire scoring system must ultimately answer:
"When the model says 70-80 in a BULL_TREND, what has actually happened
historically over 5 trading days?"

For every (regime, score_bucket) cell:
  t1_hit_rate   — % of trades that reached Target 1
  sl_hit_rate   — % of trades that were stopped out
  timeout_rate  — % that were time-stopped (neither T1 nor SL in 5 days)
  avg_mae       — average max adverse excursion (% from entry, always negative)
  avg_mfe       — average max favorable excursion (% from entry)
  profit_factor — gross wins / gross losses
  expectancy    — avg $ returned per $ risked (T1-based win rate × avg win + loss rate × avg loss)
  avg_return    — mean realised return per trade after commissions
  sample_count  — number of qualifying signals

Expensive first run (5–15 min depending on universe size).
Results are cached in strategy_data.db; subsequent calls are instant.
"""
import json
import sqlite3
import threading
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from signals import compute_signals
from technical_enhanced import compute_enhanced, compute_score100

_DB_PATH = Path(__file__).parent / "strategy_data.db"
_lock    = threading.Lock()

FORWARD_DAYS  = 5
# Single source of truth for costs — imported so this module and backtester.py
# can never drift apart. See backtester._round_trip_cost() for the breakdown
# (brokerage + STT + exchange + SEBI + stamp duty + GST ≈ 0.46% round trip).
from backtester import COMMISSION, SLIPPAGE   # noqa: E402
ANALYSIS_BARS = 756     # ~3 years of daily bars
MIN_WINDOW    = 150     # bars the indicator suite needs
MIN_CELL_N    = 10      # cells with fewer trades show "insufficient data"
STEP          = 3       # sample every 3rd bar — 3x speedup, stats still valid

SCORE_BUCKETS = [(50, 60, "50-60"), (60, 70, "60-70"), (70, 80, "70-80"), (80, 101, "80+")]

# Unfiltered control group — every tradeable bar in a regime, no score filter.
BASELINE_LABEL = "BASELINE"

# Bump when the calculation changes in a way that invalidates stored results.
#   v1 → v2 (29 Aug 2026): fixed expectancy formula, realistic ~0.56% costs,
#                          added the unfiltered baseline control group.
#   v2 → v3 (29 Aug 2026): removed one-bar look-ahead — entry moved from bar i's
#                          close to bar i+1's OPEN, the first executable price.
METHODOLOGY_VERSION = 3
REGIMES       = ["BULL_TREND", "BULL_WEAK", "RECOVERY", "SIDEWAYS", "BEAR_TREND"]

_DDL = """
CREATE TABLE IF NOT EXISTS expectancy_cache (
    cache_key   TEXT PRIMARY KEY,
    result_json TEXT NOT NULL,
    run_at      TEXT NOT NULL
);
"""

_status = {"running": False, "progress": 0, "total": 0, "symbol": "", "error": None}


def _init_db():
    with _lock:
        conn = sqlite3.connect(str(_DB_PATH))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(_DDL)
        conn.commit()
        conn.close()


_init_db()


# ── Historical regime series ───────────────────────────────────

def _adx_series(high: pd.Series, low: pd.Series, close: pd.Series, p: int = 14) -> pd.Series:
    h, l, c = high.to_numpy(float), low.to_numpy(float), close.to_numpy(float)
    pc  = np.concatenate([[c[0]], c[:-1]])
    tr  = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    up  = np.diff(h, prepend=h[0])
    dn  = -np.diff(l, prepend=l[0])
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    atr = pd.Series(tr).ewm(com=p - 1, adjust=False).mean()
    pdi = 100 * pd.Series(pdm).ewm(com=p - 1, adjust=False).mean() / atr.replace(0, np.nan)
    mdi = 100 * pd.Series(mdm).ewm(com=p - 1, adjust=False).mean() / atr.replace(0, np.nan)
    dx  = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    out = dx.ewm(com=p - 1, adjust=False).mean()
    out.index = close.index
    return out


def build_regime_frame(nifty_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by date with columns:
      regime, above_200, above_50, adx
    Used to inject historical market state into each signal for accurate scoring.
    """
    close  = nifty_df["close"]
    ema50  = close.ewm(span=50,  adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()
    adx    = _adx_series(nifty_df["high"], nifty_df["low"], close)

    above_200 = close > ema200
    above_50  = close > ema50
    strong    = adx > 25

    def _classify(row):
        a200, a50, s = row["above_200"], row["above_50"], row["strong"]
        if a200 and s and a50:   return "BULL_TREND"
        if a200 and not s:       return "BULL_WEAK"
        if not a200 and a50:     return "RECOVERY"
        if not a200 and s:       return "BEAR_TREND"
        return "SIDEWAYS"

    frame = pd.DataFrame({
        "above_200": above_200,
        "above_50":  above_50,
        "strong":    strong,
    })
    frame["regime"] = frame.apply(_classify, axis=1)
    return frame[["regime", "above_200", "above_50"]]


# ── Trade simulation ───────────────────────────────────────────

def _simulate(df: pd.DataFrame, i: int, sl: float, t1: float) -> dict | None:
    """
    Simulate the trade generated by the signal on bar `i`.

    Entry is the NEXT session's OPEN, not bar i's close. The signal is derived
    from bar i's closing price, which is only known once the market has shut —
    entering at that same close is a one-bar look-ahead and is not executable.
    For a momentum strategy the overnight gap usually continues the move, so
    the old assumption systematically flattered results.

    A gap through the stop on the entry bar is a real outcome and is kept.
    """
    if i + 1 >= len(df):
        return None
    entry_open = float(df.iloc[i + 1]["open"])
    if not np.isfinite(entry_open) or entry_open <= 0:
        return None

    entry = entry_open * (1 + SLIPPAGE)
    mae, mfe = 0.0, 0.0
    outcome, exit_price = "timeout", None

    for j in range(1, FORWARD_DAYS + 1):
        k = i + j
        if k >= len(df):
            break
        lo = float(df.iloc[k]["low"])
        hi = float(df.iloc[k]["high"])
        mae = min(mae, (lo - entry) / entry * 100)
        mfe = max(mfe, (hi - entry) / entry * 100)
        sl_hit = lo <= sl
        t1_hit = hi >= t1
        if sl_hit and t1_hit:
            exit_price = float(sl) * (1 - SLIPPAGE); outcome = "sl"; break
        if sl_hit:
            exit_price = float(sl) * (1 - SLIPPAGE); outcome = "sl"; break
        if t1_hit:
            exit_price = float(t1) * (1 - SLIPPAGE); outcome = "t1"; break

    if exit_price is None:
        end_k      = min(i + FORWARD_DAYS, len(df) - 1)
        exit_price = float(df.iloc[end_k]["close"]) * (1 - SLIPPAGE)

    ret = ((exit_price - entry) / entry - COMMISSION * 2) * 100
    return {"outcome": outcome, "ret": ret, "mae": mae, "mfe": mfe}


def _bucket(score: float) -> str | None:
    for lo, hi, label in SCORE_BUCKETS:
        if lo <= score < hi:
            return label
    return None


def _aggregate(trades: list) -> dict:
    n = len(trades)
    if n < MIN_CELL_N:
        return {"sample_count": n, "insufficient_data": True}

    rets     = np.array([t["ret"]  for t in trades])
    maes     = np.array([t["mae"]  for t in trades])
    mfes     = np.array([t["mfe"]  for t in trades])
    t1_hits  = sum(1 for t in trades if t["outcome"] == "t1")
    sl_hits  = sum(1 for t in trades if t["outcome"] == "sl")
    timeouts = n - t1_hits - sl_hits

    wins    = rets[rets > 0]
    losses  = rets[rets <= 0]
    gross_p = float(wins.sum())        if len(wins)   else 0.0
    gross_l = float(abs(losses.sum())) if len(losses) else 0.0
    pf      = round(gross_p / gross_l, 2) if gross_l > 0 else None

    # Expectancy must classify trades the SAME way its win/loss averages do.
    # Previously this used the T1-hit rate as `wr` while avg_w/avg_l were keyed
    # off the sign of the return — two different groupings. A trade that times
    # out profitably counts in `wins` but never in `t1_hits`, so `wr` (~10%)
    # badly understated the true positive rate (~35%) and weighted avg_loss by
    # ~90%. Every cell came out negative even where avg_return was positive.
    # Classifying by return sign makes expectancy == mean(returns) by identity.
    win_rate = len(wins) / n
    avg_w    = float(wins.mean())   if len(wins)   else 0.0
    avg_l    = float(losses.mean()) if len(losses) else 0.0
    exp      = round(win_rate * avg_w + (1 - win_rate) * avg_l, 2)

    return {
        "sample_count":   n,
        "t1_hit_rate":    round(t1_hits  / n * 100, 1),
        "sl_hit_rate":    round(sl_hits  / n * 100, 1),
        "timeout_rate":   round(timeouts / n * 100, 1),
        "win_rate":       round(win_rate * 100, 1),   # positive-return rate
        "avg_win":        round(avg_w, 2),
        "avg_loss":       round(avg_l, 2),
        "avg_mae":        round(float(maes.mean()), 2),
        "avg_mfe":        round(float(mfes.mean()), 2),
        "profit_factor":  pf,
        "expectancy":     exp,
        "avg_return":     round(float(rets.mean()), 2),
        "insufficient_data": False,
    }


# ── Main computation ───────────────────────────────────────────

def get_status() -> dict:
    return dict(_status)


def build_expectancy_table(
    symbols:   list,
    data:      dict,
    nifty_df:  pd.DataFrame,
    cache_key: str  = "default",
    force:     bool = False,
) -> dict:
    """
    Build (or load from cache) the regime x score-bucket expectancy table.
    nifty_df must be a DataFrame with lowercase OHLCV columns from data_manager.
    """
    if not force:
        cached = _load_cache(cache_key)
        if cached:
            return cached

    _status.update({"running": True, "progress": 0, "error": None, "symbol": ""})

    try:
        print("[Expectancy] Computing historical Nifty regime series...")
        regime_frame = build_regime_frame(nifty_df)

        cells: dict[tuple, list] = {}
        total_signals = 0

        valid_symbols = [
            s for s in symbols
            if data.get(s) is not None and len(data[s]) >= MIN_WINDOW + FORWARD_DAYS + 20
        ]
        _status["total"] = len(valid_symbols)

        for sym_i, symbol in enumerate(valid_symbols):
            _status["progress"] = sym_i + 1
            _status["symbol"]   = symbol

            df = data[symbol].copy()
            df.columns = [c.lower() for c in df.columns]

            start_i = max(MIN_WINDOW, len(df) - ANALYSIS_BARS - FORWARD_DAYS)

            for i in range(start_i, len(df) - FORWARD_DAYS, STEP):
                w_start = max(0, i - MIN_WINDOW)
                subset  = df.iloc[w_start: i + 1]
                if len(subset) < 50:
                    continue

                try:
                    sig = compute_signals(subset)
                    if sig is None:
                        continue
                    sig = compute_enhanced(subset, sig, is_intraday=False)
                except Exception:
                    continue

                # Look up historical Nifty state for this bar's date
                bar_date = df.index[i]
                try:
                    prior = regime_frame[regime_frame.index <= bar_date]
                    if prior.empty:
                        continue
                    row         = prior.iloc[-1]
                    regime_str  = row["regime"]
                    above_200   = bool(row["above_200"])
                    above_50    = bool(row["above_50"])
                except Exception:
                    continue

                # Inject historical market state so bucket 6 is accurate
                sig["regime"] = {"regime": regime_str, "regime_label": regime_str}
                sig["market"] = {
                    "nifty_above_200ema": above_200,
                    "nifty_above_50ema":  above_50,
                    "vix_high":           False,
                }

                sl = sig.get("sl_buy")
                t1 = sig.get("target1")
                if not sl or not t1:
                    continue
                bar_close = float(df.iloc[i]["close"])
                if bar_close <= 0 or sl >= bar_close:
                    continue

                trade = _simulate(df, i, sl, t1)
                if trade is None:
                    continue

                # BASELINE: every tradeable bar in this regime, regardless of
                # score. Without this there is no way to tell whether a bucket's
                # profit factor reflects the score's skill or just the market's
                # drift in that regime — same stocks, same dates, same exit
                # rules, the only difference being the filter. Any bucket that
                # fails to beat its own regime's baseline is adding nothing.
                cells.setdefault((regime_str, BASELINE_LABEL), []).append(trade)

                score  = compute_score100(sig)
                blabel = _bucket(score)
                if blabel is None:
                    continue

                cells.setdefault((regime_str, blabel), []).append(trade)
                total_signals += 1

            if (sym_i + 1) % 5 == 0 or sym_i + 1 == len(valid_symbols):
                print(f"[Expectancy] {sym_i+1}/{len(valid_symbols)} done | {total_signals} signals")

        print(f"[Expectancy] Aggregating {total_signals} signals across {len(cells)} cells...")

        result_cells = []
        for regime in REGIMES:
            # Baseline first so buckets can be scored against it.
            base_cell = _aggregate(cells.get((regime, BASELINE_LABEL), []))
            base_cell["regime"]       = regime
            base_cell["score_bucket"] = BASELINE_LABEL
            base_cell["is_baseline"]  = True
            result_cells.append(base_cell)

            base_ret = base_cell.get("avg_return")
            base_pf  = base_cell.get("profit_factor")

            for _, _, blabel in SCORE_BUCKETS:
                trades = cells.get((regime, blabel), [])
                cell   = _aggregate(trades)
                cell["regime"]       = regime
                cell["score_bucket"] = blabel
                cell["is_baseline"]  = False

                # Edge over doing nothing clever in the same regime.
                # Positive => the score filter added something. Negative or
                # zero => the bucket is just tracking the market.
                if base_ret is not None and cell.get("avg_return") is not None:
                    cell["edge_vs_baseline"] = round(cell["avg_return"] - base_ret, 3)
                if base_pf and cell.get("profit_factor") is not None:
                    cell["pf_vs_baseline"] = round(cell["profit_factor"] - base_pf, 2)

                result_cells.append(cell)

        result = {
            "cells": result_cells,
            "metadata": {
                "total_signals":    total_signals,
                "symbols_analysed": len(valid_symbols),
                "analysis_bars":    ANALYSIS_BARS,
                "forward_days":     FORWARD_DAYS,
                "step":             STEP,
                "min_cell_n":       MIN_CELL_N,
                "methodology_version": METHODOLOGY_VERSION,
                "round_trip_cost_pct": round((COMMISSION * 2 + SLIPPAGE * 2) * 100, 4),
                "run_at":           datetime.now().isoformat(),
            },
        }

        _save_cache(cache_key, result)
        return result

    except Exception as e:
        import traceback
        traceback.print_exc()
        _status["error"] = str(e)
        raise
    finally:
        _status["running"] = False


# ── SQLite cache ───────────────────────────────────────────────

def _save_cache(key: str, result: dict):
    with _lock:
        conn = sqlite3.connect(str(_DB_PATH))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            "INSERT OR REPLACE INTO expectancy_cache(cache_key, result_json, run_at) VALUES(?,?,?)",
            (key, json.dumps(result), datetime.now().isoformat()),
        )
        conn.commit()
        conn.close()


def _load_cache(key: str) -> dict | None:
    with _lock:
        conn = sqlite3.connect(str(_DB_PATH))
        row  = conn.execute(
            "SELECT result_json FROM expectancy_cache WHERE cache_key=?", (key,)
        ).fetchone()
        conn.close()
    if not row:
        return None

    cached = json.loads(row[0])

    # Refuse results produced by an older methodology. Tables built before
    # METHODOLOGY_VERSION 2 used the broken expectancy formula (T1-hit rate as
    # the win rate), understated costs at 0.30% round trip, and carried no
    # baseline column. Serving them silently would keep the corrected numbers
    # invisible, so treat them as a cache miss and rebuild.
    ver = (cached.get("metadata") or {}).get("methodology_version", 1)
    if ver < METHODOLOGY_VERSION:
        print(f"[Expectancy] Cached table is v{ver}, current is "
              f"v{METHODOLOGY_VERSION} — discarding, rebuild required.")
        return None
    return cached


def clear_cache(key: str = "default"):
    with _lock:
        conn = sqlite3.connect(str(_DB_PATH))
        conn.execute("DELETE FROM expectancy_cache WHERE cache_key=?", (key,))
        conn.commit()
        conn.close()
