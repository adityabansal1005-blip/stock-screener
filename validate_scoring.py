"""
Validate scoring rule changes against realised returns — across every stock,
not one we already know the answer to.

Method
------
For each symbol in a past scan:
  1. Rebuild the indicator state using ONLY bars up to the scan date.
  2. Derive the market regime from the Nifty's own state on that date.
  3. Score it under several rule variants.
  4. Measure the ACTUAL forward return, entering at the next session's open.
  5. Rank-correlate each variant's scores against those returns.

The variant that ranks stocks closer to their realised outcome is the better
rule. A whole threshold sweep is reported rather than the best value, because
picking the winner after seeing the results is how backtests get manufactured.

Everything runs point-in-time: no bar after the scan date reaches the scoring.
"""

from __future__ import annotations

import datetime as dt
import sqlite3
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_DB = Path(__file__).parent / "signals.db"
# The scanner targets a 3-15 day swing, so measuring only 63 days can miss a
# genuine short-horizon signal. All three are computed from one price fetch.
HORIZONS = {"5d": 5, "21d": 21, "63d": 63}
HORIZON_DAYS = max(HORIZONS.values())
RS_THRESHOLDS = [None]     # regime-relaxation sweep retired — measured flat

try:
    from backtester import COMMISSION, SLIPPAGE
    COST_PCT = (COMMISSION * 2 + SLIPPAGE * 2) * 100
except Exception:
    COST_PCT = 0.5582


def scan_symbols(date_str: str, universe: str) -> list[str]:
    with sqlite3.connect(_DB) as c:
        rows = c.execute(
            "SELECT DISTINCT symbol FROM scan_log "
            "WHERE date(scanned_at)=? AND universe=? AND scan_type='swing'",
            (date_str, universe)).fetchall()
    return [r[0].replace(".NS", "").upper() for r in rows]


def nifty_state(nifty: pd.DataFrame, upto: dt.date) -> tuple[str, dict]:
    """Classify the index on `upto` using only prior bars."""
    c = nifty[nifty.index <= pd.Timestamp(upto)]["close"]
    if len(c) < 200:
        return "UNKNOWN", {}
    e50, e200 = c.ewm(span=50).mean().iloc[-1], c.ewm(span=200).mean().iloc[-1]
    px = c.iloc[-1]
    above50, above200 = px > e50, px > e200
    slope = (e50 - c.ewm(span=50).mean().iloc[-21]) / e50 * 100

    if above200 and above50:
        regime = "BULL_TREND" if slope > 0.5 else "BULL_WEAK"
    elif above200:
        regime = "SIDEWAYS"
    elif above50:
        regime = "RECOVERY"
    else:
        regime = "BEAR_TREND"
    return regime, {"nifty_above_200ema": bool(above200),
                    "nifty_above_50ema": bool(above50), "vix_high": False}


def build_state(symbol: str, scan_date: dt.date, nifty: pd.DataFrame):
    """Point-in-time indicator state, mirroring scanner.fetch_swing exactly."""
    import groww_client
    from signals import compute_signals
    from technical_enhanced import compute_enhanced

    start = scan_date - dt.timedelta(days=900)
    d = groww_client.get_daily_bars_range(symbol, start, scan_date)
    if d is None or len(d) < 120:
        return None
    d.columns = [c.lower() for c in d.columns]
    d = d[d.index <= pd.Timestamp(scan_date)]
    if len(d) < 120:
        return None

    w = d.resample("W").agg({"open": "first", "high": "max", "low": "min",
                             "close": "last", "volume": "sum"}).dropna(subset=["close"])
    nc = nifty[nifty.index <= pd.Timestamp(scan_date)]["close"].tail(252)

    try:
        s = compute_signals(d)
        if s is None:
            return None
        s = compute_enhanced(d, s, is_intraday=False,
                             weekly_df=w if len(w) >= 15 else None,
                             nifty_close=nc if len(nc) >= 60 else None)
    except Exception:
        return None

    regime, market = nifty_state(nifty, scan_date)
    s["regime"] = {"regime": regime, "regime_label": regime}
    s["market"] = market
    s.setdefault("fund_score", None)
    return s


def score_variant(state: dict, rs_floor: float | None) -> int:
    """
    rs_floor=None  -> current rules.
    rs_floor=X     -> a stock outperforming the index by >= X% has demonstrated
                      independence from it, so it is scored under BULL_WEAK
                      conditions instead of inheriting a weak index's discount.
                      This is a principled relaxation, not a fitted parameter.
    """
    from technical_enhanced import compute_score100
    s = dict(state)
    if rs_floor is not None:
        rs = s.get("rs_vs_nifty")
        if isinstance(rs, (int, float)) and not np.isnan(rs) and rs >= rs_floor:
            cur = (s.get("regime") or {}).get("regime", "")
            if cur in ("BEAR_TREND", "SIDEWAYS", "RECOVERY", "HIGH_VOLATILITY"):
                s["regime"] = {"regime": "BULL_WEAK", "regime_label": "BULL_WEAK"}
    return compute_score100(s)


def forward_return(symbol: str, scan_date: dt.date) -> dict | None:
    """Realised return at every horizon, entering at the next session's open."""
    import groww_client
    end = min(scan_date + dt.timedelta(days=HORIZON_DAYS * 2), dt.date.today())
    d = groww_client.get_daily_bars_range(symbol, scan_date, end)
    if d is None or d.empty:
        return None
    fwd = d[d.index > pd.Timestamp(scan_date)]
    if len(fwd) < 5:
        return None
    entry = float(fwd.iloc[0]["Open"])
    if entry <= 0 or not np.isfinite(entry):
        return None
    out = {}
    for label, n in HORIZONS.items():
        if len(fwd) >= n:
            out[label] = (float(fwd.iloc[n - 1]["Close"]) / entry - 1) * 100 - COST_PCT
        else:
            out[label] = np.nan
    return out


def run(date_str: str, universe: str, limit: int | None = None) -> pd.DataFrame:
    import data_manager
    nifty = data_manager.load("^NSEI")
    if nifty is None:
        raise SystemExit("Nifty history unavailable")

    scan_date = pd.Timestamp(date_str).date()
    syms = scan_symbols(date_str, universe)
    if limit:
        syms = syms[:limit]
    regime, _ = nifty_state(nifty, scan_date)
    print(f"[{date_str}] {universe} — {len(syms)} symbols | index regime: {regime}")

    rows = []
    for i, sym in enumerate(syms, 1):
        st = build_state(sym, scan_date, nifty)
        if st is None:
            continue
        fr = forward_return(sym, scan_date)
        if fr is None:
            continue
        rec = {"symbol": sym, "rs": st.get("rs_vs_nifty"),
               "stage": st.get("stage"), "score": score_variant(st, None)}
        rec.update(fr)
        rows.append(rec)
        if i % 50 == 0:
            print(f"  {i}/{len(syms)} processed ({len(rows)} usable)")

    return pd.DataFrame(rows)


def report(df: pd.DataFrame) -> None:
    if df.empty or len(df) < 30:
        print("  insufficient data"); return
    from scipy import stats

    print(f"\n  {len(df)} stocks, net {COST_PCT:.2f}% costs")
    print(f"  {'horizon':<9}{'n':>5}{'univ ret':>10}{'rank corr':>11}{'p':>8}"
          f"{'top-dec':>10}{'edge':>8}{'bot-dec':>10}{'edge':>8}")
    print("  " + "-" * 78)

    for label in HORIZONS:
        sub = df.dropna(subset=[label])
        if len(sub) < 30:
            continue
        base = sub[label].mean()
        rho, p = stats.spearmanr(sub["score"], sub[label])
        k = max(int(len(sub) * 0.10), 5)
        top = sub.nlargest(k, "score")[label].mean()
        bot = sub.nsmallest(k, "score")[label].mean()
        print(f"  {label:<9}{len(sub):>5}{base:>+9.2f}%{rho:>+11.4f}{p:>8.3f}"
              f"{top:>+9.2f}%{top - base:>+8.2f}{bot:>+9.2f}%{bot - base:>+8.2f}")

    print("\n  rank corr = Spearman(score, realised return); p<0.05 = real.")
    print("  A working scanner needs top-decile edge POSITIVE and bottom NEGATIVE.")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    import config, groww_client
    groww_client.init_groww(config.GROWW_API_KEY, config.GROWW_API_SECRET)

    date = sys.argv[1] if len(sys.argv) > 1 else "2026-05-29"
    univ = sys.argv[2] if len(sys.argv) > 2 else "nifty500"
    lim = int(sys.argv[3]) if len(sys.argv) > 3 else None
    report(run(date, univ, lim))
