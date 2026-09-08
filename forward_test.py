"""
Forward test — evaluate scores recorded in the past against what actually happened.

This is the only genuinely uncontaminated evidence available in the project.
The scores in `scan_log` were computed live, months ago, from data available at
the time. Nothing about this test can be affected by:

  * look-ahead        — the scores existed before the outcomes did
  * survivorship      — the universe is whatever was actually scanned that day
  * multiple testing  — no parameters are fitted here, nothing is swept
  * hindsight bias    — the predictions are immutable, already written to disk

Entry is the OPEN of the first session AFTER the scan, mirroring the backtest
convention: a score computed from a day's close is only actionable next morning.

    python forward_test.py              # all scan dates with enough history
    python forward_test.py 2026-05-27   # one date
"""

from __future__ import annotations

import sqlite3
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_DB = Path(__file__).parent / "signals.db"
HORIZONS = {"5d": 5, "1mo": 21, "3mo": 63}
BUCKETS = [(70, 101, "70+"), (60, 70, "60-69"), (50, 60, "50-59"),
           (40, 50, "40-49"), (0, 40, "<40")]

# Round-trip cost, shared with the backtester so figures stay comparable.
try:
    from backtester import COMMISSION, SLIPPAGE
    ROUND_TRIP_PCT = (COMMISSION * 2 + SLIPPAGE * 2) * 100
except Exception:
    ROUND_TRIP_PCT = 0.5582


def scan_dates(min_symbols: int = 100) -> list[tuple]:
    with sqlite3.connect(_DB) as c:
        return c.execute(
            "SELECT date(scanned_at) d, universe, COUNT(DISTINCT symbol) n "
            "FROM scan_log WHERE scan_type='swing' "
            "GROUP BY d, universe HAVING n >= ? ORDER BY d", (min_symbols,)
        ).fetchall()


def load_scan(date_str: str, universe: str | None = None) -> pd.DataFrame:
    """Last recorded score per symbol for that date, never the best intraday score."""
    q = ("SELECT symbol, score100 score, universe "
         "FROM scan_log WHERE date(scanned_at)=? AND scan_type='swing' ")
    args = [date_str]
    if universe:
        q += "AND universe=? "
        args.append(universe)
    q += "ORDER BY scanned_at, id"
    with sqlite3.connect(_DB) as c:
        rows = c.execute(q, args).fetchall()
    df = pd.DataFrame(rows, columns=["symbol", "score", "universe"])
    df["symbol"] = df["symbol"].str.replace(".NS", "", regex=False).str.upper()
    return df.drop_duplicates('symbol', keep='last').dropna(subset=["score"])


def _prices(symbol: str, start, end):
    """Daily bars for the evaluation window, Groww then nselib."""
    try:
        import groww_client
        if groww_client.is_connected() and not groww_client.data_forbidden():
            df = groww_client.get_daily_bars_range(symbol, start, end)
            if df is not None and len(df) > 3:
                return df
    except Exception:
        pass
    try:
        import scanner
        df = scanner._nselib_ohlcv(symbol)
        if df is not None and len(df) > 3:
            return df[df.index >= pd.Timestamp(start)]
    except Exception:
        pass
    return None


def forward_returns(scan_df: pd.DataFrame, date_str: str,
                    progress_every: int = 50) -> pd.DataFrame:
    """Attach realised forward returns to each scored symbol."""
    import datetime
    scan_date = pd.Timestamp(date_str).date()
    end = datetime.date.today()

    out = []
    for i, r in enumerate(scan_df.itertuples(), 1):
        px = _prices(r.symbol, scan_date, end)
        if px is None or px.empty:
            continue
        # First session strictly AFTER the scan — the first executable bar.
        fwd = px[px.index > pd.Timestamp(scan_date)]
        if len(fwd) < 5:
            continue
        entry = float(fwd.iloc[0]["Open"])
        if not np.isfinite(entry) or entry <= 0:
            continue

        row = {"symbol": r.symbol, "score": float(r.score), "entry": entry}
        for label, n in HORIZONS.items():
            if len(fwd) >= n:
                exit_px = float(fwd.iloc[n - 1]["Close"])
                row[label] = (exit_px / entry - 1) * 100 - ROUND_TRIP_PCT
            else:
                row[label] = np.nan
        out.append(row)

        if progress_every and i % progress_every == 0:
            print(f"    {i}/{len(scan_df)} priced ({len(out)} usable)")

    return pd.DataFrame(out)


def _bucket(score: float) -> str:
    for lo, hi, lab in BUCKETS:
        if lo <= score < hi:
            return lab
    return "<40"


def report(res: pd.DataFrame, date_str: str, universe: str) -> None:
    if res.empty:
        print("  no usable rows")
        return
    res = res.copy()
    res["bucket"] = res["score"].apply(_bucket)

    print(f"\n  {date_str} · {universe} · {len(res)} symbols priced · "
          f"net of {ROUND_TRIP_PCT:.2f}% costs")
    hdr = f"  {'bucket':<8}{'n':>5}" + "".join(f"{h:>22}" for h in HORIZONS)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    base = {h: res[h].dropna() for h in HORIZONS}

    for lo, hi, lab in BUCKETS:
        sub = res[(res["score"] >= lo) & (res["score"] < hi)]
        if len(sub) < 5:
            continue
        line = f"  {lab:<8}{len(sub):>5}"
        for h in HORIZONS:
            v = sub[h].dropna()
            if len(v) < 5:
                line += f"{'-':>22}"
                continue
            edge = v.mean() - base[h].mean()
            t = (v.mean() - base[h].mean()) / np.sqrt(
                v.var(ddof=1) / len(v) + base[h].var(ddof=1) / len(base[h])
            ) if len(base[h]) > 5 else float("nan")
            line += f"{v.mean():>+8.2f}% {edge:>+7.2f} t{t:>4.1f}"
        print(line)

    line = f"  {'ALL':<8}{len(res):>5}"
    for h in HORIZONS:
        v = base[h]
        line += f"{v.mean():>+8.2f}% {'baseline':>14}" if len(v) else f"{'-':>22}"
    print(line)
    print(f"\n  columns per horizon: mean return | edge vs baseline | t-stat of the edge")


def run(date_str: str, universe: str | None = None) -> pd.DataFrame:
    scan = load_scan(date_str, universe)
    print(f"\n[{date_str}] {universe or 'all'} — {len(scan)} scored symbols")
    if scan.empty:
        return pd.DataFrame()
    res = forward_returns(scan, date_str)
    report(res, date_str, universe or "all")
    return res


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    import config, groww_client
    groww_client.init_groww(config.GROWW_API_KEY, config.GROWW_API_SECRET)

    if len(sys.argv) > 1:
        run(sys.argv[1])
    else:
        print("Scan dates with enough symbols:")
        for d, u, n in scan_dates():
            print(f"   {d}  {u:<10} {n:>4} symbols")
        print("\nRunning the two largest May scans (~3 months of hindsight)...")
        run("2026-05-27", "all_nse")
        run("2026-05-29", "nifty500")
