"""
OHLCV sanity checks — catch bad data at the boundary, not downstream.

Two real incidents motivated this module:

  1. Groww's intraday path parsed epoch SECONDS as NANOSECONDS, dating every
     candle to 1970. It emitted lowercase column names at the same time, so
     the scanner silently received nothing usable. No error was ever raised.

  2. Groww's V2 candle endpoint returns null OPEN on every row and roughly
     double the true volume. Migrating to it — as their own docs recommend —
     would have broken candlestick patterns without a single warning.

Both were silent. The scan completed, the dashboard rendered, the numbers were
wrong. Validation converts that class of failure into a loud, attributable one.

Usage:
    ok, issues = validate_ohlcv(df, "RELIANCE", source="groww")
    if not ok:
        ...fall through to the next data source...
"""

from __future__ import annotations

import datetime

import pandas as pd
import numpy as np

REQUIRED_COLS = ("Open", "High", "Low", "Close", "Volume")

# A frame older than this is almost certainly an epoch-unit bug, not real data.
_MIN_PLAUSIBLE_YEAR = 1995

# How old the newest bar may be before the frame is treated as a different
# period rather than the present one. NSE's longest holiday-plus-weekend
# cluster is about five days, so twelve is generous while still catching the
# failure this was written for.
_MAX_STALE_DAYS = 12


def validate_ohlcv(df, symbol: str = "?", source: str = "?",
                   intraday: bool = False, min_rows: int = 30,
                   max_stale_days: int | None = _MAX_STALE_DAYS
                   ) -> tuple[bool, list[str]]:
    """
    Check an OHLCV frame for structural and internal-consistency problems.

    Returns (ok, issues). `ok` is False only for faults that make the frame
    unusable — callers should fall through to the next source. Issues are
    returned even when ok is True so callers can log soft warnings.
    """
    issues: list[str] = []

    # ── Structure ────────────────────────────────────────────────────────────
    if df is None:
        return False, ["frame is None"]
    if not isinstance(df, pd.DataFrame):
        return False, [f"not a DataFrame (got {type(df).__name__})"]
    if df.empty:
        return False, ["frame is empty"]

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        # The classic lowercase-columns bug lands here.
        lower = [c for c in df.columns if str(c).lower() in
                 {r.lower() for r in REQUIRED_COLS}]
        hint = f" (found differently-cased {lower})" if lower else ""
        return False, [f"missing columns {missing}{hint}"]

    if not isinstance(df.index, pd.DatetimeIndex):
        return False, [f"index is {type(df.index).__name__}, expected DatetimeIndex"]

    if len(df) < min_rows:
        return False, [f"only {len(df)} rows (< {min_rows} needed for indicators)"]

    # ── Dates ────────────────────────────────────────────────────────────────
    if df.index.hasnans:
        return False, ["invalid timestamps (NaT)"]
    first, last = df.index.min(), df.index.max()
    if first.year < _MIN_PLAUSIBLE_YEAR:
        # This is the epoch-seconds-as-nanoseconds signature.
        return False, [
            f"dates start {first.date()} — implausible, likely an epoch-unit bug"
        ]

    now = pd.Timestamp.now(tz=df.index.tz)
    tomorrow = now + pd.Timedelta(days=1)
    if last > tomorrow:
        return False, [f"latest bar {last} is in the future"]

    # ── Staleness ────────────────────────────────────────────────────────────
    # The gate checked that the newest bar was not in the FUTURE and never that
    # it was not ancient, so a frame ending fifteen months ago passed cleanly
    # and every indicator was computed on it as though it were today.
    #
    # RELINFRA, 31 Aug 2026: nselib returned history ending 04 Jun 2025 because
    # the stock had moved from the EQ series to BE and the EQ filter truncated
    # it there. The scan scored it 76 — second highest of 2,373 stocks —
    # "HIGH-QUALITY SETUP", stop Rs 323.9, target Rs 494, on a stock trading at
    # Rs 59.4. The dashboard then rendered the stale close beside the live price
    # and showed +11.27%. No error was raised anywhere. 23 of 899 nselib-sourced
    # rows in that scan carried the same defect; 13 scored 60+.
    #
    # Callers replaying history deliberately (backtests, point-in-time studies)
    # pass max_stale_days=None.
    if max_stale_days is not None and not intraday:
        age = (now - last).days
        if age > max_stale_days:
            return False, [
                f"newest bar is {last.date()}, {age} days old — the frame is not "
                f"current (source may have truncated the series)"
            ]

    if not df.index.is_monotonic_increasing:
        return False, ["index not sorted ascending"]
    if df.index.has_duplicates:
        return False, [f"{int(df.index.duplicated().sum())} duplicate timestamps"]

    # ── Nulls ────────────────────────────────────────────────────────────────
    ohlc = list(REQUIRED_COLS[:4])
    nulls = {c: int(df[c].isna().sum()) for c in ohlc if df[c].isna().any()}
    if nulls:
        # Groww V2's null-OPEN defect lands here.
        total = sum(nulls.values())
        if total >= len(df) * 0.5:
            return False, [f"null OHLC values: {nulls} — source is not returning full bars"]
        issues.append(f"null OHLC values: {nulls}")

    # ── Prices ───────────────────────────────────────────────────────────────
    num = df[ohlc].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(num.to_numpy(dtype=float)).all():
        return False, ["missing, nonnumeric or infinite OHLC values"]
    if (num <= 0).any().any():
        bad = {c: int((num[c] <= 0).sum()) for c in ohlc if (num[c] <= 0).any()}
        return False, [f"non-positive prices: {bad}"]

    # ── Internal consistency ─────────────────────────────────────────────────
    valid = num.dropna()
    if not valid.empty:
        hl = int((valid["High"] < valid["Low"]).sum())
        if hl:
            return False, [f"{hl} bars where High < Low"]

        hi_break = int((valid["High"] < valid[["Open", "Close"]].max(axis=1)).sum())
        lo_break = int((valid["Low"] > valid[["Open", "Close"]].min(axis=1)).sum())
        # Allow a tiny tolerance for vendor rounding.
        if (valid["High"] + 0.011 < valid[["Open", "Close"]].max(axis=1)).any():
            return False, [f"{hi_break} bars where High < max(Open, Close)"]
        if (valid["Low"] - 0.011 > valid[["Open", "Close"]].min(axis=1)).any():
            return False, [f"{lo_break} bars where Low > min(Open, Close)"]

    # ── Volume ───────────────────────────────────────────────────────────────
    vol = pd.to_numeric(df["Volume"], errors="coerce")
    if not np.isfinite(vol.to_numpy(dtype=float)).all():
        return False, ["missing, nonnumeric or infinite volume"]
    if (vol < 0).any():
        return False, [f"{int((vol < 0).sum())} bars with negative volume"]
    if vol.fillna(0).eq(0).all():
        issues.append("volume is zero on every bar")

    return True, issues


def describe(df) -> str:
    """One-line summary for logs."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return "empty"
    return (f"{len(df)} rows {df.index.min().date()}→{df.index.max().date()}, "
            f"last close {df['Close'].iloc[-1]:.2f}")


def self_test(symbol: str = "RELIANCE") -> dict:
    """
    Fetch `symbol` from every configured source and validate each result.
    Run at startup or on demand to confirm the data layer is sound.
    """
    results: dict[str, dict] = {}

    def _record(name, df, intraday=False):
        ok, issues = validate_ohlcv(df, symbol, name, intraday=intraday)
        results[name] = {"ok": ok, "issues": issues, "summary": describe(df)}

    try:
        import groww_client
        if groww_client.is_connected() and not groww_client.data_forbidden():
            _record("groww_daily", groww_client.get_daily_bars(symbol, 550))
            _record("groww_intraday",
                    groww_client.get_intraday_candles(symbol, 15, days=5),
                    intraday=True)
    except Exception as e:
        results["groww"] = {"ok": False, "issues": [str(e)], "summary": "-"}

    try:
        import dhan_client
        if dhan_client.is_configured() and not dhan_client.data_forbidden():
            _record("dhan_daily", dhan_client.get_daily_bars(symbol, 550))
    except Exception as e:
        results["dhan"] = {"ok": False, "issues": [str(e)], "summary": "-"}

    try:
        import scanner
        _record("nselib", scanner._nselib_ohlcv(symbol))
    except Exception as e:
        results["nselib"] = {"ok": False, "issues": [str(e)], "summary": "-"}

    return results


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    import config, groww_client
    groww_client.init_groww(config.GROWW_API_KEY, config.GROWW_API_SECRET)

    print("\nData-layer self test — RELIANCE\n" + "─" * 62)
    for name, r in self_test().items():
        mark = "PASS" if r["ok"] else "FAIL"
        print(f"  [{mark}] {name:16} {r['summary']}")
        for i in r["issues"]:
            print(f"         ! {i}")
    print()
