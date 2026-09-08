"""
Survivorship-bias tooling — point-in-time universes from NSE bhavcopy.

The problem: every backtest in this project draws its universe from *today's*
index membership. Stocks that were trading during the study window but have
since been delisted, suspended, or dropped are invisible. Because the missing
names are disproportionately the failures, the bias is always favourable.

Published work on NIFTY Smallcap 250 puts the effect at roughly a 23%
overstatement of annual returns (26.17% survivor-only vs 21.23% true universe).

The fix: NSE bhavcopy lists every instrument that actually traded on a given
date, including names that later vanished. Union that historical universe with
today's and the dead names come back into the sample.

Limitation worth knowing: nselib only serves bhavcopy from 2024 onward (NSE
retired the older format), so point-in-time reconstruction cannot reach further
back than that with free sources.

    python survivorship.py          # measure the bias in this project
"""

from __future__ import annotations

import datetime
import functools

_EQ_SERIES = "EQ"


def _bhav_eq(date_str: str) -> set[str]:
    """EQ-series tickers that actually traded on `date_str` (DD-MM-YYYY)."""
    from nselib import capital_market as cm
    df = cm.bhav_copy_equities(date_str)
    if df is None or len(df) == 0 or "SctySrs" not in df.columns:
        return set()
    eq = df[df["SctySrs"].astype(str).str.strip() == _EQ_SERIES]
    return set(eq["TckrSymb"].astype(str).str.strip().str.upper())


@functools.lru_cache(maxsize=32)
def pit_universe(date_str: str) -> frozenset[str]:
    """Cached point-in-time EQ universe for one date (DD-MM-YYYY)."""
    try:
        return frozenset(_bhav_eq(date_str))
    except Exception as e:
        print(f"[Survivorship] bhavcopy failed for {date_str}: {e}")
        return frozenset()


def last_trading_date(max_lookback: int = 10) -> str:
    """
    Most recent date with a published bhavcopy, as DD-MM-YYYY.

    Calling this with today's date on a weekend or holiday returns an EMPTY
    universe, which would make every listed stock look like it had vanished.
    Walk back until a real trading day is found.
    """
    d = datetime.date.today()
    for _ in range(max_lookback):
        s = d.strftime("%d-%m-%Y")
        if pit_universe(s):
            return s
        d -= datetime.timedelta(days=1)
    return datetime.date.today().strftime("%d-%m-%Y")


def vanished_since(past_date: str, recent_date: str) -> set[str]:
    """
    Tickers trading on `past_date` but not on `recent_date`.

    NOTE: this over-counts. A ticker also disappears from the EQ series when it
    is renamed, merged, or moved to another series — none of which are failures.
    Treat the result as an upper bound and verify with `has_history()` before
    drawing conclusions about any individual name.
    """
    past, recent = pit_universe(past_date), pit_universe(recent_date)
    if not past or not recent:
        return set()
    return set(past) - set(recent)


def has_history(symbol: str, min_rows: int = 60) -> int:
    """
    Rows of daily history retrievable for `symbol`, 0 if none.
    A delisted name typically returns a truncated series ending on its last
    trading day — which is exactly what a survivorship-free backtest needs.
    """
    try:
        import groww_client
        if groww_client.is_connected() and not groww_client.data_forbidden():
            df = groww_client.get_daily_bars(symbol, 550)
            if df is not None and len(df) >= min_rows:
                return len(df)
    except Exception:
        pass
    try:
        import scanner
        df = scanner._nselib_ohlcv(symbol)
        if df is not None and len(df) >= min_rows:
            return len(df)
    except Exception:
        pass
    return 0


def build_unbiased_universe(base: list[str], past_date: str,
                            recent_date: str | None = None,
                            verify: bool = False) -> tuple[list[str], dict]:
    """
    Union `base` (today's index members) with names that were trading on
    `past_date` but have since vanished — the stocks a survivor-only backtest
    silently drops.

    Set `verify=True` to keep only vanished names whose history is actually
    retrievable. That is slower but avoids padding the universe with symbols
    that contribute no data.

    Returns (universe, stats).
    """
    if recent_date is None:
        recent_date = last_trading_date()

    clean = [s.replace(".NS", "").replace(".BO", "").upper() for s in base]
    base_set = set(clean)

    gone = vanished_since(past_date, recent_date)
    added = sorted(gone - base_set)

    verified = None
    if verify and added:
        verified = [s for s in added if has_history(s)]
        added = verified

    stats = {
        "base_size":        len(base_set),
        "vanished_total":   len(gone),
        "added":            len(added),
        "verified":         verified is not None,
        "past_date":        past_date,
        "recent_date":      recent_date,
        "pit_universe_size": len(pit_universe(past_date)),
    }
    return sorted(base_set | set(added)), stats


def measure_bias(past_date: str = "02-01-2024",
                 recent_date: str | None = None,
                 universe: str = "nifty500",
                 sample: int = 25) -> dict:
    """Quantify how much of the historical universe the study never sees."""
    from stocks_nse import get_all_symbols

    if recent_date is None:
        recent_date = last_trading_date()

    past   = pit_universe(past_date)
    recent = pit_universe(recent_date)
    study  = {s.replace(".NS", "").replace(".BO", "").upper()
              for s in get_all_symbols(universe)}

    gone = set(past) - set(recent)

    # Sample the vanished names to estimate how many are genuinely gone
    # rather than renamed, and how many still yield usable history.
    checked = sorted(gone)[:sample]
    retrievable = sum(1 for s in checked if has_history(s))

    return {
        "pit_universe":       len(past),
        "current_universe":   len(recent),
        "vanished":           len(gone),
        "vanished_pct":       round(len(gone) / len(past) * 100, 1) if past else 0,
        "study_universe":     len(study),
        "study_live_then":    len(study & set(past)),
        "study_coverage_pct": round(len(study & set(past)) / len(past) * 100, 1) if past else 0,
        "never_seen":         len(set(past) - study),
        "sampled":            len(checked),
        "sample_retrievable": retrievable,
    }


if __name__ == "__main__":
    import sys
    import warnings
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")

    import config, groww_client
    groww_client.init_groww(config.GROWW_API_KEY, config.GROWW_API_SECRET)

    print("\nSurvivorship bias — measured\n" + "─" * 58)
    m = measure_bias()
    print(f"  point-in-time universe (Jan-2024) : {m['pit_universe']:,}")
    print(f"  current universe                  : {m['current_universe']:,}")
    print(f"  vanished since                    : {m['vanished']:,} "
          f"({m['vanished_pct']}%, upper bound — includes renames)")
    print()
    print(f"  study universe                    : {m['study_universe']:,}")
    print(f"  of those, live in Jan-2024        : {m['study_live_then']:,}")
    print(f"  study covers                      : {m['study_coverage_pct']}% "
          f"of the then-live market")
    print(f"  never seen by the study           : {m['never_seen']:,}")
    print()
    print(f"  sampled {m['sampled']} vanished names → "
          f"{m['sample_retrievable']} have retrievable history "
          f"({m['sample_retrievable'] / max(m['sampled'], 1) * 100:.0f}%)")
    print()
