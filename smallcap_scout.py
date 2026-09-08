"""
Small-cap scout — find the profile, not the prediction.

Ather Energy and Hindusthan Insulators were not predicted by this system; they
were noticed afterwards. What they DID share is a describable profile, and a
screen for that profile is something we can build honestly:

  * small enough to be under-followed — outside the large-cap indices
  * liquid enough to actually trade — above the turnover floor, but well
    below mega-cap liquidity where nothing is under-researched
  * volume expanding — participation arriving before the crowd notices
  * price trending — above its own moving averages, outperforming the index

This is a SCREEN, not a forecast. Everything measured today says the scoring
has an information coefficient of ~0.12 at five days and no reliable edge at
three months, so nothing here should be read as "these will multiply". It
narrows ~2,700 stocks to a shortlist worth reading about. The research after
that is yours.

    python smallcap_scout.py            # show cached results
    python smallcap_scout.py scan       # run the screen (slow)
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_CACHE = Path(__file__).parent / "smallcap_scout.json"

# Under-followed band: liquid enough to exit, small enough to be overlooked.
MIN_TURNOVER = 2_000_000          # Rs 20 lakh/day  — below this you cannot exit
MAX_TURNOVER = 500_000_000        # Rs 50 crore/day — above this it is well covered
MIN_VOL_EXPANSION = 1.4           # recent 20d volume vs prior 60d
MIN_RS = 0.0                      # must at least match the index
EXCLUDE_LARGE_CAPS = True         # drop Nifty 100 members


def _profile(df: pd.DataFrame) -> dict | None:
    """Volume expansion, trend and strength from an OHLCV frame."""
    if df is None or len(df) < 120:
        return None
    c = pd.to_numeric(df["Close"], errors="coerce")
    v = pd.to_numeric(df["Volume"], errors="coerce").fillna(0)
    if c.isna().all() or c.iloc[-1] <= 0:
        return None

    recent_v = v.tail(20).mean()
    prior_v = v.tail(80).head(60).mean()
    if prior_v <= 0:
        return None

    ema50 = c.ewm(span=50, adjust=False).mean().iloc[-1]
    ema200 = c.ewm(span=200, adjust=False).mean().iloc[-1] if len(c) >= 200 else ema50

    return {
        "close": float(c.iloc[-1]),
        "vol_expansion": float(recent_v / prior_v),
        "above_ema50": bool(c.iloc[-1] > ema50),
        "above_ema200": bool(c.iloc[-1] > ema200),
        "ret_3m": float((c.iloc[-1] / c.iloc[-63] - 1) * 100) if len(c) > 63 else 0.0,
        "ret_1m": float((c.iloc[-1] / c.iloc[-21] - 1) * 100) if len(c) > 21 else 0.0,
        "from_high": float((c.iloc[-1] / c.tail(252).max() - 1) * 100),
    }


def _score(p: dict, liq: dict, rs: float) -> float:
    """
    Rank within the shortlist. Deliberately simple and unfitted — a tuned
    formula here would just be overfitting to the two examples that prompted it.
    """
    s = 0.0
    s += min(p["vol_expansion"], 4.0) * 12      # participation arriving
    s += 20 if p["above_ema50"] else 0
    s += 15 if p["above_ema200"] else 0
    s += max(min(rs, 40), -20) * 0.5            # strength vs index
    s += max(min(p["ret_3m"], 60), -30) * 0.3
    s += 10 if p["from_high"] > -15 else 0      # near its own highs
    return round(s, 1)


def scan(universe: str = "all_nse", include_bse: bool = True,
         limit: int | None = None) -> list[dict]:
    import concurrent.futures as cf
    import groww_client
    import liquidity
    from stocks_nse import get_all_symbols, NIFTY_50, NIFTY_NEXT50

    large = set(NIFTY_50) | set(NIFTY_NEXT50) if EXCLUDE_LARGE_CAPS else set()

    syms = [s.replace(".NS", "") for s in get_all_symbols(universe)]
    syms = [s for s in syms if s not in large]
    if include_bse:
        try:
            import bse_universe
            syms += [s.replace(".BO", "") for s in bse_universe.symbols()]
        except Exception:
            pass
    if limit:
        syms = syms[:limit]

    # Index baseline for relative strength
    try:
        import data_manager
        nif = data_manager.load("^NSEI")["close"]
        nif_3m = (nif.iloc[-1] / nif.iloc[-63] - 1) * 100
    except Exception:
        nif_3m = 0.0

    print(f"[Scout] screening {len(syms):,} names "
          f"(large caps excluded, index 3m {nif_3m:+.1f}%)")

    out, done = [], 0

    def _check(sym):
        try:
            df = groww_client.get_daily_bars(sym, 400)
            if df is None or len(df) < 120:
                return None
            liq = liquidity.assess(df)
            t = liq.get("median_turnover") or 0
            if not liq["tradeable"] or not (MIN_TURNOVER <= t <= MAX_TURNOVER):
                return None
            p = _profile(df)
            if p is None:
                return None
            if p["vol_expansion"] < MIN_VOL_EXPANSION or not p["above_ema50"]:
                return None
            rs = p["ret_3m"] - nif_3m
            if rs < MIN_RS:
                return None
            return {"symbol": sym, "exchange": "BSE" if sym.isdigit() else "NSE",
                    "turnover": round(t), "max_position": liq["max_position_value"],
                    "rs_vs_nifty": round(rs, 1), "score": _score(p, liq, rs),
                    **{k: round(v, 2) if isinstance(v, float) else v
                       for k, v in p.items()}}
        except Exception:
            return None

    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        for r in ex.map(_check, syms):
            done += 1
            if r:
                out.append(r)
            if done % 250 == 0:
                print(f"  {done}/{len(syms)} — {len(out)} passing")

    out.sort(key=lambda x: -x["score"])
    try:
        names = {}
        try:
            import bse_universe
            names = bse_universe.name_map()
        except Exception:
            pass
        for r in out:
            r["name"] = names.get(r["symbol"], r["symbol"])
        _CACHE.write_text(json.dumps({"built": time.time(), "rows": out}),
                          encoding="utf-8")
    except Exception:
        pass
    print(f"[Scout] {len(out)} names match the profile")
    return out


def load() -> list[dict]:
    if not _CACHE.exists():
        return []
    try:
        return json.loads(_CACHE.read_text(encoding="utf-8")).get("rows", [])
    except Exception:
        return []


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) > 1 and sys.argv[1] == "scan":
        import config, groww_client
        groww_client.init_groww(config.GROWW_API_KEY, config.GROWW_API_SECRET)
        lim = int(sys.argv[2]) if len(sys.argv) > 2 else None
        rows = scan(limit=lim)
    else:
        rows = load()
    if not rows:
        print("\n  nothing cached — run:  python smallcap_scout.py scan\n")
    else:
        print(f"\n  {len(rows)} names matching the small-cap accumulation profile\n")
        print(f"  {'symbol':<11}{'ex':<5}{'turnover/day':>15}{'volx':>7}"
              f"{'RS':>8}{'3m':>8}{'maxpos':>11}")
        print("  " + "-" * 66)
        for r in rows[:25]:
            print(f"  {r['symbol']:<11}{r['exchange']:<5}Rs{r['turnover']:>13,}"
                  f"{r['vol_expansion']:>7.1f}{r['rs_vs_nifty']:>+8.1f}"
                  f"{r['ret_3m']:>+8.1f}Rs{r['max_position']:>9,}")
        print("\n  A SCREEN, not a forecast. Volume expansion + trend + strength is a"
              "\n  profile worth researching, not evidence that a stock will rise.\n")
