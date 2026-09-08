"""
Liquidity gate — refuse to recommend a position you cannot actually exit.

Why
---
Position sizing currently risks 2% of the account and caps a single holding at
15% of it — ~Rs 75,000 on a Rs 5L account — with no reference to whether the
stock trades enough to absorb that. Hindusthan Insulators (BSE 539984) turns
over about Rs 3.5 lakh on a median day, so Rs 75,000 is 21% of the entire
day's volume. At that size you are the market: you push the price up buying
and drop it selling, and an urgent exit takes days.

The backtest assumes 0.05% slippage per leg. On a name like that the true cost
is several percent, which would erase any edge the scanner ever found.

This module answers two questions:
  1. Is this stock liquid enough to appear at all?
  2. Given its liquidity, what is the largest position that is actually safe?

Rule of thumb applied: a single order should stay under ~2.5% of median daily
turnover. Above ~5% market impact becomes a first-order cost rather than a
rounding error.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Below this a stock is not investable at retail size — it is a quote, not a market.
MIN_MEDIAN_TURNOVER = 2_000_000      # Rs 20 lakh/day
# Share of median daily turnover a single position may represent.
MAX_TURNOVER_SHARE = 0.025           # 2.5%
# Stocks that stop trading for long stretches cannot be exited on schedule.
MIN_TRADING_DAYS_PCT = 80            # % of sessions with non-zero volume
LOOKBACK = 60                        # sessions used to measure


def measure(df: pd.DataFrame, lookback: int = LOOKBACK) -> dict:
    """
    Liquidity profile from an OHLCV frame (Capitalised columns).
    Returns medians rather than means — a single block deal should not make an
    illiquid stock look tradeable.
    """
    out = {"median_turnover": None, "median_volume": None,
           "trading_days_pct": None, "bars": 0}
    if df is None or len(df) < 20:
        return out
    d = df.tail(lookback)
    vol = pd.to_numeric(d.get("Volume"), errors="coerce").fillna(0)
    close = pd.to_numeric(d.get("Close"), errors="coerce")
    turn = (vol * close).replace([np.inf, -np.inf], np.nan).dropna()
    if turn.empty:
        return out
    out.update({
        "median_turnover": float(turn.median()),
        "median_volume": float(vol.median()),
        "trading_days_pct": float((vol > 0).mean() * 100),
        "bars": len(d),
    })
    return out


def max_position_value(median_turnover: float | None) -> float | None:
    """Largest rupee position that stays within MAX_TURNOVER_SHARE of a day."""
    if not median_turnover or median_turnover <= 0:
        return None
    return median_turnover * MAX_TURNOVER_SHARE


def assess(df: pd.DataFrame, intended_value: float | None = None) -> dict:
    """
    Full verdict. `tradeable` False means the stock should not be surfaced as
    an opportunity at all — no score is worth a position you cannot unwind.
    """
    m = measure(df)
    t = m["median_turnover"]
    cap = max_position_value(t)

    reasons = []
    if t is None:
        reasons.append("no turnover data")
    else:
        if t < MIN_MEDIAN_TURNOVER:
            reasons.append(f"turnover Rs{t:,.0f}/day below Rs{MIN_MEDIAN_TURNOVER:,} floor")
        if (m["trading_days_pct"] or 0) < MIN_TRADING_DAYS_PCT:
            reasons.append(f"trades only {m['trading_days_pct']:.0f}% of sessions")

    result = {
        **m,
        "max_position_value": round(cap) if cap else None,
        "tradeable": not reasons,
        "reasons": reasons,
    }

    if intended_value and t:
        share = intended_value / t * 100
        result["intended_share_pct"] = round(share, 1)
        result["impact_risk"] = (
            "severe" if share > 10 else
            "high" if share > 5 else
            "moderate" if share > 2.5 else "low"
        )
        # Shrink the position to what the market can absorb rather than
        # refusing outright — a good setup in a thin stock is still tradeable
        # at the right size.
        result["suggested_value"] = round(min(intended_value, cap)) if cap else None
    return result


def apply_to_result(result: dict, df: pd.DataFrame) -> dict:
    """
    Attach liquidity fields to a scan result and shrink position sizing to fit.
    Mutates and returns `result`.
    """
    intended = result.get("position_value")
    a = assess(df, intended_value=intended)

    result["liq_turnover"] = round(a["median_turnover"]) if a["median_turnover"] else None
    result["liq_trading_days_pct"] = round(a["trading_days_pct"]) if a["trading_days_pct"] else None
    result["liq_tradeable"] = a["tradeable"]
    result["liq_reasons"] = a["reasons"]
    result["liq_max_position"] = a["max_position_value"]
    result["liq_impact_risk"] = a.get("impact_risk")

    # Resize the position to something the market can actually absorb.
    sugg = a.get("suggested_value")
    close = result.get("close") or 0
    if sugg and intended and sugg < intended and close > 0:
        result["position_value_unconstrained"] = intended
        result["position_value"] = sugg
        result["position_size"] = max(int(sugg // close), 0)
        result["position_capped_by_liquidity"] = True
    else:
        result["position_capped_by_liquidity"] = False
    return result


if __name__ == "__main__":
    import sys
    import warnings
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")

    import config, groww_client
    groww_client.init_groww(config.GROWW_API_KEY, config.GROWW_API_SECRET)

    tests = [("RELIANCE", "NSE mega-cap"), ("HFCL", "NSE mid"),
             ("539984", "BSE-only micro (Hindusthan Insulators)"),
             ("530689", "BSE-only micro (Lykis)")]
    print(f"\n  {'symbol':<10}{'turnover/day':>16}{'days%':>8}{'max pos':>12}"
          f"{'ok':>5}  note")
    print("  " + "-" * 74)
    for sym, note in tests:
        df = groww_client.get_daily_bars(sym, 120)
        a = assess(df, intended_value=75_000)
        t = f"Rs{a['median_turnover']:,.0f}" if a["median_turnover"] else "—"
        mp = f"Rs{a['max_position_value']:,}" if a["max_position_value"] else "—"
        print(f"  {sym:<10}{t:>16}{(a['trading_days_pct'] or 0):>7.0f}%{mp:>12}"
              f"{'YES' if a['tradeable'] else 'NO':>5}  {note}")
        if a["reasons"]:
            print(f"    {'':8}-> {'; '.join(a['reasons'])}")
        if a.get("intended_share_pct"):
            print(f"    {'':8}-> Rs75,000 = {a['intended_share_pct']}% of a day "
                  f"({a['impact_risk']} impact)")
    print()
