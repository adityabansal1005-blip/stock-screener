"""
Scoring v2 — an absolute bar, not a ranking.

Why v1 is being replaced
------------------------
v1 was measured this session and its top end is the broken part: the top decile
underperformed the universe average, a top-20 portfolio returned ~0% while a
wider basket returned +8%, and six stocks the user named as winners (Ather,
Hindusthan Insulators, Morepen, Shilpa, Sudeep, Aye) all scored below the buy
threshold — Morepen at 45 ("AVOID") before rising 142%.

The causes were structural, not tuning:
  * ~90% of the score came from short-horizon technicals, the factor family
    with the WEAKEST empirical support. Fundamentals were 6 points of 79.
  * Extension was penalised unconditionally, with no test of whether the move
    was justified. That systematically marks down stocks that have started
    working — precisely the multibagger profile.
  * Regime conditioning taxed scores using a classifier measured at 52%
    accuracy with inverted separation.

Design principles agreed with the user
--------------------------------------
  1. ABSOLUTE BAR, never top-N. A relative ranking always returns a list, even
     when everything on it is bad. If nothing clears the bar, the answer is
     nothing — cash is a position.
  2. NO fixed counts, allocations or horizons. The system describes; the user
     allocates.
  3. Sector-aware. The same metric means different things across sectors —
     debt/equity is distress for a manufacturer and business-as-usual for a
     lender.
  4. Momentum is confirmation, never penalty.
  5. Hard limits only where they prevent ruin (liquidity), not to shape returns.

Pass/fail bar for this module: its qualifying set must beat the universe
average out-of-sample. v1's did not. Nothing here is accepted because it
sounds right.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

# ── Sector handling ─────────────────────────────────────────────────────────
_FIN = ("financial", "bank", "insurance", "capital markets", "credit", "nbfc")


def _is_financial(sector: str | None, industry: str | None = None) -> bool:
    s = f"{sector or ''} {industry or ''}".lower()
    return any(k in s for k in _FIN)


def _num(v, default=None):
    """Coerce to float, treating NaN/None/garbage as missing."""
    try:
        f = float(v)
        return default if math.isnan(f) or math.isinf(f) else f
    except (TypeError, ValueError):
        return default


def _debt_equity(v) -> float | None:
    """Read a ratio normalized by the provider adapter; never guess units."""
    d = _num(v)
    if d is None:
        return None
    # Source adapters supply ratios. Never infer units from magnitude.
    return d


# ── Factor 1: Quality ───────────────────────────────────────────────────────

def quality(res: dict) -> tuple[float | None, list[str]]:
    """
    Return on capital and balance-sheet safety, sector-adjusted.
    Returns (0-100, notes) or (None, ...) when there is too little to judge.
    """
    roe = _num(res.get("roe"))
    de = _debt_equity(res.get("debt_equity"))
    nm = _num(res.get("net_margin"))
    fin = _is_financial(res.get("sector"), res.get("industry"))
    notes, pts, weight = [], 0.0, 0.0

    if roe is not None:
        weight += 50
        if roe >= 20:   pts += 50; notes.append(f"ROE {roe:.0f}% (strong)")
        elif roe >= 15: pts += 40; notes.append(f"ROE {roe:.0f}% (good)")
        elif roe >= 10: pts += 25
        elif roe >= 5:  pts += 10
        else:           notes.append(f"ROE {roe:.0f}% (weak)")

    # Leverage is only a risk signal for non-lenders.
    if de is not None and not fin:
        weight += 30
        if de < 0.3:   pts += 30; notes.append("low debt")
        elif de < 1.0: pts += 22
        elif de < 2.0: pts += 12
        elif de < 3.0: pts += 5
        else:          notes.append(f"D/E {de:.1f} (high)")
    elif fin:
        weight += 30
        pts += 18          # neutral: leverage is the business model

    if nm is not None:
        weight += 20
        bar_hi, bar_lo = (25, 10) if fin else (15, 3)
        if nm >= bar_hi:  pts += 20; notes.append(f"margin {nm:.0f}%")
        elif nm >= bar_lo: pts += 12
        else:              notes.append(f"margin {nm:.0f}% (thin)")

    if weight < 40:
        return None, ["insufficient fundamental data"]
    return round(pts / weight * 100, 1), notes


# ── Factor 2: Growth ────────────────────────────────────────────────────────

def growth(res: dict) -> tuple[float | None, list[str]]:
    """
    Revenue and earnings expansion. This is the factor that answers "is the
    move justified" — extension backed by accelerating growth is a re-rating,
    extension without it is a squeeze. v1 could not tell them apart.
    """
    rev = _num(res.get("revenue_growth"))
    earn = _num(res.get("earnings_growth"))
    notes, pts, weight = [], 0.0, 0.0

    if rev is not None:
        weight += 50
        if rev >= 25:   pts += 50; notes.append(f"revenue +{rev:.0f}%")
        elif rev >= 15: pts += 38; notes.append(f"revenue +{rev:.0f}%")
        elif rev >= 8:  pts += 22
        elif rev >= 0:  pts += 8
        else:           notes.append(f"revenue {rev:.0f}% (shrinking)")

    if earn is not None:
        weight += 50
        if earn >= 30:   pts += 50; notes.append(f"earnings +{earn:.0f}%")
        elif earn >= 15: pts += 38
        elif earn >= 5:  pts += 22
        elif earn >= 0:  pts += 8
        else:            notes.append(f"earnings {earn:.0f}% (falling)")

    # A single metric must not produce a perfect score. `earnings_growth` is
    # missing for most NSE names, so scoring on revenue alone let RELIANCE —
    # a refiner — reach 100/100 growth off one number with nothing to
    # corroborate it. Uncorroborated evidence is capped, not rewarded.
    if weight < 40:
        return None, ["no growth data"]
    raw = pts / weight * 100
    if weight < 100:
        raw = min(raw, 70)
        notes.append("single metric only — capped")
    return round(raw, 1), notes


# ── Factor 3: Momentum (12-1) ───────────────────────────────────────────────

def momentum(df: pd.DataFrame, nifty_ret_12m: float | None = None
             ) -> tuple[float | None, list[str]]:
    """
    The canonical academic momentum factor: 12-month return EXCLUDING the most
    recent month (the last month tends to mean-revert). Robust across decades
    and markets. Replaces v1's RSI bands and extension penalty entirely —
    there is no penalty here for having already moved.
    """
    if df is None or len(df) < 260:
        return None, ["under 12 months of history"]
    c = pd.to_numeric(df["Close"], errors="coerce").dropna()
    if len(c) < 260:
        return None, ["under 12 months of history"]

    r12_1 = (c.iloc[-21] / c.iloc[-252] - 1) * 100      # 12m ago -> 1m ago
    r3 = (c.iloc[-1] / c.iloc[-63] - 1) * 100
    notes = [f"12-1m {r12_1:+.0f}%"]

    pts = 0.0
    if r12_1 >= 60:   pts += 55
    elif r12_1 >= 30: pts += 45
    elif r12_1 >= 15: pts += 32
    elif r12_1 >= 0:  pts += 18
    elif r12_1 >= -20: pts += 6

    if r3 >= 15:   pts += 25; notes.append(f"3m {r3:+.0f}%")
    elif r3 >= 5:  pts += 18
    elif r3 >= -5: pts += 10

    if nifty_ret_12m is not None:
        rs = r12_1 - nifty_ret_12m
        notes.append(f"vs index {rs:+.0f}%")
        if rs >= 25:  pts += 20
        elif rs >= 10: pts += 14
        elif rs >= 0:  pts += 8

    return round(min(pts, 100), 1), notes


# ── Factor 4: Participation ─────────────────────────────────────────────────

def participation(df: pd.DataFrame, res: dict) -> tuple[float | None, list[str]]:
    """Volume expansion and delivery — evidence real buyers are arriving."""
    if df is None or len(df) < 100:
        return None, ["insufficient history"]
    v = pd.to_numeric(df["Volume"], errors="coerce").fillna(0)
    recent, prior = v.tail(20).mean(), v.tail(80).head(60).mean()
    if prior <= 0:
        return None, ["no volume"]

    expansion = recent / prior
    notes = [f"volume {expansion:.1f}x"]
    pts = 0.0
    if expansion >= 2.5:   pts += 60
    elif expansion >= 1.6: pts += 48
    elif expansion >= 1.2: pts += 34
    elif expansion >= 0.9: pts += 20
    else:                  notes.append("volume fading")

    delv = _num(res.get("delivery_pct"))
    if delv is not None:
        if delv >= 55:   pts += 40; notes.append(f"delivery {delv:.0f}%")
        elif delv >= 40: pts += 28
        elif delv >= 25: pts += 15
    else:
        pts += 22            # neutral when unavailable, never a penalty

    return round(min(pts, 100), 1), notes


# ── Classification + absolute bars ──────────────────────────────────────────
# Deliberately absolute. A relative cut ("top 20") always returns a list even
# when nothing on it is good; these bars return nothing when nothing qualifies.

BARS = {
    "COMPOUNDER": {"quality": 60, "growth": 55, "momentum": 25},
    "MOMENTUM":   {"momentum": 60, "participation": 50, "quality": 35},
    "TURNAROUND": {"growth": 70, "participation": 55, "momentum": 40},
}


def score_v2(res: dict, df: pd.DataFrame,
             nifty_ret_12m: float | None = None) -> dict:
    """
    Score one stock. `qualifies` lists every category whose bar it clears —
    a stock may clear several, or none. None is a valid and common answer.
    """
    q, qn = quality(res)
    g, gn = growth(res)
    m, mn = momentum(df, nifty_ret_12m)
    p, pn = participation(df, res)

    factors = {"quality": q, "growth": g, "momentum": m, "participation": p}
    known = {k: v for k, v in factors.items() if v is not None}

    qualifies = []
    for name, bar in BARS.items():
        # A category cannot be claimed on missing evidence.
        if any(factors.get(k) is None for k in bar):
            continue
        if all(factors[k] >= v for k, v in bar.items()):
            qualifies.append(name)

    composite = round(sum(known.values()) / len(known), 1) if known else None

    return {
        "v2_quality": q, "v2_growth": g,
        "v2_momentum": m, "v2_participation": p,
        "v2_composite": composite,
        "v2_coverage": round(len(known) / 4 * 100),
        "v2_qualifies": qualifies,
        "v2_notes": qn + gn + mn + pn,
    }
