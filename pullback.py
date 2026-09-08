"""
Pullback quality — is this dip a rest or a top?

Why this exists
---------------
The scanner scores entries. It has never had an opinion about a position it is
already holding, so every drawdown looks identical: a red number. That is the
gap that cost the most in the 30 Aug review.

MOREPENLAB fell ~16% from its July high. Holding through it was right — the
stock went on to a new high. The dip was distinguishable at the time, from four
numbers the system was not computing:

    decline volume      0.42x the volume of the advance that preceded it
    50-day average      held  (close 56.7 vs MA 53.0)
    pre-breakout base   held  (close 51.4 vs base 43.5)
    vs the index        -9.4% while Nifty was -0.2%

Three of those four said "rest". The fourth is the interesting one and is
handled honestly below.

What each test is actually claiming
-----------------------------------
T1  VOLUME    A decline on lighter volume than the advance is supply drying up,
              not supply arriving. This is the oldest test in the book and the
              one that carried the most weight on Morepen.
T2  TREND     Price still above a rising 50-day average means the intermediate
              trend that produced the gain is intact. Losing it is the single
              most common definition of "the move is over".
T3  STRUCTURE How much of the advance has been given back, and is price still
              above the base it broke out of. Below the base, the breakout is
              simply undone.
T4  RELATIVE  Stock's decline against the index over the same days. Reported as
              CONTEXT ONLY and never scored. On the one case examined it
              pointed the WRONG way — Morepen's dip was sharply stock-specific
              and it recovered anyway. A single case cannot establish a rule in
              either direction, so this module refuses to pretend it has one.
T5  SUPPLY    Distribution days: sessions in the decline closing down over 1.5%
              on above-average volume. Institutions leaving leave footprints.

The verdict is the count of T1/T2/T3/T5, nothing cleverer. No weights were
fitted, because fitting weights on the handful of cases we have examined would
produce a number that looks precise and means nothing.

UNVALIDATED. These four tests have not been scored against outcomes across a
population — only reconstructed on cases already known to have recovered, which
is the weakest possible evidence. Treat the output as a structured description
of the dip, not a probability that it ends well.

    python pullback.py MOREPENLAB
    python pullback.py MOREPENLAB SHILPAMED SUDEEPPHAR
    python pullback.py --watchlist
"""

from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

LOOKBACK = 120          # sessions searched for the swing high / advance base
MIN_DECLINE_PCT = 3.0   # below this there is no pullback to assess

_PASS, _WARN, _FAIL, _NA = "PASS", "WARN", "FAIL", "N/A"


def _swing_low(df: pd.DataFrame, peak: int, floor: int, k: int = 10) -> int:
    """
    Start of the advance that produced the peak: the most recent SWING low
    before it — a bar whose close is the lowest within k bars either side.

    Not the minimum of the whole window, which was the first version and was
    wrong in a way that inverted a verdict. On MOREPENLAB the window minimum
    sat months back in quiet base-building, so the "advance" averaged in weeks
    of low volume and the July decline scored 1.35x — heavy selling — when
    measured against the actual up-leg it was 0.76x. A leg definition that
    reaches back further than the move it is describing will flag every active
    stock as distributing.
    """
    for i in range(peak - 1, floor + k - 1, -1):
        lo, hi = max(floor, i - k), min(peak, i + k + 1)
        if float(df["close"].iloc[i]) == float(df["close"].iloc[lo:hi].min()):
            return i
    return floor + int(df["close"].iloc[floor:peak + 1].values.argmin())


def _norm(df: pd.DataFrame) -> pd.DataFrame | None:
    """Accept either the lowercase data_manager frame or the scanner's."""
    if df is None or df.empty:
        return None
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    need = {"high", "low", "close", "volume"}
    if not need.issubset(out.columns):
        return None
    for c in need:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.dropna(subset=["close"])


def _load(symbol: str) -> pd.DataFrame | None:
    try:
        import data_manager
        return _norm(data_manager.load(symbol))
    except Exception:
        return None


# ── The tests ───────────────────────────────────────────────────────────────

def _t_volume(df: pd.DataFrame, lo: int, peak: int) -> tuple[str, str, float | None]:
    adv = df["volume"].iloc[lo:peak + 1]
    dec = df["volume"].iloc[peak:]
    if len(adv) < 3 or len(dec) < 2 or adv.mean() <= 0:
        return _NA, "not enough sessions to compare volume", None
    ratio = float(dec.mean() / adv.mean())
    if ratio <= 0.8:
        return _PASS, f"decline volume {ratio:.2f}x the advance — supply drying up", ratio
    if ratio <= 1.2:
        return _WARN, f"decline volume {ratio:.2f}x the advance — no clear signal", ratio
    return _FAIL, f"decline volume {ratio:.2f}x the advance — heavier selling than buying", ratio


def _t_trend(df: pd.DataFrame) -> tuple[str, str, dict]:
    c = df["close"]
    if len(c) < 60:
        return _NA, "under 60 sessions of history", {}
    ma = c.rolling(50).mean()
    last, m = float(c.iloc[-1]), float(ma.iloc[-1])
    slope = float(ma.iloc[-1] - ma.iloc[-11]) if len(ma.dropna()) > 11 else 0.0
    above, rising = last > m, slope >= 0
    d = {"close": last, "ma50": round(m, 2), "above_ma50": above,
         "ma50_rising": rising}
    if above and rising:
        return _PASS, f"above a rising 50-day ({last:,.1f} vs {m:,.1f})", d
    if above:
        return _WARN, f"above the 50-day but it has rolled over ({m:,.1f})", d
    if rising:
        return _WARN, f"below the 50-day ({last:,.1f} vs {m:,.1f}) but it is still rising", d
    return _FAIL, f"below a falling 50-day ({last:,.1f} vs {m:,.1f})", d


def _t_structure(df: pd.DataFrame, lo: int, peak: int) -> tuple[str, str, dict]:
    base = float(df["close"].iloc[lo])
    top = float(df["close"].iloc[peak])
    last = float(df["close"].iloc[-1])
    advance = top - base
    if advance <= 0:
        return _NA, "no measurable advance to retrace", {}
    retrace = (top - last) / advance * 100
    d = {"base": round(base, 2), "peak": round(top, 2),
         "retrace_pct": round(retrace, 1), "above_base": last > base}
    if last <= base:
        return _FAIL, f"back below the base it rose from (₹{base:,.1f}) — the move is undone", d
    if retrace <= 38:
        return _PASS, f"gave back {retrace:.0f}% of the advance — shallow", d
    if retrace <= 62:
        return _WARN, f"gave back {retrace:.0f}% of the advance — normal but deep", d
    return _FAIL, f"gave back {retrace:.0f}% of the advance — most of the move is gone", d


def _t_supply(df: pd.DataFrame, peak: int) -> tuple[str, str, int]:
    """Distribution days inside the decline."""
    dec = df.iloc[peak:]
    if len(dec) < 3:
        return _NA, "decline too short to count distribution", 0
    avg_v = float(df["volume"].iloc[max(0, peak - 50):peak].mean())
    if not np.isfinite(avg_v) or avg_v <= 0:
        return _NA, "no volume baseline", 0
    chg = dec["close"].pct_change() * 100
    hits = int(((chg <= -1.5) & (dec["volume"] > avg_v)).sum())
    if hits == 0:
        return _PASS, "no heavy-volume down days", 0
    if hits <= 2:
        return _WARN, f"{hits} heavy-volume down day(s)", hits
    return _FAIL, f"{hits} heavy-volume down days — that is distribution", hits


def _t_relative(df: pd.DataFrame, peak: int) -> tuple[str, dict]:
    """Context only — deliberately unscored. See the module docstring."""
    stock = (float(df["close"].iloc[-1]) / float(df["close"].iloc[peak]) - 1) * 100
    try:
        import data_manager
        n = _norm(data_manager.load("^NSEI"))
    except Exception:
        n = None
    if n is None or n.empty:
        return f"stock {stock:+.1f}% from its high; index unavailable", {"stock_pct": round(stock, 1)}

    start = df.index[peak]
    nf = n[n.index >= start]
    if len(nf) < 2:
        return f"stock {stock:+.1f}% from its high; index window too short", {"stock_pct": round(stock, 1)}
    idx = (float(nf["close"].iloc[-1]) / float(nf["close"].iloc[0]) - 1) * 100
    kind = "market-wide" if idx <= -3 else "stock-specific"
    return (f"stock {stock:+.1f}% vs index {idx:+.1f}% — {kind} "
            f"(context only, not scored)",
            {"stock_pct": round(stock, 1), "index_pct": round(idx, 1),
             "kind": kind})


# ── Assessment ──────────────────────────────────────────────────────────────

def assess(symbol: str, df: pd.DataFrame | None = None,
           entry_price: float | None = None,
           lookback: int = LOOKBACK) -> dict:
    """
    Describe the current pullback in one held name.

    The swing high is the highest CLOSE in the lookback window; the advance is
    measured from the lowest close BEFORE that high. Closes rather than highs
    deliberately — a single intraday spike should not define either end of a
    leg the rest of the tests are measured against.
    """
    sym = symbol.strip().upper()
    d = _norm(df) if df is not None else _load(sym)
    if d is None or len(d) < 60:
        return {"symbol": sym, "status": "NO_DATA",
                "note": "under 60 sessions of usable history"}

    w = d.tail(lookback)
    offset = len(d) - len(w)
    peak_rel = int(w["close"].values.argmax())
    peak = offset + peak_rel
    if peak == 0:
        return {"symbol": sym, "status": "NO_PULLBACK",
                "note": "the window opens at its own high — nothing to measure"}

    lo = _swing_low(d, peak, offset)
    top = float(d["close"].iloc[peak])
    last = float(d["close"].iloc[-1])
    drawdown = (last / top - 1) * 100
    days_since = len(d) - 1 - peak

    head = {
        "symbol": sym,
        "close": round(last, 2),
        "peak": round(top, 2),
        "peak_date": str(d.index[peak].date()),
        "days_since_peak": days_since,
        "drawdown_pct": round(drawdown, 2),
    }
    if entry_price:
        head["entry"] = entry_price
        head["vs_entry_pct"] = round((last / entry_price - 1) * 100, 2)
        head["above_entry"] = last > entry_price

    if drawdown > -MIN_DECLINE_PCT:
        head.update({"status": "NO_PULLBACK",
                     "note": f"only {drawdown:.1f}% off the high — at or near highs"})
        return head

    v_s, v_n, v_ratio = _t_volume(d, lo, peak)
    t_s, t_n, t_d = _t_trend(d)
    s_s, s_n, s_d = _t_structure(d, lo, peak)
    u_s, u_n, u_hits = _t_supply(d, peak)
    r_n, r_d = _t_relative(d, peak)

    tests = [
        {"id": "volume",    "label": "Decline vs advance volume", "status": v_s, "note": v_n},
        {"id": "trend",     "label": "50-day average",            "status": t_s, "note": t_n},
        {"id": "structure", "label": "Retracement and base",      "status": s_s, "note": s_n},
        {"id": "supply",    "label": "Distribution days",         "status": u_s, "note": u_n},
    ]
    scored = [t for t in tests if t["status"] != _NA]
    passes = sum(1 for t in scored if t["status"] == _PASS)
    fails = sum(1 for t in scored if t["status"] == _FAIL)

    if not scored:
        status, note = "UNKNOWN", "no test had enough data"
    elif fails == 0 and passes >= 3:
        status = "ORDERLY"
        note = "every test that could run says rest, not top"
    elif fails >= 2:
        status = "DETERIORATING"
        note = f"{fails} of {len(scored)} tests failed"
    elif fails == 1 and passes >= 2:
        status = "MIXED"
        note = "mostly intact, one test broken — the one to watch"
    else:
        status = "MIXED"
        note = f"{passes} pass / {fails} fail — no clear reading"

    head.update({
        "status": status,
        "note": note,
        "passes": passes,
        "fails": fails,
        "tests": tests,
        "context": {"relative": r_n, **r_d},
        "detail": {"volume_ratio": None if v_ratio is None else round(v_ratio, 2),
                   "distribution_days": u_hits, **t_d, **s_d},
    })
    return head


def assess_many(symbols: list[str], entries: dict[str, float] | None = None) -> list[dict]:
    entries = entries or {}
    out = []
    for s in symbols:
        try:
            out.append(assess(s, entry_price=entries.get(s.strip().upper())))
        except Exception as e:
            out.append({"symbol": s.strip().upper(), "status": "ERROR", "note": str(e)})
    order = {"DETERIORATING": 0, "MIXED": 1, "UNKNOWN": 2, "ORDERLY": 3,
             "NO_PULLBACK": 4, "NO_DATA": 5, "ERROR": 6}
    out.sort(key=lambda r: (order.get(r.get("status"), 9), r.get("drawdown_pct", 0)))
    return out


# ── CLI ─────────────────────────────────────────────────────────────────────

def show(r: dict) -> None:
    mark = {"ORDERLY": "OK  ", "MIXED": "MIX ", "DETERIORATING": "BAD ",
            "NO_PULLBACK": "--  ", "NO_DATA": "?   ", "UNKNOWN": "?   ",
            "ERROR": "!   "}.get(r["status"], "    ")
    print(f"\n{'─' * 66}")
    print(f"  {mark}{r['symbol']}   {r['status']}")
    if r["status"] in ("NO_DATA", "ERROR"):
        print(f"       {r.get('note', '')}")
        return
    print(f"       ₹{r['close']:,.1f}  |  {r['drawdown_pct']:+.1f}% from "
          f"₹{r['peak']:,.1f} ({r['peak_date']}, {r['days_since_peak']} sessions ago)")
    if "vs_entry_pct" in r:
        print(f"       vs entry ₹{r['entry']:,.1f}: {r['vs_entry_pct']:+.1f}%")
    if r["status"] == "NO_PULLBACK":
        print(f"       {r['note']}")
        return
    print()
    for t in r["tests"]:
        print(f"       [{t['status']:<4}] {t['label']:<28} {t['note']}")
    print(f"       [ctx ] {'Relative to index':<28} {r['context']['relative']}")
    print(f"\n       → {r['note']}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    args = [a for a in sys.argv[1:] if not a.startswith("-")]

    if "--watchlist" in sys.argv:
        from stocks_nse import load_watchlist
        args = load_watchlist()

    if not args:
        print(__doc__.strip().splitlines()[-4].strip())
        sys.exit(0)

    try:
        import config, groww_client
        groww_client.init_groww(config.GROWW_API_KEY, config.GROWW_API_SECRET)
    except Exception:
        pass

    print(f"\n{'=' * 66}")
    print(f"  PULLBACK QUALITY — {len(args)} position(s)")
    print(f"  UNVALIDATED: describes the dip, does not predict the outcome.")
    print("=" * 66)
    for r in assess_many(args):
        show(r)
    print()
