"""
Exit strategy lab — hold the signal constant, vary only the exit.

Why this exists
---------------
Average favourable excursion on a signal is about +4.5%. Realised return is
about +0.17%. Roughly 96% of the move the scanner correctly identifies is
reached and then given back. That is an exit problem, not a signal problem,
and it is the largest single lever in the system.

Design
------
Signal generation is expensive (the full indicator suite, per bar, per symbol —
~40 minutes) while exit simulation is trivially cheap. So phase 1 walks the
history ONCE and caches every signal to disk; phase 2 replays any number of
exit rules against that fixed set in seconds. Because every rule sees the
identical signals, differences between them are attributable to the exit alone.

Discipline
----------
  * Signals are split by DATE into train and test. Rules are compared on train;
    the winner is then confirmed on test, which is never used for selection.
  * Every variant tried is counted and reported, so the multiple-testing cost
    is visible rather than hidden.
  * Entry is the next session's OPEN, matching backtester.py.

    python exit_lab.py build     # phase 1 — cache signals (slow, once)
    python exit_lab.py run       # phase 2 — sweep exits (fast, repeatable)
"""

from __future__ import annotations

import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_CACHE = Path(__file__).parent / "exit_lab_signals.pkl"
TRAIN_END = pd.Timestamp("2025-06-30")     # rules are chosen using data up to here
MIN_SCORE = 50
MAX_HOLD = 63                              # longest exit any variant may use
ANALYSIS_BARS = 2600                       # ~10 years — must span up AND down markets
STEP = 5                                   # wider step keeps the full-history build tractable

try:
    from backtester import COMMISSION, SLIPPAGE
    COST = (COMMISSION * 2 + SLIPPAGE * 2)
except Exception:
    COST = 0.005582


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — generate and cache signals
# ─────────────────────────────────────────────────────────────────────────────

def build_signals(symbols: list[str] | None = None, limit: int | None = None) -> int:
    """Walk history once, store every qualifying signal with its forward path."""
    import data_manager
    from signals import compute_signals
    from technical_enhanced import compute_enhanced, compute_score100
    from stocks_nse import get_all_symbols

    if symbols is None:
        symbols = [s.replace(".NS", "") for s in get_all_symbols("nifty500")]
    if limit:
        symbols = symbols[:limit]

    # Market context per date: what the INDEX did over the same forward window.
    # Stops are insurance — judging them only in a rising market is unfair to
    # them, so every signal is tagged with the market's realised direction and
    # results are reported separately for falling and rising conditions.
    nif = data_manager.load("^NSEI")
    mkt_fwd = {}
    if nif is not None:
        nc = nif["close"]
        f = (nc.shift(-MAX_HOLD) / nc - 1) * 100
        mkt_fwd = {d.normalize(): v for d, v in f.dropna().items()}

    out = []
    for si, sym in enumerate(symbols, 1):
        df = data_manager.load(f"{sym}.NS")
        if df is None or len(df) < 400:
            continue
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]

        start_i = max(200, len(df) - ANALYSIS_BARS - MAX_HOLD)
        for i in range(start_i, len(df) - MAX_HOLD - 1, STEP):
            sub = df.iloc[max(0, i - 200): i + 1]
            if len(sub) < 120:
                continue
            try:
                s = compute_signals(sub)
                if s is None:
                    continue
                s = compute_enhanced(sub, s, is_intraday=False)
                score = compute_score100(s)
            except Exception:
                continue
            if score < MIN_SCORE:
                continue

            entry = float(df.iloc[i + 1]["open"])
            atr = float(s.get("atr") or 0)
            if not np.isfinite(entry) or entry <= 0 or atr <= 0:
                continue

            path = df.iloc[i + 1: i + 1 + MAX_HOLD]
            if len(path) < 10:
                continue

            out.append({
                "symbol": sym,
                "date": df.index[i],
                "mkt_fwd": mkt_fwd.get(pd.Timestamp(df.index[i]).normalize(), np.nan),
                "score": score,
                "entry": entry,
                "atr": atr,
                "sl": s.get("sl_buy"),
                "t1": s.get("target1"),
                "high": path["high"].to_numpy(dtype=float),
                "low": path["low"].to_numpy(dtype=float),
                "close": path["close"].to_numpy(dtype=float),
            })

        if si % 25 == 0:
            print(f"  {si}/{len(symbols)} symbols — {len(out):,} signals")

    with open(_CACHE, "wb") as fh:
        pickle.dump(out, fh)
    print(f"cached {len(out):,} signals -> {_CACHE.name}")
    return len(out)


def load_signals() -> list[dict]:
    if not _CACHE.exists():
        raise SystemExit("No cached signals. Run:  python exit_lab.py build")
    with open(_CACHE, "rb") as fh:
        return pickle.load(fh)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — exit rules
# ─────────────────────────────────────────────────────────────────────────────

def _finish(entry: float, exit_px: float) -> float:
    return (exit_px / entry - 1 - COST) * 100


def exit_fixed(sig, hold=5, **_):
    """Current behaviour: stop, target, else time-stop. Stop wins on a tied bar."""
    e, sl, t1 = sig["entry"], sig["sl"], sig["t1"]
    hi, lo, cl = sig["high"][:hold], sig["low"][:hold], sig["close"][:hold]
    for j in range(len(cl)):
        if sl and lo[j] <= sl:
            return _finish(e, sl)
        if t1 and hi[j] >= t1:
            return _finish(e, t1)
    return _finish(e, cl[-1])


def exit_trail_atr(sig, hold=21, mult=2.5, **_):
    """Chandelier: stop trails `mult` ATR below the highest high since entry."""
    e, atr = sig["entry"], sig["atr"]
    hi, lo, cl = sig["high"][:hold], sig["low"][:hold], sig["close"][:hold]
    peak = e
    for j in range(len(cl)):
        stop = peak - mult * atr
        if lo[j] <= stop:
            return _finish(e, stop)
        peak = max(peak, hi[j])
    return _finish(e, cl[-1])


def exit_breakeven_trail(sig, hold=21, trigger=1.0, mult=2.5, **_):
    """Initial stop, moved to breakeven once +trigger ATR, then trails."""
    e, atr, sl = sig["entry"], sig["atr"], sig["sl"]
    hi, lo, cl = sig["high"][:hold], sig["low"][:hold], sig["close"][:hold]
    stop = sl if sl else e - 2 * atr
    peak, armed = e, False
    for j in range(len(cl)):
        if lo[j] <= stop:
            return _finish(e, stop)
        peak = max(peak, hi[j])
        if not armed and peak >= e + trigger * atr:
            armed = True
        if armed:
            stop = max(stop, peak - mult * atr)
    return _finish(e, cl[-1])


def exit_time(sig, hold=21, **_):
    """Pure time exit — no stop, no target. The control."""
    return _finish(sig["entry"], sig["close"][:hold][-1])


VARIANTS = []
for h in (5, 10, 21):
    VARIANTS.append((f"fixed h{h}", exit_fixed, {"hold": h}))
for h in (10, 21, 63):
    for m in (1.5, 2.5, 3.5):
        VARIANTS.append((f"trail {m}atr h{h}", exit_trail_atr, {"hold": h, "mult": m}))
for h in (21, 63):
    VARIANTS.append((f"be+trail 2.5 h{h}", exit_breakeven_trail, {"hold": h, "mult": 2.5}))
for h in (5, 21, 63):
    VARIANTS.append((f"time only h{h}", exit_time, {"hold": h}))


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(sigs: list[dict], fn, kw: dict) -> dict:
    rets, caps = [], []
    for s in sigs:
        try:
            r = fn(s, **kw)
        except Exception:
            continue
        if not np.isfinite(r):
            continue
        rets.append(r)
        mfe = (s["high"][:kw.get("hold", 21)].max() / s["entry"] - 1) * 100
        if mfe > 0.5:
            caps.append(r / mfe * 100)
    if len(rets) < 50:
        return {}
    a = np.array(rets)
    wins, losses = a[a > 0], a[a <= 0]
    gp, gl = wins.sum(), abs(losses.sum())
    return {
        "n": len(a),
        "avg": float(a.mean()),
        "med": float(np.median(a)),
        "win%": float((a > 0).mean() * 100),
        "pf": float(gp / gl) if gl > 0 else np.nan,
        "sharpe": float(a.mean() / a.std() * np.sqrt(252 / kw.get("hold", 21))) if a.std() > 0 else np.nan,
        "capture%": float(np.median(caps)) if caps else np.nan,
    }


def _benchmark(sigs: list[dict], hold: int = MAX_HOLD) -> float:
    """Hold every signal for `hold` bars with no stop and no target."""
    r = [_finish(s["entry"], s["close"][:hold][-1]) for s in sigs
         if len(s["close"]) >= hold]
    return float(np.mean(r)) if r else float("nan")


def run() -> None:
    sigs = load_signals()
    down = [s for s in sigs if np.isfinite(s.get("mkt_fwd", np.nan)) and s["mkt_fwd"] < 0]
    up = [s for s in sigs if np.isfinite(s.get("mkt_fwd", np.nan)) and s["mkt_fwd"] >= 0]

    print(f"\n{len(sigs):,} signals across the full archive")
    print(f"  market FELL over the next {MAX_HOLD} bars : {len(down):,} signals")
    print(f"  market ROSE                              : {len(up):,} signals")
    print(f"variants tried: {len(VARIANTS)} — all reported\n")

    for label, group in (("MARKET FELL", down), ("MARKET ROSE", up)):
        if len(group) < 200:
            print(f"  {label}: only {len(group)} signals — skipped\n")
            continue
        bench = _benchmark(group)
        print(f"  ── {label} ── {len(group):,} signals | "
              f"buy-and-hold benchmark {bench:+.2f}%")
        hdr = (f"    {'exit rule':<20}{'n':>7}{'avg%':>8}{'vs bench':>10}"
               f"{'win%':>7}{'PF':>7}{'cap%':>7}")
        print(hdr); print("    " + "-" * (len(hdr) - 4))
        for name, fn, kw in VARIANTS:
            r = evaluate(group, fn, kw)
            if not r:
                continue
            print(f"    {name:<20}{r['n']:>7}{r['avg']:>+8.2f}{r['avg']-bench:>+10.2f}"
                  f"{r['win%']:>7.1f}{r['pf']:>7.2f}{r['capture%']:>7.1f}")
        print()

    print("  'vs bench' is the only column that matters: return minus simply")
    print("  holding the same signals for the same period with no exit rule.")
    print("  A rule that cannot beat that is destroying value, not managing risk.")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "build":
        lim = int(sys.argv[2]) if len(sys.argv) > 2 else None
        build_signals(limit=lim)
    else:
        run()
