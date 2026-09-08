"""
How good is the regime classifier at predicting what the market does NEXT?

Why this matters
----------------
The exit lab produced a clean, actionable result: stops earn about +3 points
when the market falls over the following weeks and lose 7-11 points when it
rises. So the right exit is regime-conditional — tight stops in weak markets,
loose trails in strong ones.

But that result classified "fell" and "rose" with HINDSIGHT. Whether any of it
is capturable depends entirely on one question nobody has measured: can the
classifier tell, in advance, which case it is in?

This is the gate on everything. If it can, there is a concrete route to
profitability. If it cannot, the honest conclusion from the same data is to
stop trading and hold.

Method
------
Classify each historical bar using only data available up to that bar, then
compare against what the index actually did over the following N sessions.
Report it as a confusion matrix, plus the metric that actually matters:
conditional on the classifier saying "weak", how often did the market fall?

    python regime_accuracy.py
"""

from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HORIZONS = (5, 21, 63)
# Regimes the system treats as risk-off, i.e. where tight stops would be used.
WEAK = {"BEAR_TREND", "RECOVERY", "SIDEWAYS"}
STRONG = {"BULL_TREND", "BULL_WEAK"}


def build() -> pd.DataFrame:
    import data_manager
    from regime_expectancy import build_regime_frame

    n = data_manager.load("^NSEI")
    if n is None:
        raise SystemExit("Nifty history unavailable")
    n = n.copy()
    n.columns = [c.lower() for c in n.columns]

    frame = build_regime_frame(n)
    df = pd.DataFrame({"close": n["close"], "regime": frame["regime"]}).dropna()
    for h in HORIZONS:
        df[f"fwd{h}"] = (df["close"].shift(-h) / df["close"] - 1) * 100
    return df


def report(df: pd.DataFrame) -> None:
    print(f"\n  Nifty {df.index.min().date()} → {df.index.max().date()} "
          f"({len(df):,} sessions)\n")

    counts = df["regime"].value_counts()
    print("  Regime frequency:")
    for r, c in counts.items():
        print(f"    {r:<14}{c:>6}  ({c / len(df) * 100:4.1f}%)")

    for h in HORIZONS:
        col = f"fwd{h}"
        sub = df.dropna(subset=[col])
        if sub.empty:
            continue
        base_down = (sub[col] < 0).mean() * 100

        print(f"\n  ── Forward {h} sessions ── "
              f"(market fell {base_down:.1f}% of the time overall)")
        print(f"    {'regime':<14}{'n':>6}{'avg fwd':>10}{'P(fall)':>10}"
              f"{'lift':>8}")
        print("    " + "-" * 48)
        for r in counts.index:
            s = sub[sub["regime"] == r]
            if len(s) < 30:
                continue
            pf = (s[col] < 0).mean() * 100
            print(f"    {r:<14}{len(s):>6}{s[col].mean():>+9.2f}%{pf:>9.1f}%"
                  f"{pf - base_down:>+8.1f}")

        # The decision the exits actually make: risk-off vs risk-on.
        weak = sub[sub["regime"].isin(WEAK)]
        strong = sub[sub["regime"].isin(STRONG)]
        if len(weak) > 30 and len(strong) > 30:
            pw = (weak[col] < 0).mean() * 100
            ps = (strong[col] < 0).mean() * 100
            print(f"\n    classifier says WEAK   -> market fell {pw:.1f}% of the time "
                  f"(avg {weak[col].mean():+.2f}%)")
            print(f"    classifier says STRONG -> market fell {ps:.1f}% of the time "
                  f"(avg {strong[col].mean():+.2f}%)")
            print(f"    SEPARATION: {pw - ps:+.1f} pts  "
                  f"— {'usable' if abs(pw - ps) >= 10 else 'too weak to act on'}")

            # Accuracy of the binary call, and how often it fires.
            tp = ((weak[col] < 0)).sum()
            fp = ((weak[col] >= 0)).sum()
            tn = ((strong[col] >= 0)).sum()
            fn = ((strong[col] < 0)).sum()
            acc = (tp + tn) / (tp + fp + tn + fn) * 100
            prec = tp / (tp + fp) * 100 if (tp + fp) else 0
            rec = tp / (tp + fn) * 100 if (tp + fn) else 0
            print(f"    accuracy {acc:.1f}% | precision {prec:.1f}% "
                  f"(when it says weak, it's right this often) | recall {rec:.1f}%")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    report(build())
    print("\n  Interpretation: separation is the number that matters. The exit"
          "\n  lab showed a ~10 point swing between tight and loose stops, so a"
          "\n  classifier separating by less than that cannot pay for switching.\n")
