"""
"What if I had actually bought what it recommended?"

Takes the stocks the scanner scored at or above a threshold on a past date,
simulates buying them equal-weight at the next session's open, holds to today,
and reports the position — against a Nifty 50 buy-and-hold over the same days.

The benchmark is the point. A portfolio that made money in a rising market has
proved nothing; the only question that matters is whether picking these stocks
beat simply buying the index on the same day.

    python portfolio_sim.py                     # default: 2026-05-29, score>=60
    python portfolio_sim.py 2026-05-29 60
"""

from __future__ import annotations

import datetime
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from backtester import COMMISSION, SLIPPAGE
    ROUND_TRIP_PCT = (COMMISSION * 2 + SLIPPAGE * 2) * 100
except Exception:
    ROUND_TRIP_PCT = 0.5582


def simulate(date_str: str, min_score: float = 60, capital: float = 500_000,
             universe: str | None = None, max_positions: int | None = None) -> dict:
    """
    `max_positions` caps how many names the capital is split across.

    Spreading a small account over every qualifying stock strands most of it:
    Rs1,00,000 across 71 picks is Rs1,408 each, and 31 of those had share
    prices above that (POWERINDIA at Rs38,700), so those slots bought zero
    shares and their allocation sat in cash. Concentrating into the highest
    scoring names that the capital can actually buy deploys the whole amount.
    """
    from forward_test import load_scan, _prices

    scan = load_scan(date_str, universe)
    picks = scan[scan["score"] >= min_score].sort_values("score", ascending=False)
    if picks.empty:
        return {"error": f"no symbols scored >= {min_score} on {date_str}"}

    scan_date = pd.Timestamp(date_str).date()
    today = datetime.date.today()

    # Price everything first so allocation can be decided with prices known.
    priced = []
    for r in picks.itertuples():
        px = _prices(r.symbol, scan_date, today)
        if px is None or px.empty:
            continue
        fwd = px[px.index > pd.Timestamp(scan_date)]
        if len(fwd) < 5:
            continue
        entry = float(fwd.iloc[0]["Open"])
        if not np.isfinite(entry) or entry <= 0:
            continue
        priced.append((r.symbol, r.score, entry, float(fwd.iloc[-1]["Close"]), len(fwd)))

    if not priced:
        return {"error": "no positions could be priced"}

    # Take the top N by score that the capital can genuinely fund.
    n = max_positions or len(priced)
    chosen = priced[:n]
    per_stock = capital / max(len(chosen), 1)
    chosen = [c for c in chosen if c[2] <= per_stock]      # affordable at this slice
    if not chosen:
        return {"error": "capital too small for even one position"}
    per_stock = capital / len(chosen)

    rows, skipped = [], len(priced) - len(chosen)
    for symbol, score, entry, last, days in chosen:
        qty = int(per_stock // entry)
        if qty <= 0:
            skipped += 1
            continue
        r = type("R", (), {"symbol": symbol, "score": score})()
        invested  = qty * entry
        cur_value = qty * last
        # Charge the full round trip so the figure is what you'd bank on exit.
        costs     = invested * ROUND_TRIP_PCT / 100
        pnl       = cur_value - invested - costs

        rows.append({
            "symbol": r.symbol, "score": r.score, "qty": qty,
            "entry": entry, "last": last, "invested": invested,
            "value": cur_value, "pnl": pnl,
            "pnl_pct": pnl / invested * 100,
            "days": days,
        })

    if not rows:
        return {"error": "no positions could be priced"}

    df = pd.DataFrame(rows)
    invested = df["invested"].sum()
    value    = df["value"].sum()
    pnl      = df["pnl"].sum()

    # ── Benchmark: Nifty 50 over the identical window ────────────────────
    bench = None
    try:
        import data_manager
        n = data_manager.load("^NSEI")
        if n is not None:
            nf = n[n.index > pd.Timestamp(scan_date)]
            if len(nf) > 2:
                b_entry = float(nf.iloc[0]["open"])
                b_last  = float(nf.iloc[-1]["close"])
                bench = (b_last / b_entry - 1) * 100
    except Exception:
        pass

    return {
        "date": date_str, "min_score": min_score, "capital": capital,
        "positions": len(df), "skipped": skipped,
        "invested": invested, "value": value, "pnl": pnl,
        "pnl_pct": pnl / invested * 100 if invested else 0,
        "winners": int((df["pnl"] > 0).sum()),
        "losers": int((df["pnl"] <= 0).sum()),
        "best": df.nlargest(5, "pnl_pct")[["symbol", "score", "pnl_pct"]].to_dict("records"),
        "worst": df.nsmallest(5, "pnl_pct")[["symbol", "score", "pnl_pct"]].to_dict("records"),
        "days_held": int(df["days"].median()),
        "benchmark_pct": bench,
        "detail": df,
    }


def show(r: dict) -> None:
    if "error" in r:
        print("  " + r["error"]); return
    inr = lambda x: f"₹{x:,.0f}"
    sign = "+" if r["pnl"] >= 0 else ""

    print(f"\n{'=' * 62}")
    print(f"  If you had bought every stock scoring >= {r['min_score']:.0f}")
    print(f"  on {r['date']}, equal-weight, and still held it today")
    print(f"{'=' * 62}")
    print(f"  positions        : {r['positions']}  ({r['skipped']} unpriceable)")
    print(f"  held             : ~{r['days_held']} sessions")
    print(f"  invested         : {inr(r['invested'])}")
    print(f"  value today      : {inr(r['value'])}")
    print(f"  net P&L          : {sign}{inr(r['pnl'])}   ({sign}{r['pnl_pct']:.2f}%)")
    print(f"                     (after {ROUND_TRIP_PCT:.2f}% round-trip costs)")
    print(f"  winners / losers : {r['winners']} / {r['losers']}")

    if r["benchmark_pct"] is not None:
        edge = r["pnl_pct"] - r["benchmark_pct"]
        print()
        print(f"  Nifty 50 same days: {r['benchmark_pct']:+.2f}%")
        print(f"  YOUR EDGE         : {edge:+.2f} pts  "
              f"{'— beat the index' if edge > 0 else '— LOST to just buying the index'}")

    print(f"\n  best:  " + ", ".join(f"{b['symbol']} {b['pnl_pct']:+.1f}%" for b in r["best"]))
    print(f"  worst: " + ", ".join(f"{b['symbol']} {b['pnl_pct']:+.1f}%" for b in r["worst"]))
    print()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    import config, groww_client
    groww_client.init_groww(config.GROWW_API_KEY, config.GROWW_API_SECRET)

    date  = sys.argv[1] if len(sys.argv) > 1 else "2026-05-29"
    score = float(sys.argv[2]) if len(sys.argv) > 2 else 60
    cap   = float(getattr(config, "ACCOUNT_SIZE_INR", 500_000))

    show(simulate(date, score, cap))
