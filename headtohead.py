"""
Head-to-head: does the scanner's score beat a one-line breakout, and does
either beat doing nothing?

This is the test the project has never run. Everything before it was either
measured on stocks already known to have gone up (selection bias), or compared
signals on FIRST-FIRE DATE only, which structurally favours whichever signal is
noisiest — a signal that fires 14 times will always have an early one among them.

Design
------
  universe   a random, seeded sample of the archive — not hand-picked names
  dates      every 5th trading day, so signals are not counted 21 times each
  signal     computed on bars up to and including day t, nothing after
  entry      the OPEN of day t+1        (entering at t's close is look-ahead;
                                         fixing that once turned BULL_TREND
                                         60-70 from +0.17% to -0.11%)
  exit       the CLOSE 21 sessions later
  costs      0.5582% round trip, charged to every arm including the control

The control is the whole point. "Score 60+ returned +1.2%" means nothing if
every stock returned +1.2% over the same days. Only the EDGE column counts.

    python headtohead.py [n_symbols] [start] [end]
"""

from __future__ import annotations

import random
import sys
import time
import warnings
import argparse
import json
import math
import sqlite3
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

COST_PCT = 0.5582        # matches backtester.ROUND_TRIP
HOLD = 21                # matches backtester.FORWARD_DAYS
STEP = 5                 # sample signals weekly, not daily
BAR = 60.0               # the scanner's "GOOD SETUP" line
SEED = 42


def rsi(c: pd.Series, n: int = 14) -> pd.Series:
    from signals import rsi as shared_rsi
    return shared_rsi(c, n)


class ScoreReplay:
    """Memoize identical point-in-time states used for levels and crossings."""
    def __init__(self, daily, nifty):
        self.daily, self.nifty = daily, nifty
        self.cache = {}
        self.computations = 0

    def at(self, i):
        if i in self.cache:
            return self.cache[i]
        from signals import compute_signals
        from technical_enhanced import compute_enhanced
        self.computations += 1
        hist = self.daily.iloc[:i+1]
        wk = hist.resample('W').agg({'Open':'first','High':'max','Low':'min',
                                    'Close':'last','Volume':'sum'}).dropna(subset=['Close'])
        nc = self.nifty.loc[self.nifty.index<=hist.index[-1]] if self.nifty is not None else None
        r=compute_signals(hist)
        if r is not None:
            r=compute_enhanced(hist,r,is_intraday=False,
                               weekly_df=wk if len(wk)>=15 else None,nifty_close=nc)
        self.cache[i]=r.get('score100') if r else None
        return self.cache[i]


def block_edge_interval(df, mask, draws=1000):
    """Date-block bootstrap: all stocks on a date stay together.

    Conservative 21 observation-date blocks cover at least the holding period
    when observation dates are daily. This is not a correction for trying many
    strategies and does not establish out-of-sample validity.
    """
    tmp=df[['date','fwd']].copy()
    tmp['chosen']=tmp.fwd.where(mask)
    daily=tmp.groupby('date').agg(total=('fwd','sum'),n=('fwd','count'),
                                 selected=('chosen','sum'),k=('chosen','count')).sort_index()
    n=len(daily); width=HOLD
    if n < 2*width or daily.k.sum()<20:
        return None
    vals=daily.to_numpy(dtype=float); rng=np.random.default_rng(SEED)
    estimates=[]
    for _ in range(draws):
        starts=rng.integers(0,n-width+1,size=math.ceil(n/width))
        ix=np.concatenate([np.arange(s,s+width) for s in starts])[:n]
        total,count,selected,k=vals[ix].sum(axis=0)
        if k and count: estimates.append(selected/k-total/count)
    return tuple(np.quantile(estimates,[.025,.975])) if estimates else None


def main() -> int:
    parser=argparse.ArgumentParser(description='Offline technical replay; never fetches or changes archive data.')
    parser.add_argument('n_symbols',type=int,nargs='?',default=400)
    parser.add_argument('start',nargs='?',default='2026-01-02')
    parser.add_argument('end',nargs='?',default='2026-07-31')
    parser.add_argument('--archive',type=Path,default=Path(__file__).parent)
    parser.add_argument('--output',type=Path,default=Path('headtohead_results'))
    args=parser.parse_args()
    n_syms,start,end=args.n_symbols,args.start,args.end
    if n_syms<1 or pd.Timestamp(start)>pd.Timestamp(end): parser.error('Invalid sample size or date range')
    args.output.mkdir(parents=True,exist_ok=True)
    archive=args.archive/'strategy_data.db'
    con=sqlite3.connect(archive.resolve().as_uri()+'?mode=ro',uri=True)
    def load(sym):
        d=pd.read_sql_query('SELECT date,open,high,low,close,volume FROM ohlcv WHERE symbol=? ORDER BY date',con,params=[sym])
        d['date']=pd.to_datetime(d.date)
        return d.set_index('date').rename(columns=str.title)
    nd=load('^NSEI')
    NIFTY=nd.Close if not nd.empty else None
    if NIFTY is None: raise RuntimeError('Archive lacks Nifty history; refusing a different-input replay')
    cached=[r[0] for r in con.execute("SELECT DISTINCT symbol FROM ohlcv WHERE symbol LIKE '%.NS' ORDER BY symbol")]
    random.seed(SEED)
    random.shuffle(cached)
    picked = cached[:n_syms]

    print("=" * 74)
    print("  HEAD-TO-HEAD — technical replay vs 20-day breakout vs unfiltered stocks")
    print("=" * 74)
    print(f"  archive          : {len(cached):,} symbols")
    print(f"  sampled          : {len(picked):,} (seed {SEED}, not hand-picked)")
    print(f"  window           : {start} .. {end}, every {STEP}th session")
    print(f"  entry / exit     : next OPEN -> close +{HOLD} sessions")
    print(f"  costs            : {COST_PCT}% round trip, charged to every arm")
    print("=" * 74, flush=True)

    recs: list[dict] = []
    t0 = time.time()
    done = 0
    computations = 0
    score_errors = 0
    manifest=dict(start=start,end=end,seed=SEED,sampled=len(picked),holding_sessions=HOLD,
                  step=STEP,cost_pct=COST_PCT,archive=str(archive.resolve()),
                  method='technical replay; next open; 21 sessions including entry',
                  code_sha256={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                               for name in ['headtohead.py','signals.py','technical_enhanced.py']})
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (args.output/'progress.json').write_text(json.dumps(dict(status='running',completed=0,
        total=len(picked),observations=0,elapsed_seconds=0)),encoding='utf-8')

    for sym in picked:
        done += 1
        if done % 25 == 0:
            print(f"    {done}/{len(picked)}  {len(recs):,} observations  "
                  f"{(time.time()-t0)/60:.1f} min", flush=True)
            (args.output/'progress.json').write_text(json.dumps(dict(status='running',
                completed=done-1,total=len(picked),observations=len(recs),elapsed_seconds=time.time()-t0)),encoding='utf-8')
        try:
            d = load(sym)
        except Exception:
            continue
        if d is None or len(d) < 300:
            continue
        d = d.copy()
        d.columns = [c.title() for c in d.columns]
        if not {"Open", "High", "Low", "Close", "Volume"}.issubset(d.columns):
            continue

        # Benchmarks, vectorised. shift(1) keeps the fire day out of its own
        # lookback window.
        brk = d["Close"] > d["High"].shift(1).rolling(20).max()
        rs = rsi(d["Close"])
        replay=ScoreReplay(d,NIFTY)

        idx = d.loc[start:end].index
        for ts in idx[::STEP]:
            i = d.index.get_loc(ts)
            if i + HOLD >= len(d) or i < 260:
                continue
            hist = d.iloc[: i + 1]

            entry = float(d["Open"].iloc[i + 1])
            exit_ = float(d["Close"].iloc[i + HOLD])
            if not np.isfinite(entry) or entry <= 0:
                continue
            fwd = (exit_ / entry - 1) * 100 - COST_PCT

            score = None
            try:
                score = replay.at(i)
            except Exception:
                score_errors += 1

            prev_score = None
            if score is not None and i - STEP >= 260:
                # Only needed to identify a CROSSING rather than a level.
                try:
                    prev_score = replay.at(i-STEP)
                except Exception:
                    pass

            recs.append({
                "symbol": sym, "date": ts, "fwd": fwd,
                "score": score, "prev_score": prev_score,
                "breakout": bool(brk.iloc[i]) if pd.notna(brk.iloc[i]) else False,
                "rsi60": bool(rs.iloc[i] > 60) if pd.notna(rs.iloc[i]) else False,
            })
        computations += replay.computations

    con.close()
    df = pd.DataFrame(recs)
    if df.empty:
        print("\n  no observations — check the archive")
        return 1

    df.to_csv(args.output/"headtohead_raw.csv", index=False)
    raw_count=len(df)
    df=df.dropna(subset=['score','fwd']).copy()  # identical eligible control for every arm
    if df.empty:
        print('No scoreable observations; inspect raw output.'); return 1
    base = df["fwd"].mean()

    print(f"\n{'=' * 74}")
    print(f"  {len(df):,} observations · {df.symbol.nunique():,} symbols · "
          f"{df.date.nunique()} dates · {(time.time()-t0)/60:.1f} min")
    print("=" * 74)
    print(f"\n  {'ARM':<26}{'N':>8}{'MEAN %':>10}{'EDGE':>9}{'WIN%':>8}   95% date-block interval")
    print(f"  {'-'*26}{'-'*8}{'-'*10}{'-'*9}{'-'*8}{'-'*8}")

    def arm(name, mask):
        s = df.loc[mask, "fwd"].dropna()
        if len(s) < 20:
            print(f"  {name:<26}{len(s):>8}{'too few':>10}")
            return
        edge = s.mean() - base
        # Welch t against the control, so "beat the universe" is testable
        # rather than asserted.
        ci=block_edge_interval(df,mask)
        interval=f'[{ci[0]:+.2f}, {ci[1]:+.2f}]' if ci else 'insufficient dates/signals'
        print(f"  {name:<26}{len(s):>8}{s.mean():>9.2f}%{edge:>+9.2f}"
              f"{(s > 0).mean()*100:>7.0f}%   {interval}")

    print(f"  {'BASELINE (unfiltered)':<26}{len(df):>8}{base:>9.2f}%"
          f"{0.0:>+9.2f}{(df.fwd > 0).mean()*100:>7.0f}%{'—':>8}")
    has = df["score"].notna()
    arm("SCORE >= 60", has & (df["score"] >= BAR))
    arm("SCORE >= 70", has & (df["score"] >= 70))
    arm("SCORE crosses into 60", has & df["prev_score"].notna()
        & (df["prev_score"] < BAR) & (df["score"] >= BAR))
    arm("20-day breakout", df["breakout"])
    arm("RSI(14) > 60", df["rsi60"])
    arm("breakout AND score>=60", df["breakout"] & has & (df["score"] >= BAR))

    print(f"\n  Spearman score vs forward return: "
          f"{df['score'].rank().corr(df['fwd'].rank()):+.4f}")
    print(f"\n  {raw_count-len(df)} unscoreable rows excluded from ALL arms; {score_errors} scoring errors")
    print('  Technical replay only: historical fundamentals and regime inputs are unavailable.')
    print('  Block intervals account for date dependence, not model selection or survivorship bias.')
    print('  Frozen rules still require unseen chronological validation before an edge claim.')
    (args.output/'progress.json').write_text(json.dumps(dict(status='complete',completed=done,
        total=len(picked),observations=len(df),score_computations=computations,
        elapsed_seconds=time.time()-t0,score_errors=score_errors)),encoding='utf-8')
    return 0


if __name__ == "__main__":
    sys.exit(main())
