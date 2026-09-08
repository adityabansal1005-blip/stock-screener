"""Small frozen family of monthly hypotheses; no learned indicator weights."""
import hashlib
import numpy as np
import pandas as pd
from .data import valid_bars

NAMES=('stable_control','momentum','momentum_trend','year_high')


def precompute(frames):
    features={}
    for s,d in frames.items():
        c=d.close
        good=valid_bars(d)
        volatility=c.pct_change(fill_method=None).rolling(252).std()*np.sqrt(252)
        # Skip latest month; both legs end at t-21. No claim of exact NSE replication.
        strength=((c.shift(21)/c.shift(126)-1)+(c.shift(21)/c.shift(252)-1))/2/volatility.replace(0,np.nan)
        high_ratio=c/d.high.shift(1).rolling(252).max()
        features[s]=pd.DataFrame({'eligible':good.rolling(253).sum().eq(253)&(c>=10)&((c*d.volume).rolling(20).median()>=1e7),
                                  'valid':good,'tradable':good&(d.high>d.low),
                                  'momentum':strength,'year_high':high_ratio,
                                  'turnover':(c*d.volume).rolling(20).median()},index=d.index)
    return features


def rank(features,date,name,market_risk_on=True):
    if name not in NAMES: raise ValueError('Unregistered strategy')
    if name=='momentum_trend' and not market_risk_on: return []
    rows=[]
    for s,f in features.items():
        r=f.loc[date]
        if not bool(r.eligible):continue
        if name=='stable_control':
            value=int(hashlib.sha256(s.encode()).hexdigest(),16)
        else:
            value=float(r.year_high if name=='year_high' else r.momentum)
            if not np.isfinite(value):continue
            if name=='year_high' and value<.95:continue
            value=-value
        rows.append((value,s))
    return [s for _,s in sorted(rows)]
