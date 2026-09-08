"""Fixed-endpoint, no-stop event measurement. Does not model portfolio fills."""
import numpy as np
import pandas as pd

def valid_bars(d):
    return (np.isfinite(d).all(axis=1)&(d[['open','high','low','close']]>0).all(axis=1)&
            (d.high>=d[['open','low','close']].max(axis=1))&(d.low<=d[['open','high','close']].min(axis=1))&(d.volume>0))

def outcomes(d,cost_pct=.5582):
    good=valid_bars(d)
    entry=good.shift(-1,fill_value=False);exit=good.shift(-21,fill_value=False)
    interior=pd.concat([good.shift(-i,fill_value=False) for i in range(2,21)],axis=1).all(axis=1)
    status=pd.Series(np.where(~entry,'unknown_entry',np.where(~exit,'unresolved_exit',np.where(interior,'observed','observed_with_interior_gap'))),index=d.index)
    ret=(100*(d.close.shift(-21)/d.open.shift(-1)-1)-cost_pct).where(entry&exit)
    return ret,status
