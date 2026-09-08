"""Eligibility and completed-session policy; no trading or network side effects."""
import datetime as dt
import json
from pathlib import Path
import pandas as pd

VERSION='measurement-v2'
IST=dt.timezone(dt.timedelta(hours=5,minutes=30))

def completed_history(frame,now=None):
    now=now or dt.datetime.now(IST)
    if now.tzinfo is None:raise ValueError('Timezone-aware time required')
    now=now.astimezone(IST)
    cutoff=now.date() if now.time()>=dt.time(16,15) else now.date()-dt.timedelta(days=1)
    d=frame.copy()
    ix=pd.DatetimeIndex(d.index)
    if ix.tz is not None:ix=ix.tz_convert('Asia/Kolkata').tz_localize(None)
    d.index=ix
    d=d[d.index.date<=cutoff].sort_index()
    if d.index.has_duplicates:raise ValueError('Duplicate daily dates')
    w=d.resample('W').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna(subset=['Close'])
    return d,w if len(w)>=15 else None

def assess(row,expected_session,risk_registry):
    reasons=[]
    if row.get('liq_tradeable') is not True:reasons.append('liquidity_failed_or_unknown')
    if not expected_session:reasons.append('reference_session_unknown')
    elif row.get('bar_date')!=str(expected_session):reasons.append('stale_or_incomplete_session')
    review=risk_registry.get(row.get('symbol'),{})
    if review.get('status')=='blocked':reasons.append('material_financial_risk')
    status=review.get('status','unreviewed')
    return {'eligibility_version':VERSION,'screen_eligible':not reasons,
            'eligibility_reasons':reasons,'financial_review_status':status,
            'financial_review_notes':review.get('notes',[]),
            'actionable':False,'actionability_reason':'No validated strategy; research only',
            'signal_basis':'completed_session'}

def load_risks():
    path=Path(__file__).with_name('financial_risk_registry.json')
    try:
        data=json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(data,dict):raise ValueError('Invalid risk registry')
        return data
    except Exception:
        # Registry failure must not silently clear known blocks.
        raise RuntimeError('Financial risk registry unavailable or invalid') from None
