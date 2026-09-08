"""Offline integration checks for the public scan entry points."""
import datetime as dt,sys,unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import scanner,screening_policy

class ScanPolicy(unittest.TestCase):
    def test_fetch_uses_completed_history(self):
        ix=pd.bdate_range('2025-01-01','2026-09-08');n=len(ix)
        close=pd.Series(range(100,100+n),index=ix,dtype=float)
        d=pd.DataFrame({'Open':close,'High':close+1,'Low':close-1,'Close':close,'Volume':100000.})
        prepare=screening_policy.completed_history
        fixed=dt.datetime(2026,9,8,14,45,tzinfo=screening_policy.IST)
        def history(frame):return prepare(frame,fixed)
        with patch.object(scanner,'_get_ohlcv',return_value=(d,None)),patch.object(scanner,'get_fundamentals',return_value={}),patch.object(screening_policy,'completed_history',side_effect=history),patch('ml_signal.score_signal',return_value=None):
            result=scanner.fetch_swing('TEST.NS',close.iloc[:-1],bulk_fundamentals=True)
        self.assertIsNotNone(result)
        self.assertEqual(result['bar_date'],'2026-09-07')
        self.assertEqual(result['score_version'],'measurement-v2-no-rr')
        self.assertEqual(result['_df'].index[-1],pd.Timestamp('2026-09-07'))

if __name__=='__main__':unittest.main()
