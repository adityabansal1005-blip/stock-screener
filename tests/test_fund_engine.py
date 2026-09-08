import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import tempfile
import unittest
import numpy as np
import pandas as pd
from fund_engine.data import Provenance, latest_known, save_snapshot, valid_bars
from fund_engine.ledger import Ledger, Limits
from fund_engine.strategies import precompute, rank
from fund_engine.review import assess_memo
from fund_engine.paper import PaperJournal


class FundTests(unittest.TestCase):
    def test_prospective_journal_rejects_backfill_and_fake_fills(self):
        with tempfile.TemporaryDirectory() as tmp:
            j=PaperJournal(Path(tmp)/'journal.db')
            j.register('year_high','hash','2026-09-08T10:00:00Z')
            with self.assertRaises(ValueError):j.append('observation',{'session_completed_at':'2026-09-07T10:00:00Z'},'2026-09-08T11:00:00Z','old')
            j.append('observation',{'session_completed_at':'2026-09-09T10:00:00Z'},'2026-09-09T11:00:00Z','new')
            with self.assertRaises(ValueError):j.append('fill',{},'2026-09-09T11:00:00Z','fake')
            with self.assertRaises(ValueError):j.append('intent',{'not_before':'2026-09-08T10:00:00Z'},'2026-09-09T11:00:00Z','past')
            self.assertTrue(j.verify())

    def test_ai_agreement_does_not_authorize_trade(self):
        evidence={'id':'e','available_at':'2020-01-01T00:00:00Z','source_url':'https://example.com/filing'}
        claim={'claim':'test','evidence_ids':['e'],'invalidation_condition':'next filing contradicts it'}
        m={'evidence':[evidence],'bull':[claim],'bear':[claim]}
        result=assess_memo(m,'2020-02-01T00:00:00Z')
        self.assertTrue(result['memo_structurally_complete']);self.assertFalse(result['actionable'])
        self.assertFalse(assess_memo(m,'2019-02-01T00:00:00Z')['memo_structurally_complete'])

    def test_float_rounding_is_not_bad_ohlc_but_real_error_is(self):
        d=pd.DataFrame({'open':[100.],'close':[124.4632339477539],'high':[124.46323394775389],'low':[99.],'volume':[100.]})
        self.assertTrue(valid_bars(d).iloc[0])
        d.loc[0,'high']=124.45
        self.assertFalse(valid_bars(d).iloc[0])

    def verified(self):
        return Provenance('synthetic_test','raw','raw',True,True,True,True,'synthetic fixture')

    def ledger(self,**kwargs):
        return Ledger(Limits(max_positions=2,max_weight=.5,max_sector_weight=.5,**kwargs),provenance=self.verified())

    def test_unknown_data_never_passes_strict_gate(self):
        p=Provenance('legacy','unknown','unknown')
        self.assertEqual(len(p.blockers()),7)
        with self.assertRaises(ValueError):Ledger(provenance=p)
        with self.assertRaises(ValueError):Ledger()

    def test_future_fundamentals_not_visible(self):
        r=pd.DataFrame({'symbol':['A','A'],'available_at':['2020-05-01','2020-08-01'],'profit':[10,999]})
        self.assertEqual(latest_known(r,'2020-06-01').iloc[0].profit,10)

    def test_cash_fees_and_whole_shares(self):
        l=self.ledger()
        l.rebalance('d',{'A':50000},{'A':101},{'A':1e8},{'A':True},{'A':'X'})
        self.assertEqual(l.positions['A'],495)
        self.assertAlmostEqual(l.cash,100000-495*101*(1+l.limits.side_cost))
        self.assertAlmostEqual(l.nav({'A':101}),100000-l.fees)

    def test_no_duplicate_holdings_or_negative_cash(self):
        l=self.ledger()
        for day in range(3):l.rebalance(day,{'A':50000,'B':50000},{'A':100,'B':100},{'A':1e8,'B':1e8},{'A':True,'B':True},{'A':'X','B':'Y'})
        self.assertEqual(len(l.positions),2);self.assertGreaterEqual(l.cash,0)

    def test_sector_unknown_and_sector_cap(self):
        l=self.ledger()
        l.rebalance('d',{'A':50000,'B':50000},{'A':100,'B':100},{'A':1e8,'B':1e8},{'A':True,'B':True},{'A':'X','B':'X'})
        self.assertNotIn('B',l.positions)
        other=self.ledger();other.rebalance('d',{'A':50000},{'A':100},{'A':1e8},{'A':True})
        self.assertFalse(other.positions)

    def test_missing_mark_not_flat_return(self):
        l=self.ledger();l.positions={'A':10}
        self.assertIsNone(l.nav({}))

    def test_unfilled_sale_cannot_fund_new_buy_or_exceed_slots(self):
        l=Ledger(Limits(max_positions=1,max_weight=1,max_sector_weight=1,side_cost=0),provenance=self.verified())
        l.rebalance('d',{'A':100000},{'A':100},{'A':1e8},{'A':True},{'A':'X'})
        l.rebalance('e',{'B':100000},{'A':100,'B':100},{'A':1e8,'B':1e8},{'A':False,'B':True},{'A':'X','B':'Y'})
        self.assertEqual(l.positions,{'A':1000});self.assertEqual(l.cash,0)

    def test_participation_cap(self):
        l=self.ledger();l.rebalance('d',{'A':50000},{'A':100},{'A':10000},{'A':True},{'A':'X'})
        self.assertEqual(l.positions['A'],1)

    def test_split_preserves_wealth_and_duplicate_rejected(self):
        l=self.ledger();l.positions={'A':10};before=l.nav({'A':100})
        event={'id':'split1','symbol':'A','type':'split','ratio':5}
        l.action('d',event);self.assertEqual(l.nav({'A':20}),before)
        with self.assertRaises(ValueError):l.action('d',event)

    def test_dividend_entitlement_does_not_use_current_holdings(self):
        l=self.ledger();l.positions={'A':2}
        l.action('d',{'id':'div1','symbol':'A','type':'dividend_payment','entitled_quantity':10,'cash_per_share':3})
        self.assertEqual(l.cash,100030)

    def test_demerger_and_fractional_split_quarantine(self):
        l=self.ledger();l.positions={'A':3}
        l.action('d',{'id':'sp','symbol':'A','type':'split','ratio':.5})
        self.assertIsNone(l.nav({'A':200}))
        k=self.ledger();k.positions={'A':3};k.action('d',{'id':'dm','symbol':'A','type':'demerger'})
        self.assertIsNone(k.nav({'A':100}))

    def test_future_bars_do_not_change_rank(self):
        dates=pd.bdate_range('2015-01-01',periods=300)
        c=np.linspace(100,200,300)
        d=pd.DataFrame({'open':c,'close':c,'high':c+1,'low':c-1,'volume':1e6},index=dates)
        f=precompute({'A':d});before=f['A'].iloc[260].copy()
        d.iloc[261:]*=20;after=precompute({'A':d})['A'].iloc[260]
        pd.testing.assert_series_equal(before,after)
        self.assertEqual(rank(f,dates[260],'momentum_trend',False),[])

    def test_snapshots_never_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=pd.DataFrame({'open':[10],'close':[10],'high':[11],'low':[9],'volume':[100]},index=pd.to_datetime(['2020-01-01']))
            path=Path(tmp)/'snap';p=Provenance('test','raw','raw')
            save_snapshot(d,path,p)
            with self.assertRaises(FileExistsError):save_snapshot(d,path,p)


if __name__=='__main__':unittest.main()
