"""Offline correctness checks. Run: python audit_tests.py. No network or trading."""
import ast
import pathlib
import unittest
import contextlib
import importlib.util
import sys
import tempfile
import sqlite3
from types import ModuleType
from unittest.mock import patch
from unittest.mock import Mock
import numpy as np
import pandas as pd
import signals
import technical_enhanced as te
import backtester as bt
import data_quality as dq
import scoring_v2 as v2
import ml_signal as ml
import headtohead as h2h

@contextlib.contextmanager
def isolated_scanner():
    """Real scanner with network/config/database boundaries replaced by fakes."""
    specs={
      'yfinance':{'Ticker':Mock()},
      'market_context':{'get_market_context':Mock(return_value={})},
      'fundamentals':{'get_fundamentals':Mock(return_value={})},
      'ai_analyst':{}, 'config':{},
      'groww_client':{'is_connected':Mock(return_value=True),'data_forbidden':Mock(return_value=False),
                      'get_daily_bars':Mock()},
      'truedata_client':{},
      'stocks_nse':{'get_all_symbols':Mock(return_value=[]),'load_watchlist':Mock(return_value=[])},
      'data_manager':{'_is_stale':Mock(return_value=True),'_store':Mock()},
      'dhan_client':{'is_configured':Mock(return_value=False)}
    }
    modules={}
    for name,attrs in specs.items():
        mod=ModuleType(name); mod.__dict__.update(attrs); modules[name]=mod
    with patch.dict(sys.modules,modules):
        spec=importlib.util.spec_from_file_location('scanner_under_test',ROOT/'scanner.py')
        mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        yield mod

ROOT = pathlib.Path(__file__).parent

def frame(n=300):
    c = 100 + np.arange(n) * .1 + np.sin(np.arange(n))
    return pd.DataFrame(dict(Open=c, High=c+2, Low=c-2, Close=c,
                             Volume=np.full(n, 100000.)),
                        index=pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=n))

class Audit(unittest.TestCase):
    def test_all_python_syntax(self):
        for p in ROOT.glob('*.py'):
            ast.parse(p.read_text(encoding='utf-8-sig'), filename=str(p))

    def test_basic_pipeline(self):
        d = frame()
        r = te.compute_enhanced(d, signals.compute_signals(d))
        self.assertTrue(0 <= r['score100'] <= 100)
        self.assertTrue(np.isfinite(r['close']))

    def test_rsi_all_gains_is_100(self):
        self.assertEqual(signals.rsi(pd.Series(np.arange(100.))).iloc[-1], 100)

    def test_cci_matches_reference(self):
        d = frame()
        tp = (d.High+d.Low+d.Close)/3
        expected = (tp.iloc[-1]-tp.tail(20).mean()) / (.015 * np.abs(tp.tail(20)-tp.tail(20).mean()).mean())
        self.assertAlmostEqual(te.cci(d.High,d.Low,d.Close).iloc[-1], expected)

    def test_targets_are_ordered(self):
        r = te.compute_trade_levels(dict(close=100., atr=2.))
        self.assertGreater(r['target2'], r['target1'], str(r))

    def test_adx_outside_bar_selects_only_larger_dm(self):
        h = pd.Series([100., 110.]); l = pd.Series([90., 85.]); c = pd.Series([95., 100.])
        _, plus, minus = te.adx(h,l,c)
        self.assertGreater(plus.iloc[-1], 0)
        self.assertEqual(minus.iloc[-1], 0)

    def test_adx_equal_moves_have_zero_dm(self):
        h=pd.Series([100.,110.]); l=pd.Series([90.,80.]); c=pd.Series([95.,95.])
        _, plus, minus=te.adx(h,l,c)
        self.assertEqual(plus.iloc[-1],0)
        self.assertEqual(minus.iloc[-1],0)

    def test_validator_accepts_timezone(self):
        d = frame(); d.index = d.index.tz_localize('Asia/Kolkata')
        self.assertTrue(dq.validate_ohlcv(d)[0])

    def test_validator_rejects_non_numeric(self):
        d = frame()
        for col in ['Open','High','Low','Close']: d[col] = 'invalid'
        self.assertFalse(dq.validate_ohlcv(d)[0])

    def test_validator_rejects_duplicate_bars(self):
        d = frame(); d = pd.concat([d, d.tail(1)])
        self.assertFalse(dq.validate_ohlcv(d)[0])

    def test_v2_high_ratio_not_divided_by_100(self):
        self.assertEqual(v2._debt_equity(6.), 6.)

    def test_backtest_gap_stop_uses_executable_price(self):
        d = frame(200)
        d.loc[:, ['Open','High','Low','Close']] = [100.,101.,99.,100.]
        # First signal at 119, entry at 120; next session gaps below stop 95.
        d.iloc[121, d.columns.get_indexer(['Open','High','Low','Close'])] = [80.,82.,79.,81.]
        def signal(sub):
            return dict(score100=60 if sub.index[-1] == d.index[119] else 0,
                        sl_buy=95., target1=110.)
        with patch.object(bt,'compute_signals',side_effect=signal), patch.object(bt,'compute_enhanced',side_effect=lambda d,s,**k:s):
            r = bt.backtest_win_rate(d)
        expected = ((80*(1-bt.SLIPPAGE)/(100*(1+bt.SLIPPAGE)))-1-bt.COMMISSION*2)*100
        self.assertAlmostEqual(r['avg_return'], round(expected,2))

    def test_backtest_preserves_200bar_history(self):
        seen=[]
        with patch.object(bt,'compute_signals',side_effect=lambda d: seen.append(len(d)) or None):
            bt.backtest_win_rate(frame(400))
        self.assertGreaterEqual(max(seen), 200)

    def test_vwap_resets_daily(self):
        ix=pd.to_datetime(['2026-09-01 10:00','2026-09-01 11:00','2026-09-02 10:00'])
        c=pd.Series([100.,110.,200.],index=ix); v=pd.Series([1.,1.,1.],index=ix)
        self.assertEqual(te.vwap(c,c,c,v).tolist(),[100.,105.,200.])

    def test_ml_ambiguous_bar_not_automatic_win(self):
        d=frame(30).rename(columns=str.lower)
        d.loc[:, ['open','high','low','close']]=[100.,101.,99.,100.]
        d.iloc[1,d.columns.get_indexer(['high','low'])]=[105.,95.]
        self.assertEqual(ml.build_labels(d).iloc[0],0)

    def test_ml_last_bar_has_unknown_outcome(self):
        d=frame(30).rename(columns=str.lower)
        self.assertTrue(pd.isna(ml.build_labels(d).iloc[-1]))

    def test_rsi_flat_and_falling(self):
        self.assertEqual(signals.rsi(pd.Series([10.]*40)).iloc[-1],50.)
        self.assertEqual(signals.rsi(pd.Series(np.arange(100.,0.,-1))).iloc[-1],0.)

    def test_cci_is_present_in_pipeline(self):
        d=frame()
        self.assertIn('cci',te.compute_enhanced(d,signals.compute_signals(d)))

    def test_validator_rejects_infinity_and_unsorted(self):
        d=frame(); d.iloc[-1,d.columns.get_loc('High')]=np.inf
        self.assertFalse(dq.validate_ohlcv(d)[0])
        self.assertFalse(dq.validate_ohlcv(frame().iloc[::-1])[0])

    def test_validator_rejects_bad_volume_and_impossible_range(self):
        d=frame(); d.iloc[-1,d.columns.get_loc('Volume')]=np.nan
        self.assertFalse(dq.validate_ohlcv(d)[0])
        d=frame(); d.iloc[-1,d.columns.get_loc('High')]=d.Close.iloc[-1]-1
        self.assertFalse(dq.validate_ohlcv(d)[0])

    def test_ml_uses_next_open(self):
        d=frame(30).rename(columns=str.lower)
        d.loc[:,['open','high','low','close']]=[120.,121.,119.5,120.]
        d.iloc[0,d.columns.get_indexer(['open','high','low','close'])]=[100.,101.,99.,100.]
        self.assertEqual(ml.build_labels(d).iloc[0],0)

    def test_ml_horizon_is_purged_at_split(self):
        d=frame(40); d.index=pd.bdate_range('2022-12-01',periods=40)
        ends=ml.label_horizon_end(d)
        selected=(d.index<=pd.Timestamp(ml.TRAIN_END)) & (ends<=pd.Timestamp(ml.TRAIN_END))
        self.assertTrue((ends[selected]<=pd.Timestamp(ml.TRAIN_END)).all())
        self.assertFalse(bool(selected.loc['2022-12-30']))

    def test_old_ml_model_not_served(self):
        with patch.object(ml,'load_model',return_value=object()):
            self.assertIsNone(ml.score_signal(frame().rename(columns=str.lower)))

    def test_replay_reuses_identical_state(self):
        r=h2h.ScoreReplay(frame(),None)
        self.assertEqual(r.at(270),r.at(270))
        self.assertEqual(r.computations,1)

    def test_replay_has_no_future_bar_leakage(self):
        d=frame(); score=h2h.ScoreReplay(d,None).at(270)
        d.loc[d.index[271]:,['Open','High','Low','Close']]*=10
        self.assertEqual(h2h.ScoreReplay(d,None).at(270),score)

    def test_bootstrap_requires_enough_dates(self):
        d=pd.DataFrame({'date':pd.date_range('2026-01-01',periods=10),'fwd':np.arange(10.)})
        self.assertIsNone(h2h.block_edge_interval(d,np.ones(10,dtype=bool)))

    def test_cache_refresh_replaces_current_bar(self):
        with isolated_scanner() as s:
            d=frame(); fresh=d.tail(7).copy(); fresh.iloc[-1,fresh.columns.get_loc('Close')]+=1
            s.groww_client.get_daily_bars.side_effect=[d,fresh]
            with patch.object(s.time,'time',return_value=100):
                first,_=s._get_ohlcv('TEST.NS')
                again,_=s._get_ohlcv('TEST.NS')
            self.assertEqual(s.groww_client.get_daily_bars.call_count,1)
            with patch.object(s.time,'time',return_value=401):
                updated,_=s._get_ohlcv('TEST.NS')
            self.assertEqual(s.groww_client.get_daily_bars.call_count,2)
            self.assertEqual(updated.Close.iloc[-1],fresh.Close.iloc[-1])
            self.assertEqual(len(updated),len(first))

    def test_validation_exception_never_accepts_source(self):
        with isolated_scanner() as s:
            s.groww_client.get_daily_bars.return_value=frame()
            with (patch.object(dq,'validate_ohlcv',side_effect=TypeError('bad frame')),
                 patch.object(s,'_nselib_ohlcv',return_value=None),
                 patch.object(s,'_yf_history',return_value=frame())):
                self.assertEqual(s._get_ohlcv('TEST.NS'),(None,None))

    def test_yahoo_session_is_owned_by_library(self):
        with isolated_scanner() as s:
            s._yf_history('TEST.NS',period='2y')
            s.yf.Ticker.assert_called_once_with('TEST.NS')
            self.assertEqual(s.yf.Ticker.return_value.history.call_args.kwargs['timeout'],15)

    def test_database_migration_preserves_old_scans_and_versions_new_ones(self):
        import db
        with tempfile.TemporaryDirectory() as folder:
            path=pathlib.Path(folder)/'signals.db'
            with sqlite3.connect(path) as con:
                con.executescript(db._CREATE_SQL)
                con.execute("INSERT INTO scan_log(scanned_at,symbol) VALUES ('2026-01-01','OLD')")
            con.close()
            with patch.object(db,'_DB_PATH',path),patch.object(db,'_USE_POSTGRES',False):
                db.init_db(); db.init_db()
                db.save_scan_results([{'symbol':'NEW','score_version':'codex-repair-1','bar_date':'2026-09-07'}])
            with sqlite3.connect(path) as con:
                rows=con.execute('SELECT symbol,score_version,bar_date FROM scan_log ORDER BY id').fetchall()
            con.close()
            self.assertEqual(rows,[('OLD',None,None),('NEW','codex-repair-1','2026-09-07')])

if __name__ == '__main__':
    unittest.main(verbosity=2)
