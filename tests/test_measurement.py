import ast,datetime as dt,math,sys,unittest
from pathlib import Path
import numpy as np,pandas as pd
P=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(P))
from screening_policy import completed_history,assess,IST
from research_measurement import outcomes

class Measurement(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({'open':100.,'high':105.,'low':95.,'close':102.,'volume':1000.},index=pd.bdate_range('2026-01-01',periods=40))
    def test_interior_missing_not_total_loss(self):
        d=self.frame();d.iloc[8]=np.nan;r,s=outcomes(d)
        self.assertAlmostEqual(r.iloc[0],2-.5582);self.assertEqual(s.iloc[0],'observed_with_interior_gap')
    def test_missing_exit_unresolved(self):
        d=self.frame();d.iloc[21]=np.nan;r,s=outcomes(d)
        self.assertTrue(pd.isna(r.iloc[0]));self.assertEqual(s.iloc[0],'unresolved_exit')
    def test_missing_entry_no_fill(self):
        d=self.frame();d.iloc[1]=np.nan;r,s=outcomes(d)
        self.assertTrue(pd.isna(r.iloc[0]));self.assertEqual(s.iloc[0],'unknown_entry')
    def test_zero_volume_exit_unresolved(self):
        d=self.frame();d.iloc[21,d.columns.get_loc('volume')]=0;r,s=outcomes(d)
        self.assertEqual(s.iloc[0],'unresolved_exit')
    def test_partial_day_excluded_from_weekly(self):
        d=pd.concat([self.frame().set_axis(pd.bdate_range('2025-01-01',periods=40)),self.frame().set_axis(pd.bdate_range('2026-01-01',periods=40))]);d.columns=d.columns.str.title();day=d.index[-1].date()
        first,w=completed_history(d,dt.datetime.combine(day,dt.time(14),IST))
        d.iloc[-1]=1e9;second,w2=completed_history(d,dt.datetime.combine(day,dt.time(14),IST))
        pd.testing.assert_frame_equal(first,second);pd.testing.assert_frame_equal(w,w2)
        complete,_=completed_history(d,dt.datetime.combine(day,dt.time(16,30),IST));self.assertEqual(len(complete),len(d))
    def test_gates(self):
        r={'symbol':'X','liq_tradeable':True,'bar_date':'2026-09-07'}
        self.assertTrue(assess(r,'2026-09-07',{})['screen_eligible'])
        self.assertFalse(assess(r,'2026-09-07',{'X':{'status':'blocked'}})['screen_eligible'])
        self.assertFalse(assess(r,'2026-09-08',{})['screen_eligible'])
        self.assertFalse(assess(dict(r,liq_tradeable=None),'2026-09-07',{})['screen_eligible'])
        self.assertFalse(assess(r,'2026-09-07',{})['actionable'])
    def test_manufactured_rr_does_not_change_score(self):
        tree=ast.parse((P/'technical_enhanced.py').read_text(encoding='utf-8'));ns={'_math_isnan':math.isnan}
        for n in tree.body:
            if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and (t.id.startswith('_REGIME_') or t.id=='MOMENTUM_MODE') for t in n.targets):
                for t in n.targets:ns[t.id]=ast.literal_eval(n.value)
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='compute_score100')
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'score','exec'),ns)
        r={'close':100,'atr':3,'sl_buy':91,'rr_ratio':2}
        a=ns['compute_score100'](r.copy());b=ns['compute_score100'](dict(r,rr_ratio=10,resistance_blocks_target=True,stop_capped=True))
        self.assertEqual(a,b)

if __name__=='__main__':unittest.main()
