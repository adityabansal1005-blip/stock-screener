import sys,tempfile,unittest,sqlite3
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from unittest.mock import patch
import pandas as pd
import data_manager as dm


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.patch=patch.object(dm,'_DB_PATH',Path(self.tmp.name)/'archive.db');self.patch.start();dm._init()
        self.d=pd.DataFrame({'open':[10.,11.],'high':[11.,12.],'low':[9.,10.],'close':[10.,11.],'volume':[100.,100.]},index=pd.to_datetime(['2020-01-01','2020-01-02']))
        self.d.attrs={'provider':'test_source','price_basis':'raw'}
    def tearDown(self):self.patch.stop();self.tmp.cleanup()
    def close(self):
        with dm._conn() as c:return c.execute("SELECT close FROM ohlcv WHERE symbol='A' ORDER BY date").fetchall()
    def test_new_then_same_source_complete_refresh(self):
        self.assertTrue(dm._store('A',self.d))
        self.assertTrue(dm._store('A',self.d))
        self.assertEqual(self.close(),[(10.,),(11.,)])
    def test_provider_change_quarantines_incoming(self):
        dm._store('A',self.d);other=self.d*5;other.attrs={'provider':'other','price_basis':'raw'}
        self.assertFalse(dm._store('A',other));self.assertEqual(self.close(),[(10.,),(11.,)])
        self.assertFalse(dm._is_stale('A'))
        self.assertEqual(len(list(Path(self.tmp.name).glob('data/source_snapshots/*/*/provenance.json'))),2)
    def test_unknown_legacy_provenance_cannot_be_overwritten(self):
        with dm._conn() as c:c.execute("INSERT INTO ohlcv VALUES('A','2020-01-01',10,11,9,10,100)")
        self.assertFalse(dm._store('A',self.d));self.assertEqual(self.close(),[(10.,)])
    def test_partial_refresh_does_not_splice_adjustments(self):
        dm._store('A',self.d)
        self.assertFalse(dm._store('A',self.d.iloc[1:].copy()));self.assertEqual(self.close(),[(10.,),(11.,)])


if __name__=='__main__':unittest.main()
