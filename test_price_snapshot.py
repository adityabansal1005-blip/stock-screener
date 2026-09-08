import datetime as dt
import tempfile
from pathlib import Path
import unittest
import importlib.util
import json
from unittest.mock import patch
import numpy as np
import pandas as pd
from research_price_snapshot import validate


class SnapshotTests(unittest.TestCase):
    def frame(self):
        index = pd.bdate_range(end='2026-09-07', periods=240)
        return pd.DataFrame({'Open':100., 'High':102., 'Low':99., 'Close':101., 'Volume':1000.}, index=index)

    def test_bad_latest_close_rejected(self):
        frame = self.frame(); frame.iloc[-1, frame.columns.get_loc('Close')] = np.nan
        with self.assertRaises(ValueError): validate(frame, dt.date(2026,9,7))

    def test_future_row_excluded(self):
        frame = self.frame(); frame.loc[pd.Timestamp('2026-09-08')] = [100,102,99,101,1000]
        self.assertEqual(str(validate(frame, dt.date(2026,9,7)).index[-1].date()), '2026-09-07')

    def test_stale_rejected(self):
        with self.assertRaises(ValueError): validate(self.frame(), dt.date(2026,9,20))

    @unittest.skipUnless(importlib.util.find_spec('tradingagents'), 'Requires the optional TradingAgents environment')
    def test_all_price_routes_use_snapshot(self):
        from research_price_adapter import install
        from tradingagents.dataflows import stockstats_utils, market_data_validator, y_finance, interface
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder); frame = self.frame(); frame.index.name = 'Date'; frame.to_csv(path/'prices.csv')
            (path/'price_source.json').write_text(json.dumps({'ticker':'AARTIPHARM.NS','source':'test','last_bar':'2026-09-07'}))
            install(path)
            with patch('yfinance.download', side_effect=AssertionError('Yahoo prices forbidden')), patch('yfinance.Ticker', side_effect=AssertionError('Yahoo prices forbidden')):
                for loader in (stockstats_utils.load_ohlcv, market_data_validator.load_ohlcv, y_finance.load_ohlcv):
                    self.assertEqual(len(loader('AARTIPHARM.NS','2026-09-07')),240)
                    with self.assertRaises(ValueError): loader('RELIANCE.NS','2026-09-07')
                self.assertIn('test',interface.VENDOR_METHODS['get_stock_data']['local_snapshot']('AARTIPHARM.NS','2026-09-01','2026-09-07'))
                self.assertIn('101.00',market_data_validator.build_verified_market_snapshot('AARTIPHARM.NS','2026-09-07'))


if __name__ == '__main__': unittest.main()
