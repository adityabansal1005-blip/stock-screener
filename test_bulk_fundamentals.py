import unittest
from unittest.mock import patch
import fundamentals


class BulkTests(unittest.TestCase):
    def test_bulk_hit_does_not_scrape(self):
        with patch('tv_fundamentals.get', return_value={'pe':20}) as tv, patch.object(fundamentals,'_enrich_screener') as scrape:
            self.assertEqual(fundamentals.get_fundamentals('TCS.NS',bulk_only=True)['pe'],20)
            scrape.assert_not_called()

    def test_bulk_miss_does_not_call_yahoo(self):
        with patch('tv_fundamentals.get',return_value=None), patch.object(fundamentals.yf,'Ticker') as yahoo:
            self.assertEqual(fundamentals.get_fundamentals('MISSING.NS',bulk_only=True),{})
            yahoo.assert_not_called()

    def test_regular_mode_keeps_enrichment(self):
        with patch('tv_fundamentals.get',return_value={'pe':20}), patch.object(fundamentals,'_enrich_screener') as scrape:
            fundamentals.get_fundamentals('TCS.NS')
            scrape.assert_called_once()


if __name__ == '__main__': unittest.main()
