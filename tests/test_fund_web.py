import sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from unittest.mock import patch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from fund_engine import web


class WebTests(unittest.TestCase):
    def setUp(self):
        app=FastAPI();app.include_router(web.router);self.client=TestClient(app)
    def test_empty_install_has_no_fake_results(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(web,'ROOT',Path(tmp)):
            response=self.client.get('/api/fund-research')
            self.assertEqual(response.status_code,200)
            self.assertEqual(response.json()['state'],'not_run')
            self.assertEqual(response.json()['promotion'],'not_approved')
    def test_page_loads(self):
        response=self.client.get('/fund-research')
        self.assertEqual(response.status_code,200);self.assertIn('Fund Research',response.text)
    def test_debate_cannot_approve_trade(self):
        response=self.client.post('/api/fund-research/review',json={'memo':{},'asof':'2020-01-01T00:00:00Z'})
        self.assertEqual(response.status_code,200);self.assertFalse(response.json()['actionable'])
        self.assertFalse(response.json()['memo_structurally_complete'])
    def test_bad_memo_is_validation_error(self):
        response=self.client.post('/api/fund-research/review',json={'memo':{'evidence':[3]},'asof':'nonsense'})
        self.assertEqual(response.status_code,422)


if __name__=='__main__':unittest.main()
