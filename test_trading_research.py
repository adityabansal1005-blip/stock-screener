import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from fastapi import FastAPI
from fastapi.testclient import TestClient
import trading_research as tr


class ResearchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.patches = [patch.object(tr, 'ROOT', root), patch.object(tr, 'STORE', root / 'reports'),
                        patch.object(tr, 'PYTHON', root / 'python.exe'),
                        patch.object(tr, 'settings', return_value=('test-key', tr.MODEL))]
        for p in self.patches:
            p.start(); self.addCleanup(p.stop)
        tr.PYTHON.touch()
        tr._active = None
        app = FastAPI(); app.include_router(tr.router)
        self.client = TestClient(app)
        self.addCleanup(self.client.close)

    def start(self, symbol='RELIANCE'):
        return self.client.post('/api/trading-research/start', params={'symbol':symbol},
                                headers={'X-Research-Action':'start'})

    def test_no_key_no_worker(self):
        with patch.object(tr, 'settings', return_value=('', tr.MODEL)), patch.object(tr.threading, 'Thread') as thread:
            self.assertEqual(self.start().status_code, 503)
            thread.assert_not_called()

    def test_cross_origin_and_missing_header(self):
        self.assertEqual(self.client.post('/api/trading-research/start?symbol=TCS').status_code, 403)
        r = self.client.post('/api/trading-research/start?symbol=TCS', headers={
            'Origin':'https://evil.example', 'X-Research-Action':'start'})
        self.assertEqual(r.status_code, 403)

    def test_invalid_symbol_and_traversal(self):
        for symbol in ('../secret', 'TCS;cmd', '<script>', 'AAPL.US'):
            self.assertEqual(self.start(symbol).status_code, 422)
        self.assertEqual(tr.ticker_for('M&M'), 'M&M.NS')
        self.assertEqual(tr.ticker_for('500325.BO'), '500325.BO')

    def test_duplicate_and_concurrent_work(self):
        with patch.object(tr.threading, 'Thread') as thread:
            a = self.start().json(); b = self.start().json()
            self.assertEqual(a['id'], b['id']); self.assertTrue(b['reused'])
            self.assertEqual(self.start('TCS').status_code, 409)
            self.assertEqual(thread.call_count, 1)

    def test_finished_cached_and_model_change(self):
        with patch.object(tr.threading, 'Thread'):
            identifier = self.start().json()['id']
            job = tr.read_job(identifier); job['status'] = 'completed'; tr.save_job(job); tr._active = None
            self.assertEqual(self.start().json()['id'], identifier)
            with patch.object(tr, 'settings', return_value=('test-key', 'different-model')):
                self.assertNotEqual(self.start().json()['id'], identifier)

    def test_interrupted_history(self):
        with patch.object(tr.threading, 'Thread'):
            identifier = self.start().json()['id']
        tr._active = None
        self.assertEqual(self.client.get('/api/trading-research/jobs/' + identifier).json()['status'], 'interrupted')

    def test_timeout_releases_slot(self):
        with patch.object(tr.threading, 'Thread'):
            identifier = self.start().json()['id']
        with patch.object(tr.subprocess, 'run', side_effect=subprocess.TimeoutExpired('worker', 600)):
            tr.run_job(tr.read_job(identifier), 'test-key')
        self.assertIsNone(tr._active)
        self.assertEqual(tr.read_job(identifier)['status'], 'failed')

    def test_results_saved_and_credentials_isolated(self):
        with patch.object(tr.threading, 'Thread'):
            identifier = self.start().json()['id']
        (tr.STORE / identifier / 'result.json').write_text(json.dumps({'status':'completed', 'reports':{'news_report':'sample'}}))
        (tr.STORE / identifier / 'price_source.json').write_text(json.dumps({'source':'test snapshot'}))
        with patch.dict(tr.os.environ, {'GROWW_API_KEY':'private-broker-key', 'TELEGRAM_BOT_TOKEN':'private-alert-key'}):
            with patch.object(tr.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0)) as run:
                tr.run_job(tr.read_job(identifier), 'test-key')
                env = run.call_args.kwargs['env']
                self.assertNotIn('GROWW_API_KEY', env); self.assertNotIn('TELEGRAM_BOT_TOKEN', env)
                self.assertEqual(env['GOOGLE_API_KEY'], 'test-key')
        self.assertEqual(tr.read_job(identifier)['status'], 'completed')
        self.assertNotIn('test-key', json.dumps(tr.read_job(identifier)))

    def test_missing_worker_result(self):
        with patch.object(tr.threading, 'Thread'):
            identifier = self.start().json()['id']
        with patch.object(tr.subprocess, 'run', return_value=subprocess.CompletedProcess([], 1)):
            tr.run_job(tr.read_job(identifier), 'test-key')
        self.assertEqual(tr.read_job(identifier)['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
