"""Optional, local-only TradingAgents jobs. No scanner or broker dependencies."""
import datetime as dt
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import uuid
from urllib.parse import urlsplit

from dotenv import dotenv_values
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

ROOT = Path(__file__).resolve().parent
STORE = ROOT / 'research_reports'
PYTHON = ROOT / '.venv-tradingagents' / 'Scripts' / 'python.exe'
MODEL = 'gemini-3.5-flash'
REVISION = '9dee508c44662702281a8dbaad1f7b42179b5ba7'
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
router = APIRouter(prefix='/api/trading-research')
_lock = threading.Lock()
_active = None


def settings():
    # Read on demand so adding a key does not require another restart.
    values = {**os.environ, **dotenv_values(ROOT / '.env')}
    key = (values.get('GOOGLE_API_KEY') or values.get('GEMINI_API_KEY') or '').strip()
    if key.lower().startswith(('your_', 'paste_', 'replace_')):
        key = ''
    return key, values.get('TRADINGAGENTS_MODEL') or MODEL


def local_request(request):
    if not request.client or request.client.host not in ('127.0.0.1', '::1', 'testclient'):
        raise HTTPException(403, 'Research is available from this computer only.')
    if request.url.hostname not in ('localhost', '127.0.0.1', '::1', 'testserver'):
        raise HTTPException(403, 'Use the localhost dashboard address.')
    origin = request.headers.get('origin')
    if origin and urlsplit(origin).netloc != request.url.netloc:
        raise HTTPException(403, 'Cross-site research requests are blocked.')


def ticker_for(symbol):
    symbol = symbol.strip().upper()
    if not re.fullmatch(r'[A-Z0-9][A-Z0-9&-]{0,29}(?:\.NS|\.BO)?', symbol):
        raise HTTPException(422, 'Choose a valid NSE or BSE ticker.')
    return symbol if symbol.endswith(('.NS', '.BO')) else symbol + '.NS'


def read_job(job_id):
    if not re.fullmatch(r'[0-9a-f]{32}', job_id):
        raise HTTPException(404, 'Research report not found.')
    try:
        return json.loads((STORE / job_id / 'job.json').read_text(encoding='utf-8'))
    except FileNotFoundError:
        raise HTTPException(404, 'Research report not found.')


def save_job(job):
    folder = STORE / job['id']
    folder.mkdir(parents=True, exist_ok=True)
    tmp = folder / 'job.tmp'
    tmp.write_text(json.dumps(job, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    tmp.replace(folder / 'job.json')


def recent_jobs():
    jobs = []
    if STORE.exists():
        for path in STORE.glob('*/job.json'):
            try:
                job = json.loads(path.read_text(encoding='utf-8'))
                if job['status'] == 'running' and job['id'] != _active:
                    job = {**job, 'status': 'interrupted', 'error': 'Dashboard restarted before this job finished.'}
                jobs.append(job)
            except (ValueError, KeyError, OSError):
                continue
    return sorted(jobs, key=lambda j: j['created_at'], reverse=True)


def run_job(job, key):
    global _active
    phase = 'price preparation'
    try:
        prepared = subprocess.run(
            [sys.executable, str(ROOT / 'research_price_snapshot.py'), job['id']],
            cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=90, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        if prepared.returncode:
            job.update(status='failed', error='No validated recent prices from Groww or the local archive. Research was not sent to Gemini.')
            return
        job['price_source'] = json.loads((STORE / job['id'] / 'price_source.json').read_text(encoding='utf-8'))
        save_job(job)
        # Do not pass brokerage, Telegram, or unrelated provider credentials.
        env = {k: v for k, v in os.environ.items() if k.upper() in {
            'SYSTEMROOT', 'WINDIR', 'PATH', 'TEMP', 'TMP', 'USERPROFILE',
            'APPDATA', 'LOCALAPPDATA', 'HOMEDRIVE', 'HOMEPATH', 'COMSPEC'
        }}
        env.update(GOOGLE_API_KEY=key, PYTHONUTF8='1', PYTHONIOENCODING='utf-8')
        phase = 'AI research'
        result = subprocess.run(
            [str(PYTHON), str(ROOT / 'trading_research_worker.py'), job['id']],
            cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=600, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
        )
        output_path = STORE / job['id'] / 'result.json'
        if output_path.exists():
            output = json.loads(output_path.read_text(encoding='utf-8'))
            job.update(output)
        else:
            job.update(status='failed', error='Research worker failed. Check the isolated installation and retry.')
        if result.returncode and job['status'] == 'completed':
            job.update(status='failed', error='Research worker exited unexpectedly.')
    except subprocess.TimeoutExpired:
        job.update(status='failed', error=('Price preparation exceeded 90 seconds and was stopped.' if phase == 'price preparation' else 'Research exceeded the 10-minute limit and was stopped.'))
    except Exception:
        job.update(status='failed', error='Research could not finish. Check local storage and the isolated installation.')
    finally:
        job['finished_at'] = dt.datetime.now(IST).isoformat()
        with _lock:
            try:
                save_job(job)
            finally:
                _active = None


@router.get('/status')
def status(request: Request):
    local_request(request)
    key, model = settings()
    return {'installed': PYTHON.is_file(), 'key_configured': bool(key),
            'model': model, 'active_job': _active,
            'ready': PYTHON.is_file() and bool(key)}


@router.get('/jobs')
def jobs(request: Request, symbol: str = ''):
    local_request(request)
    ticker = ticker_for(symbol) if symbol else None
    with _lock:
        found = recent_jobs()
    return [{k: j.get(k) for k in ('id', 'ticker', 'created_at', 'status', 'model', 'error')}
            for j in found if not ticker or j['ticker'] == ticker][:30]


@router.post('/start')
def start(request: Request, symbol: str):
    global _active
    local_request(request)
    if request.headers.get('x-research-action') != 'start':
        raise HTTPException(403, 'Start research from the dashboard.')
    ticker = ticker_for(symbol)
    key, model = settings()
    if not PYTHON.is_file():
        raise HTTPException(503, 'TradingAgents environment is not installed.')
    if not key:
        raise HTTPException(503, 'Add GOOGLE_API_KEY to the app .env file, then retry.')
    now = dt.datetime.now(IST)
    with _lock:
        for old in recent_jobs():
            if old.get('price_flow') == 'groww_archive_v1' and old['ticker'] == ticker and old['model'] == model and old['status'] in ('running', 'completed'):
                if (now - dt.datetime.fromisoformat(old['created_at'])).total_seconds() < 6 * 3600:
                    return {'id': old['id'], 'reused': True}
        if _active:
            raise HTTPException(409, 'Another stock is being researched. Wait for it to finish.')
        job = {'id': uuid.uuid4().hex, 'ticker': ticker, 'model': model,
               'analysis_date': now.date().isoformat(), 'created_at': now.isoformat(),
               'revision': REVISION, 'status': 'running', 'price_flow': 'groww_archive_v1'}
        save_job(job)
        _active = job['id']
        try:
            threading.Thread(target=run_job, args=(job, key), daemon=True).start()
        except Exception:
            _active = None
            job.update(status='failed', error='Could not start research worker.')
            save_job(job)
            raise HTTPException(503, job['error'])
    return {'id': job['id'], 'reused': False}


@router.get('/jobs/{job_id}')
def job_status(request: Request, job_id: str):
    local_request(request)
    with _lock:
        job = read_job(job_id)
        if job['status'] == 'running' and job_id != _active:
            job.update(status='interrupted', error='Dashboard restarted before this job finished.')
    return job


@router.get('/ui.js')
def ui_script():
    return FileResponse(ROOT / 'templates' / 'trading_research.js', media_type='text/javascript')
