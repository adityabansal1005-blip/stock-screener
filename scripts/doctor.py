"""Local diagnostic: never print credentials or call providers."""
import importlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
failed=[]
print('Python:',sys.version.split()[0]);print('Project:',ROOT)
for name in ['fastapi','uvicorn','pandas','numpy','yfinance','growwapi','dotenv','requests','sklearn','lightgbm','nselib','tradingview_screener']:
    try:importlib.import_module(name);print('OK:',name)
    except Exception as exc:failed.append(name);print('MISSING/FAILED:',name,type(exc).__name__)
if not failed:
    from dotenv import dotenv_values
    values=dotenv_values(ROOT/'.env')
    for name in ['GROWW_API_KEY','GROWW_API_SECRET','ANTHROPIC_API_KEY','GOOGLE_API_KEY']:
        print(name+':', 'configured (not authenticated)' if values.get(name) else 'not configured (optional)')
    try:
        import screening_policy
        screening_policy.load_risks()
        print('OK: eligibility policy and financial-risk registry')
    except Exception:failed.append('risk registry');print('FAILED: risk registry')
print('No market data, AI requests, alerts or orders were sent.')
sys.exit(1 if failed else 0)
