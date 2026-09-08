"""Inspect exactly the staged Git tree; never print sensitive content."""
import re,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
files=subprocess.check_output(['git','diff','--cached','--name-only','--diff-filter=ACMR','-z'],cwd=ROOT).decode().split('\0')
bad=[]
for name in filter(None,files):
    path=Path(name)
    if (name.startswith(('.venv','research_reports/','data/','output/','paper_records/')) or
        (path.name.startswith('.env') and path.name!='.env.example') or
        path.suffix.lower() in ('.db','.pkl','.pickle','.csv','.parquet','.pem','.key','.log')):
        bad.append((name,'excluded local data/configuration'))
    data=subprocess.check_output(['git','show',':'+name],cwd=ROOT)
    if len(data)>2_000_000:bad.append((name,'unexpected large file'))
    text=data.decode('utf-8',errors='ignore')
    for pattern in [r'AIza[0-9A-Za-z_-]{30,}',r'sk-ant-[A-Za-z0-9_-]{20,}',r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',r'eyJ[A-Za-z0-9_-]{30,}\.[A-Za-z0-9_-]{30,}\.[A-Za-z0-9_-]{20,}']:
        if re.search(pattern,text):bad.append((name,'credential-like content'))
if bad:
    for name,reason in bad:print('BLOCKED:',name,reason)
    sys.exit(1)
print('PASS: staged files contain no excluded data or recognized credential patterns.')
