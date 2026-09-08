"""Simple local audit launcher. No downloads or strategy execution by default."""
from datetime import datetime
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parents[1]
archive=root/'strategy_data.db'
if not archive.exists():
    print('No local historical archive. This download does not include private scan data.')
    print('Ask your coding assistant to read docs/FUND_PROTOCOL.md and help prepare a source-verified archive.')
    raise SystemExit(1)
output=root/'fund_runs'/('audit_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
raise SystemExit(subprocess.call([sys.executable,'-m','fund_engine.run','--archive',str(archive),'--output',str(output)],cwd=root))
