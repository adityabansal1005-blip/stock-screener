"""
Standalone ML training runner. Run: python run_ml_training.py
Loads .env, runs researcher, downloads data, trains LightGBM.
"""
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

# ── load .env manually ────────────────────────────────────────
_env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(_env_path):
    with open(_env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    print("[Init] .env loaded")

UNIVERSE = sys.argv[1] if len(sys.argv) > 1 else "nifty500"
print(f"[Init] Universe: {UNIVERSE}")
print(f"[Init] Anthropic key: {'set' if os.environ.get('ANTHROPIC_API_KEY','').startswith('sk-') else 'MISSING'}")

# ── Phase 0: web research ─────────────────────────────────────
print("\n" + "="*60)
print("PHASE 0: Strategy Researcher (web scraping + Claude)")
print("="*60)
try:
    import strategy_researcher
    strategy_researcher.run_research()
except Exception as e:
    print(f"[Phase 0] Researcher failed: {e} — continuing with base library")

# ── Phase 1: download 10-year OHLCV ──────────────────────────
print("\n" + "="*60)
print(f"PHASE 1: Downloading 10-year OHLCV for {UNIVERSE}")
print("="*60)
import data_manager
from stocks_nse import get_all_symbols

symbols_raw = list(get_all_symbols(UNIVERSE))
ns_symbols  = [f"{s.replace('.NS','').replace('.BO','')}.NS" for s in symbols_raw]
print(f"[Phase 1] {len(ns_symbols)} symbols to load")

data = {}
t0 = time.time()
for i, sym in enumerate(ns_symbols):
    df = data_manager.load(sym)
    if df is not None:
        data[sym.replace(".NS", "")] = df
    if (i + 1) % 25 == 0 or (i + 1) == len(ns_symbols):
        elapsed = time.time() - t0
        rate = (i + 1) / elapsed
        remaining = (len(ns_symbols) - i - 1) / rate if rate > 0 else 0
        print(f"  {i+1}/{len(ns_symbols)} loaded ({len(data)} ok) "
              f"— {elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining")

print(f"[Phase 1] Done: {len(data)} symbols with data")

if len(data) < 10:
    print("[Phase 1] Too few symbols — aborting")
    sys.exit(1)

# ── Phase 2: train ────────────────────────────────────────────
print("\n" + "="*60)
print("PHASE 2: LightGBM Training (84 features, T1/SL target)")
print("="*60)
import ml_signal

t1 = time.time()
metrics = ml_signal.train(list(data.keys()), data=data, model_id="default")
elapsed = time.time() - t1

print("\n" + "="*60)
print("TRAINING COMPLETE")
print("="*60)
print(f"  Symbols used:       {metrics.get('symbols_used')}")
print(f"  Features:           {metrics.get('n_features')}")
print(f"  Train samples:      {metrics.get('n_train'):,}")
print(f"  Test samples:       {metrics.get('n_test'):,}")
print(f"  Val samples:        {metrics.get('n_val'):,}")
print(f"  Win rate in data:   {metrics.get('win_rate_in_data',0):.1%}")
print(f"  Train accuracy:     {metrics.get('train_acc',0):.1%}")
print(f"  Test accuracy:      {metrics.get('test_acc',0):.1%}")
print(f"  Test AUC:           {metrics.get('test_auc','N/A')}")
print(f"  Val accuracy:       {metrics.get('val_acc','N/A')}")
print(f"  Val AUC:            {metrics.get('val_auc','N/A')}")
print(f"  Training time:      {elapsed:.0f}s")
print(f"  Target:             {metrics.get('target')}")
print()
print("Top 10 features by importance:")
for name, imp in metrics.get("top_features", [])[:10]:
    bar = "#" * int(imp * 100)
    print(f"  {name:<35} {imp:.4f}  {bar}")
print()
print("Model saved as 'default' in strategy_data.db")
print("Restart the server to use the new model in live scoring.")
