# ============================================================
#  CONFIGURATION — reads from environment variables.
#  Locally: create a .env file (see .env.example).
#  On Railway: set variables in the Railway dashboard.
# ============================================================
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).with_name(".env"), override=True)  # .env always wins over OS env vars (fixes empty stale vars)

# --- Groww API ---
GROWW_API_KEY    = os.environ.get("GROWW_API_KEY", "")
GROWW_API_SECRET = os.environ.get("GROWW_API_SECRET", "")

# --- Telegram Alerts ---
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "YOUR_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID", "YOUR_CHAT_ID")

# --- Claude AI Analyst ---
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "YOUR_ANTHROPIC_API_KEY")
AI_ANALYSE_TOP_N  = int(os.environ.get("AI_ANALYSE_TOP_N", "3"))

# --- Position Sizing ---
ACCOUNT_SIZE_INR   = float(os.environ.get("ACCOUNT_SIZE_INR", "500000"))
RISK_PER_TRADE_PCT = float(os.environ.get("RISK_PER_TRADE_PCT", "1.0"))

# --- TrueData ---
TRUEDATA_USER     = os.environ.get("TRUEDATA_USER", "")
TRUEDATA_PASSWORD = os.environ.get("TRUEDATA_PASSWORD", "")

# DhanHQ v2 — primary OHLCV source (daily + intraday). Inert until both are set.
DHAN_CLIENT_ID    = os.environ.get("DHAN_CLIENT_ID", "")
DHAN_ACCESS_TOKEN = os.environ.get("DHAN_ACCESS_TOKEN", "")

# --- Scan Settings ---
SCAN_UNIVERSE   = os.environ.get("SCAN_UNIVERSE", "nifty500")
ALERT_MIN_SCORE = int(os.environ.get("ALERT_MIN_SCORE", "4"))
AUTO_SCAN_HOURS = [h.strip() for h in os.environ.get("AUTO_SCAN_HOURS", "").split(",") if h.strip()]
STARTUP_SCAN_ENABLED = os.environ.get("STARTUP_SCAN_ENABLED", "false").lower() == "true"

# --- Daily full-universe scan (feeds the crossing detector) ---
# The intraday cadence above scans a narrow universe for speed. That is fine for
# refreshing large caps, and useless for finding a stock crossing into the bar:
# measured 31 Aug 2026, NONE of MOREPENLAB / SHILPAMED / SUDEEPPHRM / ATHERENERG
# is in nifty50, and only ATHERENERG is in nifty500. A crossing detector fed a
# 50-stock universe can only ever report crossings among 50 large caps.
# So the full universe runs once daily after the close, when it costs nothing in
# latency and the day's bars are final.
FULL_SCAN_AT       = os.environ.get("FULL_SCAN_AT", "18:30")
FULL_SCAN_UNIVERSE = os.environ.get("FULL_SCAN_UNIVERSE", "all_india")
FULL_SCAN_ENABLED  = os.environ.get("FULL_SCAN_ENABLED", "false").lower() == "true"
