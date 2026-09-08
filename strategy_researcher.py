"""
Strategy Researcher — uses Scrapling to fetch financial education pages
and Claude claude-opus-4-7 to extract + implement additional trading strategies.

Extends strategy_library.py with strategies found on:
  - Investopedia technical analysis articles
  - StockCharts ChartSchool (where accessible)
  - Wikipedia: List of technical indicators
  - Other finance education sources

Run once: python strategy_researcher.py
Output: strategy_library_researched.py (auto-appended to strategy_library.py)
"""
import re
import sys
import time
import json
import os
from pathlib import Path

# Load .env before anthropic client is created
_env = Path(__file__).parent / ".env"
if _env.exists():
    for _line in _env.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

import anthropic

# Scrapling for anti-bot web fetching
from scrapling import Fetcher

_ROOT = Path(__file__).parent
_OUT  = _ROOT / "strategy_library_researched.py"

client = anthropic.Anthropic()

# ── URLs to research ─────────────────────────────────────────────────────────
RESEARCH_URLS = [
    # Investopedia strategy/pattern articles
    ("https://www.investopedia.com/terms/t/technicalanalysis.asp",
     "overview of technical analysis methods and indicators"),
    ("https://www.investopedia.com/articles/trading/09/trade-the-trend.asp",
     "trend trading strategies"),
    ("https://www.investopedia.com/terms/c/chartpattern.asp",
     "chart patterns used for trading"),
    ("https://www.investopedia.com/terms/m/momentum_investing.asp",
     "momentum investing strategies"),
    ("https://www.investopedia.com/terms/m/mean-reversion.asp",
     "mean reversion trading strategies"),
    ("https://www.investopedia.com/articles/trading/06/swingtradingintro.asp",
     "swing trading strategies and setups"),
    ("https://www.investopedia.com/terms/b/breakout.asp",
     "breakout trading strategies"),
    ("https://en.wikipedia.org/wiki/Technical_analysis",
     "comprehensive list of all technical indicators and analysis methods"),
    ("https://www.investopedia.com/terms/r/relative-strength-index.asp",
     "RSI-based trading strategies"),
    ("https://www.investopedia.com/terms/m/macd.asp",
     "MACD-based trading strategies"),
]

# ── System prompt for Claude ──────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert quantitative trader and Python developer.
Your job: given raw text from a finance education website, extract EVERY trading strategy,
signal, or pattern mentioned and implement each as a Python function.

Rules for each function:
1. Name: snake_case, descriptive (e.g., `s_rsi_divergence`, `s_fibonacci_retracement`)
2. Signature: `def s_NAME(df: pd.DataFrame) -> pd.Series:` where df has lowercase OHLCV columns
3. Returns boolean Series — True where the entry signal fires
4. No external imports beyond numpy (np) and pandas (pd)
5. Must be self-contained (define any helper calculations inside or use provided helpers)
6. Add a one-line docstring naming the strategy and its source/author

Only implement strategies NOT already in this list (skip duplicates):
{existing}

Output ONLY valid Python code. No explanations, no markdown, no ``` blocks.
Start directly with the function definitions.
If no new strategies are found, output: # No new strategies found
"""

EXISTING_STRATEGIES = [
    "ema_stack", "golden_cross", "price_above_200ema", "supertrend", "adx_bull",
    "psar", "keltner_trend", "rsi_momentum", "rsi_cross_50", "macd", "cci_breakout",
    "roc_positive", "stoch_golden_cross", "mfi", "williams_r", "turtle", "donchian",
    "bollinger_upper", "inside_bar", "consolidation_breakout", "cup_handle", "vcp",
    "rsi_oversold", "bollinger_lower", "cci_oversold", "stoch_oversold", "pullback",
    "ema21_bounce", "vwap", "bb_squeeze", "keltner_bb_squeeze", "atr_expansion",
    "narrow_range", "high_tight_flag", "bullish_engulfing", "hammer", "morning_star",
    "harami", "three_white_soldiers", "pin_bar", "marubozu", "doji", "obv",
    "volume_surge", "cmf", "mfi_cross", "volume_dry_up", "weinstein_stage2",
    "elder_triple_screen", "darvas_box", "canslim", "livermore_pivot",
]


def fetch_page(url: str) -> str:
    """Fetch page text using Scrapling (anti-bot fingerprinting)."""
    try:
        f = Fetcher(auto_match=False)
        page = f.get(url, timeout=25, stealthy_headers=True)
        # Extract readable text — skip scripts, styles, nav, footer
        texts = []
        for tag in ["p", "h1", "h2", "h3", "h4", "li", "td", "th"]:
            for el in page.css(tag):
                t = el.text
                if t and len(t.strip()) > 20:
                    texts.append(t.strip())
        content = "\n".join(texts)
        # Limit to ~8000 chars to fit in context
        return content[:8000]
    except Exception as e:
        print(f"  [Scrape] {url} failed: {e}")
        return ""


def extract_strategies_from_text(url: str, description: str, text: str) -> str:
    """
    Ask Claude to extract and implement trading strategies from the page text.
    Returns Python code string (may be empty if no new strategies found).
    """
    if not text.strip():
        return ""

    existing_str = ", ".join(EXISTING_STRATEGIES)
    prompt = f"""URL: {url}
Description: {description}

Page content:
{text}

Extract and implement all NEW trading strategies as Python functions.
"""
    try:
        msg = client.messages.create(
            model="claude-opus-4-7",
            max_tokens=4096,
            thinking={"type": "adaptive"},
            system=SYSTEM_PROMPT.format(existing=existing_str),
            messages=[{"role": "user", "content": prompt}],
        )
        # Get text content from response
        code = ""
        for block in msg.content:
            if hasattr(block, "text"):
                code += block.text
        code = code.strip()
        if code == "# No new strategies found" or not code:
            return ""
        return code
    except Exception as e:
        print(f"  [Claude] API error: {e}")
        return ""


def _validate_code(code: str) -> bool:
    """Quick syntax check — don't add broken code."""
    try:
        compile(code, "<string>", "exec")
        return True
    except SyntaxError as e:
        print(f"  [Validate] Syntax error: {e}")
        return False


def _extract_function_names(code: str) -> list[str]:
    return re.findall(r"^def (s_\w+)\(", code, re.MULTILINE)


def run_research(force: bool = False):
    """
    Main research loop. Scrapes all URLs, asks Claude for new strategies,
    writes validated code to strategy_library_researched.py.
    """
    if _OUT.exists() and not force:
        print(f"[Researcher] {_OUT.name} already exists. Use force=True to re-research.")
        print(f"  Found existing file — loading into strategy_library...")
        _register_researched()
        return

    print("[Researcher] Starting strategy research from", len(RESEARCH_URLS), "sources...")
    all_code_blocks = [
        "# Auto-generated by strategy_researcher.py",
        "# Do not edit manually — re-run strategy_researcher.py to refresh",
        "",
        "import numpy as np",
        "import pandas as pd",
        "",
    ]
    all_new_functions: list[str] = []

    for url, description in RESEARCH_URLS:
        print(f"\n[Scrape] {url[:60]}...")
        text = fetch_page(url)
        if not text:
            continue
        print(f"  Got {len(text)} chars. Asking Claude for new strategies...")
        code = extract_strategies_from_text(url, description, text)
        if not code:
            print("  No new strategies found.")
            continue

        new_fns = _extract_function_names(code)
        if not new_fns:
            print("  Claude returned code with no s_* functions — skipping.")
            continue

        if not _validate_code(code):
            print("  Code has syntax errors — skipping.")
            continue

        # Deduplicate
        new_fns = [f for f in new_fns if f not in all_new_functions]
        if not new_fns:
            print("  All functions already collected.")
            continue

        all_new_functions.extend(new_fns)
        all_code_blocks.append(f"\n# ── From: {url} ────")
        all_code_blocks.append(code)
        print(f"  Added {len(new_fns)} new strategies: {', '.join(new_fns)}")
        time.sleep(1.5)  # be polite to servers + Claude rate limit

    if not all_new_functions:
        print("\n[Researcher] No new strategies discovered beyond existing library.")
        _OUT.write_text("# No additional strategies found\n", encoding="utf-8")
        return

    # Build STRATEGIES_RESEARCHED dict
    entries = [f'    "{fn.replace("s_","",1)}": {fn},' for fn in all_new_functions]
    registry = "\n".join([
        "",
        "STRATEGIES_RESEARCHED = {",
        *entries,
        "}",
    ])
    all_code_blocks.append(registry)

    final_code = "\n".join(all_code_blocks)
    _OUT.write_text(final_code, encoding="utf-8")
    print(f"\n[Researcher] Done. {len(all_new_functions)} new strategies written to {_OUT.name}")
    _register_researched()


def _register_researched():
    """Merge STRATEGIES_RESEARCHED into strategy_library.STRATEGIES at runtime."""
    try:
        import importlib, sys
        if str(_ROOT) not in sys.path:
            sys.path.insert(0, str(_ROOT))
        spec = importlib.util.spec_from_file_location("strategy_library_researched", _OUT)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        extra = getattr(mod, "STRATEGIES_RESEARCHED", {})
        if extra:
            import strategy_library
            before = len(strategy_library.STRATEGIES)
            strategy_library.STRATEGIES.update(extra)
            print(f"[Researcher] Merged {len(extra)} researched strategies into library "
                  f"({before} → {len(strategy_library.STRATEGIES)} total)")
    except Exception as e:
        print(f"[Researcher] Could not register researched strategies: {e}")


def get_all_strategies():
    """
    Returns the full strategy dict — base library + researched additions.
    Call this instead of importing strategy_library.STRATEGIES directly.
    """
    import strategy_library
    if _OUT.exists():
        _register_researched()
    return strategy_library.STRATEGIES


if __name__ == "__main__":
    force = "--force" in sys.argv
    run_research(force=force)
