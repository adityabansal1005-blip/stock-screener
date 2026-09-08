"""
Claude AI synthesis — regime-aware analysis with position sizing.
Uses claude-haiku-4-5 for speed. Adapts its strategy guidance based on
market regime (bull/bear/sideways) so recommendations are contextually correct.
"""
import json
import math
import config

_client = None


def _get_client():
    global _client
    if _client is not None:
        return _client
    key = getattr(config, "ANTHROPIC_API_KEY", "")
    if not key or key == "YOUR_ANTHROPIC_API_KEY":
        return None
    try:
        import anthropic
        _client = anthropic.Anthropic(api_key=key)
        return _client
    except ImportError:
        print("[AI] anthropic package not installed — run: pip install anthropic")
        return None


def _null_result():
    return {
        "ai_verdict": None, "ai_confidence": None,
        "ai_summary": None, "ai_strengths": None,
        "ai_risk_factors": None, "ai_suggested_approach": None,
        "ai_bull_case": None, "ai_bear_case": None, "ai_invalidation": None,
        "ai_catalyst": None, "ai_trade_plan": None, "position_size": None,
        "position_value": None, "risk_amount": None,
    }


def compute_position_size(tech: dict) -> dict:
    """
    Risk-based position sizing: risk 2% of account per trade.
    position_size = floor((account × risk%) / (close − SL))
    """
    account  = float(getattr(config, "ACCOUNT_SIZE_INR", 500_000))
    risk_pct = float(getattr(config, "RISK_PER_TRADE_PCT", 2.0))

    try:
        close = float(tech.get("close") or 0)
        sl    = float(tech.get("sl_buy") or 0)
    except (TypeError, ValueError):
        return {"position_size": None, "position_value": None, "risk_amount": None}

    if math.isnan(close) or math.isnan(sl) or close <= 0 or sl <= 0 or sl >= close:
        return {"position_size": None, "position_value": None, "risk_amount": None}

    risk_per_share = close - sl
    risk_capital   = account * risk_pct / 100
    shares         = math.floor(risk_capital / risk_per_share)

    # Hard cap: no single position > 15% of account
    max_shares = math.floor(account * 0.15 / close)
    shares     = min(shares, max_shares)

    # Liquidity cap: a position must stay small against the stock's own daily
    # turnover. Sizing purely off account risk ignores whether the market can
    # absorb the order — on a thin name a "2% risk" position can be 20% of a
    # day's volume, where you move the price against yourself on entry AND
    # exit and real slippage dwarfs the 0.05% the backtest assumes.
    liq_cap = tech.get("liq_max_position")
    capped_by_liquidity = False
    if liq_cap and liq_cap > 0:
        liq_shares = math.floor(liq_cap / close)
        if liq_shares < shares:
            shares = liq_shares
            capped_by_liquidity = True

    if shares <= 0:
        return {"position_size": None, "position_value": None, "risk_amount": None,
                "position_capped_by_liquidity": bool(liq_cap)}

    return {
        "position_size":  shares,
        "position_value": round(shares * close, 2),
        "risk_amount":    round(shares * risk_per_share, 2),
        "position_capped_by_liquidity": capped_by_liquidity,
    }


def _extract_json(text: str) -> dict | None:
    """
    Robustly extract the first complete JSON object from Claude's response.
    Handles: raw JSON, ```json fences, leading prose before {, trailing text after }.
    Does NOT patch incomplete JSON — returns None if no valid object is found.
    """
    if not text:
        return None

    # Strip markdown code fences
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            candidate = part.strip()
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
            if candidate.startswith("{"):
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    pass

    # Find first { and try progressively shorter substrings from the last }
    start = text.find("{")
    if start == -1:
        return None
    end = text.rfind("}")
    if end == -1 or end < start:
        return None

    candidate = text[start:end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Last resort: scan backwards from last } to find a valid closing point
    for i in range(len(text) - 1, start, -1):
        if text[i] == "}":
            try:
                return json.loads(text[start:i + 1])
            except json.JSONDecodeError:
                continue

    return None


def analyze_stock(symbol: str, tech: dict, fundamentals: dict, market: dict,
                  timeout: int = 40) -> dict:
    """
    Regime-aware Claude analysis. The prompt adapts its strategy guidance
    based on current market regime so AI never recommends momentum buys
    in a confirmed bear market, etc.
    """
    client = _get_client()
    pos    = compute_position_size(tech)

    if client is None:
        return {**_null_result(), **pos}

    signals_text = "\n".join(f"  • {s}" for s in tech.get("signals", []))
    if not signals_text:
        signals_text = "  • No strong signals detected"

    # Market regime context
    regime_data    = tech.get("regime", market)
    regime         = regime_data.get("regime", "UNKNOWN") if isinstance(regime_data, dict) else "UNKNOWN"
    regime_label   = regime_data.get("regime_label", "") if isinstance(regime_data, dict) else ""
    regime_strategy= regime_data.get("strategy", "") if isinstance(regime_data, dict) else ""

    # FII/DII context
    fii_dii  = tech.get("fii_dii", {})
    fii_text = "N/A"
    if fii_dii:
        fnet = fii_dii.get("fii_net", 0)
        dnet = fii_dii.get("dii_net", 0)
        fii_text = (
            f"FII net ₹{fnet:+,.0f} Cr | DII net ₹{dnet:+,.0f} Cr | "
            f"Combined ₹{fii_dii.get('combined_net', 0):+,.0f} Cr"
        )

    # Delivery %
    delv = tech.get("delivery_pct")
    delv_str = f"{delv:.1f}% {'(high — genuine buying)' if delv and delv > 50 else '(low — speculative)' if delv and delv < 25 else ''}" if delv else "N/A"

    # Relative strength
    rs = tech.get("rs_vs_nifty")
    rs_str = f"{rs:+.1f}% vs Nifty (3M)" if rs is not None else "N/A"

    # Stage analysis
    stage_label   = tech.get("stage_label", "Unknown")
    stage_desc    = tech.get("stage_desc", "")
    setup_status  = tech.get("setup_status", "WATCH")
    breakout_lvl  = tech.get("breakout_level")
    breakout_dist = tech.get("breakout_dist_pct")
    key_support   = tech.get("key_support")
    ema150        = tech.get("ema150")
    ema150_slope  = tech.get("ema150_slope_pct")
    dist_52w_high = tech.get("dist_52w_high") or tech.get("near_high_pct")
    dist_52w_low  = tech.get("dist_52w_low")

    breakout_str = f"₹{breakout_lvl} ({breakout_dist:+.1f}% away)" if breakout_lvl and breakout_dist is not None else "None detected"
    support_str  = f"₹{key_support}" if key_support else "None detected"
    ema150_str   = f"₹{ema150} (slope {ema150_slope:+.3f}%/4wk)" if ema150 and ema150_slope is not None else "N/A"

    # Weekly trend
    wt = tech.get("weekly_trend", "N/A")

    # Ichimoku
    ichi = tech.get("ichimoku_signal", "N/A")

    # Parabolic SAR
    psar = tech.get("psar_signal", "N/A")
    psar_val = tech.get("psar_val")
    psar_str = f"{psar} (₹{psar_val})" if psar_val else psar

    # Stoch RSI
    srsi_k = tech.get("stoch_rsi_k")
    srsi_str = f"K={srsi_k:.0f}" if srsi_k is not None else "N/A"

    # Williams %R
    wr = tech.get("williams_r")
    wr_str = f"{wr:.0f}" if wr is not None else "N/A"

    # OBV
    obv_t = tech.get("obv_trend", "N/A")

    # Fundamentals block
    fund_score   = fundamentals.get("fund_score")
    pe           = fundamentals.get("pe")
    pb           = fundamentals.get("pb")
    roe          = fundamentals.get("roe")
    de           = fundamentals.get("debt_equity")
    rev_g        = fundamentals.get("revenue_growth")
    earn_g       = fundamentals.get("earnings_growth")
    net_margin   = fundamentals.get("net_margin")
    sector       = fundamentals.get("sector", "N/A") or "N/A"

    def _fmt(v, suffix="", prefix=""):
        return f"{prefix}{v}{suffix}" if v is not None else "N/A"

    fund_text = f"""  Fundamental Score: {fund_score if fund_score is not None else 'N/A'}/100  |  Sector: {sector}
  P/E: {_fmt(pe, 'x')}  |  P/B: {_fmt(pb, 'x')}  |  ROE: {_fmt(roe, '%')}
  Revenue Growth (YoY): {_fmt(rev_g, '%')}  |  Earnings Growth: {_fmt(earn_g, '%')}
  Debt/Equity: {_fmt(de, 'x')}  |  Net Margin: {_fmt(net_margin, '%')}"""

    prompt = f"""You are a senior Indian equity analyst (NSE/BSE) with 30 years of experience across bull and bear cycles.
You write for BEGINNER investors. Plain English only — no jargon. If you must use a technical term, explain it in brackets.

IMPORTANT LANGUAGE RULES:
- Never say "big investors are accumulating" or "smart money is buying" — you cannot know intent from price data alone
- Instead describe OBSERVABLE patterns: "delivery percentage is above average, meaning more buyers are holding overnight rather than selling the same day"
- Never assign certainty to price targets — use "potential upside zone" framing
- Always include genuine reasons the trade could fail — not just token warnings
- Setup Quality score reflects how many indicators align, NOT probability of profit

━━━ MARKET REGIME: {regime_label} ━━━
{regime_strategy}

━━━ STOCK: {symbol} ━━━

TECHNICAL SCORE: {tech.get('score100', 'N/A')}/100  |  Verdict: {tech.get('verdict', 'N/A')}
Close: ₹{tech.get('close')}  |  Change: {tech.get('change_pct', 0):+.2f}%  |  ATR: ₹{tech.get('atr')}

TREND INDICATORS:
  Supertrend: {tech.get('supertrend_label', 'N/A')}  |  Parabolic SAR: {psar_str}
  Ichimoku Cloud: {ichi}  |  Weekly Trend: {wt}
  ADX: {tech.get('adx', 'N/A')} (+DI {tech.get('adx_plus_di','N/A')} / -DI {tech.get('adx_minus_di','N/A')})

MOMENTUM:
  RSI(14): {tech.get('rsi')}  |  Stoch RSI K: {srsi_str}  |  Williams %R: {wr_str}
  Stoch K/D: {tech.get('stoch_k','N/A')}/{tech.get('stoch_d','N/A')}  |  CCI: {tech.get('cci','N/A')}
  MACD Hist: {tech.get('macd_hist', 0):+.4f}

VOLUME & ACCUMULATION:
  Vol Ratio: {tech.get('vol_ratio')}x avg  |  OBV: {obv_t}
  Delivery %: {delv_str}

RELATIVE PERFORMANCE:
  RS vs Nifty (3M): {rs_str}

SIGNALS TRIGGERED:
{signals_text}

STAGE ANALYSIS: {stage_label} — {stage_desc}
  Setup Status: {setup_status}  |  EMA150: {ema150_str}
  Distance from 52W High: {f"{dist_52w_high:+.1f}%" if dist_52w_high is not None else "N/A"}  |  Distance from 52W Low: {f"{dist_52w_low:+.1f}%" if dist_52w_low is not None else "N/A"}

BREAKOUT LEVELS:
  Breakout Trigger: {breakout_str}
  Key Support: {support_str}
  Stop Loss: ₹{tech.get('sl_buy')}  |  Target 1: ₹{tech.get('target1')}  |  Target 2: ₹{tech.get('target2')}
  R:R Ratio: {tech.get('rr_ratio')}:1  |  Pivot Support: ₹{tech.get('support')}  |  Pivot Resistance: ₹{tech.get('resistance')}

CANDLESTICK PATTERNS: {', '.join(tech.get('patterns', [])) or 'None'}

FUNDAMENTALS:
{fund_text}

BROAD MARKET:
  Nifty: ₹{market.get('nifty_price')} ({market.get('nifty_change',0):+.2f}%) — {market.get('nifty_trend')}
  India VIX: {market.get('vix')}{'  ⚠ HIGH VOLATILITY' if market.get('vix_high') else ''}
  FII/DII Today: {fii_text}

━━━ TASK ━━━
Assess this setup honestly — both what supports it AND what could make it fail.

Respond with ONLY valid JSON:
{{
  "verdict": "<STRONG BUY | BUY | HOLD | SELL | STRONG SELL>",
  "setup_quality": <integer 0-100 — reflects indicator alignment, NOT win probability>,
  "summary": "<one plain-English sentence: what is this stock doing and why does it stand out (or not)>",
  "strengths": [
    "<observable reason 1 the setup looks good — specific, data-grounded, no hype>",
    "<observable reason 2>",
    "<observable reason 3 — omit if fewer than 3 genuine reasons>"
  ],
  "risk_factors": [
    "<specific reason this setup may fail — e.g. RSI near overbought, resistance ahead, weak market, high debt>",
    "<specific risk 2>",
    "<specific risk 3 — omit if fewer than 3 genuine risks>"
  ],
  "suggested_approach": "<one sentence on position sizing or timing — e.g. 'Wait for a close above ₹X before entering' or 'Consider half position given current market weakness'>",
  "bull_case": "<one sentence: what observable conditions in the next 1-2 days would confirm this setup is working>",
  "bear_case": "<one sentence: what observable conditions would warn that this setup is failing>",
  "invalidation": "<specific price level or technical condition at which this setup is no longer valid — e.g. 'A daily close below ₹1,290 would invalidate this setup'>",
  "catalyst": "<one sentence: what specific event or condition would make this trade work>",
  "trade_plan": "<one sentence plain-English entry plan: entry trigger price/condition, what to wait for, target zone, and what invalidates it — e.g. 'Watch for breakout above ₹243 with volume; entry on breakout close, first target ₹280, invalid if closes below ₹215'>"
}}"""

    try:
        # Use streaming to avoid HTTP timeout on long responses.
        # get_final_message() collects all chunks into a complete Message.
        # 1100 tokens was too tight for the eleven fields this prompt asks for
        # (summary, 3 strengths, 3 risks, approach, bull, bear, invalidation,
        # catalyst, trade plan). Responses were being truncated mid-JSON, which
        # surfaced as "no valid JSON object found" — a message that pointed at
        # the parser rather than the real cause.
        with client.messages.stream(
            model="claude-sonnet-4-6",
            max_tokens=2000,
            timeout=timeout,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            message = stream.get_final_message()
        text = message.content[0].text.strip()

        # Extract the first complete JSON object from the response.
        # Claude occasionally wraps JSON in markdown fences or adds a leading sentence.
        data = _extract_json(text)
        if data is None:
            # Report WHY it failed. Truncation and malformed output need
            # different fixes, and the old message could not tell them apart.
            if getattr(message, "stop_reason", None) == "max_tokens":
                print(f"[AI] {symbol}: response truncated at max_tokens "
                      f"({message.usage.output_tokens} out) — raise max_tokens")
            else:
                print(f"[AI] JSON parse failed for {symbol} "
                      f"(stop={getattr(message, 'stop_reason', '?')}, "
                      f"{len(text)} chars): {text[:160]!r}")
            return {**_null_result(), **pos}

        sq = data.get("setup_quality") or data.get("confidence") or 50
        try:
            sq = int(float(sq))
        except (TypeError, ValueError):
            sq = 50
        return {
            "ai_verdict":           data.get("verdict"),
            "ai_confidence":        sq,   # kept for backward compat (drives color bars)
            "ai_summary":           data.get("summary"),
            "ai_strengths":         data.get("strengths") or [],
            "ai_risk_factors":      data.get("risk_factors") or [],
            "ai_suggested_approach":data.get("suggested_approach"),
            "ai_bull_case":         data.get("bull_case"),
            "ai_bear_case":         data.get("bear_case"),
            "ai_invalidation":      data.get("invalidation"),
            "ai_catalyst":          data.get("catalyst"),
            "ai_trade_plan":        data.get("trade_plan"),
            **pos,
        }
    except Exception as e:
        print(f"[AI] Analysis failed for {symbol}: {e}")
        return {**_null_result(), **pos}
