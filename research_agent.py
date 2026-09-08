"""
AI Research Agent — Claude analyses strategy_lab results and ML model metrics,
produces regime-aware strategy recommendations.
"""
import json
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
        return None


def _null():
    return {
        "overall_verdict":    None,
        "best_strategy":      None,
        "worst_strategy":     None,
        "regime_guidance":    None,
        "overfitting_risks":  [],
        "improvements":       [],
        "edge_sources":       [],
        "raw_text":           None,
    }


def analyse_strategies(lab_results: dict, ml_metrics: dict | None = None) -> dict:
    """
    Send strategy lab output to Claude for research-grade analysis.
    Returns structured insights.
    """
    client = _get_client()
    if client is None:
        return _null()

    strategies = lab_results.get("strategies", {})
    ranked     = lab_results.get("ranked", [])

    # Build compact summary for each strategy
    strat_lines = []
    for name in (ranked or list(strategies.keys())):
        s = strategies.get(name, {})
        if not s.get("valid"):
            strat_lines.append(f"  • {name}: insufficient trades (<{10})")
            continue
        bull = s.get("regime_bull", {})
        bear = s.get("regime_bear", {})
        side = s.get("regime_sideways", {})
        strat_lines.append(
            f"  • {name} | Trades={s['total_trades']} | Symbols={s['symbol_count']} "
            f"| WinRate={s['avg_win_rate']}% | AvgRet={s['avg_return']:+.2f}% "
            f"| Sharpe={s.get('avg_sharpe') or 'N/A'} "
            f"| PF={s.get('avg_profit_factor') or 'N/A'} "
            f"| MaxDD={s.get('avg_max_drawdown', 0):.1f}%\n"
            f"    Regime: Bull WR={bull.get('avg_win_rate') or 'N/A'}% "
            f"| Bear WR={bear.get('avg_win_rate') or 'N/A'}% "
            f"| Sideways WR={side.get('avg_win_rate') or 'N/A'}%"
        )

    ml_section = ""
    if ml_metrics and not ml_metrics.get("error"):
        top_features = ml_metrics.get("top_features", [])[:5]
        tf_text = ", ".join(f"{n}({v:.3f})" for n, v in top_features)
        ml_section = f"""
━━━ ML MODEL METRICS ━━━
  Train accuracy:    {ml_metrics.get('train_acc', 'N/A')}
  Test accuracy:     {ml_metrics.get('test_acc', 'N/A')}
  Validation acc:    {ml_metrics.get('val_acc', 'N/A')}
  Test AUC-ROC:      {ml_metrics.get('test_auc', 'N/A')}
  Overfitting gap:   {ml_metrics.get('overfitting_gap', 'N/A')} (train-test accuracy delta)
  Train/Test/Val N:  {ml_metrics.get('n_train')}/{ml_metrics.get('n_test')}/{ml_metrics.get('n_val')}
  Top features:      {tf_text}
"""

    prompt = f"""You are a quantitative research analyst with 20 years of experience in Indian equity markets.
You are reviewing backtested strategy results for NSE stocks and ML model metrics.
Write for a BEGINNER investor — plain English, define any jargon.

RULES:
- Past backtest performance does NOT guarantee future returns — say so clearly
- Flag overfitting risks explicitly (high train-test accuracy gap, few symbols, short backtest period)
- Regime awareness is critical: a strategy that works in bull markets may destroy capital in bear markets
- Be honest about which strategies have a genuine edge vs. which are likely curve-fitted noise

━━━ STRATEGY BACKTEST RESULTS (10 years, NSE universe) ━━━
{chr(10).join(strat_lines)}
{ml_section}

━━━ TASK ━━━
Analyse these results and provide research-grade insights.

Respond with ONLY valid JSON:
{{
  "overall_verdict": "<one sentence: what does this research tell us about the overall signal quality across strategies>",
  "best_strategy": {{
    "name": "<strategy name>",
    "reason": "<plain English: why this strategy has a genuine edge — be specific about what drives its performance>",
    "best_regime": "<bull | bear | sideways | all>",
    "caution": "<one specific risk or limitation even for the best strategy>"
  }},
  "worst_strategy": {{
    "name": "<strategy name>",
    "reason": "<why this strategy likely lacks edge or is curve-fitted>",
    "salvageable": "<yes/no + one line on whether it could be improved>"
  }},
  "regime_guidance": "<one paragraph: which strategies to use in bull / bear / sideways markets — specific, actionable>",
  "overfitting_risks": [
    "<specific signal that one or more strategies may be overfit — e.g. win rate drops sharply in validation period, bear-regime sample too small>",
    "<second overfitting risk>",
    "<third if applicable>"
  ],
  "edge_sources": [
    "<observable market inefficiency or behavioral pattern that explains why the best strategy works>",
    "<second edge source>"
  ],
  "improvements": [
    "<concrete, implementable improvement suggestion — e.g. 'Add a volume filter to Breakout: require volume > 2x 20d avg instead of 1.5x'>",
    "<second improvement>",
    "<third improvement>"
  ],
  "ml_interpretation": "<if ML metrics provided: plain English interpretation of what the model learned and whether it adds value over rule-based strategies. Otherwise: null>"
}}"""

    try:
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=900,
            timeout=40,
            messages=[{"role": "user", "content": prompt}],
        )
        text = msg.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            patched = text.rstrip().rstrip(",")
            if not patched.endswith("}"):
                patched += '"}'
            try:
                data = json.loads(patched)
            except Exception:
                data = {}

        return {
            "overall_verdict":   data.get("overall_verdict"),
            "best_strategy":     data.get("best_strategy"),
            "worst_strategy":    data.get("worst_strategy"),
            "regime_guidance":   data.get("regime_guidance"),
            "overfitting_risks": data.get("overfitting_risks") or [],
            "improvements":      data.get("improvements") or [],
            "edge_sources":      data.get("edge_sources") or [],
            "ml_interpretation": data.get("ml_interpretation"),
            "raw_text":          text,
        }
    except Exception as e:
        print(f"[ResearchAgent] Analysis failed: {e}")
        return _null()
