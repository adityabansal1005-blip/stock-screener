"""Isolated research process; receives no brokerage credentials or execution tools."""
import copy
import json
import os
from pathlib import Path
import re
import sys


def main():
    root = Path(__file__).resolve().parent
    job_id = sys.argv[1]
    if not re.fullmatch(r'[0-9a-f]{32}', job_id):
        raise ValueError('Invalid job identifier')
    folder = root / 'research_reports' / job_id
    job = json.loads((folder / 'job.json').read_text(encoding='utf-8'))
    try:
        from tradingagents.default_config import DEFAULT_CONFIG
        from tradingagents.graph.trading_graph import TradingAgentsGraph
        from research_price_adapter import install
        price_source = install(folder)
        cfg = copy.deepcopy(DEFAULT_CONFIG)
        cfg.update(llm_provider='google', backend_url=None,
                   deep_think_llm=job['model'], quick_think_llm=job['model'],
                   max_debate_rounds=1, max_risk_discuss_rounds=1,
                   max_recur_limit=70, max_tokens=4096, llm_max_retries=1,
                   google_thinking_level='minimal', output_language='English',
                   checkpoint_enabled=False, memory_log_path=None,
                   results_dir=str(folder / 'upstream'),
                   data_cache_dir=str(root / 'research_reports' / 'data_cache'),
                   global_news_queries=['India RBI monetary policy inflation rupee',
                                        'India NSE corporate earnings economy'])
        # Exact vendor selection, no undeclared paid market-data providers.
        cfg['data_vendors'].update(core_stock_apis='local_snapshot', technical_indicators='local_snapshot',
                                   fundamental_data='yfinance', news_data='yfinance')
        graph = TradingAgentsGraph(selected_analysts=['market', 'fundamentals', 'news'],
                                   config=cfg, debug=False)
        state, decision = graph.propagate(job['ticker'], job['analysis_date'])
        fields = ['market_report', 'fundamentals_report', 'news_report',
                  'investment_debate_state', 'trader_investment_plan',
                  'risk_debate_state', 'final_trade_decision']
        reports = {k: state.get(k, '') for k in fields}
        result = {'status': 'completed', 'decision': decision, 'reports': reports, 'price_source': price_source,
                  'limitations': 'AI research, not validated edge. Sources and quotes require checking. '
                  'Analysis date is the request date; prices may be from an earlier session. '
                  'This report does not change scanner scores or place trades.'}
        # Store a readable export without HTML rendering or executing model text.
        text = '# TradingAgents research: ' + job['ticker'] + '\n\n' + job['created_at'] + '\n\n' + json.dumps(price_source, indent=2) + '\n\n'
        for name, value in reports.items():
            text += '## ' + name.replace('_', ' ').title() + '\n\n'
            text += (value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)) + '\n\n'
        key = os.environ.get('GOOGLE_API_KEY', '')
        if key:
            text = text.replace(key, '[redacted]')
        (folder / 'report.md').write_text(text, encoding='utf-8')
    except Exception as exc:
        message = str(exc).lower()
        if any(s in message for s in ('429', 'quota', 'resource_exhausted')):
            error = 'Google API quota or rate limit reached. Check API billing/quota and retry later.'
        elif any(s in message for s in ('api key', 'api_key', '401', '403', 'permission_denied')):
            error = 'Google rejected the credentials or access. Check the local API key and model permissions.'
        elif any(s in message for s in ('404', 'model not found', 'not supported')):
            error = 'Configured model is unavailable. Set TRADINGAGENTS_MODEL to an available Gemini model.'
        else:
            error = 'Research failed (' + type(exc).__name__ + '). Data/model service may be unavailable; retry later.'
        result = {'status': 'failed', 'error': error}
    payload = json.dumps(result, ensure_ascii=False, default=str)
    key = os.environ.get('GOOGLE_API_KEY', '')
    if key:
        payload = payload.replace(key, '[redacted]')
    (folder / 'result.json').write_text(payload, encoding='utf-8')


if __name__ == '__main__':
    main()
