# Bull/bear review and market regimes

The existing optional TradingAgents integration supplies the multi-agent analysis workflow. It is not a substitute for a profitable strategy. Google/Claude API quotas and costs still apply; the structural rebuild itself does not start paid debates.

Two separate functions:

1. Market exposure: the frozen momentum_trend portfolio targets cash at a monthly decision when Nifty is below its200-session average. This is an explicit testable rule, not an AI guess about a bull or bear market. It can miss rebounds and does not guarantee a smaller drawdown.
2. Company review: a bullish analyst and a bearish analyst use the same evidence pack. Each claim must cite evidence and state what would invalidate it. A reviewer records unresolved disagreements. They cannot override unknown data, audited financial risks, position limits, or failed strategy validation.

The fund_engine.review contract and /api/fund-research/review endpoint validate memo completeness and availability timestamps. They do not call an LLM, verify the truth of a citation, assign a win probability, or authorize an order. Existing free-form AI reports are not automatically converted to this contract; an adapter must supply the cited evidence structure first.

Input memo keys: evidence (id, source_url, available_at with timezone), bull and bear (lists of claim, evidence_ids, invalidation_condition). The request also requires asof with timezone. Future or unsupported evidence fails the structural check. Human source verification is always explicitly pending. A completed memo remains research-only.
