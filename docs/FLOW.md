# How the system works

1. `main.py` serves a local FastAPI dashboard. The supplied configuration starts manually and binds to localhost.
2. A Rescan calls `scanner.scan_universe`. Stock lists come from the universe modules. Data comes from the local archive and enabled providers, with fallbacks; source labels matter.
3. `screening_policy.completed_history` removes unfinished daily bars before swing indicators and weekly aggregates. The daily reference index follows the same cutoff.
4. `signals.py` and `technical_enhanced.py` calculate indicators and an explicitly versioned research score. The new denominator is64 when all components are available. Risk/reward levels remain diagnostics and contribute no score points.
5. Fundamentals and market context are attached. `screening_policy.assess` checks liquidity, bar date and known financial risks before candidate filters/ranking. Financial review can still be unreviewed. Passing these checks is not trading approval.
6. `templates/index.html` shows results, live quotes, dates and research/eligibility labels. A cached legacy scan needs a new eligibility evaluation before it enters candidate filters.
7. Local SQLite files store scans/history. These are created/downloaded on the user's machine and excluded from GitHub.
8. Optional Claude and TradingAgents analysis are separate from score calculation. Reports do not prove edge. TradingAgents obtains a validated price snapshot and sends only its AI-provider credential into its isolated worker.

## Research sequence

Data integrity -> frozen entry/exit rules -> development -> chronological validation -> costs/capacity stress tests -> prospective paper records. Stages can stop with no surviving strategy. Definitions and unresolved issues are in RESEARCH_PROTOCOL.md.
