# Instructions for coding assistants

Read README.md first. Help nontechnical users one clear step at a time. Use Windows Python3.12 in `.venv`; do not require the author's computer paths. API keys belong in `.env`, never in chat, logs, commits or screenshots. Inspect key presence only. Never copy private data into the repository.

Primary flow: START.cmd -> main.py -> scanner.py -> signals.py / technical_enhanced.py -> screening_policy.py -> templates/index.html. Persistence lives in db.py and data_manager.py. All generated databases/caches are local and ignored.

Run `python scripts/doctor.py`, `python audit_tests.py`, and `python smoke_app.py` in the project environment. The smoke test blocks external network. New behavior needs meaningful tests, especially entry/exit timing, partial sessions, missing-data treatment and financial-risk gates. Do not claim installation or market connectivity succeeded unless tested.

Preserve completed-session swing scoring. Keep live quotes separate. Score version measurement-v2-no-rr removes modelled risk/reward points and denominator capacity. No probability claims. `screen_eligible` means eligible for research, not approved to trade; `actionable` remains false. Known financial risks require documented human review to clear. Unknown is not passed.

Read docs/RESEARCH_PROTOCOL.md before research changes. Do not impute a -100% loss for an interior gap in a fixed-endpoint, no-stop test. Keep unresolved entry/exit states explicit. Never silently splice price-adjustment scales. Record all trials and preserve frozen artifacts. Historical periods already used in tuning are not pristine holdouts.

Do not place orders, send alerts, enable schedules, or make paid AI calls without the user's instruction. TradingAgents is optional and runs in `.venv-tradingagents`, never the main environment. Do not install the whole upstream repository into this source tree or commit its environment.

Before publishing: inspect the staged file list and run scripts/check_release.py. Do not commit `.env`, databases, pickles, raw provider datasets, generated reports, account data or local environments. Never force-push or discard someone else's changes to make publishing easier.
