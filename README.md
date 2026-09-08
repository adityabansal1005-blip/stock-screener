# NSE Stock Screener

A local Windows dashboard for screening Indian stocks and reviewing research. **Research software: its scores have not demonstrated a reliable trading edge.** A high score is not a buy recommendation or a win probability.

## Start here — no coding needed

1. On this GitHub page choose **Code → Download ZIP**. Right-click the ZIP → **Extract All**. Open the extracted folder. Do not run files from inside the ZIP.
2. Install **Python 3.12, 64-bit**, from [python.org](https://www.python.org/downloads/windows/). Include the **Python Launcher** during installation. This release's simple launchers target Windows.
3. Double-click **SETUP.cmd** once. It creates a private Python environment and downloads the required libraries. Keep the window open until it says setup finished.
4. Optional: double-click **EDIT_KEYS.cmd**, enter **your own** Groww API credentials in the local `.env`, and save. Never share that file. No keys are required just to open the dashboard; successful live scanning depends on provider access.
5. Double-click **START.cmd**. A browser opens at **http://localhost:8000**. Keep the terminal window open while using the app. Closing it stops the app.
6. Choose **Nifty 50** and click **Rescan** first. Once that works, try **All NSE**. The first large scan can take much longer while data downloads.

If anything fails, double-click **CHECK.cmd**. It checks installed libraries and whether keys are present without displaying them or contacting providers.

## What you need

| Requirement | Needed when? |
|---|---|
| Windows 10/11, Python 3.12 64-bit, internet, free disk space for libraries/cache | Initial setup and operation |
| Your own Groww API access and credentials | Recommended market-data connection; account entitlements and authentication must work |
| Anthropic API key | Optional existing Claude analysis feature |
| Google API key + separate research environment | Optional TradingAgents reports; see [optional AI setup](docs/OPTIONAL_AI.md) |
| Telegram bot credentials | Optional alerts; blank by default |
| Git | Not needed for ZIP download/basic setup; needed for optional pinned TradingAgents installation |

AI and broker APIs may require separate access, quotas or billing. Installing the app does not provide those services. A coding assistant can help operate and change the project; it does not supply the API keys.

## Everyday flow

### Fund research

Open **Fund Research** from the dashboard to inspect local portfolio tests and data readiness. **RESEARCH.cmd** audits your local historical archive; it does not download data, place trades, or run a new strategy search. The new engine separates source provenance, investment rules, cash/holdings accounting and validation. Read [the frozen fund protocol](docs/FUND_PROTOCOL.md) before running research.

For a developer-run legacy diagnostic only:

```powershell
.\.venv\Scripts\python.exe -m fund_engine.run --archive strategy_data.db --output fund_runs\my_first_diagnostic --diagnostic
```

The output folder must be new. This uses 2016–2020 only, with four fixed monthly portfolio arms and doubled-cost checks. Legacy adjusted prices produce normalized-unit diagnostics, not verified real-share backtests. Missing held valuations stop a run. The source archive is never modified.

**START → choose universe → Rescan → inspect the scan date → review eligible candidates → inspect financial risks → record a research decision.**

- Swing scores use completed daily sessions. During market hours, live quotes can move while the swing score remains based on the previous completed session.
- Eligibility checks reject failed/unknown liquidity, mismatched session dates and recorded material financial risks. Unreviewed finances are labelled pending, not cleared.
- Candidates remain **research only**. There is no validated strategy authorizing real-money trades.
- Score version `measurement-v2-no-rr` excludes points for constructed risk/reward targets. Its scale differs from previous scores; do not compare across versions as if they were identical.
- Scans, databases and reports accumulate locally. **This download does not include the owner's historical database, account, watchlist or keys.** A fresh installation starts without those records. Old research results cannot be reproduced without the corresponding frozen data.
- Background scheduled scans are disabled in the example configuration. Start with manual rescans.

## Ask Claude or Codex for help

Open this extracted folder in your coding assistant and paste:

> Read AGENTS.md and README.md. Help me run this project on Windows. Check Python 3.12, create the local environment, install requirements, and run scripts/doctor.py. Do not read or print API-key values. Tell me where to enter keys myself. Run the offline tests before changing code. Start the dashboard locally, help me run a Nifty 50 scan, and explain any data or eligibility warnings. Do not place trades or enable paid AI calls or alerts without my instruction.

[Application flow](docs/FLOW.md) · [Troubleshooting](docs/TROUBLESHOOTING.md) · [Research protocol](docs/RESEARCH_PROTOCOL.md)
