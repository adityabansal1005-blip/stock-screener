# Optional AI reports

The scanner works without an AI key. Configure these only after a basic Nifty50 scan succeeds.

## Existing Claude analysis

Put your own `ANTHROPIC_API_KEY` in `.env`. `AI_ANALYSE_TOP_N=0` keeps automatic top-stock AI analysis off in the supplied configuration. If you deliberately enable automatic analysis, choose a small positive number and review your API usage. Credentials and model access must be valid for the developer API.

## TradingAgents / Google research

This is a separate optional feature, not a prerequisite for SETUP or START. It uses a second environment to avoid dependency conflicts. Requires Python3.12, Git, an internet connection and your own Google developer API key/quota.

Ask your assistant to run these from the project folder:

```powershell
py -3.12 -m venv .venv-tradingagents
.\.venv-tradingagents\Scripts\python.exe -m pip install "git+https://github.com/TauricResearch/TradingAgents.git@9dee508c44662702281a8dbaad1f7b42179b5ba7"
```

Then add `GOOGLE_API_KEY` to `.env` locally. The example model is `gemini-3.5-flash`; set `TRADINGAGENTS_MODEL` to an available compatible model if your account lacks it. The project checks availability only when a research call runs. Model access and quota can change.

Open a stock's details and select **Research this stock**. This makes billable/provider-quota API requests. There is a process timeout and bounded agent rounds, but no guaranteed rupee spending cap. The adapter currently requires at least220 valid completed daily bars, so recent listings can be rejected. A cached archive is a fallback only if it exists on your machine.

The worker receives the Google credential and a frozen price snapshot; it is not given brokerage credentials. It does not place orders, send alerts or change scores. Outputs are research opinions, not validated predictions. Original provider attribution and dates remain necessary for interpreting a report.

Upstream TradingAgents is Apache-2.0, maintained at https://github.com/TauricResearch/TradingAgents. Its source and dependencies are downloaded by the optional installation and are not bundled in this repository. This optional environment is not installed by SETUP.cmd.
