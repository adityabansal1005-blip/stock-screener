# Troubleshooting

The basic installation intentionally excludes the optional TrueData WebSocket SDK because its pinned compression dependency conflicts with the old requirements. Leave TrueData credentials blank for the supported Groww/basic flow. The legacy client code remains available for a separately tested integration.

| Symptom | What to do |
|---|---|
| SETUP says Python3.12 is missing | Install64-bit Python3.12 with its launcher, close the window and rerun SETUP. |
| Package installation fails | Check internet and free disk space. Keep the error window open; ask your assistant to inspect it. Do not paste keys. |
| Browser did not open | Open http://localhost:8000 manually. Keep START's terminal open. |
| Port8000 already used | Close the older app, or set PORT=8001 in `.env`, restart, and use localhost:8001. |
| Groww not connected | Enter your own credentials locally; verify access/entitlements with Groww. CHECK only tests presence, not authentication. |
| No candidates after an update | Rescan to compute eligibility under the new policy. All rows remain inspectable. Data freshness, liquidity and financial blocks can reject candidates. |
| Score stays the same during trading | Swing scores use completed sessions. Read the scan date and separate live quote. |
| AI not available | Optional feature. Check the relevant developer API key, quota/model access and optional environment. Scanning does not require AI. |
| Missing historical data | The repository excludes private/provider archives. Download through your own permitted provider; do not expect the author's past scans to be bundled. |

To update a ZIP installation: stop the app, extract the new ZIP into a separate folder, copy your `.env` locally if desired, run SETUP and START there. Keep your old folder until the new installation works. Never upload the `.env` or databases when asking for help.
