# Measurement v2 — frozen before the revised calculations

This revision corrects measurement choices after viewing v1 results. It is retrospective research, not a fresh holdout. Preserve all v1 artifacts and do not tune thresholds to rescue a failed rule.

## Historical event study

Keep the six original entry rules, universe eligibility, development years 2016–2020, validation years 2021–2025, weekly sampling and t+1 open/t+21 close horizon unchanged. Keep the 0.5582 percentage-point cost assumption and original advancement thresholds.

For the fixed-horizon/no-stop experiment, valid entry and exit endpoints permit an observed return even if an interior candle is missing. Mark interior gaps separately. Missing/invalid entry means unknown execution, not a loss. Valid entry with missing/invalid exit means unresolved valuation, not a zero or -100% realized return. Preserve every event and report group-specific missingness. Observed-return statistics are conditional on observable endpoints and may be biased.

Publish separate hypothetical net-return scenarios of -100%, -20%, 0%, +20% and +100% for unresolved exits; these are sensitivity assumptions, not observed outcomes or mathematical upper bounds. Missing entries remain explicitly unknown and are not imputed as filled trades. No strategy receives a deployable-edge declaration while missingness, adjustment provenance or historical universe coverage is unresolved. Do not run later-stage selection merely because a conditional mean improves.

Attempt targeted Groww reconciliation of the first 12 missing symbol/date pairs in chronological order from the existing diagnostic list. Save responses and overlap differences separately. Never silently splice providers or infer that a missing candle is a delisting. Broader repair requires confirmed adjustment compatibility and historical membership evidence.

## One complete prospective research rule

Freeze `pullback_trend` as the first paper hypothesis for simplicity, not because it passed validation. Signal after the completed session: close > SMA50 > SMA200, SMA200 > its value 20 sessions earlier, and three-session return <0. Require 220 consecutive valid exchange sessions, price >= INR10, trailing20 median turnover >= INR1 crore, and no known blocked financial-risk review. Unknown financial review remains labelled unreviewed; this is research, not real-money eligibility.

Initial virtual equity INR100,000; no leverage, no shorting, at most10 positions, at most10% of current equity allocated to each new position. At a signal date choose eligible candidates alphabetically, excluding already-held symbols; unused slots stay cash. Orders are prospective intents for the next exchange-session open, not retroactive fills. Position value also cannot exceed1% of trailing20 median turnover. Integer shares, fees included in the cash budget. Exit at the close of the 21st session after the signal. No target, stop or intraday optimization in this first hypothesis. Missing executable prices leave an intent/valuation unresolved rather than inventing a fill. Suspensions/circuits and corporate actions require explicit treatment. Fixed research round-trip cost0.5582% is a starting assumption, not a broker-accurate fee model.

Future capture starts with sessions completed AFTER this protocol was frozen. Already-seen September8 intraday results and earlier scans are not prospective observations. Keep append-only captures and record every skip. A paper ledger requires confirmed exchange-session calendars and entry/exit quote evidence before fills; capturing a signal does not establish execution.

## Application changes

New score version `measurement-v2-no-rr`: remove the manufactured risk/reward points and their15-point denominator capacity (79 ->64); do not optimize other weights. This is a changed research scale and is not directly comparable to old scores. Existing scores remain unchanged in historical artifacts. Swing computations exclude the current session before16:15 IST, including weekly and Nifty inputs. Live quotes remain separate.

Research-candidate filters require positive liquidity verification, a completed bar matching the reference session and no recorded material financial-risk block. Unreviewed finances are not described as cleared. Every candidate remains non-actionable until a strategy is validated. These gates reduce misleading presentation; they do not create predictive edge.
