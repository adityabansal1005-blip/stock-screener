# Fund engine v1: frozen research specification

This structural revision follows earlier results. It is exploratory, not a pristine holdout. No automatic promotion to live trading exists. No broker orders or alerts are part of this module.

## Architecture

Source-specific immutable data snapshots -> point-in-time eligibility -> separately named strategies -> prior-close budget intents -> next-open execution -> cash/holdings/fees ledger -> complete daily valuation -> risk and benchmark report.

Strict mode requires reviewed raw execution prices, raw volume, corporate actions, signal adjustments, historical membership and reference calendar. Legacy data cannot pass by guessing its provider or basis. Public-availability timestamps, not fiscal period end, control financial data access. Splits preserve shareholder units and cost-neutral wealth; dividend cash requires an explicit ex-date entitlement and payment date. Unsupported corporate actions quarantine valuation. Missing marks prevent performance publication rather than implying a flat or zero-price holding.

## Frozen first batch

2015 warm-up, 2016-2020 development only. Do not load later years. Initial capital INR100,000; monthly signal at last archived Nifty session of month; rebalance at NEXT session open. Twenty positions maximum, 5% NAV maximum target per name, no leverage or shorting, cash earns zero. Targets sized from signal-close NAV; sells first and only actual proceeds can fund purchases. No replacement ranking at the execution open. Unfilled buys expire; stranded holdings remain and consume limits. One position per symbol; monthly rebalance existing positions rather than count repeated weekly entries as separate trades. Participation capped at1% of prior20 median turnover per order. Flat OHLC bars are conservatively execution-unverified in this batch for both buys and sells; that does not claim all such orders were impossible.

Require253 consecutive positive-volume, coherent historical OHLC bars, close>=10 and prior20 median turnover>=INR1crore. These legacy price/turnover thresholds inherit uncertain adjustment basis. Strict real-share simulation requires proper raw price/volume and reviewed signals. Unknown sector data blocks new strict-mode buys; 25% sector limit applies to strict new allocations. Weights can drift between rebalances and unsuccessful sells can leave breaches; report actual maximum weights rather than claim continuous compliance.

Four arms, no tuning:

- `stable_control`: top20 eligible symbols by deterministic SHA256(symbol) ordering; fixed without returns. Same cash/cost/position constraints. One control draw is not a population baseline or a benchmark index.
- `momentum`: top20 by mean of (close[t-21]/close[t-126]-1) and (close[t-21]/close[t-252]-1), divided by252-session annualized daily-return volatility. Deliberately simple, excludes latest month. No positive-score floor. This is not an exact Nifty200 Momentum30 reproduction.
- `momentum_trend`: same ranking, but targets cash when Nifty close <=200-session mean. Monthly decision, not an intraday stop.
- `year_high`: top20 by close/previous252-session high, requiring ratio>=.95. Fewer candidates leaves cash. This changes prior event study to a deployable-style allocation hypothesis; not confirmation of its earlier event result.

Base cost0.2791% each buy/sell notional, including fees/slippage as a single research assumption. Repeat fixed arms at twice costs; count sensitivity separately, not as new tuned variants. No independent transaction-tax accuracy claimed. Individual income tax excluded. End NAV is marked holdings, not forced liquidation; reports deduct estimated one-side exit cost for a separate liquidation estimate when fully valued.

Legacy diagnostic mode uses fractional normalized units because the archive is adjusted/unknown. It demonstrates finite-capital allocation and turnover mechanics; NOT real-share fills or verified total returns. Sector constraints cannot be claimed with absent historical sectors. Report all integrity blockers. Stop the run on first unresolved held valuation or obvious >40% close jump, preserving the incomplete ledger and prefix results; never quietly discard the position. No full-period performance inferred from a truncated run.

Output: daily NAV/cash/holdings/max weight, append-only-in-run order events, fees/turnover, total return/CAGR/max drawdown only for a complete simulation, yearly returns, control-relative performance and twice-cost results. Nifty close is a PRICE-index context only, not TRI and not an investable after-fee benchmark. No alpha assertion. Statistical evaluation and further period testing require data integrity and a suitable total-return benchmark first.

Advancement: complete full-period portfolio with no unresolved holdings; positive liquidation-adjusted return at doubled costs; lower drawdown or higher annual return than same-constraint control to merit further investigation, not automatic promotion; then benchmark and dependence-aware uncertainty, historical-universe reconciliation and untouched future paper observation. No threshold relaxation. Record all arms, failures and source hashes.

Sources motivating hypotheses: https://www.niftyindices.com/indices/equity/strategy-indices/nifty200-momentum-30 and https://www.aqr.com/Insights/Research/Working-Paper/Trading-Costs-of-Asset-Pricing-Anomalies . They do not establish profitability of these implementations.

## Implementation correction after run001, before run002

Run001 stopped momentum/year-high portfolios on machine-rounding differences between adjusted high and close (~1e-14). Add OHLC coherence tolerance abs(price)*1e-10+1e-8; it is far below an exchange tick and does not excuse material errors. Preserve run001. This is a documented validity-check correction, not strategy-threshold optimization. It can change historical eligibility as well as held-price validation. Rerun all arms/costs together. Other legacy studies have not been silently rewritten.

## Explicit diagnostic data repair before run003

Run002 found a missing PGHL candle on2018-06-20. A Groww fetch supplied it; all OHLCV fields matched exactly on five surrounding shared sessions. Create a separate copy of the archive and insert only that missing record, retaining the response and repair manifest. Original archive remains unchanged. This local overlap reconciliation does not certify global provenance. Rerun every arm and both cost assumptions without retuning. CGPOWER's2016 demerger discontinuity is not replaced with a fabricated return.
