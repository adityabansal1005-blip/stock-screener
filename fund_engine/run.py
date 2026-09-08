"""Run a frozen diagnostic batch. Default command is audit-only, never trading."""
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from .data import load_legacy, sha256, valid_bars
from .ledger import Ledger, Limits
from .strategies import NAMES, precompute, rank


def simulate(calendar, frames, reference, features, name, limits):
    ledger=Ledger(limits,diagnostic=True)
    rows=[]; pending=None; previous_close={}; stop=None
    trading=calendar[calendar.year>=2016]
    trend=reference>reference.rolling(200).mean()
    offsets={day:i for i,day in enumerate(calendar)}
    opens={s:d.open.to_numpy() for s,d in frames.items()}
    marks={s:d.close.where(features[s]['valid']).to_numpy() for s,d in frames.items()}
    execution={s:features[s].tradable.to_numpy() for s in frames}
    for i,date in enumerate(trading):
        offset=offsets[date]
        prices={s:float(v[offset]) for s,v in opens.items()}
        closes={s:float(v[offset]) for s,v in marks.items()}
        # Discontinuous held prices cannot be assumed an economic gain/loss.
        jumps=[s for s in ledger.positions if s in previous_close and np.isfinite(closes[s])
               and abs(closes[s]/previous_close[s]-1)>.4]
        if jumps:
            ledger.record(date,'run_stopped',reason='held_price_discontinuity',symbols=jumps)
            stop='held_price_discontinuity';break
        if pending is not None:
            targets,turnover,signal_date=pending
            tradable={s:bool(v[offset]) for s,v in execution.items()}
            ledger.record(date,'execute_intent',signal_date=str(signal_date))
            ledger.rebalance(date,targets,prices,turnover,tradable)
            pending=None
        nav=ledger.nav(closes)
        if nav is None:
            ledger.record(date,'run_stopped',reason='unresolved_held_valuation',symbols=[s for s in ledger.positions if not math.isfinite(closes.get(s,float('nan'))) or closes[s]<=0])
            stop='unresolved_held_valuation';break
        rows.append({'date':str(date.date()),'nav':nav,'cash':ledger.cash,'positions':len(ledger.positions),
                     'max_weight':max([q*closes[s]/nav for s,q in ledger.positions.items()] or [0.])})
        previous_close={s:closes[s] for s in ledger.positions}
        # A confirmed exchange calendar supplies month-end, not observed stock outcomes.
        if i+1<len(trading) and trading[i+1].month!=date.month:
            picks=rank(features,date,name,bool(trend.loc[date]))[:limits.max_positions]
            targets={s:nav*limits.max_weight for s in picks}
            turnover={s:float(features[s].loc[date,'turnover']) for s in set(picks)|set(ledger.positions)}
            pending=(targets,turnover,date)
            ledger.record(date,'rebalance_intent',targets=targets)
    navs=pd.DataFrame(rows)
    complete=stop is None and len(navs)==len(trading)
    metrics={'complete':complete,'stop_reason':stop,'valued_sessions':len(navs),
             'last_valued_date':None if navs.empty else navs.iloc[-1].date,
             'fees':ledger.fees,'traded_notional':ledger.traded_notional,
             'orders_filled_under_model':sum(x['kind'] in ('buy','sell') for x in ledger.events),
             'skipped_orders':sum(x['kind']=='order_skipped' for x in ledger.events),
             'max_observed_position_weight':float(navs.max_weight.max()) if len(navs) else None,
             'ending_positions':ledger.positions,'mode':'legacy_normalized_units_diagnostic',
             'total_return_pct':None,'cagr_pct':None,'max_drawdown_pct':None,'liquidation_return_pct':None}
    if complete:
        values=np.r_[limits.capital,navs.nav.to_numpy()]
        years=(trading[-1]-trading[0]).days/365.25
        end=float(values[-1]); cash=float(navs.iloc[-1].cash)
        metrics.update(total_return_pct=(end/limits.capital-1)*100,
                       cagr_pct=((end/limits.capital)**(1/years)-1)*100,
                       max_drawdown_pct=float((values/np.maximum.accumulate(values)-1).min()*100),
                       liquidation_return_pct=((end-(end-cash)*limits.side_cost)/limits.capital-1)*100)
        year_ends=navs.assign(year=pd.to_datetime(navs.date).dt.year).groupby('year').nav.last()
        prior=limits.capital; yearly={}
        for year,end_nav in year_ends.items():
            yearly[str(year)]=(end_nav/prior-1)*100;prior=end_nav
        metrics['year_returns_pct']=yearly
    return metrics,navs,ledger.events


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--diagnostic',action='store_true',help='Explicitly permit unverified normalized-unit research, not raw-share backtesting')
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1]
    manifest={'created_utc':datetime.now(timezone.utc).isoformat(),
              'hashes':{str(f):sha256(f) for f in [args.archive,root/'docs/FUND_PROTOCOL.md',*sorted((root/'fund_engine').glob('*.py'))]},
              'diagnostic':args.diagnostic,'limits':asdict(Limits()),'end':'2020-12-31'}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    calendar,frames,reference,provenance=load_legacy(args.archive)
    audit={'provenance':asdict(provenance),'blockers':provenance.blockers(),
           'symbols':len(frames),'rows':sum(int(d.close.notna().sum()) for d in frames.values()),
           'invalid_rows':sum(int((~valid_bars(d)&d.close.notna()).sum()) for d in frames.values())}
    results={'audit':audit,'mode':'audit_only','strategies':{},'promotion':'not_approved'}
    if args.diagnostic:
        results['mode']='legacy_normalized_units_diagnostic'
        print('Precomputing shared features once',flush=True)
        features=precompute(frames)
        for mult in (1,2):
            for name in NAMES:
                key=f'{name}_cost{mult}'
                print('Simulating',key,flush=True)
                metrics,navs,events=simulate(calendar,frames,reference,features,name,replace(Limits(),side_cost=Limits().side_cost*mult))
                results['strategies'][key]=metrics
                navs.to_csv(args.output/f'{key}_nav.csv',index=False)
                with (args.output/f'{key}_ledger.jsonl').open('x') as f:
                    for event in events:f.write(json.dumps(event,allow_nan=False)+'\n')
                print(key,metrics['complete'],metrics['stop_reason'],metrics['total_return_pct'],flush=True)
        complete=[v for v in results['strategies'].values() if v['complete']]
        results['complete_runs']=len(complete)
    (args.output/'results.json').write_text(json.dumps(results,indent=2,allow_nan=False))
    lines=['# Fund engine research report','',f"Mode: {results['mode']}. Development 2016-2020 only.",
           '', 'Legacy prices are not certified raw shares or total returns. Sector/membership/action data are incomplete. These are finite-capital diagnostic portfolios, not validated fund performance.',
           '', '| Arm | Complete | Total return % | CAGR % | Max drawdown % | Last valued | Stop reason |',
           '|---|---|---:|---:|---:|---|---|']
    def fmt(v):return 'unavailable' if v is None else f'{v:.2f}'
    for key,m in results['strategies'].items():
        lines.append(f"| {key} | {m['complete']} | {fmt(m['total_return_pct'])} | {fmt(m['cagr_pct'])} | {fmt(m['max_drawdown_pct'])} | {m['last_valued_date']} | {m['stop_reason'] or '-'} |")
    lines+=['','No full-period returns are reported for stopped portfolios. Saved ledgers expose the exact event that needs reconciliation.', '', 'Strict-mode blockers: '+', '.join(audit['blockers']), '',
            'No later-year data loaded, no orders submitted, no strategy promoted. All trials and cost stresses are retained.']
    (args.output/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'output':str(args.output),'complete_runs':results.get('complete_runs',0),'blockers':audit['blockers']},indent=2))


if __name__=='__main__':main()
