"""Cash-accounting kernel: raw shares in strict mode, normalized units in diagnostics."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Limits:
    capital: float = 100000.
    max_positions: int = 20
    max_weight: float = .05
    max_sector_weight: float = .25
    participation: float = .01
    side_cost: float = .002791

    def __post_init__(self):
        if not (self.capital > 0 and self.max_positions > 0 and 0 < self.max_weight <= 1
                and 0 < self.max_sector_weight <= 1 and 0 < self.participation <= 1 and 0 <= self.side_cost < 1):
            raise ValueError('Invalid portfolio limits')


class Ledger:
    def __init__(self, limits=Limits(), diagnostic=False, provenance=None):
        if not diagnostic and (provenance is None or provenance.blockers()):
            raise ValueError('Strict portfolio requires reviewed provenance; use explicit diagnostic mode for legacy data')
        self.limits = limits
        self.diagnostic = diagnostic
        self.cash = limits.capital
        self.positions = {}
        self.last_marks = {}
        self.events = []
        self.fees = 0.
        self.traded_notional = 0.
        self.unresolved_actions = set()
        self.action_ids = set()

    def record(self, date, kind, **fields):
        self.events.append({'date':str(date), 'kind':kind, **fields})

    def nav(self, prices):
        missing = [s for s in self.positions if not math.isfinite(prices.get(s, float('nan'))) or prices[s] <= 0]
        if missing or self.unresolved_actions:
            return None
        return self.cash + sum(q * prices[s] for s,q in self.positions.items())

    def action(self, date, event):
        """Supplied action amounts apply before trading on the effective date.

        Dividend event date must be the payable date; caller must capture entitlement
        at ex-date and provide entitled_quantity (never infer it from payment-day holdings).
        Demergers/rights/mergers remain unresolved until an explicit reviewed lifecycle exists.
        """
        identity = event['id']
        if identity in self.action_ids:
            raise ValueError('Duplicate corporate action')
        s = event['symbol']; kind = event['type']
        if kind == 'split':
            ratio = float(event['ratio'])
            if not math.isfinite(ratio) or ratio <= 0:
                raise ValueError('Invalid split ratio')
            if s in self.positions:
                self.positions[s] *= ratio
                if not self.diagnostic and abs(self.positions[s]-round(self.positions[s]))>1e-9:
                    self.unresolved_actions.add(s)  # fractional entitlement/cash-in-lieu needs evidence
                if s in self.last_marks: self.last_marks[s] /= ratio
        elif kind == 'dividend_payment':
            qty = float(event['entitled_quantity']); amount = float(event['cash_per_share'])
            if not math.isfinite(qty + amount) or qty < 0 or amount < 0:
                raise ValueError('Invalid dividend entitlement')
            self.cash += qty * amount
        else:
            if s in self.positions:
                self.unresolved_actions.add(s)
        self.action_ids.add(identity)
        self.record(date, 'corporate_action', action=event)

    def rebalance(self, date, targets, prices, turnover, tradable, sectors=None):
        """Targets are budgets frozen at PRIOR close, never ranked using this open.

        Sell reductions first. Existing holdings count against all limits. Missing,
        flat or unreviewed execution bars are not assumed filled. Unfilled buys expire.
        """
        sectors = sectors or {}
        nav = self.nav(prices)
        if nav is None:
            self.record(date, 'rebalance_blocked', reason='unresolved_valuation_or_action')
            return
        if any(not math.isfinite(v) or v < 0 for v in targets.values()):
            raise ValueError('Invalid target budget')
        cap = self.limits.max_weight * nav
        desired = {s:min(v,cap) for s,v in targets.items()}
        # Respect the given ranking order for buys, with deterministic sell ordering.
        sequence = [(s, False) for s in sorted(self.positions)] + [(s,True) for s in desired]
        for s,buy_phase in sequence:
            px = prices.get(s, float('nan'))
            if not tradable.get(s,False) or not math.isfinite(px) or px <= 0:
                if s in self.positions or desired.get(s,0)>0:
                    self.record(date,'order_skipped',symbol=s,reason='execution_unverified')
                continue
            held = self.positions.get(s,0.)
            target = desired.get(s,0.) / px
            if not self.diagnostic: target = math.floor(target)
            delta = target-held
            if (delta <= 1e-10 and buy_phase) or (delta >= -1e-10 and not buy_phase): continue
            liquidity = turnover.get(s,0.)
            if not math.isfinite(liquidity) or liquidity <= 0:
                self.record(date,'order_skipped',symbol=s,reason='liquidity_unknown');continue
            quantity = min(abs(delta), self.limits.participation*liquidity/px)
            if buy_phase:
                if s not in self.positions and len(self.positions)>=self.limits.max_positions:
                    self.record(date,'order_skipped',symbol=s,reason='position_limit');continue
                if not self.diagnostic:
                    sector = sectors.get(s)
                    if sector is None:
                        self.record(date,'order_skipped',symbol=s,reason='sector_unknown');continue
                    used = sum(q*prices[k] for k,q in self.positions.items() if sectors.get(k)==sector)
                    quantity = min(quantity,max(0,self.limits.max_sector_weight*nav-used)/px)
                quantity = min(quantity,self.cash/(px*(1+self.limits.side_cost)))
            if not self.diagnostic: quantity=math.floor(quantity)
            if quantity<=1e-10: continue
            notional = quantity*px; fee=notional*self.limits.side_cost
            if buy_phase:
                self.cash -= notional+fee; self.positions[s]=held+quantity
            else:
                self.cash += notional-fee; self.positions[s]=held-quantity
                if self.positions[s]<1e-9: del self.positions[s]
            self.fees+=fee;self.traded_notional+=notional
            self.record(date,'buy' if buy_phase else 'sell',symbol=s,quantity=quantity,price=px,fee=fee)
        if self.cash < -1e-7: raise AssertionError('Negative cash')
        if len(self.positions)>self.limits.max_positions: raise AssertionError('Position limit breached')
