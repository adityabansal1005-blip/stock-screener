"""Explicit provenance and point-in-time data contracts."""
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json
import sqlite3
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Provenance:
    source: str
    price_basis: str
    volume_basis: str
    calendar_verified: bool = False
    membership_verified: bool = False
    actions_verified: bool = False
    adjustment_verified: bool = False
    review_reference: str = ''

    def blockers(self):
        checks = {'execution_prices_not_raw': self.price_basis == 'raw',
                  'volume_basis_unverified': self.volume_basis == 'raw',
                  'calendar_unverified': self.calendar_verified,
                  'historic_membership_unverified': self.membership_verified,
                  'corporate_actions_unverified': self.actions_verified,
                  'signal_adjustments_unverified': self.adjustment_verified,
                  'review_evidence_missing': bool(self.review_reference)}
        return [name for name, ok in checks.items() if not ok]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def valid_bars(d):
    cols = ['open', 'high', 'low', 'close', 'volume']
    x = d[cols]
    # Adjusted float series can differ by machine rounding at the high/low.
    # This tolerance is far smaller than a market tick; never repair material OHLC errors.
    eps = x[['open','high','low','close']].abs().max(axis=1)*1e-10 + 1e-8
    return (np.isfinite(x).all(axis=1) & (x[['open','high','low','close']] > 0).all(axis=1)
            & (x.high + eps >= x[['open','low','close']].max(axis=1))
            & (x.low - eps <= x[['open','high','close']].min(axis=1)) & (x.volume > 0))


def load_legacy(path, end='2020-12-31'):
    """Explicitly unverified. Never silently upgrade legacy candles to raw prices."""
    with sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True) as con:
        d = pd.read_sql_query('SELECT symbol,date,open,high,low,close,volume FROM ohlcv WHERE date<=? ORDER BY symbol,date', con, params=[end])
    d.date = pd.to_datetime(d.date)
    if d.duplicated(['symbol','date']).any():
        raise ValueError('Duplicate symbol dates')
    calendar = pd.DatetimeIndex(d.loc[d.symbol == '^NSEI','date'])
    if len(calendar) < 260 or not calendar.is_monotonic_increasing:
        raise ValueError('Invalid reference calendar')
    frames = {s:g.set_index('date').drop(columns='symbol').reindex(calendar)
              for s,g in d[d.symbol.str.endswith('.NS')].groupby('symbol', sort=True)}
    reference = d[d.symbol == '^NSEI'].set_index('date').close.reindex(calendar)
    return calendar, frames, reference, Provenance('legacy_multi_provider', 'unknown', 'unknown')


def save_snapshot(frame, directory, provenance):
    """Immutable source-specific snapshot: original bars and provenance stay together."""
    directory = Path(directory)
    if frame.index.duplicated().any() or not frame.index.is_monotonic_increasing:
        raise ValueError('Duplicate or unsorted dates')
    if not provenance.source or provenance.price_basis not in ('raw','split_adjusted','total_return_adjusted','unknown'):
        raise ValueError('Explicit source and supported price basis required')
    directory.mkdir(parents=True, exist_ok=False)
    frame.to_csv(directory/'bars.csv')
    metadata = {**asdict(provenance), 'bars_sha256': sha256(directory/'bars.csv'),
                'invalid_rows': int((~valid_bars(frame)).sum())}
    (directory/'provenance.json').write_text(json.dumps(metadata, indent=2))
    return metadata


def latest_known(records, asof):
    """Financial/sector/membership records need public availability timestamps."""
    r = records.copy()
    if 'available_at' not in r:
        raise ValueError('available_at is required; fiscal period end is not availability')
    r['available_at'] = pd.to_datetime(r.available_at, utc=True)
    cutoff = pd.Timestamp(asof)
    cutoff = cutoff.tz_localize('UTC') if cutoff.tzinfo is None else cutoff.tz_convert('UTC')
    r = r[r.available_at <= cutoff]
    return r.sort_values('available_at').drop_duplicates('symbol', keep='last')
