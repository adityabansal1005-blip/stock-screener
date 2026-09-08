"""Freeze validated daily prices using Groww, then the local archive (no Yahoo)."""
import datetime as dt
import json
from pathlib import Path
import re
import sqlite3
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
COLS = ['Open', 'High', 'Low', 'Close', 'Volume']


def validate(frame, cutoff):
    if frame is None or frame.empty:
        raise ValueError('No daily bars available')
    frame = frame.copy()
    frame.index = pd.DatetimeIndex(frame.index)
    if frame.index.tz is not None:
        frame.index = frame.index.tz_localize(None)
    frame.index = frame.index.normalize()
    frame = frame.loc[frame.index <= pd.Timestamp(cutoff), COLS].sort_index()
    if frame.index.has_duplicates or len(frame) < 220:
        raise ValueError('Need at least 220 unique daily bars')
    frame = frame.apply(pd.to_numeric, errors='coerce')
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError('Non-finite daily prices')
    if (frame[COLS[:4]] <= 0).any().any() or (frame.Volume < 0).any():
        raise ValueError('Invalid prices or volume')
    if ((frame.High < frame[['Open', 'Close', 'Low']].max(axis=1)) |
        (frame.Low > frame[['Open', 'Close', 'High']].min(axis=1))).any():
        raise ValueError('Inconsistent daily OHLC ranges')
    if (pd.Timestamp(cutoff) - frame.index[-1]).days > 5:
        raise ValueError('Latest completed-session price is more than five days old')
    frame.index.name = 'Date'
    return frame


def prepare(folder):
    job = json.loads((folder / 'job.json').read_text(encoding='utf-8'))
    ticker = job['ticker']
    if not re.fullmatch(r'[A-Z0-9][A-Z0-9&-]{0,29}\.NS', ticker):
        raise ValueError('This price adapter currently supports NSE tickers only')
    # Only completed sessions: never present an intraday candle as a daily close.
    requested = dt.date.fromisoformat(job['analysis_date'])
    created = dt.datetime.fromisoformat(job['created_at']).astimezone(dt.timezone(dt.timedelta(hours=5,minutes=30)))
    cutoff = requested if created.time() >= dt.time(16,15) else requested - dt.timedelta(days=1)
    failures = []
    source = 'Groww daily candles'
    frame = None
    try:
        import config
        import groww_client
        if not groww_client.is_connected():
            groww_client.init_groww(config.GROWW_API_KEY, config.GROWW_API_SECRET)
        frame = validate(groww_client.get_daily_bars(ticker[:-3], days=800), cutoff)
    except Exception as exc:
        failures.append('Groww unavailable or failed validation: ' + type(exc).__name__)
    if frame is None:
        source = 'Local strategy_data.db archive (original vendor/adjustment provenance not recorded)'
        database = ROOT / 'strategy_data.db'
        with sqlite3.connect(database.as_uri() + '?mode=ro', uri=True) as conn:
            frame = pd.read_sql_query(
                'SELECT date, open AS Open, high AS High, low AS Low, close AS Close, volume AS Volume '
                'FROM ohlcv WHERE symbol IN (?, ?) AND date <= ? ORDER BY date', conn,
                params=(ticker, ticker[:-3], cutoff.isoformat()), index_col='date')
        frame = validate(frame, cutoff)
    frame.to_csv(folder / 'prices.csv')
    metadata = {'ticker': ticker, 'source': source, 'first_bar': str(frame.index[0].date()),
                'last_bar': str(frame.index[-1].date()), 'rows': len(frame),
                'daily_close': float(frame.Close.iloc[-1]), 'cutoff': cutoff.isoformat(),
                'warnings': failures, 'price_policy': 'Completed daily bars only; no cross-vendor stitching'}
    (folder / 'price_source.json').write_text(json.dumps(metadata), encoding='utf-8')


if __name__ == '__main__':
    identifier = sys.argv[1]
    if not re.fullmatch(r'[0-9a-f]{32}', identifier):
        raise ValueError('Invalid job identifier')
    target = ROOT / 'research_reports' / identifier
    try:
        prepare(target)
    except Exception as exc:
        # Never persist provider exceptions, which may contain signed URLs.
        (target / 'price_error.json').write_text(json.dumps({
            'error': 'No validated, recent daily history from Groww or the local archive.',
            'category': type(exc).__name__}), encoding='utf-8')
        sys.exit(1)
