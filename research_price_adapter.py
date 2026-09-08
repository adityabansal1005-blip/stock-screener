"""Bind every TradingAgents OHLCV/indicator path to the frozen local snapshot."""
import json
import pandas as pd


def install(folder):
    from tradingagents.dataflows import stockstats_utils, market_data_validator, y_finance, interface
    metadata = json.loads((folder / 'price_source.json').read_text(encoding='utf-8'))
    frame = pd.read_csv(folder / 'prices.csv', parse_dates=['Date'])

    def load(symbol, curr_date):
        if symbol.upper() != metadata['ticker']:
            raise ValueError('No frozen price snapshot for this ticker')
        subset = frame[frame.Date <= pd.Timestamp(curr_date)].copy()
        if subset.empty:
            raise ValueError('Requested date precedes available history')
        return subset

    def prices(symbol, start_date, end_date):
        data = load(symbol, end_date)
        data = data[data.Date >= pd.Timestamp(start_date)]
        return '# Price source: ' + metadata['source'] + '\n# Last completed bar: ' + metadata['last_bar'] + '\n' + data.to_csv(index=False)

    # Upstream imports these functions by value in several modules. Bind all
    # entry points inside this isolated process, without modifying upstream code.
    stockstats_utils.load_ohlcv = load
    market_data_validator.load_ohlcv = load
    y_finance.load_ohlcv = load
    interface.VENDOR_METHODS['get_stock_data']['local_snapshot'] = prices
    interface.VENDOR_METHODS['get_indicators']['local_snapshot'] = y_finance.get_stock_stats_indicators_window
    return metadata
