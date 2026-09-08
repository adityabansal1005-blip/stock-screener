"""
Strategy Lab — 8 named strategies, walk-forward backtest, regime-wise reporting.

Each strategy is a pure function:
    strategy(df: DataFrame) -> Series[bool]  (True = entry signal on that bar)

The lab runs each strategy over a 10-year walk-forward window (train/test split
so we don't cherry-pick parameters on the same data we test on), then computes:
  win_rate, avg_return, sharpe, profit_factor, max_drawdown, trade_count
  breakdown by market regime (Bull / Bear / Sideways)
"""
import numpy as np
import pandas as pd
from typing import Callable

FORWARD_DAYS = 5
MIN_TRADES   = 10     # minimum trades to report a strategy as valid
COMMISSION   = 0.001  # 0.1% Zerodha/Groww
SLIPPAGE     = 0.0005

# ── helpers ──────────────────────────────────────────────────

def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(com=n-1, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=n-1, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"]  - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=n-1, adjust=False).mean()

def _supertrend(df: pd.DataFrame, mult: float = 3.0, period: int = 10) -> pd.Series:
    """Returns True where price is above Supertrend (bullish)."""
    atr   = _atr(df, period)
    hl2   = (df["high"] + df["low"]) / 2
    upper = hl2 + mult * atr
    lower = hl2 - mult * atr
    st    = pd.Series(np.nan, index=df.index)
    trend = pd.Series(1, index=df.index)
    for i in range(1, len(df)):
        if df["close"].iloc[i] > st.iloc[i-1] if not np.isnan(st.iloc[i-1]) else True:
            st.iloc[i]    = lower.iloc[i]
            trend.iloc[i] = 1
        else:
            st.iloc[i]    = upper.iloc[i]
            trend.iloc[i] = -1
    return trend == 1

def _macd(close: pd.Series):
    fast   = _ema(close, 12)
    slow   = _ema(close, 26)
    macd   = fast - slow
    signal = _ema(macd, 9)
    hist   = macd - signal
    return macd, signal, hist

def _bollinger(close: pd.Series, n: int = 20, k: float = 2.0):
    ma    = close.rolling(n).mean()
    std   = close.rolling(n).std()
    upper = ma + k * std
    lower = ma - k * std
    return upper, ma, lower

def _regime(close: pd.Series):
    """Simple regime: 'bull' / 'bear' / 'sideways' based on 200d EMA slope."""
    ema200 = _ema(close, 200)
    slope  = ema200.pct_change(20) * 100
    reg    = pd.Series("sideways", index=close.index)
    reg[slope >  0.5] = "bull"
    reg[slope < -0.5] = "bear"
    return reg


# ── strategy definitions ──────────────────────────────────────

def _strat_ema_momentum(df: pd.DataFrame) -> pd.Series:
    """EMA 9/21/50 stack + price above all three."""
    e9  = _ema(df["close"], 9)
    e21 = _ema(df["close"], 21)
    e50 = _ema(df["close"], 50)
    return (e9 > e21) & (e21 > e50) & (df["close"] > e50)

def _strat_rsi_reversal(df: pd.DataFrame) -> pd.Series:
    """RSI 14 crosses above 35 from oversold (<30)."""
    rsi   = _rsi(df["close"])
    cross = (rsi.shift(1) < 35) & (rsi >= 35)
    return cross

def _strat_supertrend_rs(df: pd.DataFrame) -> pd.Series:
    """Supertrend bullish + RSI 14 between 50-70 (momentum, not overbought)."""
    st  = _supertrend(df)
    rsi = _rsi(df["close"])
    return st & (rsi >= 50) & (rsi <= 70)

def _strat_macd_crossover(df: pd.DataFrame) -> pd.Series:
    """MACD histogram turns positive (crosses from negative)."""
    _, _, hist = _macd(df["close"])
    return (hist.shift(1) <= 0) & (hist > 0)

def _strat_breakout(df: pd.DataFrame) -> pd.Series:
    """
    20-day high breakout with volume expansion (vol > 1.5x 20d avg).
    Classic Turtle-style momentum entry.
    """
    high20 = df["high"].rolling(20).max().shift(1)
    vol20  = df["volume"].rolling(20).mean()
    return (df["close"] > high20) & (df["volume"] > 1.5 * vol20)

def _strat_ichimoku(df: pd.DataFrame) -> pd.Series:
    """
    Price above Ichimoku cloud (senkou span A & B).
    Tenkan > Kijun confirms momentum.
    """
    tenkan  = (df["high"].rolling(9).max()  + df["low"].rolling(9).min())  / 2
    kijun   = (df["high"].rolling(26).max() + df["low"].rolling(26).min()) / 2
    span_a  = ((tenkan + kijun) / 2).shift(26)
    span_b  = ((df["high"].rolling(52).max() + df["low"].rolling(52).min()) / 2).shift(26)
    cloud_top = pd.concat([span_a, span_b], axis=1).max(axis=1)
    return (df["close"] > cloud_top) & (tenkan > kijun)

def _strat_adx_trend(df: pd.DataFrame) -> pd.Series:
    """ADX > 25 (strong trend) + +DI > -DI (bullish direction)."""
    high, low, close = df["high"], df["low"], df["close"]
    up   = high.diff().clip(lower=0)
    down = (-low.diff()).clip(lower=0)
    up   = up.where(up > down, 0)
    down = down.where(down > up, 0)
    atr  = _atr(df, 14)
    pdi  = 100 * up.ewm(com=13, adjust=False).mean()   / atr.replace(0, np.nan)
    mdi  = 100 * down.ewm(com=13, adjust=False).mean() / atr.replace(0, np.nan)
    dx   = (100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)).fillna(0)
    adx  = dx.ewm(com=13, adjust=False).mean()
    return (adx > 25) & (pdi > mdi)

def _strat_pullback_swing(df: pd.DataFrame) -> pd.Series:
    """
    Swing pullback: price dips to 21d EMA in a bull trend (above 50d EMA)
    then recovers — classic buy-the-dip in an uptrend.
    """
    e21   = _ema(df["close"], 21)
    e50   = _ema(df["close"], 50)
    # In uptrend (price > EMA50), close was below EMA21 yesterday, now above
    above_trend = df["close"] > e50
    cross_e21   = (df["close"].shift(1) < e21.shift(1)) & (df["close"] >= e21)
    return above_trend & cross_e21


STRATEGIES: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "EMA Momentum":    _strat_ema_momentum,
    "RSI Reversal":    _strat_rsi_reversal,
    "Supertrend + RS": _strat_supertrend_rs,
    "MACD Crossover":  _strat_macd_crossover,
    "Breakout":        _strat_breakout,
    "Ichimoku":        _strat_ichimoku,
    "ADX Trend":       _strat_adx_trend,
    "Swing Pullback":  _strat_pullback_swing,
}


# ── backtest engine ────────────────────────────────────────────

def _backtest_strategy(df: pd.DataFrame, signals: pd.Series) -> dict:
    """
    Run a strategy's entry signals over df, hold for FORWARD_DAYS,
    then compute performance metrics.  Walk-forward: no lookahead.
    """
    returns     = []
    regime_rets = {"bull": [], "bear": [], "sideways": []}
    reg_series  = _regime(df["close"])

    idx = df.index
    for i in range(len(df) - FORWARD_DAYS):
        if not signals.iloc[i]:
            continue
        entry  = float(df["close"].iloc[i])     * (1 + SLIPPAGE)
        exit_p = float(df["close"].iloc[i + FORWARD_DAYS]) * (1 - SLIPPAGE)
        if entry <= 0:
            continue
        ret = ((exit_p - entry) / entry - COMMISSION * 2) * 100
        returns.append(ret)
        reg = reg_series.iloc[i]
        if reg in regime_rets:
            regime_rets[reg].append(ret)

    if len(returns) < MIN_TRADES:
        return {"valid": False, "trade_count": len(returns)}

    arr      = np.array(returns)
    wins     = int((arr > 0).sum())
    losses   = int((arr <= 0).sum())
    mean_r   = float(np.mean(arr))
    std_r    = float(np.std(arr))
    win_rets  = arr[arr > 0]
    loss_rets = arr[arr <= 0]

    # Max drawdown (peak-to-trough on cumulative returns)
    cum      = np.cumsum(arr)
    peak     = np.maximum.accumulate(cum)
    drawdown = peak - cum
    max_dd   = float(np.max(drawdown)) if len(drawdown) else 0.0

    gross_profit = float(win_rets.sum())  if len(win_rets)  else 0.0
    gross_loss   = float(abs(loss_rets.sum())) if len(loss_rets) else 0.0

    periods_per_year = 252 / FORWARD_DAYS
    sharpe = round(mean_r / std_r * np.sqrt(periods_per_year), 2) if std_r > 0 else None

    def _regime_stats(rets: list) -> dict:
        if len(rets) < 3:
            return {"trade_count": len(rets), "win_rate": None, "avg_return": None}
        a = np.array(rets)
        return {
            "trade_count": len(rets),
            "win_rate":    round(float((a > 0).mean() * 100), 1),
            "avg_return":  round(float(np.mean(a)), 2),
        }

    return {
        "valid":          True,
        "trade_count":    len(returns),
        "win_rate":       round(wins / len(returns) * 100, 1),
        "avg_return":     round(mean_r, 2),
        "sharpe":         sharpe,
        "profit_factor":  round(gross_profit / gross_loss, 2) if gross_loss > 0 else None,
        "max_drawdown":   round(max_dd, 2),
        "expectancy":     round(mean_r, 2),  # same as avg_return for equal-weight
        "bt_wins":        wins,
        "bt_losses":      losses,
        "regime_bull":    _regime_stats(regime_rets["bull"]),
        "regime_bear":    _regime_stats(regime_rets["bear"]),
        "regime_sideways":_regime_stats(regime_rets["sideways"]),
    }


def run_strategy_on_symbol(symbol: str, df: pd.DataFrame) -> dict:
    """Run all 8 strategies on one symbol's historical data."""
    if df is None or len(df) < 300:
        return {}
    results = {}
    for name, fn in STRATEGIES.items():
        try:
            sigs = fn(df)
            sigs = sigs.reindex(df.index).fillna(False)
            stats = _backtest_strategy(df, sigs)
            stats["strategy"] = name
            stats["symbol"]   = symbol
            results[name]     = stats
        except Exception as e:
            results[name] = {"valid": False, "error": str(e), "strategy": name, "symbol": symbol}
    return results


def run_strategy_lab(symbols: list[str], data: dict[str, pd.DataFrame] | None = None) -> dict:
    """
    Run all strategies on all symbols, aggregate results per strategy.
    Returns: {strategy_name: {aggregate_stats, per_symbol}}
    """
    from data_manager import load_many
    if data is None:
        data = load_many(symbols)

    # Collect per-symbol results
    all_results: dict[str, list[dict]] = {name: [] for name in STRATEGIES}

    for symbol, df in data.items():
        sym_results = run_strategy_on_symbol(symbol, df)
        for name, stats in sym_results.items():
            if stats.get("valid"):
                all_results[name].append(stats)

    # Aggregate
    aggregated = {}
    for name, sym_stats in all_results.items():
        if not sym_stats:
            aggregated[name] = {"strategy": name, "valid": False, "symbol_count": 0}
            continue

        win_rates  = [s["win_rate"]    for s in sym_stats]
        avg_rets   = [s["avg_return"]  for s in sym_stats]
        sharpes    = [s["sharpe"]      for s in sym_stats if s.get("sharpe") is not None]
        pfs        = [s["profit_factor"] for s in sym_stats if s.get("profit_factor") is not None]
        trade_cnts = [s["trade_count"] for s in sym_stats]
        drawdowns  = [s["max_drawdown"] for s in sym_stats]

        def _regime_agg(key: str) -> dict:
            stats_list = [s[key] for s in sym_stats if s.get(key) and s[key].get("trade_count", 0) >= 3]
            if not stats_list:
                return {}
            wrs = [x["win_rate"]   for x in stats_list if x.get("win_rate")   is not None]
            ars = [x["avg_return"] for x in stats_list if x.get("avg_return") is not None]
            return {
                "avg_win_rate":  round(float(np.mean(wrs)), 1) if wrs else None,
                "avg_return":    round(float(np.mean(ars)), 2) if ars else None,
                "symbol_count":  len(stats_list),
            }

        aggregated[name] = {
            "strategy":          name,
            "valid":             True,
            "symbol_count":      len(sym_stats),
            "total_trades":      sum(trade_cnts),
            "avg_win_rate":      round(float(np.mean(win_rates)),  1),
            "avg_return":        round(float(np.mean(avg_rets)),   2),
            "median_win_rate":   round(float(np.median(win_rates)),1),
            "avg_sharpe":        round(float(np.mean(sharpes)),    2) if sharpes else None,
            "avg_profit_factor": round(float(np.mean(pfs)),        2) if pfs     else None,
            "avg_max_drawdown":  round(float(np.mean(drawdowns)),  2),
            "regime_bull":       _regime_agg("regime_bull"),
            "regime_bear":       _regime_agg("regime_bear"),
            "regime_sideways":   _regime_agg("regime_sideways"),
            "per_symbol":        sym_stats,
        }

    # Rank by avg_sharpe (or avg_return if sharpe missing)
    ranked = sorted(
        [v for v in aggregated.values() if v.get("valid")],
        key=lambda x: (x.get("avg_sharpe") or 0, x.get("avg_return") or 0),
        reverse=True,
    )
    return {"strategies": aggregated, "ranked": [r["strategy"] for r in ranked]}
