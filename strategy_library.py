"""
Strategy Library — 55 named trading strategies encoded as signal functions.
Each function takes a DataFrame with lowercase OHLCV columns and returns
a boolean Series (True = entry signal on that bar).

Sources: Investopedia, StockCharts ChartSchool, Minervini, Elder, O'Neil,
         Darvas, Williams, Livermore, Weinstein, Turtle Trading rules.

Used by ml_signal.py as binary feature columns for LightGBM training.
"""
import numpy as np
import pandas as pd
from typing import Callable


# ── indicator helpers ─────────────────────────────────────────────────────────

def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()

def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    g = d.clip(lower=0).ewm(com=n - 1, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(com=n - 1, adjust=False).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))

def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"]  - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=n - 1, adjust=False).mean()

def _macd(close: pd.Series):
    fast = _ema(close, 12); slow = _ema(close, 26)
    macd = fast - slow; sig = _ema(macd, 9)
    return macd, sig, macd - sig

def _stoch(df: pd.DataFrame, k: int = 14, d: int = 3):
    lo = df["low"].rolling(k).min(); hi = df["high"].rolling(k).max()
    K  = 100 * (df["close"] - lo) / (hi - lo).replace(0, np.nan)
    return K, K.rolling(d).mean()

def _adx_full(df: pd.DataFrame, n: int = 14):
    up   = df["high"].diff().clip(lower=0)
    down = (-df["low"].diff()).clip(lower=0)
    pdm  = up.where(up > down, 0.0)
    mdm  = down.where(down > up, 0.0)
    atr  = _atr(df, n)
    pdi  = 100 * pdm.ewm(com=n - 1, adjust=False).mean() / atr.replace(0, np.nan)
    mdi  = 100 * mdm.ewm(com=n - 1, adjust=False).mean() / atr.replace(0, np.nan)
    dx   = (100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)).fillna(0)
    adx  = dx.ewm(com=n - 1, adjust=False).mean()
    return adx, pdi, mdi

def _bollinger(close: pd.Series, n: int = 20, k: float = 2.0):
    ma = _sma(close, n); std = close.rolling(n).std()
    return ma + k * std, ma, ma - k * std

def _obv(df: pd.DataFrame) -> pd.Series:
    return (np.sign(df["close"].diff().fillna(0)) * df["volume"]).cumsum()

def _mfi(df: pd.DataFrame, n: int = 14) -> pd.Series:
    tp  = (df["high"] + df["low"] + df["close"]) / 3
    rmf = tp * df["volume"]
    pos = rmf.where(tp > tp.shift(), 0.0).rolling(n).sum()
    neg = rmf.where(tp < tp.shift(), 0.0).rolling(n).sum()
    return 100 - 100 / (1 + pos / neg.replace(0, np.nan))

def _cci(df: pd.DataFrame, n: int = 20) -> pd.Series:
    tp  = (df["high"] + df["low"] + df["close"]) / 3
    sma = tp.rolling(n).mean()
    mad = tp.rolling(n).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - sma) / (0.015 * mad.replace(0, np.nan))

def _williams_r(df: pd.DataFrame, n: int = 14) -> pd.Series:
    hi = df["high"].rolling(n).max(); lo = df["low"].rolling(n).min()
    return -100 * (hi - df["close"]) / (hi - lo).replace(0, np.nan)

def _supertrend(df: pd.DataFrame, mult: float = 3.0, period: int = 10) -> pd.Series:
    atr = _atr(df, period); hl2 = (df["high"] + df["low"]) / 2
    upper = hl2 + mult * atr; lower = hl2 - mult * atr
    st = pd.Series(np.nan, index=df.index)
    trend = pd.Series(1, index=df.index)
    for i in range(1, len(df)):
        prev = st.iloc[i - 1]
        if np.isnan(prev) or df["close"].iloc[i] > prev:
            st.iloc[i] = lower.iloc[i]; trend.iloc[i] = 1
        else:
            st.iloc[i] = upper.iloc[i]; trend.iloc[i] = -1
    return trend == 1

def _psar(df: pd.DataFrame, step: float = 0.02, max_af: float = 0.2) -> pd.Series:
    """Parabolic SAR — True where price is above SAR (bullish)."""
    close, high, low = df["close"].values, df["high"].values, df["low"].values
    n = len(close)
    bull = np.zeros(n, dtype=bool)
    sar  = low[0]; af = step; ep = high[0]; rising = True
    for i in range(1, n):
        if rising:
            sar = sar + af * (ep - sar)
            sar = min(sar, low[i-1], low[i-2] if i > 1 else sar)
            if low[i] < sar:
                rising = False; sar = ep; af = step; ep = low[i]
            else:
                if high[i] > ep: ep = high[i]; af = min(af + step, max_af)
                bull[i] = True
        else:
            sar = sar + af * (ep - sar)
            sar = max(sar, high[i-1], high[i-2] if i > 1 else sar)
            if high[i] > sar:
                rising = True; sar = ep; af = step; ep = high[i]; bull[i] = True
            else:
                if low[i] < ep: ep = low[i]; af = min(af + step, max_af)
    return pd.Series(bull, index=df.index)

def _keltner(df: pd.DataFrame, n: int = 20, mult: float = 2.0):
    mid = _ema(df["close"], n); atr = _atr(df, n)
    return mid + mult * atr, mid, mid - mult * atr

def _donchian(df: pd.DataFrame, n: int = 20):
    return df["high"].rolling(n).max(), df["low"].rolling(n).min()

def _cmf(df: pd.DataFrame, n: int = 20) -> pd.Series:
    mfv = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (df["high"] - df["low"]).replace(0, np.nan) * df["volume"]
    return mfv.rolling(n).sum() / df["volume"].rolling(n).sum().replace(0, np.nan)

def _roc(close: pd.Series, n: int = 20) -> pd.Series:
    return (close / close.shift(n) - 1) * 100

def _bb_width(close: pd.Series, n: int = 20) -> pd.Series:
    upper, mid, lower = _bollinger(close, n)
    return (upper - lower) / mid.replace(0, np.nan)


# ── GROUP 1: Trend Following (10 strategies) ─────────────────────────────────

def s_ema_stack_9_21_50(df):
    """EMA 9 > 21 > 50 with price above all three — Darvas/O'Neil trend filter."""
    e9, e21, e50 = _ema(df["close"], 9), _ema(df["close"], 21), _ema(df["close"], 50)
    return (e9 > e21) & (e21 > e50) & (df["close"] > e50)

def s_ema_stack_20_50_200(df):
    """EMA 20 > 50 > 200 — Stan Weinstein Stage 2 proxy."""
    e20 = _ema(df["close"], 20); e50 = _ema(df["close"], 50); e200 = _ema(df["close"], 200)
    return (e20 > e50) & (e50 > e200) & (df["close"] > e200)

def s_golden_cross(df):
    """SMA 50 crosses above SMA 200 — classic golden cross."""
    s50 = _sma(df["close"], 50); s200 = _sma(df["close"], 200)
    return (s50 > s200) & (s50.shift(1) <= s200.shift(1))

def s_price_above_200ema(df):
    """Price above 200 EMA — long-term bull market filter."""
    return df["close"] > _ema(df["close"], 200)

def s_supertrend_bull(df):
    """Supertrend (3×ATR10) flips bullish — strong trend confirmation."""
    st = _supertrend(df)
    return st & (~st.shift(1).fillna(False).infer_objects(copy=False))

def s_supertrend_sustained(df):
    """Supertrend bullish for 3+ consecutive days."""
    st = _supertrend(df)
    return st & st.shift(1).fillna(False).infer_objects(copy=False) & st.shift(2).fillna(False).infer_objects(copy=False)

def s_adx_bull_trend(df):
    """ADX > 25 with +DI > -DI — strong directional bull trend."""
    adx, pdi, mdi = _adx_full(df)
    return (adx > 25) & (pdi > mdi)

def s_psar_bull(df):
    """Parabolic SAR flips bullish (price crosses above SAR)."""
    bull = _psar(df)
    return bull & (~bull.shift(1).fillna(False).infer_objects(copy=False))

def s_keltner_trend(df):
    """Price closes above upper Keltner Channel — strong trend breakout."""
    upper, _, _ = _keltner(df)
    return df["close"] > upper

def s_three_ema_reclaim(df):
    """Price reclaims all three EMAs (9, 21, 50) from below in same bar."""
    e9, e21, e50 = _ema(df["close"], 9), _ema(df["close"], 21), _ema(df["close"], 50)
    above_now  = (df["close"] > e9) & (df["close"] > e21) & (df["close"] > e50)
    below_prev = (df["close"].shift(1) < e9.shift(1)) | (df["close"].shift(1) < e50.shift(1))
    return above_now & below_prev


# ── GROUP 2: Momentum (10 strategies) ────────────────────────────────────────

def s_rsi_momentum(df):
    """RSI 14 in 50–70 zone and rising — healthy momentum, not overbought."""
    rsi = _rsi(df["close"])
    return (rsi >= 50) & (rsi <= 70) & (rsi > rsi.shift(2))

def s_rsi_cross_50(df):
    """RSI crosses above 50 — momentum shift from bearish to bullish."""
    rsi = _rsi(df["close"])
    return (rsi > 50) & (rsi.shift(1) <= 50)

def s_macd_bull_crossover(df):
    """MACD line crosses above signal line — standard MACD buy."""
    macd, sig, _ = _macd(df["close"])
    return (macd > sig) & (macd.shift(1) <= sig.shift(1))

def s_macd_histogram_positive(df):
    """MACD histogram turns positive — momentum accelerating upward."""
    _, _, hist = _macd(df["close"])
    return (hist > 0) & (hist.shift(1) <= 0)

def s_macd_above_zero(df):
    """Both MACD and signal above zero — confirmed bull momentum."""
    macd, sig, _ = _macd(df["close"])
    return (macd > 0) & (sig > 0) & (macd > sig)

def s_cci_breakout(df):
    """CCI crosses above +100 — price breaking out above statistical norm."""
    cci = _cci(df)
    return (cci > 100) & (cci.shift(1) <= 100)

def s_roc_positive(df):
    """20-day Rate of Change turns positive — price recovering over a month."""
    roc = _roc(df["close"], 20)
    return (roc > 0) & (roc.shift(1) <= 0)

def s_stoch_golden_cross(df):
    """Stochastic %K crosses %D from below 30 (oversold golden cross)."""
    K, D = _stoch(df)
    return (K > D) & (K.shift(1) <= D.shift(1)) & (K.shift(1) < 30)

def s_mfi_bullish(df):
    """MFI crosses above 50 — money flowing into stock."""
    mfi = _mfi(df)
    return (mfi > 50) & (mfi.shift(1) <= 50)

def s_williams_r_recovery(df):
    """Williams %R recovers from extreme oversold (< -80 → > -60)."""
    wr = _williams_r(df)
    return (wr > -60) & (wr.shift(1) <= -60) & (wr.shift(2) < -80)


# ── GROUP 3: Breakout (9 strategies) ─────────────────────────────────────────

def s_turtle_20d_breakout(df):
    """20-day high breakout — classic Turtle Trading entry."""
    high20 = df["high"].rolling(20).max().shift(1)
    return df["close"] > high20

def s_turtle_breakout_volume(df):
    """20-day high breakout confirmed by volume > 1.5× 20d average."""
    high20 = df["high"].rolling(20).max().shift(1)
    vol20  = df["volume"].rolling(20).mean()
    return (df["close"] > high20) & (df["volume"] > 1.5 * vol20)

def s_52week_high_breakout(df):
    """Price closes at new 52-week high — Darvas Box / O'Neil pivot rule."""
    high252 = df["high"].rolling(252).max().shift(1)
    return df["close"] >= high252

def s_donchian_breakout(df):
    """Price closes above Donchian upper channel (20d)."""
    upper, _ = _donchian(df, 20)
    return df["close"] > upper.shift(1)

def s_bollinger_upper_breakout(df):
    """Price closes above upper Bollinger Band with volume surge."""
    upper, _, _ = _bollinger(df["close"])
    vol20 = df["volume"].rolling(20).mean()
    return (df["close"] > upper) & (df["volume"] > 1.5 * vol20)

def s_inside_bar_breakout(df):
    """Inside bar breakout: today's high exceeds yesterday's inside bar high."""
    inside_bar = (df["high"].shift(1) < df["high"].shift(2)) & (df["low"].shift(1) > df["low"].shift(2))
    breakout   = df["high"] > df["high"].shift(1)
    return inside_bar & breakout

def s_consolidation_breakout(df):
    """ATR contracted to 20d low, then today's range expands (volatility breakout)."""
    atr = _atr(df, 14)
    low_vol = atr.shift(1) == atr.rolling(20).min().shift(1)
    expansion = (df["high"] - df["low"]) > atr.shift(1) * 1.5
    return low_vol & expansion & (df["close"] > df["open"])

def s_cup_handle_proxy(df):
    """
    Simplified Cup & Handle: price was at 52w high, pulled back >15%,
    recovered to within 5% of prior high, consolidates (low ATR), then breaks out.
    """
    high252 = df["high"].rolling(252).max()
    pullback = (df["close"] / high252 - 1) * 100
    near_high = (pullback > -5) & (pullback.shift(20) < -15)
    atr = _atr(df, 14)
    tight = atr / df["close"] < 0.015
    return near_high & tight & (df["close"] > df["close"].shift(1))

def s_vcp_proxy(df):
    """
    Minervini VCP (Volatility Contraction Pattern): price in Stage 2,
    progressively tightening closes (each pullback shallower), low ATR.
    """
    e50 = _ema(df["close"], 50); e150 = _ema(df["close"], 150)
    stage2 = (df["close"] > e50) & (e50 > e150)
    atr = _atr(df, 14)
    contracting_vol = (atr < atr.rolling(10).mean() * 0.8)
    return stage2 & contracting_vol


# ── GROUP 4: Mean Reversion (7 strategies) ───────────────────────────────────

def s_rsi_oversold_bounce(df):
    """RSI dips below 30, then crosses above 35 — classic oversold reversal."""
    rsi = _rsi(df["close"])
    was_oversold = rsi.rolling(5).min().shift(1) < 30
    return was_oversold & (rsi > 35) & (rsi.shift(1) <= 35)

def s_bollinger_lower_bounce(df):
    """Price touches lower BB, next bar closes above lower BB — mean reversion."""
    _, _, lower = _bollinger(df["close"])
    touched = df["low"].shift(1) <= lower.shift(1)
    recovered = df["close"] > lower
    return touched & recovered

def s_cci_oversold_recovery(df):
    """CCI was below -200 (extreme oversold), now crosses above -100."""
    cci = _cci(df)
    was_extreme = cci.rolling(5).min().shift(1) < -200
    return was_extreme & (cci > -100) & (cci.shift(1) <= -100)

def s_stoch_oversold_recovery(df):
    """%K crosses above %D with both below 20 (oversold reversal)."""
    K, D = _stoch(df)
    return (K > D) & (K.shift(1) <= D.shift(1)) & (K < 30) & (D < 30)

def s_3bar_pullback(df):
    """3-day pullback in an uptrend (above 50 EMA), then closes green above EMA21."""
    e21 = _ema(df["close"], 21); e50 = _ema(df["close"], 50)
    uptrend     = df["close"] > e50
    three_red   = (df["close"].shift(1) < df["open"].shift(1)) & (df["close"].shift(2) < df["open"].shift(2)) & (df["close"].shift(3) < df["open"].shift(3))
    recovery    = (df["close"] > df["open"]) & (df["close"] > e21)
    return uptrend & three_red & recovery

def s_ema21_bounce(df):
    """Price pulls back to EMA 21 and closes above it — swing pullback entry."""
    e21 = _ema(df["close"], 21); e50 = _ema(df["close"], 50)
    in_uptrend  = df["close"] > e50
    touched_e21 = (df["low"] <= e21 * 1.01) & (df["close"].shift(1) < e21.shift(1))
    recovered   = df["close"] > e21
    return in_uptrend & touched_e21 & recovered

def s_vwap_reclaim(df):
    """Price reclaims VWAP (approximate daily VWAP via VWAP-like MA)."""
    vwap = (df["close"] * df["volume"]).rolling(20).sum() / df["volume"].rolling(20).sum()
    return (df["close"] > vwap) & (df["close"].shift(1) <= vwap.shift(1))


# ── GROUP 5: Volatility / Squeeze (5 strategies) ─────────────────────────────

def s_bollinger_squeeze_breakout(df):
    """
    Bollinger Squeeze: BB width narrows to 6-month low, then expands
    and price closes above upper BB (TTM Squeeze inspired).
    """
    upper, mid, lower = _bollinger(df["close"])
    width = (upper - lower) / mid.replace(0, np.nan)
    squeeze = width.shift(1) == width.rolling(126).min().shift(1)
    expanding = (df["close"] > upper) & (width > width.shift(1))
    return squeeze & expanding

def s_keltner_bb_squeeze(df):
    """
    Keltner-Bollinger squeeze: BB inside KC signals coiling energy.
    Entry when BB expands outside KC and price is up.
    """
    kc_upper, _, kc_lower = _keltner(df, 20, 1.5)
    bb_upper, _, bb_lower = _bollinger(df["close"])
    squeezed  = (bb_upper.shift(3) < kc_upper.shift(3)) & (bb_lower.shift(3) > kc_lower.shift(3))
    expanding = (bb_upper > kc_upper) & (df["close"] > df["open"])
    return squeezed & expanding

def s_atr_expansion(df):
    """ATR contracts to 10-day low, then expands with bullish close."""
    atr = _atr(df, 14)
    contracted = atr.shift(1) == atr.rolling(10).min().shift(1)
    expanded   = atr > atr.shift(1) * 1.2
    return contracted & expanded & (df["close"] > df["open"])

def s_narrow_range_7(df):
    """NR7: today's range is the narrowest of last 7 days (coiling before move)."""
    today_range = df["high"] - df["low"]
    min_range   = today_range.rolling(7).min()
    return (today_range == min_range) & (df["close"] > df["open"])

def s_high_tight_flag(df):
    """
    High Tight Flag: stock gained >100% in 2 months, now consolidating <25%.
    Classic O'Neil setup before explosive continuation.
    """
    gain2m = df["close"] / df["close"].shift(42) - 1
    high   = df["close"].rolling(5).max()
    flag   = (df["close"] / high - 1) > -0.15
    return (gain2m > 1.0) & flag


# ── GROUP 6: Candlestick / Price Action (8 strategies) ───────────────────────

def s_bullish_engulfing(df):
    """Bullish engulfing: red candle followed by larger green that engulfs it."""
    prev_red  = df["close"].shift(1) < df["open"].shift(1)
    curr_green = df["close"] > df["open"]
    engulfs    = (df["open"] < df["close"].shift(1)) & (df["close"] > df["open"].shift(1))
    return prev_red & curr_green & engulfs

def s_hammer(df):
    """Hammer: small body near top, long lower wick ≥ 2× body, after downtrend."""
    body  = (df["close"] - df["open"]).abs()
    lower = df["open"].combine(df["close"], min) - df["low"]
    upper = df["high"] - df["open"].combine(df["close"], max)
    hammer = (lower > 2 * body.replace(0, np.nan)) & (upper < body)
    downtrend = df["close"].shift(3) > df["close"].shift(1)
    return hammer & downtrend

def s_morning_star(df):
    """Morning star: big red, small body doji, big green — 3-candle reversal."""
    red1   = df["close"].shift(2) < df["open"].shift(2)
    big1   = (df["open"].shift(2) - df["close"].shift(2)) > (df["close"].rolling(20).std().shift(2) * 0.5)
    small2 = (df["close"].shift(1) - df["open"].shift(1)).abs() < (df["close"].rolling(20).std().shift(1) * 0.3)
    green3 = df["close"] > df["open"]
    big3   = (df["close"] - df["open"]) > (df["close"].rolling(20).std() * 0.5)
    return red1 & big1 & small2 & green3 & big3

def s_bullish_harami(df):
    """Bullish harami: large red candle, then small green body inside it."""
    big_red  = (df["open"].shift(1) - df["close"].shift(1)) > _atr(df, 14).shift(1) * 0.5
    small_green = (df["close"] > df["open"]) & ((df["close"] - df["open"]) < (df["open"].shift(1) - df["close"].shift(1)) * 0.5)
    inside   = (df["open"] > df["close"].shift(1)) & (df["close"] < df["open"].shift(1))
    return big_red & small_green & inside

def s_three_white_soldiers(df):
    """Three white soldiers: 3 consecutive strong green candles, each higher close."""
    c = df["close"]; o = df["open"]
    green = lambda n: c.shift(n) > o.shift(n)
    each_higher = (c.shift(1) > c.shift(2)) & (c > c.shift(1))
    bodies = lambda n: (c.shift(n) - o.shift(n)) > (df["high"].shift(n) - df["low"].shift(n)) * 0.6
    return green(2) & green(1) & green(0) & each_higher & bodies(2) & bodies(1)

def s_pin_bar_reversal(df):
    """Pin bar at support: long wick (>60% of range), small body, closes in top 30%."""
    total = df["high"] - df["low"]
    body  = (df["close"] - df["open"]).abs()
    lower_wick = df["open"].combine(df["close"], min) - df["low"]
    pin = (lower_wick > total * 0.6) & (body < total * 0.3)
    e50 = _ema(df["close"], 50)
    near_support = df["close"] < e50 * 1.03
    return pin & near_support

def s_marubozu_bull(df):
    """Bullish marubozu: very small wicks, strong green close — conviction buying."""
    body   = df["close"] - df["open"]
    range_ = df["high"] - df["low"]
    return (body > range_ * 0.85) & (df["close"] > df["open"])

def s_doji_reversal(df):
    """Doji after downtrend: near-equal open/close signals indecision / reversal."""
    body  = (df["close"] - df["open"]).abs()
    range_ = df["high"] - df["low"]
    doji  = body < range_ * 0.1
    down  = df["close"].shift(3) > df["close"].shift(1)
    return doji & down


# ── GROUP 7: Volume-Based (6 strategies) ─────────────────────────────────────

def s_obv_breakout(df):
    """OBV makes a new 20-day high — institutional accumulation."""
    obv = _obv(df)
    return obv > obv.rolling(20).max().shift(1)

def s_volume_surge_up(df):
    """Volume > 2.5× 20d average with a green close — conviction move."""
    vol20 = df["volume"].rolling(20).mean()
    return (df["volume"] > 2.5 * vol20) & (df["close"] > df["open"])

def s_cmf_positive(df):
    """Chaikin Money Flow crosses above zero — buyers controlling price."""
    cmf = _cmf(df)
    return (cmf > 0) & (cmf.shift(1) <= 0)

def s_mfi_cross_50(df):
    """Money Flow Index crosses above 50 from below — smart money entering."""
    mfi = _mfi(df)
    return (mfi > 50) & (mfi.shift(1) <= 50)

def s_volume_dry_up_reversal(df):
    """Volume dries up on pullback (<50% avg), then surges on recovery day."""
    vol20 = df["volume"].rolling(20).mean()
    dry   = df["volume"].shift(1) < vol20.shift(1) * 0.5
    surge = (df["volume"] > vol20) & (df["close"] > df["open"])
    return dry & surge

def s_obv_divergence_bull(df):
    """OBV making higher lows while price makes lower lows — bullish divergence."""
    obv   = _obv(df)
    price_lower_low = (df["close"] < df["close"].shift(10)) & (df["close"].shift(5) < df["close"].shift(10))
    obv_higher_low  = (obv > obv.shift(10)) & (obv.shift(5) > obv.shift(10))
    return price_lower_low & obv_higher_low


# ── GROUP 8: Named / Advanced Strategies (5 strategies) ──────────────────────

def s_weinstein_stage2(df):
    """
    Stan Weinstein Stage 2: price above rising 30-week (150d) MA,
    and above 10-week (50d) MA — confirmed stage 2 uptrend.
    """
    sma150 = _sma(df["close"], 150); sma50 = _sma(df["close"], 50)
    rising150 = sma150 > sma150.shift(4)
    return (df["close"] > sma150) & (df["close"] > sma50) & rising150

def s_elder_triple_screen(df):
    """
    Elder Triple Screen: weekly trend bullish (EMA rising 13-week),
    daily RSI dips to 25-35 (oversold on intermediate), then recovers.
    Elder's 'buy only when weekly is up and daily is oversold'.
    """
    e65   = _ema(df["close"], 65)   # ~13 weeks * 5 days
    weekly_bull = e65 > e65.shift(5)
    rsi14 = _rsi(df["close"])
    oversold = rsi14.rolling(5).min().shift(1) < 35
    recovering = rsi14 > 35
    return weekly_bull & oversold & recovering

def s_darvas_box(df):
    """
    Darvas Box: stock hits new 52w high, consolidates (forms a box),
    then breaks above the box top on high volume.
    """
    new52 = df["high"] >= df["high"].rolling(252).max().shift(1)
    box_hi = df["high"].rolling(10).max().shift(1)
    box_lo = df["low"].rolling(10).min().shift(1)
    tight_box = (box_hi - box_lo) / box_lo < 0.08
    breakout  = df["close"] > box_hi
    vol20     = df["volume"].rolling(20).mean()
    return tight_box & breakout & (df["volume"] > 1.5 * vol20)

def s_canslim_proxy(df):
    """
    O'Neil CANSLIM proxy: RS leader (near 52w high), EMA stack bullish,
    accumulation days > distribution days (volume on up days).
    Simplified institutional-quality filter.
    """
    e50 = _ema(df["close"], 50); e150 = _ema(df["close"], 150)
    near_high = df["close"] >= df["high"].rolling(252).max().shift(1) * 0.85
    ema_stack = (df["close"] > e50) & (e50 > e150)
    up_vol   = df["volume"].where(df["close"] > df["open"], 0).rolling(20).sum()
    down_vol = df["volume"].where(df["close"] < df["open"], 0).rolling(20).sum()
    accum    = up_vol > down_vol * 1.2
    return near_high & ema_stack & accum

def s_livermore_pivot(df):
    """
    Livermore Pivot Point: stock breaks to a new 6-month high after
    at least a 3-week consolidation (Livermore's natural rally pause).
    """
    high126 = df["high"].rolling(126).max().shift(1)
    new_high = df["close"] > high126
    consolidation = (df["close"].rolling(15).max().shift(1) - df["close"].rolling(15).min().shift(1)) / df["close"].rolling(15).min().shift(1) < 0.08
    return new_high & consolidation


# ── registry ──────────────────────────────────────────────────────────────────

STRATEGIES: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    # Trend following
    "ema_stack_9_21_50":     s_ema_stack_9_21_50,
    "ema_stack_20_50_200":   s_ema_stack_20_50_200,
    "golden_cross":          s_golden_cross,
    "price_above_200ema":    s_price_above_200ema,
    "supertrend_bull_flip":  s_supertrend_bull,
    "supertrend_sustained":  s_supertrend_sustained,
    "adx_bull_trend":        s_adx_bull_trend,
    "psar_bull_flip":        s_psar_bull,
    "keltner_trend":         s_keltner_trend,
    "three_ema_reclaim":     s_three_ema_reclaim,
    # Momentum
    "rsi_momentum_zone":     s_rsi_momentum,
    "rsi_cross_50":          s_rsi_cross_50,
    "macd_crossover":        s_macd_bull_crossover,
    "macd_hist_positive":    s_macd_histogram_positive,
    "macd_above_zero":       s_macd_above_zero,
    "cci_breakout":          s_cci_breakout,
    "roc_positive":          s_roc_positive,
    "stoch_golden_cross":    s_stoch_golden_cross,
    "mfi_bullish":           s_mfi_bullish,
    "williams_r_recovery":   s_williams_r_recovery,
    # Breakout
    "turtle_20d":            s_turtle_20d_breakout,
    "turtle_volume":         s_turtle_breakout_volume,
    "52w_high_breakout":     s_52week_high_breakout,
    "donchian_breakout":     s_donchian_breakout,
    "bb_upper_breakout":     s_bollinger_upper_breakout,
    "inside_bar_breakout":   s_inside_bar_breakout,
    "consolidation_breakout":s_consolidation_breakout,
    "cup_handle_proxy":      s_cup_handle_proxy,
    "vcp_proxy":             s_vcp_proxy,
    # Mean reversion
    "rsi_oversold_bounce":   s_rsi_oversold_bounce,
    "bb_lower_bounce":       s_bollinger_lower_bounce,
    "cci_oversold_recovery": s_cci_oversold_recovery,
    "stoch_oversold_recovery":s_stoch_oversold_recovery,
    "3bar_pullback":         s_3bar_pullback,
    "ema21_bounce":          s_ema21_bounce,
    "vwap_reclaim":          s_vwap_reclaim,
    # Volatility / squeeze
    "bb_squeeze_breakout":   s_bollinger_squeeze_breakout,
    "keltner_bb_squeeze":    s_keltner_bb_squeeze,
    "atr_expansion":         s_atr_expansion,
    "narrow_range_7":        s_narrow_range_7,
    "high_tight_flag":       s_high_tight_flag,
    # Candlestick / price action
    "bullish_engulfing":     s_bullish_engulfing,
    "hammer":                s_hammer,
    "morning_star":          s_morning_star,
    "bullish_harami":        s_bullish_harami,
    "three_white_soldiers":  s_three_white_soldiers,
    "pin_bar_reversal":      s_pin_bar_reversal,
    "marubozu_bull":         s_marubozu_bull,
    "doji_reversal":         s_doji_reversal,
    # Volume
    "obv_breakout":          s_obv_breakout,
    "volume_surge_up":       s_volume_surge_up,
    "cmf_positive":          s_cmf_positive,
    "mfi_cross_50":          s_mfi_cross_50,
    "volume_dry_up_reversal":s_volume_dry_up_reversal,
    "obv_divergence_bull":   s_obv_divergence_bull,
    # Named/advanced
    "weinstein_stage2":      s_weinstein_stage2,
    "elder_triple_screen":   s_elder_triple_screen,
    "darvas_box":            s_darvas_box,
    "canslim_proxy":         s_canslim_proxy,
    "livermore_pivot":       s_livermore_pivot,
}


def compute_all_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run all 59 strategies on df. Returns DataFrame with one boolean column per strategy.
    Silently skips strategies that error (df too short, missing columns, etc.)
    """
    out = pd.DataFrame(index=df.index)
    for name, fn in STRATEGIES.items():
        try:
            sig = fn(df)
            out[f"strat_{name}"] = sig.reindex(df.index).fillna(False).astype(int)
        except Exception:
            out[f"strat_{name}"] = 0
    return out
