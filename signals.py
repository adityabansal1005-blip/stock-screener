import pandas as pd
import numpy as np


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(com=length - 1, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(com=length - 1, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    value = 100 - (100 / (1 + rs))
    value = value.mask((loss == 0) & (gain > 0), 100.0)
    return value.mask((loss == 0) & (gain == 0), 50.0)


def macd(series: pd.Series, fast=12, slow=26, signal=9):
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bbands(series: pd.Series, length=20, std=2):
    mid = series.rolling(length).mean()
    stddev = series.rolling(length).std()
    upper = mid + std * stddev
    lower = mid - std * stddev
    return upper, mid, lower


def atr(high: pd.Series, low: pd.Series, close: pd.Series, length=14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(com=length - 1, adjust=False).mean()


def compute_signals(df: pd.DataFrame) -> dict:
    if df is None or len(df) < 50:
        return None

    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    # yfinance includes today as last row with NaN close when market hasn't opened yet.
    # Strip all trailing rows where close is NaN so last = last completed trading day.
    df = df[df["close"].notna()]
    if len(df) < 50:
        return None

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    df["ema9"] = ema(close, 9)
    df["ema21"] = ema(close, 21)
    df["ema50"] = ema(close, 50)
    df["ema200"] = ema(close, 200) if len(df) >= 200 else pd.Series([np.nan] * len(df), index=df.index)

    df["macd"], df["macd_signal"], df["macd_hist"] = macd(close)
    df["rsi"] = rsi(close, 14)
    df["bb_upper"], df["bb_mid"], df["bb_lower"] = bbands(close)
    df["atr"] = atr(high, low, close, 14)
    df["vol_avg20"] = volume.rolling(20).mean()
    df["vol_ratio"] = volume / df["vol_avg20"]

    last = df.iloc[-1]
    prev = df.iloc[-2]

    signals = []
    score = 0

    ema_bull = last["ema9"] > last["ema21"] > last["ema50"]
    ema_bear = last["ema9"] < last["ema21"] < last["ema50"]

    if ema_bull:
        signals.append("EMA Bullish Alignment (9>21>50)")
        score += 2
    if ema_bear:
        signals.append("EMA Bearish Alignment (9<21<50)")
        score -= 2

    if prev["ema9"] < prev["ema21"] and last["ema9"] > last["ema21"]:
        signals.append("EMA 9 crossed above EMA 21 (Bullish)")
        score += 3
    if prev["ema9"] > prev["ema21"] and last["ema9"] < last["ema21"]:
        signals.append("EMA 9 crossed below EMA 21 (Bearish)")
        score -= 3

    if prev["macd_hist"] < 0 and last["macd_hist"] > 0:
        signals.append("MACD Histogram turned Positive (Bullish)")
        score += 2
    if prev["macd_hist"] > 0 and last["macd_hist"] < 0:
        signals.append("MACD Histogram turned Negative (Bearish)")
        score -= 2
    if last["macd"] > last["macd_signal"] and last["macd"] > 0:
        signals.append("MACD above Signal above Zero (Strong Bull)")
        score += 1

    rsi_val = last["rsi"]
    if rsi_val < 35:
        signals.append(f"RSI Oversold ({rsi_val:.1f}) — Potential reversal up")
        score += 2
    elif rsi_val > 65:
        signals.append(f"RSI Overbought ({rsi_val:.1f}) — Potential reversal down")
        score -= 2
    elif 45 < rsi_val < 60:
        signals.append(f"RSI in Bullish Zone ({rsi_val:.1f})")
        score += 1

    if last["vol_ratio"] > 1.5:
        if score > 0:
            signals.append(f"High Volume ({last['vol_ratio']:.1f}x avg) confirms Bull move")
            score += 1
        elif score < 0:
            signals.append(f"High Volume ({last['vol_ratio']:.1f}x avg) confirms Bear move")
            score -= 1

    if last["close"] > last["bb_upper"]:
        signals.append("Price above Bollinger Upper Band (Breakout / Overbought)")
    if last["close"] < last["bb_lower"]:
        signals.append("Price below Bollinger Lower Band (Breakdown / Oversold)")
        score += 1

    above_200 = not np.isnan(last["ema200"]) and last["close"] > last["ema200"]
    below_200 = not np.isnan(last["ema200"]) and last["close"] < last["ema200"]
    if above_200:
        signals.append("Price above 200 EMA (Long-term Uptrend)")
        score += 1
    if below_200:
        signals.append("Price below 200 EMA (Long-term Downtrend)")
        score -= 1

    recent_change = (close.iloc[-1] - close.iloc[-5]) / close.iloc[-5] * 100
    intraday_signal = "NEUTRAL"
    if recent_change > 0.5 and last["vol_ratio"] > 1.3:
        intraday_signal = "BULLISH MOMENTUM"
    elif recent_change < -0.5 and last["vol_ratio"] > 1.3:
        intraday_signal = "BEARISH MOMENTUM"

    if score >= 4:
        verdict = "STRONG BUY"
    elif score >= 2:
        verdict = "BUY"
    elif score <= -4:
        verdict = "STRONG SELL / SHORT"
    elif score <= -2:
        verdict = "SELL / AVOID"
    else:
        verdict = "NEUTRAL / WAIT"

    atr_val = last["atr"]
    sl_buy = round(last["close"] - 1.5 * atr_val, 2)
    target1_buy = round(last["close"] + 2 * atr_val, 2)
    target2_buy = round(last["close"] + 3.5 * atr_val, 2)

    change_pct = (last["close"] - df.iloc[-2]["close"]) / df.iloc[-2]["close"] * 100

    return {
        "close": round(last["close"], 2),
        "change_pct": round(change_pct, 2),
        "rsi": round(rsi_val, 1),
        "macd_hist": round(float(last["macd_hist"]), 4) if not np.isnan(last["macd_hist"]) else 0,
        "ema9": round(last["ema9"], 2),
        "ema21": round(last["ema21"], 2),
        "ema50": round(last["ema50"], 2),
        "vol_ratio": round(float(last["vol_ratio"]), 2),
        "score": score,
        "verdict": verdict,
        "intraday_signal": intraday_signal,
        "signals": signals,
        "sl_buy": sl_buy,
        "target1": target1_buy,
        "target2": target2_buy,
        "atr": round(float(atr_val), 2) if not np.isnan(atr_val) else 0,
    }
