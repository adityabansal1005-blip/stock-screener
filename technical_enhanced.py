"""
Enhanced technical analysis: Supertrend, CCI, VWAP, CPR, ADX, Stochastic,
Parabolic SAR, Ichimoku Cloud, Stochastic RSI, Williams %R, OBV,
Relative Strength vs Nifty, weekly trend, candlestick patterns,
pivot S/R, SL/Target/R:R, and 0-100 score.
"""
import pandas as pd
import numpy as np


# ── Supertrend ───────────────────────────────────────────────

def supertrend(high: pd.Series, low: pd.Series, close: pd.Series,
               period: int = 10, multiplier: float = 3.0):
    """
    Returns (supertrend_line, direction) where direction: 1=bullish, -1=bearish.
    Uses numpy arrays for the loop — avoids pandas CoW slowness.
    """
    h = high.to_numpy(dtype=float)
    l = low.to_numpy(dtype=float)
    c = close.to_numpy(dtype=float)
    n = len(c)

    # True Range
    pc = np.empty(n); pc[0] = c[0]; pc[1:] = c[:-1]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))

    # EWM ATR
    alpha = 2.0 / (period + 1)
    atr = np.empty(n)
    atr[0] = tr[0]
    for i in range(1, n):
        atr[i] = alpha * tr[i] + (1 - alpha) * atr[i - 1]

    hl = (h + l) / 2
    ub = hl + multiplier * atr
    lb = hl - multiplier * atr

    upper = ub.copy()
    lower = lb.copy()
    for i in range(1, n):
        upper[i] = ub[i] if (ub[i] < upper[i-1] or c[i-1] > upper[i-1]) else upper[i-1]
        lower[i] = lb[i] if (lb[i] > lower[i-1] or c[i-1] < lower[i-1]) else lower[i-1]

    st  = np.empty(n)
    dirn = np.empty(n, dtype=int)
    st[0] = upper[0]; dirn[0] = -1
    for i in range(1, n):
        if st[i-1] == upper[i-1]:
            if c[i] <= upper[i]: st[i] = upper[i]; dirn[i] = -1
            else:                st[i] = lower[i]; dirn[i] =  1
        else:
            if c[i] >= lower[i]: st[i] = lower[i]; dirn[i] =  1
            else:                st[i] = upper[i]; dirn[i] = -1

    return pd.Series(st, index=close.index), pd.Series(dirn, index=close.index)


# ── CCI (Commodity Channel Index) ────────────────────────────

def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20):
    tp = (high + low + close) / 3
    rolling_mean = tp.rolling(period).mean()
    rolling_mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - rolling_mean) / (0.015 * rolling_mad.replace(0, np.nan))


# ── VWAP ─────────────────────────────────────────────────────

def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series):
    """
    Session-resetting VWAP — resets at each calendar day boundary.
    For a 5-day × 15-min dataset this produces true intraday VWAP per session.
    If the index has no date variation (single-session data), it degrades to
    cumulative VWAP from bar 0 (correct for that case too).
    """
    tp     = (high + low + close) / 3
    tp_vol = tp * volume
    result = pd.Series(np.nan, index=high.index)

    try:
        dates = high.index.normalize()   # datetime64 at midnight per bar
        for d in dates.unique():
            mask = dates == d
            cv   = volume[mask].cumsum().replace(0, np.nan)
            result[mask] = tp_vol[mask].cumsum() / cv
    except Exception:
        # Fallback for non-datetime or tz-naive indices
        cumvol = volume.cumsum().replace(0, np.nan)
        result = tp_vol.cumsum() / cumvol

    return result


# ── Stochastic ───────────────────────────────────────────────

def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
               k_period: int = 14, d_period: int = 3):
    lowest = low.rolling(k_period).min()
    highest = high.rolling(k_period).max()
    denom = (highest - lowest).replace(0, np.nan)
    k = 100 * (close - lowest) / denom
    d = k.rolling(d_period).mean()
    return k, d


# ── ADX ──────────────────────────────────────────────────────

def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    """Returns (adx_series, plus_di, minus_di)."""
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_mask = (plus_dm > minus_dm) & (plus_dm > 0)
    minus_mask = (minus_dm > plus_dm) & (minus_dm > 0)
    plus_dm = plus_dm.where(plus_mask, 0.0)
    minus_dm = minus_dm.where(minus_mask, 0.0)

    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr14 = tr.ewm(com=period - 1, adjust=False).mean().replace(0, np.nan)
    plus_di = 100 * plus_dm.ewm(com=period - 1, adjust=False).mean() / atr14
    minus_di = 100 * minus_dm.ewm(com=period - 1, adjust=False).mean() / atr14
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_val = dx.ewm(com=period - 1, adjust=False).mean()
    return adx_val, plus_di, minus_di


# ── Candlestick patterns ─────────────────────────────────────

def detect_patterns(df: pd.DataFrame) -> list[str]:
    if len(df) < 3:
        return []
    patterns = []

    last  = df.iloc[-1]
    prev  = df.iloc[-2]
    prev2 = df.iloc[-3]

    o, h, l, c = last["open"], last["high"], last["low"], last["close"]
    body      = abs(c - o)
    wick_up   = h - max(o, c)
    wick_dn   = min(o, c) - l
    full_range = h - l
    ma20 = df["close"].rolling(20).mean().iloc[-1]

    if full_range > 0 and body / full_range < 0.08:
        patterns.append("Doji (indecision — watch breakout direction)")

    if body > 0 and full_range > 0 and wick_dn >= 2.5 * body and wick_up <= body * 0.4 and c < ma20:
        patterns.append("Hammer (Bullish Reversal Signal)")

    if body > 0 and full_range > 0 and wick_up >= 2.5 * body and wick_dn <= body * 0.4 and c < ma20:
        patterns.append("Inverted Hammer (Potential Bullish Reversal)")

    if body > 0 and full_range > 0 and wick_up >= 2.5 * body and wick_dn <= body * 0.4 and c > ma20:
        patterns.append("Shooting Star (Bearish Reversal Signal)")

    prev_body = prev["close"] - prev["open"]
    curr_body = c - o
    if prev_body < 0 and curr_body > 0 and o <= prev["close"] and c >= prev["open"]:
        patterns.append("Bullish Engulfing (Strong Buy Signal)")

    if prev_body > 0 and curr_body < 0 and o >= prev["close"] and c <= prev["open"]:
        patterns.append("Bearish Engulfing (Strong Sell Signal)")

    p2_body = prev2["close"] - prev2["open"]
    if (p2_body < 0 and abs(prev_body) < abs(p2_body) * 0.35
            and curr_body > 0 and c > (prev2["open"] + prev2["close"]) / 2):
        patterns.append("Morning Star (Strong 3-Candle Bullish Reversal)")

    if (p2_body > 0 and abs(prev_body) < abs(p2_body) * 0.35
            and curr_body < 0 and c < (prev2["open"] + prev2["close"]) / 2):
        patterns.append("Evening Star (Strong 3-Candle Bearish Reversal)")

    if full_range > 0 and body / full_range > 0.92:
        if c > o:
            patterns.append("Bullish Marubozu (Strong Buying Pressure)")
        else:
            patterns.append("Bearish Marubozu (Strong Selling Pressure)")

    # Double rejection candle — two pin bars rejecting the same support level
    if len(df) >= 5:
        pin_bars = []
        for k in range(max(0, len(df) - 5), len(df)):
            bk = df.iloc[k]
            ok2, hk2, lk2, ck2 = bk["open"], bk["high"], bk["low"], bk["close"]
            body_k   = abs(ck2 - ok2)
            wick_dn_k = min(ok2, ck2) - lk2
            range_k   = hk2 - lk2
            if range_k > 0 and wick_dn_k / range_k > 0.55 and (body_k == 0 or wick_dn_k > 1.5 * body_k):
                pin_bars.append((k, lk2))
        if len(pin_bars) >= 2:
            for bi in range(len(pin_bars) - 1):
                l1, l2 = pin_bars[bi][1], pin_bars[bi + 1][1]
                if l1 > 0 and abs(l1 - l2) / l1 < 0.015:
                    patterns.append(
                        f"Double Rejection Candle at ₹{round((l1 + l2) / 2, 2)} — "
                        "strong support, watch for breakout"
                    )
                    break

    return patterns


# ── Pivot S/R + CPR ──────────────────────────────────────────

def pivot_support_resistance(df: pd.DataFrame) -> dict:
    """Classic pivot points + Central Pivot Range from previous session."""
    if len(df) < 2:
        return {}
    prev = df.iloc[-2]
    h, l, c = prev["high"], prev["low"], prev["close"]
    pp     = (h + l + c) / 3
    r1     = 2 * pp - l
    r2     = pp + (h - l)
    s1     = 2 * pp - h
    s2     = pp - (h - l)
    cpr_top = (h + l) / 2
    cpr_bot = 2 * pp - cpr_top
    cpr_width_pct = abs(cpr_top - cpr_bot) / pp * 100
    return {
        "pivot":      round(pp,   2),
        "resistance": round(r1,   2),
        "resistance2": round(r2,  2),
        "support":    round(s1,   2),
        "support2":   round(s2,   2),
        "cpr_top":    round(cpr_top,   2),
        "cpr_bottom": round(cpr_bot,   2),
        "cpr_width_pct": round(cpr_width_pct, 2),
        "cpr_narrow": bool(cpr_width_pct < 0.5),   # narrow CPR = trending day expected
    }


# ── Minervini Stage Analysis ─────────────────────────────────

def compute_stage(df: pd.DataFrame) -> dict:
    """
    Minervini Stage Analysis (1-4).
    Stage 1: Basing — price consolidating, EMA150 flat/declining
    Stage 2: Advancing — price above EMA150, EMA150 rising, EMA50 > EMA150  ← entries here
    Stage 3: Topping — EMA150 flattening, price stalling near highs
    Stage 4: Declining — price below EMA150, EMA150 falling
    """
    if df is None or len(df) < 30:
        return {"stage": 0, "stage_label": "Unknown", "stage_desc": "Insufficient data"}

    close = df["close"]
    high  = df["high"]
    low   = df["low"]
    curr  = float(close.iloc[-1])

    ema50_s  = close.ewm(span=50,  adjust=False).mean()
    ema150_s = close.ewm(span=150, adjust=False).mean()

    e50  = float(ema50_s.iloc[-1])
    e150 = float(ema150_s.iloc[-1])

    # EMA150 slope over last 20 bars (4 weeks)
    lb = min(20, len(ema150_s) - 1)
    e150_base = float(ema150_s.iloc[-lb - 1])
    e150_slope = (e150 - e150_base) / e150_base * 100 if e150_base > 0 else 0

    # EMA50 slope over last 10 bars
    lb50 = min(10, len(ema50_s) - 1)
    e50_base = float(ema50_s.iloc[-lb50 - 1])
    e50_slope = (e50 - e50_base) / e50_base * 100 if e50_base > 0 else 0

    above_150     = curr > e150
    above_50      = curr > e50
    e50_above_150 = e50 > e150
    e150_rising   = e150_slope > 0.3
    e150_falling  = e150_slope < -0.3

    period_len = min(252, len(df))
    h52 = float(high.rolling(period_len).max().iloc[-1])
    l52 = float(low.rolling(period_len).min().iloc[-1])
    dist_high = round((curr - h52) / h52 * 100, 1) if h52 > 0 else 0
    dist_low  = round((curr - l52) / l52 * 100, 1) if l52 > 0 else 0

    if above_150 and above_50 and e50_above_150 and e150_rising:
        stage = 2
        stage_label = "Stage 2 ▲"
        stage_desc  = "Advancing uptrend — preferred stage for entries"
    elif not above_150 and e150_falling:
        stage = 4
        stage_label = "Stage 4 ▼"
        stage_desc  = "Declining downtrend — avoid longs"
    elif above_150 and (not e150_rising or not e50_above_150):
        stage = 3
        stage_label = "Stage 3 ◆"
        stage_desc  = "Topping / distribution — reduce exposure"
    else:
        if dist_low > 25 and abs(e150_slope) < 0.5:
            stage = 1
            stage_label = "Stage 1★"
            stage_desc  = "Late base — EMA flattening, possible Stage 2 breakout forming"
        else:
            stage = 1
            stage_label = "Stage 1"
            stage_desc  = "Basing / accumulation — not ready yet"

    return {
        "stage":             stage,
        "stage_label":       stage_label,
        "stage_desc":        stage_desc,
        "ema150":            round(e150, 2),
        "ema150_slope_pct":  round(e150_slope, 3),
        "dist_52w_high":     dist_high,
        "dist_52w_low":      dist_low,
    }


# ── Pivot-based Horizontal S/R ────────────────────────────────

def compute_pivot_levels(df: pd.DataFrame, lookback: int = 60, n: int = 3) -> dict:
    """
    Find horizontal S/R from confirmed pivot highs and lows.
    breakout_level = nearest pivot high above current price (the trigger).
    key_support    = nearest pivot low below current price.
    """
    if df is None or len(df) < n * 2 + 5:
        return {}

    h    = df["high"].values
    l    = df["low"].values
    curr = float(df["close"].values[-1])

    start = max(n, len(df) - lookback)
    end   = len(df) - n  # last n bars are unconfirmed pivots

    pivot_highs = []
    pivot_lows  = []

    for i in range(start, end):
        if (all(h[i] >= h[i - j] for j in range(1, n + 1)) and
                all(h[i] >= h[i + j] for j in range(1, n + 1))):
            pivot_highs.append(round(float(h[i]), 2))
        if (all(l[i] <= l[i - j] for j in range(1, n + 1)) and
                all(l[i] <= l[i + j] for j in range(1, n + 1))):
            pivot_lows.append(round(float(l[i]), 2))

    # Nearest resistance at least 0.5% above price
    resistances = sorted([p for p in pivot_highs if p > curr * 1.005])
    breakout_level = resistances[0] if resistances else None

    # Nearest support at least 0.5% below price
    supports = sorted([p for p in pivot_lows if p < curr * 0.995], reverse=True)
    key_support = supports[0] if supports else None

    result = {}
    if breakout_level:
        result["breakout_level"]    = breakout_level
        result["breakout_dist_pct"] = round((breakout_level - curr) / curr * 100, 1)
    if key_support:
        result["key_support"] = key_support

    return result


# ── Setup Status (WATCH / ENTRY / AVOID) ─────────────────────

def compute_setup_status(result: dict) -> str:
    """
    Two-stage output replacing the binary GOOD SETUP / NOT SUITABLE.
    ENTRY  — trigger ready, conditions met, act now
    WATCH  — setup forming, wait for breakout confirmation
    AVOID  — downtrend or score too low
    """
    score          = result.get("score100", 0)
    stage          = result.get("stage", 0)
    breakout_watch = result.get("breakout_watch", False)
    breakout_level = result.get("breakout_level")
    close          = result.get("close", 0) or 0
    st_dir         = result.get("supertrend_dir", 0)
    rs             = result.get("rs_vs_nifty", 0) or 0

    # Already broken above pivot resistance with strong score
    if breakout_level and close > breakout_level * 1.005 and score >= 55:
        return "ENTRY"

    # Stage 2 advancing with solid score
    if stage == 2 and score >= 60:
        return "ENTRY"

    # VCP / base actively forming
    if breakout_watch:
        return "WATCH"

    # Late Stage 1 base with positive RS
    if stage == 1 and rs > 0 and score >= 40:
        return "WATCH"

    # Stage 2 but score building
    if stage == 2 and score >= 45:
        return "WATCH"

    # Clear downtrend
    if stage == 4 or (st_dir == -1 and rs < -5):
        return "AVOID"

    if score < 35:
        return "AVOID"

    return "WATCH"


# ── Trade levels (SL / Targets / R:R) ────────────────────────

def compute_trade_levels(result: dict) -> dict:
    """
    Compute stop loss, targets, and R:R.

    SL logic (structure-first, volatility-validated):
      ATR stop  = close − 1.5 × ATR
      If structural support exists below close:
          support SL = S1 × 0.995  (small buffer below widely-watched level)
          Final SL   = min(ATR stop, support SL)  — more breathing room
      Maximum stop distance cap: 2.5 × ATR.
          A stop wider than 2.5 ATR makes T1 unreachable in a normal 5-day move
          and inflates position sizes to unsafe levels.

    T1 = strict 2R  (consistent across all stocks, never adjusted to resistance)
    T2 = R2 pivot if > T1, else close + 3.5 × ATR
    """
    close   = result.get("close", 0)
    atr_val = result.get("atr", 0) or 0
    support = result.get("support")
    resist  = result.get("resistance")

    if atr_val <= 0:
        return result

    # Stop widened 1.5 -> 3.0 ATR, cap 2.5 -> 4.0 ATR (29 Aug 2026).
    #
    # Measured across 16,343 historical signals: a 1.5 ATR stop on a 5-day
    # clock was the WORST of 17 exit rules tested, returning -0.37% blended
    # against +1.82% for a 3.5 ATR trail held longer. Tight stops were being
    # triggered by ordinary volatility rather than by the thesis breaking —
    # trades exited during a normal wobble and forfeited the move. Average
    # favourable excursion was +4.5% while realised return was +0.17%, i.e.
    # roughly 4% of the correctly-identified move was actually captured.
    atr_sl = close - 3.0 * atr_val

    if support and support < close:
        support_sl = support * 0.99
        sl = min(atr_sl, support_sl)
    else:
        sl = atr_sl

    # Hard cap: stop cannot be more than 4.0 ATR from entry
    max_sl_distance = 4.0 * atr_val
    if (close - sl) > max_sl_distance:
        sl = close - max_sl_distance
        result["stop_capped"] = True   # auditable flag — tells scoring this happened

    sl   = round(sl, 2)
    risk = close - sl
    if risk <= 0:
        return result

    t1 = round(close + 2.0 * risk, 2)

    if resist and resist > close and resist < t1:
        result["resistance_blocks_target"] = True

    resist2 = result.get("resistance2")
    t2 = round(resist2, 2) if (resist2 and resist2 > t1) else round(close + 3.0 * risk, 2)
    t2 = max(t2, round(t1 + 0.01, 2))

    result["sl_buy"]   = sl
    result["target1"]  = t1
    result["target2"]  = t2
    result["rr_ratio"] = round((t1 - close) / risk, 2)
    return result


# ── Parabolic SAR ────────────────────────────────────────────

def parabolic_sar(high: pd.Series, low: pd.Series,
                  af_start: float = 0.02, af_step: float = 0.02, af_max: float = 0.2):
    """
    Returns (sar_series, signal_series) where signal: 1=price above SAR (bull), -1=below (bear).
    Standard Wilder Parabolic SAR.
    """
    h = high.to_numpy(float)
    l = low.to_numpy(float)
    n = len(h)

    sar    = np.empty(n)
    sig    = np.empty(n, dtype=int)
    ep     = h[0]          # extreme point
    af     = af_start
    bull   = True          # start bullish
    sar[0] = l[0]
    sig[0] = 1

    for i in range(1, n):
        if bull:
            sar[i] = sar[i-1] + af * (ep - sar[i-1])
            sar[i] = min(sar[i], l[i-1], l[max(0, i-2)])
            if l[i] < sar[i]:           # flip to bearish
                bull   = False
                sar[i] = ep
                ep     = l[i]
                af     = af_start
                sig[i] = -1
            else:
                sig[i] = 1
                if h[i] > ep:
                    ep = h[i]
                    af = min(af + af_step, af_max)
        else:
            sar[i] = sar[i-1] + af * (ep - sar[i-1])
            sar[i] = max(sar[i], h[i-1], h[max(0, i-2)])
            if h[i] > sar[i]:           # flip to bullish
                bull   = True
                sar[i] = ep
                ep     = h[i]
                af     = af_start
                sig[i] = 1
            else:
                sig[i] = -1
                if l[i] < ep:
                    ep = l[i]
                    af = min(af + af_step, af_max)

    return pd.Series(sar, index=high.index), pd.Series(sig, index=high.index)


# ── Ichimoku Cloud ────────────────────────────────────────────

def ichimoku(high: pd.Series, low: pd.Series, close: pd.Series,
             tenkan: int = 9, kijun: int = 26, senkou_b: int = 52):
    """
    Returns dict with all 5 Ichimoku components and a composite signal.
    Signal: BULLISH (price above cloud + TK bull), BEARISH, or NEUTRAL.
    """
    def midpoint(h, l, p): return (h.rolling(p).max() + l.rolling(p).min()) / 2

    tenkan_sen  = midpoint(high, low, tenkan)
    kijun_sen   = midpoint(high, low, kijun)
    senkou_a    = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
    senkou_b_s  = midpoint(high, low, senkou_b).shift(kijun)
    chikou_span = close.shift(-kijun)

    curr_close  = float(close.iloc[-1])
    t_val = float(tenkan_sen.iloc[-1]) if not np.isnan(tenkan_sen.iloc[-1]) else None
    k_val = float(kijun_sen.iloc[-1])  if not np.isnan(kijun_sen.iloc[-1])  else None
    a_val = float(senkou_a.iloc[-1])   if not np.isnan(senkou_a.iloc[-1])   else None
    b_val = float(senkou_b_s.iloc[-1]) if not np.isnan(senkou_b_s.iloc[-1]) else None

    cloud_top    = max(a_val, b_val) if a_val and b_val else None
    cloud_bottom = min(a_val, b_val) if a_val and b_val else None
    cloud_green  = (a_val > b_val)   if a_val and b_val else None  # green cloud = bullish

    above_cloud  = (curr_close > cloud_top)    if cloud_top    else None
    below_cloud  = (curr_close < cloud_bottom) if cloud_bottom else None
    tk_bull      = (t_val > k_val) if t_val and k_val else None
    tk_bear      = (t_val < k_val) if t_val and k_val else None

    if above_cloud and tk_bull and cloud_green:
        signal = "BULLISH"
    elif below_cloud and tk_bear and not cloud_green:
        signal = "BEARISH"
    elif above_cloud:
        signal = "WEAK_BULL"
    elif below_cloud:
        signal = "WEAK_BEAR"
    else:
        signal = "NEUTRAL"  # price inside cloud

    return {
        "ichimoku_signal":  signal,
        "ichimoku_tenkan":  round(t_val, 2) if t_val else None,
        "ichimoku_kijun":   round(k_val, 2) if k_val else None,
        "ichimoku_cloud_top":    round(cloud_top,    2) if cloud_top    else None,
        "ichimoku_cloud_bottom": round(cloud_bottom, 2) if cloud_bottom else None,
        "ichimoku_cloud_green":  bool(cloud_green) if cloud_green is not None else None,
    }


# ── Stochastic RSI ────────────────────────────────────────────

def stoch_rsi(close: pd.Series, rsi_period: int = 14, stoch_period: int = 14,
              k_smooth: int = 3, d_smooth: int = 3):
    """
    Faster than standard RSI — catches reversals 1-2 days earlier.
    Returns (K%, D%) both 0-100.
    """
    from signals import rsi
    rsi_s = rsi(close, rsi_period)

    rsi_min = rsi_s.rolling(stoch_period).min()
    rsi_max = rsi_s.rolling(stoch_period).max()
    stoch   = 100 * (rsi_s - rsi_min) / (rsi_max - rsi_min).replace(0, np.nan)
    k       = stoch.rolling(k_smooth).mean()
    d       = k.rolling(d_smooth).mean()
    return k, d


# ── Williams %R ───────────────────────────────────────────────

def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    """
    Range: -100 (oversold) to 0 (overbought).
    Below -80 = oversold (potential reversal up).
    Above -20 = overbought (potential reversal down).
    """
    hh = high.rolling(period).max()
    ll = low.rolling(period).min()
    return -100 * (hh - close) / (hh - ll).replace(0, np.nan)


# ── OBV (On-Balance Volume) ───────────────────────────────────

def obv_trend(close: pd.Series, volume: pd.Series, period: int = 20):
    """
    Returns (obv_series, trend_str).
    Trend: RISING (accumulation) | FALLING (distribution) | NEUTRAL.
    """
    direction  = np.sign(close.diff()).fillna(0)
    obv_series = (direction * volume).cumsum()
    obv_ema    = obv_series.ewm(span=period, adjust=False).mean()

    curr_obv   = float(obv_series.iloc[-1])
    curr_ema   = float(obv_ema.iloc[-1])
    prev_ema   = float(obv_ema.iloc[-2]) if len(obv_ema) > 1 else curr_ema

    if curr_obv > curr_ema and curr_ema > prev_ema:
        trend = "RISING"
    elif curr_obv < curr_ema and curr_ema < prev_ema:
        trend = "FALLING"
    else:
        trend = "NEUTRAL"

    return obv_series, trend


# ── Relative Strength vs Nifty ────────────────────────────────

def relative_strength_vs_nifty(stock_close: pd.Series,
                                nifty_close: pd.Series,
                                period: int = 63) -> float | None:
    """
    RS = stock return over period  −  Nifty return over same period.
    Positive = outperforming (bullish), Negative = underperforming.
    Period: 63 ≈ 3 months (standard for momentum screening).
    """
    try:
        sc = stock_close.dropna()
        nc = nifty_close.dropna()
        if len(sc) < period or len(nc) < period:
            return None
        s_ret = (float(sc.iloc[-1]) / float(sc.iloc[-period]) - 1) * 100
        n_ret = (float(nc.iloc[-1]) / float(nc.iloc[-period]) - 1) * 100
        return round(s_ret - n_ret, 2)
    except Exception:
        return None


def ibd_rs_rank(stock_close: pd.Series, nifty_close: pd.Series) -> float | None:
    """
    IBD-style composite RS score (0-100).
    Formula: 40% × 3M excess return + 20% × 6M + 20% × 9M + 20% × 12M
    vs Nifty as benchmark. Higher = stronger relative momentum.
    RS > 80 = Top tier. RS > 70 = Strong. RS > 50 = Average.
    """
    try:
        sc = stock_close.dropna()
        nc = nifty_close.dropna()
        if len(sc) < 63 or len(nc) < 63:
            return None

        def _excess(p):
            if len(sc) < p or len(nc) < p:
                return None
            s = (float(sc.iloc[-1]) / float(sc.iloc[-p]) - 1) * 100
            n = (float(nc.iloc[-1]) / float(nc.iloc[-p]) - 1) * 100
            return s - n

        r3  = _excess(63)
        r6  = _excess(126)
        r9  = _excess(189)
        r12 = _excess(252)

        weights = []
        if r3  is not None: weights.append((r3,  0.40))
        if r6  is not None: weights.append((r6,  0.20))
        if r9  is not None: weights.append((r9,  0.20))
        if r12 is not None: weights.append((r12, 0.20))

        if not weights:
            return None

        total_w = sum(w for _, w in weights)
        score = sum(v * w for v, w in weights) / total_w
        # Normalize to 0-100 using sigmoid-like mapping: score=0 → 50, score=±20 → ~70/30
        normalized = 50 + score * 1.25
        return round(max(0.0, min(100.0, normalized)), 1)
    except Exception:
        return None


def mansfield_rs(stock_close: pd.Series, nifty_close: pd.Series) -> float | None:
    """
    Mansfield Relative Strength: ATR-normalized position of RS line vs its 52W SMA.
    Positive = RS line above its long-term average (sustained outperformance).
    Negative = RS line below average (structural underperformance).
    A reading of 0.0 is the neutral boundary.
    """
    try:
        sc = stock_close.dropna()
        nc = nifty_close.dropna()
        n  = min(len(sc), len(nc))
        if n < 60:
            return None

        sc_aligned = sc.iloc[-n:]
        nc_aligned = nc.iloc[-n:]
        nc_norm    = nc_aligned / float(nc_aligned.iloc[0])
        rs_line    = sc_aligned / (nc_norm * float(sc_aligned.iloc[0]))

        sma52 = rs_line.rolling(52).mean()
        if sma52.isna().all():
            sma52 = rs_line.rolling(min(52, len(rs_line))).mean()

        rs_last  = float(rs_line.iloc[-1])
        sma_last = float(sma52.iloc[-1])
        if np.isnan(sma_last) or sma_last == 0:
            return None

        return round((rs_last - sma_last) / sma_last * 100, 2)
    except Exception:
        return None


# ── Weekly Trend Confirmation ─────────────────────────────────

def weekly_trend(weekly_df: pd.DataFrame) -> dict:
    """
    Confirms daily signal with weekly timeframe.
    A BUY signal on daily in a weekly downtrend has ~70% lower success rate.
    Returns: weekly_trend (BULLISH/BEARISH/NEUTRAL), weekly_supertrend.
    """
    if weekly_df is None or len(weekly_df) < 15:
        return {"weekly_trend": None, "weekly_supertrend": None}

    wdf = weekly_df.copy()
    wdf.columns = [c.lower() for c in wdf.columns]
    close  = wdf["close"]
    high   = wdf["high"]
    low    = wdf["low"]

    wema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
    curr   = float(close.iloc[-1])

    try:
        st_line, st_dir = supertrend(high, low, close, period=10, multiplier=3.0)
        w_st = int(st_dir.iloc[-1])
    except Exception:
        w_st = 0

    above_wema = curr > wema20
    if above_wema and w_st == 1:
        w_trend = "BULLISH"
    elif not above_wema and w_st == -1:
        w_trend = "BEARISH"
    elif above_wema:
        w_trend = "WEAK_BULL"
    else:
        w_trend = "WEAK_BEAR"

    return {
        "weekly_trend":      w_trend,
        "weekly_supertrend": w_st,
        "weekly_ema20":      round(wema20, 2),
    }


# ── Data Confidence (separate from Setup Quality) ─────────────

def compute_data_confidence(result: dict) -> str:
    """
    Assesses how complete the data inputs are.
    Returned as a label, NOT added to the numeric score.

    Rationale: a stock should not score higher just because Yahoo Finance
    happened to provide fundamentals. Data completeness affects trust in the
    score, not the quality of the setup itself.
    """
    deductions = 0
    if not result.get("rs_vs_nifty"):       deductions += 3
    if not result.get("ichimoku_signal"):   deductions += 2
    if not result.get("weekly_trend"):      deductions += 2
    if not result.get("adx"):              deductions += 1
    if result.get("fund_score") is None:   deductions += 3
    if not result.get("rr_ratio"):         deductions += 2
    if not result.get("obv_trend"):        deductions += 1
    if not result.get("supertrend_dir"):   deductions += 1

    if   deductions == 0:  return "HIGH"
    elif deductions <= 3:  return "MEDIUM"
    elif deductions <= 7:  return "LOW"
    else:                  return "INCOMPLETE"


# ── Regime-specific score modifiers ───────────────────────────
#
# Trend and momentum signals from individual stocks are less reliable
# when the broad market is in a bear or high-volatility regime.
# A bullish EMA stack during a BEAR_TREND is more often a bear-market
# rally than a new uptrend. These multipliers reflect that.
#
# Score caps prevent any setup from reaching "GOOD SETUP" or higher
# when the macro environment makes it structurally dangerous.

# Regime modifiers tell you HOW MUCH to trust a bullish trend signal given
# the macro environment. In a bear market a bullish EMA stack is less reliable
# than in a bull market — but it is still a signal worth showing.
# Previous values (0.25 / 0.30 for BEAR_TREND) were too punishing and made the
# dashboard useless in down markets. Adjusted to allow genuine outperformers
# (sector rotation: gold, power, pharma in bear markets) to surface.
_REGIME_TREND_MOD = {
    "BULL_TREND":     1.00,
    "BULL_WEAK":      0.90,
    "RECOVERY":       0.75,
    "BEAR_TREND":     0.55,
    "SIDEWAYS":       0.65,
    "HIGH_VOLATILITY":0.55,
    "UNKNOWN":        0.85,
}

_REGIME_MOM_MOD = {
    "BULL_TREND":     1.00,
    "BULL_WEAK":      0.92,
    "RECOVERY":       0.78,
    "BEAR_TREND":     0.60,
    "SIDEWAYS":       0.68,
    "HIGH_VOLATILITY":0.58,
    "UNKNOWN":        0.85,
}

# Score caps: the regime tells you the RISK CONTEXT, not whether a setup is valid.
# Raised significantly — a stock with genuine relative strength and bullish
# technicals deserves to show as GOOD SETUP even in a weak market.
# The "size down in bear market" message is shown via the regime label in the
# AI prompt and market banner — not by hiding valid signals from the user.
_REGIME_SCORE_CAP = {
    "BULL_TREND":     100,
    "BULL_WEAK":      92,
    "RECOVERY":       84,
    "BEAR_TREND":     72,
    "SIDEWAYS":       82,
    "HIGH_VOLATILITY":74,
    "UNKNOWN":        90,
}


# ── 0-100 Score ───────────────────────────────────────────────

# ── Momentum handling switch ────────────────────────────────────────────────
# v1 treats an extended, strongly-trending stock as RISKY: it docks points for
# trading far above the 50-day average and caps the momentum bucket once RSI
# passes 65. That is what scored Morepen 45 ("AVOID") before it rose 142% and
# capped Ather at 62 before it rose 66%.
#
# A smoke test on known winners vs controls found momentum and participation
# separated them cleanly while multi-year growth did not — which suggests the
# penalty is fighting the one factor that works. This switch exists so the
# alternatives can be MEASURED against realised returns rather than argued.
#
#   "penalise" — v1 behaviour: extension docked, RSI above 72 scores 1
#   "neutral"  — no extension penalty, RSI ceiling flattened
#   "reward"   — strong trends score higher, in line with momentum research
# Switched to "reward" 30 Aug 2026 on measured evidence.
#
# Tested across two universes (all_nse 757 stocks, nifty500 497 stocks),
# scoring each stock three ways and comparing against realised returns:
#   reward beat penalise on TOP-DECILE return in 6 of 6 tests,
#   improving the edge over the universe by ~+0.8 to +1.2 points.
#
#   top-decile edge over universe    penalise    reward
#     all_nse   5d / 21d / 63d      +0.08 +0.04 +1.84 | +1.27 +1.25 +2.44
#     nifty500  5d / 21d / 63d      -0.47 +1.68 -0.48 | -0.03 +2.42 +1.30
#
# The effect sits in the tail, which is why rank correlation barely moves
# (+0.121 -> +0.126) while top-decile performance improves sharply: the
# penalty was pushing the strongest trends OUT of the top decile. That is
# the mechanism that scored Morepen 45 before a 142% rise.
#
# LIMITATION: both scans are from late May, so their forward windows overlap
# the same market episode. Replicated across universes, NOT across time.
# Momentum is known to reverse violently in sharp corrections and neither
# sample contains one. Revert to "penalise" if live results diverge.
MOMENTUM_MODE = "reward"


def _math_isnan(v) -> bool:
    import math as _m
    try:
        return _m.isnan(float(v))
    except (TypeError, ValueError):
        return True


def compute_score100(result: dict) -> int:
    """
    measurement-v2-no-rr: R:R is excluded; full base capacity is 64.
    This remains an unvalidated technical research score, not a probability.
    The historical design notes below describe the preceding version.
    Regime-conditional 0-100 setup quality score. Zero double-counting.
    Data Confidence is NOT included here — it is a separate label
    (see compute_data_confidence). A stock should not score higher just
    because a data provider happened to return fundamentals.

    Seven buckets + VCP bonus (raw max = 80), scaled to 0-100, then:

    Trend/Momentum: regime-conditional modifiers. In BEAR_TREND the modifier
    depends on the stock's own EMA50 position and RS — not a blanket discount.

    Score cap: regime-dependent. BEAR_TREND default = 52 (no GOOD SETUP).
    Rises to 65 only under strict criteria: above EMA50, RS >+10%, weekly not
    bearish, volume confirming. Fundamentals gate can further reduce cap on
    severe red flags regardless of regime.

    Bucket                  Raw Max
    ──────────────────────────────────
    1  Trend Structure          15
    2  Momentum Quality         15
    3  Relative Strength        10   ← sector RS needed for full 15pt version
    4  Volume / Participation   10
    5  Risk-Reward Quality      15
    6  Market Regime             6   ← reduced: already in modifiers + cap
    7  Fundamental Quality       6   ← swing tool; fundamentals = risk gate
    8  VCP Bonus              0–3   ← tiered by squeeze quality
    ──────────────────────────────────
    Raw total                   80   → scaled to 0-100
    """
    # Determine regime for modifiers
    regime_obj = result.get("regime") or {}
    if isinstance(regime_obj, dict):
        regime_str = regime_obj.get("regime", "UNKNOWN")
    else:
        regime_str = "UNKNOWN"
    # Also check market context for VIX override
    market    = result.get("market") or {}
    vix_high  = market.get("vix_high", False)
    if vix_high and regime_str not in ("BEAR_TREND", "HIGH_VOLATILITY"):
        regime_str = "HIGH_VOLATILITY"

    close = result.get("close", 0) or 0
    e9    = result.get("ema9",  0) or 0
    e21   = result.get("ema21", 0) or 0
    e50   = result.get("ema50", 0) or 0
    adx_v = result.get("adx",   0) or 0
    rs_v  = result.get("rs_vs_nifty") or 0

    # ── Conditional BEAR_TREND modifiers ─────────────────────────
    # A counter-trend rally in a bear market is not the same as a stock
    # genuinely proving independence from the index. Modifiers split by
    # where the stock actually sits — not a blanket discount for all longs.
    if regime_str == "BEAR_TREND":
        above_ema50 = e50 > 0 and close >= e50
        if above_ema50 and rs_v > 0:
            # Stock holding above EMA50 AND outperforming Nifty — genuine leader
            trend_mod = 0.50
            mom_mod   = 0.55
        elif above_ema50:
            # Above EMA50 but not outperforming — possible bounce, partial credit
            trend_mod = 0.35
            mom_mod   = 0.40
        else:
            # Below EMA50 in a bear market — trend signal is mostly noise
            trend_mod = 0.25
            mom_mod   = 0.30
    else:
        trend_mod = _REGIME_TREND_MOD.get(regime_str, 0.80)
        mom_mod   = _REGIME_MOM_MOD.get(regime_str, 0.80)

    # ── Conditional BEAR_TREND cap ───────────────────────────────
    # Default stays 52 — GOOD SETUP (60+) is not valid in a bear market
    # unless the stock is proving genuine independence from the index.
    # Cap rises only under strict, specific conditions — not broadly.
    score_cap = _REGIME_SCORE_CAP.get(regime_str, 86)

    if regime_str == "BEAR_TREND":
        above_ema50  = e50 > 0 and close >= e50
        weekly_ok    = result.get("weekly_trend") not in ("BEARISH", "WEAK_BEAR")
        vol_confirm  = (result.get("vol_ratio") or 0) >= 1.2
        if above_ema50 and rs_v > 10 and weekly_ok and vol_confirm:
            # Strict exceptional leader: above EMA50, RS >10%, weekly not bearish,
            # volume confirming. These are sector rotation leaders, not bounces.
            score_cap = 65
        # Otherwise stays at 52 — bear market = bear market

    # ── 1. Trend Structure (0-15) ─────────────────────────────
    # EMA stack + slope freshness + Supertrend + weekly trend.
    # ADX < 18: ranging market — halve trend points (alignment is coincidental).
    # Extension check: if close >15% above EMA50, trend is likely exhausted.
    if e9 > e21 > e50 and close > e50:
        trend_pts = 7
    elif e9 > e21 > e50:
        trend_pts = 5
    elif e9 > e21:
        trend_pts = 2
    elif e9 < e21 < e50:
        trend_pts = -3
    else:
        trend_pts = 0

    # EMA slope bonus: both EMA21 and EMA50 rising = trend is gaining, not fading
    ema21_slope = result.get("ema21_slope", 0) or 0
    ema50_slope = result.get("ema50_slope", 0) or 0
    if ema21_slope > 0.05 and ema50_slope > 0.02:
        trend_pts += 2   # slopes confirm trend is accelerating

    # Freshness bonus: recently reclaimed EMA21 (1-5 days = fresh, 6-20 = sustained)
    days_above = result.get("days_above_ema21", 0) or 0
    if 1 <= days_above <= 5:
        trend_pts += 2   # fresh breakout above EMA21 = early entry
    elif 6 <= days_above <= 20:
        trend_pts += 1   # sustained but not exhausted

    # Extension penalty: >15% above EMA50 = likely overextended
    ema50_dist_pct = result.get("ema50_dist_pct", 0) or 0
    if MOMENTUM_MODE == "penalise":
        if ema50_dist_pct > 15:
            trend_pts -= 2
        elif ema50_dist_pct > 10:
            trend_pts -= 1
    elif MOMENTUM_MODE == "reward":
        # Distance above the mean is evidence the trend is working, not that
        # it is due to revert — the standard momentum-continuation reading.
        if ema50_dist_pct > 15:
            trend_pts += 2
        elif ema50_dist_pct > 10:
            trend_pts += 1
    # "neutral": extension neither helps nor hurts

    st_dir = result.get("supertrend_dir", 0)
    trend_pts += 5 if st_dir == 1 else (-4 if st_dir == -1 else 0)

    # Weekly trend contributes up to 3 of this bucket's 15. When it is UNKNOWN
    # (thin history — typically a recent listing) we must not silently score it
    # as neutral-bad; instead its 3 points leave the denominator entirely.
    wt = result.get("weekly_trend") or ""
    _wt_known = wt in ("BULLISH", "WEAK_BULL", "WEAK_BEAR", "BEARISH")
    trend_max = 15 if _wt_known else 12
    trend_pts += (3 if wt == "BULLISH" else
                  1 if wt == "WEAK_BULL" else
                 -1 if wt == "WEAK_BEAR" else
                 -3 if wt == "BEARISH" else 0)

    if 0 < adx_v < 18:
        trend_pts = int(trend_pts * 0.5)

    trend_pts = max(0, min(trend_max, int(trend_pts * trend_mod)))

    # ── 2. Momentum Quality (0-15) ────────────────────────────
    # RSI swing zones + MACD (must be RISING, not just positive) +
    # Ichimoku + ADX directional confirmation.
    # Stochastic / Williams %R / CCI / StochRSI excluded — co-linear with RSI.
    rsi_v = result.get("rsi", 50) or 50
    if MOMENTUM_MODE == "penalise":
        if   55 <= rsi_v <= 65:  mom_pts = 6   # ideal swing zone
        elif 45 <= rsi_v <  55:  mom_pts = 5   # building momentum
        elif 65 <  rsi_v <= 72:  mom_pts = 3   # extended but not exhausted
        elif 35 <= rsi_v <  45:  mom_pts = 3   # recovering
        elif rsi_v > 72:         mom_pts = 1   # overheated
        else:                    mom_pts = 1   # oversold alone ≠ bullish
    else:
        # Flatten the ceiling: a strong RSI is strength, not exhaustion. In
        # "reward" mode the strongest readings score highest.
        top = 7 if MOMENTUM_MODE == "reward" else 6
        if   rsi_v > 72:         mom_pts = top
        elif 65 <  rsi_v <= 72:  mom_pts = 6
        elif 55 <= rsi_v <= 65:  mom_pts = 6
        elif 45 <= rsi_v <  55:  mom_pts = 4
        elif 35 <= rsi_v <  45:  mom_pts = 2
        else:                    mom_pts = 1

    # MACD: fresh/rising histogram > stale positive histogram
    # A declining positive MACD histogram signals momentum fading — do not reward equally.
    macd_h      = result.get("macd_hist",      0) or 0
    macd_h_prev = result.get("macd_hist_prev", 0) or 0
    if macd_h > 0 and macd_h > macd_h_prev:
        mom_pts += 3   # rising positive histogram — genuine momentum
    elif macd_h > 0:
        mom_pts += 1   # positive but declining — stale, partial credit only

    # Ichimoku contributes up to 4 of this bucket's 15 — same treatment as
    # weekly trend: unknown drops out of the denominator rather than scoring 0.
    ichi = result.get("ichimoku_signal") or ""
    _ichi_known = ichi in ("BULLISH", "WEAK_BULL", "WEAK_BEAR", "BEARISH")
    mom_max = 15 if _ichi_known else 11
    mom_pts += (4 if ichi == "BULLISH" else
                2 if ichi == "WEAK_BULL" else
               -2 if ichi == "BEARISH" else
               -1 if ichi == "WEAK_BEAR" else 0)

    if adx_v > 25:
        plus_di  = result.get("adx_plus_di",  0) or 0
        minus_di = result.get("adx_minus_di", 0) or 0
        mom_pts += 2 if plus_di > minus_di else (-2 if minus_di > plus_di else 0)

    mom_pts = max(0, min(mom_max, int(mom_pts * mom_mod)))

    # ── 3. Relative Strength vs Nifty (0-10) ────────────────────
    # No regime modifier — outperforming in a bear market is genuine signal.
    # Kept at 10 pts (not 15) because the full split (stock vs sector + sector
    # vs Nifty + RS consistency = 15) requires sector RS data we don't have yet.
    # Do not increase weight until sector RS is available.
    # `or 0` used to collapse UNKNOWN into the "slightly underperforming" band,
    # so a stock with no RS reading scored identically to one genuinely lagging
    # the index by 4%. That penalised recent listings hardest — exactly the
    # high-momentum names this scanner exists to find. Unknown now leaves the
    # denominator instead of being scored as weakness.
    _rs_raw = result.get("rs_vs_nifty")
    _rs_known = isinstance(_rs_raw, (int, float)) and not _math_isnan(_rs_raw)
    rs_max = 10 if _rs_known else 0
    if not _rs_known:
        rs_pts = 0
    else:
        rs = float(_rs_raw)
        if   rs > 10:  rs_pts = 10
        elif rs > 5:   rs_pts = 8
        elif rs > 0:   rs_pts = 5
        elif rs > -5:  rs_pts = 3
        elif rs > -10: rs_pts = 1
        else:          rs_pts = 0
    rs_pts = max(0, min(10, rs_pts))

    # ── 4. Volume & Participation (0-10) ─────────────────────
    vol   = result.get("vol_ratio", 1.0) or 1.0
    obv_t = result.get("obv_trend", "")

    if   vol >= 2.0: vol_pts = 6
    elif vol >= 1.5: vol_pts = 5
    elif vol >= 1.2: vol_pts = 4
    elif vol >= 0.8: vol_pts = 2
    elif result.get("breakout_watch"):
        vol_pts = 3   # VCP: quiet volume is a feature, not a flaw
    else:            vol_pts = 1

    vol_pts += (3 if obv_t == "RISING" else 1 if obv_t == "NEUTRAL" else 0)
    # Delivery % is more reliable than OBV — bonus if available and meaningful
    delv = result.get("delivery_pct")
    if delv and delv > 55:
        vol_pts += 1   # high delivery = genuine buy conviction
    vol_pts = max(0, min(10, vol_pts))

    # ── 5. Risk-Reward Quality (0-15) ────────────────────────
    rr    = result.get("rr_ratio", 0) or 0
    sl    = result.get("sl_buy") or 0
    atr_v = result.get("atr", 0) or 0

    if   rr >= 3.0: rr_pts = 10
    elif rr >= 2.5: rr_pts = 8
    elif rr >= 2.0: rr_pts = 6
    elif rr >= 1.5: rr_pts = 3
    else:           rr_pts = 0

    if sl > 0 and close > 0 and atr_v > 0:
        sl_dist_pct = (close - sl) / close * 100
        atr_pct     = atr_v / close * 100
        ratio       = sl_dist_pct / atr_pct if atr_pct > 0 else 1.0
        rr_pts += (5 if ratio >= 1.2 else 3 if ratio >= 0.8 else 1 if ratio >= 0.5 else 0)

    if result.get("resistance_blocks_target"):
        rr_pts = max(0, rr_pts - 3)

    # If stop was capped at 2.5 ATR, the trade is marginal — small penalty
    if result.get("stop_capped"):
        rr_pts = max(0, rr_pts - 2)

    rr_pts = max(0, min(15, rr_pts))

    # ── Participant OI signal (institutional futures positioning) ─
    # FII net long index futures = institutional bullish bias → +1 to regime pts
    # Applied before regime bucket so it feeds into market context section
    poi = result.get("participant_oi", {}) or {}
    fii_fut_sig = poi.get("fii_futures_signal", "")
    retail_sig  = poi.get("retail_contrarian_signal", "")
    poi_bonus = 0
    if fii_fut_sig == "BULLISH":             poi_bonus += 1
    elif fii_fut_sig == "BEARISH":           poi_bonus -= 1
    if retail_sig == "BUY":                  poi_bonus += 1   # retail heavily short = contrarian buy
    elif retail_sig == "SELL":               poi_bonus -= 1
    poi_bonus = max(-1, min(1, poi_bonus))

    # ── Bulk/Block deal bonus ─────────────────────────────────
    bulk_bonus = 1 if result.get("bulk_deals") else 0

    # ── 6. Market Regime (0-10) ───────────────────────────────
    above_200 = market.get("nifty_above_200ema")
    above_50  = market.get("nifty_above_50ema")

    if above_200 is not None:
        if   above_200 and above_50:   regime_pts = 8
        elif above_200:                regime_pts = 5
        elif above_50:                 regime_pts = 3
        else:                          regime_pts = 1
    else:
        regime_pts = 5

    if vix_high:
        regime_pts = max(0, regime_pts - 2)
    regime_pts = max(0, min(6, regime_pts))   # capped at 6 — already in modifiers + cap

    # ── 7. Fundamental Quality (0-6) ─────────────────────────
    # Swing trading is 3-15 days — price action drives outcomes, not P/E ratios.
    # Reduced from 10 to 6 pts. Still rewards good fundamentals without
    # penalising the many NSE stocks where yfinance returns no data.
    fund_score = result.get("fund_score")
    import math as _math
    _fs_valid  = fund_score is not None and isinstance(fund_score, (int, float)) and not _math.isnan(fund_score)
    fund_pts   = round(fund_score / 100 * 6) if _fs_valid else 3   # neutral default
    fund_pts   = max(0, min(6, fund_pts))

    # ── VCP / Breakout bonus (0-2, tiered by quality) ────────────
    # Relative strength is a GATE here, not a point. It already scores up to 10
    # in bucket 3 and gates the BEAR_TREND cap; awarding another point for
    # rs_v > 0 counted the same variable a third time. A stock with negative RS
    # simply does not qualify for the VCP bonus at all.
    vcp_bonus = 0
    if result.get("breakout_watch") and rs_v > 0:
        vcp_bonus += 1                                      # squeeze detected
        if result.get("vol_dryup"):
            vcp_bonus += 1                                  # volume actually drying up
    vcp_bonus = min(vcp_bonus, 2)

    # ── Stage bonus / penalty (Minervini Stage Analysis) ─────────
    # Stage 2 = advancing uptrend = structurally correct entry zone.
    # Stage 4 = declining = hard cap even with good technicals.
    stage = result.get("stage", 0)
    stage_bonus = 0
    if stage == 2:
        # Confirmed Stage 2: EMA50 > EMA150, EMA150 rising, price above both
        stage_bonus = 2
    elif stage == 3:
        stage_bonus = -1     # topping/distribution
        score_cap = min(score_cap, 65)
    elif stage == 4:
        stage_bonus = -3     # declining
        score_cap = min(score_cap, 50)
    stage_bonus = max(-3, min(2, stage_bonus))

    # ── IBD RS Rank bonus ────────────────────────────────────────
    # Already counted in RS bucket but top-tier RS (>80) gets an extra point
    ibd_rs_val = result.get("ibd_rs", 0) or 0
    ibd_bonus = 1 if ibd_rs_val >= 80 else 0

    # ── Screener quality gate ────────────────────────────────────
    # Promoter pledging is a risk, not a scoring input — it caps the score.
    # High ROCE gets a small bonus as it validates fundamental quality.
    pledged = result.get("promoter_pledged", 0) or 0
    screener_qs = result.get("quality_score")     # 0-10 from screener_client
    screener_bonus = 0
    if screener_qs is not None:
        if screener_qs >= 8:
            screener_bonus = 2    # high quality: ROCE>15, low D/E, FII buying, zero pledge
        elif screener_qs >= 6:
            screener_bonus = 1
        elif screener_qs <= 2:
            screener_bonus = -1   # multiple red flags
    if pledged > 50:
        score_cap = min(score_cap, 48)    # extremely high pledge = structural danger
    elif pledged > 30:
        score_cap = min(score_cap, 58)
    elif pledged > 20:
        score_cap = min(score_cap, 68)
    screener_bonus = max(-1, min(2, screener_bonus))

    # ── OI Spurt bonus ───────────────────────────────────────────
    # Stock showing unusual open interest buildup = smart money entering
    oi_bonus = 1 if result.get("oi_spurt") else 0

    # ── ML probability bonus — DISABLED 29 Aug 2026 ──────────────
    # The LightGBM model was measured at test AUC 0.529 / validation AUC 0.537
    # against 0.500 for a coin flip, i.e. no demonstrable ranking ability. Its
    # feature importance is 71.4% concentrated on atr_pct, meaning it largely
    # learned that volatile stocks touch a fixed ±2% band more often — an
    # arithmetic property of the label, not a predictive signal.
    # Granting it up to +3 points was injecting noise into the score.
    # `ml_prob` is still computed and displayed; it just no longer scores.
    # Re-enable ONLY if a retrained model clears AUC ~0.55 out-of-sample.
    ml_bonus = 0

    # ── Fundamentals risk gate ────────────────────────────────────
    # Fundamentals don't boost the score — they cap it on severe red flags.
    # Purpose: prevent a technically clean chart on a fundamentally dangerous
    # stock from reaching GOOD SETUP for a beginner.
    if _fs_valid:
        # Lenders carry leverage as their business model, not as distress.
        # A bank or NBFC routinely runs 6-10x debt/equity (BAJFINANCE reports
        # ~315), so a blanket ">3.0 = danger" rule fired on every financial in
        # the universe and capped their scores regardless of technicals —
        # suppressing a large share of the Nifty 50 by construction.
        _sec = f"{result.get('sector') or ''} {result.get('industry') or ''}".lower()
        _is_financial = any(k in _sec for k in
                            ("financial", "bank", "insurance", "capital markets", "credit"))

        debt_eq  = result.get("debt_equity")
        rev_g    = result.get("revenue_growth")
        _debt_high   = (not _is_financial
                        and isinstance(debt_eq, (int, float))
                        and not _math.isnan(debt_eq) and debt_eq > 3.0)
        _loss_making = fund_score < 20
        _rev_shrink  = isinstance(rev_g, (int, float)) and not _math.isnan(rev_g) and rev_g < -10
        if _loss_making and _debt_high:
            # Worst case: losing money + heavily in debt — cap at low WATCHLIST
            score_cap = min(score_cap, 52)
        elif _loss_making and _rev_shrink:
            # Losing money + shrinking revenue — cap at WATCHLIST
            score_cap = min(score_cap, 57)
        elif _loss_making:
            # Loss-making but debt/revenue not extreme — allow GOOD SETUP with strong technicals
            score_cap = min(score_cap, 63)

    # ── Final: scale raw → 0-100, apply all caps ─────────────────
    # Base buckets max = 79 (trend15 + mom15 + rs10 + vol10 + rr15 + regime6 + fund6 + vcp2)
    #   VCP dropped 3→2 when relative strength was demoted from a point to a gate.
    # Bonus pool max = +8 / min = -5  (stage+2, ibd+1, screener+2, oi+1, poi+1, bulk+1;
    #   ML removed — measured AUC 0.529, no edge).
    # NOTE: base points are rescaled by 100/79 (worth ~1.27 each) while bonuses
    # are added unscaled (worth 1.0). That asymmetry is inherited, not intended —
    # revisit once the ablation test shows which terms actually earn their weight.
    # ── Renormalise over the evidence that actually exists ───────────────
    # Buckets whose inputs are UNKNOWN contribute neither points nor capacity,
    # so a stock is no longer punished for data we failed to fetch. The full
    # denominator is 79; it shrinks as inputs go missing.
    # Version measurement-v2-no-rr: constructed 2R targets are diagnostics,
    # not independent evidence. Remove the entire R:R bucket and capacity.
    _FULL_MAX = 64
    result["score_version"] = "measurement-v2-no-rr"
    base_max = (trend_max + mom_max + rs_max + 10   # vol
                + 6 + 6 + 2)                         # regime, fund, vcp
    base = trend_pts + mom_pts + rs_pts + vol_pts + regime_pts + fund_pts + vcp_bonus
    base = max(0, base)
    if _math.isnan(base) or _math.isinf(base) or base_max <= 0:
        base, base_max = 0, _FULL_MAX
    scaled_base = int(round(base / base_max * 100))

    # ── Coverage guard ────────────────────────────────────────────────────
    # Renormalising alone would let a stock with almost no data score highly on
    # the two buckets it happens to have. Confidence must track evidence: the
    # thinner the coverage, the lower the ceiling. A 90 earned on 55% of the
    # inputs is not a 90.
    coverage = base_max / _FULL_MAX
    if   coverage < 0.70: score_cap = min(score_cap, 58)   # WATCHLIST at best
    elif coverage < 0.85: score_cap = min(score_cap, 72)   # no HIGH-QUALITY call
    result["score_coverage_pct"] = round(coverage * 100)
    bonus_total = stage_bonus + ibd_bonus + screener_bonus + oi_bonus + ml_bonus + poi_bonus + bulk_bonus
    scaled = scaled_base + bonus_total
    return max(0, min(score_cap, min(100, scaled)))


def compute_verdict(score100: int) -> str:
    """Safer, non-advisory verdict labels."""
    if   score100 >= 75: return "HIGH-QUALITY SETUP"
    elif score100 >= 60: return "GOOD SETUP"
    elif score100 <= 30: return "NOT SUITABLE"
    elif score100 <= 45: return "WEAK / AVOID"
    else:                return "WATCHLIST"


# ── Main enrichment entry point ───────────────────────────────

def compute_enhanced(df: pd.DataFrame, base_result: dict,
                     is_intraday: bool = False,
                     weekly_df: pd.DataFrame | None = None,
                     nifty_close: pd.Series | None = None) -> dict:
    """
    Receives signals.compute_signals() output and raw OHLCV dataframe.
    Adds Supertrend, CCI, VWAP, CPR, ADX, Stochastic, patterns, S/R,
    better SL/Target/R:R, and 0-100 score.
    """
    if df is None or len(df) < 20:
        return base_result

    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    if "open" not in df.columns:
        df["open"] = df["close"].shift(1).fillna(df["close"])

    close, high, low = df["close"], df["high"], df["low"]

    # ── EMA slope, freshness, extension (pre-computed for score) ─
    try:
        close_s = close  # alias for clarity
        ema21_s = close_s.ewm(span=21, adjust=False).mean()
        ema50_s = close_s.ewm(span=50, adjust=False).mean()

        if len(ema21_s) >= 6:
            prev21 = float(ema21_s.iloc[-6])
            curr21 = float(ema21_s.iloc[-1])
            base_result["ema21_slope"] = round((curr21 - prev21) / prev21 * 100, 3) if prev21 else 0

        if len(ema50_s) >= 6:
            prev50 = float(ema50_s.iloc[-6])
            curr50 = float(ema50_s.iloc[-1])
            base_result["ema50_slope"] = round((curr50 - prev50) / prev50 * 100, 3) if prev50 else 0
            ema50_last = float(ema50_s.iloc[-1])
            base_result["ema50_dist_pct"] = round(
                (float(close_s.iloc[-1]) - ema50_last) / ema50_last * 100, 2
            ) if ema50_last else 0

        # Consecutive days close > EMA21 (up to 30 bars back)
        days_above = 0
        for k in range(len(df) - 1, max(0, len(df) - 31) - 1, -1):
            if float(close_s.iloc[k]) > float(ema21_s.iloc[k]):
                days_above += 1
            else:
                break
        base_result["days_above_ema21"] = days_above
    except Exception:
        pass

    # ── MACD previous histogram (for rising-vs-stale distinction) ─
    try:
        from signals import macd as _macd_fn
        _, _, hist_s = _macd_fn(close)
        if len(hist_s) >= 2 and not np.isnan(float(hist_s.iloc[-2])):
            base_result["macd_hist_prev"] = round(float(hist_s.iloc[-2]), 4)
    except Exception:
        pass

    # ── Candlestick patterns ──────────────────────────────────
    base_result["patterns"] = detect_patterns(df)

    # ── Pivot S/R + CPR ──────────────────────────────────────
    sr = pivot_support_resistance(df)
    base_result.update(sr)

    # ── Supertrend ────────────────────────────────────────────
    try:
        if len(df) >= 15:
            st_line, st_dir = supertrend(high, low, close, period=10, multiplier=3.0)
            st_val  = float(st_line.iloc[-1])
            dir_val = int(st_dir.iloc[-1])
            prev_dir = int(st_dir.iloc[-2]) if len(st_dir) > 1 else dir_val
            if not np.isnan(st_val):
                base_result["supertrend_val"] = round(st_val, 2)
                base_result["supertrend_dir"] = dir_val  # 1=bullish, -1=bearish
                base_result["supertrend_label"] = "BUY ▲" if dir_val == 1 else "SELL ▼"
                # Freshly flipped signal
                if dir_val == 1 and prev_dir == -1:
                    base_result["signals"].append("Supertrend flipped BULLISH — strong buy signal")
                    base_result["score"] += 2
                elif dir_val == -1 and prev_dir == 1:
                    base_result["signals"].append("Supertrend flipped BEARISH — exit / short signal")
                    base_result["score"] -= 2
                elif dir_val == 1:
                    base_result["signals"].append(f"Supertrend BUY (support at ₹{st_val:.2f})")
                    base_result["score"] += 1
                else:
                    base_result["signals"].append(f"Supertrend SELL (resistance at ₹{st_val:.2f})")
                    base_result["score"] -= 1
    except Exception:
        pass

    # ── CCI ───────────────────────────────────────────────────
    try:
        if len(df) >= 20:
            cci_s = cci(high, low, close, period=20)
            cci_val = float(cci_s.iloc[-1])
            if not np.isnan(cci_val):
                base_result["cci"] = round(cci_val, 1)
                if cci_val > 100:
                    base_result["signals"].append(f"CCI Overbought ({cci_val:.0f}) — possible reversal")
                    base_result["score"] -= 1
                elif cci_val < -100:
                    base_result["signals"].append(f"CCI Oversold ({cci_val:.0f}) — potential bounce")
                    base_result["score"] += 1
    except Exception:
        pass

    # ── VWAP (intraday only) ──────────────────────────────────
    if is_intraday and "volume" in df.columns:
        try:
            vwap_s = vwap(high, low, close, df["volume"])
            vwap_val = float(vwap_s.iloc[-1])
            if not np.isnan(vwap_val):
                base_result["vwap"] = round(vwap_val, 2)
                if close.iloc[-1] > vwap_val:
                    base_result["signals"].append(f"Price above VWAP ₹{vwap_val:.2f} — bullish bias")
                    base_result["score"] += 1
                else:
                    base_result["signals"].append(f"Price below VWAP ₹{vwap_val:.2f} — bearish bias")
                    base_result["score"] -= 1
        except Exception:
            pass

    # ── ADX ───────────────────────────────────────────────────
    try:
        adx_s, plus_di, minus_di = adx(high, low, close, 14)
        adx_val  = float(adx_s.iloc[-1])
        plus_val = float(plus_di.iloc[-1])
        minus_val = float(minus_di.iloc[-1])
        if not (np.isnan(adx_val) or np.isnan(plus_val)):
            base_result["adx"] = round(adx_val, 1)
            base_result["adx_plus_di"]  = round(plus_val, 1)
            base_result["adx_minus_di"] = round(minus_val, 1)
            if adx_val > 25 and plus_val > minus_val:
                base_result["signals"].append(f"ADX {adx_val:.0f} — Strong Uptrend Confirmed")
                base_result["score"] += 1
            elif adx_val > 25 and minus_val > plus_val:
                base_result["signals"].append(f"ADX {adx_val:.0f} — Strong Downtrend Confirmed")
                base_result["score"] -= 1
    except Exception:
        pass

    # ── Stochastic ────────────────────────────────────────────
    try:
        k, d = stochastic(high, low, close, 14, 3)
        k_val, d_val = float(k.iloc[-1]), float(d.iloc[-1])
        k_prev = float(k.iloc[-2])
        if not np.isnan(k_val):
            base_result["stoch_k"] = round(k_val, 1)
            base_result["stoch_d"] = round(d_val, 1) if not np.isnan(d_val) else None
            if k_val < 20:
                base_result["signals"].append(f"Stochastic Oversold ({k_val:.0f}) — Bullish")
                base_result["score"] += 1
            elif k_val > 80:
                base_result["signals"].append(f"Stochastic Overbought ({k_val:.0f}) — Bearish")
                base_result["score"] -= 1
            if k_prev < d_val and k_val > d_val and k_val < 50:
                base_result["signals"].append("Stochastic Bullish Crossover in oversold zone")
                base_result["score"] += 1
    except Exception:
        pass

    # ── 52W High / Low from available data ───────────────────
    try:
        period_len = min(252, len(df))
        high_52w = float(high.rolling(period_len).max().iloc[-1])
        low_52w  = float(low.rolling(period_len).min().iloc[-1])
        curr     = float(close.iloc[-1])
        if not np.isnan(high_52w) and high_52w > 0:
            near_high_pct = round((curr - high_52w) / high_52w * 100, 1)
            near_low_pct  = round((curr - low_52w)  / low_52w  * 100, 1)
            base_result["high_52w"]      = round(high_52w, 2)
            base_result["low_52w"]       = round(low_52w, 2)
            base_result["near_high_pct"] = near_high_pct
            base_result["near_low_pct"]  = near_low_pct
            if near_high_pct > -3:
                base_result["signals"].append(f"Near 6M High ({near_high_pct:+.1f}%) — breakout watch")
            if near_low_pct < 5:
                base_result["signals"].append(f"Near 6M Low ({near_low_pct:+.1f}%) — support zone")
    except Exception:
        pass

    # ── Candlestick pattern scoring ───────────────────────────
    for p in base_result["patterns"]:
        p_lower = p.lower()
        if any(w in p_lower for w in ["bullish", "morning star", "hammer"]):
            base_result["score"] += 1
        elif any(w in p_lower for w in ["bearish", "evening star", "shooting"]):
            base_result["score"] -= 1

    # ── Parabolic SAR ─────────────────────────────────────────
    try:
        if len(df) >= 10:
            psar_s, psar_dir = parabolic_sar(high, low)
            psar_val = float(psar_s.iloc[-1])
            psar_sig_val = int(psar_dir.iloc[-1])
            psar_prev    = int(psar_dir.iloc[-2]) if len(psar_dir) > 1 else psar_sig_val
            if not np.isnan(psar_val):
                base_result["psar_val"]    = round(psar_val, 2)
                base_result["psar_signal"] = "BUY" if psar_sig_val == 1 else "SELL"
                base_result["psar_distance_pct"] = round(
                    (float(close.iloc[-1]) - psar_val) / psar_val * 100, 2
                )
                if psar_sig_val == 1 and psar_prev == -1:
                    base_result["signals"].append("Parabolic SAR flipped BUY — trend reversal up")
                    base_result["score"] += 2
                elif psar_sig_val == -1 and psar_prev == 1:
                    base_result["signals"].append("Parabolic SAR flipped SELL — trend reversal down")
                    base_result["score"] -= 2
                elif psar_sig_val == 1:
                    base_result["signals"].append(f"Parabolic SAR BUY (SAR support ₹{psar_val:.2f})")
                else:
                    base_result["signals"].append(f"Parabolic SAR SELL (SAR resistance ₹{psar_val:.2f})")
    except Exception:
        pass

    # ── Ichimoku Cloud ────────────────────────────────────────
    try:
        if len(df) >= 52:
            ichi = ichimoku(high, low, close)
            base_result.update(ichi)
            sig = ichi.get("ichimoku_signal", "")
            if sig == "BULLISH":
                base_result["signals"].append("Ichimoku: Price above green cloud + TK bullish cross")
                base_result["score"] += 2
            elif sig == "WEAK_BULL":
                base_result["signals"].append("Ichimoku: Price above cloud (weak bull)")
                base_result["score"] += 1
            elif sig == "BEARISH":
                base_result["signals"].append("Ichimoku: Price below red cloud + TK bearish cross")
                base_result["score"] -= 2
            elif sig == "WEAK_BEAR":
                base_result["signals"].append("Ichimoku: Price below cloud (weak bear)")
                base_result["score"] -= 1
            else:
                base_result["signals"].append("Ichimoku: Price inside cloud — indecision zone")
    except Exception:
        pass

    # ── Stochastic RSI ────────────────────────────────────────
    try:
        if len(df) >= 30:
            srsi_k, srsi_d = stoch_rsi(close)
            sk_val = float(srsi_k.iloc[-1])
            sd_val = float(srsi_d.iloc[-1])
            sk_prev = float(srsi_k.iloc[-2]) if len(srsi_k) > 1 else sk_val
            if not np.isnan(sk_val):
                base_result["stoch_rsi_k"] = round(sk_val, 1)
                base_result["stoch_rsi_d"] = round(sd_val, 1) if not np.isnan(sd_val) else None
                if sk_val < 20:
                    base_result["signals"].append(f"Stoch RSI Oversold ({sk_val:.0f}) — fast reversal signal")
                    base_result["score"] += 1
                elif sk_val > 80:
                    base_result["signals"].append(f"Stoch RSI Overbought ({sk_val:.0f}) — caution")
                    base_result["score"] -= 1
                if sk_prev < sd_val and sk_val > sd_val and sk_val < 50:
                    base_result["signals"].append("Stoch RSI bullish K/D crossover in oversold zone")
                    base_result["score"] += 1
    except Exception:
        pass

    # ── Williams %R ───────────────────────────────────────────
    try:
        if len(df) >= 14:
            wr_s = williams_r(high, low, close, 14)
            wr_val = float(wr_s.iloc[-1])
            if not np.isnan(wr_val):
                base_result["williams_r"] = round(wr_val, 1)
                if wr_val < -80:
                    base_result["signals"].append(f"Williams %R Oversold ({wr_val:.0f}) — bullish reversal zone")
                    base_result["score"] += 1
                elif wr_val > -20:
                    base_result["signals"].append(f"Williams %R Overbought ({wr_val:.0f}) — bearish reversal zone")
                    base_result["score"] -= 1
    except Exception:
        pass

    # ── OBV ───────────────────────────────────────────────────
    try:
        if "volume" in df.columns and len(df) >= 20:
            obv_s, obv_t = obv_trend(close, df["volume"])
            base_result["obv_trend"] = obv_t
            if obv_t == "RISING":
                base_result["signals"].append("OBV Rising — institutional accumulation detected")
                base_result["score"] += 1
            elif obv_t == "FALLING":
                base_result["signals"].append("OBV Falling — distribution / selling pressure")
                base_result["score"] -= 1
    except Exception:
        pass

    # ── ATR Squeeze + Volume Dry-up → VCP / Pre-Breakout Setup ──
    # Volatility Contraction Pattern (Minervini): price near highs,
    # daily range contracting (ATR falling), volume going quiet.
    # These three together mean a move is being coiled, not fading.
    try:
        if "volume" in df.columns and len(df) >= 25:
            from signals import atr as _atr_fn
            atr_series  = _atr_fn(high, low, close, 14)
            atr_cur     = float(atr_series.iloc[-1])
            atr_avg20   = float(atr_series.rolling(20).mean().iloc[-1])

            if not np.isnan(atr_cur) and not np.isnan(atr_avg20) and atr_avg20 > 0:
                atr_ratio = round(atr_cur / atr_avg20, 2)
                base_result["atr_ratio"] = atr_ratio
                atr_squeezing = atr_ratio < 0.85

                vol_5d    = float(df["volume"].iloc[-5:].mean())
                vol_avg20 = float(df["volume"].rolling(20).mean().iloc[-1])
                vol_dryup = (not np.isnan(vol_5d) and not np.isnan(vol_avg20)
                             and vol_avg20 > 0 and vol_5d < vol_avg20 * 0.80)
                base_result["vol_dryup"] = vol_dryup

                # Not in freefall: within 20% of 6M high OR above EMA50
                # (avoids flagging stocks in a straight downtrend)
                near_high  = base_result.get("near_high_pct")
                ema50_v    = base_result.get("ema50", 0) or 0
                close_v    = base_result.get("close", 0) or 0
                not_freefall = (
                    (near_high is not None and near_high > -20) or
                    (ema50_v > 0 and close_v >= ema50_v * 0.97)
                )

                obv_rising = base_result.get("obv_trend") == "RISING"
                obv_ok     = base_result.get("obv_trend") in ("RISING", "NEUTRAL")

                # Reject clear downtrends — Supertrend SELL + negative RS
                # = stock grinding lower, not building a base before a move
                st_dir = base_result.get("supertrend_dir", 0)
                rs_val = base_result.get("rs_vs_nifty", 0) or 0
                not_downtrend = not (st_dir == -1 and rs_val < 0)

                if not_freefall and not_downtrend and atr_squeezing and (vol_dryup or obv_rising):
                    base_result["breakout_watch"] = True
                    squeeze_str = f"ATR at {atr_ratio:.2f}x avg"
                    high_ref = base_result.get("high_52w")
                    high_str = f"₹{high_ref}" if high_ref else "6M high"
                    if vol_dryup:
                        base_result["signals"].append(
                            f"VCP / Pre-Breakout Setup — range squeezing ({squeeze_str}), "
                            f"volume quiet ({vol_5d/vol_avg20:.1f}x avg) — watch for breakout above {high_str}"
                        )
                    else:
                        base_result["signals"].append(
                            f"Volatility Squeeze ({squeeze_str}) — range contracting, "
                            f"OBV {base_result.get('obv_trend','RISING')} — potential breakout building near {high_str}"
                        )
                    base_result["score"] += 1
                else:
                    base_result["breakout_watch"] = False
    except Exception:
        pass

    # ── Relative Strength vs Nifty ────────────────────────────
    try:
        if nifty_close is not None and len(df) >= 63:
            rs = relative_strength_vs_nifty(close, nifty_close, period=63)
            if rs is not None:
                base_result["rs_vs_nifty"] = rs
                if rs > 10:
                    base_result["signals"].append(f"Strong RS vs Nifty (+{rs:.1f}%) — market leader")
                    base_result["score"] += 2
                elif rs > 0:
                    base_result["signals"].append(f"Outperforming Nifty (+{rs:.1f}%)")
                    base_result["score"] += 1
                elif rs < -10:
                    base_result["signals"].append(f"Weak RS vs Nifty ({rs:.1f}%) — laggard, avoid")
                    base_result["score"] -= 2
                elif rs < 0:
                    base_result["signals"].append(f"Underperforming Nifty ({rs:.1f}%)")
                    base_result["score"] -= 1

            # IBD composite RS rank (0-100) and Mansfield RS
            try:
                ibd_rs = ibd_rs_rank(close, nifty_close)
                if ibd_rs is not None:
                    base_result["ibd_rs"] = ibd_rs
                    if ibd_rs >= 80:
                        base_result["signals"].append(f"IBD RS Rank {ibd_rs:.0f} — top-tier momentum leader")
                        base_result["score"] += 1
            except Exception:
                pass
            try:
                mf_rs = mansfield_rs(close, nifty_close)
                if mf_rs is not None:
                    base_result["mansfield_rs"] = mf_rs
            except Exception:
                pass
    except Exception:
        pass

    # ── Weekly Trend Confirmation ─────────────────────────────
    try:
        if not is_intraday and weekly_df is not None:
            wt = weekly_trend(weekly_df)
            base_result.update(wt)
            wt_val = wt.get("weekly_trend")
            if wt_val == "BULLISH":
                base_result["signals"].append("Weekly trend BULLISH — daily signal confirmed by weekly")
                base_result["score"] += 1
            elif wt_val == "BEARISH":
                base_result["signals"].append("Weekly trend BEARISH — daily buy signal at risk")
                base_result["score"] -= 1
    except Exception:
        pass

    # ── Better SL / Target / R:R ─────────────────────────────
    base_result = compute_trade_levels(base_result)

    # ── Minervini Stage Analysis ──────────────────────────────
    try:
        stage_data = compute_stage(df)
        base_result.update(stage_data)
    except Exception:
        pass

    # ── Pivot-based horizontal S/R (breakout level) ───────────
    try:
        pivot_data = compute_pivot_levels(df)
        base_result.update(pivot_data)
    except Exception:
        pass

    # ── 0-100 Score + data confidence label ───────────────────
    base_result["score100"]         = compute_score100(base_result)
    base_result["verdict"]          = compute_verdict(base_result["score100"])
    base_result["data_confidence"]  = compute_data_confidence(base_result)

    # ── Setup status (WATCH / ENTRY / AVOID) ──────────────────
    base_result["setup_status"] = compute_setup_status(base_result)

    return base_result
