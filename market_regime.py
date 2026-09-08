"""
Market regime detection — determines the current market environment.
Regime drives which strategies work: bull favours momentum/breakouts,
bear favours mean-reversion/oversold bounces, sideways favours CPR/range.
Cached 30 minutes.
"""
import time
import numpy as np
import pandas as pd
from market_context import get_nifty_daily, get_nifty_weekly, get_market_context

_cache = {"data": None, "ts": 0}
CACHE_TTL = 1800  # 30 min


def _adx(high, low, close, period=14):
    try:
        h, l, c = high.to_numpy(float), low.to_numpy(float), close.to_numpy(float)
        pc = np.empty(len(c)); pc[0] = c[0]; pc[1:] = c[:-1]
        tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
        up   = np.diff(h, prepend=h[0])
        down = -np.diff(l, prepend=l[0])
        pdm = np.where((up > down) & (up > 0), up, 0.0)
        mdm = np.where((down > up) & (down > 0), down, 0.0)
        s = pd.Series
        atr_s = s(tr).ewm(com=period - 1, adjust=False).mean()
        pdi   = 100 * s(pdm).ewm(com=period - 1, adjust=False).mean() / atr_s.replace(0, np.nan)
        mdi   = 100 * s(mdm).ewm(com=period - 1, adjust=False).mean() / atr_s.replace(0, np.nan)
        dx    = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
        return float(dx.ewm(com=period - 1, adjust=False).mean().iloc[-1])
    except Exception:
        return None


# Strategy guidance per regime — used by AI analyst
REGIME_STRATEGY = {
    "BULL_TREND": (
        "Market is in a confirmed bull trend (above 200 EMA, ADX>25). "
        "Favour momentum breakouts and trend-following entries. "
        "Prioritise stocks with strong RS vs Nifty and Supertrend BUY. "
        "Use wider stops — pullbacks are opportunities, not reversals."
    ),
    "BULL_WEAK": (
        "Market is above 200 EMA but trend is weak (ADX<25, sideways). "
        "Be selective — only take the highest-conviction setups (score>=70). "
        "CPR and pivot-based range trades work better than breakouts here."
    ),
    "RECOVERY": (
        "Market is recovering — below 200 EMA but reclaiming 50 EMA. "
        "Buy leading sectors and stocks with high delivery %. "
        "Keep position sizes at 50-60% of normal. Confirm with weekly trend."
    ),
    "BEAR_TREND": (
        "Market is in a confirmed bear trend (below 200 EMA, ADX>25). "
        "Only take oversold bounce setups with RSI<35 + Stoch RSI oversold + high delivery. "
        "Strict 1×ATR stops. Avoid breakout trades — they fail in bear markets. "
        "Consider reducing exposure and holding cash."
    ),
    "SIDEWAYS": (
        "Market is range-bound (ADX<20). "
        "CPR and classic pivot S/R are the most reliable tools. "
        "Mean-reversion strategies: buy near support, sell near resistance. "
        "Breakout signals have a high false-signal rate in this regime — wait for confirmation."
    ),
    "HIGH_VOLATILITY": (
        "VIX is elevated (>20). Market is stressed. "
        "Reduce position sizes by 30-50%. Widen stops or avoid fresh longs. "
        "Focus only on the most liquid large-caps. Consider hedging."
    ),
}


def get_regime() -> dict:
    now = time.time()
    if _cache["data"] and now - _cache["ts"] < CACHE_TTL:
        return _cache["data"]

    result = _fetch_regime()
    _cache["data"] = result
    _cache["ts"] = now
    return result


def _fetch_regime() -> dict:
    default = {
        "regime": "UNKNOWN", "regime_label": "Unknown",
        "nifty_close": None, "nifty_ema50": None, "nifty_ema200": None,
        "above_200ema": None, "above_50ema": None,
        "adx": None, "strong_trend": False,
        "weekly_trend": None, "vix": None, "vix_extreme": False,
        "return_1m": None, "return_3m": None,
        "strategy": "Market regime data unavailable. Do not treat signals as confirmed. Reduce position sizes and avoid aggressive entries until regime can be determined.",
    }

    try:
        # Use shared Nifty DataFrames — no yfinance call here
        df = get_nifty_daily()
        if df is None or len(df) < 50:
            return default

        close = df["Close"]
        curr  = float(close.iloc[-1])
        ema50  = float(close.ewm(span=50,  adjust=False).mean().iloc[-1])
        ema200 = float(close.ewm(span=200, adjust=False).mean().iloc[-1]) if len(close) >= 200 else None
        adx_v  = _adx(df["High"], df["Low"], close)
        strong = adx_v is not None and adx_v > 25

        above200 = ema200 is not None and curr > ema200
        above50  = curr > ema50

        r1m = round((curr / float(close.iloc[-21]) - 1) * 100, 2) if len(close) >= 21 else None
        r3m = round((curr / float(close.iloc[-63]) - 1) * 100, 2) if len(close) >= 63 else None

        # Weekly trend — shared DataFrame, no extra fetch
        weekly_trend = None
        wdf = get_nifty_weekly()
        if wdf is not None and len(wdf) >= 20:
            wclose = wdf["Close"]
            wema20 = float(wclose.ewm(span=20, adjust=False).mean().iloc[-1])
            weekly_trend = "BULLISH" if float(wclose.iloc[-1]) > wema20 else "BEARISH"

        # VIX — from NSE live API via market_context (already fetched, no extra call)
        vix_val, vix_extreme = None, False
        ctx = get_market_context()
        if ctx.get("vix") is not None:
            vix_val     = ctx["vix"]
            vix_extreme = ctx.get("vix_high", False)

        # Regime classification
        if vix_extreme and vix_val and vix_val > 25:
            regime, label = "HIGH_VOLATILITY", "High Volatility / Stressed"
        elif above200 and strong and above50:
            regime, label = "BULL_TREND", "Bull Trend"
        elif above200 and not strong:
            regime, label = "BULL_WEAK", "Bull — Weak Trend"
        elif not above200 and above50:
            regime, label = "RECOVERY", "Recovery Phase"
        elif not above200 and strong:
            regime, label = "BEAR_TREND", "Bear Trend"
        else:
            regime, label = "SIDEWAYS", "Sideways / Consolidation"

        return {
            "regime": regime,
            "regime_label": label,
            "nifty_close": round(curr, 2),
            "nifty_ema50":  round(ema50, 2),
            "nifty_ema200": round(ema200, 2) if ema200 else None,
            "above_200ema": above200,
            "above_50ema":  above50,
            "adx": round(adx_v, 1) if adx_v else None,
            "strong_trend": strong,
            "weekly_trend": weekly_trend,
            "vix": vix_val,
            "vix_extreme": vix_extreme,
            "return_1m": r1m,
            "return_3m": r3m,
            "strategy": REGIME_STRATEGY.get(regime, ""),
        }

    except Exception as e:
        print(f"[Regime] Error: {e}")
        return default
