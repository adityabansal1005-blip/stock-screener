"""
Walk-forward backtester.
Exit logic: simulates the actual trade plan.
  - Checks each day's HIGH against T1 (target hit → WIN)
  - Checks each day's LOW against SL (stop hit → LOSS)
  - If SL and T1 hit on the same bar → LOSS (conservative)
  - If neither hit in FORWARD_DAYS → exit at close[t+5] (time stop)

This is materially more honest than the previous "5-day future close" exit,
which implicitly assumed the trader holds regardless of whether their stop
or target was reached.
"""
import numpy as np
import pandas as pd
from signals import compute_signals
from technical_enhanced import compute_enhanced

# Holding period raised 5 -> 21 sessions (29 Aug 2026).
#
# Costs are charged PER TRADE (0.5582% round trip), not per day, so a 5-day
# clock paid that toll ~50 times a year to collect a measured edge of +0.21%
# each time — a structurally losing trade regardless of signal quality.
# Measured net edge by horizon (all_nse, out-of-sample):
#     5d  gross +0.21%  net -0.35%  ->  ~-16%/yr
#    21d  gross +0.07%  net -0.49%
#    63d  gross +1.78%  net +1.22%  ->  ~+5%/yr
# 21 is a compromise: it cuts turnover ~4x while keeping trades short enough
# that the 5-day signal (IC 0.12) has not fully decayed.
FORWARD_DAYS  = 21
MIN_SCORE100  = 55
LOOKBACK_DAYS = 60

# ── Transaction costs — NSE delivery equity, realistic all-in ────────────────
# The old model charged 0.30% round trip (0.1% brokerage + 0.05% slippage per
# leg) and ignored statutory charges entirely. STT alone is 0.1% on BOTH legs,
# so the real figure is closer to 0.50%. Against a best-bucket average return
# of +0.48% per trade, that understatement was consuming a third to a half of
# the entire measured edge — the difference between a live strategy and a
# breakeven one. Broken out so each component is auditable.
BROKERAGE_PCT   = 0.0010   # 0.10% per leg (Groww/Zerodha delivery, capped ₹20)
STT_PCT         = 0.0010   # 0.10% per leg on delivery — the big one
EXCHANGE_TXN    = 0.0000297  # NSE 0.00297% per leg
SEBI_FEES       = 0.000001   # ₹10 per crore
STAMP_DUTY_BUY  = 0.00015  # 0.015%, buy side only
GST_RATE        = 0.18     # on brokerage + exchange + SEBI
SLIPPAGE        = 0.0005   # 0.05% market impact per leg

def _round_trip_cost() -> float:
    """All-in round-trip cost as a decimal fraction of notional."""
    per_leg_taxable = BROKERAGE_PCT + EXCHANGE_TXN + SEBI_FEES
    gst   = per_leg_taxable * GST_RATE * 2          # both legs
    return (
        BROKERAGE_PCT * 2
        + STT_PCT * 2
        + EXCHANGE_TXN * 2
        + SEBI_FEES * 2
        + STAMP_DUTY_BUY
        + gst
    )

COMMISSION = _round_trip_cost() / 2   # per-leg equivalent, keeps call sites intact
ROUND_TRIP = _round_trip_cost() + SLIPPAGE * 2


def backtest_win_rate(df, nifty_close=None) -> dict:
    """
    Walk-forward backtest: for each bar in the last 60 days that produces a
    score100 ≥ 55 signal, simulate the trade using the actual SL and T1
    computed by the signal pipeline.

    Returns performance metrics including regime-aware stats.
    """
    null = {
        "win_rate":        None,
        "total_signals":   0,
        "avg_return":      None,
        "bt_wins":         0,
        "bt_losses":       0,
        "bt_timeouts":     0,
        "sharpe":          None,
        "profit_factor":   None,
        "expectancy":      None,
        "max_consec_loss": None,
        "avg_mae":         None,   # max adverse excursion
        "avg_mfe":         None,   # max favorable excursion
        "t1_hit_rate":     None,
        "bt_method":      "technical replay; historical fundamentals/regime unavailable",
    }

    if df is None or len(df) < LOOKBACK_DAYS + FORWARD_DAYS + 52:
        return null

    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    wins, losses, timeouts = 0, 0, 0
    target_hits = 0
    returns   = []
    mae_list  = []   # max adverse excursion per trade
    mfe_list  = []   # max favorable excursion per trade

    test_start = len(df) - LOOKBACK_DAYS - FORWARD_DAYS

    for i in range(test_start, len(df) - FORWARD_DAYS):
        subset = df.iloc[: i + 1]
        if len(subset) < 50:
            continue

        try:
            sig = compute_signals(subset)
            if sig is None:
                continue
            weekly = subset.resample("W").agg({"open":"first", "high":"max",
                "low":"min", "close":"last", "volume":"sum"}).dropna(subset=["close"])
            nc = nifty_close.loc[nifty_close.index <= subset.index[-1]] if nifty_close is not None else None
            sig = compute_enhanced(subset, sig, is_intraday=False,
                                   weekly_df=weekly if len(weekly)>=15 else None,
                                   nifty_close=nc)
        except Exception:
            continue

        if sig.get("score100", 0) < MIN_SCORE100:
            continue

        # Enter at the NEXT session's OPEN — the first executable price.
        # The signal is derived from bar i's close, which is only known after
        # the market shuts, so entering at that close is a one-bar look-ahead.
        # Momentum gaps usually continue the move, meaning the old assumption
        # systematically overstated returns.
        if i + 1 >= len(df):
            continue
        entry_open = float(df.iloc[i + 1]["open"])
        if not np.isfinite(entry_open) or entry_open <= 0:
            continue
        entry = entry_open * (1 + SLIPPAGE)

        sl = sig.get("sl_buy")
        t1 = sig.get("target1")
        if sl is not None and t1 is not None:
            if not (np.isfinite(sl) and np.isfinite(t1) and 0 < sl < entry < t1):
                continue  # opening gap has invalidated the proposed entry

        # ── Simulate trade path using daily high/low ──────────────
        hit_sl = False
        hit_t1 = False
        exit_price = None
        mae = 0.0   # worst unrealised loss seen (as % of entry)
        mfe = 0.0   # best unrealised gain seen

        for j in range(1, FORWARD_DAYS + 1):
            bar      = df.iloc[i + j]
            bar_low  = float(bar["low"])
            bar_high = float(bar["high"])
            bar_close= float(bar["close"])
            bar_open = float(bar["open"])

            # Opening prices precede the high/low sequence. A gap through a
            # stop cannot fill at yesterday's stop price.
            if sl is not None and t1 is not None and (bar_open <= sl or bar_open >= t1):
                hit_sl = bar_open <= sl
                hit_t1 = not hit_sl
                exit_price = bar_open * (1 - SLIPPAGE)
                open_ret = (bar_open / entry - 1) * 100
                mae = min(mae, open_ret)
                mfe = max(mfe, open_ret)
                break

            # Track excursion
            bar_min_ret = (bar_low  - entry) / entry * 100
            bar_max_ret = (bar_high - entry) / entry * 100
            mae = min(mae, bar_min_ret)
            mfe = max(mfe, bar_max_ret)

            if sl is not None and t1 is not None:
                sl_hit = bar_low  <= sl
                t1_hit = bar_high >= t1
                if sl_hit and t1_hit:
                    # Same bar: conservative assumption — stop was hit first
                    exit_price = float(sl) * (1 - SLIPPAGE)
                    hit_sl = True
                    break
                elif sl_hit:
                    exit_price = float(sl) * (1 - SLIPPAGE)
                    hit_sl = True
                    break
                elif t1_hit:
                    exit_price = float(t1) * (1 - SLIPPAGE)
                    hit_t1 = True
                    break
            # no SL/T1 available — fall through to time-stop

        if exit_price is None:
            # Time stop: exit at close[t + FORWARD_DAYS]
            exit_price = float(df.iloc[i + FORWARD_DAYS]["close"]) * (1 - SLIPPAGE)
            timeouts += 1
            # Do NOT also count as win/loss — timeout is its own outcome category
        else:
            target_hits += int(hit_t1)
            ret = ((exit_price - entry) / entry - COMMISSION * 2) * 100
            returns.append(ret)
            mae_list.append(mae)
            mfe_list.append(mfe)
            if ret > 0:
                wins += 1
            else:
                losses += 1
            continue

        # Timeout: record the return for expectancy/MAE/MFE but don't count in win/loss
        ret = ((exit_price - entry) / entry - COMMISSION * 2) * 100
        returns.append(ret)
        mae_list.append(mae)
        mfe_list.append(mfe)

    total = wins + losses + timeouts
    if total == 0:
        return null

    arr      = np.array(returns)
    mean_r   = float(np.mean(arr))
    std_r    = float(np.std(arr))

    win_rets  = [r for r in returns if r > 0]
    loss_rets = [r for r in returns if r <= 0]

    avg_win   = float(np.mean(win_rets))  if win_rets  else 0.0
    avg_loss  = float(np.mean(loss_rets)) if loss_rets else 0.0

    gross_profit = sum(win_rets)
    gross_loss   = abs(sum(loss_rets))
    profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else None

    # Expectancy must classify trades the SAME way avg_win / avg_loss do.
    # `wins` counts only target-exits while `total` includes timeouts, so using
    # wins/total here mixed two different groupings: a trade that timed out at
    # +2% landed in win_rets but never in `wins`. That deflated the weighting
    # on avg_win and produced systematically negative expectancy even when
    # avg_return was positive. Classifying by return sign makes expectancy
    # identical to mean(returns), which is the correct definition.
    pos_rate     = len(win_rets) / len(returns) if returns else 0.0
    expectancy   = round(pos_rate * avg_win + (1 - pos_rate) * avg_loss, 2)

    max_cl = cur_cl = 0
    for r in returns:
        if r <= 0:
            cur_cl += 1
            max_cl  = max(max_cl, cur_cl)
        else:
            cur_cl  = 0

    # Overlapping signal returns are not a funded daily portfolio equity curve.
    sharpe = None

    return {
        # Two distinct, honestly-named rates — they are NOT interchangeable:
        #   win_rate    = share of trades that ended profitable (intuitive sense)
        #   t1_hit_rate = share that actually reached target 1 (much lower)
        # The dashboard's "win rate" card reads win_rate.
        "win_rate":        round(pos_rate * 100, 1),
        "t1_hit_rate":     round(target_hits / total * 100, 1),
        "bt_method":      null["bt_method"],
        "excursion_method": "daily-bar bounds; intrabar sequence unknown",
        "avg_win":         round(avg_win, 2),
        "avg_loss":        round(avg_loss, 2),
        "total_signals":   total,
        "avg_return":      round(mean_r, 2),
        "bt_wins":         wins,
        "bt_losses":       losses,
        "bt_timeouts":     timeouts,
        "sharpe":          sharpe,
        "profit_factor":   profit_factor,
        "expectancy":      round(expectancy, 2),
        "max_consec_loss": max_cl,
        "avg_mae":         round(float(np.mean(mae_list)), 2) if mae_list else None,
        "avg_mfe":         round(float(np.mean(mfe_list)), 2) if mfe_list else None,
    }
