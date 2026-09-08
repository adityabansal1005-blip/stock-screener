"""
ML Signal Scorer — LightGBM trained on 10 years of NSE data.

Features: ~24 technical indicators + 59 strategy signals = 83+ features.
Target:   Price hits +2% (T1) before hitting -1% (SL) within 15 days.
          This models the actual swing trade outcome, not just forward return.

Split:
  Train:    2015-01-01 → 2022-12-31
  Test:     2023-01-01 → 2024-12-31
  Validate: 2025-01-01 → latest

Persistence: model stored in strategy_data.db (serialised via pickle bytes).
"""
import io
import pickle
import sqlite3
import threading
import numpy as np
import pandas as pd
from pathlib import Path

_DB_PATH = Path(__file__).parent / "strategy_data.db"
_lock    = threading.Lock()
_model_cache: dict = {}
LABEL_VERSION = "next-open-stop-first-purged-v2"

# ── Trade outcome parameters ──────────────────────────────────────────────────
TARGET_PROFIT_PCT = 2.0      # T1: +2% from entry
TARGET_SL_PCT     = 1.0      # SL: -1% from entry (1:2 risk/reward)
TARGET_DAYS       = 15       # max hold days
COMMISSION        = 0.001    # 0.1%
SLIPPAGE          = 0.0005   # 0.05%

TRAIN_END  = "2022-12-31"
TEST_START = "2023-01-01"
TEST_END   = "2024-12-31"
VAL_START  = "2025-01-01"


# ── technical indicator helpers ───────────────────────────────────────────────

def _ema(s, n):     return s.ewm(span=n, adjust=False).mean()
def _sma(s, n):     return s.rolling(n).mean()

def _rsi(close, n=14):
    from signals import rsi
    return rsi(close, n)

def _atr(df, n=14):
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"]  - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=n-1, adjust=False).mean()

def _macd_hist(close):
    fast = _ema(close, 12); slow = _ema(close, 26)
    sig  = _ema(fast - slow, 9)
    return (fast - slow) - sig

def _obv(df):
    return (np.sign(df["close"].diff().fillna(0)) * df["volume"]).cumsum()

def _cci(df, n=20):
    tp  = (df["high"] + df["low"] + df["close"]) / 3
    sma = tp.rolling(n).mean()
    mad = tp.rolling(n).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - sma) / (0.015 * mad.replace(0, np.nan))

def _adx_vals(df, n=14):
    up   = df["high"].diff().clip(lower=0)
    down = (-df["low"].diff()).clip(lower=0)
    pdm  = up.where(up > down, 0.0)
    mdm  = down.where(down > up, 0.0)
    atr  = _atr(df, n)
    pdi  = 100 * pdm.ewm(com=n-1, adjust=False).mean() / atr.replace(0, np.nan)
    mdi  = 100 * mdm.ewm(com=n-1, adjust=False).mean() / atr.replace(0, np.nan)
    dx   = (100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)).fillna(0)
    return dx.ewm(com=n-1, adjust=False).mean(), pdi, mdi


# ── feature names ─────────────────────────────────────────────────────────────

TECHNICAL_FEATURES = [
    "rsi14", "rsi_delta",
    "macd_hist", "macd_hist_delta",
    "ema9_dist", "ema21_dist", "ema50_dist", "ema200_dist",
    "ema_stack", "price_vs_ema200",
    "atr_pct", "vol_ratio",
    "cci", "adx", "adx_pdi_mdi",
    "obv_slope",
    "bb_pct",
    "close_pct_5d", "close_pct_20d", "close_pct_60d",
    "high52w_pct", "low52w_pct",
    "day_range_pct", "vol_regime",
]


def _get_strategy_feature_names() -> list[str]:
    try:
        import strategy_library
        return [f"strat_{k}" for k in strategy_library.STRATEGIES]
    except Exception:
        return []


def get_all_feature_names() -> list[str]:
    return TECHNICAL_FEATURES + _get_strategy_feature_names()


# ── feature engineering ───────────────────────────────────────────────────────

def build_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"].copy()
    e9, e21, e50, e200 = _ema(c,9), _ema(c,21), _ema(c,50), _ema(c,200)
    rsi    = _rsi(c)
    mh     = _macd_hist(c)
    atr    = _atr(df)
    obv    = _obv(df)
    cci    = _cci(df)
    adx, pdi, mdi = _adx_vals(df)

    vol20  = df["volume"].rolling(20).mean()
    ma20   = c.rolling(20).mean(); std20 = c.rolling(20).std()
    bb_upper = ma20 + 2*std20; bb_lower = ma20 - 2*std20
    bb_range = (bb_upper - bb_lower).replace(0, np.nan)

    high52 = c.rolling(252).max(); low52 = c.rolling(252).min()
    obv_norm  = obv / obv.rolling(20).std().replace(0, np.nan)
    obv_slope = obv_norm.diff(5)
    vol252    = df["volume"].rolling(252).std()
    vol20s    = df["volume"].rolling(20).std()
    vol_regime = (vol20s / vol252.replace(0, np.nan)).fillna(1)

    ft = pd.DataFrame(index=df.index)
    ft["rsi14"]            = rsi
    ft["rsi_delta"]        = rsi.diff(3)
    ft["macd_hist"]        = mh
    ft["macd_hist_delta"]  = mh.diff(3)
    ft["ema9_dist"]        = (c/e9  - 1)*100
    ft["ema21_dist"]       = (c/e21 - 1)*100
    ft["ema50_dist"]       = (c/e50 - 1)*100
    ft["ema200_dist"]      = (c/e200- 1)*100
    ft["ema_stack"]        = ((e9>e21)&(e21>e50)&(e50>e200)).astype(int)
    ft["price_vs_ema200"]  = (c/e200 - 1)*100
    ft["atr_pct"]          = atr / c.replace(0, np.nan)*100
    ft["vol_ratio"]        = df["volume"] / vol20.replace(0, np.nan)
    ft["cci"]              = cci.clip(-300, 300)
    ft["adx"]              = adx
    ft["adx_pdi_mdi"]      = (pdi-mdi)/adx.replace(0, np.nan)
    ft["obv_slope"]        = obv_slope.clip(-10, 10)
    ft["bb_pct"]           = (c - bb_lower)/bb_range
    ft["close_pct_5d"]     = c.pct_change(5)*100
    ft["close_pct_20d"]    = c.pct_change(20)*100
    ft["close_pct_60d"]    = c.pct_change(60)*100
    ft["high52w_pct"]      = (c/high52 - 1)*100
    ft["low52w_pct"]       = (c/low52  - 1)*100
    ft["day_range_pct"]    = (df["high"]-df["low"])/c.replace(0, np.nan)*100
    ft["vol_regime"]       = vol_regime.clip(0, 5)
    return ft.replace([np.inf, -np.inf], np.nan)


def build_strategy_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run all 59 strategy functions, return 0/1 feature columns."""
    try:
        import strategy_library
        return strategy_library.compute_all_signals(df)
    except Exception:
        return pd.DataFrame(index=df.index)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    tech  = build_technical_features(df)
    strat = build_strategy_features(df)
    return pd.concat([tech, strat], axis=1)


# ── target: T1 hit before SL within N days ───────────────────────────────────

def build_labels(df: pd.DataFrame) -> pd.Series:
    """
    1 = price hits +TARGET_PROFIT_PCT% before hitting -TARGET_SL_PCT% within TARGET_DAYS.
    0 = SL hit first, or neither hit (timeout = loss).
    This models the actual swing trade outcome.
    """
    n = len(df)
    labels = np.full(n, np.nan)
    opens = df["open"].to_numpy(dtype=float)
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)

    # Require a complete horizon, including entry day. Unresolved recent
    # observations cannot become training losses simply because time ran out.
    for i in range(n - TARGET_DAYS):
        entry = opens[i + 1] * (1 + SLIPPAGE)
        if not np.isfinite(entry) or entry <= 0:
            continue
        t1    = entry * (1 + TARGET_PROFIT_PCT / 100)
        sl    = entry * (1 - TARGET_SL_PCT    / 100)
        end = i + TARGET_DAYS + 1
        if not (np.isfinite(highs[i+1:end]).all() and np.isfinite(lows[i+1:end]).all()
                and np.isfinite(opens[i+1:end]).all()):
            continue
        labels[i] = 0
        for j in range(i + 1, end):
            if opens[j] <= sl:
                break
            if opens[j] >= t1:
                labels[i] = 1
                break
            if lows[j] <= sl:
                break
            if highs[j] >= t1:
                labels[i] = 1
                break

    return pd.Series(labels, index=df.index)


def label_horizon_end(df):
    """Conservative last information date used for purging split boundaries."""
    return pd.Series(df.index, index=df.index).shift(-TARGET_DAYS)


# ── model DB helpers ──────────────────────────────────────────────────────────

def _ensure_model_table():
    with _lock, sqlite3.connect(str(_DB_PATH)) as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS ml_models (
                model_id     TEXT PRIMARY KEY,
                trained_at   TEXT NOT NULL,
                n_features   INTEGER,
                train_acc    REAL,
                test_acc     REAL,
                val_acc      REAL,
                test_auc     REAL,
                val_auc      REAL,
                n_train      INTEGER,
                n_test       INTEGER,
                feature_imp  TEXT,
                blob         BLOB NOT NULL
            )
        """)
        # Add columns introduced in this version (safe on existing DBs)
        for col, typ in [("test_auc", "REAL"), ("val_auc", "REAL")]:
            try:
                c.execute(f"ALTER TABLE ml_models ADD COLUMN {col} {typ}")
            except sqlite3.OperationalError:
                pass   # column already exists
        c.commit()


def save_model(model_id: str, estimator, metrics: dict):
    _ensure_model_table()
    import json
    from datetime import datetime
    blob = pickle.dumps(estimator)
    fi   = json.dumps(metrics.get("feature_importance", {}))
    with _lock, sqlite3.connect(str(_DB_PATH)) as c:
        c.execute("""
            INSERT OR REPLACE INTO ml_models
            (model_id, trained_at, n_features, train_acc, test_acc, val_acc,
             test_auc, val_auc, n_train, n_test, feature_imp, blob)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            model_id,
            datetime.now().strftime("%Y-%m-%d %H:%M"),
            metrics.get("n_features"),
            metrics.get("train_acc"),
            metrics.get("test_acc"),
            metrics.get("val_acc"),
            metrics.get("test_auc"),
            metrics.get("val_auc"),
            metrics.get("n_train"),
            metrics.get("n_test"),
            fi, blob,
        ))
        c.commit()
    _model_cache[model_id] = estimator
    print(f"[ML] Model '{model_id}' saved — test_acc={metrics.get('test_acc','?')}, "
          f"test_auc={metrics.get('test_auc','?')}")


def load_model(model_id: str = "default"):
    if model_id in _model_cache:
        return _model_cache[model_id]
    _ensure_model_table()
    with _lock, sqlite3.connect(str(_DB_PATH)) as c:
        row = c.execute("SELECT blob FROM ml_models WHERE model_id=?", (model_id,)).fetchone()
    if row is None:
        return None
    est = pickle.loads(row[0])
    _model_cache[model_id] = est
    return est


def get_model_meta(model_id: str = "default") -> dict:
    _ensure_model_table()
    import json
    with _lock, sqlite3.connect(str(_DB_PATH)) as c:
        row = c.execute("""
            SELECT model_id, trained_at, n_features, train_acc, test_acc, val_acc,
                   test_auc, val_auc, n_train, n_test, feature_imp
            FROM ml_models WHERE model_id=?
        """, (model_id,)).fetchone()
    if row is None:
        return {}
    keys = ["model_id","trained_at","n_features","train_acc","test_acc","val_acc",
            "test_auc","val_auc","n_train","n_test","feature_importance"]
    d = dict(zip(keys, row))
    try:
        d["feature_importance"] = json.loads(d.get("feature_importance") or "{}")
    except Exception:
        d["feature_importance"] = {}
    return d


# ── training ──────────────────────────────────────────────────────────────────

def train(symbols: list[str], data: dict[str, pd.DataFrame] | None = None,
          model_id: str = "default") -> dict:
    """
    Train LightGBM on all symbols. Target = T1 hit before SL.
    First runs strategy_researcher to add web-sourced strategies.
    Returns metrics dict.
    """
    import lightgbm as lgb
    from sklearn.metrics import accuracy_score, roc_auc_score

    # Try to load researched strategies (non-blocking)
    try:
        import strategy_researcher
        strategy_researcher._register_researched()
    except Exception:
        pass

    if data is None:
        from data_manager import load_many
        data = load_many(symbols)

    feature_names = get_all_feature_names()
    print(f"[ML] Building features from {len(data)} symbols ({len(feature_names)} features)...")
    print(f"[ML] Target: T1=+{TARGET_PROFIT_PCT}% before SL=-{TARGET_SL_PCT}% within {TARGET_DAYS}d")

    X_train, y_train = [], []
    X_test,  y_test  = [], []
    X_val,   y_val   = [], []

    for i, (symbol, df) in enumerate(data.items()):
        if len(df) < 300:
            continue
        try:
            ft  = build_features(df)
            lbl = build_labels(df)

            # Only keep columns we know about (handles variable strategy library)
            avail_cols = [c for c in feature_names if c in ft.columns]
            combined = pd.concat([ft[avail_cols], lbl.rename("target")], axis=1).dropna()
            if len(combined) < 100:
                continue

            label_end = label_horizon_end(df).reindex(combined.index)
            train_m = (combined.index <= pd.Timestamp(TRAIN_END)) & (label_end <= pd.Timestamp(TRAIN_END))
            test_m  = (combined.index >= pd.Timestamp(TEST_START)) & (label_end <= pd.Timestamp(TEST_END))
            val_m   = combined.index >= pd.Timestamp(VAL_START)

            def _s(mask):
                sub = combined[mask]
                return sub[avail_cols].values, sub["target"].values

            Xt, yt   = _s(train_m)
            Xte, yte = _s(test_m)
            Xv, yv   = _s(val_m)

            if len(Xt) >= 50:  X_train.append(Xt); y_train.append(yt)
            if len(Xte) >= 20: X_test.append(Xte); y_test.append(yte)
            if len(Xv) >= 10:  X_val.append(Xv);   y_val.append(yv)

        except Exception as e:
            print(f"[ML] Skipping {symbol}: {e}")

        if (i+1) % 50 == 0:
            print(f"[ML] Features built: {i+1}/{len(data)} symbols")

    if not X_train:
        return {"error": "No training data"}

    Xt  = np.vstack(X_train); yt  = np.concatenate(y_train)
    Xte = np.vstack(X_test)  if X_test else np.empty((0, len(avail_cols)))
    yte = np.concatenate(y_test)  if y_test  else np.array([])
    Xv  = np.vstack(X_val)   if X_val  else np.empty((0, len(avail_cols)))
    yv  = np.concatenate(y_val)   if y_val   else np.array([])

    # Fill NaN with column medians from training set
    medians = np.nanmedian(Xt, axis=0)
    medians = np.where(np.isnan(medians), 0, medians)
    for arr in [Xt, Xte, Xv]:
        if arr.shape[0] == 0:
            continue
        inds = np.where(np.isnan(arr))
        arr[inds] = np.take(medians, inds[1])

    print(f"[ML] Train={len(Xt)}, Test={len(Xte)}, Val={len(Xv)} samples")
    pos_rate = float(yt.mean())
    print(f"[ML] Win rate in training data: {pos_rate:.1%}")

    # ── LightGBM ──────────────────────────────────────────────
    dtrain = lgb.Dataset(Xt, label=yt, feature_name=avail_cols, free_raw_data=False)
    dval   = lgb.Dataset(Xte, label=yte, reference=dtrain, free_raw_data=False) if len(Xte) > 0 else None

    params = {
        "objective":       "binary",
        "metric":          ["binary_logloss", "auc"],
        "verbosity":       -1,
        "boosting_type":   "gbdt",
        "num_leaves":      63,
        "max_depth":       7,
        "learning_rate":   0.05,
        "n_estimators":    500,
        "min_child_samples": 30,
        "subsample":       0.8,
        "colsample_bytree": 0.8,
        "reg_alpha":       0.1,
        "reg_lambda":      0.1,
        "scale_pos_weight": (1 - pos_rate) / max(pos_rate, 0.01),
        "n_jobs":          -1,
        "random_state":    42,
    }

    callbacks = [lgb.log_evaluation(50)]
    if dval:
        callbacks.append(lgb.early_stopping(30, verbose=False))
    valid_sets  = [dval] if dval else None
    valid_names = ["test"] if dval else None

    model = lgb.train(
        params,
        dtrain,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )

    def _acc(X, y):
        if len(X) == 0: return None
        pred = (model.predict(X) > 0.5).astype(int)
        return round(float(accuracy_score(y, pred)), 4)

    def _auc(X, y):
        if len(X) == 0 or len(np.unique(y)) < 2: return None
        try:
            return round(float(roc_auc_score(y, model.predict(X))), 4)
        except Exception:
            return None

    train_acc = _acc(Xt, yt)
    test_acc  = _acc(Xte, yte)
    val_acc   = _acc(Xv, yv)
    test_auc  = _auc(Xte, yte)
    val_auc   = _auc(Xv, yv)

    # Feature importance (gain-based)
    fi_vals   = model.feature_importance(importance_type="gain")
    fi_sum    = fi_vals.sum() or 1
    fi        = {name: round(float(v)/fi_sum, 4) for name, v in zip(avail_cols, fi_vals)}
    top10     = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:10]

    metrics = {
        "n_features":          len(avail_cols),
        "n_train":             len(Xt),
        "n_test":              len(Xte),
        "n_val":               len(Xv),
        "train_acc":           train_acc,
        "test_acc":            test_acc,
        "val_acc":             val_acc,
        "test_auc":            test_auc,
        "val_auc":             val_auc,
        "win_rate_in_data":    round(pos_rate, 4),
        "feature_importance":  fi,
        "top_features":        top10,
        "symbols_used":        len(X_train),
        "target":              f"T1=+{TARGET_PROFIT_PCT}% before SL=-{TARGET_SL_PCT}% in {TARGET_DAYS}d",
    }

    model.screener_label_version = LABEL_VERSION
    model.screener_feature_medians = medians
    save_model(model_id, model, metrics)
    return metrics


# ── inference ─────────────────────────────────────────────────────────────────

def score_signal(df: pd.DataFrame, model_id: str = "default") -> float | None:
    """
    Return ML probability [0-1] that the current bar is a winning setup.
    (Probability of T1 being hit before SL within TARGET_DAYS.)
    Returns None if model not trained or features unavailable.
    """
    model = load_model(model_id)
    if model is None:
        return None
    if getattr(model, "screener_label_version", None) != LABEL_VERSION:
        return None  # archived models used incompatible labels; retrain first
    try:
        ft = build_features(df)
        feature_names = model.feature_name()
        if any(c not in ft.columns for c in feature_names):
            return None
        row = ft[feature_names].iloc[-1]
        if row.isna().all():
            return None
        medians = getattr(model, "screener_feature_medians", None)
        if medians is None or len(medians) != len(feature_names):
            return None
        values = row.to_numpy(dtype=float)
        X = np.where(np.isfinite(values), values, medians).reshape(1, -1)

        # LightGBM predict returns array
        prob = float(model.predict(X)[0])
        return round(prob, 3)
    except Exception as e:
        print(f"[ML] score_signal failed: {e}")
        return None
