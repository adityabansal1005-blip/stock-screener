"""
NSE India specific data: FII/DII flows and delivery percentage.
Both are critical signals unavailable in standard technical analysis.

FII/DII: Foreign vs Domestic institutional net buying — drives large-cap direction.
Delivery %: % of trades resulting in actual delivery — high = genuine accumulation.
"""
import io
import time
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta

log = logging.getLogger(__name__)

_fii_cache  = {"data": None, "ts": 0}
_delv_cache = {"data": {}, "ts": 0}
CACHE_TTL   = 3600  # 1 hour

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
    "Connection": "keep-alive",
}


def _nse_session() -> requests.Session:
    sess = requests.Session()
    sess.headers.update(_HEADERS)
    try:
        sess.get("https://www.nseindia.com", timeout=10)
        time.sleep(0.3)
    except Exception:
        pass
    return sess


# ── FII / DII ────────────────────────────────────────────────

def get_fii_dii() -> dict:
    """
    Returns today's FII and DII net buy/sell in ₹ Crores.
    Positive net = net buyers. Negative = net sellers.
    """
    now = time.time()
    if _fii_cache["data"] and now - _fii_cache["ts"] < CACHE_TTL:
        return _fii_cache["data"]

    try:
        sess = _nse_session()
        resp = sess.get(
            "https://www.nseindia.com/api/fiidiiTradeReact",
            timeout=15,
        )
        if resp.status_code != 200:
            return {}

        rows = resp.json()
        if not rows or not isinstance(rows, list):
            return {}

        fii_net = dii_net = fii_buy = fii_sell = dii_buy = dii_sell = 0.0
        date_str = ""

        for row in rows:
            cat = str(row.get("category", "")).upper()
            net = _safe_float(row.get("netValue") or row.get("net_value") or row.get("NET_VALUE", 0))
            buy = _safe_float(row.get("buyValue")  or row.get("buy_value")  or row.get("BUY_VALUE",  0))
            sell= _safe_float(row.get("sellValue") or row.get("sell_value") or row.get("SELL_VALUE", 0))
            date_str = row.get("date", date_str)

            if "FII" in cat or "FPI" in cat:
                fii_net += net; fii_buy += buy; fii_sell += sell
            elif "DII" in cat:
                dii_net += net; dii_buy += buy; dii_sell += sell

        result = {
            "date":     date_str,
            "fii_net":  round(fii_net,  2),
            "fii_buy":  round(fii_buy,  2),
            "fii_sell": round(fii_sell, 2),
            "dii_net":  round(dii_net,  2),
            "dii_buy":  round(dii_buy,  2),
            "dii_sell": round(dii_sell, 2),
            "combined_net": round(fii_net + dii_net, 2),
        }
        _fii_cache["data"] = result
        _fii_cache["ts"]   = now
        return result

    except Exception as e:
        log.warning(f"[NSE] FII/DII failed: {e}")
        return {}


# ── Delivery % ───────────────────────────────────────────────

def get_delivery_pct() -> dict:
    """
    Returns a dict of {SYMBOL: delivery_pct} from the latest NSE bhav copy.
    Delivery % > 50 = genuine buying; < 20 = speculative/intraday noise.
    """
    now = time.time()
    if _delv_cache["data"] and now - _delv_cache["ts"] < CACHE_TTL:
        return _delv_cache["data"]

    sess = _nse_session()

    for days_back in range(1, 6):
        dt = datetime.now() - timedelta(days=days_back)
        if dt.weekday() >= 5:   # skip weekends
            continue
        date_str = dt.strftime("%d%m%Y")
        url = (
            f"https://archives.nseindia.com/products/content/"
            f"sec_bhavdata_full_{date_str}.csv"
        )
        try:
            resp = sess.get(url, timeout=20)
            if resp.status_code != 200:
                continue

            df = pd.read_csv(io.StringIO(resp.text))
            df.columns = [c.strip() for c in df.columns]

            # Keep EQ series only
            if "SERIES" in df.columns:
                df = df[df["SERIES"].str.strip() == "EQ"]

            # Find delivery column (NSE has changed its name over time)
            delv_col = next(
                (c for c in df.columns if "DELIV_PER" in c.upper() or "DELIVERY" in c.upper()),
                None,
            )
            sym_col = next((c for c in df.columns if c.strip() == "SYMBOL"), None)

            if delv_col and sym_col:
                df[delv_col] = pd.to_numeric(df[delv_col], errors="coerce")
                result = dict(zip(df[sym_col].str.strip(), df[delv_col]))
                result = {k: round(float(v), 1) for k, v in result.items() if pd.notna(v)}
                _delv_cache["data"] = result
                _delv_cache["ts"]   = now
                return result

        except Exception as e:
            log.debug(f"[NSE] Delivery {date_str} failed: {e}")
            time.sleep(1)
            continue

    log.warning("[NSE] Could not fetch delivery % from any recent bhav copy")
    return {}


def _safe_float(v) -> float:
    try:
        return float(str(v).replace(",", ""))
    except Exception:
        return 0.0


# ── Participant-Wise OI (FII / DII / Client / Pro) ────────────

_pwoi_cache = {"data": None, "ts": 0}

def get_participant_oi() -> dict:
    """
    NSE participant-wise open interest — who is positioned where in F&O.

    Returns:
      fii_index_long, fii_index_short, fii_stock_long, fii_stock_short,
      client_net_index, pro_net_index (all in lots)

    Signal logic:
      FII net long index futures = institutional bulls (bullish overlay)
      FII net short index futures = institutional hedging/bearish
      Client (retail) net short at extremes = contrarian bullish signal
      Pro (prop desks) net long = smart money accumulating
    """
    now = time.time()
    if _pwoi_cache["data"] and now - _pwoi_cache["ts"] < CACHE_TTL:
        return _pwoi_cache["data"]

    try:
        from nselib import derivatives
        df = derivatives.participant_wise_open_interest()
        if df is None or df.empty:
            return {}

        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        result = {}

        for _, row in df.iterrows():
            cat = str(row.get("client_type", row.get("category", ""))).upper().strip()
            long_v  = _safe_float(row.get("future_index_long",  row.get("fut_index_long",  0)))
            short_v = _safe_float(row.get("future_index_short", row.get("fut_index_short", 0)))
            stk_long  = _safe_float(row.get("future_stock_long",  0))
            stk_short = _safe_float(row.get("future_stock_short", 0))

            if "FII" in cat or "FPI" in cat:
                result["fii_index_long"]  = long_v
                result["fii_index_short"] = short_v
                result["fii_index_net"]   = round(long_v - short_v, 0)
                result["fii_stock_long"]  = stk_long
                result["fii_stock_short"] = stk_short
                result["fii_stock_net"]   = round(stk_long - stk_short, 0)
            elif "CLIENT" in cat or "RETAIL" in cat:
                result["client_index_net"] = round(long_v - short_v, 0)
            elif "PRO" in cat:
                result["pro_index_net"] = round(long_v - short_v, 0)
            elif "DII" in cat:
                result["dii_index_net"] = round(long_v - short_v, 0)

        if result:
            fii_net = result.get("fii_index_net", 0)
            result["fii_futures_signal"] = (
                "BULLISH" if fii_net > 5000 else
                "SLIGHTLY_BULLISH" if fii_net > 0 else
                "SLIGHTLY_BEARISH" if fii_net > -5000 else
                "BEARISH"
            )
            client_net = result.get("client_index_net", 0)
            result["retail_contrarian_signal"] = (
                "BUY" if client_net < -20000 else    # retail heavily short = contrarian buy
                "SELL" if client_net > 20000 else    # retail heavily long = contrarian sell
                "NEUTRAL"
            )

        _pwoi_cache["data"] = result
        _pwoi_cache["ts"]   = now
        return result

    except Exception as e:
        log.debug(f"[NSE] Participant OI failed: {e}")
        return {}


# ── Bulk / Block Deals (last 7 days) ─────────────────────────

_bulk_cache = {"data": {}, "ts": 0}

def get_bulk_block_deals(days_back: int = 7) -> dict[str, list]:
    """
    Returns {SYMBOL: [list of deal dicts]} for bulk/block deals in last N days.
    Institutional bulk deals = major conviction signal.
    Block deal ≥ ₹5 Cr or 5L shares between institutions.
    Bulk deal ≥ 0.5% of listed shares.
    """
    now = time.time()
    if _bulk_cache["data"] and now - _bulk_cache["ts"] < CACHE_TTL:
        return _bulk_cache["data"]

    try:
        from nselib import capital_market
        # Try bulk deals
        result: dict[str, list] = {}
        for method_name in ["bulk_deal_data", "block_deals_data"]:
            try:
                fn = getattr(capital_market, method_name, None)
                if fn is None:
                    continue
                df = fn()
                if df is None or df.empty:
                    continue
                df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
                sym_col = next((c for c in df.columns if "symbol" in c), None)
                qty_col = next((c for c in df.columns if "qty" in c or "quantity" in c), None)
                price_col = next((c for c in df.columns if "price" in c or "rate" in c), None)
                type_col = next((c for c in df.columns if "buy" in c or "sell" in c or "type" in c), None)

                for _, row in df.iterrows():
                    sym = str(row.get(sym_col, "")).upper().strip()
                    if not sym:
                        continue
                    deal = {
                        "qty":   _safe_float(row.get(qty_col, 0)) if qty_col else 0,
                        "price": _safe_float(row.get(price_col, 0)) if price_col else 0,
                        "type":  str(row.get(type_col, "")).upper() if type_col else "",
                        "source": method_name,
                    }
                    result.setdefault(sym, []).append(deal)
            except Exception:
                continue

        _bulk_cache["data"] = result
        _bulk_cache["ts"]   = now
        return result

    except Exception as e:
        log.debug(f"[NSE] Bulk/block deals failed: {e}")
        return {}
