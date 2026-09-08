"""
BSE-only equity universe — build, validate, cache.

Roughly 8,600 instruments trade on BSE and not NSE, but that headline is
misleading: most are debt (series F), government securities (G), or shells that
barely trade. The scanner was blind to all of them, which is how a 4x move in
Hindusthan Insulators (BSE 539984) never once appeared in a scan.

Two filters turn that list into something worth scanning:

  1. Structural — ISIN must start with INE (equity, not debt or fund), the BSE
     series must be an equity group, and the name must not look like a bond.
     Excludes series F/G (debt, govt sec), Z (non-compliant) and T
     (trade-to-trade, settlement-restricted).

  2. Liquidity — actually fetch each candidate and apply the liquidity gate.
     A BSE-only listing is illiquid by default; if it were liquid it would
     likely be on NSE too. Only names that clear the turnover floor survive.

Symbols are emitted with a .BO suffix so groww_client._route() sends them to
the BSE endpoint using the scrip code.

    python bse_universe.py build    # validate + cache (slow, once)
    python bse_universe.py          # show what is cached
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

_CACHE = Path(__file__).parent / "bse_universe.json"
_TTL = 7 * 24 * 3600

# BSE groups that are equities we could actually trade.
#   A/B  mainboard   X/XT  smallcap   M/MT  SME   P  permitted
# Excluded: F (debt), G (govt sec), Z (non-compliant), T (trade-to-trade)
EQUITY_SERIES = {"A", "B", "X", "XT", "M", "MT", "P", "IF", "ZP"}
_DEBT_RE = re.compile(r"\d+\.?\d*\s*%|NCD|SDL|GOI|BOND|DEBENT|\bGS\d|STRIPS|TBILL")


def candidates() -> list[dict]:
    """Structural filter only — no network calls."""
    import symbol_registry as R
    reg = R.registry()
    out = []
    for v in reg["by_isin"].values():
        if v.get("nse_symbol") or not v.get("bse_code"):
            continue                                  # NSE-listed or not BSE
        if not v["isin"].startswith("INE"):
            continue                                  # debt / mutual fund
        if (v.get("series") or "").strip().upper() not in EQUITY_SERIES:
            continue
        if _DEBT_RE.search((v.get("name") or "").upper()):
            continue
        out.append(v)
    return out


def build(limit: int | None = None, workers: int = 6) -> dict:
    """Fetch each candidate and keep only those that clear the liquidity gate."""
    import concurrent.futures as cf
    import groww_client
    import liquidity

    cands = candidates()
    if limit:
        cands = cands[:limit]
    print(f"[BSE] {len(cands):,} structural candidates — validating liquidity...")

    kept, checked = [], 0
    lock_report = max(len(cands) // 20, 1)

    def _check(v):
        try:
            df = groww_client.get_daily_bars(v["bse_code"], 150)
            if df is None or len(df) < 100:
                return None
            a = liquidity.assess(df)
            if not a["tradeable"]:
                return None
            return {"bse_code": v["bse_code"], "isin": v["isin"],
                    "name": v["name"], "series": v.get("series"),
                    "turnover": round(a["median_turnover"]),
                    "max_position": a["max_position_value"]}
        except Exception:
            return None

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(_check, cands):
            checked += 1
            if res:
                kept.append(res)
            if checked % lock_report == 0:
                print(f"  {checked}/{len(cands)} checked — {len(kept)} tradeable")

    kept.sort(key=lambda x: -x["turnover"])
    payload = {"built": time.time(), "candidates": len(cands),
               "tradeable": len(kept), "symbols": kept}
    _CACHE.write_text(json.dumps(payload), encoding="utf-8")
    print(f"[BSE] {len(kept):,} tradeable of {len(cands):,} candidates "
          f"-> {_CACHE.name}")
    return payload


def load() -> list[dict]:
    if not _CACHE.exists():
        return []
    try:
        return json.loads(_CACHE.read_text(encoding="utf-8")).get("symbols", [])
    except Exception:
        return []


def symbols(min_turnover: float | None = None) -> list[str]:
    """BSE scrip codes with a .BO suffix, ready for the scan universe."""
    rows = load()
    if min_turnover:
        rows = [r for r in rows if r["turnover"] >= min_turnover]
    return [f"{r['bse_code']}.BO" for r in rows]


def name_map() -> dict[str, str]:
    """{scrip_code: company name} so the dashboard shows names, not numbers."""
    return {r["bse_code"]: r["name"] for r in load()}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    import warnings
    warnings.filterwarnings("ignore")

    if len(sys.argv) > 1 and sys.argv[1] == "build":
        import config, groww_client
        groww_client.init_groww(config.GROWW_API_KEY, config.GROWW_API_SECRET)
        lim = int(sys.argv[2]) if len(sys.argv) > 2 else None
        build(limit=lim)
    else:
        rows = load()
        if not rows:
            print("\n  nothing cached — run:  python bse_universe.py build\n")
        else:
            print(f"\n  {len(rows):,} tradeable BSE-only equities cached")
            print(f"  {'code':<9}{'turnover/day':>16}{'max pos':>12}  name")
            print("  " + "-" * 62)
            for r in rows[:20]:
                print(f"  {r['bse_code']:<9}Rs{r['turnover']:>14,}"
                      f"Rs{r['max_position']:>10,}  {r['name'][:32]}")
            print()
