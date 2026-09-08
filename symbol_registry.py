"""
Unified instrument registry — resolve a stock by ticker, ISIN, or BSE code.

Motivation
----------
Symbols are the least stable identifier a listed company has. Zomato became
ETERNAL; the project's universe still said ZOMATO, so Groww returned nothing
and the fetch fell through to a rate-limited yfinance and took a training run
down with it. ISIN never changes, and the BSE scrip code rarely does, so both
are better keys for anything long-lived than the trading symbol.

Source: DhanHQ's detailed instrument master, which carries ISIN alongside the
NSE and BSE identifiers. Cached to disk (24h) because it is a ~40 MB download.

    python symbol_registry.py                 # audit the project universe
    python symbol_registry.py RELIANCE        # look one up
    python symbol_registry.py INE002A01018    # by ISIN
    python symbol_registry.py 500325          # by BSE code
"""

from __future__ import annotations

import csv
import io
import pickle
import sys
import time
from pathlib import Path

import requests

_URL = "https://images.dhan.co/api-data/api-scrip-master-detailed.csv"
_CACHE = Path(__file__).parent / "symbol_registry.pkl"
_TTL = 24 * 3600

_reg: dict | None = None


def _download() -> dict:
    """Build lookup indices from the detailed instrument master."""
    resp = requests.get(_URL, timeout=240)
    resp.raise_for_status()

    by_symbol: dict[str, dict] = {}
    by_isin: dict[str, dict] = {}
    by_bse: dict[str, dict] = {}

    for row in csv.DictReader(io.StringIO(resp.text)):
        if row.get("INSTRUMENT") != "EQUITY":
            continue
        exch = (row.get("EXCH_ID") or "").strip()
        if exch not in ("NSE", "BSE"):
            continue

        sym = (row.get("SYMBOL_NAME") or "").strip().upper()
        isin = (row.get("ISIN") or "").strip().upper()
        sec = (row.get("SECURITY_ID") or "").strip()
        name = (row.get("DISPLAY_NAME") or "").strip()
        series = (row.get("SERIES") or "").strip()

        if not isin:
            continue

        rec = by_isin.setdefault(isin, {
            "isin": isin, "name": name,
            "nse_symbol": None, "nse_id": None,
            "bse_code": None, "series": series,
        })
        if name and not rec["name"]:
            rec["name"] = name

        if exch == "NSE":
            # Trading symbol lives in the compact master; the detailed file's
            # SYMBOL_NAME is the long company name, so key NSE rows by id and
            # backfill the ticker from the compact file below.
            rec["nse_id"] = sec
            rec["series"] = series or rec["series"]
        else:
            rec["bse_code"] = sec          # BSE security id IS the scrip code
            by_bse[sec] = rec

    # Backfill NSE trading symbols from the compact master (has SEM_TRADING_SYMBOL)
    try:
        c = requests.get("https://images.dhan.co/api-data/api-scrip-master.csv", timeout=240)
        idmap = {}
        for row in csv.DictReader(io.StringIO(c.text)):
            if (row.get("SEM_EXM_EXCH_ID") == "NSE"
                    and row.get("SEM_SEGMENT") == "E"
                    and row.get("SEM_INSTRUMENT_NAME") == "EQUITY"):
                idmap[(row.get("SEM_SMST_SECURITY_ID") or "").strip()] = \
                    (row.get("SEM_TRADING_SYMBOL") or "").strip().upper()
        for rec in by_isin.values():
            if rec["nse_id"] and rec["nse_id"] in idmap:
                rec["nse_symbol"] = idmap[rec["nse_id"]]
                by_symbol[rec["nse_symbol"]] = rec
    except Exception as e:
        print(f"[Registry] compact master backfill failed: {e}")

    return {"by_symbol": by_symbol, "by_isin": by_isin,
            "by_bse": by_bse, "built": time.time()}


def registry(force: bool = False) -> dict:
    global _reg
    if _reg and not force:
        return _reg
    if not force and _CACHE.exists() and time.time() - _CACHE.stat().st_mtime < _TTL:
        try:
            with open(_CACHE, "rb") as fh:
                _reg = pickle.load(fh)
                return _reg
        except Exception:
            pass
    print("[Registry] downloading instrument master (~40 MB, first run only)...")
    _reg = _download()
    try:
        with open(_CACHE, "wb") as fh:
            pickle.dump(_reg, fh)
    except Exception:
        pass
    print(f"[Registry] {len(_reg['by_isin']):,} instruments "
          f"({len(_reg['by_symbol']):,} NSE, {len(_reg['by_bse']):,} BSE)")
    return _reg


def resolve(query: str) -> dict | None:
    """
    Look up by NSE ticker, ISIN, or BSE scrip code — whichever you have.
    ISIN is the stable key; symbols change on rename, BSE codes rarely do.
    """
    if not query:
        return None
    q = str(query).strip().upper()
    r = registry()

    if q in r["by_symbol"]:
        return r["by_symbol"][q]
    if q in r["by_isin"]:
        return r["by_isin"][q]
    if q.isdigit() and q in r["by_bse"]:
        return r["by_bse"][q]

    # Fall back to a name search so a partial company name still finds it.
    hits = [v for v in r["by_isin"].values() if q in (v["name"] or "").upper()]
    return hits[0] if len(hits) == 1 else None


def search(query: str, limit: int = 10) -> list[dict]:
    """Fuzzy search across ticker, ISIN, BSE code and company name."""
    q = str(query).strip().upper()
    r = registry()
    out, seen = [], set()
    for v in r["by_isin"].values():
        hay = f"{v.get('nse_symbol') or ''} {v['isin']} {v.get('bse_code') or ''} {v['name'].upper()}"
        if q in hay and v["isin"] not in seen:
            seen.add(v["isin"])
            out.append(v)
            if len(out) >= limit:
                break
    return out


def audit_universe(universe: str = "all_nse") -> dict:
    """Find symbols in the project's universe that no longer trade on NSE."""
    from stocks_nse import get_all_symbols
    r = registry()
    syms = [s.replace(".NS", "").replace(".BO", "").upper()
            for s in get_all_symbols(universe)]
    stale = [s for s in syms if s not in r["by_symbol"]]
    return {"universe": universe, "total": len(syms),
            "ok": len(syms) - len(stale), "stale": sorted(stale)}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    if len(sys.argv) > 1:
        q = sys.argv[1]
        rec = resolve(q)
        if rec:
            print(f"\n  {rec['name']}")
            print(f"    NSE symbol : {rec.get('nse_symbol') or '—'}")
            print(f"    ISIN       : {rec['isin']}")
            print(f"    BSE code   : {rec.get('bse_code') or '—'}")
            print(f"    NSE id     : {rec.get('nse_id') or '—'}   series {rec.get('series') or '—'}\n")
        else:
            print(f"\n  no exact match for '{q}' — closest:")
            for h in search(q, 5):
                print(f"    {h.get('nse_symbol') or '?':<14} {h['isin']:<14} "
                      f"BSE {h.get('bse_code') or '—':<8} {h['name']}")
            print()
    else:
        for u in ("nifty50", "nifty500", "all_nse"):
            try:
                a = audit_universe(u)
            except Exception as e:
                print(f"  {u}: {e}")
                continue
            print(f"\n  {a['universe']:<10} {a['ok']}/{a['total']} resolve on NSE today")
            if a["stale"]:
                print(f"    stale / renamed ({len(a['stale'])}): {', '.join(a['stale'][:25])}")
                if len(a["stale"]) > 25:
                    print(f"    ...and {len(a['stale']) - 25} more")
        print()
