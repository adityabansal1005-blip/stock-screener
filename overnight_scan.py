"""
Overnight full-universe scan — runs the same pipeline as the server's scheduled
scan, but in the foreground with logging so a failure is visible rather than
swallowed by a detached thread.

Writes results to the SQLite scan cache, which the dashboard loads on startup —
so the morning dashboard shows this scan without re-running anything.

    python overnight_scan.py                # all_india (NSE 2,559 + BSE 152)
    python overnight_scan.py all_nse
    python overnight_scan.py nifty500 --no-ai
"""

import sys
import time
import traceback
import warnings

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")


def main() -> int:
    universe = "all_india"
    do_ai = True
    for a in sys.argv[1:]:
        if a == "--no-ai":
            do_ai = False
        elif not a.startswith("-"):
            universe = a

    t0 = time.time()
    print("=" * 68)
    print(f"  OVERNIGHT SCAN — {universe}")
    print("=" * 68)

    import config
    import groww_client
    import db

    if not groww_client.init_groww(config.GROWW_API_KEY, config.GROWW_API_SECRET):
        print("[WARN] Groww not connected — falling back to nselib (much slower)")

    db.init_db()

    import technical_enhanced as TE
    import backtester
    from stocks_nse import get_all_symbols
    from scanner import scan_universe, enrich_with_backtest, enrich_with_ai

    syms = get_all_symbols(universe)
    print(f"  universe        : {len(syms):,} symbols")
    print(f"  momentum mode   : {TE.MOMENTUM_MODE}")
    print(f"  hold period     : {backtester.FORWARD_DAYS} sessions")
    print(f"  round-trip cost : {backtester.ROUND_TRIP * 100:.4f}%")
    print(f"  AI phase        : {'on' if do_ai else 'off'}\n")

    # ── Phase 1: score ──────────────────────────────────────────────────
    print("[1/3] Scoring...")
    p1 = time.time()
    try:
        results = scan_universe(universe)
    except Exception:
        traceback.print_exc()
        return 1
    print(f"[1/3] done — {len(results):,} scored in {(time.time()-p1)/60:.1f} min")
    if not results:
        print("[FATAL] no results")
        return 1

    # Persist immediately: a crash in a later phase must not lose the scan.
    try:
        db.save_full_scan(results, universe=universe, scan_type="swing")
        print(f"      cached to DB ({len(results):,} stocks)")
    except Exception as e:
        print(f"      [WARN] cache write failed: {e}")

    # ── Crossings ───────────────────────────────────────────────────────
    # Runs between Phase 1 and Phase 2 for the same reason the server does it
    # there: the crossing is known as soon as scoring ends.
    try:
        import crossings
        cx = crossings.check_and_alert(results)
        print(f"\n      crossings into {cx['bar']:.0f}+ : {len(cx['crossed'])} "
              f"({cx['alerted']} newly alerted)")
        for c in cx["crossed"][:12]:
            print(f"        {c['symbol']:<14}{c['prev_score']:>4.0f} → "
                  f"{c['score']:>4.0f}  in {c['days_between'] or '?'}d")
        print(f"      first sightings (no prior reading, NOT crossings): "
              f"{len(cx['new'])}")
    except Exception as e:
        print(f"      [WARN] crossings failed: {e}")

    top = sorted(results, key=lambda r: r.get("score100") or 0, reverse=True)[:15]
    print("\n      top 15 by score:")
    for r in top:
        liq = r.get("liq_turnover")
        print(f"        {r.get('symbol',''):<14}{r.get('score100',0):>4}  "
              f"{str(r.get('verdict',''))[:18]:<20}"
              f"{('Rs' + format(liq, ',')) if liq else '-':>16}")

    # ── Phase 2: backtest ───────────────────────────────────────────────
    print("\n[2/3] Backtesting...")
    p2 = time.time()
    try:
        results = enrich_with_backtest(results)
        print(f"[2/3] done in {(time.time()-p2)/60:.1f} min")
    except Exception:
        traceback.print_exc()
        print("[2/3] failed — continuing with Phase 1 results")

    # ── Phase 3: AI ─────────────────────────────────────────────────────
    if do_ai:
        print("\n[3/3] AI analysis on top-N...")
        p3 = time.time()
        try:
            results = enrich_with_ai(results)
            n = sum(1 for r in results if r.get("ai_verdict"))
            print(f"[3/3] done in {(time.time()-p3)/60:.1f} min — {n} verdicts")
        except Exception:
            traceback.print_exc()
            print("[3/3] failed — continuing")
    else:
        print("\n[3/3] skipped")

    try:
        db.save_full_scan(results, universe=universe, scan_type="swing")
        db.save_scan_results(results, universe=universe, scan_type="swing")
        print("\n      final results saved to DB + scan_log")
    except Exception as e:
        print(f"\n      [WARN] final save failed: {e}")

    good = [r for r in results if (r.get("score100") or 0) >= 60]
    tradeable = [r for r in good if r.get("liq_tradeable")]
    print(f"\n{'=' * 68}")
    print(f"  COMPLETE in {(time.time()-t0)/60:.1f} min")
    print(f"  {len(results):,} scanned | {len(good)} scored 60+ | "
          f"{len(tradeable)} of those pass the liquidity gate")
    print(f"{'=' * 68}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
