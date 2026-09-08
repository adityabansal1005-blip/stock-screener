"""
Crossing detector — the transition, not the level.

Why this exists
---------------
Measured 30 Aug 2026: the scanner's ranked list is not where its information
lives. Its top decile UNDERPERFORMS the universe. But the stocks that actually
ran were all visible at the moment they CROSSED the bar, not while sitting
above it:

    MOREPEN  crossed 60 on 09 Jun at Rs 48.6  ->  Rs 106.2   (+118% remaining)
    SHILPA   crossed 60 on 02 Jun at Rs 517.4 ->  Rs 905.5   (+75%  remaining)
    SUDEEP   crossed 60 on 09 Jul at Rs 802.3 ->  Rs 1155.0  (+44%  remaining)

Every one of those was flagged by the existing score. Nobody saw them because
`all_nse` had been scanned exactly ONCE (27 May, and only 777 of 2,559 symbols
were logged) while the scheduled scans covered 50 large caps. The failure was
operational, not analytical.

So this module answers one question: which stocks moved INTO the bar since the
last time we looked, and how fast.

Three honesty rules baked in
----------------------------
1. A symbol we have never scored before is NEW, never CROSSED. Otherwise the
   first all_india scan after a run of nifty50 scans would report ~2,600
   fictional crossings.
2. Duplicate scan runs are collapsed. `scan_log` contains hundreds of identical
   nifty50 rows written 12 seconds apart on 29 Aug by a loop that should not
   have been writing there; comparing consecutive raw runs would compare a scan
   to itself and find nothing, forever.
3. The prior observation is looked up per SYMBOL across universes, not per
   universe. The score does not depend on which list the stock was scanned in,
   so a nifty500 reading is a valid predecessor to an all_india one.

    python crossings.py              # crossings in the latest scan
    python crossings.py 55           # against a different bar
"""

from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

_DB = Path(__file__).parent / "signals.db"

# The bar. 60 is the current "GOOD SETUP" line in the dashboard. The 29 Aug
# expectancy study supports 60+ in BULL_TREND and 70+ in BULL_WEAK; it does not
# support trading SIDEWAYS or RECOVERY at any score. This module reports the
# crossing — it does not decide whether the regime makes it worth taking.
DEFAULT_BAR = 60.0

# How far back to look for a symbol's previous reading. Beyond this the two
# observations are too far apart for "crossed" to mean anything — a stock that
# was 45 three months ago and is 62 today did not cross today.
DEFAULT_LOOKBACK_DAYS = 21

_CREATE = """
CREATE TABLE IF NOT EXISTS crossings (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    detected_at  TEXT NOT NULL,
    symbol       TEXT NOT NULL,
    bar          REAL NOT NULL,
    prev_score   REAL,
    prev_at      TEXT,
    score        REAL,
    scanned_at   TEXT,
    days_between INTEGER,
    close        REAL,
    verdict      TEXT,
    sector       TEXT,
    universe     TEXT,
    alerted      INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ix_crossings_symbol ON crossings(symbol);
CREATE INDEX IF NOT EXISTS ix_crossings_at     ON crossings(detected_at);
"""


def init() -> None:
    conn = sqlite3.connect(str(_DB))
    try:
        for stmt in _CREATE.strip().split(";"):
            if stmt.strip():
                conn.execute(stmt)
        conn.commit()
    finally:
        conn.close()


# ── Observation history ─────────────────────────────────────────────────────

def _observations(days: int) -> dict[str, list[tuple[str, float, dict]]]:
    """
    One observation per symbol per calendar day, newest last.

    Collapsing to a day is what makes the duplicate-run pollution harmless: the
    29 Aug nifty50 loop wrote ~230 identical runs, which become one reading.
    Within a day the LAST scan wins — it saw the most price history.
    """
    since = (datetime.now() - timedelta(days=days + 2)).strftime("%Y-%m-%d")
    conn = sqlite3.connect(str(_DB))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT scanned_at, symbol, score100, close, verdict, sector, universe
            FROM scan_log
            WHERE score100 IS NOT NULL AND scanned_at >= ?
            ORDER BY scanned_at ASC
            """,
            (since,),
        ).fetchall()
    finally:
        conn.close()

    by_symbol: dict[str, dict[str, tuple[str, float, dict]]] = {}
    for r in rows:
        sym = (r["symbol"] or "").strip().upper()
        if not sym:
            continue
        day = r["scanned_at"][:10]
        by_symbol.setdefault(sym, {})[day] = (
            r["scanned_at"],
            float(r["score100"]),
            {
                "close": r["close"],
                "verdict": r["verdict"],
                "sector": r["sector"],
                "universe": r["universe"],
            },
        )

    return {
        sym: [days_[d] for d in sorted(days_)]
        for sym, days_ in by_symbol.items()
    }


# ── Detection ───────────────────────────────────────────────────────────────

def detect(bar: float = DEFAULT_BAR,
           lookback_days: int = DEFAULT_LOOKBACK_DAYS,
           results: list[dict] | None = None) -> dict:
    """
    Compare each symbol's newest reading to its previous one.

    `results` is an optional live scan-result list; when given, the newest
    reading comes from it rather than the DB (so this can run the moment a
    scan finishes, before the log is written) and each crossing is enriched
    with fields the log does not carry — liquidity above all.

    Returns four disjoint groups:
      crossed  — was below the bar, now at or above it. The signal.
      rising   — still below the bar but gained ground. The watchlist.
      lost     — was at or above, now below. Exit-relevant.
      new      — no prior reading. NOT a crossing; stated separately so it can
                 never be counted as one.
    """
    hist = _observations(lookback_days)

    latest: dict[str, tuple[str, float, dict]] = {}
    if results:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for r in results:
            sym = (r.get("symbol") or "").strip().upper()
            sc = r.get("score100")
            if not sym or sc is None:
                continue
            latest[sym] = (now, float(sc), r)
    else:
        for sym, obs in hist.items():
            latest[sym] = obs[-1]

    crossed, rising, lost, new = [], [], [], []
    cutoff = datetime.now() - timedelta(days=lookback_days)

    for sym, (at, score, meta) in latest.items():
        # The previous reading is the newest one from a DIFFERENT day.
        prior = None
        for p_at, p_score, p_meta in reversed(hist.get(sym, [])):
            if p_at[:10] != at[:10]:
                prior = (p_at, p_score, p_meta)
                break

        if prior is None:
            if score >= bar:
                new.append(_row(sym, None, None, score, at, meta, bar))
            continue

        p_at, p_score, _ = prior
        try:
            p_dt = datetime.strptime(p_at[:19], "%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
        if p_dt < cutoff:
            # Too stale to call a crossing; treat as a first sighting.
            if score >= bar:
                new.append(_row(sym, None, None, score, at, meta, bar))
            continue

        row = _row(sym, p_score, p_at, score, at, meta, bar)
        if p_score < bar <= score:
            crossed.append(row)
        elif score < bar <= p_score:
            lost.append(row)
        elif score < bar and score - p_score >= 5:
            rising.append(row)

    key = lambda r: (-(r["delta"] or 0), -r["score"])
    crossed.sort(key=key)
    rising.sort(key=key)
    lost.sort(key=lambda r: r["delta"] or 0)
    new.sort(key=lambda r: -r["score"])

    return {
        "bar": bar,
        "lookback_days": lookback_days,
        "as_of": max((v[0] for v in latest.values()), default=None),
        "symbols_compared": len(latest),
        "crossed": crossed,
        "rising": rising,
        "lost": lost,
        "new": new,
    }


def _row(sym, p_score, p_at, score, at, meta, bar) -> dict:
    days = None
    if p_at:
        try:
            days = (datetime.strptime(at[:19], "%Y-%m-%d %H:%M:%S")
                    - datetime.strptime(p_at[:19], "%Y-%m-%d %H:%M:%S")).days
        except ValueError:
            pass
    delta = None if p_score is None else round(score - p_score, 1)

    # Velocity separates "woke up this week" from "drifted up over a month".
    # Morepen went 45 -> 62 in a single session; that is a different event from
    # the same 17 points spread across 20 sessions.
    velocity = None
    if delta is not None and days:
        velocity = round(delta / max(days, 1), 2)

    return {
        "symbol": sym,
        "score": round(score, 1),
        "prev_score": None if p_score is None else round(p_score, 1),
        "delta": delta,
        "days_between": days,
        "velocity": velocity,
        "scanned_at": at,
        "prev_at": p_at,
        "close": meta.get("close"),
        "live_price": meta.get("live_price"),
        "verdict": meta.get("verdict"),
        "sector": meta.get("sector"),
        "universe": meta.get("universe"),
        "liq_turnover": meta.get("liq_turnover"),
        "liq_tradeable": meta.get("liq_tradeable"),
        "ai_verdict": meta.get("ai_verdict"),
        "margin": round(score - bar, 1),
    }


# ── Persistence + alerting ──────────────────────────────────────────────────

def record(crossed: list[dict], bar: float = DEFAULT_BAR) -> list[dict]:
    """
    Write crossings and return only the ones not already recorded today.

    De-duplication matters more than it looks: the auto-scan loop can run the
    same universe several times a day, and a stock sitting at 61 would be
    re-alerted on every run if the newest reading were compared against a
    day-old one each time. One alert per symbol per day per bar.
    """
    if not crossed:
        return []
    init()
    today = datetime.now().strftime("%Y-%m-%d")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fresh = []
    conn = sqlite3.connect(str(_DB))
    try:
        for c in crossed:
            seen = conn.execute(
                "SELECT 1 FROM crossings WHERE symbol=? AND bar=? "
                "AND detected_at >= ? LIMIT 1",
                (c["symbol"], bar, today),
            ).fetchone()
            if seen:
                continue
            conn.execute(
                "INSERT INTO crossings (detected_at, symbol, bar, prev_score, "
                "prev_at, score, scanned_at, days_between, close, verdict, "
                "sector, universe, alerted) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,0)",
                (now, c["symbol"], bar, c["prev_score"], c["prev_at"],
                 c["score"], c["scanned_at"], c["days_between"], c["close"],
                 c["verdict"], c["sector"], c["universe"]),
            )
            fresh.append(c)
        conn.commit()
    finally:
        conn.close()
    return fresh


def history(days: int = 90, symbol: str = "") -> list[dict]:
    """Past crossings — the record that lets tomorrow grade today's calls."""
    init()
    since = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    conn = sqlite3.connect(str(_DB))
    conn.row_factory = sqlite3.Row
    try:
        sql = "SELECT * FROM crossings WHERE detected_at >= ?"
        params: list = [since]
        if symbol:
            sql += " AND symbol = ?"
            params.append(symbol.strip().upper())
        sql += " ORDER BY detected_at DESC"
        return [dict(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


def format_alert(crossed: list[dict], bar: float = DEFAULT_BAR) -> str:
    """Telegram body. Deliberately states the transition, not the level."""
    if not crossed:
        return ""
    lines = [f"*{len(crossed)} stock(s) crossed into {bar:.0f}+*", ""]
    for c in crossed[:15]:
        px = c.get("live_price") or c.get("close")
        move = (f"{c['prev_score']:.0f} → {c['score']:.0f}"
                if c["prev_score"] is not None else f"{c['score']:.0f}")
        line = f"`{c['symbol']:<12}` {move}"
        if c.get("days_between"):
            line += f"  in {c['days_between']}d"
        if px:
            line += f"  ₹{px:,.1f}"
        if c.get("liq_tradeable") is False:
            line += "  ⚠ illiquid"
        lines.append(line)
    if len(crossed) > 15:
        lines.append(f"_…and {len(crossed) - 15} more_")
    lines += ["", "_A crossing is an event, not a recommendation._"]
    return "\n".join(lines)


def check_and_alert(results: list[dict] | None = None,
                    bar: float = DEFAULT_BAR) -> dict:
    """
    Full pipeline: detect → record → notify. Safe to call after every scan.
    Returns the detection payload with `alerted` added.
    """
    out = detect(bar=bar, results=results)
    fresh = record(out["crossed"], bar=bar)
    out["alerted"] = len(fresh)

    if fresh:
        try:
            import config
            from notifier import send_telegram
            token = getattr(config, "TELEGRAM_BOT_TOKEN", "")
            chat = getattr(config, "TELEGRAM_CHAT_ID", "")
            if token and chat and token != "YOUR_BOT_TOKEN":
                send_telegram(token, chat, format_alert(fresh, bar))
        except Exception as e:
            print(f"[Crossings] alert failed: {e}")
    return out


# ── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    bar = float(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_BAR
    out = detect(bar=bar)

    print(f"\n{'=' * 66}")
    print(f"  CROSSINGS INTO {bar:.0f}+   ({out['symbols_compared']:,} symbols "
          f"with a reading, {out['lookback_days']}d lookback)")
    print(f"  as of {out['as_of']}")
    print("=" * 66)

    def dump(title: str, rows: list[dict], n: int = 20) -> None:
        print(f"\n  {title}  [{len(rows)}]")
        if not rows:
            print("    none")
            return
        for r in rows[:n]:
            prev = f"{r['prev_score']:.0f}" if r["prev_score"] is not None else "  -"
            d = f"{r['days_between']}d" if r["days_between"] else "-"
            px = f"₹{r['close']:,.1f}" if r["close"] else "-"
            print(f"    {r['symbol']:<14}{prev:>4} → {r['score']:>4.0f}"
                  f"{d:>6}{px:>13}   {str(r['verdict'] or '')[:22]}")
        if len(rows) > n:
            print(f"    …{len(rows) - n} more")

    dump(f"CROSSED into {bar:.0f}+", out["crossed"])
    dump("RISING (below bar, +5 or more)", out["rising"], 10)
    dump(f"LOST {bar:.0f}+", out["lost"], 10)
    dump("FIRST SIGHTING (no prior reading — NOT a crossing)", out["new"], 10)
    print()
