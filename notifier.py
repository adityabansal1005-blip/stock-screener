import urllib.request
import urllib.parse
import json
import datetime


def send_telegram(bot_token: str, chat_id: str, message: str) -> bool:
    if not bot_token or bot_token == "YOUR_BOT_TOKEN":
        print("[Telegram] Not configured — skipping notification")
        return False
    if not chat_id or chat_id == "YOUR_CHAT_ID":
        print("[Telegram] Chat ID not set — skipping notification")
        return False
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = json.dumps({
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "Markdown",
        }).encode("utf-8")
        req = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status == 200
    except Exception as e:
        print(f"[Telegram] Failed: {e}")
        return False


def _verdict_emoji(verdict: str) -> str:
    v = (verdict or "").upper()
    if "STRONG BUY" in v:  return "🚀"
    if "BUY" in v:         return "✅"
    if "STRONG SELL" in v: return "🔴"
    if "SELL" in v:        return "⚠️"
    return "⏸"


def format_alert(stocks: list[dict]) -> str:
    now = datetime.datetime.now().strftime("%d %b %Y, %I:%M %p")

    # Market context from first stock (all carry it)
    market = stocks[0].get("market", {}) if stocks else {}
    nifty_trend = market.get("nifty_trend", "")
    vix = market.get("vix")
    nifty_price = market.get("nifty_price")
    nifty_chg = market.get("nifty_change")

    lines = [f"*AI Stock Alert* — {now}\n"]

    # Market banner
    if nifty_price:
        chg_str = f"{nifty_chg:+.2f}%" if nifty_chg is not None else ""
        vix_str = f"  |  VIX: {vix}" if vix else ""
        lines.append(f"_Nifty: ₹{nifty_price} ({chg_str}) — {nifty_trend}{vix_str}_\n")

    strong_buy = [s for s in stocks if s.get("score", 0) >= 4]
    buy        = [s for s in stocks if 2 <= s.get("score", 0) < 4]

    if strong_buy:
        lines.append("*🚀 STRONG BUY*")
        for s in strong_buy[:5]:
            ai_v = s.get("ai_verdict")
            ai_c = s.get("ai_confidence")
            ai_tag = f" | AI: {ai_v} ({ai_c}%)" if ai_v else ""
            lines.append(
                f"• *{s['symbol']}* ₹{s['close']}{ai_tag}\n"
                f"  RSI {s['rsi']} | SL ₹{s['sl_buy']} | T1 ₹{s['target1']} | T2 ₹{s['target2']}"
            )
            reasoning = s.get("ai_reasoning")
            if reasoning:
                lines.append(f"  _{reasoning}_")
            lines.append("")

    if buy:
        lines.append("*✅ BUY*")
        for s in buy[:5]:
            ai_v = s.get("ai_verdict")
            ai_c = s.get("ai_confidence")
            ai_tag = f" | AI: {ai_v} ({ai_c}%)" if ai_v else ""
            lines.append(
                f"• *{s['symbol']}* ₹{s['close']}{ai_tag}\n"
                f"  RSI {s['rsi']} | SL ₹{s['sl_buy']} | T1 ₹{s['target1']}"
            )

    lines.append("\n_Not investment advice. Do your own research._")
    return "\n".join(lines)
