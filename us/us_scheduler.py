"""
美股多空雷達 V9.9 ─ LINE 推播排程

兩種使用方式：
1) 常駐模式（Render Background Worker / 自己的主機）：
       python us_scheduler.py
   以「美東時間」排程，夏令/冬令自動處理，不再需要手動判斷 DST。

2) 單次執行模式（GitHub Actions cron、或手動測試）：
       python us_scheduler.py auto      # 依目前美東時間自動判斷該跑哪一階段，不在時段內就不動作
       python us_scheduler.py stage1    # 開盤後半小時
       python us_scheduler.py stage2    # 午盤（已停止排程，只留手動測試用）
       python us_scheduler.py stage3    # 收盤結算
       python us_scheduler.py test      # 只送一則測試訊息
       python us_scheduler.py flextest  # 送一張白頭海鵰卡片，確認卡片版推播正常

需要的套件：requests pandas numpy yfinance apscheduler tzdata
需要的環境變數：LINE_CHANNEL_TOKEN、LINE_USER_ID
選填：ASSET_BASE（海鵰圖片所在的公開網址資料夾，預設為本 repo 的 assets/）
"""
import os
import sys
import time
import logging
import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf

import risk_plan  # 大盤狀態／移動停損階段／部位試算（us/risk_plan.py）
import dip_scan  # 抄底模式：上升趨勢拉回（us/dip_scan.py，網頁與推播共用）

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("us_scheduler")

LINE_CHANNEL_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_TOKEN", "").strip()
LINE_USER_ID = os.environ.get("LINE_USER_ID", "").strip()
APP_URL = os.environ.get("US_APP_URL", "").strip() or "https://pro-terminal.onrender.com"  # GitHub Actions 未設 secret 時會傳入空字串

ET = ZoneInfo("America/New_York")
TPE = ZoneInfo("Asia/Taipei")

# (代號, 名稱, 板塊)；與網頁版同一個股票池
UNIVERSE = [
    ("NVDA", "輝達", "SOX"), ("TSM", "台積電ADR", "SOX"), ("AMD", "超微", "SOX"), ("AVGO", "博通", "SOX"),
    ("QCOM", "高通", "SOX"), ("MU", "美光", "SOX"), ("INTC", "英特爾", "SOX"), ("ARM", "安謀", "SOX"),
    ("ASML", "艾司摩爾", "SOX"), ("TXN", "德儀", "SOX"),
    ("AAPL", "蘋果", "NASDAQ"), ("MSFT", "微軟", "NASDAQ"), ("GOOGL", "Google", "NASDAQ"), ("AMZN", "亞馬遜", "NASDAQ"),
    ("META", "Meta", "NASDAQ"), ("TSLA", "特斯拉", "NASDAQ"), ("NFLX", "Netflix", "NASDAQ"), ("COST", "Costco", "NASDAQ"),
    ("ADBE", "Adobe", "NASDAQ"), ("COIN", "Coinbase", "NASDAQ"),
    ("CAT", "開拓重工", "DOW"), ("BA", "波音", "DOW"), ("UNH", "聯合健康", "DOW"), ("JNJ", "嬌生", "DOW"),
    ("MCD", "麥當勞", "DOW"), ("HD", "家得寶", "DOW"), ("V", "Visa", "DOW"), ("WMT", "沃爾瑪", "DOW"),
    ("IBM", "IBM", "DOW"), ("GS", "高盛", "DOW"),
    ("PLTR", "Palantir", "NYSE"), ("JPM", "摩根大通", "NYSE"), ("BAC", "美國銀行", "NYSE"), ("XOM", "埃克森", "NYSE"),
    ("CVX", "雪佛龍", "NYSE"), ("LLY", "禮來", "NYSE"), ("NKE", "Nike", "NYSE"), ("UBER", "Uber", "NYSE"),
    ("SPY", "標普500ETF", "ETF"), ("QQQ", "那指100ETF", "ETF"), ("SOXX", "費半ETF", "ETF"),
    ("DIA", "道瓊ETF", "ETF"), ("IWM", "小型股ETF", "ETF"),
]
NAME = {s: n for s, n, _ in UNIVERSE}
SYMBOLS = [s for s, _, _ in UNIVERSE]
WATCH_CORE = ["NVDA", "TSLA", "AAPL", "MSFT", "PLTR", "AMZN", "AMD", "SPY", "QQQ"]  # 開盤/午盤快報的核心觀察股

NECK_DAYS = int(os.environ.get("NECK_DAYS", "20"))
VOL_RATIO = float(os.environ.get("VOL_RATIO", "1.2"))
TOP_N = int(os.environ.get("TOP_N", "6"))
DIP_TOP_N = int(os.environ.get("DIP_TOP_N", "5"))          # 收盤戰報最多列幾檔抄底標的（跌越深越前面）
DIP_DEFAULT_CAPITAL = 91000.0                               # 沒設定 ACCOUNT_USD 時，抄底股數以 Joe 提供的約 91,000 美元試算（卡片會註明）


# ------------------------------------------------------------------------------
# LINE
# ------------------------------------------------------------------------------
def send_line_message(message: str) -> bool:
    if not LINE_CHANNEL_ACCESS_TOKEN or not LINE_USER_ID:
        log.warning("未設定 LINE 憑證，以下為模擬推播：\n%s", message)
        return False
    # LINE 單則文字上限 5000 字，保守切成 4500
    chunks = [message[i:i + 4500] for i in range(0, len(message), 4500)] or [message]
    ok_all = True
    for ch in chunks:
        try:
            r = requests.post(
                "https://api.line.me/v2/bot/message/push",
                headers={"Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}", "Content-Type": "application/json"},
                json={"to": LINE_USER_ID, "messages": [{"type": "text", "text": ch}]},
                timeout=10,
            )
            log.info("LINE 回應 %s %s", r.status_code, r.text[:120])
            ok_all = ok_all and r.status_code == 200
        except Exception as e:
            log.error("LINE 推播失敗：%s", e)
            ok_all = False
    return ok_all


# ------------------------------------------------------------------------------
# LINE Flex（卡片式戰報，白頭海鵰主題）
# 圖片必須是公開 https 網址；預設放在本 repo 的 assets/。
# repo 改成 Private 後 raw 網址會失效 → 把 assets/ 放到另一個公開 repo，並設定 ASSET_BASE。
# ------------------------------------------------------------------------------
ASSET_BASE = (os.environ.get("ASSET_BASE", "").strip()
              or "https://raw.githubusercontent.com/Joesung0719/pro-terminal/main/assets").rstrip("/")
IMG_HEADER = f"{ASSET_BASE}/eagle_header.png"
IMG_WATERMARK = f"{ASSET_BASE}/eagle_watermark.png"

NAVY, NAVY_SUB, PAGE_BG, CARD_LINE = "#0b2a52", "#b9cdea", "#eef3fa", "#cddbef"
UP, DOWN, MUTED, INFO = "#1f7a3a", "#c0281e", "#5f5e5a", "#185fa5"   # 美股慣例：綠漲紅跌


def _t(text, size="md", color="#1a1a1a", bold=False, **kw):
    d = {"type": "text", "text": str(text), "size": size, "color": color, "wrap": True}
    if bold:
        d["weight"] = "bold"
    d.update(kw)
    return d


def _box(layout, contents, **kw):
    d = {"type": "box", "layout": layout, "contents": contents}
    d.update(kw)
    return d


def _card(contents, **kw):
    return _box("vertical", contents, backgroundColor="#ffffff", cornerRadius="12px",
                borderWidth="1px", borderColor=CARD_LINE, paddingAll="14px", **kw)


def _flex_header(title, sub):
    stripe = _box("horizontal", [_box("vertical", [], flex=1, backgroundColor=c)
                                 for c in ("#b22234", "#ffffff", "#b22234", "#ffffff", "#b22234")], height="6px")
    top = _box("horizontal", [
        _box("vertical", [_t(title, "xl", "#ffffff", True), _t(sub, "sm", NAVY_SUB, margin="sm")],
             flex=7, justifyContent="center"),
        _box("vertical", [{"type": "image", "url": IMG_HEADER, "size": "full",
                           "aspectRatio": "3:2", "aspectMode": "fit"}], flex=2, justifyContent="center"),
    ], backgroundColor=NAVY, paddingAll="16px", spacing="md")
    return _box("vertical", [top, stripe], paddingAll="0px")


def _flex_index_card(idx_rows, vix):
    rows = []
    for r in idx_rows:
        if r.get("pct") is None:
            rows.append(_box("horizontal", [_t(r["label"], "lg", NAVY, flex=3), _t("暫無資料", "md", MUTED, flex=2, align="end")]))
            continue
        color = UP if r["pct"] >= 0 else DOWN
        arrow = "▲" if r["pct"] >= 0 else "▼"
        rows.append(_box("horizontal", [
            _t(f"{r['label']} {r['close']:,.0f}", "lg", NAVY, flex=3),
            _t(f"{arrow}{abs(r['pct']):.2f}%", "lg", color, True, flex=2, align="end"),
        ]))
    if vix:
        rows.append(_box("horizontal", [_t(f"VIX {vix['value']:.2f}", "lg", NAVY, flex=3),
                                        _t(vix["state"], "md", MUTED, flex=2, align="end")]))
    return _card(rows, spacing="sm")


def _market_state():
    """SPY／QQQ 近 1 年日K → 大盤狀態；抓不到回傳空串列（卡片顯示「—」，不捏造）。"""
    try:
        raw = yf.download("SPY QQQ", period="1y", group_by="ticker", threads=True, progress=False, auto_adjust=False)
        return risk_plan.market_state(_split(raw, ["SPY", "QQQ"]))
    except Exception as e:
        log.warning("大盤狀態略過：%s", e)
        return []


def _ma_phrase(ms):
    return "—" if not ms else f"{'上' if ms['above'] else '下'} {ms['dist']:+.1f}%"


def _market_text(state):
    if not state:
        return "🧭 大盤狀態 —（暫時抓不到）"
    lines = ["🧭 大盤狀態（僅標示，不過濾名單）"]
    for m in state:
        lines.append(f"{m['symbol']} {m['close']:,.2f}｜季線{_ma_phrase(m['q'])}｜200日線{_ma_phrase(m['y'])}")
    lab = risk_plan.market_label(state)
    if lab:
        lines.append(lab)
    return "\n".join(lines)


def _flex_market_card(state):
    if not state:
        return _card([_t("🧭 大盤狀態 —（暫時抓不到）", "md", MUTED)])
    rows = [_t("🧭 大盤狀態（僅標示，不過濾名單）", "lg", NAVY, True)]
    for m in state:
        for label, ms in (("季線", m["q"]), ("200日線", m["y"])):
            color = MUTED if not ms else (UP if ms["above"] else DOWN)
            # LINE 不接受空字串文字（會讓整張卡片 400 退回純文字），沒有左欄文字時改放空白區塊
            left = (_t(f"{m['symbol']} {m['close']:,.2f}", "md", NAVY, True, flex=3) if label == "季線"
                    else _box("vertical", [], flex=3))
            rows.append(_box("horizontal", [left,
                                            _t(f"{label}{_ma_phrase(ms)}", "md", color, True, flex=3, align="end")]))
    lab = risk_plan.market_label(state)
    if lab:
        up = all(m["q"]["above"] for m in state if m.get("q"))
        rows.append(_t(lab, "md", UP if up else DOWN, True, margin="sm"))
    rows.append(_t("美股 5 年回測：大盤在季線下期望值仍為正（不像台股為負），所以只標示、不過濾名單。",
                   "sm", MUTED, margin="sm"))
    return _card(rows, spacing="xs")


def _plan_flex(r):
    """多方卡片下方：移動停損階段（%）＋部位試算。回傳 Flex 元素列表。"""
    out = [_t("移動停損（% 以實際成交價計）", "md", NAVY, True, margin="md")]
    out += [_t(x, "md", "#1a1a1a", margin="xs") for x in risk_plan.plan_lines(r.get("plan"))]
    sz = risk_plan.size_text(r)
    if sz:
        out.append(_t(sz, "md", DOWN if r.get("over_cap") else INFO, True, margin="sm"))
    return out


def _plan_text(r):
    s = "\n".join("   " + x for x in risk_plan.plan_lines(r.get("plan")))
    sz = risk_plan.size_text(r)
    return "\n   📐 移動停損（% 以實際成交價計）\n" + s + (f"\n   💰 {sz}" if sz else "")


def _risk_footer_lines(capital, alloc):
    """收盤卡片的風險提醒；沒設定本金就明講，不放假數字。"""
    if not capital:
        head = "💰 部位試算：未設定本金（設定 ACCOUNT_USD 後會顯示股數；網頁版可直接輸入）"
    else:
        head = (f"💰 本金 ${capital:,.0f}｜每筆最多虧 {risk_plan.RISK_PER_TRADE_PCT:g}%｜"
                f"同時總風險上限 {risk_plan.TOTAL_RISK_CAP_PCT:g}%（最多 {int(risk_plan.TOTAL_RISK_CAP_PCT // risk_plan.RISK_PER_TRADE_PCT)} 檔）")
        if alloc and alloc["skipped"]:
            head += f"｜今日名單 {alloc['skipped']} 檔超過上限"
    return [head, f"🛑 月虧達本金 {risk_plan.MONTH_STOP_PCT:g}% 就停手（需自行對帳）｜同日標的高度相關，勿全開"]


def _event_texts(lines):
    """事件提醒文字列：緊急（3 天內）用琥珀色粗體，其餘深藍。"""
    return [_t(txt, "md", "#a8650a" if urgent else NAVY, True, margin="xs") for txt, urgent in (lines or [])]


def _flex_macro_card(lines):
    body = [_t("📅 未來 7 天重大事件", "lg", NAVY, True)]
    body += _event_texts(lines) if lines else [_t("未來 7 天無 FOMC／CPI／非農公布", "md", MUTED, margin="xs")]
    return _card(body, spacing="xs")


def _get_events(symbols):
    """總經事件＋個股事件；任何錯誤都不中斷推播（個股抓不到時顯示「—」）。"""
    macro, per = None, {}
    try:
        from us_events import macro_events, stock_events
        try:
            macro = macro_events()
        except Exception as e:
            log.warning("總經事件略過：%s", e)
        try:
            per = stock_events(symbols)
        except Exception as e:
            log.warning("個股事件略過：%s", e)
            per = {s: [("📌 事件提醒 —", False)] for s in symbols}
    except Exception as e:
        log.warning("事件模組載入失敗：%s", e)
    return macro, per


def _macro_block(macro):
    if macro is None:
        return [_card([_t("📅 重大事件 —（暫時抓不到）", "md", MUTED)])]
    return [_flex_macro_card(macro)]


def _events_text(lines, indent="   "):
    return "".join(f"\n{indent}{txt}" for txt, _ in (lines or []))


def _macro_text(macro):
    if macro is None:
        return "📅 重大事件 —"
    return "📅 未來 7 天重大事件：" + ("".join(f"\n{txt}" for txt, _ in macro) if macro else "無 FOMC／CPI／非農公布")


def _flex_pick_card(r, mode, events=None):
    accent = UP if mode == "LONG" else DOWN
    pct_color = UP if r["pct"] >= 0 else DOWN
    arrow = "▲" if r["pct"] >= 0 else "▼"
    tag = "首日" if r["first"] else "延續"
    body = [
        _box("horizontal", [
            _t(f"{r['symbol']} {NAME.get(r['symbol'], '')}", "lg", NAVY, True, flex=5),
            _t(f"${r['price']:.2f} {arrow}{abs(r['pct']):.1f}%", "lg", pct_color, True, flex=5, align="end"),
        ]),
        _t(f"{tag}｜量比 {r['ratio']:.1f}x｜距{'頸線' if mode == 'LONG' else '跌破線'} {r['dist']:+.1f}%",
           "md", INFO, margin="sm"),
    ]
    if r["dist"] > 5:
        body.append(_t("⚠️ 距離偏遠，追價風險高", "md", "#a8650a", margin="xs"))
    body.append(_t(f"停損參考：${r['neck']:.2f} ／ 月線 ${r['ma20']:.2f}", "md", "#444444", margin="xs"))
    if mode == "LONG":
        body += _plan_flex(r)
    body += _event_texts(events)
    return _box("horizontal", [
        _box("vertical", [], width="6px", backgroundColor=accent),
        _box("vertical", body, flex=1, paddingAll="12px"),
    ], backgroundColor="#ffffff", cornerRadius="12px", borderWidth="1px", borderColor=CARD_LINE)


def _flex_mover_card(r, events=None):
    color = UP if r["vs_prev"] >= 0 else DOWN
    arrow = "▲" if r["vs_prev"] >= 0 else "▼"
    return _box("horizontal", [
        _box("vertical", [], width="6px", backgroundColor=color),
        _box("vertical", [
            _box("horizontal", [
                _t(f"{r['symbol']} {NAME.get(r['symbol'], '')}", "lg", NAVY, True, flex=5),
                _t(f"${r['last']:.2f} {arrow}{abs(r['vs_prev']):.1f}%", "lg", color, True, flex=5, align="end"),
            ]),
            _t(f"較昨收 {r['vs_prev']:+.1f}%｜較今開 {r['vs_open']:+.1f}%", "md", INFO, margin="sm"),
        ] + _event_texts(events), flex=1, paddingAll="12px"),
    ], backgroundColor="#ffffff", cornerRadius="12px", borderWidth="1px", borderColor=CARD_LINE)


def _section_title(text, color):
    return _t(text, "lg", color, True, margin="xl")


def _flex_bubble(title, body_contents, foot_note):
    watermark = _box("vertical", [{"type": "image", "url": IMG_WATERMARK, "size": "full",
                                   "aspectRatio": "3:2", "aspectMode": "fit"}],
                     position="absolute", offsetEnd="0px", offsetBottom="0px", width="70%")
    return {
        "type": "bubble",
        "size": "giga",
        "header": _flex_header(title, header_line()),
        "body": _box("vertical", [watermark] + body_contents,
                     backgroundColor=PAGE_BG, paddingAll="14px", spacing="md"),
        "footer": _box("vertical", [
            {"type": "button", "style": "primary", "color": NAVY, "height": "md",
             "action": {"type": "uri", "label": "打開美股戰情室", "uri": APP_URL}},
            _t(foot_note, "sm", MUTED, align="center", margin="md"),
        ], backgroundColor=PAGE_BG, paddingAll="14px"),
        "styles": {"header": {"backgroundColor": NAVY}},
    }


def send_line_flex(alt_text: str, bubble: dict, fallback_text: str) -> bool:
    """送 Flex 卡片；失敗（格式錯誤、圖片網址失效等）就自動改送原本的純文字版，確保一定收得到。"""
    if not LINE_CHANNEL_ACCESS_TOKEN or not LINE_USER_ID:
        import json
        log.warning("未設定 LINE 憑證，Flex JSON 約 %d bytes：\n%s", len(json.dumps(bubble, ensure_ascii=False).encode()),
                    json.dumps(bubble, ensure_ascii=False)[:2000])
        return False
    try:
        r = requests.post(
            "https://api.line.me/v2/bot/message/push",
            headers={"Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}", "Content-Type": "application/json"},
            json={"to": LINE_USER_ID, "messages": [{"type": "flex", "altText": alt_text[:400], "contents": bubble}]},
            timeout=10,
        )
        log.info("LINE Flex 回應 %s %s", r.status_code, r.text[:300])
        if r.status_code == 200:
            return True
    except Exception as e:
        log.error("LINE Flex 推播失敗：%s", e)
    log.warning("Flex 失敗，改送純文字版")
    return send_line_message(fallback_text)


# ------------------------------------------------------------------------------
# 資料
# ------------------------------------------------------------------------------
def _strip_tz(df: pd.DataFrame) -> pd.DataFrame:
    if getattr(df.index, "tz", None) is not None:
        df = df.copy()
        df.index = df.index.tz_localize(None)
    return df


def _split(raw, symbols, keep_tz=False):
    out = {}
    if raw is None or raw.empty:
        return out
    multi = isinstance(raw.columns, pd.MultiIndex)
    lvl0 = set(raw.columns.get_level_values(0)) if multi else set()
    for s in symbols:
        sub = None
        if multi and s in lvl0:
            sub = raw[s].copy()
        elif (not multi) and len(symbols) == 1:
            sub = raw.copy()
        if sub is not None and "Close" in sub.columns:
            sub = sub.dropna(subset=["Close"])
            if not sub.empty:
                out[s] = sub if keep_tz else _strip_tz(sub)
    return out


def download_daily(symbols, period="6mo"):
    for attempt in range(3):
        try:
            raw = yf.download(" ".join(symbols), period=period, group_by="ticker",
                              threads=True, progress=False, auto_adjust=False)
            data = _split(raw, symbols)
            if len(data) >= len(symbols) * 0.6:
                return data
        except Exception as e:
            log.warning("日K下載失敗（第 %d 次）：%s", attempt + 1, e)
        time.sleep(1.5 * (attempt + 1))
    return {}


def download_intraday(symbols):
    for attempt in range(3):
        try:
            raw = yf.download(" ".join(symbols), period="5d", interval="5m", group_by="ticker",
                              threads=True, progress=False, auto_adjust=False, prepost=False)
            data = _split(raw, symbols, keep_tz=True)
            if data:
                return data
        except Exception as e:
            log.warning("分時下載失敗（第 %d 次）：%s", attempt + 1, e)
        time.sleep(1.5 * (attempt + 1))
    return {}


# NYSE 休市日（週一至週五的國定假日）。以日曆判斷，不再依賴 Yahoo 回應：
# 舊版用 yfinance 查 SPY 是否有「今天」的K線，Yahoo 一限流/延遲就回 False，整場推播被靜默略過。
NYSE_HOLIDAYS = {
    # 2026
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25", "2026-06-19",
    "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25",
    # 2027
    "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31", "2027-06-18",
    "2027-07-05", "2027-09-06", "2027-11-25", "2027-12-24",
}


def market_open_today() -> bool:
    now = datetime.datetime.now(ET)
    d = now.date().isoformat()
    if now.weekday() >= 5:
        log.info("今天(美東 %s)是週末，不推播", d)
        return False
    if d in NYSE_HOLIDAYS:
        log.info("今天(美東 %s)是 NYSE 休市日，不推播", d)
        return False
    return True


def notify_failure(stage: str, err):
    """推播程式本身出錯時，至少讓你在 LINE 看到「出錯了」，而不是整天沒消息。"""
    log.error("%s 失敗：%s", stage, err)
    try:
        send_line_message(f"⚠️ 美股推播 {stage} 執行失敗：{type(err).__name__} {str(err)[:120]}\n（資料來源 Yahoo 可能暫時限流，請稍後看網頁版：{APP_URL}）")
    except Exception:
        pass


def index_data():
    """回傳 (指數列, VIX)；抓不到的指數 pct=None，不捏造數字。"""
    items = [("道瓊", "^DJI"), ("那指", "^IXIC"), ("費半", "^SOX"), ("標普", "^GSPC")]
    rows = []
    for label, sym in items:
        try:
            df = yf.Ticker(sym).history(period="7d").dropna(subset=["Close"])
            c, p = float(df["Close"].iloc[-1]), float(df["Close"].iloc[-2])
            rows.append({"label": label, "close": c, "pct": (c / p - 1) * 100})
        except Exception:
            rows.append({"label": label, "close": None, "pct": None})
    vix = None
    try:
        v = yf.Ticker("^VIX").history(period="7d").dropna(subset=["Close"])
        c = float(v["Close"].iloc[-1])
        st = "極度恐慌" if c >= 30 else ("恐慌警戒" if c >= 20 else ("正常平穩" if c >= 15 else "樂觀偏低"))
        vix = {"value": c, "state": st}
    except Exception:
        pass
    return rows, vix


def index_summary(rows=None, vix=None):
    if rows is None:
        rows, vix = index_data()
    lines = [f"{'▲' if r['pct'] >= 0 else '▼'} {r['label']} {r['close']:,.0f} ({r['pct']:+.2f}%)"
             if r["pct"] is not None else f"· {r['label']} 暫無資料" for r in rows]
    vix_line = f"🔥 VIX {vix['value']:.2f}（{vix['state']}）" if vix else ""
    return lines, vix_line


# ------------------------------------------------------------------------------
# 篩選（與網頁版相同邏輯：頸線突破/跌破 + 收紅/黑K + 站上/跌破月線 + 量比）
# ------------------------------------------------------------------------------
def screen_symbol(df: pd.DataFrame, mode: str):
    if df is None or len(df) < max(NECK_DAYS + 6, 25):
        return None
    c, o = float(df["Close"].iloc[-1]), float(df["Open"].iloc[-1])
    prev_c = float(df["Close"].iloc[-2])
    v = float(df["Volume"].iloc[-1])
    ma20 = df["Close"].rolling(20, min_periods=20).mean().iloc[-1]
    vma5 = df["Volume"].shift(1).rolling(5, min_periods=3).mean().iloc[-1]
    if pd.isna(ma20) or pd.isna(vma5) or vma5 <= 0:
        return None
    ratio = v / float(vma5)
    win = df.iloc[-(NECK_DAYS + 1):-1]
    prev_win = df.iloc[-(NECK_DAYS + 2):-2]
    if mode == "LONG":
        neck, prev_neck = float(win["High"].max()), float(prev_win["High"].max())
        ok = c > neck and c > o and c > float(ma20) and ratio >= VOL_RATIO
        dist, first = (c / neck - 1) * 100, prev_c <= prev_neck
    else:
        neck, prev_neck = float(win["Low"].min()), float(prev_win["Low"].min())
        ok = c < neck and c < o and c < float(ma20) and ratio >= VOL_RATIO
        dist, first = (neck / c - 1) * 100, prev_c >= prev_neck
    if not ok:
        return None
    return {"price": c, "pct": (c / prev_c - 1) * 100, "ratio": ratio, "dist": dist, "first": first,
            "neck": neck, "ma20": float(ma20), "atr_pct": risk_plan.atr_pct(df)}


def run_screen(mode: str):
    data = download_daily(SYMBOLS)
    out = []
    for s, df in data.items():
        r = screen_symbol(df, mode)
        if r:
            r["symbol"] = s
            r["date"] = df.index[-1].strftime("%Y-%m-%d")   # 價格所屬K棒日期（推播績效追蹤用）
            r["plan"] = risk_plan.stop_plan(r["price"], r.get("atr_pct")) if mode == "LONG" else None
            out.append(r)
    out.sort(key=lambda x: x["ratio"], reverse=True)
    return out, len(data)


def fmt_pick(r, mode, events=None):
    arrow = "▲" if r["pct"] >= 0 else "▼"
    tag = "首日" if r["first"] else "延續"
    far = " ⚠️距離偏遠" if r["dist"] > 5 else ""
    return (f"{'🔥' if mode == 'LONG' else '⚡'} {r['symbol']} {NAME.get(r['symbol'], '')} ${r['price']:.2f} ({arrow}{abs(r['pct']):.1f}%)\n"
            f"   {tag}｜量比 {r['ratio']:.1f}x｜距{'頸線' if mode == 'LONG' else '跌破線'} {r['dist']:+.1f}%{far}\n"
            f"   🛡️ 停損參考：${r['neck']:.2f} ／ 月線 ${r['ma20']:.2f}"
            + (_plan_text(r) if mode == "LONG" else "") + _events_text((events or {}).get(r["symbol"])))


# ------------------------------------------------------------------------------
# 盤中快報：修正「開盤半小時戰報」原本拿 5 天前的開盤價來算漲跌的錯誤
# ------------------------------------------------------------------------------
def intraday_moves(symbols):
    data = download_intraday(symbols)
    today = datetime.datetime.now(ET).date()
    rows = []
    for s, df in data.items():
        try:
            idx = df.index.tz_convert(ET) if getattr(df.index, "tz", None) is not None else df.index
            dates = pd.Series([d.date() for d in idx], index=df.index)
            td = df[dates == today]
            prev = df[dates < today]
            if td.empty or prev.empty:
                continue
            last_c = float(td["Close"].iloc[-1])
            open_p = float(td["Open"].iloc[0])
            prev_c = float(prev["Close"].iloc[-1])
            rows.append({"symbol": s, "last": last_c, "vs_prev": (last_c / prev_c - 1) * 100,
                         "vs_open": (last_c / open_p - 1) * 100})
        except Exception:
            continue
    return rows


def movers_text(rows, events=None):
    if not rows:
        return "（暫無分時資料，可能是 Yahoo 延遲，稍後請看戰情室）"
    rows = sorted(rows, key=lambda x: abs(x["vs_prev"]), reverse=True)[:TOP_N]
    events = events or {}
    return "\n".join(
        f"{'▲' if r['vs_prev'] >= 0 else '▼'} {r['symbol']}: ${r['last']:.2f} (較昨收 {r['vs_prev']:+.1f}%｜較今開 {r['vs_open']:+.1f}%)"
        + _events_text(events.get(r["symbol"]))
        for r in rows
    )


LATE_MIN = 0   # GitHub 排程實際執行時間比預定晚幾分鐘（auto() 設定；>=90 分鐘時標題會註明延遲）


def header_line():
    now_et = datetime.datetime.now(ET)
    now_tp = datetime.datetime.now(TPE)
    dst = "夏令" if bool(now_et.dst()) else "冬令"
    late = f"｜⏱️排程延遲約 {LATE_MIN / 60:.1f} 小時" if LATE_MIN >= 90 else ""
    return f"{dst}｜台北 {now_tp.strftime('%m/%d %H:%M')}／美東 {now_et.strftime('%H:%M')}{late}"


def _stage1():
    """開盤後半小時（美東 10:00）"""
    if not market_open_today():
        log.info("今天美股休市，略過 stage1")
        return
    rows = intraday_moves(WATCH_CORE)
    top = sorted(rows, key=lambda x: abs(x["vs_prev"]), reverse=True)[:TOP_N]
    macro, ev = _get_events([r["symbol"] for r in top])
    text = (
        f"⚡ 【美股多空雷達 V9.9｜開盤半小時戰報】\n{header_line()}\n"
        f"{_macro_text(macro)}\n\n"
        f"早盤初段量能已消化，核心觀察股：\n{movers_text(rows, ev)}\n"
        f"━━━━━━━━━━━━━━\n※ 早盤波動大，請避免追價；僅供量化輔助監控。\n👉 {APP_URL}"
    )
    body = _macro_block(macro) + [_section_title("核心觀察股（波動最大）", NAVY)]
    body += [_flex_mover_card(r, ev.get(r["symbol"])) for r in top] or [_card([_t("暫無分時資料（Yahoo 可能延遲），請看戰情室", "md", MUTED)])]
    bubble = _flex_bubble("美股多空雷達｜開盤半小時", body, "早盤波動大，避免追價｜僅供量化輔助監控")
    send_line_flex("美股多空雷達｜開盤半小時戰報", bubble, text)


def _stage2():
    """午盤快報（美東 12:30）：指數 + 核心觀察股較昨收漲跌"""
    if not market_open_today():
        log.info("今天美股休市，略過 stage2")
        return
    idx_rows, vix = index_data()
    idx_lines, vix_line = index_summary(idx_rows, vix)
    rows = intraday_moves(WATCH_CORE)
    top = sorted(rows, key=lambda x: abs(x["vs_prev"]), reverse=True)[:TOP_N]
    macro, ev = _get_events([r["symbol"] for r in top])
    mstate = _market_state()
    text = (
        f"🌙 【美股多空雷達 V9.9｜午盤戰報】\n{header_line()}\n\n"
        + "\n".join(idx_lines) + (f"\n{vix_line}" if vix_line else "")
        + f"\n\n{_market_text(mstate)}"
        + f"\n\n{_macro_text(macro)}"
        + f"\n\n核心觀察股：\n{movers_text(rows, ev)}\n"
        f"━━━━━━━━━━━━━━\n※ 僅供量化輔助監控。\n👉 {APP_URL}"
    )
    body = [_flex_index_card(idx_rows, vix), _flex_market_card(mstate)] + _macro_block(macro) + [_section_title("核心觀察股（波動最大）", NAVY)]
    body += [_flex_mover_card(r, ev.get(r["symbol"])) for r in top] or [_card([_t("暫無分時資料（Yahoo 可能延遲），請看戰情室", "md", MUTED)])]
    bubble = _flex_bubble("美股多空雷達｜午盤戰報", body, "僅供量化輔助監控")
    send_line_flex("美股多空雷達｜午盤戰報", bubble, text)


def _review_box(longs, shorts, macro, ev):
    """盤後每日覆盤區塊；任何錯誤回傳 None，不影響收盤戰報。"""
    try:
        import daily_review
        syms = [r["symbol"] for r in longs[:TOP_N] + shorts[:TOP_N]]
        # 近 5 個推播日的標的也要抓報價
        _, past = daily_review.picks_rows("9999-12-31")
        want = sorted({"SPY", "QQQ"} | {s for s, _ in daily_review.SECTORS} | {r["symbol"].strip() for r in past} | set(syms))
        data = daily_review.fetch_daily(want)
        spy = data.get("SPY")
        if spy is None:
            return None
        exp_date = spy.index[-1].strftime("%Y-%m-%d")
        return daily_review.build_review_box(exp_date, data, macro, ev, syms, NAME, NYSE_HOLIDAYS, CARD_LINE)
    except Exception as e:
        log.warning("每日覆盤略過：%s %s", type(e).__name__, e)
        return None


# ------------------------------------------------------------------------------
# 抄底模式｜上升趨勢拉回（收盤 > 200 日線、5 日跌 ≥ 10%；跌越深越前面）
#   條件與回測見 us/dip_scan.py；只做提示與試算，由使用者自行決定要不要買
# ------------------------------------------------------------------------------
def _dip_universe():
    import json
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "us_universe.json")
    try:
        rows = json.load(open(p, encoding="utf-8"))
        syms = [str(r[0]).strip().upper().replace(".", "-") for r in rows if r and r[0]]
    except Exception as e:
        log.warning("抄底：讀不到 us_universe.json（%s），改用內建股票池", e)
        syms = list(SYMBOLS)
    return sorted(set(syms) | {"SPY"})


def run_dip():
    """回傳 {market, rows, scanned, capital, default_capital}；任何錯誤回 None（不影響原本推播）。"""
    try:
        syms = _dip_universe()
        data = {}
        for i in range(0, len(syms), 100):
            batch = syms[i:i + 100]
            for attempt in range(2):
                try:
                    raw = yf.download(" ".join(batch), period="1y", group_by="ticker", threads=True,
                                      progress=False, auto_adjust=False)
                    data.update(_split(raw, batch))
                    break
                except Exception as e:
                    log.warning("抄底日K下載失敗（%d，第 %d 次）：%s", i, attempt + 1, e)
                    time.sleep(2)
        spy = data.pop("SPY", None)
        if spy is None or len(data) < len(syms) * 0.6:
            log.warning("抄底：資料不足（%d/%d）或缺 SPY，略過", len(data), len(syms))
            return None
        cap = risk_plan.account_usd()
        use_cap = cap or DIP_DEFAULT_CAPITAL
        res = dip_scan.scan(data, spy, NAME, use_cap)
        res.update({"capital": use_cap, "default_capital": cap is None, "universe": len(syms) - 1})
        return res
    except Exception as e:
        log.warning("抄底模式略過：%s", e)
        return None


def _dip_stats_lines():
    st = dip_scan.STATS
    wy = f"｜最差年度 {st['worst_year']} 平均 {st['worst_year_avg']:+.1f}%" if st.get("worst_year") else ""
    return [f"回測 {st['period']}（持有 {dip_scan.HOLD_DAYS} 天、3×ATR 停損）：勝率 {st['win']}%｜平均 {st['avg']:+.2f}%｜超額 {st['excess']:+.2f}%{wy}",
            f"組合模擬（9.1萬美元、每筆風險1%）：年均約 {st['port_ann']:+.1f}%｜最大回撤約 {st['port_dd']:.0f}%（最差 {st['port_dd_worst']:.0f}%）",
            f"大盤恐慌日（SPY 5日跌≥5%）回測：勝率 {st['panic_win']}%｜平均 {st['panic_avg']:+.2f}%｜停損率 {st['panic_stop']}%（持有10天版）",
            f"※ {st['note']}"]


def _dip_pick_lines(r, panic):
    sz = r.get("size")
    lines = [f"{r['symbol']} {r.get('name') or ''} ${r['price']:.2f}｜5日 {r['ret5']:+.1f}%",
             f"   距200日線 {r['dist200']:+.1f}%｜ATR {r['atr_pct']:.1f}%",
             f"   🛑 停損 ${r['stop']:.2f}（{-r['stop_pct']:.1f}%，3×ATR）",
             (f"   📐 {sz['shares']} 股（${sz['value']:,.0f}｜佔本金 {sz['value_pct']:.0f}%｜最大虧損約 ${sz['risk_usd']:,.0f}）" + ("（已受單檔20%上限）" if sz["capped"] else "")) if sz else "   📐 股數：—",
             f"   ⏳ 建議持有 {dip_scan.HOLD_DAYS} 個交易日（約至 {r['exit_date'] or '—'}）"]
    if panic:
        lines.append("   ⚠️ 恐慌日，風險高")
    return lines


def _dip_head_text(dip):
    m = dip["market"]
    cap_txt = f"${dip['capital']:,.0f}" + ("（預設試算，未設定 ACCOUNT_USD）" if dip["default_capital"] else "")
    ms = f"SPY 5日 {m['ret5']:+.1f}%｜{'在200日線上' if m['above200'] else '在200日線下'}" if m else "SPY：—"
    return ms, cap_txt


def _dip_text(dip):
    if not dip:
        return "\n━━ 抄底模式（上升趨勢拉回）━━\n今日資料不足，未計算"
    ms, cap_txt = _dip_head_text(dip)
    panic = bool(dip["market"] and dip["market"]["panic"])
    rows = dip["rows"]
    out = [f"\n━━ 抄底模式｜上升趨勢拉回（共 {len(rows)} 檔，顯示前 {min(len(rows), DIP_TOP_N)}）━━",
           f"條件：收盤>200日線且5日跌≥{dip_scan.DROP_PCT:g}%，跌越深越前｜{ms}｜本金 {cap_txt}｜單筆風險 {dip_scan.RISK_PCT:g}%"]
    if panic:
        out.append("⚠️ 大盤恐慌日，風險高（仍照列，請自行決定）")
    out += _dip_stats_lines()
    if rows:
        for r in rows[:DIP_TOP_N]:
            out.append("\n".join(_dip_pick_lines(r, panic)))
    else:
        out.append("今日無符合條件之標的")
    return "\n".join(out)


def _flex_dip_section(dip):
    """收盤卡片的抄底區；回傳 Flex 元素列表（資料不足時只給一張說明卡）。"""
    out = [_section_title(f"抄底模式｜上升趨勢拉回  共 {len(dip['rows']) if dip else '—'} 檔", INFO)]
    if not dip:
        return out + [_card([_t("今日資料不足，未計算", "md", MUTED)])]
    ms, cap_txt = _dip_head_text(dip)
    panic = bool(dip["market"] and dip["market"]["panic"])
    head = [_t(f"收盤>200日線且5日跌≥{dip_scan.DROP_PCT:g}%，跌越深越前", "md", NAVY, True),
            _t(f"{ms}｜本金 {cap_txt}｜單筆風險 {dip_scan.RISK_PCT:g}%", "md", "#1a1a1a", margin="xs")]
    if panic:
        head.append(_t("⚠️ 大盤恐慌日，風險高（仍照列，請自行決定）", "lg", DOWN, True, margin="sm"))
    head += [_t(x, "sm", MUTED, margin="xs") for x in _dip_stats_lines()]
    out.append(_card(head))
    for r in dip["rows"][:DIP_TOP_N]:
        sz = r.get("size")
        body = [
            _box("horizontal", [
                _t(f"{r['symbol']} {r.get('name') or ''}", "lg", NAVY, True, flex=5),
                _t(f"${r['price']:.2f} ▼{abs(r['ret5']):.1f}%", "lg", DOWN, True, flex=5, align="end")]),
            _t(f"距200日線 {r['dist200']:+.1f}%｜ATR {r['atr_pct']:.1f}%", "md", INFO, margin="sm"),
            _t(f"🛑 停損 ${r['stop']:.2f}（{-r['stop_pct']:.1f}%，3×ATR）", "md", DOWN, True, margin="xs"),
            _t((f"📐 {sz['shares']} 股｜${sz['value']:,.0f}｜佔本金 {sz['value_pct']:.0f}%｜最大虧損約 ${sz['risk_usd']:,.0f}" + ("（單檔20%上限）" if sz["capped"] else "")) if sz else "📐 股數：—",
               "md", "#1a1a1a", margin="xs"),
            _t(f"⏳ 建議持有 {dip_scan.HOLD_DAYS} 個交易日（約至 {r['exit_date'] or '—'}）", "md", "#444444", margin="xs"),
        ]
        if panic:
            body.append(_t("⚠️ 恐慌日，風險高", "md", DOWN, True, margin="xs"))
        out.append(_box("horizontal", [
            _box("vertical", [], width="6px", backgroundColor=INFO),
            _box("vertical", body, flex=1, paddingAll="12px"),
        ], backgroundColor="#ffffff", cornerRadius="12px", borderWidth="1px", borderColor=CARD_LINE))
    if not dip["rows"]:
        out.append(_card([_t("今日無符合條件之標的", "md", MUTED)]))
    return out


def build_close_bubble(idx_rows, vix, longs, shorts, scanned, macro=None, ev=None, mstate=None, capital=None, alloc=None, dip=None, dip_on=False):
    ev = ev or {}
    body = [_flex_index_card(idx_rows, vix), _flex_market_card(mstate)]
    if macro is not None or ev:
        body += _macro_block(macro)
    body.append(_section_title(f"多方突破  共 {len(longs)} 檔", UP))
    body += [_flex_pick_card(r, "LONG", ev.get(r["symbol"])) for r in longs[:TOP_N]] or [_card([_t("今日無符合條件之多方突破標的", "md", MUTED)])]
    body.append(_section_title(f"空方破底  共 {len(shorts)} 檔", DOWN))
    body += [_flex_pick_card(r, "SHORT", ev.get(r["symbol"])) for r in shorts[:TOP_N]] or [_card([_t("今日無符合條件之空方破底標的", "md", MUTED)])]
    if dip_on:
        try:
            body += _flex_dip_section(dip)
        except Exception as e:
            log.warning("抄底卡片略過：%s", e)
    body.append(_card([_t(x, "md", NAVY, True, margin="xs") for x in _risk_footer_lines(capital, alloc)]))
    review = _review_box(longs, shorts, macro, ev)
    if review:
        body.append(review)
    body.append(_t(f"已掃描 {scanned}/{len(SYMBOLS)} 檔｜{NECK_DAYS}日頸線、量比≥{VOL_RATIO}x、收紅/黑K、站上/跌破月線",
                   "sm", MUTED, margin="lg"))
    return _flex_bubble("美股多空雷達｜收盤戰報", body, "僅為量化條件篩選結果，非投資建議")


def _stage3():
    """收盤結算（美東 16:15）：指數 + 全股票池 多方突破 / 空方破底 名單"""
    if not market_open_today():
        log.info("今天美股休市，略過 stage3")
        return
    idx_rows, vix = index_data()
    idx_lines, vix_line = index_summary(idx_rows, vix)
    longs, scanned = run_screen("LONG")
    shorts, _ = run_screen("SHORT")
    mstate = _market_state()
    capital = risk_plan.account_usd()
    alloc = risk_plan.allocate(longs[:TOP_N], capital)   # 只對實際推播的多方標的試算；空方不做部位試算
    macro, ev = _get_events([r["symbol"] for r in longs[:TOP_N] + shorts[:TOP_N]])
    dip = run_dip()                      # 抄底模式（上升趨勢拉回）；失敗回 None，不影響原本推播
    try:
        dip_txt = _dip_text(dip)
    except Exception as e:
        log.warning("抄底文字略過：%s", e); dip_txt = ""
    parts = [
        f"📊 【美股多空雷達 V9.9｜全日收盤戰報】\n{header_line()}\n",
        "\n".join(idx_lines) + (f"\n{vix_line}" if vix_line else ""),
        "\n" + _market_text(mstate),
        "\n" + _macro_text(macro),
        f"\n━━ 多方突破（共 {len(longs)} 檔，顯示前 {min(len(longs), TOP_N)}）━━",
        "\n".join(fmt_pick(r, "LONG", ev) for r in longs[:TOP_N]) if longs else "今日無符合條件之多方突破標的",
        f"\n━━ 空方破底（共 {len(shorts)} 檔，顯示前 {min(len(shorts), TOP_N)}）━━",
        "\n".join(fmt_pick(r, "SHORT", ev) for r in shorts[:TOP_N]) if shorts else "今日無符合條件之空方破底標的",
        dip_txt,
        "\n" + "\n".join(_risk_footer_lines(capital, alloc)),
        f"\n已掃描 {scanned}/{len(SYMBOLS)} 檔｜條件：{NECK_DAYS}日頸線、量比≥{VOL_RATIO}x、收紅/黑K、站上/跌破月線",
        "※ 僅為量化條件篩選結果，非投資建議。",
        f"👉 {APP_URL}",
    ]
    bubble = build_close_bubble(idx_rows, vix, longs, shorts, scanned, macro, ev, mstate, capital, alloc, dip, True)
    ok = send_line_flex(f"美股多空雷達｜收盤戰報：多方 {len(longs)} 檔／空方 {len(shorts)} 檔", bubble, "\n".join(parts))
    # 推播績效追蹤：只記錄真的送達 LINE 的標的；任何錯誤都不影響推播
    if ok and (longs or shorts):
        try:
            from push_perf import record_push
            record_push({"LONG": longs, "SHORT": shorts}, NAME, TOP_N)
        except Exception as e:
            log.warning("推播紀錄略過：%s", e)


def _safe(fn, label):
    def run():
        log.info("開始執行 %s（美東 %s）", label, datetime.datetime.now(ET).strftime("%m/%d %H:%M"))
        try:
            fn()
            log.info("%s 完成", label)
        except Exception as e:
            notify_failure(label, e)
    return run


stage1 = _safe(_stage1, "stage1 開盤半小時")
stage2 = _safe(_stage2, "stage2 午盤")
stage3 = _safe(_stage3, "stage3 收盤結算")


def _intended_et(schedule: str):
    """把觸發這次執行的 cron（UTC，例如 "5 15 * * 1-5"）換成「今天預定的美東時間」。

    GitHub Actions 排程常延遲 30–60 分鐘，若用「實際執行時」的美東鐘點判斷，
    冬令時早一班（夏令用的那班）可能延遲到正確鐘點而重複推，正確那班也可能延遲到下一個鐘點而漏推。
    改用「預定時間」判斷：每班只對應一個美東時間，延遲多久都不會跑錯階段。
    cron 解析不了（例如舊格式一行兩個鐘點）就回傳 None，改用目前時間判斷。
    """
    try:
        minute, hour = schedule.split()[:2]
        mi, hr = int(minute), int(hour)
    except Exception:
        return None
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    sched = now_utc.replace(hour=hr, minute=mi, second=0, microsecond=0)
    if sched > now_utc + datetime.timedelta(minutes=5):   # 跨過 UTC 午夜才開始跑
        sched -= datetime.timedelta(days=1)
    return sched.astimezone(ET)


def auto():
    """依美東時間決定階段；給 GitHub Actions 這類「固定 UTC 時間觸發」的排程使用。
    夏令/冬令由 America/New_York 時區自動判斷，每年不用手動改。"""
    sched = os.environ.get("SCHEDULE", "").strip()
    t = _intended_et(sched) if sched else None
    now_et = datetime.datetime.now(ET)
    if t is not None:
        log.info("排程 %s → 預定美東 %s（%s），實際美東 %s",
                 sched, t.strftime("%m/%d %H:%M"), "夏令" if t.dst() else "冬令", now_et.strftime("%H:%M"))
    else:
        t = now_et
    global LATE_MIN
    LATE_MIN = max(0, int((now_et - t).total_seconds() // 60))
    h, wd = t.hour, t.weekday()
    if wd >= 5:
        log.info("週末，略過")
        return
    if h == 10:
        # 過時保護：開盤半小時快報若拖到美東 11:00 之後才跑（GitHub 排程常嚴重延遲），內容已不是開盤初段，寧可不推
        if now_et.date() != t.date() or now_et.hour >= 11:
            log.warning("開盤半小時班次延遲至美東 %s，已過時，略過不推", now_et.strftime("%m/%d %H:%M"))
            return
        stage1()
    elif h == 16:
        stage3()
    else:
        log.info("預定美東 %02d 點不在推播時段（另一個季節用的班次），略過", h)


def run_daemon():
    from apscheduler.schedulers.blocking import BlockingScheduler
    sch = BlockingScheduler(timezone=ET)   # 以美東時間排程：夏令/冬令自動切換
    kw = dict(day_of_week="mon-fri", misfire_grace_time=900, coalesce=True)
    sch.add_job(stage1, "cron", hour=10, minute=0, **kw)
    sch.add_job(stage3, "cron", hour=16, minute=15, **kw)
    log.info("🚀 美股兩段式推播排程已啟動（美東 10:00 / 16:15，週一至週五；午盤已停推）")
    sch.start()


if __name__ == "__main__":
    cmd = sys.argv[1].lower() if len(sys.argv) > 1 else "daemon"
    if cmd == "daemon":
        run_daemon()
    elif cmd == "auto":
        auto()
    elif cmd == "stage1":
        stage1()
    elif cmd == "stage2":
        stage2()
    elif cmd == "stage3":
        stage3()
    elif cmd == "test":
        send_line_message(f"✅ 美股雷達推播測試成功\n{header_line()}")
    elif cmd == "flextest":
        # 只抓指數（真實資料），確認卡片版型與海鵰圖片能正常顯示
        rows, vix = index_data()
        bubble = _flex_bubble("美股多空雷達｜卡片版測試", [
            _flex_index_card(rows, vix),
            _card([_t("看到白頭海鵰和這張卡片，代表 Flex 卡片版推播設定成功。", "md", NAVY)]),
        ], "測試訊息")
        send_line_flex("美股多空雷達｜卡片版測試", bubble, f"✅ 卡片版測試（Flex 失敗，已改送文字）\n{header_line()}")
    else:
        print(__doc__)
