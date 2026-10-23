"""
美股多空雷達 V9.9 ─ LINE 推播排程

兩種使用方式：
1) 常駐模式（Render Background Worker / 自己的主機）：
       python us_scheduler.py
   以「美東時間」排程，夏令/冬令自動處理，不再需要手動判斷 DST。

2) 單次執行模式（GitHub Actions cron、或手動測試）：
       python us_scheduler.py auto      # 依目前美東時間自動判斷該跑哪一階段，不在時段內就不動作
       python us_scheduler.py stage1    # 開盤後半小時
       python us_scheduler.py stage2    # 午盤
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


def _flex_pick_card(r, mode):
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
    return _box("horizontal", [
        _box("vertical", [], width="6px", backgroundColor=accent),
        _box("vertical", body, flex=1, paddingAll="12px"),
    ], backgroundColor="#ffffff", cornerRadius="12px", borderWidth="1px", borderColor=CARD_LINE)


def _flex_mover_card(r):
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
        ], flex=1, paddingAll="12px"),
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
            "neck": neck, "ma20": float(ma20)}


def run_screen(mode: str):
    data = download_daily(SYMBOLS)
    out = []
    for s, df in data.items():
        r = screen_symbol(df, mode)
        if r:
            r["symbol"] = s
            out.append(r)
    out.sort(key=lambda x: x["ratio"], reverse=True)
    return out, len(data)


def fmt_pick(r, mode):
    arrow = "▲" if r["pct"] >= 0 else "▼"
    tag = "首日" if r["first"] else "延續"
    far = " ⚠️距離偏遠" if r["dist"] > 5 else ""
    return (f"{'🔥' if mode == 'LONG' else '⚡'} {r['symbol']} {NAME.get(r['symbol'], '')} ${r['price']:.2f} ({arrow}{abs(r['pct']):.1f}%)\n"
            f"   {tag}｜量比 {r['ratio']:.1f}x｜距{'頸線' if mode == 'LONG' else '跌破線'} {r['dist']:+.1f}%{far}\n"
            f"   🛡️ 停損參考：${r['neck']:.2f} ／ 月線 ${r['ma20']:.2f}")


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


def movers_text(rows):
    if not rows:
        return "（暫無分時資料，可能是 Yahoo 延遲，稍後請看戰情室）"
    rows = sorted(rows, key=lambda x: abs(x["vs_prev"]), reverse=True)[:TOP_N]
    return "\n".join(
        f"{'▲' if r['vs_prev'] >= 0 else '▼'} {r['symbol']}: ${r['last']:.2f} (較昨收 {r['vs_prev']:+.1f}%｜較今開 {r['vs_open']:+.1f}%)"
        for r in rows
    )


def header_line():
    now_et = datetime.datetime.now(ET)
    now_tp = datetime.datetime.now(TPE)
    dst = "夏令" if bool(now_et.dst()) else "冬令"
    return f"{dst}｜台北 {now_tp.strftime('%m/%d %H:%M')}／美東 {now_et.strftime('%H:%M')}"


def _stage1():
    """開盤後半小時（美東 10:00）"""
    if not market_open_today():
        log.info("今天美股休市，略過 stage1")
        return
    rows = intraday_moves(WATCH_CORE)
    text = (
        f"⚡ 【美股多空雷達 V9.9｜開盤半小時戰報】\n{header_line()}\n"
        f"早盤初段量能已消化，核心觀察股：\n{movers_text(rows)}\n"
        f"━━━━━━━━━━━━━━\n※ 早盤波動大，請避免追價；僅供量化輔助監控。\n👉 {APP_URL}"
    )
    top = sorted(rows, key=lambda x: abs(x["vs_prev"]), reverse=True)[:TOP_N]
    body = [_section_title("核心觀察股（波動最大）", NAVY)]
    body += [_flex_mover_card(r) for r in top] or [_card([_t("暫無分時資料（Yahoo 可能延遲），請看戰情室", "md", MUTED)])]
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
    text = (
        f"🌙 【美股多空雷達 V9.9｜午盤戰報】\n{header_line()}\n\n"
        + "\n".join(idx_lines) + (f"\n{vix_line}" if vix_line else "")
        + f"\n\n核心觀察股：\n{movers_text(rows)}\n"
        f"━━━━━━━━━━━━━━\n※ 僅供量化輔助監控。\n👉 {APP_URL}"
    )
    top = sorted(rows, key=lambda x: abs(x["vs_prev"]), reverse=True)[:TOP_N]
    body = [_flex_index_card(idx_rows, vix), _section_title("核心觀察股（波動最大）", NAVY)]
    body += [_flex_mover_card(r) for r in top] or [_card([_t("暫無分時資料（Yahoo 可能延遲），請看戰情室", "md", MUTED)])]
    bubble = _flex_bubble("美股多空雷達｜午盤戰報", body, "僅供量化輔助監控")
    send_line_flex("美股多空雷達｜午盤戰報", bubble, text)


def build_close_bubble(idx_rows, vix, longs, shorts, scanned):
    body = [_flex_index_card(idx_rows, vix)]
    body.append(_section_title(f"多方突破  共 {len(longs)} 檔", UP))
    body += [_flex_pick_card(r, "LONG") for r in longs[:TOP_N]] or [_card([_t("今日無符合條件之多方突破標的", "md", MUTED)])]
    body.append(_section_title(f"空方破底  共 {len(shorts)} 檔", DOWN))
    body += [_flex_pick_card(r, "SHORT") for r in shorts[:TOP_N]] or [_card([_t("今日無符合條件之空方破底標的", "md", MUTED)])]
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
    parts = [
        f"📊 【美股多空雷達 V9.9｜全日收盤戰報】\n{header_line()}\n",
        "\n".join(idx_lines) + (f"\n{vix_line}" if vix_line else ""),
        f"\n━━ 多方突破（共 {len(longs)} 檔，顯示前 {min(len(longs), TOP_N)}）━━",
        "\n".join(fmt_pick(r, "LONG") for r in longs[:TOP_N]) if longs else "今日無符合條件之多方突破標的",
        f"\n━━ 空方破底（共 {len(shorts)} 檔，顯示前 {min(len(shorts), TOP_N)}）━━",
        "\n".join(fmt_pick(r, "SHORT") for r in shorts[:TOP_N]) if shorts else "今日無符合條件之空方破底標的",
        f"\n已掃描 {scanned}/{len(SYMBOLS)} 檔｜條件：{NECK_DAYS}日頸線、量比≥{VOL_RATIO}x、收紅/黑K、站上/跌破月線",
        "※ 僅為量化條件篩選結果，非投資建議。",
        f"👉 {APP_URL}",
    ]
    bubble = build_close_bubble(idx_rows, vix, longs, shorts, scanned)
    send_line_flex(f"美股多空雷達｜收盤戰報：多方 {len(longs)} 檔／空方 {len(shorts)} 檔", bubble, "\n".join(parts))


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


def auto():
    """依目前美東時間決定階段；給 GitHub Actions 這類「固定 UTC 時間觸發」的排程使用。"""
    h = datetime.datetime.now(ET).hour
    wd = datetime.datetime.now(ET).weekday()
    if wd >= 5:
        log.info("週末，略過")
        return
    if h == 10:
        stage1()
    elif h == 12:
        stage2()
    elif h == 16:
        stage3()
    else:
        log.info("目前美東 %02d 點不在推播時段，略過", h)


def run_daemon():
    from apscheduler.schedulers.blocking import BlockingScheduler
    sch = BlockingScheduler(timezone=ET)   # 以美東時間排程：夏令/冬令自動切換
    kw = dict(day_of_week="mon-fri", misfire_grace_time=900, coalesce=True)
    sch.add_job(stage1, "cron", hour=10, minute=0, **kw)
    sch.add_job(stage2, "cron", hour=12, minute=30, **kw)
    sch.add_job(stage3, "cron", hour=16, minute=15, **kw)
    log.info("🚀 美股三段式推播排程已啟動（美東 10:00 / 12:30 / 16:15，週一至週五）")
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
