"""盤後「每日覆盤」區塊（美股）。只用真實數據；抓不到就顯示「—」。美股綠漲紅跌。

內容：① SPY／QQQ 與季線、200日線位置變化 ② 近 5 個推播日標的後續表現 ③ 類股強弱（SPDR 11 類股 ETF）
④ 明日關注。每段各自 try/except；LINE Flex 內不放空字串（PR #14 教訓）。
"""
import csv
import datetime
import logging
import os

import pandas as pd
import yfinance as yf

log = logging.getLogger("us_review")
UP, DOWN, MUTED, NAVY, INK, AMBER = "#1f7a3a", "#c0281e", "#5f5e5a", "#0b2a52", "#1a1a1a", "#a8650a"   # 綠漲紅跌
BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tracking")
TRACK_FILES = ("us_push_perf.csv", "us_push_log.csv")
LOOKBACK_DAYS = 5
MAX_PICK_ROWS = 8
SECTORS = [("XLK", "科技"), ("XLF", "金融"), ("XLE", "能源"), ("XLV", "醫療保健"), ("XLY", "非必需消費"),
           ("XLP", "必需消費"), ("XLI", "工業"), ("XLU", "公用事業"), ("XLB", "原物料"),
           ("XLRE", "房地產"), ("XLC", "通訊服務")]


def _t(text, size="md", color=INK, bold=False, **kw):
    if not text:
        return None
    d = {"type": "text", "text": str(text), "size": size, "color": color, "wrap": True, "margin": "xs"}
    if bold:
        d["weight"] = "bold"
    d.update(kw)
    return d


def _col(v):
    if v is None or v != v or abs(v) < 1e-9:
        return MUTED
    return UP if v > 0 else DOWN


def _title(text):
    return _t(text, "lg", NAVY, True, margin="lg")


def fetch_daily(symbols):
    """一次下載日K（1 年，夠算 200 日線）；回傳 {代號: DataFrame(Close…)}，失敗回傳 {}。"""
    try:
        raw = yf.download(" ".join(symbols), period="1y", group_by="ticker", threads=True,
                          progress=False, auto_adjust=False)
    except Exception as e:
        log.warning("覆盤日K下載失敗：%s", e)
        return {}
    out = {}
    if raw is None or raw.empty:
        return out
    multi = isinstance(raw.columns, pd.MultiIndex)
    lvl0 = set(raw.columns.get_level_values(0)) if multi else set()
    for s in symbols:
        sub = raw[s].copy() if (multi and s in lvl0) else (raw.copy() if (not multi and len(symbols) == 1) else None)
        if sub is not None and "Close" in sub.columns:
            sub = sub.dropna(subset=["Close"])
            if not sub.empty:
                if getattr(sub.index, "tz", None) is not None:
                    sub.index = sub.index.tz_localize(None)
                out[s] = sub
    return out


def _ma_move(close, n):
    """回傳 (今日收盤, 均線, 距離%, 今日是否在線上, 昨日是否在線上)；資料不足回傳 None。"""
    if len(close) < n + 1:
        return None
    ma = close.rolling(n).mean()
    c, m, pc, pm = float(close.iloc[-1]), float(ma.iloc[-1]), float(close.iloc[-2]), float(ma.iloc[-2])
    if pd.isna(m) or pd.isna(pm) or m <= 0:
        return None
    return c, m, (c / m - 1) * 100, c >= m, pc >= pm


def index_section(data):
    out = []
    for sym in ("SPY", "QQQ"):
        df = data.get(sym)
        if df is None or len(df) < 2:
            out.append(_t(f"{sym}：—", color=MUTED))
            continue
        c = df["Close"].astype(float)
        pct = (c.iloc[-1] / c.iloc[-2] - 1) * 100
        out.append(_t(f"{sym} {c.iloc[-1]:,.2f}（{pct:+.2f}%）", "md", _col(pct), True))
        for label, n in (("季線", 60), ("200日線", 200)):
            r = _ma_move(c, n)
            if r is None:
                out.append(_t(f"　{label}：—", color=MUTED))
                continue
            _, m, gap, above, was = r
            move = "｜🔺今日站上" if (above and not was) else ("｜🔻今日跌破" if (not above and was) else ("｜續在線上" if above else "｜續在線下"))
            out.append(_t(f"　{label} {m:,.2f}（{gap:+.1f}%）{move}", "md", UP if above else DOWN, True))
    return out


def _read_tracking():
    for fn in TRACK_FILES:
        p = os.path.join(BASE, fn)
        if os.path.exists(p):
            try:
                with open(p, encoding="utf-8-sig", newline="") as f:
                    return list(csv.DictReader(f))
            except Exception:
                continue
    return []


def _f(x):
    try:
        v = float(x)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def picks_rows(exp_date, rows=None):
    """近 LOOKBACK_DAYS 個推播日（不含今天）推過的標的（同代號同方向只算最早一次）。"""
    rows = _read_tracking() if rows is None else rows
    rows = [r for r in rows if (r.get("date") or "") < exp_date and (r.get("symbol") or "").strip()
            and r.get("mode") in ("LONG", "SHORT")]
    dates = sorted({r["date"] for r in rows}, reverse=True)[:LOOKBACK_DAYS]
    first = {}
    for r in sorted((r for r in rows if r["date"] in dates), key=lambda r: (r["date"], r.get("time", ""))):
        first.setdefault((r["symbol"].strip(), r["mode"]), r)
    return dates, list(first.values())


def picks_section(exp_date, data, rows=None):
    dates, picks = picks_rows(exp_date, rows)
    if not picks:
        return [_t("近期尚無推播紀錄（—）", color=MUTED)]
    items = []
    for r in picks:
        sym, mode = r["symbol"].strip(), r["mode"]
        px, neck = _f(r.get("push_px")), _f(r.get("neck"))
        df = data.get(sym)
        cur = None
        if df is not None and df.index[-1].strftime("%Y-%m-%d") == exp_date:
            cur = float(df["Close"].iloc[-1])
        move = (cur / px - 1) * 100 if (cur and px and px > 0) else None
        pnl = None if move is None else (move if mode == "LONG" else -move)
        # 多方跌回頸線下＝突破失敗；空方站回頸線上＝破底失敗
        failed = None if (cur is None or not neck) else ((cur < neck) if mode == "LONG" else (cur > neck))
        items.append({"date": r["date"], "sym": sym, "name": r.get("name", ""), "mode": mode, "px": px,
                      "cur": cur, "move": move, "pnl": pnl, "failed": failed})
    valid = [i for i in items if i["pnl"] is not None]
    out = []
    if valid:
        wins = sum(1 for i in valid if i["pnl"] > 0)
        fails = sum(1 for i in valid if i["failed"])
        avg = sum(i["pnl"] for i in valid) / len(valid)
        out.append(_t(f"近 {len(dates)} 個推播日共 {len(valid)} 檔：{wins} 檔方向正確、{fails} 檔突破／破底失敗｜平均損益 {avg:+.2f}%",
                      "md", _col(avg), True))
    else:
        out.append(_t("今日報價暫無，無法計算後續表現（—）", color=MUTED))
    items.sort(key=lambda i: (i["pnl"] is not None, i["date"], i["sym"]), reverse=True)
    for i in items[:MAX_PICK_ROWS]:
        md = i["date"][5:].replace("-", "/")
        side = "多" if i["mode"] == "LONG" else "空"
        if i["pnl"] is None:
            out.append(_t(f"{md} {i['sym']}（{side}）今日報價 —", color=MUTED))
            continue
        flag = "｜⚠️失敗" if i["failed"] else ("｜守住" if i["failed"] is False else "")
        out.append(_t(f"{md} {i['sym']}（{side}）{i['px']:g}→{i['cur']:g}｜價 {i['move']:+.1f}%｜損益 {i['pnl']:+.1f}%{flag}",
                      "md", _col(i["move"]), bool(i["failed"])))
    if len(items) > MAX_PICK_ROWS:
        out.append(_t(f"…其餘 {len(items) - MAX_PICK_ROWS} 檔請看戰情室", "sm", MUTED))
    out.append(_t("損益：多方＝價格漲跌；空方＝價格跌為正。顏色依價格漲跌（綠漲紅跌）。", "sm", MUTED))
    return out


def sector_section(exp_date, data):
    rows = []
    for sym, zh in SECTORS:
        df = data.get(sym)
        if df is not None and len(df) >= 2 and df.index[-1].strftime("%Y-%m-%d") == exp_date:
            c = df["Close"].astype(float)
            rows.append(((c.iloc[-1] / c.iloc[-2] - 1) * 100, f"{zh}（{sym}）"))
    if len(rows) < 6:
        return [_t("類股資料不足（—）", color=MUTED)]
    rows.sort(reverse=True)
    out = [_t("最強", "md", INK, True)]
    out += [_t(f"{n}　{p:+.2f}%", "md", _col(p), True) for p, n in rows[:3]]
    out.append(_t("最弱", "md", INK, True, margin="sm"))
    out += [_t(f"{n}　{p:+.2f}%", "md", _col(p), True) for p, n in rows[-3:][::-1]]
    out.append(_t("類股＝SPDR 11 類股 ETF 當日漲跌", "sm", MUTED))
    return out


def next_trading_day(d, holidays):
    n = d + datetime.timedelta(days=1)
    while n.weekday() >= 5 or n.isoformat() in holidays:
        n += datetime.timedelta(days=1)
    return n


def tomorrow_section(exp_date, holidays, macro, events, symbols, names):
    out = []
    try:
        d = datetime.date.fromisoformat(exp_date)
        n = next_trading_day(d, holidays)
        wd = "一二三四五六日"[n.weekday()]
        gap = (n - d).days
        tail = "（中間有休市日）" if gap > 1 and not (d.weekday() == 4 and gap == 3) else ""
        out.append(_t(f"下一個交易日：{n.month}/{n.day}（{wd}）{tail}", "md", NAVY, True))
    except Exception:
        out.append(_t("下一個交易日：—", color=MUTED))
    if macro is None:
        out.append(_t("總經事件 —（暫時抓不到）", color=MUTED))
    else:
        urgent = [txt for txt, u in macro if u]
        out += [_t(x, "md", AMBER, True) for x in urgent[:3]]
        if not urgent:
            out.append(_t("3 天內無 FOMC／CPI／非農公布", color=MUTED))
    shown = 0
    for s in symbols:
        for txt, u in (events or {}).get(s, []):
            if u and txt and shown < 4:
                out.append(_t(f"{s} {names.get(s, '')}：{txt}", "md", AMBER, True))
                shown += 1
    return out


def _stale_note(exp_date):
    """資料日不是美東今天（Yahoo 還沒更新今日收盤）就註明，避免把舊資料當今日。"""
    try:
        from zoneinfo import ZoneInfo
        now = datetime.datetime.now(ZoneInfo("America/New_York"))
        if now.weekday() < 5 and now.hour >= 16 and now.date().isoformat() != exp_date:
            return "｜⚠️ 非今日收盤資料（報價來源尚未更新）"
    except Exception:
        pass
    return ""


def build_review_box(exp_date, data, macro, events, symbols, names, holidays, card_line="#cddbef"):
    def safe(fn, *a):
        try:
            return fn(*a)
        except Exception as e:
            log.warning("覆盤段落略過：%s %s %s", fn.__name__, type(e).__name__, e)
            return [_t("暫無資料（—）", color=MUTED)]
    contents = [_t("📋 今日覆盤", "xl", NAVY, True, margin="none"),
                _t(f"資料日 {exp_date}" + _stale_note(exp_date), "sm", MUTED),
                _title("① 大盤位置變化"), *safe(index_section, data),
                _title("② 推播標的後續表現"), *safe(picks_section, exp_date, data),
                _title("③ 類股強弱"), *safe(sector_section, exp_date, data),
                _title("④ 明日關注"), *safe(tomorrow_section, exp_date, holidays, macro, events, symbols, names)]
    return {"type": "box", "layout": "vertical", "backgroundColor": "#ffffff", "cornerRadius": "12px",
            "borderWidth": "1px", "borderColor": card_line, "paddingAll": "14px",
            "contents": [c for c in contents if c]}
