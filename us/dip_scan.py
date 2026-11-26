"""
美股「抄底模式｜上升趨勢拉回」（推播與網頁共用同一份計算）

條件（收盤後已知）：收盤 > 200 日均線，且 5 日跌幅 ≥ 10%。跌越深排越前面。
每檔給：建議停損（進場價 − 3×ATR14）、依單筆風險 1% 算的股數、建議持有 20 個交易日。
大盤恐慌日（SPY 5 日跌幅 ≥ 5%）照列，但標「恐慌日，風險高」，由使用者自行決定。

回測數字（backtest_reports/dip_trend_pullback_backtest.md；S&P 500 時點成分股，2004–2026，含 2008／2020／2022；
成本 0.1%）寫在 STATS，網頁與推播直接引用，不自行編造。倖存者偏差：曾移出成分股 Yahoo 只抓到 38%，數字偏樂觀。
資料原則：算不出來就不放進名單（不補假數字）。
"""
import math

import numpy as np
import pandas as pd

DROP_PCT = 10.0        # 5 日跌幅門檻（%）
HOLD_DAYS = 20         # 建議持有交易日
ATR_MULT = 3.0         # 停損 = 進場價 − 3×ATR14
RISK_PCT = 1.0         # 單筆風險（佔本金 %）
POS_CAP_PCT = 20.0     # 單檔部位上限（佔本金 %，與回測一致）
PANIC_SPY5 = -5.0      # SPY 5 日跌幅 ≤ 此值＝恐慌日

# 回測（持有 20 天、3×ATR 盤中停損）與恐慌日（持有 10 天版）；來源檔見上方說明
STATS = {
    "n": 3132, "events": 241, "win": 54, "avg": 1.78, "excess": 0.82, "t": 2.2,
    "worst_year": None, "worst_year_avg": None,     # 由回測 20 天分年度表填入
    "panic_win": 52, "panic_avg": -0.23, "panic_stop": 27,   # 大盤 5 日跌 ≥5%，持有 10 天版
    "period": "2004–2026",
    "note": "S&P 500 時點成分股；移出股僅抓到 38%，數字偏樂觀；t 值 2.2 屬勉強顯著",
}


def _atr14(df: pd.DataFrame):
    pc = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - pc).abs(), (df["Low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()


def spy_state(spy: pd.DataFrame):
    """SPY 5 日跌幅、是否在 200 日線上；資料不足回 None。"""
    if spy is None or len(spy) < 205:
        return None
    c = spy["Close"].dropna()
    if len(c) < 205:
        return None
    r5 = float((c.iloc[-1] / c.iloc[-6] - 1) * 100)
    ma200 = float(c.rolling(200).mean().iloc[-1])
    return {"ret5": r5, "above200": bool(c.iloc[-1] > ma200), "dist200": float((c.iloc[-1] / ma200 - 1) * 100),
            "panic": r5 <= PANIC_SPY5, "close": float(c.iloc[-1]), "date": c.index[-1]}


def screen_one(df: pd.DataFrame):
    """符合條件回傳 dict，否則 None。"""
    if df is None or len(df) < 205 or not {"Open", "High", "Low", "Close"} <= set(df.columns):
        return None
    d = df.dropna(subset=["Close", "High", "Low"])
    if len(d) < 205:
        return None
    c = d["Close"]
    last = float(c.iloc[-1])
    ma200 = float(c.rolling(200).mean().iloc[-1])
    atr = float(_atr14(d).iloc[-1]) if len(d) >= 20 else float("nan")
    if not (last > 0 and ma200 > 0 and atr > 0) or math.isnan(atr) or atr / last >= 0.25:
        return None
    ret5 = float((last / float(c.iloc[-6]) - 1) * 100)
    if not (last > ma200 and ret5 <= -DROP_PCT):
        return None
    risk_ps = ATR_MULT * atr
    return {"price": last, "ret5": ret5, "ma200": ma200, "dist200": (last / ma200 - 1) * 100, "atr": atr,
            "atr_pct": atr / last * 100, "stop": last - risk_ps, "stop_pct": risk_ps / last * 100,
            "risk_ps": risk_ps, "date": d.index[-1].strftime("%Y-%m-%d")}


def size(r, capital):
    """依單筆風險 1% 算股數（單檔上限 20% 本金）；沒有本金就回 None。"""
    if not capital or capital <= 0 or not r or r["risk_ps"] <= 0:
        return None
    sh_risk = math.floor(capital * RISK_PCT / 100 / r["risk_ps"])
    sh_cap = math.floor(capital * POS_CAP_PCT / 100 / r["price"])
    sh = max(0, min(sh_risk, sh_cap))
    return {"shares": sh, "value": sh * r["price"], "value_pct": sh * r["price"] / capital * 100,
            "risk_usd": sh * r["risk_ps"], "capped": sh_cap < sh_risk}


def exit_date(date_str: str):
    """建議持有期滿日（略過週末；美股假日未排除，僅供參考）。"""
    try:
        return (pd.Timestamp(date_str) + pd.offsets.BDay(HOLD_DAYS)).strftime("%Y-%m-%d")
    except Exception:
        return None


def scan(data: dict, spy: pd.DataFrame, names=None, capital=None):
    """data: symbol → 日K。回傳 {ok, market, rows, scanned}；rows 依 5 日跌幅由深到淺。"""
    names = names or {}
    rows, scanned = [], 0
    for sym, df in data.items():
        if df is None or len(df) < 205:
            continue
        scanned += 1
        r = screen_one(df)
        if r:
            r["symbol"] = sym
            r["name"] = names.get(sym, "")
            r["exit_date"] = exit_date(r["date"])
            r["size"] = size(r, capital)
            rows.append(r)
    rows.sort(key=lambda x: x["ret5"])
    return {"market": spy_state(spy), "rows": rows, "scanned": scanned}
