"""
美股多空雷達 ─ 大盤狀態、移動停損階段、部位試算（推播與網頁共用，全部用 % 表示）

資料原則：全部由真實日K計算；算不出來（資料不足、沒設定本金）就回傳 None，畫面顯示「—」，不放預設數字。

【大盤狀態】SPY／QQQ 收盤相對季線（60 日）與 200 日線。只標示狀態；是否用來過濾推播，
            依 backtest_reports/us_exit_LONG.md 的「大盤濾網驗證」結果決定。

【移動停損階段】參考 analysis/trailing-stop-plan-2026-11-23.md，把 ATR 倍數全部換算成「相對成交價的 %」：
  ATR% ＝ 14 日 ATR（Wilder）÷ 收盤價
  ① 初始停損   ＝ 成交價 −T%（T＝移動回檔 %：ATR% ≤4% → 10、≤5% → 12、其餘 15）
                 （美股回測最佳出場為「最高價回檔 10%」；波動大的股票放寬，計畫書同做法）
  ② 漲到 +1 個 ATR%：停損上移到成交價 −0.5 個 ATR%
  ③ 漲到 +2 個 ATR%：停損上移到成交價（保本）
  ④ 之後：停損 ＝ max(保本價, 最高收盤 ×(1−T%))，只升不降
  % 都以「實際成交價」為基準；推播價只是估算。

【部位試算】需要本金（環境變數 ACCOUNT_USD；網頁版可直接輸入）。
  每筆最多虧本金 RISK_PER_TRADE_PCT（預設 1%）→ 股數 ＝ 本金 × 1% ÷ 每股風險（成交價 × T%）。
  同時持有的總風險上限 TOTAL_RISK_CAP_PCT（預設 5%，等於最多同時 5 檔）；當日名單超過的部分標「不建議再開」。
  月虧 MONTH_STOP_PCT（預設 5%）停手＝提醒文字（雷達不記帳，需自己對帳）。
  以上三個數字是建議值，可用環境變數調整。
"""
import os
import math

import numpy as np
import pandas as pd


def _env_f(name, default):
    try:
        v = float(os.environ.get(name, "").strip())
        return v if v > 0 else default
    except ValueError:
        return default


RISK_PER_TRADE_PCT = _env_f("RISK_PER_TRADE_PCT", 1.0)
TOTAL_RISK_CAP_PCT = _env_f("TOTAL_RISK_CAP_PCT", 5.0)
MONTH_STOP_PCT = _env_f("MONTH_STOP_PCT", 5.0)


def account_usd():
    """本金（美元）。沒設定或不是正數就回傳 None（不預設假數字）。"""
    try:
        v = float(os.environ.get("ACCOUNT_USD", "").replace(",", "").strip())
        return v if v > 0 else None
    except ValueError:
        return None


# ------------------------------------------------------------------------------
# 大盤狀態
# ------------------------------------------------------------------------------
def _ma_state(close: pd.Series, n: int):
    ma = close.rolling(n, min_periods=n).mean().iloc[-1]
    c = close.iloc[-1]
    if pd.isna(ma) or ma <= 0:
        return None
    return {"ma": float(ma), "above": bool(c > ma), "dist": float((c / ma - 1) * 100)}


def market_state(dfs: dict, symbols=("SPY", "QQQ")):
    """dfs: symbol → 日K DataFrame。回傳 [{symbol, close, q(季線), y(200日)}]；抓不到的代號不放進去。"""
    out = []
    for s in symbols:
        df = dfs.get(s)
        if df is None or df.empty or "Close" not in df.columns:
            continue
        close = df["Close"].dropna()
        if len(close) < 60:
            continue
        out.append({"symbol": s, "close": float(close.iloc[-1]), "q": _ma_state(close, 60), "y": _ma_state(close, 200)})
    return out


def market_label(state):
    """一句話總結：全部在季線上＝大盤偏多；全部在季線下＝大盤在季線下；其餘＝多空分歧；沒資料＝None。"""
    flags = [m["q"]["above"] for m in state if m.get("q")]
    if not flags:
        return None
    if all(flags):
        return "大盤偏多（SPY／QQQ 都在季線上）" if len(flags) > 1 else "大盤在季線上"
    if not any(flags):
        return "大盤在季線下" if len(flags) == 1 else "大盤偏弱（SPY／QQQ 都在季線下）"
    return "大盤多空分歧（SPY／QQQ 一個在季線上、一個在下）"


# ------------------------------------------------------------------------------
# 移動停損階段（%）
# ------------------------------------------------------------------------------
def atr_pct(df: pd.DataFrame, n: int = 14):
    """Wilder ATR(14) ÷ 最新收盤 × 100；資料不足回傳 None。"""
    if df is None or len(df) < n + 2:
        return None
    h, l, c = (df[k].astype(float) for k in ("High", "Low", "Close"))
    prev = c.shift(1)
    tr = pd.concat([h - l, (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    atr = tr.iloc[1:].ewm(alpha=1 / n, adjust=False, min_periods=n).mean().iloc[-1]
    last = float(c.iloc[-1])
    if pd.isna(atr) or not last > 0:
        return None
    return float(atr / last * 100)


def trail_pct(atr_p: float) -> int:
    return 10 if atr_p <= 4 else (12 if atr_p <= 5 else 15)


def stop_plan(price: float, atr_p):
    """回傳各階段（相對成交價的 %）與以推播價估算的停損價；atr_p 為 None 時回傳 None。"""
    if atr_p is None or not price > 0:
        return None
    t = trail_pct(atr_p)
    a = atr_p
    return {
        "atr_pct": round(a, 1), "trail": t,
        "s0": {"stop_pct": -float(t), "stop_px": price * (1 - t / 100)},
        "s1": {"gain_pct": round(a, 1), "stop_pct": round(-0.5 * a, 1), "gain_px": price * (1 + a / 100),
               "stop_px": price * (1 - 0.5 * a / 100)},
        "s2": {"gain_pct": round(2 * a, 1), "gain_px": price * (1 + 2 * a / 100), "stop_px": price},
    }


def plan_lines(plan, price_hint=True):
    """推播／網頁共用的文字（四行）；沒有 plan 回傳 ['移動停損 —（資料不足）']。"""
    if not plan:
        return ["移動停損 —（資料不足）"]
    p = plan
    return [
        f"① 先設停損 −{p['trail']}%（約 ${p['s0']['stop_px']:.2f}）",
        f"② 漲 +{p['s1']['gain_pct']}%（${p['s1']['gain_px']:.2f}）→ 停損上移到 {p['s1']['stop_pct']:+.1f}%",
        f"③ 漲 +{p['s2']['gain_pct']}%（${p['s2']['gain_px']:.2f}）→ 停損上移到成本（保本）",
        f"④ 之後從最高收盤回落 {p['trail']}% 出場，只升不降",
    ]


# ------------------------------------------------------------------------------
# 部位試算
# ------------------------------------------------------------------------------
def size_position(price: float, plan, capital, risk_pct=None):
    """股數＝本金×風險%÷每股風險。沒本金／沒停損計畫回傳 None。"""
    risk_pct = RISK_PER_TRADE_PCT if risk_pct is None else risk_pct
    if not capital or not plan or not price > 0:
        return None
    per_share = price * plan["trail"] / 100
    risk_usd = capital * risk_pct / 100
    shares = int(math.floor(risk_usd / per_share))
    value = shares * price
    return {"shares": shares, "value": value, "value_pct": value / capital * 100,
            "risk_usd": shares * per_share, "risk_pct": shares * per_share / capital * 100}


def allocate(picks, capital):
    """依名單順序（量比高到低）逐檔試算，累計總風險超過上限就標「不建議再開」。
    每檔 pick 需已有 pick['plan']；結果寫入 pick['size']（None＝沒本金或算不出）與 pick['over_cap']。
    回傳 {'max_n', 'used_risk_pct', 'skipped'}；沒本金時回傳 None。"""
    if not capital:
        return None
    used, skipped = 0.0, 0
    for r in picks:
        r["size"], r["over_cap"] = None, False
        sz = size_position(r["price"], r.get("plan"), capital)
        if sz is None:
            continue
        if used + RISK_PER_TRADE_PCT > TOTAL_RISK_CAP_PCT + 1e-9:
            r["over_cap"] = True
            skipped += 1
            continue
        r["size"] = sz
        used += sz["risk_pct"]
    return {"max_n": int(TOTAL_RISK_CAP_PCT // RISK_PER_TRADE_PCT), "used_risk_pct": used, "skipped": skipped}


def size_text(r):
    """單檔部位文字；沒本金回傳 None（卡片上就不顯示這行，由頁尾統一提示設定方式）。"""
    if r.get("over_cap"):
        return f"🚫 已達總風險上限 {TOTAL_RISK_CAP_PCT:g}%，不建議再開"
    sz = r.get("size")
    if sz is None:
        return None
    if sz["shares"] < 1:
        return "部位：本金不足以買 1 股（風險 1% 內）"
    return (f"部位：{sz['shares']} 股（${sz['value']:,.0f}，占本金 {sz['value_pct']:.0f}%）"
            f"｜停損觸及虧 ${sz['risk_usd']:,.0f}（{sz['risk_pct']:.2f}%）")
