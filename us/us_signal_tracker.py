"""
美股多空雷達 ─ 突破／破底追蹤名單（前瞻驗證）

每個交易日收盤後，把「當天」符合網頁版預設條件的多方突破與空方破底凍結存檔（事後不重算），
再追蹤之後第 3／5／7 個交易日的真實表現，並和 SPY 比較。

條件：與網頁版 /api/screener 預設相同（20日頸線、量比≥1.2x、收紅/黑K、站上/跌破月線），
      另外只記錄「首日」突破/跌破，避免同一檔連續多天重複計入。
股票池：us_universe.json（標普500＋那斯達克100）＋ 43 檔觀察名單。

用法：
    python us/us_signal_tracker.py
輸出：
    tracking/us_signals.csv    每一筆訊號與追蹤結果（累積）
    tracking/us_summary.md     分組統計

計算方式：
  - 進場：訊號隔日開盤價；報酬 = 第 N 個交易日收盤 vs 進場價，扣來回成本（預設 0.1%，空方取反向）。
  - 報酬用還原權息後的價格計算；名單上顯示的價格是當天實際價格。
  - 大盤基準：SPY 同期間（隔日開盤 → 第 N 日收盤）的報酬，空方同樣取反向。
  - 假突破：7 個交易日內任一天收盤跌回頸線下（空方：站回頸線上）。
  - 抓不到的資料一律留空，不補數字。
"""
import os
import sys
import json
import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from us_scheduler import UNIVERSE as WATCHLIST, NYSE_HOLIDAYS  # noqa: E402

ET = ZoneInfo("America/New_York")
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE, "tracking")
SIGNALS_CSV = os.path.join(OUT_DIR, "us_signals.csv")
SUMMARY_MD = os.path.join(OUT_DIR, "us_summary.md")
NECK_DAYS = 20
VOL_RATIO = 1.2
COST_PCT = float(os.environ.get("COST_PCT", "0.1"))
HORIZONS = (3, 5, 7)
MAX_H = max(HORIZONS)
MIN_N = 30
BENCH = "SPY"

COLS = ["date", "mode", "symbol", "name", "close", "pct", "neck", "dist_neck", "v_ratio",
        "above_240", "market_above_ma20",
        "entry_date", "entry_px", "days_done", "r_last", "fail_neck", "mfe", "mae"] + \
       [f"r{h}" for h in HORIZONS] + [f"bench{h}" for h in HORIZONS] + [f"ex{h}" for h in HORIZONS]
BOOL_COLS = ["above_240", "market_above_ma20", "fail_neck"]


def load_universe():
    names = {s: n for s, n, _ in WATCHLIST}
    syms = list(names)
    try:
        rows = json.load(open(os.path.join(BASE, "us_universe.json"), encoding="utf-8"))
        syms += [str(r[0]).strip().upper().replace(".", "-") for r in rows]
    except Exception as e:
        print(f"⚠️ us_universe.json 讀取失敗，只用 43 檔觀察名單：{e}")
    return sorted(set(syms)), names


def download(symbols, period):
    data = {}
    for i in range(0, len(symbols), 100):
        batch = symbols[i:i + 100]
        raw = None
        for _ in range(3):
            try:
                raw = yf.download(" ".join(batch), period=period, group_by="ticker", threads=True,
                                  progress=False, auto_adjust=False)
                if raw is not None and not raw.empty:
                    break
            except Exception:
                pass
        if raw is None or raw.empty:
            print(f"  批次 {i // 100 + 1} 下載失敗，略過")
            continue
        multi = isinstance(raw.columns, pd.MultiIndex)
        lvl0 = set(raw.columns.get_level_values(0)) if multi else set()
        for s in batch:
            sub = raw[s].copy() if (multi and s in lvl0) else (raw.copy() if (not multi and len(batch) == 1) else None)
            if sub is None or "Close" not in sub.columns:
                continue
            sub = sub.dropna(subset=["Close"])
            if sub.empty:
                continue
            if getattr(sub.index, "tz", None) is not None:
                sub.index = sub.index.tz_localize(None)
            if "Adj Close" not in sub.columns:
                sub["Adj Close"] = sub["Close"]
            data[s] = sub
    return data


def adj_factor(df):
    return (df["Adj Close"] / df["Close"]).replace([np.inf, -np.inf], np.nan).fillna(1.0)


def detect_today(sym, name, df, today, mkt_above):
    if len(df) < max(NECK_DAYS + 6, 25) or df.index[-1].strftime("%Y-%m-%d") != today:
        return []
    c, o, v = float(df["Close"].iloc[-1]), float(df["Open"].iloc[-1]), float(df["Volume"].iloc[-1])
    prev_c = float(df["Close"].iloc[-2])
    ma20 = df["Close"].rolling(20, min_periods=20).mean().iloc[-1]
    vma5 = df["Volume"].shift(1).rolling(5, min_periods=3).mean().iloc[-1]
    if pd.isna(ma20) or pd.isna(vma5) or vma5 <= 0:
        return []
    ratio = v / float(vma5)
    ma240 = df["Close"].iloc[-240:].mean() if len(df) >= 240 else np.nan
    win = df.iloc[-(NECK_DAYS + 1):-1]
    prev_win = df.iloc[-(NECK_DAYS + 2):-2]
    out = []
    for mode in ("LONG", "SHORT"):
        if mode == "LONG":
            neck, prev_neck = float(win["High"].max()), float(prev_win["High"].max())
            ok = c > neck and c > o and c > float(ma20) and ratio >= VOL_RATIO and prev_c <= prev_neck
            dist = (c / neck - 1) * 100
        else:
            neck, prev_neck = float(win["Low"].min()), float(prev_win["Low"].min())
            ok = c < neck and c < o and c < float(ma20) and ratio >= VOL_RATIO and prev_c >= prev_neck
            dist = (neck / c - 1) * 100
        if not ok:
            continue
        out.append({"date": today, "mode": mode, "symbol": sym, "name": name, "close": round(c, 2),
                    "pct": round((c / prev_c - 1) * 100, 2), "neck": round(neck, 2), "dist_neck": round(dist, 2),
                    "v_ratio": round(ratio, 2), "above_240": None if np.isnan(ma240) else bool(c >= ma240),
                    "market_above_ma20": mkt_above, "days_done": 0})
    return out


def update_outcome(row, df, bench):
    sig_dt = pd.Timestamp(row["date"])
    after = df[df.index > sig_dt]
    if after.empty:
        return row
    sgn = 1.0 if row["mode"] == "LONG" else -1.0
    f = adj_factor(df)
    f_sig = f.asof(sig_dt)
    f_sig = 1.0 if pd.isna(f_sig) else float(f_sig)
    seg = after.iloc[:MAX_H]
    fs = f.loc[seg.index]
    entry_raw = float(seg["Open"].iloc[0])
    if not entry_raw > 0:
        return row
    entry = entry_raw * float(fs.iloc[0])
    row["entry_date"] = seg.index[0].strftime("%Y-%m-%d")
    row["entry_px"] = round(entry_raw, 2)
    row["days_done"] = len(seg)
    closes = seg["Close"] * fs
    highs, lows = seg["High"] * fs, seg["Low"] * fs
    row["r_last"] = round(sgn * (float(closes.iloc[-1]) / entry - 1) * 100 - COST_PCT, 2)
    if sgn > 0:
        row["mfe"] = round((highs.max() / entry - 1) * 100, 2)
        row["mae"] = round((lows.min() / entry - 1) * 100, 2)
    else:
        row["mfe"] = round((1 - lows.min() / entry) * 100, 2)
        row["mae"] = round((1 - highs.max() / entry) * 100, 2)
    neck_adj = float(row["neck"]) * f_sig
    broke = (closes < neck_adj) if sgn > 0 else (closes > neck_adj)
    row["fail_neck"] = bool(broke.any()) if (broke.any() or len(seg) >= MAX_H) else None
    b_after = bench[bench.index > sig_dt] if bench is not None else None
    for h in HORIZONS:
        if len(seg) < h:
            continue
        r = sgn * (float(closes.iloc[h - 1]) / entry - 1) * 100 - COST_PCT
        row[f"r{h}"] = round(r, 2)
        if b_after is not None and len(b_after) >= h and b_after.index[h - 1] == seg.index[h - 1]:
            bf = adj_factor(b_after)
            b = sgn * (float(b_after["Close"].iloc[h - 1] * bf.iloc[h - 1]) / float(b_after["Open"].iloc[0] * bf.iloc[0]) - 1) * 100
            row[f"bench{h}"] = round(b, 2)
            row[f"ex{h}"] = round(r - b, 2)
    return row


def horizon_stats(g):
    """回傳 {h: {...}}，給 summary.md 與網頁 API 共用；沒有資料的欄位為 None。"""
    out = {}
    for h in HORIZONS:
        a = pd.to_numeric(g[f"r{h}"], errors="coerce").dropna()
        ex = pd.to_numeric(g[f"ex{h}"], errors="coerce").dropna()
        n = len(a)
        out[h] = {"n": n, "enough": n >= MIN_N,
                  "win": round(float((a > 0).mean() * 100), 1) if n else None,
                  "avg": round(float(a.mean()), 2) if n else None,
                  "median": round(float(a.median()), 2) if n else None,
                  "excess": round(float(ex.mean()), 2) if len(ex) else None,
                  "beat": round(float((ex > 0).mean() * 100), 1) if len(ex) else None}
    fin = g[g["fail_neck"].notna()]
    fail = {"rate": round(float((fin["fail_neck"] == True).mean() * 100), 1) if len(fin) else None, "n": len(fin)}  # noqa: E712
    return out, fail


def _p(v, pct=True):
    return "—" if v is None else (f"{v:+.2f}%" if pct else f"{v:.0f}%")


def group_lines(g, title):
    hs, fail = horizon_stats(g)
    fail_txt = "—" if fail["rate"] is None else f"{fail['rate']:.0f}%（{fail['n']} 筆）"
    lines = [f"\n**{title}**｜假突破率（7日內收盤回到頸線另一側）：{fail_txt}\n",
             "| 期間 | 樣本 | 勝率 | 平均 | 中位數 | 超額報酬(vs SPY) | 贏大盤比率 |",
             "|---|---|---|---|---|---|---|"]
    for h, s in hs.items():
        tag = "" if s["enough"] or s["n"] == 0 else " 樣本不足"
        lines.append(f"| {h}日 | {s['n']}{tag} | {_p(s['win'], False)} | {_p(s['avg'])} | {_p(s['median'])} | "
                     f"{_p(s['excess'])} | {_p(s['beat'], False)} |")
    return lines


def build_summary(sig):
    now = datetime.datetime.now(ET).strftime("%Y-%m-%d %H:%M")
    lines = ["# 美股突破／破底追蹤統計\n",
             f"- 更新：{now}（美東時間）",
             f"- 追蹤起始：{sig['date'].min() if len(sig) else '—'}｜訊號總數 {len(sig)}｜訊號天數 {sig['date'].nunique() if len(sig) else 0}",
             f"- 進場隔日開盤、扣成本 {COST_PCT}%、還原權息計算報酬；樣本少於 {MIN_N} 筆標示「樣本不足」。",
             "- 同一天的訊號受同一個大盤影響，彼此不獨立；請同時看「訊號天數」與「超額報酬」。"]
    for mode, label in (("LONG", "多方突破"), ("SHORT", "空方破底")):
        g = sig[sig["mode"] == mode]
        lines.append(f"\n## {label}")
        if g.empty:
            lines.append("\n暫無資料")
            continue
        lines += group_lines(g, "全部")
        for col, t, f in (("above_240", "年線", {True: "站上年線", False: "年線下"}),
                          ("market_above_ma20", "大盤(SPY)月線", {True: "在月線上", False: "在月線下"})):
            for k, lab in f.items():
                sub = g[g[col] == k]
                if len(sub):
                    lines += group_lines(sub, f"{t}：{lab}")
        v = pd.to_numeric(g["v_ratio"], errors="coerce")
        for cond, lab in ((v >= 2, "量比≥2x"), (v < 2, "量比<2x")):
            if cond.any():
                lines += group_lines(g[cond], lab)
    return "\n".join(lines) + "\n"


def read_signals(path_or_buf):
    df = pd.read_csv(path_or_buf, dtype={"symbol": str}, encoding="utf-8-sig")
    for col in BOOL_COLS:
        if col in df.columns:
            df[col] = df[col].map({True: True, False: False, "True": True, "False": False})
    return df


def main():
    now = datetime.datetime.now(ET)
    today = now.strftime("%Y-%m-%d")
    os.makedirs(OUT_DIR, exist_ok=True)
    old = read_signals(SIGNALS_CSV) if os.path.exists(SIGNALS_CSV) else pd.DataFrame(columns=COLS)

    bench = download([BENCH], "1y").get(BENCH)
    market_open = (now.weekday() < 5 and today not in NYSE_HOLIDAYS and bench is not None
                   and bench.index[-1].strftime("%Y-%m-%d") == today and now.hour >= 16)
    mkt_above = None
    if market_open and len(bench) >= 20:
        mkt_above = bool(bench["Close"].iloc[-1] >= bench["Close"].iloc[-20:].mean())

    new = []
    if not market_open:
        print(f"📅 美東 {today} 非交易日、尚未收盤或 SPY 沒有今天的K棒，不記錄新訊號")
    elif (old["date"] == today).any():
        print(f"ℹ️ {today} 已記錄過，不重複記錄")
    else:
        syms, names = load_universe()
        print(f"股票池 {len(syms)} 檔")
        hist = download(syms, "2y")
        print(f"可用資料 {len(hist)} 檔")
        for s in syms:
            if s in hist:
                new += detect_today(s, names.get(s, s), hist[s], today, mkt_above)
        print(f"今天新訊號：多方 {sum(r['mode'] == 'LONG' for r in new)}、空方 {sum(r['mode'] == 'SHORT' for r in new)}")

    sig = pd.concat([old, pd.DataFrame(new, columns=COLS)], ignore_index=True).astype(object)
    sig["days_done"] = pd.to_numeric(sig["days_done"], errors="coerce").fillna(0)
    todo = sig[(sig["days_done"] < MAX_H) & (sig["date"] < today)]
    if len(todo):
        hist = download(sorted(todo["symbol"].unique()), "6mo")
        for i, row in todo.iterrows():
            df = hist.get(row["symbol"])
            if df is not None:
                sig.loc[i] = update_outcome(row.copy(), df, bench)
    sig = sig[COLS].sort_values(["date", "mode", "symbol"], ascending=[False, True, True])
    sig.to_csv(SIGNALS_CSV, index=False, encoding="utf-8-sig")
    with open(SUMMARY_MD, "w", encoding="utf-8") as f:
        f.write(build_summary(sig))
    print(f"完成：累積 {len(sig)} 筆，追蹤中 {int((sig['days_done'] < MAX_H).sum())} 筆")


if __name__ == "__main__":
    main()
