"""
美股多空雷達 ─ 出場方式回測

進場條件：與網頁版突破榜預設相同（20日頸線首日突破/跌破、量比≥1.2x、收紅/黑K、站上/跌破月線），
股票池：us_universe.json（標普500＋那斯達克100）＋ 43 檔觀察名單，近 YEARS 年。
只比較「出場方式」：固定持有、移動停損 10%、ATR 移動停損，以及加上頸線出場／時間停損的組合。
規則細節見 us/exit_sim.py。

用法：
    python us/us_exit_backtest.py
輸出：
    backtest_reports/us_exit_LONG.md、backtest_reports/us_exit_SHORT.md
"""
import os
import sys
import json
import time
import datetime

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from us_scheduler import UNIVERSE as WATCHLIST  # noqa: E402
from exit_sim import atr14, strategies, simulate, report_md  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE, "backtest_reports")
YEARS = int(os.environ.get("YEARS", "5"))
COST_PCT = float(os.environ.get("COST_PCT", "0.1"))
NECK_DAYS = 20
VOL_RATIO = 1.2
MIN_GAP = 10   # 同一檔兩次訊號至少間隔 10 個交易日（與台股回測相同）


def load_universe():
    syms = [s for s, _, _ in WATCHLIST if s not in ("SPY", "QQQ", "SOXX", "DIA", "IWM")]
    rows = json.load(open(os.path.join(BASE, "us_universe.json"), encoding="utf-8"))
    syms += [str(r[0]).strip().upper().replace(".", "-") for r in rows]
    return sorted(set(syms))


def download(symbols, years):
    data = {}
    for i in range(0, len(symbols), 100):
        batch = symbols[i:i + 100]
        raw = None
        for attempt in range(3):
            try:
                raw = yf.download(" ".join(batch), period=f"{years + 1}y", group_by="ticker", threads=True,
                                  progress=False, auto_adjust=True)
                if raw is not None and not raw.empty:
                    break
            except Exception:
                time.sleep(1.5 * (attempt + 1))
        if raw is None or raw.empty:
            print(f"  批次 {i // 100 + 1} 下載失敗，略過")
            continue
        multi = isinstance(raw.columns, pd.MultiIndex)
        lvl0 = set(raw.columns.get_level_values(0)) if multi else set()
        for s in batch:
            sub = raw[s].copy() if (multi and s in lvl0) else (raw.copy() if (not multi and len(batch) == 1) else None)
            if sub is not None and "Close" in sub.columns:
                sub = sub.dropna(subset=["Close"])
                if len(sub) > 300:
                    if getattr(sub.index, "tz", None) is not None:
                        sub.index = sub.index.tz_localize(None)
                    data[s] = sub
        print(f"  已下載 {min(i + 100, len(symbols))}/{len(symbols)}，累計可用 {len(data)} 檔")
    return data


def signals(df, mode, start_date):
    c, o, v = df["Close"].values, df["Open"].values, df["Volume"].values
    ma20 = df["Close"].rolling(20, min_periods=20).mean().values
    vma5 = df["Volume"].shift(1).rolling(5, min_periods=3).mean().values
    if mode == "LONG":
        neck = df["High"].shift(1).rolling(NECK_DAYS, min_periods=NECK_DAYS).max().values
    else:
        neck = df["Low"].shift(1).rolling(NECK_DAYS, min_periods=NECK_DAYS).min().values
    prev_neck = np.r_[np.nan, neck[:-1]]
    prev_c = np.r_[np.nan, c[:-1]]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = v / vma5
    if mode == "LONG":
        cond = (c > neck) & (prev_c <= prev_neck) & (c > o) & (c > ma20)
    else:
        cond = (c < neck) & (prev_c >= prev_neck) & (c < o) & (c < ma20)
    cond &= ratio >= VOL_RATIO
    out, last = [], -999
    for i in np.where(cond)[0]:
        if i < 30 or i - last < MIN_GAP or df.index[i] < start_date:
            continue
        out.append((i, neck[i]))
        last = i
    return out


def main():
    t0 = time.time()
    syms = load_universe()
    print(f"股票池：{len(syms)} 檔")
    hist = download(syms, YEARS)
    if len(hist) < len(syms) * 0.8:
        raise SystemExit(f"❌ 只下載到 {len(hist)}/{len(syms)} 檔，資料不完整，不產生報告")
    start_date = pd.Timestamp(datetime.date.today() - datetime.timedelta(days=int(365.25 * YEARS)))
    strats = strategies()
    os.makedirs(OUT_DIR, exist_ok=True)
    for mode in ("LONG", "SHORT"):
        recs = []
        for s, df in hist.items():
            o, h, l, c = (df[k].values for k in ("Open", "High", "Low", "Close"))
            atr = atr14(h, l, c)
            for i, neck in signals(df, mode, start_date):
                for name, p in strats:
                    r = simulate(i, mode, o, h, l, c, atr, neck, p, COST_PCT)
                    if r is not None:
                        recs.append({"strategy": name, "ret": r[0], "days": r[1], "reason": r[2],
                                     "year": df.index[i].year})
        trades = pd.DataFrame(recs)
        label = "多方突破" if mode == "LONG" else "空方破底"
        n_sig = int((trades["strategy"] == strats[0][0]).sum()) if len(trades) else 0
        title = [f"# 美股出場方式回測｜{label}\n",
                 f"- 產生時間：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} (UTC)",
                 f"- 進場：網頁突破榜預設條件（{NECK_DAYS}日頸線首日{'突破' if mode == 'LONG' else '跌破'}、量比≥{VOL_RATIO}x、"
                 f"{'收紅K、站上月線' if mode == 'LONG' else '收黑K、跌破月線'}），訊號隔日開盤進場",
                 f"- 近 {YEARS} 年｜股票池 {len(syms)} 檔（標普500＋那斯達克100＋觀察名單，有資料 {len(hist)} 檔）",
                 f"- 訊號數：{n_sig} 筆（只計入之後還有滿 60 個交易日資料的訊號，各出場方式用同一批）",
                 f"- 成本：來回 {COST_PCT}%｜價格：還原權息｜停損觸發以停損價成交，跳空越過則以開盤價成交｜最長持有 60 天"]
        md = report_md(title, trades) if not trades.empty else "\n".join(title) + "\n\n暫無資料\n"
        with open(os.path.join(OUT_DIR, f"us_exit_{mode}.md"), "w", encoding="utf-8") as f:
            f.write(md)
        print(md)
    print(f"完成，耗時 {time.time() - t0:.0f} 秒")


if __name__ == "__main__":
    main()
