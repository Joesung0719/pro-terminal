"""
美股多空雷達 ─ 停損被洗檢討（只讀取與回測，不改任何篩選／推播邏輯）

A. 近期推播標的：推播後每天的盤中高低、收盤相對頸線（停損參考）→ 是否盤中觸及、收盤是否收回
B. 大盤 SPY／QQQ／VIX 近 8 個交易日 OHLC
C. 5 年歷史訊號回測（同 us_exit_backtest 的進場條件，隔日開盤進場）：
   比較「停損怎麼設」——盤中觸頸線（現行）、收盤確認、頸線外加 ATR 緩衝、連 2 日收盤、收盤確認＋盤中災難停損

輸出：backtest_reports/stop_audit.md
"""
import os
import sys
import datetime

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import us_exit_backtest as B  # noqa: E402

BASE = B.BASE
OUT = os.path.join(BASE, "backtest_reports", "stop_audit.md")
SINCE = os.environ.get("AUDIT_SINCE", "2026-11-18")
COST = B.COST_PCT / 100.0


def raw_hist(symbols, period="3mo"):
    out = {}
    raw = yf.download(" ".join(symbols), period=period, group_by="ticker", threads=True,
                      progress=False, auto_adjust=False)
    multi = isinstance(raw.columns, pd.MultiIndex)
    for s in symbols:
        sub = raw[s].copy() if multi else raw.copy()
        sub = sub.dropna(subset=["Close"])
        if getattr(sub.index, "tz", None) is not None:
            sub.index = sub.index.tz_localize(None)
        if len(sub):
            out[s] = sub
    return out


def atr_series(df, n=14):
    pc = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - pc).abs(), (df["Low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


# --------------------------------------------------------------------------- A / B
def part_ab(lines):
    perf = pd.read_csv(os.path.join(BASE, "tracking", "us_push_perf.csv"), encoding="utf-8-sig")
    perf = perf[perf["date"] >= SINCE].copy()
    syms = sorted(set(perf["symbol"]) | {"SPY", "QQQ", "^VIX"})
    H = raw_hist(syms)
    last_day = max(df.index[-1] for df in H.values())
    lines.append(f"## A. 大盤 SPY／QQQ／VIX 近 8 個交易日（資料到 {last_day:%Y-%m-%d}，未還原）\n")
    for s in ("SPY", "QQQ", "^VIX"):
        d = H.get(s)
        if d is None:
            lines.append(f"- {s}：—（抓不到）")
            continue
        d = d.tail(9)
        pc = d["Close"].shift(1)
        rows = []
        for i in range(1, len(d)):
            r = d.iloc[i]
            rows.append({"日期": f"{d.index[i]:%m-%d}", "開": round(r.Open, 2), "高": round(r.High, 2),
                         "低": round(r.Low, 2), "收": round(r.Close, 2),
                         "收盤%": round((r.Close / pc.iloc[i] - 1) * 100, 2),
                         "最低vs昨收%": round((r.Low / pc.iloc[i] - 1) * 100, 2),
                         "收盤離低點%": round((r.Close / r.Low - 1) * 100, 2)})
        lines.append(f"**{s}**\n```\n{pd.DataFrame(rows).to_string(index=False)}\n```\n")

    lines.append(f"## B. {SINCE} 之後推播標的 vs 停損參考（頸線）\n")
    lines.append("停損＝推播卡片上的「停損參考」頸線；空方＝最高價 ≥ 頸線算盤中觸及，多方＝最低價 ≤ 頸線。"
                 "「收盤收回」＝觸及當天收盤又回到頸線的有利側。\n")
    rows = []
    for _, p in perf.sort_values(["date", "mode", "rank"]).iterrows():
        d = H.get(p.symbol)
        if d is None:
            rows.append({"推播日": p.date, "方向": p["mode"], "標的": p.symbol, "備註": "—（抓不到資料）"})
            continue
        atr = atr_series(d)
        after = d[d.index > pd.Timestamp(p.date)]
        neck, px = float(p.neck), float(p.push_px)
        short = p["mode"] == "SHORT"
        atr_pct = float(atr.loc[:pd.Timestamp(p.date)].iloc[-1] / px * 100) if len(atr.loc[:pd.Timestamp(p.date)]) else np.nan
        hit_day, closed_back, closed_beyond = None, None, None
        worst = None
        for ix, r in after.iterrows():
            ext = r.High if short else r.Low
            beyond = ext >= neck if short else ext <= neck
            if beyond and hit_day is None:
                hit_day = ix
                cb = r.Close >= neck if short else r.Close <= neck
                closed_beyond = bool(cb)
                closed_back = not cb
        last = after.iloc[-1] if len(after) else None
        sgn = -1 if short else 1
        rows.append({
            "推播日": p.date, "方向": p["mode"], "標的": p.symbol, "推播價": round(px, 2), "頸線": round(neck, 2),
            "停損距離%": round(abs(neck / px - 1) * 100, 2), "ATR%": round(atr_pct, 2),
            "距離/ATR": round(abs(neck / px - 1) * 100 / atr_pct, 2) if atr_pct == atr_pct and atr_pct else np.nan,
            "盤中觸及": f"{hit_day:%m-%d}" if hit_day is not None else "未觸及",
            "觸及當日收盤": ("仍越過" if closed_beyond else "收回") if hit_day is not None else "—",
            "最新收盤報酬%": round(sgn * (last.Close / px - 1) * 100, 2) if last is not None else np.nan,
            "最新收盤vs頸線": ("仍越過" if ((last.Close >= neck) if short else (last.Close <= neck)) else "在安全側") if last is not None else "—",
        })
    df = pd.DataFrame(rows)
    lines.append("```\n" + df.to_string(index=False) + "\n```\n")
    if "盤中觸及" in df:
        t = df[df["盤中觸及"].notna() & (df["盤中觸及"] != "未觸及") & (df["盤中觸及"] != "—")] if "盤中觸及" in df else df.iloc[0:0]
        lines.append(f"- 共 {len(df)} 筆；盤中觸及頸線 {len(t)} 筆；其中觸及當日收盤已收回 {int((t['觸及當日收盤'] == '收回').sum())} 筆；"
                     f"最新收盤仍在頸線危險側 {int((df['最新收盤vs頸線'] == '仍越過').sum())} 筆。\n")


# --------------------------------------------------------------------------- C
VARIANTS = [
    "A 現行：盤中觸頸線",
    "B 收盤確認頸線",
    "C 頸線外 0.5ATR 盤中",
    "D 頸線外 1ATR 盤中",
    "E 頸線外 1.5ATR 盤中",
    "F 連2日收盤越頸線",
    "G 收盤確認＋1.5ATR盤中災難停損",
    "H 收盤越過頸線外0.5ATR",
    "I 不設停損（對照）",
]


def sim(o, h, l, c, e, H, s, L0, a):
    """回傳 {variant: (ret, hold_days, stopped)}；s=+1 多、-1 空；e＝進場日 index（開盤進場）。"""
    n = len(c)
    end = min(e + H - 1, n - 1)
    if end < e + H - 1:        # 資料不足，不納入
        return None
    entry = o[e]
    if not np.isfinite(entry) or entry <= 0 or not np.isfinite(L0) or not np.isfinite(a):
        return None
    res = {}

    def fin(exitp, k, stopped):
        return (s * (exitp / entry - 1) - COST, k + 1, stopped)

    def adverse(px, level):   # 盤中極端價是否越過停損價
        return (px >= level) if s < 0 else (px <= level)

    def worse_ext(i):
        return h[i] if s < 0 else l[i]

    def beyond_close(i, level):
        return (c[i] >= level) if s < 0 else (c[i] <= level)

    def intraday(level):
        for k, i in enumerate(range(e, end + 1)):
            if adverse(worse_ext(i), level):
                gap = (o[i] >= level) if s < 0 else (o[i] <= level)
                return fin(o[i] if gap else level, k, True)
        return fin(c[end], end - e, False)

    def close_conf(level):
        for k, i in enumerate(range(e, end + 1)):
            if beyond_close(i, level):
                return fin(c[i], k, True)
        return fin(c[end], end - e, False)

    buf = lambda m: L0 + (-s) * m * a   # 空方往上加、多方往下減
    res[VARIANTS[0]] = intraday(L0)
    res[VARIANTS[1]] = close_conf(L0)
    res[VARIANTS[2]] = intraday(buf(0.5))
    res[VARIANTS[3]] = intraday(buf(1.0))
    res[VARIANTS[4]] = intraday(buf(1.5))
    # F 連2日收盤
    r = None
    for k, i in enumerate(range(e + 1, end + 1)):
        if beyond_close(i, L0) and beyond_close(i - 1, L0):
            r = fin(c[i], k + 1, True)
            break
    res[VARIANTS[5]] = r or fin(c[end], end - e, False)
    # G 收盤確認 ＋ 盤中災難停損
    cat = buf(1.5)
    r = None
    for k, i in enumerate(range(e, end + 1)):
        if adverse(worse_ext(i), cat):
            gap = (o[i] >= cat) if s < 0 else (o[i] <= cat)
            r = fin(o[i] if gap else cat, k, True)
            break
        if beyond_close(i, L0):
            r = fin(c[i], k, True)
            break
    res[VARIANTS[6]] = r or fin(c[end], end - e, False)
    res[VARIANTS[7]] = close_conf(buf(0.5))
    res[VARIANTS[8]] = fin(c[end], end - e, False)
    return res


def part_c(lines):
    years = int(os.environ.get("YEARS", "5"))
    syms = B.load_universe()
    print(f"下載 {len(syms)} 檔…")
    data = B.download(syms, years)
    spy = B.load_index("SPY", years)
    spy_v = ((spy["Close"] / spy["Low"] - 1) >= 0.01) & (spy["Close"] > spy["Open"])  # V 轉日：收盤離盤中低點 ≥1% 且收紅
    start = pd.Timestamp(datetime.date.today() - datetime.timedelta(days=int(365.25 * years)))
    for mode, s in (("SHORT", -1), ("LONG", 1)):
        for H in (5, 10):
            recs = []
            for sym, df in data.items():
                df = df.sort_index()
                o, h, l, c = (df[k].values for k in ("Open", "High", "Low", "Close"))
                atr = atr_series(df).values
                for i, neck in B.signals(df, mode, start):
                    e = i + 1
                    if e >= len(df) or not np.isfinite(atr[i]):
                        continue
                    r = sim(o, h, l, c, e, H, s, float(neck), float(atr[i]))
                    if r is None:
                        continue
                    stop_day = None
                    for v, (ret, hold, st) in r.items():
                        recs.append({"sym": sym, "date": df.index[i], "variant": v, "ret": ret * 100,
                                     "hold": hold, "stopped": st,
                                     "vday": bool(spy_v.get(df.index[min(e + hold - 1, len(df) - 1)], False)),
                                     "year": df.index[i].year})
            t = pd.DataFrame(recs)
            if t.empty:
                lines.append(f"## C. {mode} 持有上限 {H} 天：—（無訊號）\n")
                continue
            lines.append(f"## C. {'空方破底' if mode == 'SHORT' else '多方突破'}｜最長持有 {H} 天｜訊號 {t['date'].nunique() and len(t) // len(VARIANTS)} 筆（隔日開盤進場、扣 0.1% 成本）\n")
            rows = []
            for v in VARIANTS:
                g = t[t["variant"] == v]
                r = g["ret"].values
                win, loss = r[r > 0], r[r <= 0]
                ym = g.groupby("year")["ret"].mean()
                rows.append({"停損方式": v, "勝率%": round((r > 0).mean() * 100, 1), "期望值%": round(r.mean(), 2),
                             "平均賺%": round(win.mean(), 2) if len(win) else np.nan,
                             "平均賠%": round(loss.mean(), 2) if len(loss) else np.nan,
                             "賺賠比": round(win.mean() / -loss.mean(), 2) if len(win) and len(loss) else np.nan,
                             "停損率%": round(g["stopped"].mean() * 100, 1), "最差%": round(r.min(), 1),
                             "賺錢年數": f"{int((ym > 0).sum())}/{len(ym)}"})
            lines.append("```\n" + pd.DataFrame(rows).to_string(index=False) + "\n```\n")
            # 洗盤率：現行停損被打到的單，若不設停損持有到期是否反而賺錢
            a = t[t["variant"] == VARIANTS[0]].set_index(["sym", "date"])
            nb = t[t["variant"] == VARIANTS[8]].set_index(["sym", "date"])
            j = a.join(nb["ret"].rename("ret_hold"))
            st = j[j["stopped"]]
            if len(st):
                lines.append(f"- 現行（盤中觸頸線）被停損 {len(st)} 筆；若不停損持有到期反而賺錢的比例（洗盤率）："
                             f"{(st['ret_hold'] > 0).mean() * 100:.1f}%；停損當天是 SPY「V 轉日」"
                             f"（收盤離盤中低點≥1%且收紅）的有 {int(st['vday'].sum())} 筆，其洗盤率 "
                             f"{((st[st['vday']]['ret_hold'] > 0).mean() * 100 if st['vday'].any() else float('nan')):.1f}%。\n")


def main():
    lines = [f"# 美股停損被洗檢討（產生時間 {datetime.datetime.utcnow():%Y-%m-%d %H:%M} UTC）\n",
             "只讀取與回測，不改篩選與停損邏輯。價格來自 yfinance；抓不到的欄位標「—」。\n"]
    try:
        part_ab(lines)
    except Exception as ex:  # noqa: BLE001
        lines.append(f"A/B 部分失敗：{ex!r}\n")
        print("part_ab 失敗", ex)
    try:
        part_c(lines)
    except Exception as ex:  # noqa: BLE001
        lines.append(f"C 部分失敗：{ex!r}\n")
        print("part_c 失敗", ex)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
