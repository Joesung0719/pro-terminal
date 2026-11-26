"""
抄底多單停損檢討（只讀取與回測，不改推播／篩選）
A. Joe 批1 持倉（QQQ/GOOGL/AMD/SPCX）＋觀察檔：11/20～最新 日K，對照計畫停損價（盤中是否碰到、收盤是否破）、停損距離＝幾個 ATR
B. SOXX 近日走勢
C. 歷史「恐慌抄底」多單回測：5日跌≥10% 且收在季線(60日)下 → 當日收盤買進，比較各種停損方式
輸出 backtest_reports/dip_stop_review.md
"""
import os, sys, datetime
import numpy as np, pandas as pd
import yfinance as yf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import us_exit_backtest as B
import stop_audit as S

OUT = os.path.join(B.BASE, "backtest_reports", "dip_stop_review.md")
COST = B.COST_PCT / 100.0
HOLD = {"QQQ": (5, 651.47, 606.0), "GOOGL": (10, 284.06, 264.0), "AMD": (7, 457.51, 418.0), "SPCX": (20, 136.13, 126.0)}
WATCH = {"PLTR": 144.8, "NVDA": 131.2, "COHR": 162.1, "SNDK": 64.3}   # five-picks 方案的初始停損（未必有買）


def part_a(lines):
    syms = list(HOLD) + list(WATCH) + ["SOXX", "SPY"]
    H = S.raw_hist(syms, period="6mo")
    lines.append("## A. Joe 批1 持倉 vs 計畫停損（價格未還原；停損是計畫書的「收盤價」停損）\n")
    rows = []
    for s, (sh, cost, stop) in {**HOLD, **{k: (None, None, v) for k, v in WATCH.items()}}.items():
        d = H.get(s)
        if d is None:
            rows.append({"標的": s, "備註": "—（抓不到）"}); continue
        atr = S.atr_series(d)
        a = float(atr.iloc[-1])
        after = d[d.index >= pd.Timestamp("2026-11-23")]
        for ix, r in after.iterrows():
            rows.append({"標的": s, "持倉": "批1" if sh else "未確認持有", "日期": f"{ix:%m-%d}", "開": round(r.Open, 2), "高": round(r.High, 2),
                         "低": round(r.Low, 2), "收": round(r.Close, 2), "計畫停損": stop,
                         "低點距停損%": round((r.Low / stop - 1) * 100, 2),
                         "盤中碰停損": "是" if r.Low <= stop else "否", "收盤破停損": "是" if r.Close <= stop else "否",
                         "停損距成本/ATR": round((cost - stop) / a, 2) if cost else np.nan,
                         "ATR14": round(a, 2)})
    lines.append("```\n" + pd.DataFrame(rows).to_string(index=False) + "\n```\n")
    lines.append("## B. SOXX（費半 ETF）與 SPY 近 6 日\n")
    for s in ("SOXX", "SPY"):
        d = H.get(s)
        if d is None:
            lines.append(f"- {s}：—"); continue
        d = d.tail(7); pc = d["Close"].shift(1); out = []
        for i in range(1, len(d)):
            r = d.iloc[i]
            out.append({"日期": f"{d.index[i]:%m-%d}", "開": round(r.Open, 2), "高": round(r.High, 2), "低": round(r.Low, 2), "收": round(r.Close, 2),
                        "收盤%": round((r.Close / pc.iloc[i] - 1) * 100, 2), "最低vs昨收%": round((r.Low / pc.iloc[i] - 1) * 100, 2),
                        "收盤離低點%": round((r.Close / r.Low - 1) * 100, 2)})
        lines.append(f"**{s}**\n```\n{pd.DataFrame(out).to_string(index=False)}\n```\n")


VARS = ["固定-7% 盤中", "固定-7% 收盤確認", "固定-10% 盤中", "固定-10% 收盤確認", "2×ATR 盤中", "3×ATR 盤中", "3×ATR 收盤確認",
        "恐慌低點 盤中", "恐慌低點 收盤確認", "恐慌低點-1ATR 盤中", "收盤-7%＋盤中-12%災難", "不設停損（對照）"]


def sim(o, h, l, c, i, H, atr, lowpanic):
    """i＝訊號日（收盤買進）；之後第 1..H 日判斷。回傳 {variant:(ret,hold,stopped)}"""
    n = len(c)
    if i + H >= n or not np.isfinite(atr) or atr <= 0:
        return None
    ent = c[i]
    res = {}

    def fin(px, k, st):
        return (px / ent - 1 - COST, k, st)

    def intr(level):
        for k in range(1, H + 1):
            j = i + k
            if l[j] <= level:
                return fin(o[j] if o[j] <= level else level, k, True)
        return fin(c[i + H], H, False)

    def cc(level):
        for k in range(1, H + 1):
            j = i + k
            if c[j] <= level:
                return fin(c[j], k, True)
        return fin(c[i + H], H, False)

    def hyb(cl, cat):
        for k in range(1, H + 1):
            j = i + k
            if l[j] <= cat:
                return fin(o[j] if o[j] <= cat else cat, k, True)
            if c[j] <= cl:
                return fin(c[j], k, True)
        return fin(c[i + H], H, False)

    res[VARS[0]] = intr(ent * 0.93); res[VARS[1]] = cc(ent * 0.93)
    res[VARS[2]] = intr(ent * 0.90); res[VARS[3]] = cc(ent * 0.90)
    res[VARS[4]] = intr(ent - 2 * atr); res[VARS[5]] = intr(ent - 3 * atr); res[VARS[6]] = cc(ent - 3 * atr)
    if lowpanic < ent:
        res[VARS[7]] = intr(lowpanic); res[VARS[8]] = cc(lowpanic); res[VARS[9]] = intr(lowpanic - atr)
    else:
        return None
    res[VARS[10]] = hyb(ent * 0.93, ent * 0.88)
    res[VARS[11]] = fin(c[i + H], H, False)
    return res


def part_c(lines):
    years = int(os.environ.get("YEARS", "5"))
    data = B.download(B.load_universe(), years)
    start = pd.Timestamp(datetime.date.today() - datetime.timedelta(days=int(365.25 * years)))
    for H in (10, 20):
        recs = []
        for sym, df in data.items():
            df = df.sort_index()
            o, h, l, c = (df[k].values for k in ("Open", "High", "Low", "Close"))
            ma60 = df["Close"].rolling(60, min_periods=60).mean().values
            atr = S.atr_series(df).values
            ret5 = df["Close"].pct_change(5).values
            low5 = df["Low"].rolling(5, min_periods=5).min().values
            last = -999
            for i in range(65, len(df) - H - 1):
                if df.index[i] < start or i - last < 10:
                    continue
                if ret5[i] <= -0.10 and c[i] < ma60[i]:
                    r = sim(o, h, l, c, i, H, atr[i], low5[i])
                    if r is None:
                        continue
                    last = i
                    for v, (ret, hold, st) in r.items():
                        recs.append(dict(sym=sym, date=df.index[i], year=df.index[i].year, variant=v, ret=ret * 100, hold=hold, stopped=st))
        t = pd.DataFrame(recs)
        if t.empty:
            lines.append(f"## C. 持有上限 {H} 天：—\n"); continue
        lines.append(f"## C. 恐慌抄底多單｜5日跌≥10%且收在季線下，當日收盤買｜最長持有 {H} 天｜訊號 {len(t) // len(VARS)} 筆（分布 {t['date'].nunique()} 個交易日，同日訊號不獨立）\n")
        rows = []
        for v in VARS:
            g = t[t["variant"] == v]; r = g["ret"].values; w, lo = r[r > 0], r[r <= 0]
            ym = g.groupby("year")["ret"].mean()
            rows.append({"停損方式": v, "勝率%": round((r > 0).mean() * 100, 1), "期望值%": round(r.mean(), 2),
                         "平均賺%": round(w.mean(), 2) if len(w) else np.nan, "平均賠%": round(lo.mean(), 2) if len(lo) else np.nan,
                         "停損率%": round(g["stopped"].mean() * 100, 1), "最差%": round(r.min(), 1), "賺錢年數": f"{int((ym > 0).sum())}/{len(ym)}"})
        lines.append("```\n" + pd.DataFrame(rows).to_string(index=False) + "\n```\n")
        a = t[t["variant"] == VARS[0]].set_index(["sym", "date"]); nb = t[t["variant"] == VARS[11]].set_index(["sym", "date"])
        j = a.join(nb["ret"].rename("hold")); st = j[j["stopped"]]
        if len(st):
            lines.append(f"- 固定-7% 盤中被停損 {len(st)} 筆；不停損抱到期反而賺的比例（洗盤率）{(st['hold'] > 0).mean() * 100:.1f}%\n")


def main():
    lines = [f"# 抄底多單停損檢討（{datetime.datetime.utcnow():%Y-%m-%d %H:%M} UTC）\n", "只讀取與回測，不改推播／篩選。抓不到標「—」。\n"]
    for f in (part_a, part_c):
        try:
            f(lines)
        except Exception as ex:  # noqa
            lines.append(f"{f.__name__} 失敗：{ex!r}\n"); print(f.__name__, ex)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
