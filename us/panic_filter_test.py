"""
美股空方：大盤恐慌日不推空的回測（只回測，不接推播）
訊號日（收盤後）已知的條件：SPY 當日跌幅、VIX 收盤。比較照推 vs 不推的勝率與期望值。
輸出 backtest_reports/panic_filter.md
"""
import os, sys, datetime
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import us_exit_backtest as B
import stop_audit as S

OUT = os.path.join(B.BASE, "backtest_reports", "panic_filter.md")
RULES = [("SPY單日跌≥3%", lambda spy, vix: spy <= -3.0),
         ("VIX收盤>35", lambda spy, vix: vix > 35),
         ("SPY跌≥3% 或 VIX>35（Joe 條件）", lambda spy, vix: (spy <= -3.0) | (vix > 35)),
         ("SPY跌≥2% 或 VIX>30（較寬）", lambda spy, vix: (spy <= -2.0) | (vix > 30)),
         ("SPY跌≥2%", lambda spy, vix: spy <= -2.0)]


def stats(r):
    r = np.asarray(r)
    if len(r) == 0:
        return dict(n=0)
    w, l = r[r > 0], r[r <= 0]
    return dict(n=len(r), 勝率=round((r > 0).mean() * 100, 1), 期望值=round(r.mean(), 2),
                平均賺=round(w.mean(), 2) if len(w) else np.nan, 平均賠=round(l.mean(), 2) if len(l) else np.nan,
                最差=round(r.min(), 1))


def main():
    years = int(os.environ.get("YEARS", "5"))
    data = B.download(B.load_universe(), years)
    spy = B.load_index("SPY", years)
    vix = B.load_index("^VIX", years)
    spy_ret = (spy["Close"].pct_change() * 100)
    vix_c = vix["Close"]
    start = pd.Timestamp(datetime.date.today() - datetime.timedelta(days=int(365.25 * years)))
    lines = [f"# 空方恐慌日過濾回測（{datetime.datetime.utcnow():%Y-%m-%d %H:%M} UTC）\n",
             "訊號日收盤已知的條件；隔日開盤進場、扣 0.1% 成本；同 us_exit_backtest 的空方破底條件。只回測，未接推播。\n"]
    for mode, s in (("SHORT", -1),):
        for H in (5, 10):
            recs = []
            for sym, df in data.items():
                df = df.sort_index()
                o, h, l, c = (df[k].values for k in ("Open", "High", "Low", "Close"))
                atr = S.atr_series(df).values
                for i, neck in B.signals(df, mode, start):
                    e = i + 1
                    if e >= len(df) or not np.isfinite(atr[i]):
                        continue
                    r = S.sim(o, h, l, c, e, H, s, float(neck), float(atr[i]))
                    if r is None:
                        continue
                    d = df.index[i]
                    sv = spy_ret.get(d, np.nan); vv = vix_c.get(d, np.nan)
                    for v in (S.VARIANTS[0], S.VARIANTS[1], S.VARIANTS[8]):
                        recs.append(dict(date=d, year=d.year, spy=sv, vix=vv, variant=v, ret=r[v][0] * 100))
            t = pd.DataFrame(recs)
            lines.append(f"## 空方破底｜最長持有 {H} 天\n")
            known = t[t["spy"].notna() & t["vix"].notna()]
            lines.append(f"（有大盤資料的樣本 {len(known) // 3} 筆／全部 {len(t) // 3} 筆；缺資料的不納入過濾比較）\n")
            for v in (S.VARIANTS[0], S.VARIANTS[8]):
                g = known[known["variant"] == v]
                rows = [{"分組": "全部（照推）", **stats(g["ret"])}]
                for name, f in RULES:
                    m = f(g["spy"], g["vix"])
                    rows.append({"分組": f"[不推] {name}", **stats(g[m]["ret"])})
                    rows.append({"分組": f"[保留] 其餘（套用過濾後）", **stats(g[~m]["ret"])} if False else
                                {"分組": f"[套用過濾後] {name}", **stats(g[~m]["ret"])})
                lines.append(f"**{v}**\n```\n{pd.DataFrame(rows).to_string(index=False)}\n```\n")
            # Joe 條件的年度分布與觸發日數
            g = known[known["variant"] == S.VARIANTS[0]]
            m = (g["spy"] <= -3.0) | (g["vix"] > 35)
            lines.append(f"- Joe 條件觸發：{int(m.sum())} 筆訊號，分布在 {g[m]['date'].nunique()} 個交易日；"
                         f"年度筆數 {dict(g[m].groupby('year').size())}\n")
            yr = g[m].groupby("year")["ret"].agg(["size", "mean"]).round(2)
            lines.append("- 被過濾掉那組的年度期望值%（現行停損）：" + "、".join(f"{y} n={int(r['size'])} {r['mean']:+.2f}" for y, r in yr.iterrows()) + "\n")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
