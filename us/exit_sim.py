"""
出場方式模擬（台股 exit_backtest.py 與美股 us/us_exit_backtest.py 共用，兩個 repo 各放一份相同的檔案，修改時兩邊一起改）

同一批進場訊號，套用不同出場規則，比較「哪種出場方式讓這套條件真正賺錢」。

進場：訊號隔日開盤價。
出場規則（可組合）：
  - 固定持有 N 天：第 N 天收盤出場（對照組，與原本回測相同）
  - 移動停損 %：停損價 = 持有期間最高價 ×(1 − x%)，空方為最低價 ×(1 + x%)
  - ATR 移動停損：停損價 = 持有期間最高價 − k × ATR(14)，空方反向；ATR 每天更新
  - 頸線出場：收盤跌回頸線下（空方：站回頸線上）就以收盤價出場
  - 時間停損：連續 N 天沒有創新高（空方：新低）就以收盤價出場
停損觸發：當天最低價碰到停損價 → 以停損價出場；若開盤就跳空越過停損價，以開盤價出場（較保守）。
最高價只用「前一天以前」的資料更新，當天的高點不會拿來抬高當天的停損（避免偷看）。
最長持有 MAX_HOLD 天，到期以收盤價出場。
"""
import numpy as np
import pandas as pd

MAX_HOLD = 60


def atr14(h, l, c):
    prev_c = np.r_[np.nan, c[:-1]]
    tr = np.nanmax(np.vstack([h - l, np.abs(h - prev_c), np.abs(l - prev_c)]), axis=0)
    return pd.Series(tr).rolling(14, min_periods=14).mean().values


def ma_flags(c, mode):
    """回傳 (季線＋年線都在腳下, 所有均線都在腳下) 兩個布林陣列；空方反過來（均線都在頭上）。"""
    cs = pd.Series(c)
    mas = {p: cs.rolling(p, min_periods=p).mean().values for p in (5, 10, 20, 60, 120, 240)}
    with np.errstate(invalid="ignore"):
        if mode == "LONG":
            qy = (c > mas[60]) & (c > mas[240])
            allv = np.all([c > m for m in mas.values()], axis=0)
        else:
            qy = (c < mas[60]) & (c < mas[240])
            allv = np.all([c < m for m in mas.values()], axis=0)
    return qy, allv


def market_flag(index_df, mode):
    """大盤（加權指數／SPY）收盤在季線上（空方：季線下），回傳以日期為索引的布林 Series。"""
    c = index_df["Close"]
    ma60 = c.rolling(60, min_periods=60).mean()
    f = (c > ma60) if mode == "LONG" else (c < ma60)
    return f.where(ma60.notna())


FILTER_EXITS = ("固定持有10天（對照）", "移動停損10%", "ATR×3")


def strategies():
    """回傳 [(名稱, 參數)]；名稱用於報告。"""
    out = [("固定持有10天（對照）", {"hold": 10})]
    trails = [("移動停損10%", {"trail_pct": 10.0}),
              ("ATR×2", {"atr_k": 2.0}),
              ("ATR×2.5", {"atr_k": 2.5}),
              ("ATR×3", {"atr_k": 3.0})]
    for name, p in trails:
        out.append((name, dict(p)))
    for name, p in trails:
        out.append((name + "＋頸線出場", dict(p, neck_exit=True)))
    for name, p in trails:
        out.append((name + "＋5天沒創新高出場", dict(p, time_stop=5)))
    return out


def simulate(i, mode, o, h, l, c, atr, neck, p, cost_pct):
    """訊號在第 i 根K棒；回傳 (報酬%, 持有天數, 出場原因) 或 None（資料不足）。"""
    n = len(c)
    e = i + 1
    # 所有出場方式都用同一批訊號：只取「之後還有滿 MAX_HOLD 天資料」的訊號，避免近期還沒走完的單造成偏差
    if e + MAX_HOLD - 1 >= n or not (o[e] > 0):
        return None
    entry = o[e]
    long_ = mode == "LONG"
    sgn = 1.0 if long_ else -1.0
    hwm = entry               # 多方：持有期間最高價；空方：最低價
    no_new = 0
    hold = p.get("hold")
    last = e + (hold - 1 if hold else MAX_HOLD - 1)
    for j in range(e, last + 1):
        stop = None
        if "trail_pct" in p:
            stop = hwm * (1 - p["trail_pct"] / 100) if long_ else hwm * (1 + p["trail_pct"] / 100)
        elif "atr_k" in p:
            a = atr[j - 1]
            if not np.isnan(a):
                stop = hwm - p["atr_k"] * a if long_ else hwm + p["atr_k"] * a
        if stop is not None:
            if long_ and l[j] <= stop:
                px = min(o[j], stop)
                return sgn * (px / entry - 1) * 100 - cost_pct, j - e + 1, "停損"
            if (not long_) and h[j] >= stop:
                px = max(o[j], stop)
                return sgn * (px / entry - 1) * 100 - cost_pct, j - e + 1, "停損"
        if p.get("neck_exit") and ((long_ and c[j] < neck) or ((not long_) and c[j] > neck)):
            return sgn * (c[j] / entry - 1) * 100 - cost_pct, j - e + 1, "跌回頸線" if long_ else "站回頸線"
        new_ext = h[j] > hwm if long_ else l[j] < hwm
        if new_ext:
            hwm = h[j] if long_ else l[j]
            no_new = 0
        else:
            no_new += 1
        if p.get("time_stop") and no_new >= p["time_stop"]:
            return sgn * (c[j] / entry - 1) * 100 - cost_pct, j - e + 1, "時間停損"
    return sgn * (c[last] / entry - 1) * 100 - cost_pct, last - e + 1, "到期"


def summarize(trades):
    """trades: DataFrame[strategy, ret, days, reason, year]；回傳兩張報告用的表。"""
    rows = []
    for name, g in trades.groupby("strategy", sort=False):
        r = g["ret"].values
        win, loss = r[r > 0], r[r <= 0]
        rows.append({
            "出場方式": name, "n": len(r),
            "勝率%": round((r > 0).mean() * 100, 1),
            "期望值%": round(r.mean(), 2),
            "中位數%": round(float(np.median(r)), 2),
            "平均賺%": round(win.mean(), 2) if len(win) else np.nan,
            "平均賠%": round(loss.mean(), 2) if len(loss) else np.nan,
            "賺賠比": round(win.mean() / -loss.mean(), 2) if len(win) and len(loss) and loss.mean() < 0 else np.nan,
            "最差%": round(r.min(), 1),
            "平均持有天": round(g["days"].mean(), 1),
        })
    main = pd.DataFrame(rows)
    yearly = trades.pivot_table(index="strategy", columns="year", values="ret", aggfunc="mean", sort=False).round(2)
    yearly = yearly.reindex(sorted(yearly.columns), axis=1)
    yearly.index.name = "出場方式（各年度期望值%）"
    reasons = (trades.groupby(["strategy", "reason"], sort=False).size().unstack(fill_value=0))
    reasons = (reasons.div(reasons.sum(axis=1), axis=0) * 100).round(0)
    reasons.index.name = "出場方式（出場原因占比%）"
    return main, yearly, reasons


def filter_table(trades):
    filters = [("全部（原條件）", None),
               ("季線＋年線在腳下", trades["f_qy"]),
               ("所有均線在腳下", trades["f_all"]),
               ("大盤在季線上", trades["f_mkt"] == True),  # noqa: E712
               ("大盤在季線下", trades["f_mkt"] == False),  # noqa: E712
               ("所有均線在腳下＋大盤在季線上", trades["f_all"] & (trades["f_mkt"] == True))]  # noqa: E712
    rows = []
    for fname, mask in filters:
        sub = trades if mask is None else trades[mask.astype(bool)]
        for ex in FILTER_EXITS:
            g = sub[sub["strategy"] == ex]
            r = g["ret"].values
            if len(r) == 0:
                rows.append({"進場濾網": fname, "出場方式": ex, "n": 0})
                continue
            win, loss = r[r > 0], r[r <= 0]
            ym = g.groupby("year")["ret"].mean()
            rows.append({"進場濾網": fname, "出場方式": ex, "n": len(r),
                         "勝率%": round((r > 0).mean() * 100, 1), "期望值%": round(r.mean(), 2),
                         "賺賠比": round(win.mean() / -loss.mean(), 2) if len(win) and len(loss) and loss.mean() < 0 else np.nan,
                         "最差%": round(r.min(), 1), "賺錢年數": f"{int((ym > 0).sum())}/{len(ym)}"})
    return pd.DataFrame(rows)


def report_md(title_lines, trades):
    main, yearly, reasons = summarize(trades)
    parts = list(title_lines)
    parts += ["\n### 各出場方式比較（每筆交易，已扣成本）\n", "```\n" + main.to_string(index=False) + "\n```",
              "\n### 加上進場濾網（只看三種出場）\n",
              "季線＋年線：股價在 60MA 與 240MA 之上；所有均線：在 5/10/20/60/120/240MA 之上；"
              "大盤：加權指數／SPY 在季線之上。空方全部反過來。\n",
              "```\n" + filter_table(trades).to_string(index=False) + "\n```",
              "\n### 各年度期望值（看是否穩定，不是只靠某一年）\n", "```\n" + yearly.to_string() + "\n```",
              "\n### 出場原因占比\n", "```\n" + reasons.to_string() + "\n```",
              "\n## 怎麼解讀",
              "- **期望值%** = 平均每筆賺賠，這是最重要的數字；勝率低但期望值正，代表靠賺賠比取勝。",
              "- **賺賠比** = 平均賺 ÷ 平均賠；移動停損的目的就是把這個數字拉高。",
              "- 各年度都同方向才算穩定；只有某一年特別好，多半是行情而不是方法。",
              "- 這是每筆交易的統計，沒有模擬資金分配與同時持有多檔的情況。",
              "- 存活者偏差：只含目前仍上市的股票，整體結果偏樂觀，請看「各方式之間的相對差異」。"]
    return "\n".join(parts) + "\n"
