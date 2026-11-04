"""
美股多空雷達 ─ LINE 推播績效追蹤

目的：記錄「收盤戰報實際推播到 LINE」的多方突破／空方破底標的（推播時價格、首日或延續），
      之後用真實收盤價回填第 1／5／20 個交易日的漲跌幅與最大回檔，統計勝率。
      開盤半小時、午盤戰報只是核心股漲跌快報，沒有買賣訊號，不列入。
      和 us_signal_tracker.py（全股票池訊號、D+3/5/7）不同：這裡只看真的推給你的那幾檔。

儲存：Google 試算表（分頁「美股推播紀錄」「美股績效」「美股統計」），設定方式見 sheet_store.py。
      試算表未設定或寫入失敗時，推播紀錄改存 tracking/us_push_log.csv，下次回填會自動補進試算表。
      績效另存一份 tracking/us_push_perf.csv 給網頁版讀（Render 不必設金鑰）。

計算方式：
  - 基準價：推播卡片上的價格（收盤戰報＝當日收盤價），日期＝該價格的K棒日期。
  - D+N：基準日之後第 N 個交易日收盤 vs 基準價（未扣手續費），還原權息校正；空方取反向（跌＝賺）。
  - 最大回檔（D+N 內）：之後 N 個交易日中對部位最不利的價格（多方看最低價、空方看最高價），沒有不利記 0。
  - 抓不到的資料一律留空，畫面與試算表顯示「—」，不補數字。
"""
import os
import sys
import csv
import datetime
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sheet_store  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE, "tracking")
LOG_CSV = os.path.join(OUT_DIR, "us_push_log.csv")
PERF_CSV = os.path.join(OUT_DIR, "us_push_perf.csv")
SUMMARY_MD = os.path.join(OUT_DIR, "us_push_summary.md")
TAB_LOG, TAB_PERF, TAB_STATS = "美股推播紀錄", "美股績效", "美股統計"
ET = ZoneInfo("America/New_York")
HORIZONS = (1, 5, 20)
MAX_H = max(HORIZONS)
MIN_N = 30
UP, DOWN = "#1E7E5A", "#C0392B"   # 美股綠漲紅跌
COST_NOTE = "未扣手續費"
MODE_LABEL = {"LONG": "多方突破", "SHORT": "空方破底"}

LOG_FIELDS = [("date", "日期"), ("time", "推播時間(美東)"), ("slot", "時段"), ("mode", "方向"), ("symbol", "代號"),
              ("name", "名稱"), ("push_px", "推播價"), ("pct", "推播日漲跌%"), ("v_ratio", "量比"), ("neck", "頸線"),
              ("dist", "距頸線%"), ("first", "首日突破"), ("rank", "卡片順位")]
PERF_FIELDS = LOG_FIELDS + [("days_done", "已過交易日"), ("r1", "D+1%"), ("r5", "D+5%"), ("r20", "D+20%"),
                            ("mdd1", "D+1內最大回檔%"), ("mdd5", "D+5內最大回檔%"), ("mdd20", "D+20內最大回檔%"),
                            ("r_last", "最新%")]
STATS_FIELDS = [("group", "組別"), ("period", "期間"), ("n", "樣本"), ("win", "勝率%"), ("avg", "平均%"),
                ("median", "中位數%"), ("mdd_avg", "最大回檔平均%"), ("mdd_worst", "最大回檔最差%"), ("note", "備註")]
LOG_COLS = [k for k, _ in LOG_FIELDS]
PERF_COLS = [k for k, _ in PERF_FIELDS]
LOG_NUM = {"push_px", "pct", "v_ratio", "neck", "dist", "rank"}
PCT_KEYS = ("r1", "r5", "r20", "mdd1", "mdd5", "mdd20", "r_last")
PERF_NUM = LOG_NUM | set(PCT_KEYS) | {"days_done"}


def _r(v, nd=2):
    try:
        f = float(v)
        return "" if f != f else round(f, nd)
    except (TypeError, ValueError):
        return ""


def _read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _append_csv(path, cols, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new_file = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", encoding="utf-8-sig" if new_file else "utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        if new_file:
            w.writeheader()
        w.writerows(rows)


def _key(r):
    return (str(r["date"]), str(r["mode"]), str(r["symbol"]))


def record_push(picks: dict, names: dict, top_n: int):
    """收盤戰報推播成功後呼叫。picks={"LONG": [...], "SHORT": [...]}（run_screen 的結果）。失敗只印警告。"""
    try:
        now = datetime.datetime.now(ET)
        rows = []
        for mode, lst in picks.items():
            for i, r in enumerate(lst[:top_n], 1):
                rows.append({"date": r.get("date") or now.strftime("%Y-%m-%d"), "time": now.strftime("%H:%M"),
                             "slot": "收盤", "mode": mode, "symbol": r["symbol"],
                             "name": names.get(r["symbol"], "") or r["symbol"],
                             "push_px": round(float(r["price"]), 2), "pct": _r(r.get("pct")),
                             "v_ratio": _r(r.get("ratio")), "neck": _r(r.get("neck")), "dist": _r(r.get("dist")),
                             "first": bool(r.get("first")), "rank": i})
        if sheet_store.configured():
            try:
                seen = {_key(r) for r in sheet_store.read_rows(TAB_LOG, LOG_FIELDS)}
                rows_s = [r for r in rows if _key(r) not in seen]
                sheet_store.append_rows(TAB_LOG, LOG_FIELDS, rows_s, num_keys=LOG_NUM, pct_keys=("pct",), up=UP, down=DOWN)
                print(f"📝 推播紀錄（試算表）：新增 {len(rows_s)} 檔")
                return
            except Exception as e:
                print(f"⚠️ 試算表寫入失敗，改存 CSV：{sheet_store.describe_error(e)}")
        seen = {_key(r) for r in _read_csv(LOG_CSV)}
        rows_c = [r for r in rows if _key(r) not in seen]
        _append_csv(LOG_CSV, LOG_COLS, rows_c)
        print(f"📝 推播紀錄（CSV）：新增 {len(rows_c)} 檔")
    except Exception as e:
        print(f"⚠️ 推播紀錄寫入失敗（不影響推播）：{e}")


# ------------------------------------------------------------------------------
# 回填（收盤後執行）
# ------------------------------------------------------------------------------
def prep_daily(sub):
    import numpy as np
    sub = sub.dropna(subset=["Close"])
    sub = sub[sub["Volume"].fillna(0) > 0].copy()
    if sub.empty:
        return None
    if getattr(sub.index, "tz", None) is not None:
        sub.index = sub.index.tz_localize(None)
    if "Adj Close" not in sub.columns:
        sub["Adj Close"] = sub["Close"]
    sub["f"] = (sub["Adj Close"] / sub["Close"]).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    return sub


def download(symbols, period="1y"):
    import pandas as pd
    import yfinance as yf
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
            if sub is not None and "Close" in sub.columns:
                d = prep_daily(sub)
                if d is not None:
                    data[s] = d
    return data


def compute_outcome(row: dict, df, today: str) -> dict:
    import pandas as pd
    out = {k: "" for k in ["r_last"] + [f"r{h}" for h in HORIZONS] + [f"mdd{h}" for h in HORIZONS]}
    out["days_done"] = 0
    try:
        px = float(row["push_px"])
    except (TypeError, ValueError):
        return out
    if df is None or not px > 0:
        return out
    d0 = pd.Timestamp(str(row["date"]))
    sgn = 1.0 if row.get("mode", "LONG") == "LONG" else -1.0
    day0 = df[df.index.normalize() == d0]
    if day0.empty:
        return out
    base = px * float(day0["f"].iloc[0])
    after = df[(df.index.normalize() > d0) & (df.index.normalize() <= pd.Timestamp(today))].iloc[:MAX_H]
    out["days_done"] = len(after)
    closes = after["Close"] * after["f"]
    adverse = sgn * (((after["Low"] if sgn > 0 else after["High"]) * after["f"]) / base - 1)
    for h in HORIZONS:
        if len(after) < h:
            continue
        out[f"r{h}"] = round(sgn * (float(closes.iloc[h - 1]) / base - 1) * 100, 2)
        out[f"mdd{h}"] = round(min(float(adverse.iloc[:h].min()), 0.0) * 100, 2)
    if len(after):
        out["r_last"] = round(sgn * (float(closes.iloc[-1]) / base - 1) * 100, 2)
    return out


def horizon_stats(g):
    import pandas as pd
    res = []
    for h in HORIZONS:
        a = pd.to_numeric(g[f"r{h}"], errors="coerce").dropna()
        m = pd.to_numeric(g[f"mdd{h}"], errors="coerce").dropna()
        n = len(a)
        res.append({"key": f"r{h}", "h": h, "n": n,
                    "win": round(float((a > 0).mean() * 100), 0) if n else None,
                    "avg": round(float(a.mean()), 2) if n else None,
                    "median": round(float(a.median()), 2) if n else None,
                    "mdd_avg": round(float(m.mean()), 2) if len(m) else None,
                    "mdd_worst": round(float(m.min()), 2) if len(m) else None})
    return res


def read_perf(path_or_buf=PERF_CSV):
    import pandas as pd
    df = pd.read_csv(path_or_buf, dtype={"symbol": str}, encoding="utf-8-sig")
    df["first"] = df["first"].map({True: True, False: False, "True": True, "False": False})
    return df


def groups(perf, mode):
    g = perf[perf["mode"] == mode]
    lab = MODE_LABEL[mode]
    return [(f"{lab}（全部）", g), (f"{lab}｜首日", g[g["first"] == True]),   # noqa: E712
            (f"{lab}｜延續", g[g["first"] == False])]                          # noqa: E712


def stats_rows(perf):
    rows = []
    for mode in ("LONG", "SHORT"):
        for title, g in groups(perf, mode):
            for s in horizon_stats(g):
                rows.append({"group": title, "period": f"D+{s['h']}", "n": s["n"], "win": s["win"], "avg": s["avg"],
                             "median": s["median"], "mdd_avg": s["mdd_avg"], "mdd_worst": s["mdd_worst"],
                             "note": "" if s["n"] >= MIN_N or s["n"] == 0 else "樣本不足"})
    return rows


def _p(v):
    return "—" if v is None else f"{v:+.2f}%"


def build_summary(perf):
    now = datetime.datetime.now(ET).strftime("%Y-%m-%d %H:%M")
    lines = ["# 美股 LINE 推播績效（收盤戰報）\n",
             f"- 更新：{now}（美東時間）",
             f"- 起始：{perf['date'].min() if len(perf) else '—'}｜推播 {len(perf)} 筆｜推播天數 {perf['date'].nunique() if len(perf) else 0}",
             f"- 基準價＝推播價（收盤）；D+N＝之後第 N 個交易日收盤；{COST_NOTE}；還原權息校正；空方取反向。",
             "- 最大回檔＝D+N 內對部位最不利的價格相對推播價（平均／最差）。",
             f"- 樣本少於 {MIN_N} 筆標示「樣本不足」。同一天的推播受同一個大盤影響，彼此不獨立。"]
    last = None
    for r in stats_rows(perf):
        if r["group"] != last:
            last = r["group"]
            lines += [f"\n## {last}\n", "| 期間 | 樣本 | 勝率 | 平均 | 中位數 | 最大回檔 平均 | 最大回檔 最差 |",
                      "|---|---|---|---|---|---|---|"]
        win = "—" if r["win"] is None else f"{r['win']:.0f}%"
        tag = f" {r['note']}" if r["note"] else ""
        lines.append(f"| {r['period']} | {r['n']}{tag} | {win} | {_p(r['avg'])} | {_p(r['median'])} | "
                     f"{_p(r['mdd_avg'])} | {_p(r['mdd_worst'])} |")
    return "\n".join(lines) + "\n"


def load_log():
    csv_rows = _read_csv(LOG_CSV)
    rows = list(csv_rows)
    if sheet_store.configured():
        try:
            sheet_rows = sheet_store.read_rows(TAB_LOG, LOG_FIELDS)
            have = {_key(r) for r in sheet_rows}
            missing = [r for r in csv_rows if _key(r) not in have]
            if missing:
                sheet_store.append_rows(TAB_LOG, LOG_FIELDS, missing, num_keys=LOG_NUM, pct_keys=("pct",), up=UP, down=DOWN)
                print(f"📤 CSV 暫存紀錄補進試算表 {len(missing)} 筆")
            rows = sheet_rows + missing
        except Exception as e:
            print(f"⚠️ 試算表讀取失敗，只用 CSV 紀錄：{sheet_store.describe_error(e)}")
    return rows


def main():
    import pandas as pd
    today = datetime.datetime.now(ET).strftime("%Y-%m-%d")
    raw = load_log()
    if not raw:
        print("尚無推播紀錄")
        return
    log = pd.DataFrame(raw, columns=LOG_COLS).astype(str).drop_duplicates(["date", "mode", "symbol"])
    done = {}
    if os.path.exists(PERF_CSV):   # 已滿 20 個交易日的紀錄凍結，不再重抓
        for _, r in read_perf().astype(object).iterrows():
            if pd.to_numeric(r["days_done"], errors="coerce") >= MAX_H:
                done[_key(r)] = r.to_dict()
    todo = [r for _, r in log.iterrows() if _key(r) not in done]
    hist = download(sorted({r["symbol"] for r in todo})) if todo else {}
    rows = []
    for _, r in log.iterrows():
        k = _key(r)
        if k in done:
            rows.append(done[k])
            continue
        row = {c: r[c] for c in LOG_COLS}
        row["first"] = str(r["first"]) == "True"
        row.update(compute_outcome(row, hist.get(r["symbol"]), today))
        rows.append(row)
    perf = pd.DataFrame(rows, columns=PERF_COLS)
    perf["_r"] = pd.to_numeric(perf["rank"], errors="coerce")
    perf = perf.sort_values(["date", "mode", "_r"], ascending=[False, True, True]).drop(columns="_r")
    os.makedirs(OUT_DIR, exist_ok=True)
    perf.to_csv(PERF_CSV, index=False, encoding="utf-8-sig")
    perf = read_perf()
    with open(SUMMARY_MD, "w", encoding="utf-8") as f:
        f.write(build_summary(perf))
    if sheet_store.configured():
        try:
            recs = perf.astype(object).where(perf.notna(), None).to_dict("records")
            sheet_store.replace_rows(TAB_PERF, PERF_FIELDS, recs, num_keys=PERF_NUM, pct_keys=PCT_KEYS, up=UP, down=DOWN)
            sheet_store.replace_rows(TAB_STATS, STATS_FIELDS, stats_rows(perf),
                                     num_keys={"n", "win", "avg", "median", "mdd_avg", "mdd_worst"},
                                     pct_keys=("avg", "median", "mdd_avg", "mdd_worst"), up=UP, down=DOWN)
            print("📤 試算表已更新")
        except Exception as e:
            print(f"⚠️ 試算表寫入失敗（repo 內 CSV 已更新）：{sheet_store.describe_error(e)}")
    else:
        print("ℹ️ 未設定 GOOGLE_SERVICE_ACCOUNT_JSON／PUSH_SHEET_ID，只更新 repo 內 CSV")
    live = int((pd.to_numeric(perf["days_done"], errors="coerce").fillna(0) < MAX_H).sum())
    print(f"完成：推播 {len(perf)} 筆，追蹤中 {live} 筆")


if __name__ == "__main__":
    main()
