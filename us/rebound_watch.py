"""
大盤「RSI 底背離落底確認」觀察與前瞻績效記錄（只觀察，不影響篩選邏輯與推播）

訊號定義與 rebound_backtest.py 完全相同（事先固定、不看結果調整）：
  RSI 底背離：近 10 日最低收盤 < 前一段（11~40 日前）最低收盤，但 RSI(14) 低點較高；訊號日收紅確認。
  大跌狀態：收盤相對近 120 日最高收盤回檔 >= 8% / 12%，或近 5 日累計跌幅 >= 5%。
  進場：訊號日隔日開盤；出場：進場後第 5/10/20 個交易日收盤；未扣手續費與稅。
  去叢聚：同一指數 10 個交易日內不重複記錄。

前瞻記錄：只記錄「追蹤起始日之後」才出現的訊號（不回補歷史，歷史請看回測報告）。
  tracking/rebound_signals.csv   每筆訊號與後續 5/10/20 日真實報酬、最大回撤（尚未走完的欄位留空）
  tracking/rebound_status.json   各指數最新狀態，給戰情室顯示
  tracking/rebound_summary.md    前瞻績效彙總

資料：yfinance 真實日線；抓不到就保留舊紀錄、狀態標示「—」，不填任何佔位數字。
用法：python us/rebound_watch.py us
"""
import os
import sys
import json
import datetime
import numpy as np
import pandas as pd

MARKET = os.environ.get("REBOUND_MARKET", "us")
CONFIG = {
    "tw": [("加權指數", "^TWII")],
    "us": [("S&P500", "^GSPC"), ("Nasdaq", "^IXIC")],
}
HORIZONS = (5, 10, 20)
DD_LOOKBACK, RECENT, COOLDOWN = 120, 10, 10
SHOCK_N, SHOCK_PCT = 5, 0.05
HERE = os.path.dirname(os.path.abspath(__file__))
TRACK_DIR = os.path.join(HERE, "tracking")
if os.path.basename(HERE) == "us":           # pro-terminal：腳本放在 us/，資料放 repo 根目錄 tracking/
    TRACK_DIR = os.path.join(os.path.dirname(HERE), "tracking")
SIG_CSV = os.path.join(TRACK_DIR, "rebound_signals.csv")
STATUS_JSON = os.path.join(TRACK_DIR, "rebound_status.json")
SUMMARY_MD = os.path.join(TRACK_DIR, "rebound_summary.md")
COLS = ["market", "symbol", "name", "date", "entry_date", "entry_open", "close", "dd120", "dd8", "dd12", "shock5",
        "r5", "r10", "r20", "mdd5", "mdd10", "mdd20"]


def fetch(sym, start="2015-01-01"):
    import yfinance as yf
    try:
        df = yf.download(sym, start=start, auto_adjust=False, progress=False)
        out = {}
        for c in ("Open", "High", "Low", "Close"):
            s = df[c]
            out[c] = s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s
        d = pd.DataFrame(out).dropna(subset=["Close", "Open"])
        d.index = pd.to_datetime(d.index).tz_localize(None)
        return d if len(d) > 200 else None
    except Exception as e:  # noqa
        print("fetch fail", sym, e)
        return None


def rsi(c, n=14):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def compute(d):
    """回傳含 dd120、dd8/dd12/shock5（近 10 日內曾大跌）、rsi_div 的 DataFrame。"""
    c, o = d["Close"], d["Open"]
    out = pd.DataFrame(index=d.index)
    dd = c / c.rolling(DD_LOOKBACK, min_periods=60).max() - 1
    out["dd120"] = dd
    shock = (c / c.shift(SHOCK_N) - 1)
    out["shock5"] = shock
    rec = lambda s: s.astype(float).rolling(RECENT, min_periods=1).max().astype(bool)
    out["dd8"] = rec(dd <= -0.08)
    out["dd12"] = rec(dd <= -0.12)
    out["shock_recent"] = rec(shock <= -SHOCK_PCT)
    up = (c > c.shift(1)) & (c > o)
    r = rsi(c)
    low_recent, low_prev = c.rolling(10).min(), c.shift(10).rolling(30).min()
    rsi_recent, rsi_prev = r.rolling(10).min(), r.shift(10).rolling(30).min()
    out["rsi_div"] = ((low_recent < low_prev) & (rsi_recent > rsi_prev) & up).fillna(False)
    out["rsi"] = r
    return out


def load_signals():
    if os.path.exists(SIG_CSV):
        try:
            return pd.read_csv(SIG_CSV, dtype={"symbol": str}, encoding="utf-8-sig")
        except Exception:
            pass
    return pd.DataFrame(columns=COLS)


def load_status():
    try:
        with open(STATUS_JSON, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def fwd(d, date, H):
    """訊號日 date → (報酬, 最大回撤)；資料未走完回 (None, None)。"""
    idx = d.index.get_loc(pd.Timestamp(date))
    e, x = idx + 1, idx + H
    if x >= len(d):
        return None, None
    entry = float(d["Open"].iloc[e])
    if not entry or np.isnan(entry):
        return None, None
    path = d["Close"].iloc[e:x + 1].values / entry - 1
    return float(path[-1]), float(path.min())


def main():
    os.makedirs(TRACK_DIR, exist_ok=True)
    sigs = load_signals()
    status = load_status()
    now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    for name, sym in CONFIG[MARKET]:
        d = fetch(sym)
        st = status.get(sym, {})
        if d is None:
            st.update({"name": name, "symbol": sym, "ok": False, "updated": now})
            status[sym] = st
            continue
        f = compute(d)
        last = f.iloc[-1]
        last_date = str(d.index[-1].date())
        # 追蹤起始：第一次執行時的最新一根 K 棒之後才算前瞻
        start = st.get("track_start") or last_date
        scanned = st.get("scanned_through") or last_date
        new_days = [i for i in d.index if str(i.date()) > scanned]
        for day in new_days:
            row = f.loc[day]
            if not bool(row["rsi_div"]):
                continue
            ex = sigs[sigs["symbol"] == sym]
            prev = [pd.Timestamp(x) for x in ex["date"]]
            pos = d.index.get_loc(day)
            if any(0 < pos - d.index.get_loc(p) <= COOLDOWN for p in prev if p in d.index):
                continue  # 去叢聚
            sigs.loc[len(sigs)] = {
                "market": MARKET, "symbol": sym, "name": name, "date": str(day.date()),
                "entry_date": "", "entry_open": "", "close": round(float(d.loc[day, "Close"]), 2),
                "dd120": round(float(row["dd120"]), 4) if pd.notna(row["dd120"]) else "",
                "dd8": bool(row["dd8"]), "dd12": bool(row["dd12"]), "shock5": bool(row["shock_recent"]),
                **{k: "" for k in ("r5", "r10", "r20", "mdd5", "mdd10", "mdd20")}}
        # 回填報酬
        for i, r in sigs[sigs["symbol"] == sym].iterrows():
            day = pd.Timestamp(r["date"])
            if day not in d.index:
                continue
            pos = d.index.get_loc(day)
            if pos + 1 < len(d):
                sigs.at[i, "entry_date"] = str(d.index[pos + 1].date())
                sigs.at[i, "entry_open"] = round(float(d["Open"].iloc[pos + 1]), 2)
            for H in HORIZONS:
                if str(r[f"r{H}"]) in ("", "nan"):
                    ret, mdd = fwd(d, day, H)
                    if ret is not None:
                        sigs.at[i, f"r{H}"] = round(ret * 100, 3)
                        sigs.at[i, f"mdd{H}"] = round(mdd * 100, 3)
        sdates = sigs[sigs["symbol"] == sym]["date"]
        st.update({
            "name": name, "symbol": sym, "ok": True, "updated": now, "asof": last_date,
            "close": round(float(d["Close"].iloc[-1]), 2),
            "dd120": round(float(last["dd120"]) * 100, 2) if pd.notna(last["dd120"]) else None,
            "rsi": round(float(last["rsi"]), 1) if pd.notna(last["rsi"]) else None,
            "crash8": bool(last["dd8"]), "crash12": bool(last["dd12"]), "shock5": bool(last["shock_recent"]),
            "div_today": bool(last["rsi_div"]),
            "last_signal": str(sdates.max()) if len(sdates) else None,
            "track_start": start, "scanned_through": last_date,
        })
        status[sym] = st
    sigs.to_csv(SIG_CSV, index=False, encoding="utf-8-sig")
    with open(STATUS_JSON, "w", encoding="utf-8") as f:
        json.dump(status, f, ensure_ascii=False, indent=1)
    write_summary(sigs, status)


def write_summary(sigs, status):
    lines = ["# RSI 底背離落底確認：前瞻績效（只記錄追蹤起始日之後的真實訊號）", ""]
    for sym, st in status.items():
        lines.append(f"- {st.get('name', sym)}（{sym}）追蹤起始 {st.get('track_start', '—')}，資料至 {st.get('asof', '—')}")
    lines.append("")
    for H in HORIZONS:
        a = pd.to_numeric(sigs.get(f"r{H}"), errors="coerce").dropna() if len(sigs) else pd.Series(dtype=float)
        n = len(a)
        if n:
            lines.append(f"- 持有 {H} 日：已完成 {n} 筆，勝率 {(a > 0).mean() * 100:.0f}%，平均 {a.mean():+.2f}%，中位數 {a.median():+.2f}%")
        else:
            lines.append(f"- 持有 {H} 日：已完成 0 筆（—）")
    lines += ["", "⚠️ 事件稀少（回測約每年 2～3 次），至少累積 30 筆才有統計意義；在此之前僅供觀察。"]
    with open(SUMMARY_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        MARKET = sys.argv[1]
    main()
