"""
抄底模式｜嚴格回測（只回測，不改雷達／推播／篩選）
規則：個股 5 日跌幅 ≥ 門檻 且 收盤 < 60 日均線（季線），大型股（S&P 500 成分）。
改善：時點成分股（含已移出但仍可抓到資料者）、空頭年（2008/2020/2022…）、同日訊號合併、分年度、組合層級最大回撤。
數據：yfinance 還原權息日線；抓不到一律「—」。輸出 backtest_reports/dip_strict_backtest.md
"""
import os, io, sys, time, datetime as dt
import numpy as np, pandas as pd, requests

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODE = os.environ.get("MODE", "below60")      # below60＝收盤<季線；above200＝收盤>200日線（上升趨勢中的拉回）
OUT = os.path.join(BASE, "backtest_reports", "dip_strict_backtest.md" if MODE == "below60" else "dip_trend_pullback_backtest.md")
SYNTH = os.environ.get("SYNTH") == "1"
START = "2003-06-01"
FIRST_SIG = pd.Timestamp("2004-09-01")
COST = float(os.environ.get("COST_PCT", "0.1")) / 100.0
THRS = [0.08, 0.10, 0.12, 0.15]
HMAX, COOL, NW = 20, 10, 24            # 最長持有、同股冷卻天數、窗口欄數
EQ0, RISK_PCT, POS_CAP = 91000.0, 0.01, 0.20
ETFS = ["SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB", "SOXX", "SMH", "XBI", "KRE", "IYR", "EFA", "EEM"]
WIKI = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
LOG = []


def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.append(s)


def norm(t):
    return str(t).strip().upper().replace(".", "-")


# ----------------------------------------------------------------------------- 成分股（時點）
def load_membership():
    """回傳 (members: {ticker:[(start,end),...]}, current:set, meta:str)"""
    far = pd.Timestamp("2100-01-01"); st0 = pd.Timestamp(START)
    try:
        r = requests.get(WIKI, headers={"User-Agent": "Mozilla/5.0 (research backtest)"}, timeout=40)
        r.raise_for_status()
        tabs = pd.read_html(io.StringIO(r.text))
        log("Wikipedia 表格數", len(tabs), [t.shape for t in tabs[:4]], "頁面長度", len(r.text))
        flat = lambda t: [" ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in t.columns]
        cur = next(t for t in tabs if any("Symbol" in c for c in flat(t)) and any("Date added" in c for c in flat(t)))
        for q, t in enumerate(tabs[:6]):
            log("  表", q, t.shape, flat(t)[:8])
        chg = None
        try:
            chg = next(t for t in tabs if any("Removed" in c for c in flat(t)) and any("Added" in c for c in flat(t)))
        except StopIteration:
            hdr = {"User-Agent": "Mozilla/5.0 (research backtest)"}
            titles = ["List_of_S&P_500_component_changes", "Historical_components_of_the_S&P_500", "List_of_S&P_500_companies/Changes"]
            try:
                js = requests.get("https://en.wikipedia.org/w/api.php", params={"action": "query", "list": "search", "srsearch": "S&P 500 index components changes added removed", "format": "json", "srlimit": 8}, headers=hdr, timeout=30).json()
                titles += [x["title"].replace(" ", "_") for x in js["query"]["search"]]
            except Exception as e:
                log("  搜尋失敗", repr(e)[:100])
            log("  異動表候選頁：", titles)
            for ti in titles:
                try:
                    rr = requests.get("https://en.wikipedia.org/wiki/" + ti, headers=hdr, timeout=40)
                    if rr.status_code != 200:
                        continue
                    for t in pd.read_html(io.StringIO(rr.text)):
                        fc = flat(t)
                        if any("Removed" in c for c in fc) and any("Added" in c for c in fc) and len(t) > 50:
                            chg = t; log("  找到異動表：", ti, t.shape, fc[:6]); break
                except Exception as e:
                    continue
                if chg is not None:
                    break
            if chg is None:
                raise RuntimeError("找不到成分股異動表")
        sym = [c for c in cur.columns if "Symbol" in str(c)][0]
        dcol = [c for c in cur.columns if "Date added" in str(c)][0]
        chg.columns = [" ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in chg.columns]
        cd = [c for c in chg.columns if "Date" in c][0]
        ca = [c for c in chg.columns if "Added" in c and "Ticker" in c][0]
        cr = [c for c in chg.columns if "Removed" in c and "Ticker" in c][0]
        chg["_d"] = pd.to_datetime(chg[cd], errors="coerce")
        chg = chg.dropna(subset=["_d"])
        members, current = {}, set()
        for _, row in cur.iterrows():
            t = norm(row[sym]); current.add(t)
            a = pd.to_datetime(row[dcol], errors="coerce")
            s = st0 if pd.isna(a) or a < st0 else a
            members.setdefault(t, []).append((s, far))
        added = {}
        for _, row in chg.iterrows():
            if isinstance(row[ca], str) and row[ca].strip():
                added.setdefault(norm(row[ca]), []).append(row["_d"])
        nrem = 0
        for _, row in chg.iterrows():
            if isinstance(row[cr], str) and row[cr].strip() and row["_d"] > st0:
                t = norm(row[cr]); d = row["_d"]
                prev = [x for x in added.get(t, []) if x < d]
                s = max(prev) if prev else st0
                members.setdefault(t, []).append((max(s, st0), d)); nrem += 1
        return members, current, f"Wikipedia 成分股表：目前 {len(current)} 檔、{START[:4]} 年後移出事件 {nrem} 筆"
    except Exception as e:
        import traceback; log("Wikipedia 失敗：", repr(e)); traceback.print_exc()
        import json
        p = os.path.join(BASE, "us_universe.json")
        u = [norm(x[0]) for x in json.load(open(p, encoding="utf-8"))] if os.path.exists(p) else []
        return {t: [(st0, far)] for t in u}, set(u), "Wikipedia 抓取失敗，退回 us_universe.json（無時點成分，PIT 無效）"


def in_member(members, t, d):
    for s, e in members.get(t, []):
        if s <= d < e:
            return True
    return False


# ----------------------------------------------------------------------------- 下載
def fetch(tickers):
    import yfinance as yf
    out = {}
    for i in range(0, len(tickers), 40):
        batch = tickers[i:i + 40]; raw = None
        for _ in range(3):
            try:
                raw = yf.download(batch, start=START, auto_adjust=True, group_by="ticker", threads=True, progress=False)
                break
            except Exception as e:
                log("下載重試", repr(e)[:80]); time.sleep(5)
        if raw is None or len(raw) == 0:
            continue
        multi = isinstance(raw.columns, pd.MultiIndex)
        for t in batch:
            try:
                sub = raw[t] if multi else raw
            except KeyError:
                continue
            sub = sub[["Open", "High", "Low", "Close"]].dropna()
            if getattr(sub.index, "tz", None) is not None:
                sub.index = sub.index.tz_localize(None)
            if len(sub) >= 260:
                out[t] = sub.astype(float)
        log(f"下載 {min(i + 40, len(tickers))}/{len(tickers)}，累計成功 {len(out)}")
    return out


def synth_data(n=60):
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2003-06-02", "2026-11-25")
    out = {}
    mk = rng.normal(0.0003, 0.011, len(idx))
    for k in range(n):
        r = 0.7 * mk + rng.normal(0, 0.016, len(idx))
        c = 50 * np.exp(np.cumsum(r)); o = c * (1 + rng.normal(0, 0.004, len(idx)))
        h = np.maximum(o, c) * (1 + abs(rng.normal(0, 0.006, len(idx)))); l = np.minimum(o, c) * (1 - abs(rng.normal(0, 0.006, len(idx))))
        d = pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c}, index=idx)
        if k % 7 == 3:
            d = d.iloc[: 2500]
        out[f"S{k}"] = d
    spy = pd.DataFrame({"Open": 100 * np.exp(np.cumsum(mk)), "High": 100 * np.exp(np.cumsum(mk)), "Low": 100 * np.exp(np.cumsum(mk)), "Close": 100 * np.exp(np.cumsum(mk))}, index=idx)
    out["SPY"] = spy; out["^VIX"] = pd.DataFrame({c: 15 + 10 * np.abs(np.sin(np.arange(len(idx)) / 200)) for c in ["Open", "High", "Low", "Close"]}, index=idx)
    out["QQQ"] = spy.copy()
    return out


# ----------------------------------------------------------------------------- 訊號與窗口
def build_signals(data, kinds, last_day):
    cols = {k: [] for k in ["tk", "kind", "i", "date", "ret5", "atr", "O", "H", "L", "C", "D"]}
    for t, df in data.items():
        if t not in kinds:
            continue
        o, h, l, c = (df[x].values for x in ["Open", "High", "Low", "Close"])
        cs = df["Close"]; n = len(df)
        ret5 = (cs / cs.shift(5) - 1).values
        sma = cs.rolling(60).mean().values
        sma200 = cs.rolling(200).mean().values
        pc = cs.shift(1)
        tr = pd.concat([df.High - df.Low, (df.High - pc).abs(), (df.Low - pc).abs()], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean().values
        ok = (ret5 <= -0.08) & ((c < sma) if MODE == "below60" else (c > sma200)) & ~np.isnan(atr) & (atr > 0) & (atr / c < 0.25) & (df.index >= FIRST_SIG)
        ii = np.flatnonzero(ok)
        if len(ii) == 0:
            continue
        ended = df.index[-1] < last_day - pd.Timedelta(days=10)     # 已下市／資料中止：窗口用最後收盤補齊
        pad = NW
        fill = lambda a: np.concatenate([a, np.full(pad, a[-1] if ended else np.nan)])
        O, H, L, C = fill(o), fill(h), fill(l), fill(c)
        D = np.concatenate([df.index.values.astype("datetime64[ns]"), np.full(pad, np.datetime64("NaT", "ns"))])
        off = np.arange(NW)
        win = ii[:, None] + off[None, :]
        cols["tk"] += [t] * len(ii); cols["kind"] += [kinds[t]] * len(ii)
        cols["i"] += list(ii); cols["date"] += list(df.index.values[ii]); cols["ret5"] += list(ret5[ii]); cols["atr"] += list(atr[ii])
        for k, A in zip("OHLCD", [O, H, L, C, D]):
            cols[k].append(A[win])
    S = pd.DataFrame({k: cols[k] for k in ["tk", "kind", "i", "date", "ret5", "atr"]})
    W = {k: np.vstack(cols[k]) for k in "OHLCD"}
    S["date"] = pd.to_datetime(S["date"])
    return S, W


def entry(W, mode):
    N = W["C"].shape[0]; Ow, Hw, Lw, Cw = W["O"], W["H"], W["L"], W["C"]
    if mode == "close":
        return np.ones(N, bool), np.zeros(N, int), Cw[:, 0].copy()
    if mode == "open":
        P = Ow[:, 1].copy(); return ~np.isnan(P), np.zeros(N, int), P
    # 止跌確認（最多等 3 天；期間跌破訊號日最低點 → 取消）
    done = np.zeros(N, bool); dead = np.zeros(N, bool); ej = np.zeros(N, int); P = np.full(N, np.nan)
    for k in (1, 2, 3):
        brk = Lw[:, k] < Lw[:, 0]
        if mode == "confA":      # 收紅（收>開且收>前收）且未破訊號日低點
            cond = (Cw[:, k] > Ow[:, k]) & (Cw[:, k] > Cw[:, k - 1])
        else:                    # confB：收盤站上前一日最高
            cond = Cw[:, k] > Hw[:, k - 1]
            brk = np.zeros(N, bool)
        dead |= (~done) & brk
        hit = (~done) & (~dead) & cond & ~np.isnan(Cw[:, k])
        ej[hit] = k; P[hit] = Cw[hit, k]; done |= hit
    return done, ej, P


def sim(W, S, emode, stop, H):
    valid, ej, P = entry(W, emode)
    N = len(P); ar = np.arange(N)
    idx = (ej + 1)[:, None] + np.arange(HMAX)[None, :]
    take = lambda A: np.take_along_axis(A, np.minimum(idx, NW - 1), axis=1)
    Bo, Bl, Bc, Bd = take(W["O"]), take(W["L"]), take(W["C"]), take(W["D"])
    valid = valid & ~np.isnan(Bc[:, H - 1]) & ~np.isnan(P)
    atr = S["atr"].values
    kind = stop[0]
    if kind == "none":
        dist = 3 * atr / P; exitp = Bc[:, H - 1].copy(); off = np.full(N, H - 1); stopped = np.zeros(N, bool)
    else:
        sl = P - stop[1] * atr if kind.startswith("atr") else P * (1 - stop[1])
        dist = (P - sl) / P
        if kind.endswith("intra"):
            hit = (Bl[:, :H] <= sl[:, None]) | (Bo[:, :H] <= sl[:, None])
            anyh = hit.any(1); k = hit.argmax(1); ko = Bo[ar, k]
            exitp = np.where(anyh, np.where(ko <= sl, ko, sl), Bc[:, H - 1]); off = np.where(anyh, k, H - 1)
        else:                                     # 收盤確認 → 隔日開盤出場
            br = Bc[:, :H] <= sl[:, None]; anyh = br.any(1); k = br.argmax(1)
            nxt = np.where(k < H - 1, Bo[ar, np.minimum(k + 1, HMAX - 1)], Bc[ar, k])
            exitp = np.where(anyh, nxt, Bc[:, H - 1]); off = np.where(anyh, np.minimum(k + 1, H - 1), H - 1)
        stopped = anyh
    ret = exitp / P - 1 - COST
    R = ret / dist
    ent_d = np.take_along_axis(W["D"], ej[:, None], axis=1)[:, 0]
    ex_d = Bd[ar, off]
    return dict(valid=valid, ret=ret, R=R, stopped=stopped, off=off, dist=dist, P=P, exitp=exitp, ej=ej, ent_d=ent_d, ex_d=ex_d, Bc=Bc, Bd=Bd)


# ----------------------------------------------------------------------------- 篩選與統計
def cooled(S, mask):
    """同股 10 個交易日冷卻：回傳 mask 內被保留的訊號"""
    keep = np.zeros(len(S), bool)
    sub = S[mask].sort_values(["tk", "i"])
    last_tk, last_i = None, -10 ** 9
    for ix, tk, i in zip(sub.index, sub["tk"], sub["i"]):
        if tk != last_tk:
            last_tk, last_i = tk, -10 ** 9
        if i - last_i >= COOL:
            keep[ix] = True; last_i = i
    return keep


def stats(mask, s, S, spy):
    m = mask & s["valid"]
    n = int(m.sum())
    if n < 5:
        return None
    r = s["ret"][m]; R = s["R"][m]; d = S["date"].values[m]
    dd = pd.to_datetime(d); yrs = dd.year.values
    ud = np.unique(d); gaps = np.diff(ud).astype("timedelta64[D]").astype(int)
    epi = np.concatenate([[0], np.cumsum(gaps > 10)])
    eid = epi[np.searchsorted(ud, d)]
    mean = r.mean()
    cl = pd.Series(r - mean).groupby(eid).sum()
    se = np.sqrt((cl ** 2).sum()) / n
    ep_mean = pd.Series(r).groupby(eid).mean()
    sp = spy_ret(spy, s["ent_d"][m], s["ex_d"][m])
    w = r[r > 0]; l = r[r <= 0]
    early = yrs <= 2014
    return dict(n=n, days=len(ud), eps=int(epi[-1] + 1), win=(r > 0).mean(), mean=mean, med=np.median(r), aw=w.mean() if len(w) else np.nan,
                al=l.mean() if len(l) else np.nan, ER=R.mean(), stop=s["stopped"][m].mean(), worst=r.min(), t=mean / se if se > 0 else np.nan,
                exc=np.nanmean(r - sp) if sp is not None else np.nan, epm=ep_mean.mean(), epw=(ep_mean > 0).mean(),
                e1=r[early].mean() if early.sum() >= 20 else np.nan, n1=int(early.sum()),
                e2=r[~early].mean() if (~early).sum() >= 20 else np.nan, n2=int((~early).sum()))


def spy_ret(spy, d0, d1):
    if spy is None:
        return None
    ix = spy.index.values
    a = spy["Close"].values[np.clip(np.searchsorted(ix, d0, side="right") - 1, 0, len(ix) - 1)]
    b = spy["Close"].values[np.clip(np.searchsorted(ix, d1, side="right") - 1, 0, len(ix) - 1)]
    return b / a - 1


def pc(x, d=1, sign=False):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else (f"{x * 100:+.{d}f}%" if sign else f"{x * 100:.{d}f}%")


def f2(x, d=2):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.{d}f}"


def row(label, st):
    if st is None:
        return f"| {label} | <5 | — | — | — | — | — | — | — | — | — | — | — |"
    flag = "" if st["eps"] >= 30 else "⚠"
    return (f"| {label} | {st['n']} | {st['days']} / {st['eps']}{flag} | {pc(st['win'], 0)} | {pc(st['mean'], 2, True)} | {pc(st['med'], 2, True)} | "
            f"{f2(st['ER'])} | {pc(st['exc'], 2, True)} | {f2(st['t'], 1)} | {pc(st['epm'], 2, True)}（{pc(st['epw'], 0)}） | {pc(st['stop'], 0)} | "
            f"{pc(st['e1'], 2, True)}（{st['n1']}）/ {pc(st['e2'], 2, True)}（{st['n2']}） | {pc(st['worst'], 0)} |")


HEAD = ["| 組別 | 筆數 | 訊號日數 / 獨立事件 | 勝率 | 平均 | 中位數 | 期望值(R) | 超額於SPY | t值(群聚) | 事件平均（事件勝率） | 停損率 | 2014前 / 2015後 平均（筆） | 最差 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]


# ----------------------------------------------------------------------------- 組合模擬
def portfolio(S, s, mask, spy, seeds=15):
    m = np.flatnonzero(mask & s["valid"])
    if len(m) == 0:
        return None
    cal = spy.index.values; T = len(cal)
    pos = lambda d: np.clip(np.searchsorted(cal, d, side="left"), 0, T - 1)
    dist = s["dist"][m]; V = np.minimum(RISK_PCT * EQ0 / dist, POS_CAP * EQ0)
    ent = pos(s["ent_d"][m]); ext = pos(s["ex_d"][m])
    paths = []
    for q, ix in enumerate(m):
        off = s["off"][ix]; ratio = s["Bc"][ix, :off + 1] / s["P"][ix] - 1
        ratio[-1] = s["exitp"][ix] / s["P"][ix] - 1 - COST
        paths.append((pos(s["Bd"][ix, :off + 1]), ratio))
    res = []
    for seed in range(seeds):
        rng = np.random.default_rng(seed); order = np.lexsort((rng.random(len(m)), ent))
        expo = np.zeros(T); delta = np.zeros(T); acc = 0
        for q in order:
            a, b = ent[q], max(ext[q], ent[q])
            if expo[a:b + 1].max() + V[q] > EQ0:
                continue
            expo[a:b + 1] += V[q]; acc += 1
            bp, ratio = paths[q]; pnl = V[q] * ratio; prev = 0.0
            for dpos, v in zip(bp, pnl):
                delta[dpos] += v - prev; prev = v
        eq = EQ0 + np.cumsum(delta); peak = np.maximum.accumulate(eq)
        dd = ((eq - peak) / peak).min()
        yrs = pd.to_datetime(cal).year.values
        yp = {int(y): delta[yrs == y].sum() / EQ0 for y in np.unique(yrs)}
        res.append(dict(acc=acc, tot=(eq[-1] - EQ0) / EQ0, dd=dd, maxpos=None, avgexp=expo.mean() / EQ0, yp=yp))
    return res, len(m)


# ----------------------------------------------------------------------------- 主程式
def main():
    t0 = time.time()
    members, current, meta = load_membership()
    log(meta)
    pit_ok = not meta.startswith("Wikipedia 抓取失敗")
    if SYNTH:
        pit_ok = True; data = synth_data(); allt = [k for k in data if k.startswith("S") and k != "SPY"]
        members = {t: [(pd.Timestamp(START), pd.Timestamp("2100-01-01"))] for t in allt}; current = set(allt[:40])
    else:
        want = sorted(set(members) | set(ETFS) | {"SPY", "^VIX"})
        data = fetch(want)
    miss_cur = sorted(t for t in current if t not in data); rem = sorted(t for t in members if t not in current)
    got_rem = [t for t in rem if t in data]
    spy = data.get("SPY"); vix = data.get("^VIX")
    if spy is None:
        log("SPY 抓取失敗 → 結束"); sys.exit(1)
    last_day = spy.index[-1]
    kinds = {t: "stock" for t in members if t in data}
    kinds.update({t: "etf" for t in ETFS if t in data})
    S, W = build_signals(data, kinds, last_day)
    log(f"原始訊號（5日跌≥8%且<季線，含 ETF）{len(S)} 筆；資料至 {last_day.date()}")

    # 大盤狀態（訊號日當天）
    sc = spy["Close"]; sma200 = sc.rolling(200).mean()
    feat = pd.DataFrame({"bull": sc > sma200, "spy5": sc / sc.shift(5) - 1, "spyup": sc > sc.shift(1)})
    feat["vix"] = vix["Close"].reindex(spy.index).ffill() if vix is not None else np.nan
    F = feat.reindex(S["date"].values, method="ffill").reset_index(drop=True)
    S["bull"] = F["bull"].astype(bool).values; S["spy5"] = F["spy5"].values; S["spyup"] = F["spyup"].astype(bool).values; S["vix"] = F["vix"].values

    isst = (S["kind"] == "stock").values
    pit = np.array([in_member(members, t, d) for t, d in zip(S["tk"], S["date"])]) & isst if pit_ok else np.zeros(len(S), bool)
    cur_all = S["tk"].isin(current).values & isst
    cur_pre = cur_all & ~pit                       # 現成分股，但訊號當時尚未入選（前視偏差部分）
    rem_only = pit & ~S["tk"].isin(current).values  # 當時是成分、現在已移出（倖存者偏差的反例）
    etf = (S["kind"] == "etf").values
    y = S["date"].dt.year.values

    cache = {}
    def get(e, st, H):
        k = (e, st, H)
        if k not in cache:
            cache[k] = sim(W, S, e, st, H)
        return cache[k]
    base_stop, base_e, base_H = ("atr_intra", 3), "close", 10
    thr = lambda t: (S["ret5"].values <= -t)
    cool_cache = {}
    def sel(t, uni, uname):
        k = (t, uname)
        if k not in cool_cache:
            cool_cache[k] = cooled(S, thr(t) & uni)
        return cool_cache[k]
    B = get(base_e, base_stop, base_H)
    mb = sel(0.10, pit, "pit")

    L = [f"# 抄底模式｜嚴格回測（{dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M} UTC）", "",
         "只回測，未改雷達／推播／篩選。數據：yfinance 還原權息日線，抓不到一律「—」。", "",
         "## 規則與設定",
         ("- 訊號（收盤後已知）：個股 5 日跌幅 ≥ 門檻（預設 10%）且收盤 < 60 日均線（季線）。標的＝S&P 500 時點成分股。" if MODE == "below60" else "- 訊號（收盤後已知）：【上升趨勢拉回】個股 5 日跌幅 ≥ 門檻（此版預設仍用 10% 為基準列，另列 8%/12%/15%）且收盤 > 200 日均線。標的＝S&P 500 時點成分股。"),
         f"- 預設基準：訊號日收盤買進 → 最長持有 10 個交易日 → 停損＝進場價 − 3×ATR14（盤中觸價；跳空越過以開盤價成交）。來回成本 {COST * 100:.1f}%。",
         "- 同股票訊號 10 個交易日內只算第 1 次（冷卻）。「訊號日數 / 獨立事件」＝不同訊號日數／相隔 >10 天才算新的一波行情（事件）；⚠＝獨立事件 <30，結論參考性低。",
         "- 期望值(R)＝平均報酬 ÷ 停損距離（1R＝單筆風險 1%）。超額於SPY＝同進出場期間 SPY 報酬扣除後。t值為同一波行情內訊號群聚校正後。",
         "- 事件平均＝每一波行情先平均再平均（避免 2008／2020 這種一天幾十檔把結果撐大）。",
         f"- 時點成分股：{meta}。", ""]
    L += ["## 資料覆蓋與限制（倖存者偏差）", "",
          f"- 目前成分股抓到資料：{len(current) - len(miss_cur)}/{len(current)}（抓不到：{', '.join(miss_cur[:20]) or '無'}）",
          f"- {START[:4]} 年後曾移出成分股的代號：{len(rem)} 個，其中 Yahoo 仍有日線者 {len(got_rem)} 個（{pc(len(got_rem) / max(len(rem), 1), 0)}）。"
          "已下市／被併購且 Yahoo 已刪除資料者抓不到，**這部分仍缺**——而它們多半是下跌後沒回來的股票，所以以下結果對抄底**仍偏樂觀**，只能當上限參考。",
          "- 已下市但資料還在者：窗口超過最後交易日時以最後收盤價結算（不排除）。",
          "- ETF（SPY/QQQ/IWM/部門 ETF 等）沒有倖存者偏差，作為無偏對照，但訊號少。", ""]

    # --- 表 1：基準規則 × 不同母體
    L += ["## 1. 基準規則在不同母體的結果（偏差檢驗）", "", *HEAD]
    for lab, uni in [("時點成分股（主要）", pit), ("　└ 其中：當時是成分、現已移出（反偏差樣本）", rem_only), ("目前成分股（不看當時是否已入選，偏樂觀）", cur_all),
                     ("　└ 其中：訊號當時尚未入選（前視偏差部分）", cur_pre), ("ETF 對照（無倖存者偏差）", etf)]:
        if lab.startswith("時點") and not pit_ok:
            L.append(f"| {lab} | — | | | | | | | | | | | |"); continue
        mk = sel(0.10, uni, lab)
        L.append(row(lab, stats(mk, B, S, spy)))
    L.append("")

    # --- 表 2：門檻
    L += ["## 2. 跌幅門檻（時點成分股，基準進出場）", "", *HEAD]
    for t in THRS:
        L.append(row(f"5日跌 ≥ {int(t * 100)}%", stats(sel(t, pit, "pit"), B, S, spy)))
    L.append("")

    # --- 表 3：進場
    L += ["## 3. 進場方式（10% 門檻，3×ATR 盤中停損，持有 10 天）", "", *HEAD]
    for e, lab in [("close", "訊號日收盤買"), ("open", "隔日開盤買"), ("confA", "等止跌K：3天內收紅且未破訊號日低點，收盤買"), ("confB", "等止跌K：3天內收盤站上前日高點，收盤買")]:
        L.append(row(lab, stats(mb, get(e, base_stop, base_H), S, spy)))
    L.append("")

    # --- 表 4：停損
    L += ["## 4. 停損方式（訊號日收盤買，持有 10 天）", "", *HEAD]
    for st, lab in [(("none",), "不設停損（對照）"), (("atr_intra", 2), "2×ATR 盤中"), (("atr_intra", 3), "3×ATR 盤中（Joe 現行）"), (("atr_intra", 4), "4×ATR 盤中"),
                    (("atr_close", 3), "3×ATR 收盤確認→隔日開盤出"), (("fix_intra", 0.07), "固定 −7% 盤中"), (("fix_intra", 0.10), "固定 −10% 盤中")]:
        L.append(row(lab, stats(mb, get(base_e, st, base_H), S, spy)))
    L.append("")

    # --- 表 5：持有天數
    L += ["## 5. 最長持有天數（訊號日收盤買，3×ATR 盤中停損）", "", *HEAD]
    for H in (5, 10, 20):
        L.append(row(f"最長 {H} 天", stats(mb, get(base_e, base_stop, H), S, spy)))
    L.append("")

    # --- 表 6：大盤濾網
    sl = {"全部": np.ones(len(S), bool), "SPY 在 200 日線上": S["bull"].values, "SPY 在 200 日線下": ~S["bull"].values,
          "大盤 5 日跌 ≥ 5%（系統性恐慌）": (S["spy5"].values <= -0.05), "大盤 5 日跌 < 5%": (S["spy5"].values > -0.05),
          "個股獨跌（SPY>200日線 且 SPY 5日>−2%）": S["bull"].values & (S["spy5"].values > -0.02),
          "VIX < 20": S["vix"].values < 20, "VIX 20–30": (S["vix"].values >= 20) & (S["vix"].values < 30), "VIX ≥ 30": S["vix"].values >= 30,
          "訊號日大盤收紅": S["spyup"].values, "訊號日大盤收黑": ~S["spyup"].values}
    L += ["## 6. 大盤環境切片（時點成分股，基準進出場；同一批訊號切分，互不重複加總的列才可加總）", "", *HEAD]
    for lab, m in sl.items():
        L.append(row(lab, stats(mb & m, B, S, spy)))
    L += ["", "同樣的切片，改成「等止跌K（收紅且未破低）」進場：", "", *HEAD]
    A = get("confA", base_stop, base_H)
    for lab, m in sl.items():
        L.append(row(lab, stats(mb & m, A, S, spy)))
    L.append("")

    # --- 表 7：分年度
    def yearly(title, mask, s):
        out = [f"**{title}**", "", "| 年 | 筆數 | 獨立事件 | 勝率 | 平均 | 期望值(R) | 超額於SPY | 停損率 | 最差 |", "|---|---|---|---|---|---|---|---|---|"]
        for yy in sorted(set(y[mask & s["valid"]])):
            st = stats(mask & (y == yy), s, S, spy)
            tag = " 🐻" if yy in (2008, 2009, 2011, 2015, 2018, 2020, 2022, 2025) else ""
            out.append(f"| {yy}{tag} | <5 | | | | | | | |" if st is None else
                       f"| {yy}{tag} | {st['n']} | {st['eps']}{'⚠' if st['eps'] < 3 else ''} | {pc(st['win'], 0)} | {pc(st['mean'], 2, True)} | {f2(st['ER'])} | {pc(st['exc'], 2, True)} | {pc(st['stop'], 0)} | {pc(st['worst'], 0)} |")
        return out + [""]
    L += ["## 7. 分年度（🐻＝明顯空頭／大跌年；每年獨立事件很少，單年不要過度解讀）", ""]
    L += yearly("基準：時點成分股，收盤買、3×ATR 盤中停損、10 天", mb, B)
    L += yearly("等止跌K（收紅且未破低）進場", mb, A)
    L += yearly("基準 ＋ 僅 SPY 在 200 日線上", mb & S["bull"].values, B)
    L += yearly("基準 ＋ 僅 SPY 在 200 日線下", mb & ~S["bull"].values, B)

    # --- 表 8：組合模擬
    L += ["## 8. 組合層級模擬（91,000 美元，單筆風險 1%，單檔上限 20%，總曝險 ≤100% 不加槓桿，同日超額訊號隨機取，15 組種子）", "",
          "損益以固定本金 91,000 計（不複利），回撤＝每日以收盤盯市的權益曲線，含持有期間浮虧。", "",
          "| 組別 | 可用訊號 | 平均實際下單 | 總損益（中位／最差種子） | 年均損益 | 最大回撤（中位／最差） | 平均資金使用率 | 最差年度損益 |", "|---|---|---|---|---|---|---|---|"]
    years_span = (last_day - pd.Timestamp("2004-09-01")).days / 365.25
    port_rows = [("基準（收盤買、3ATR 盤中停損、10天）", mb, B), ("等止跌K（收紅且未破低）", mb, A),
                 ("基準 ＋ SPY>200日線", mb & S["bull"].values, B), ("基準 ＋ SPY≤200日線", mb & ~S["bull"].values, B),
                 ("基準 ＋ 大盤5日跌<5%", mb & (S["spy5"].values > -0.05), B)]
    for lab, mk, s in port_rows:
        p = portfolio(S, s, mk, spy)
        if p is None:
            L.append(f"| {lab} | 0 | — | — | — | — | — | — |"); continue
        res, avail = p
        tot = sorted(r["tot"] for r in res); dd = sorted(r["dd"] for r in res)
        worst_y = min(min(r["yp"].values()) for r in res); med = res[len(res) // 2]
        L.append(f"| {lab} | {avail} | {np.mean([r['acc'] for r in res]):.0f} | {pc(np.median(tot), 1, True)} / {pc(tot[0], 1, True)} | {pc(np.median(tot) / years_span, 1, True)} | "
                 f"{pc(np.median(dd), 1)} / {pc(dd[0], 1)} | {pc(np.mean([r['avgexp'] for r in res]), 0)} | {pc(worst_y, 1, True)} |")
    L.append("")
    L += ["<details><summary>執行紀錄</summary>", "", "```", *LOG[:60], "```", "</details>", ""]
    L.append(f"_執行 {time.time() - t0:.0f} 秒；資料 {len(data)} 檔；訊號 {len(S)} 筆。_")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
