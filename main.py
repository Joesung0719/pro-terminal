import os
import re
import math
import time
import hmac
import threading
import datetime
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from zoneinfo import ZoneInfo
    ET = ZoneInfo("America/New_York")
    TPE = ZoneInfo("Asia/Taipei")
except Exception:  # 需要在 requirements.txt 加入 tzdata
    ET = None
    TPE = None

APP_VERSION = "V10.1"

app = FastAPI(title=f"US Professional Radar War Room - {APP_VERSION}")

# 修正：allow_origins="*" 時不可同時 allow_credentials=True（瀏覽器會拒絕，也有安全疑慮）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

LINE_CHANNEL_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_TOKEN", "").strip()
LINE_USER_ID = os.environ.get("LINE_USER_ID", "").strip()
ADMIN_PUSH_PIN = os.environ.get("ADMIN_PUSH_PIN", "").strip()


# ==============================================================================
# 工具：快取、時區、輸入驗證
# ==============================================================================
_CACHE = {}
_CACHE_LOCK = threading.Lock()


_LAST_ERR = {}


def _is_valid_cache_val(val) -> bool:
    """DataFrame 不能直接 bool()；這是早上圖表 HTTP 500 的根本原因。"""
    if val is None:
        return False
    if isinstance(val, (pd.DataFrame, pd.Series)):
        return not val.empty
    if isinstance(val, (dict, list, tuple, set, str)):
        return len(val) > 0
    return True


def cached(key, ttl, fn, cache_empty=False):
    """TTL 快取。例外不再被悄悄吞掉：原因記錄在 _LAST_ERR，可由 /api/debug 查看。"""
    now = time.time()
    with _CACHE_LOCK:
        hit = _CACHE.get(key)
        if hit and now - hit[0] < ttl:
            return hit[1]
    try:
        val = fn()
        _LAST_ERR.pop(key, None)
    except Exception as e:
        _LAST_ERR[key] = f"{type(e).__name__}: {str(e)[:200]}"
        return None
    if cache_empty or _is_valid_cache_val(val):
        with _CACHE_LOCK:
            _CACHE[key] = (now, val)
            if len(_CACHE) > 500:
                for k in sorted(_CACHE, key=lambda x: _CACHE[x][0])[:100]:
                    _CACHE.pop(k, None)
    return val


def _utc_now():
    return datetime.datetime.now(datetime.timezone.utc)


def et_now() -> datetime.datetime:
    if ET is not None:
        return datetime.datetime.now(ET)
    u = _utc_now()
    year = u.year
    d = datetime.datetime(year, 3, 8, 7, tzinfo=datetime.timezone.utc)
    d += datetime.timedelta(days=(6 - d.weekday()) % 7)
    e = datetime.datetime(year, 11, 1, 6, tzinfo=datetime.timezone.utc)
    e += datetime.timedelta(days=(6 - e.weekday()) % 7)
    off = -4 if d <= u < e else -5
    return u.astimezone(datetime.timezone(datetime.timedelta(hours=off)))


def tpe_now() -> datetime.datetime:
    if TPE is not None:
        return datetime.datetime.now(TPE)
    return _utc_now().astimezone(datetime.timezone(datetime.timedelta(hours=8)))


# NYSE 休市日（與 us/us_scheduler.py 的 NYSE_HOLIDAYS 相同，改一邊要同步改另一邊）
NYSE_HOLIDAYS = {
    # 2026
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25", "2026-06-19",
    "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25",
    # 2027
    "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31", "2027-06-18",
    "2027-07-05", "2027-09-06", "2027-11-25", "2027-12-24",
}

# Nasdaq 23/5 夜盤（美東 21:00–04:00）預計 2026/12/6（週日）21:00 起；若官方延期，改這個日期即可。
# 每日 20:00–21:00 全市場暫停。21:00 之後的成交算「隔天」的交易日（期貨式），假日當晚 21:00 恢復。
OVERNIGHT_START = datetime.datetime(2026, 12, 6, 21, 0)


def us_session_label() -> str:
    n = et_now()
    hm = n.hour * 100 + n.minute
    # 20:00 之後（暫停＋夜盤）屬於隔天的交易日；週日 21:00 前仍是週末
    trade_day = n.date() + datetime.timedelta(days=1 if hm >= 2000 else 0)
    if trade_day.weekday() >= 5 or (n.weekday() == 6 and hm < 2100):
        return "休市(週末)"
    if trade_day.isoformat() in NYSE_HOLIDAYS:
        return "休市(假日)"
    if 930 <= hm < 1600:
        return "盤中"
    if 400 <= hm < 930:
        return "盤前"
    if 1600 <= hm < 2000:
        return "盤後"
    overnight = n.replace(tzinfo=None) >= OVERNIGHT_START
    if overnight and (hm >= 2100 or hm < 400):
        return "夜盤"
    if overnight and 2000 <= hm < 2100:
        return "暫停"
    return "休市"


SYMBOL_RE = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,9}$")


def clean_symbol(s: str) -> str:
    sym = (s or "").strip().upper()
    if not SYMBOL_RE.match(sym):
        raise HTTPException(status_code=400, detail="股票代號格式不正確")
    return sym


def _f(x, default=0.0):
    try:
        if x is None:
            return default
        if isinstance(x, (pd.Series, np.ndarray)):
            if len(x) == 0:
                return default
            x = x.iloc[-1] if isinstance(x, pd.Series) else x[-1]
        if pd.isna(x):
            return default
        v = float(x)
        return default if (math.isnan(v) or math.isinf(v)) else v
    except Exception:
        return default


def _safe_round(x, nd=2, default=None):
    """取不到就回傳 default(預設 None)，絕不拿別的數字頂替。"""
    try:
        if x is None:
            return default
        if isinstance(x, (pd.Series, np.ndarray)):
            if len(x) == 0:
                return default
            x = x.iloc[-1] if isinstance(x, pd.Series) else x[-1]
        if pd.isna(x):
            return default
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return round(v, nd)
    except Exception:
        return default


# ==============================================================================
# Black-Scholes Greeks（移除 scipy 依賴，以 math.erf 取代，部署更輕）
# ==============================================================================
def _ncdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _npdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def calculate_greeks(S, K, T, r, sigma, option_type="put"):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return {"delta": 0.0, "theta": 0.0, "gamma": 0.0, "vega": 0.0}
    sq = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / sq
    d2 = d1 - sq
    if option_type == "call":
        delta = _ncdf(d1)
        theta = (-(S * _npdf(d1) * sigma) / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * _ncdf(d2)) / 365
    else:
        delta = _ncdf(d1) - 1
        theta = (-(S * _npdf(d1) * sigma) / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * _ncdf(-d2)) / 365
    gamma = _npdf(d1) / (S * sq)
    vega = S * _npdf(d1) * math.sqrt(T) / 100
    return {
        "delta": round(delta, 4), "theta": round(theta, 4),
        "gamma": round(gamma, 4), "vega": round(vega, 4)
    }


def risk_free_rate():
    """無風險利率：優先用 Yahoo 的 13 週美國國庫券殖利率(^IRX)；取不到才用預設並明確標示。"""
    def _load():
        df = _normalize_df(yf.Ticker("^IRX").history(period="10d"))
        if df.empty:
            return None
        return round(float(df["Close"].iloc[-1]) / 100.0, 4)

    r = cached(("irx",), 6 * 3600, _load)
    if r is not None and 0 <= r < 0.2:
        return r, "13週美國國庫券殖利率(^IRX)"
    return 0.045, "預設假設 4.5%（^IRX 取得失敗）"


WATCHLIST_43 = [
    {"symbol": "NVDA", "name": "輝達", "sector": "SOX", "market": "NASDAQ"},
    {"symbol": "TSM", "name": "台積電 ADR", "sector": "SOX", "market": "NYSE"},
    {"symbol": "AMD", "name": "超微半導體", "sector": "SOX", "market": "NASDAQ"},
    {"symbol": "AVGO", "name": "博通", "sector": "SOX", "market": "NASDAQ"},
    {"symbol": "QCOM", "name": "高通", "sector": "SOX", "market": "NASDAQ"},
    {"symbol": "MU", "name": "美光科技", "sector": "SOX", "market": "NASDAQ"},
    {"symbol": "INTC", "name": "英特爾", "sector": "SOX", "market": "NASDAQ"},
    {"symbol": "ARM", "name": "安謀科技", "sector": "SOX", "market": "NASDAQ"},
    {"symbol": "ASML", "name": "艾司摩爾", "sector": "SOX", "market": "NASDAQ"},
    {"symbol": "TXN", "name": "德州儀器", "sector": "SOX", "market": "NASDAQ"},
    {"symbol": "AAPL", "name": "蘋果", "sector": "NASDAQ", "market": "NASDAQ"},
    {"symbol": "MSFT", "name": "微軟", "sector": "NASDAQ", "market": "NASDAQ"},
    {"symbol": "GOOGL", "name": "Alphabet (Google)", "sector": "NASDAQ", "market": "NASDAQ"},
    {"symbol": "AMZN", "name": "亞馬遜", "sector": "NASDAQ", "market": "NASDAQ"},
    {"symbol": "META", "name": "Meta 臉書", "sector": "NASDAQ", "market": "NASDAQ"},
    {"symbol": "TSLA", "name": "特斯拉", "sector": "NASDAQ", "market": "NASDAQ"},
    {"symbol": "NFLX", "name": "網飛 Netflix", "sector": "NASDAQ", "market": "NASDAQ"},
    {"symbol": "COST", "name": "好市多 Costco", "sector": "NASDAQ", "market": "NASDAQ"},
    {"symbol": "ADBE", "name": "奧多比 Adobe", "sector": "NASDAQ", "market": "NASDAQ"},
    {"symbol": "COIN", "name": "Coinbase 加密所", "sector": "NASDAQ", "market": "NASDAQ"},
    {"symbol": "CAT", "name": "開拓重工", "sector": "DOW", "market": "NYSE"},
    {"symbol": "BA", "name": "波音公司", "sector": "DOW", "market": "NYSE"},
    {"symbol": "UNH", "name": "聯合健康集團", "sector": "DOW", "market": "NYSE"},
    {"symbol": "JNJ", "name": "嬌生生技", "sector": "DOW", "market": "NYSE"},
    {"symbol": "MCD", "name": "麥當勞", "sector": "DOW", "market": "NYSE"},
    {"symbol": "HD", "name": "家得寶 Home Depot", "sector": "DOW", "market": "NYSE"},
    {"symbol": "V", "name": "Visa 信用卡", "sector": "DOW", "market": "NYSE"},
    {"symbol": "WMT", "name": "沃爾瑪", "sector": "DOW", "market": "NYSE"},
    {"symbol": "IBM", "name": "國際商業機器 IBM", "sector": "DOW", "market": "NYSE"},
    {"symbol": "GS", "name": "高盛集團", "sector": "DOW", "market": "NYSE"},
    {"symbol": "PLTR", "name": "帕蘭蒂爾科技", "sector": "NYSE", "market": "NYSE"},
    {"symbol": "JPM", "name": "摩根大通", "sector": "NYSE", "market": "NYSE"},
    {"symbol": "BAC", "name": "美國銀行", "sector": "NYSE", "market": "NYSE"},
    {"symbol": "XOM", "name": "埃克森美孚石油", "sector": "NYSE", "market": "NYSE"},
    {"symbol": "CVX", "name": "雪佛龍能源", "sector": "NYSE", "market": "NYSE"},
    {"symbol": "LLY", "name": "禮來藥廠", "sector": "NYSE", "market": "NYSE"},
    {"symbol": "NKE", "name": "耐吉 Nike", "sector": "NYSE", "market": "NYSE"},
    {"symbol": "UBER", "name": "優步 Uber", "sector": "NYSE", "market": "NYSE"},
    {"symbol": "SPY", "name": "標普500 ETF", "sector": "ETF", "market": "AMEX"},
    {"symbol": "QQQ", "name": "那斯達克100 ETF", "sector": "ETF", "market": "NASDAQ"},
    {"symbol": "SOXX", "name": "費城半導體 ETF", "sector": "ETF", "market": "NASDAQ"},
    {"symbol": "DIA", "name": "道瓊工業 ETF", "sector": "ETF", "market": "AMEX"},
    {"symbol": "IWM", "name": "羅素2000小型股 ETF", "sector": "ETF", "market": "AMEX"}
]

# ------------------------------------------------------------------------------
# 掃描股票池：標普500 ＋ 那斯達克100（約 500 檔，GICS 產業分類）
#   載入順序：① 同資料夾 us_universe.json（可直接用回測產生的 us_backtest_universe.json 改名）
#            ② 上次成功抓到的 us_universe_cache.json（30 天內）
#            ③ 啟動後背景向維基百科抓最新成分股
#            ④ 以上皆失敗 → 退回原本 43 檔觀察名單（畫面會明確標示）
# 原本 43 檔固定保留為「我的觀察名單」並保留中文名稱。
# ------------------------------------------------------------------------------
import json as _json

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UNIVERSE_FILE = os.path.join(_BASE_DIR, "us_universe.json")
UNIVERSE_CACHE = os.path.join(_BASE_DIR, "us_universe_cache.json")

SECTOR_ZH = {
    "Information Technology": "💻 資訊科技",
    "Health Care": "🏥 醫療保健",
    "Financials": "🏦 金融",
    "Consumer Discretionary": "🛍️ 非必需消費",
    "Communication Services": "📡 通訊服務",
    "Industrials": "🏭 工業",
    "Consumer Staples": "🛒 必需消費",
    "Energy": "🛢️ 能源",
    "Utilities": "💡 公用事業",
    "Real Estate": "🏢 房地產",
    "Materials": "⛏️ 原物料",
    "ETF": "📊 主力 ETF",
    "OTHER": "❔ 其他／未分類",
}
# 原 43 檔中，成分股名單可能沒有的外國 ADR 的產業（人工標註）
_ORIG_SECTOR_FALLBACK = {"TSM": "Information Technology", "ASML": "Information Technology",
                         "ARM": "Information Technology"}

_UNI_LOCK = threading.Lock()
_UNI_META = {"source": "內建43檔", "built": None, "note": ""}
MARKET_UNIVERSE = []
UNIVERSE_SYMBOLS = []
UNIVERSE_BY_SYMBOL = {}


def _norm_sym(x) -> str:
    return str(x).strip().upper().replace(".", "-")


def _build_universe(rows, source):
    """rows: [[ticker, sector], ...]；與 43 檔觀察名單合併後設為目前股票池。"""
    sec_map = {}
    for r in rows or []:
        try:
            t, sec = _norm_sym(r[0]), (str(r[1]).strip() if len(r) > 1 else "")
        except Exception:
            continue
        if t and t != "NAN" and re.fullmatch(r"[A-Z0-9\-]{1,8}", t):
            sec_map.setdefault(t, sec)
    items, seen = [], set()
    for w in WATCHLIST_43:
        sym = w["symbol"]
        if w["sector"] == "ETF":
            sec = "ETF"
        else:
            sec = sec_map.get(sym) or _ORIG_SECTOR_FALLBACK.get(sym, "OTHER")
        items.append({"symbol": sym, "name": w["name"], "sector": sec if sec in SECTOR_ZH else "OTHER",
                      "market": w["market"], "watch": True})
        seen.add(sym)
    for t, sec in sorted(sec_map.items()):
        if t in seen:
            continue
        items.append({"symbol": t, "name": t, "sector": sec if sec in SECTOR_ZH else "OTHER",
                      "market": "US", "watch": False})
    global MARKET_UNIVERSE, UNIVERSE_SYMBOLS, UNIVERSE_BY_SYMBOL
    with _UNI_LOCK:
        MARKET_UNIVERSE = items
        UNIVERSE_SYMBOLS = [x["symbol"] for x in items]
        UNIVERSE_BY_SYMBOL = {x["symbol"]: x for x in items}
        _UNI_META.update({"source": source, "built": _utc_now().isoformat(timespec="seconds") if "_utc_now" in globals() else ""})
    return len(items)


def _read_universe_file(path, max_age_days=None):
    try:
        if not os.path.exists(path):
            return None
        if max_age_days is not None and (time.time() - os.path.getmtime(path)) > max_age_days * 86400:
            return None
        rows = _json.load(open(path, encoding="utf-8"))
        if isinstance(rows, list) and len(rows) >= 100:
            return rows
    except Exception:
        pass
    return None


_WIKI_UA = {"User-Agent": "Mozilla/5.0 (compatible; USRadar/10.1)"}


def _wiki_tables(url):
    import io
    r = requests.get(url, headers=_WIKI_UA, timeout=20)
    r.raise_for_status()
    try:
        return pd.read_html(io.StringIO(r.text))
    except ImportError:
        return pd.read_html(io.StringIO(r.text), flavor="bs4")


def fetch_universe_online():
    """向維基百科抓標普500＋那斯達克100成分股；回傳 (rows, 訊息)。"""
    uni, msgs = {}, []
    try:
        for t in _wiki_tables("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"):
            if "Symbol" in t.columns:
                col = "GICS Sector" if "GICS Sector" in t.columns else None
                for _, r in t.iterrows():
                    uni.setdefault(_norm_sym(r["Symbol"]), str(r[col]) if col else "")
                break
        msgs.append(f"S&P500 {len(uni)} 檔")
    except Exception as e:
        msgs.append(f"S&P500 失敗:{type(e).__name__}")
    n0 = len(uni)
    try:
        for t in _wiki_tables("https://en.wikipedia.org/wiki/Nasdaq-100"):
            col = next((c for c in t.columns if str(c).strip() in ("Ticker", "Symbol")), None)
            if col is not None and len(t) >= 80:
                sec = next((c for c in t.columns if "Sector" in str(c)), None)
                for _, r in t.iterrows():
                    uni.setdefault(_norm_sym(r[col]), str(r[sec]) if sec else "")
                break
        msgs.append(f"Nasdaq100 新增 {len(uni) - n0} 檔")
    except Exception as e:
        msgs.append(f"Nasdaq100 失敗:{type(e).__name__}")
    rows = sorted([[k, v] for k, v in uni.items() if k and k != "NAN"])
    return rows, "；".join(msgs)


def init_universe():
    """同步、快速：先用本機檔案或 43 檔，讓服務立刻可用。"""
    rows = _read_universe_file(UNIVERSE_FILE)
    if rows:
        _UNI_META["note"] = ""
        return _build_universe(rows, "us_universe.json，標普500＋那斯達克100")
    rows = _read_universe_file(UNIVERSE_CACHE, 30)
    if rows:
        return _build_universe(rows, "上次抓取的成分股快取")
    _UNI_META["note"] = "尚未取得成分股名單，暫用內建 43 檔"
    return _build_universe([], "內建43檔（成分股名單尚未取得）")


init_universe()


# ==============================================================================
# 歷史資料與指標（修正：均線一律要求滿期數才有值，不再 min_periods=1 偽造年線）
# ==============================================================================
HISTORY_PLAN = {
    # interval: (抓取區間, 顯示根數)；抓得比顯示長，均線才有足夠暖機資料
    "1d": ("2y", 252),
    "1wk": ("10y", 156),
    "1mo": ("max", 120),
}
INTERVAL_UNIT = {"1d": "日", "1wk": "週", "1mo": "月"}


def _strip_tz(df: pd.DataFrame) -> pd.DataFrame:
    if getattr(df.index, "tz", None) is not None:
        df = df.copy()
        df.index = df.index.tz_localize(None)
    return df


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """統一資料格式：攤平 MultiIndex 欄位、去除重複欄位、去時區、去除無收盤價的列。"""
    if df is None or df.empty:
        return pd.DataFrame()
    d = df.copy()
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    d = d.loc[:, ~d.columns.duplicated()]
    if getattr(d.index, "tz", None) is not None:
        d.index = d.index.tz_localize(None)
    if "Close" not in d.columns:
        return pd.DataFrame()
    return d.dropna(subset=["Close"])


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    c = d["Close"]
    for p in (5, 10, 20, 60, 120, 240):
        d[f"MA{p}"] = c.rolling(p, min_periods=p).mean()
    std20 = c.rolling(20, min_periods=20).std(ddof=0)
    d["BB_UP"] = d["MA20"] + 2.0 * std20
    d["BB_LOW"] = d["MA20"] - 2.0 * std20
    d["Vol_MA5"] = d["Volume"].shift(1).rolling(5, min_periods=3).mean()

    low9 = d["Low"].rolling(9, min_periods=9).min()
    high9 = d["High"].rolling(9, min_periods=9).max()
    rsv = ((c - low9) / (high9 - low9).replace(0, np.nan) * 100.0).fillna(50.0)
    k_vals, d_vals = [], []
    k_prev, d_prev = 50.0, 50.0
    for val in rsv.values:
        k_cur = (2.0 / 3.0) * k_prev + (1.0 / 3.0) * float(val)
        d_cur = (2.0 / 3.0) * d_prev + (1.0 / 3.0) * k_cur
        k_vals.append(k_cur)
        d_vals.append(d_cur)
        k_prev, d_prev = k_cur, d_cur
    d["K"] = k_vals
    d["D"] = d_vals

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    d["DIF"] = ema12 - ema26
    d["DEA"] = d["DIF"].ewm(span=9, adjust=False).mean()
    d["MACD_HIST"] = d["DIF"] - d["DEA"]
    return d


def fetch_history(symbol: str, interval: str) -> pd.DataFrame:
    period, _ = HISTORY_PLAN.get(interval, ("2y", 252))

    def _load():
        last = None
        for _ in range(2):
            try:
                df = _normalize_df(yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=False))
                if not df.empty:
                    return df
            except Exception as e:
                last = e
                time.sleep(0.8)
        if last:
            raise last
        return None

    return cached(("hist", symbol, interval), 120, _load)


def calc_neckline(df: pd.DataFrame, neck_days: int, mode: str):
    """N 根 K 棒頸線：多方=前N根最高、空方=前N根最低（不含當根）。"""
    if len(df) < neck_days + 2:
        return None
    win = df.iloc[-(neck_days + 1):-1]
    v = float(win["High"].max()) if mode == "LONG" else float(win["Low"].min())
    return round(v, 2)


def quality_badge(mode: str, spot: float, neck: float, ma20) -> tuple:
    if neck is None or neck <= 0:
        return "資料不足", 0.0
    if mode == "LONG":
        dist = (spot / neck - 1.0) * 100.0
        if dist <= 0:
            return "⏳ 尚未突破頸線", dist
        # 台股回測顯示「距頸線遠近」對後續報酬沒有穩定影響，不再以 5% 作為警告；只看是否站上月線
        if ma20 is None:
            return "✅ 已突破（月線資料不足）", dist
        return ("✅ 已突破且站上月線", dist) if spot >= ma20 else ("⚠️ 已突破但在月線下", dist)
    dist = (neck / spot - 1.0) * 100.0
    if dist <= 0:
        return "⏳ 尚未跌破頸線", dist
    if ma20 is None:
        return "✅ 已跌破（月線資料不足）", dist
    return ("✅ 已跌破且在月線下", dist) if spot <= ma20 else ("⚠️ 已跌破但仍在月線上", dist)


def week52_stats(df_daily):
    """52 週高低：取最近 252 根日K 的『最高價 / 最低價』(與 Yahoo 52週區間定義一致)。
    少於 240 根(上市未滿一年)一律回傳 None，不用不足一年的資料充數。"""
    if df_daily is None or len(df_daily) < 240:
        return None
    w = df_daily.tail(252)
    prior = w.iloc[:-1]
    hi, lo = float(w["High"].max()), float(w["Low"].min())
    last_c, last_h, last_l = float(w["Close"].iloc[-1]), float(w["High"].iloc[-1]), float(w["Low"].iloc[-1])
    prior_hi, prior_lo = float(prior["High"].max()), float(prior["Low"].min())
    rng = hi - lo
    return {
        "high": round(hi, 2), "high_date": w["High"].idxmax().strftime("%Y-%m-%d"),
        "low": round(lo, 2), "low_date": w["Low"].idxmin().strftime("%Y-%m-%d"),
        "from_high_pct": round((last_c / hi - 1) * 100, 1),
        "from_low_pct": round((last_c / lo - 1) * 100, 1),
        "position_pct": round((last_c - lo) / rng * 100, 0) if rng > 0 else None,
        "new_high": bool(last_h >= prior_hi),            # 盤中曾創 52 週新高
        "close_new_high": bool(last_c >= prior_hi),      # 收盤價站上前 52 週最高（較強）
        "new_low": bool(last_l <= prior_lo),
        "close_new_low": bool(last_c <= prior_lo),
        "bars": len(w),
    }


def _series(s: pd.Series, nd=2):
    return [None if pd.isna(v) else round(float(v), nd) for v in s.values]


# ==============================================================================
# 全市場日K：背景執行緒分批下載並存放，請求時直接讀取（不在請求當下下載）
# ==============================================================================
def _split_download(raw, symbols):
    out = {}
    if raw is None or raw.empty:
        return out
    multi = isinstance(raw.columns, pd.MultiIndex)
    if multi:
        lvl0 = set(raw.columns.get_level_values(0))
        lvl1 = set(raw.columns.get_level_values(1))
        for s in symbols:
            try:
                if s in lvl0:
                    sub = raw[s].copy()
                elif s in lvl1:
                    sub = raw.xs(s, axis=1, level=1).copy()
                else:
                    continue
                sub = _normalize_df(sub)
                if not sub.empty:
                    out[s] = sub
            except Exception:
                continue
    elif len(symbols) == 1:
        sub = _normalize_df(raw)
        if not sub.empty:
            out[symbols[0]] = sub
    return out


_STORE = {"data": {}, "stamp": {}, "cycle_done": None, "cycle_secs": None,
          "progress": 0, "total": 0, "running": False, "last_err": "", "cycles": 0}
_STORE_LOCK = threading.Lock()
_BATCH = 40


def load_universe_daily():
    """回傳背景程序已抓好的 1 年日K（symbol → DataFrame）。不在請求當下才下載，避免逾時與限流。"""
    with _STORE_LOCK:
        return dict(_STORE["data"])


def _download_batch(syms):
    raw = yf.download(" ".join(syms), period="1y", group_by="ticker", threads=4,
                      progress=False, auto_adjust=False)
    return _split_download(raw, syms)


def _refresh_cycle():
    """抓一輪全部股票：觀察名單優先；每批 40 檔；缺的逐檔補抓；失敗者保留舊資料。"""
    syms = list(UNIVERSE_SYMBOLS)
    watch = [x["symbol"] for x in MARKET_UNIVERSE if x.get("watch")]
    wset = set(watch)
    order = watch + [s for s in syms if s not in wset]
    t0 = time.time()
    with _STORE_LOCK:
        _STORE["total"] = len(order)
        _STORE["progress"] = 0
        _STORE["running"] = True
    missing = []
    for k in range(0, len(order), _BATCH):
        batch = order[k:k + _BATCH]
        got = {}
        try:
            got = _download_batch(batch)
        except Exception as e:
            _STORE["last_err"] = f"{type(e).__name__}: {str(e)[:120]}"
        now = time.time()
        with _STORE_LOCK:
            for sym, df in got.items():
                _STORE["data"][sym] = df
                _STORE["stamp"][sym] = now
            _STORE["progress"] = min(len(order), k + len(batch))
        missing += [s for s in batch if s not in got]
        time.sleep(1.5)
    # 逐檔補抓缺的（最多 60 檔，避免拖太久）
    for sym in missing[:60]:
        try:
            got = _download_batch([sym])
            if sym in got:
                with _STORE_LOCK:
                    _STORE["data"][sym] = got[sym]
                    _STORE["stamp"][sym] = time.time()
        except Exception as e:
            _STORE["last_err"] = f"{sym} {type(e).__name__}"
        time.sleep(0.6)
    with _STORE_LOCK:
        _STORE["cycle_done"] = time.time()
        _STORE["cycle_secs"] = round(time.time() - t0)
        _STORE["running"] = False
        _STORE["cycles"] += 1
        # 清掉已不在股票池的舊資料
        keep = set(UNIVERSE_SYMBOLS)
        for sym in list(_STORE["data"].keys()):
            if sym not in keep:
                _STORE["data"].pop(sym, None)
                _STORE["stamp"].pop(sym, None)


def _refresher_loop():
    # ① 若本機沒有名單檔，先上網抓一次成分股
    if _UNI_META["source"].startswith(("內建", "上次")):
        try:
            rows, msg = fetch_universe_online()
            if len(rows) >= 300:
                _build_universe(rows, "維基百科即時抓取（標普500＋那斯達克100）")
                _UNI_META["note"] = msg
                try:
                    _json.dump(rows, open(UNIVERSE_CACHE, "w", encoding="utf-8"), ensure_ascii=False)
                except Exception:
                    pass
            else:
                _UNI_META["note"] = f"成分股名單抓取不完整（{msg}），沿用 {_UNI_META['source']}"
        except Exception as e:
            _UNI_META["note"] = f"成分股名單抓取失敗：{type(e).__name__}"
    # ② 持續更新日K
    while True:
        try:
            _refresh_cycle()
        except Exception as e:
            with _STORE_LOCK:
                _STORE["running"] = False
                _STORE["last_err"] = f"{type(e).__name__}: {str(e)[:120]}"
        sess = us_session_label()
        time.sleep(600 if sess in ("盤中", "盤前") else 3600)


_REFRESHER_STARTED = False


@app.on_event("startup")
def _start_refresher():
    global _REFRESHER_STARTED
    if _REFRESHER_STARTED:
        return
    _REFRESHER_STARTED = True
    threading.Thread(target=_refresher_loop, daemon=True, name="universe-refresher").start()


def universe_status():
    with _STORE_LOCK:
        loaded = len([s for s in UNIVERSE_SYMBOLS if s in _STORE["data"]])
        done, secs = _STORE["cycle_done"], _STORE["cycle_secs"]
        prog, total, running, err, cycles = (_STORE["progress"], _STORE["total"], _STORE["running"],
                                             _STORE["last_err"], _STORE["cycles"])
    age_min = round((time.time() - done) / 60) if done else None
    return {
        "total": len(UNIVERSE_SYMBOLS), "loaded": loaded,
        "refreshing": running, "progress": prog, "progress_total": total,
        "cycles": cycles, "cycle_secs": secs, "age_min": age_min,
        "source": _UNI_META["source"], "note": _UNI_META["note"], "last_err": err,
        "ready": loaded >= max(1, int(len(UNIVERSE_SYMBOLS) * 0.6)),
    }


def screen_one(df: pd.DataFrame, mode: str, neck_days: int, vol_ratio: float):
    if df is None or len(df) < max(neck_days + 6, 25):
        return None
    c = float(df["Close"].iloc[-1])
    o = float(df["Open"].iloc[-1])
    prev_c = float(df["Close"].iloc[-2])
    v = float(df["Volume"].iloc[-1])
    ma20 = df["Close"].rolling(20, min_periods=20).mean().iloc[-1]
    vma5 = df["Volume"].shift(1).rolling(5, min_periods=3).mean().iloc[-1]
    if pd.isna(ma20) or pd.isna(vma5) or vma5 <= 0:
        return None
    ratio = v / float(vma5)
    win = df.iloc[-(neck_days + 1):-1]
    prev_win = df.iloc[-(neck_days + 2):-2]
    if mode == "LONG":
        neck = float(win["High"].max())
        prev_neck = float(prev_win["High"].max())
        ok = c > neck and c > o and c > float(ma20) and ratio >= vol_ratio
        dist = (c / neck - 1.0) * 100.0
        first_day = prev_c <= prev_neck
    else:
        neck = float(win["Low"].min())
        prev_neck = float(prev_win["Low"].min())
        ok = c < neck and c < o and c < float(ma20) and ratio >= vol_ratio
        dist = (neck / c - 1.0) * 100.0
        first_day = prev_c >= prev_neck
    if not ok:
        return None
    return {
        "price": round(c, 2),
        "pct": round((c / prev_c - 1.0) * 100.0, 2),
        "dist_neck_pct": round(dist, 1),
        "v_ratio": round(ratio, 1),
        "first_day": bool(first_day),
        "last_date": df.index[-1].strftime("%Y-%m-%d"),
    }


@app.get("/healthz")
def healthz():
    return {"ok": True, "version": APP_VERSION}


@app.get("/api/market/indices")
def get_indices():
    def _load():
        indices_map = {"dji": "^DJI", "ixic": "^IXIC", "sox": "^SOX", "spx": "^GSPC", "vix": "^VIX"}

        def _one(item):
            k, sym = item
            try:
                df = yf.Ticker(sym).history(period="7d")
                df = df.dropna(subset=["Close"])
                c = float(df["Close"].iloc[-1])
                prev_c = float(df["Close"].iloc[-2])
                chg = round(c - prev_c, 2)
                pct = round(chg / prev_c * 100.0, 2)
                if k == "vix":
                    if c >= 30:
                        status = "極度恐慌"
                    elif c >= 20:
                        status = "恐慌警戒"
                    elif c >= 15:
                        status = "正常平穩"
                    else:
                        status = "樂觀偏低"
                    return k, {"close": round(c, 2), "chg": chg, "pct": pct, "status": status}
                return k, {"close": int(round(c)), "chg": chg, "pct": pct}
            except Exception:
                return k, None

        with ThreadPoolExecutor(max_workers=5) as ex:
            res = dict(ex.map(_one, indices_map.items()))
        return res

    res = cached(("indices",), 60, _load)
    out = {k: (v if v else {"close": 0, "chg": 0.0, "pct": 0.0, "status": "—"}) for k, v in (res or {}).items()}
    for k in ("dji", "ixic", "sox", "spx", "vix"):
        out.setdefault(k, {"close": 0, "chg": 0.0, "pct": 0.0, "status": "—"})
    out["session"] = us_session_label()
    out["et_time"] = et_now().strftime("%m/%d %H:%M")
    return out


def _pool(sector: str):
    uni = list(MARKET_UNIVERSE)
    if sector == "ALL":
        return uni
    if sector == "WATCH":
        return [x for x in uni if x.get("watch")]
    return [x for x in uni if x["sector"] == sector]


@app.get("/api/universe")
def api_universe():
    """股票池概況：各產業檔數＋背景更新進度（前端用來建選單與顯示載入狀態）。"""
    uni = list(MARKET_UNIVERSE)
    counts = {}
    for x in uni:
        counts[x["sector"]] = counts.get(x["sector"], 0) + 1
    order = [k for k in SECTOR_ZH if k in counts]
    return {
        "sectors": [{"key": k, "label": SECTOR_ZH[k], "count": counts[k]} for k in order],
        "watch_count": sum(1 for x in uni if x.get("watch")),
        "status": universe_status(),
    }


@app.get("/api/screener")
def run_screener(sector: str = "ALL", mode: str = "LONG", neck_days: int = 20, vol_ratio: float = 1.2):
    mode = "SHORT" if mode == "SHORT" else "LONG"
    neck_days = max(5, min(60, int(neck_days)))
    vol_ratio = max(1.0, min(5.0, float(vol_ratio)))
    pool = _pool(sector)

    data = load_universe_daily()
    results, failed = [], 0
    for item in pool:
        df = data.get(item["symbol"])
        if df is None:
            failed += 1
            continue
        r = screen_one(df, mode, neck_days, vol_ratio)
        if r is None:
            continue
        r.update({
            "symbol": item["symbol"], "name": item["name"], "sector": item["sector"],
            "score": ("🔥 暴量突破" if r["v_ratio"] >= 2.0 else "📈 放量突破") if mode == "LONG"
            else ("⚡ 暴量破底" if r["v_ratio"] >= 2.0 else "📉 破底轉弱"),
        })
        results.append(r)
    results.sort(key=lambda x: x["v_ratio"], reverse=True)
    session = us_session_label()
    return {
        "screener_list": results,
        "scanned": len(pool) - failed,
        "failed": failed,
        "partial_volume": session == "盤中",
        "session": session,
        "universe": universe_status(),
    }


@app.get("/api/week52")
def get_week52(kind: str = "high", sector: str = "ALL", near: float = 3.0):
    """52 週新高 / 新低榜（只用真實日K計算；上市未滿一年者不納入並計數）。"""
    kind = "low" if kind == "low" else "high"
    near = max(0.5, min(10.0, float(near)))
    pool = _pool(sector)
    data = load_universe_daily()
    out, failed, insufficient = [], 0, 0
    for item in pool:
        df = data.get(item["symbol"])
        if df is None or df.empty:
            failed += 1
            continue
        w = week52_stats(df)
        if w is None:
            insufficient += 1
            continue
        if kind == "high":
            hit = w["new_high"] or w["from_high_pct"] >= -near
        else:
            hit = w["new_low"] or w["from_low_pct"] <= near
        if not hit:
            continue
        prev_c = _f(df["Close"].iloc[-2], 0.0)
        c = _f(df["Close"].iloc[-1], 0.0)
        vma5 = df["Volume"].shift(1).rolling(5, min_periods=3).mean().iloc[-1]
        ratio = _safe_round(_f(df["Volume"].iloc[-1]) / vma5, 1) if (not pd.isna(vma5) and vma5 > 0) else None
        out.append({
            "symbol": item["symbol"], "name": item["name"], "sector": item["sector"],
            "price": round(c, 2), "pct": round((c / prev_c - 1) * 100, 2) if prev_c > 0 else None,
            "v_ratio": ratio, **w,
        })
    if kind == "high":
        out.sort(key=lambda x: (not x["close_new_high"], not x["new_high"], -x["from_high_pct"]))
    else:
        out.sort(key=lambda x: (not x["close_new_low"], not x["new_low"], x["from_low_pct"]))
    session = us_session_label()
    return {"list": out, "scanned": len(pool) - failed - insufficient, "failed": failed,
            "insufficient": insufficient, "partial": session == "盤中", "session": session,
            "universe": universe_status()}


@app.get("/api/debug/{symbol}")
def debug_symbol(symbol: str):
    """診斷：直接對 Yahoo 做一次請求並回報真實錯誤，用來查「無期權合約」這類問題。"""
    sym = clean_symbol(symbol)
    out = {"yfinance": getattr(yf, "__version__", "?"), "symbol": sym, "time_et": et_now().strftime("%Y-%m-%d %H:%M")}
    try:
        df = _normalize_df(yf.Ticker(sym).history(period="5d"))
        out["history"] = {"ok": not df.empty, "rows": len(df), "last": df.index[-1].strftime("%Y-%m-%d") if not df.empty else None}
    except Exception as e:
        out["history"] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}"}
    try:
        exps = list(yf.Ticker(sym).options or [])
        out["options"] = {"ok": len(exps) > 0, "count": len(exps), "first": exps[:3]}
    except Exception as e:
        out["options"] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}"}
    # 原始請求探測：看 Yahoo / Cboe 的期權端點實際回什麼（狀態碼與開頭文字，不做任何加工）
    probes = {
        "yahoo_options_query2": f"https://query2.finance.yahoo.com/v7/finance/options/{sym}",
        "yahoo_options_query1": f"https://query1.finance.yahoo.com/v7/finance/options/{sym}",
        "cboe_delayed_options": f"https://cdn.cboe.com/api/global/delayed_quotes/options/{sym}.json",
    }
    raw = {}
    for name, url in probes.items():
        try:
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
            raw[name] = {"status": r.status_code, "bytes": len(r.content), "head": r.text[:160].replace("\n", " ")}
        except Exception as e:
            raw[name] = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
    try:
        r = requests.get(f"https://cdn.cboe.com/api/global/delayed_quotes/options/{sym}.json", headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
        j = r.json()
        raw["cboe_top_keys"] = list(j.keys())
        raw["cboe_data_keys"] = list((j.get("data") or {}).keys())[:20]
        raw["cboe_sample_options"] = ((j.get("data") or {}).get("options") or [])[100:102]
    except Exception as e:
        raw["cboe_sample_error"] = f"{type(e).__name__}: {str(e)[:120]}"
    out["raw_probe"] = raw
    out["recent_cache_errors"] = {str(k): v for k, v in list(_LAST_ERR.items())[-10:]}
    return out


def _name_for(symbol: str):
    it = UNIVERSE_BY_SYMBOL.get(symbol)
    if it and it["name"] != symbol:          # 只有真的有中文名才直接用；否則往下查英文簡稱
        return it["name"], it["market"]

    def _load():
        try:
            info = yf.Ticker(symbol).info
            return info.get("shortName") or info.get("longName") or symbol, info.get("exchange", "US")
        except Exception:
            return None

    got = cached(("name", symbol), 86400, _load)
    return got if got else (symbol, "US")


@app.get("/api/quote/{symbol}")
def get_quote_data(symbol: str, interval: str = "1d", mode: str = "LONG", neck_days: int = 20):
    try:
        sym = clean_symbol(symbol)
        if interval not in HISTORY_PLAN:
            interval = "1d"
        mode = "SHORT" if mode == "SHORT" else "LONG"
        neck_days = max(5, min(60, int(neck_days)))

        raw = fetch_history(sym, interval)
        if (raw is None or len(raw) < 3) and interval == "1d":
            raw = load_universe_daily().get(sym)
        if raw is None or len(raw) < 3:
            why = _LAST_ERR.get(("hist", sym, interval))
            raise HTTPException(status_code=404, detail=f"查無 {sym} 資料或 Yahoo 暫時無回應" + (f"（{why}）" if why else ""))

        df = add_indicators(raw)
        spot = _f(df["Close"].iloc[-1], 0.0)
        prev_spot = _f(df["Close"].iloc[-2], 0.0)
        chg = round(spot - prev_spot, 2) if prev_spot > 0 else 0.0
        pct = round(chg / prev_spot * 100.0, 2) if prev_spot > 0 else 0.0
        ma20_val = _safe_round(df["MA20"].iloc[-1], 2, None)       # 取不到就是 None，不拿現價頂替
        ma240_val = _safe_round(df["MA240"].iloc[-1], 2, None)

        neck = calc_neckline(df, neck_days, mode)
        badge, dist = quality_badge(mode, spot, neck, ma20_val)

        daily = raw if interval == "1d" else fetch_history(sym, "1d")
        w52 = week52_stats(daily)

        _, show_n = HISTORY_PLAN[interval]
        v = df.tail(show_n)
        name_str, exchange_str = _name_for(sym)

        return {
            "symbol": sym, "name": str(name_str), "exchange": str(exchange_str),
            "interval": interval, "mode": mode,
            "spot": round(spot, 2), "chg": chg, "pct": pct,
            "last_date": v.index[-1].strftime("%Y-%m-%d"),
            "ma20_val": ma20_val, "ma240_val": ma240_val,
            "neckline": neck, "neck_days": neck_days, "neck_unit": INTERVAL_UNIT[interval],
            "dist_neck": round(dist, 1), "quality_badge": badge,
            "week52": w52,
            "dates": [d.strftime("%Y-%m-%d") for d in v.index],
            "opens": _series(v["Open"]), "highs": _series(v["High"]),
            "lows": _series(v["Low"]), "closes": _series(v["Close"]),
            "volumes": [int(_f(x, 0)) for x in v["Volume"].values],
            "ma5": _series(v["MA5"]), "ma10": _series(v["MA10"]), "ma20": _series(v["MA20"]),
            "ma60": _series(v["MA60"]), "ma120": _series(v["MA120"]), "ma240": _series(v["MA240"]),
            "bb_up": _series(v["BB_UP"]), "bb_low": _series(v["BB_LOW"]),
            "vol_ma5": _series(v["Vol_MA5"], 0),
            "k": _series(v["K"], 1), "d": _series(v["D"], 1),
            "macd_hist": _series(v["MACD_HIST"]), "macd_dif": _series(v["DIF"]), "macd_dea": _series(v["DEA"]),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"後端處理異常 ({type(e).__name__}): {str(e)[:200]}")


@app.get("/api/earnings/{symbol}")
def get_next_earnings(symbol: str):
    """下一個財報日：賣方策略（Short Put / Covered Call）最需要避開的事件風險。"""
    sym = clean_symbol(symbol)

    def _load():
        try:
            cal = yf.Ticker(sym).calendar
            dates = None
            if isinstance(cal, dict):
                dates = cal.get("Earnings Date")
            elif isinstance(cal, pd.DataFrame) and "Earnings Date" in cal.index:
                dates = list(cal.loc["Earnings Date"].values)
            if not dates:
                return {"date": None}
            if not isinstance(dates, (list, tuple, np.ndarray)):
                dates = [dates]
            today = et_now().date()
            for d in dates:
                dd = pd.Timestamp(d).date()
                if dd >= today:
                    return {"date": dd.isoformat()}
            return {"date": None}
        except Exception:
            return {"date": None}

    return cached(("earn", sym), 6 * 3600, _load, cache_empty=True)


# ==============================================================================
# 選擇權：改為「先取到期日、選了再取該到期日合約」，不再一次抓全部到期日
# ==============================================================================
_OPT_SRC = {}          # sym -> "yahoo" | "cboe"，標示這檔目前用哪個資料來源
_OCC_RE = re.compile(r"^([A-Z]{1,6})(\d{6})([CP])(\d{8})$")
_SRC_LABEL = {"yahoo": "Yahoo Finance", "cboe": "Cboe 延遲報價（約延遲 15 分鐘）"}


def _num(x):
    """數字或 None；不把缺值當 0。"""
    try:
        if x is None or isinstance(x, bool):
            return None
        v = float(x)
        return None if (math.isnan(v) or math.isinf(v)) else v
    except Exception:
        return None


def _cboe_options(sym: str):
    """Cboe 公開延遲報價：回傳 {到期日: {"calls": [...], "puts": [...]}}。
    欄位缺值一律保持 None；IV 單位不確定時(>=10 視為異常)寧可不用也不猜。"""
    def _load():
        r = requests.get(f"https://cdn.cboe.com/api/global/delayed_quotes/options/{sym}.json",
                         headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
        r.raise_for_status()
        opts = ((r.json() or {}).get("data") or {}).get("options") or []
        out = {}
        for o in opts:
            m = _OCC_RE.match(str(o.get("option", "")))
            if not m or m.group(1) != sym:       # 排除調整後合約(不同 root)
                continue
            ymd = m.group(2)
            exp = f"20{ymd[0:2]}-{ymd[2:4]}-{ymd[4:6]}"
            iv = _num(o.get("iv"))
            row = {
                "strike": int(m.group(4)) / 1000.0,
                "bid": _num(o.get("bid")) or 0.0,
                "ask": _num(o.get("ask")) or 0.0,
                "last": _num(o.get("last_trade_price", o.get("last"))) or 0.0,
                "impliedVolatility": iv if (iv is not None and 0.01 < iv < 10) else None,
                "volume": int(_num(o.get("volume")) or 0),
                "openInterest": int(_num(o.get("open_interest")) or 0),
            }
            out.setdefault(exp, {"calls": [], "puts": []})["calls" if m.group(3) == "C" else "puts"].append(row)
        today = et_now().strftime("%Y-%m-%d")
        out = {e: v for e, v in out.items() if e >= today}
        for v in out.values():
            v["calls"].sort(key=lambda x: x["strike"]); v["puts"].sort(key=lambda x: x["strike"])
        if not out:
            raise ValueError("Cboe 回傳中沒有此標的的有效合約")
        return out
    return cached(("cboe", sym), 120, _load)


def _expirations(sym: str):
    """先試 Yahoo；Yahoo 回空或出錯就改用 Cboe。兩者都不是猜的，來源會回報給前端。"""
    def _load():
        yerr = None
        try:
            exps = list(yf.Ticker(sym).options or [])
            if exps:
                _OPT_SRC[sym] = "yahoo"
                return exps
        except Exception as e:
            yerr = e
        c = _cboe_options(sym)
        if c:
            _OPT_SRC[sym] = "cboe"
            return sorted(c.keys())
        if yerr:
            raise yerr
        return []
    return cached(("exps", sym), 600, _load) or []


@app.get("/api/options/{symbol}")
def get_option_expirations(symbol: str):
    sym = clean_symbol(symbol)
    exps = _expirations(sym)
    if not exps:
        why = _LAST_ERR.get(("exps", sym)) or _LAST_ERR.get(("cboe", sym))
        msg = ("取得到期日失敗：" + why) if why else "Yahoo 與 Cboe 都沒有此標的的期權合約（可能該標的沒有掛牌期權）"
        return {"expirations": [], "error": msg + f"｜可開 /api/debug/{sym} 查看原因"}
    src = _OPT_SRC.get(sym, "yahoo")
    return {"expirations": exps, "source": src, "source_label": _SRC_LABEL[src]}


@app.get("/api/options/{symbol}/chain")
def get_option_chain(symbol: str, exp: str, spot: float = 0.0):
    sym = clean_symbol(symbol)
    if exp not in (_expirations(sym) or []):
        raise HTTPException(status_code=400, detail="到期日不在此標的的合約清單內")
    src = _OPT_SRC.get(sym, "yahoo")

    def _in_range(k):
        return not (spot > 0 and not (spot * 0.5 <= k <= spot * 1.5))

    def _load():
        if src == "cboe":
            c = _cboe_options(sym) or {}
            d = c.get(exp) or {"calls": [], "puts": []}
            return {"puts": [r for r in d["puts"] if _in_range(r["strike"])],
                    "calls": [r for r in d["calls"] if _in_range(r["strike"])]}
        oc = yf.Ticker(sym).option_chain(exp)

        def _rows(frame):
            rows = []
            for _, r in frame.iterrows():
                k = _f(r.get("strike"))
                if not _in_range(k):
                    continue
                iv = _f(r.get("impliedVolatility"), 0.0)
                rows.append({
                    "strike": k, "bid": _f(r.get("bid")), "ask": _f(r.get("ask")), "last": _f(r.get("lastPrice")),
                    "impliedVolatility": iv if iv > 0.01 else None,
                    "volume": int(_f(r.get("volume"))), "openInterest": int(_f(r.get("openInterest"))),
                })
            return rows
        return {"puts": _rows(oc.puts), "calls": _rows(oc.calls)}

    key = ("chain", sym, exp, src, round(_f(spot), 0))
    data = cached(key, 120, _load)
    if not data or (not data["puts"] and not data["calls"]):
        why = _LAST_ERR.get(key) or _LAST_ERR.get(("cboe", sym))
        raise HTTPException(status_code=502, detail="取得期權鏈失敗" + (f"：{why}" if why else "，請稍後重試"))
    return {**data, "source": src, "source_label": _SRC_LABEL[src]}


@app.get("/api/greeks")
def get_greeks(S: float, K: float, T: float, sigma: float, type: str = "put"):
    S_c, K_c, T_c, sig_c = _f(S), _f(K), _f(T), _f(sigma)
    if S_c <= 0 or K_c <= 0 or T_c <= 0 or sig_c <= 0:
        raise HTTPException(status_code=400, detail="參數不完整，無法計算 Greeks")
    r, r_src = risk_free_rate()
    g = calculate_greeks(S_c, K_c, T_c, r, sig_c, "call" if type == "call" else "put")
    g["r"], g["r_source"] = r, r_src
    return g


# ==============================================================================
# LINE 推播：加入 PIN 驗證與冷卻；回傳真實 LINE 狀態
# ==============================================================================
STRATEGY_NAMES = {
    "csp": "Short Put (低接收租)", "cc": "Covered Call (現股包租公)",
    "bull_put": "Bull Put Spread", "long_call": "Long Call", "long_put": "Long Put",
    "bull_call": "Bull Call Spread", "bear_put": "Bear Put Spread", "straddle": "Long Straddle",
}
_PUSH_LOCK = threading.Lock()
_PUSH_STATE = {"last": 0.0, "fails": 0, "lock_until": 0.0}


class PushPayload(BaseModel):
    symbol: str = Field(default="SPY", max_length=12)
    strategy: str = Field(default="csp", max_length=20)
    mode: str = Field(default="LONG", max_length=5)
    neck_days: int = Field(default=20, ge=5, le=60)
    summary: str = Field(default="", max_length=500)


@app.post("/api/line/push")
def line_push(payload: PushPayload, x_push_pin: str = Header(default="")):
    if not ADMIN_PUSH_PIN:
        raise HTTPException(status_code=403, detail="伺服器尚未設定 ADMIN_PUSH_PIN，推播功能已停用（避免任何人都能觸發你的 LINE）")
    now = time.time()
    with _PUSH_LOCK:
        if now < _PUSH_STATE["lock_until"]:
            raise HTTPException(status_code=429, detail=f"錯誤次數過多，請 {int(_PUSH_STATE['lock_until'] - now)} 秒後再試")
        if not hmac.compare_digest(x_push_pin.encode("utf-8"), ADMIN_PUSH_PIN.encode("utf-8")):
            _PUSH_STATE["fails"] += 1
            if _PUSH_STATE["fails"] >= 5:
                _PUSH_STATE["lock_until"] = now + 300
                _PUSH_STATE["fails"] = 0
            raise HTTPException(status_code=401, detail="通行碼錯誤")
        _PUSH_STATE["fails"] = 0
        if now - _PUSH_STATE["last"] < 30:
            raise HTTPException(status_code=429, detail="請等待 30 秒後再推播")
        _PUSH_STATE["last"] = now

    sym = clean_symbol(payload.symbol)
    mode = "SHORT" if payload.mode == "SHORT" else "LONG"
    raw = fetch_history(sym, "1d")
    if raw is None or len(raw) < 5:
        raw = load_universe_daily().get(sym)
    if raw is None or len(raw) < 5:
        raise HTTPException(status_code=502, detail="取得行情失敗，未推播")
    df = add_indicators(raw)
    spot = _f(df["Close"].iloc[-1])
    prev = _f(df["Close"].iloc[-2])
    pct = _f((spot / prev - 1.0) * 100.0) if prev > 0 else None
    ma20 = _safe_round(df["MA20"].iloc[-1], 2, None)
    neck = calc_neckline(df, payload.neck_days, mode)
    badge, dist = quality_badge(mode, spot, neck, ma20)
    w52 = week52_stats(raw)
    arrow = "▲" if (pct or 0) >= 0 else "▼"
    pct_txt = f"({arrow}{abs(pct):.2f}%)" if pct is not None else "(漲跌未知)"
    neck_txt = f"${neck:.2f}（{dist:+.1f}%）" if neck is not None else "資料不足"
    if ma20 is None:
        ma_txt = "月線資料不足（未滿 20 根）"
    else:
        ma_txt = f"月線 MA20 ${ma20:.2f}（現價在{'月線之上' if spot >= ma20 else '月線之下'}）"
    if w52:
        flag = "｜🏔️ 收盤創52週新高" if w52["close_new_high"] else ("｜🏔️ 盤中創52週新高" if w52["new_high"] else "")
        flag = flag or ("｜🕳️ 創52週新低" if w52["new_low"] else "")
        w52_txt = f"🏔️ 52週 高 ${w52['high']} ({w52['from_high_pct']:+.1f}%)／低 ${w52['low']}{flag}\n"
    else:
        w52_txt = "🏔️ 52週高低：上市未滿一年，資料不足\n"
    summary = re.sub(r"[\r\x00-\x08]", "", payload.summary).strip()

    text = (
        f"📈【美股多空雷達 {APP_VERSION}｜個股戰報】\n"
        f"🕒 {tpe_now().strftime('%Y-%m-%d %H:%M')} (台北) / {et_now().strftime('%H:%M')} (美東)\n"
        f"🌡️ {sym} 現價 ${spot:.2f} {pct_txt}\n"
        f"📏 距{payload.neck_days}日{'頸線' if mode == 'LONG' else '跌破線'}：{neck_txt}\n"
        f"🧭 {ma_txt}\n"
        + w52_txt +
        f"🛡️ 品質：{badge}\n"
        f"⚡ 策略：{STRATEGY_NAMES.get(payload.strategy, payload.strategy)}\n"
        + (f"🧮 {summary}\n" if summary else "")
        + "━━━━━━━━━━━━━━\n※ 僅為量化條件篩選結果，非投資建議。"
    )

    line_status = "not_configured"
    if LINE_CHANNEL_ACCESS_TOKEN and LINE_USER_ID:
        try:
            r = requests.post(
                "https://api.line.me/v2/bot/message/push",
                headers={"Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}", "Content-Type": "application/json"},
                json={"to": LINE_USER_ID, "messages": [{"type": "text", "text": text}]},
                timeout=8,
            )
            line_status = str(r.status_code)
        except Exception as e:
            line_status = f"error:{type(e).__name__}"
    return {"status": "success" if line_status == "200" else "failed", "line_status": line_status, "preview_text": text}


# ==============================================================================
# 前端（單頁）。V9.9：選擇權逐腿計價、Plotly Payoff、畫線保留、偏好記憶、錯誤可見
# ==============================================================================
HTML_CONTENT = r"""
<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>美股多空雷達戰情室 V10.1</title>
<script src="https://cdn.tailwindcss.com"></script>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
:root{--bg-main:#0b0e14;--bg-card:#121824;--border-col:#1e293b;--text-main:#e2e8f0;--text-muted:#94a3b8;--input-bg:#0b0e14;}
.theme-light{--bg-main:#f1f5f9;--bg-card:#ffffff;--border-col:#cbd5e1;--text-main:#0f172a;--text-muted:#475569;--input-bg:#f8fafc;}
body{background-color:var(--bg-main);color:var(--text-main);}
.bg-card{background-color:var(--bg-card);} .border-theme{border-color:var(--border-col);}
.text-muted{color:var(--text-muted);} .bg-input{background-color:var(--input-bg);}
#leftSidebar{width:clamp(270px,21vw,384px)!important;}
#rightSidebar{width:clamp(310px,25vw,461px)!important;}
/* 平板／小筆電（821–1280px）：左＝選股清單（固定在視窗內可捲動）；右＝上圖表、下選擇權。清單先看、圖表居中，與桌機順序一致 */
@media (min-width:821px) and (max-width:1280px){
 #mainLayout{display:grid!important;grid-template-columns:clamp(250px,27vw,320px) minmax(0,1fr);grid-template-rows:auto auto;align-content:start;height:auto!important;overflow-y:auto!important;}
 #leftSidebar{grid-column:1;grid-row:1 / span 2;width:auto!important;height:calc(100vh - 8px)!important;max-height:none!important;position:sticky;top:0;align-self:start;border-right:1px solid var(--border-col)!important;}
 #centerMain{grid-column:2;grid-row:1;width:auto!important;height:auto!important;min-width:0;}
 #rightSidebar{grid-column:2;grid-row:2;width:auto!important;height:auto!important;max-height:none!important;border-left:none!important;border-top:1px solid var(--border-col)!important;}
 body{overflow-y:auto!important;height:auto!important;}
}
/* 手機／窄螢幕（≤820px）：全部直向堆疊 */
@media (max-width:820px){
 #mainLayout{flex-direction:column!important;flex-wrap:nowrap!important;}
 #leftSidebar,#rightSidebar,#centerMain{width:100%!important;height:auto!important;border:none!important;border-bottom:1px solid var(--border-col)!important;}
 #centerMain{order:1!important;} #rightSidebar{order:2!important;max-height:none!important;} #leftSidebar{order:3!important;max-height:480px!important;}
}
#optToggle,#optBackdrop{display:none;}
@media (max-width:1280px){.optClose{display:inline-block!important;}}
/* ≤1280px：選擇權面板改成右側滑出抽屜，預設收起；點右側「選擇權」分頁或按 ✕／點背景關閉 */
@media (max-width:1280px){
 #rightSidebar{position:fixed!important;top:0!important;right:0!important;bottom:0!important;height:100vh!important;max-height:none!important;
   width:min(480px,92vw)!important;transform:translateX(105%);transition:transform .25s ease;z-index:60;border:none!important;
   border-left:1px solid var(--border-col)!important;box-shadow:-8px 0 24px rgba(0,0,0,.45);overflow-y:auto!important;order:0!important;}
 #rightSidebar.open{transform:translateX(0);}
 #optToggle{display:flex;position:fixed;right:0;top:45%;z-index:55;writing-mode:vertical-rl;align-items:center;gap:4px;
   background:#38bdf8;color:#0b0e14;font-weight:900;padding:14px 8px;border-radius:10px 0 0 10px;box-shadow:-3px 2px 10px rgba(0,0,0,.4);cursor:pointer;border:none;font-size:0.95rem;}
 #optBackdrop.show{display:block;position:fixed;inset:0;background:rgba(0,0,0,.5);z-index:58;}
 #centerMain{grid-row:1 / span 2!important;padding-right:44px!important;}  /* 讓出右緣給「選擇權」分頁，不蓋住價格軸 */
}
@media (max-width:1024px){
 .grid-indices{grid-template-columns:repeat(2,minmax(0,1fr))!important;}
 .grid-indices>div:last-child{grid-column:span 2/span 2!important;}
 .grid-indices>div{flex-wrap:wrap;gap:2px 6px;}
 .grid-indices>div>span:first-child{white-space:nowrap;}
 .grid-indices{font-size:0.82rem;}
}
.pill-active{background-color:#234E70!important;color:#fff!important;font-weight:700!important;}
.pill-inactive{background-color:var(--bg-card)!important;color:#64748b!important;}
select,input{color:var(--text-main);}
.tool-btn{padding:2px 8px;border-radius:6px;border:1px solid var(--border-col);background:var(--input-bg);font-weight:700;}
</style>
</head>
<body id="bodyRoot" class="font-sans antialiased min-h-screen flex flex-col">
<header class="bg-card border-b border-theme px-3.5 py-2 flex flex-col gap-1.5 shadow-lg shrink-0">
  <div class="flex justify-between items-center flex-wrap gap-2">
    <div class="flex items-center flex-wrap gap-2">
      <span class="text-lg md:text-xl font-black text-[#38bdf8] tracking-wider">⚡ 美股多空雷達 V10.1</span>
      <span class="text-xs bg-[#1e293b] text-amber-400 px-2 py-0.5 rounded-full font-bold border border-amber-500/30">真實數據版</span>
      <span id="sessionTag" class="text-xs px-2 py-0.5 rounded-full border border-theme font-bold text-muted">--</span>
      <button onclick="toggleSettingsDrawer()" class="text-xs bg-card px-2.5 py-0.5 rounded border border-theme font-bold">⚙️ 設定</button>
      <button onclick="openChangeLogModal()" class="text-xs bg-card text-amber-500 px-2 py-0.5 rounded border border-amber-500/40 font-bold">📜 改版紀錄</button>
    </div>
    <button onclick="fetchIndicesData()" class="bg-[#1e293b] text-white px-2 py-1 rounded border border-[#38bdf8]/40 text-xs font-bold">🔄 刷新溫度計</button>
  </div>
  <div class="grid grid-cols-5 grid-indices gap-1.5 text-sm font-mono">
    <div class="bg-input border border-theme px-2 py-1.5 rounded flex justify-between items-center"><span class="text-muted font-bold">🏛️ 道瓊</span><span id="idx_dji" class="font-bold">--</span></div>
    <div class="bg-input border border-theme px-2 py-1.5 rounded flex justify-between items-center"><span class="text-muted font-bold">💻 那指</span><span id="idx_ixic" class="font-bold">--</span></div>
    <div class="bg-input border border-theme px-2 py-1.5 rounded flex justify-between items-center"><span class="text-muted font-bold">⚡ 費半</span><span id="idx_sox" class="font-bold">--</span></div>
    <div class="bg-input border border-theme px-2 py-1.5 rounded flex justify-between items-center"><span class="text-muted font-bold">📊 標普</span><span id="idx_spx" class="font-bold">--</span></div>
    <div class="bg-input border border-theme px-2 py-1.5 rounded flex justify-between items-center"><span class="text-amber-500 font-bold">🔥 恐慌</span><span id="idx_vix" class="font-bold text-amber-500">--</span></div>
  </div>
  <div id="statusBar" class="hidden text-xs px-2 py-1 rounded border"></div>
  <div id="settingsDrawer" class="hidden bg-input border border-theme rounded-lg p-3 mt-1.5 text-sm space-y-3">
    <div class="flex flex-wrap items-center justify-between gap-3 border-b border-theme pb-2">
      <div class="flex items-center gap-3">
        <span class="font-bold">🎨 主題：</span>
        <label class="cursor-pointer flex items-center gap-1.5"><input type="radio" name="themeRadio" value="dark" checked onchange="switchTheme('dark')"><span class="font-bold">深色</span></label>
        <label class="cursor-pointer flex items-center gap-1.5"><input type="radio" name="themeRadio" value="light" onchange="switchTheme('light')"><span class="font-bold text-amber-500">淺色(護眼米灰)</span></label>
      </div>
      <div class="flex items-center gap-2">
        <span class="font-bold">👓 字體：</span>
        <button onclick="setFontSize('normal')" id="btn_font_normal" class="px-2.5 py-1 rounded bg-card border border-theme">標準</button>
        <button onclick="setFontSize('large')" id="btn_font_large" class="px-2.5 py-1 rounded bg-card border border-theme">大字(推薦)</button>
        <button onclick="setFontSize('xlarge')" id="btn_font_xlarge" class="px-2.5 py-1 rounded bg-card border border-theme">特大字</button>
      </div>
    </div>
    <div class="flex flex-wrap items-center gap-4 text-muted">
      <div class="flex items-center gap-2"><span>頸線天數 (N)：</span>
        <input type="number" id="cfg_neckDays" value="20" min="5" max="60" class="w-16 bg-card border border-theme rounded px-1.5 py-0.5 text-center" onchange="onParamChange()"></div>
      <div class="flex items-center gap-2"><span>5日均量倍數：</span>
        <input type="number" id="cfg_volRatio" value="1.2" step="0.1" min="1.0" max="5.0" class="w-16 bg-card border border-theme rounded px-1.5 py-0.5 text-center" onchange="onParamChange()"></div>
      <span>※ 頸線天數同時套用於左側篩選與中間 K 線圖</span>
    </div>
  </div>
</header>

<button id="optToggle" onclick="toggleOptions()">📊 選擇權風控 ◀</button>
<div id="optBackdrop" onclick="toggleOptions(false)"></div>
<div id="mainLayout" class="flex flex-1 overflow-hidden">
  <aside id="leftSidebar" class="w-80 bg-card border-r border-theme flex flex-col shrink-0">
    <div class="p-3 border-b border-theme space-y-2">
      <div class="grid grid-cols-3 gap-1 text-xs font-bold">
        <button id="tab_break" onclick="setListTab('break')" class="py-1.5 rounded bg-[#234E70] text-white">🚀 突破榜</button>
        <button id="tab_high" onclick="setListTab('high')" class="py-1.5 rounded bg-input text-muted">🏔️ 52週新高</button>
        <button id="tab_low" onclick="setListTab('low')" class="py-1.5 rounded bg-input text-muted">🕳️ 52週新低</button>
      </div>
      <div id="breakModeBox">
        <label class="text-xs text-muted font-bold block mb-1">🎯 選股多空方向：</label>
        <div class="grid grid-cols-2 gap-1.5">
          <button onclick="setRadarMode('LONG')" id="mode_long" class="py-1 rounded text-xs font-bold bg-[#C0392B] text-white">📈 多方突破</button>
          <button onclick="setRadarMode('SHORT')" id="mode_short" class="py-1 rounded text-xs font-bold bg-input text-muted">📉 空方破底</button>
        </div>
      </div>
      <div>
        <label class="text-xs text-muted font-bold block mb-1">🏛️ 市場板塊：</label>
        <select id="sectorSelect" onchange="onSectorChange()" class="w-full bg-input border border-theme rounded px-2.5 py-1.5 text-sm text-[#38bdf8] font-bold">
          <option value="ALL">🌐 全部（載入中…）</option>
          <option value="WATCH">⭐ 我的觀察名單</option>
        </select>
      </div>
      <div class="flex gap-2">
        <input id="customSymbol" type="text" placeholder="任意美股代號 (NVDA, IONQ...)" class="flex-1 bg-input border border-theme rounded px-3 py-1.5 text-sm uppercase" onkeydown="if(event.key==='Enter')addCustomSymbol()">
        <button onclick="addCustomSymbol()" class="bg-[#38bdf8] text-slate-950 font-black px-3 py-1.5 rounded text-sm">診斷</button>
      </div>
      <div id="screenerNote" class="text-xs text-muted"></div>
    </div>
    <div id="watchlist" class="flex-1 overflow-y-auto p-2 space-y-1.5"></div>
  </aside>

  <main id="centerMain" class="flex-1 flex flex-col p-3 overflow-y-auto">
    <div class="flex flex-wrap justify-between items-center gap-2 mb-1.5">
      <div class="flex items-center gap-2 flex-wrap">
        <h1 id="chartTitle" class="text-base md:text-xl font-black tracking-wide">載入中...</h1>
        <div class="flex bg-card border border-theme rounded p-0.5 gap-1 text-xs">
          <button onclick="changeInterval('1d')" id="btn_1d" class="px-2.5 py-0.5 rounded font-bold bg-[#38bdf8] text-slate-950">日K</button>
          <button onclick="changeInterval('1wk')" id="btn_1wk" class="px-2.5 py-0.5 rounded font-bold text-muted">週K</button>
          <button onclick="changeInterval('1mo')" id="btn_1mo" class="px-2.5 py-0.5 rounded font-bold text-muted">月K</button>
        </div>
      </div>
      <div class="flex items-center gap-2 flex-wrap">
        <div id="earningsTag" class="hidden text-xs bg-amber-500/20 text-amber-500 px-2 py-0.5 rounded border border-amber-500/40 font-bold"></div>
        <div id="w52Tag" class="hidden text-xs bg-sky-500/15 text-sky-400 px-2 py-0.5 rounded border border-sky-500/40 font-bold"></div>
        <div id="qualityTag" class="text-xs bg-emerald-500/20 text-emerald-500 px-2 py-0.5 rounded border border-emerald-500/40 font-bold">—</div>
        <div id="currentPriceTag" class="text-sm md:text-base font-bold bg-card px-3 py-1 rounded-lg border border-theme shadow">收盤: $-- (--%)</div>
      </div>
    </div>

    <div class="flex flex-wrap items-center gap-1.5 mb-1.5 text-xs">
      <span class="text-muted font-bold mr-1">📐 均線/指標:</span>
      <button onclick="toggleIndicator('ma5')" id="pill_ma5" class="pill-active px-2 py-0.5 rounded border border-[#D97706]">5MA</button>
      <button onclick="toggleIndicator('ma10')" id="pill_ma10" class="pill-inactive px-2 py-0.5 rounded border border-[#2563EB]">10MA</button>
      <button onclick="toggleIndicator('ma20')" id="pill_ma20" class="pill-active px-2 py-0.5 rounded border border-[#7C3AED]">20MA</button>
      <button onclick="toggleIndicator('ma60')" id="pill_ma60" class="pill-active px-2 py-0.5 rounded border border-[#0D9488]">60MA</button>
      <button onclick="toggleIndicator('ma120')" id="pill_ma120" class="pill-inactive px-2 py-0.5 rounded border border-[#DB2777]">120MA</button>
      <button onclick="toggleIndicator('ma240')" id="pill_ma240" class="pill-active px-2 py-0.5 rounded border border-[#64748b]">240MA(年線)</button>
      <button onclick="toggleIndicator('bb')" id="pill_bb" class="pill-inactive px-2 py-0.5 rounded border border-[#64748b]">布林</button>
      <button onclick="toggleIndicator('w52')" id="pill_w52" class="pill-active px-2 py-0.5 rounded border border-[#0EA5E9]">52週高低</button>
      <button onclick="toggleIndicator('neck')" id="pill_neck" class="pill-active px-2 py-0.5 rounded border border-[#B45309]"><span id="neckPillText">頸線</span></button>
    </div>

    <div class="flex flex-wrap justify-center items-center bg-card border border-theme py-1.5 px-3 rounded-lg text-sm mb-2 gap-2 shadow">
      <span class="font-bold text-[#38bdf8] mr-1">🛠️ 畫線:</span>
      <button class="tool-btn" onclick="setPlotlyMode('drawline')">📏 直線</button>
      <button class="tool-btn" onclick="setPlotlyMode('drawrect')">📦 方框</button>
      <button class="tool-btn" onclick="setPlotlyMode('drawcircle')">⭕ 圓形</button>
      <span class="text-muted">|</span>
      <button class="tool-btn text-red-500" onclick="setPlotlyMode('eraseshape')">🧹 點選刪除</button>
      <button class="tool-btn text-red-500" onclick="clearAllShapes()">🗑️ 全部清空</button>
      <span class="text-muted">|</span>
      <button class="tool-btn" onclick="setPlotlyMode('pan')">✋ 平移(預設)</button>
      <button class="tool-btn" onclick="setPlotlyMode('zoom')">🔍 框選放大</button>
      <button class="tool-btn" onclick="resetPlotlyAxes()">🏠 還原</button>
      <span id="touchHint" class="hidden text-xs text-muted">觸控裝置預設鎖定圖表以便捲動頁面，點上方按鈕即可操作</span>
    </div>

    <div id="plotlyChartContainer" class="bg-card border border-theme rounded-xl p-2 w-full h-[620px] md:h-[680px]"></div>
  </main>

  <aside id="rightSidebar" class="w-96 bg-card border-l border-theme flex flex-col p-4 shrink-0 overflow-y-auto">
    <h2 class="text-sm font-black mb-2.5 flex items-center justify-between">
      <span>📊 選擇權策略風控</span>
      <span class="flex items-center gap-2">
        <span class="text-xs bg-emerald-500 text-slate-950 font-black px-2 py-0.5 rounded">逐腿真實報價</span>
        <button onclick="toggleOptions(false)" class="optClose text-muted text-lg px-1" style="display:none">✕</button>
      </span>
    </h2>
    <div class="space-y-2.5 text-sm">
      <div class="grid grid-cols-1 md:grid-cols-2 gap-2">
        <div>
          <label class="text-muted block mb-1 text-xs">到期日</label>
          <select id="expDateSelect" onchange="onExpirationChange()" class="w-full bg-input border border-theme rounded p-2 font-bold"><option value="">載入中...</option></select>
        </div>
        <div>
          <label class="text-muted block mb-1 text-xs">策略 (8大主流)</label>
          <select id="strategySelect" onchange="onStrategyChange()" class="w-full bg-input border border-theme rounded p-2 font-bold">
            <optgroup label="💰 賣方穩健收租">
              <option value="csp" selected>Short Put (低接收租)</option>
              <option value="cc">Covered Call (現股包租公)</option>
              <option value="bull_put">Bull Put Spread (賣權多頭價差)</option>
            </optgroup>
            <optgroup label="🚀 買方方向波段">
              <option value="long_call">Long Call</option>
              <option value="long_put">Long Put</option>
              <option value="bull_call">Bull Call Spread</option>
              <option value="bear_put">Bear Put Spread</option>
            </optgroup>
            <optgroup label="⚡ 波動度交易">
              <option value="straddle">Long Straddle</option>
            </optgroup>
          </select>
        </div>
      </div>

      <div class="bg-input p-2.5 rounded-lg border border-theme space-y-1">
        <div class="flex justify-between"><span class="text-muted">現價 / 20MA</span><span id="spotDisplay" class="font-bold">-</span></div>
        <div class="flex justify-between"><span class="text-muted">停損參考</span><span id="stopLossDisplay" class="text-amber-500 font-bold">-</span></div>
      </div>

      <div class="bg-input p-2.5 rounded-lg border border-theme space-y-2">
        <div>
          <div class="flex justify-between items-center mb-1">
            <label class="text-muted text-xs" id="strike1Label">履約價 (Strike)</label>
            <span class="text-xs text-[#38bdf8] cursor-pointer hover:underline" onclick="autoAlignMA20()">🎯 對齊 MA20</span>
          </div>
          <select id="strikeSelect" onchange="recompute()" class="w-full bg-card border border-theme rounded p-2 font-bold"><option value="">請先選到期日</option></select>
        </div>
        <div id="strike2Wrap" class="hidden">
          <label class="text-muted text-xs block mb-1" id="strike2Label">第二腿履約價</label>
          <select id="strike2Select" onchange="recompute()" class="w-full bg-card border border-theme rounded p-2 font-bold"></select>
        </div>
        <div id="chainNote" class="text-xs text-amber-500"></div>
        <div id="chainSrc" class="text-xs text-muted"></div>
        <div id="chainDte" class="text-xs text-amber-500"></div>
        <button id="retryChainBtn" onclick="onExpirationChange()" class="hidden w-full text-xs py-1 rounded border border-amber-500/50 text-amber-500 font-bold">↻ 重試取得期權鏈</button>
      </div>

      <div class="bg-input p-2.5 rounded-lg border border-theme space-y-1">
        <div class="font-bold text-[#38bdf8] text-xs flex justify-between"><span>📈 到期損益 (每口 100 股)</span><span id="netPremium" class="text-muted">--</span></div>
        <div class="grid grid-cols-2 gap-2 text-xs bg-card p-1.5 rounded border border-theme">
          <div>最大獲利: <strong id="payoffMax" class="text-emerald-500">--</strong></div>
          <div>最大虧損: <strong id="payoffMin" class="text-red-500">--</strong></div>
          <div class="col-span-2">損益兩平: <strong id="payoffBE">--</strong></div>
        </div>
        <div id="payoffChart" class="w-full" style="height:190px;"></div>
      </div>

      <div class="bg-input p-2.5 rounded-lg border border-theme space-y-1">
        <div class="font-bold text-[#38bdf8] border-b border-theme pb-0.5 text-xs">⚡ Greeks（各腿加總，每股）</div>
        <div class="grid grid-cols-2 gap-2 text-sm">
          <div>Delta: <strong id="valDelta">-</strong></div><div>Theta: <strong id="valTheta" class="text-amber-500">-</strong></div>
          <div>Gamma: <strong id="valGamma">-</strong></div><div>Vega: <strong id="valVega">-</strong></div>
        </div>
      </div>

      <div id="riskBox" class="hidden bg-input p-2.5 rounded-lg border border-amber-500/40 text-xs whitespace-pre-wrap leading-relaxed"></div>

      <div class="grid grid-cols-2 gap-2 pt-1">
        <button onclick="calculateOptionStrategy()" class="bg-[#38bdf8] text-slate-950 font-black py-2 rounded-lg shadow text-sm">風控檢核</button>
        <button onclick="sendCloudPush()" class="bg-[#06C755] text-white font-black py-2 rounded-lg shadow text-sm">📲 LINE 推播</button>
      </div>
      <div class="text-xs text-muted leading-relaxed">※ 買進以 Ask、賣出以 Bid 計價（較保守）；盤後無報價時改用最後成交價並提示。損益為「到期日」理論值，未含手續費、提前履約與保證金。</div>
    </div>
  </aside>
</div>

<div id="changeLogModal" class="fixed inset-0 bg-black/80 flex items-center justify-center hidden z-50 p-4">
  <div class="bg-card border border-theme rounded-2xl p-6 w-full max-w-xl shadow-2xl space-y-4">
    <div class="flex justify-between items-center border-b border-theme pb-2">
      <h3 class="text-base font-black">📜 美股多空雷達戰情室 ｜ 版本紀錄</h3>
      <button onclick="closeChangeLogModal()" class="text-muted text-base">✕</button>
    </div>
    <div class="space-y-3 text-sm max-h-[420px] overflow-y-auto pr-2 text-muted">
      <div class="border-l-2 border-slate-600 pl-3 py-1">
        <div class="font-bold text-base" style="color:var(--text-main)">V10.1 擴大股票池 (2026-10-18)</div>
        <ul class="list-disc list-inside mt-1 space-y-0.5">
          <li>掃描範圍由 43 檔擴大到標普500＋那斯達克100（約 500 檔），原 43 檔保留為「我的觀察名單」。</li>
          <li>板塊選單改為 11 大 GICS 產業＋ETF；每檔標示所屬產業，並顯示各產業檔數。</li>
          <li>日K改由背景程式分批下載並存放（每 10 分鐘盤中／1 小時盤後），請求時直接讀取，不再於點選當下才抓而逾時。</li>
          <li>啟動初期資料尚在載入時會明確顯示進度，而不是顯示不完整的結果。</li>
        </ul>
      </div>
      <div class="border-l-2 border-slate-600 pl-3 py-1">
        <div class="font-bold text-base" style="color:var(--text-main)">V10.0 真實數據版 (2026-10-18)</div>
        <ul class="list-disc list-inside mt-1 space-y-0.5">
          <li>修正圖表載入失敗(HTTP 500)：快取判斷誤用 DataFrame 真值，已修。</li>
          <li>新增 52 週新高／新低榜、個股 52 週區間與位置、圖上 52 週高低虛線；上市未滿一年者顯示「資料不足」而非亂填。</li>
          <li>全面移除假資料預設值：缺 IV 時 Greeks 顯示「—」(不再假設 35%)；均線不足期數不顯示；月線取不到不再用現價頂替。</li>
          <li>Greeks 無風險利率改用 13 週美國國庫券殖利率(^IRX)，取不到時明確標示為預設假設。</li>
          <li>LINE 推播：必須設定通行碼才啟用、錯誤 5 次鎖 5 分鐘、冷卻 30 秒，只在 LINE 回 200 才顯示成功。</li>
          <li>新增 /api/debug/代號：直接回報 Yahoo 期權／行情請求的真實錯誤，用來查「無期權合約」。</li>
          <li>x 軸日期移到最下方；手機指數列不再斷字。</li>
        </ul>
      </div>
      <div class="border-l-2 border-slate-600 pl-3 py-1">
        <div class="font-bold text-base" style="color:var(--text-main)">V9.9 風控強化版 (2026-10-18)</div>
        <ul class="list-disc list-inside mt-1 space-y-0.5">
          <li>修正均線暖機：年線/半年線不再以不足期數的平均值偽造，改抓更長歷史計算。</li>
          <li>頸線依多空方向計算（空方用N日最低），並與設定的天數同步到 K 線圖；品質標籤依方向判斷。</li>
          <li>選擇權改為逐腿真實計價：價差、跨式可選第二履約價；Payoff 與損益兩平以真實報價計算；Greeks 為各腿加總。</li>
          <li>期權鏈改為選到期日才抓取並加快取，大幅降低 Yahoo 限流造成的「無合約」。</li>
          <li>篩選器批次下載＋5分鐘快取，並修正全市場只掃前 30 檔的問題。</li>
          <li>新增財報日提醒（財報在到期日前時警示）。</li>
          <li>LINE 推播加入通行碼、冷卻與真實送出狀態；推播時間改為台北/美東。</li>
          <li>切換指標/週期時保留已畫的線；主題與字體記憶；錯誤訊息可見。</li>
        </ul>
      </div>
      <div class="border-l-2 border-slate-600 pl-3 py-1">
        <div class="font-bold">V9.8 旗艦完全體 (2026-10-17)</div>
        <ul class="list-disc list-inside mt-1"><li>月K、Plotly 四層圖、中文畫線工具、8 大選擇權策略。</li></ul>
      </div>
    </div>
    <button onclick="closeChangeLogModal()" class="w-full bg-[#38bdf8] text-slate-950 font-black py-2 rounded-lg text-sm">關 閉</button>
  </div>
</div>

<div id="lineModal" class="fixed inset-0 bg-black/75 flex items-center justify-center hidden z-50 p-4">
  <div class="bg-card border border-theme rounded-2xl p-5 w-full max-w-md shadow-2xl space-y-3">
    <div class="flex justify-between items-center border-b border-theme pb-2">
      <h3 class="text-sm font-black">📲 LINE 戰報</h3>
      <button onclick="closeLineModal()" class="text-muted text-sm">✕</button>
    </div>
    <div id="lineResult" class="text-sm font-bold"></div>
    <div class="bg-input p-3.5 rounded-xl border border-theme text-xs font-mono text-muted whitespace-pre-wrap max-h-[300px] overflow-y-auto" id="linePreviewText"></div>
    <button onclick="closeLineModal()" class="w-full bg-[#38bdf8] text-slate-950 font-black py-2 rounded-lg text-sm">確 認</button>
  </div>
</div>

<script>
const isTouch = window.matchMedia('(pointer: coarse)').matches;
let currentSpot = 0, currentMA20 = 0, currentNeckline = null;
let currentSymbol = "PLTR", currentInterval = "1d";
let currentSector = "ALL", currentRadarMode = "LONG";
let universeData = [], currentRawData = null;
let chainCache = {}, currentChain = null, currentExpirations = [];
let lastRender = {symbol: null, interval: null};
let nextEarnings = null;
let listTab = 'break';
let indState = {ma5:true, ma10:false, ma20:true, ma60:true, ma120:false, ma240:true, bb:false, neck:true, w52:true};

const esc = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt$ = v => (v < 0 ? '-$' : '+$') + Math.abs(v).toLocaleString(undefined, {maximumFractionDigits: 0});

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    let m = 'HTTP ' + r.status;
    try { const j = await r.json(); if (j.detail) m = j.detail; } catch (e) {}
    throw new Error(m);
  }
  return r.json();
}
function setStatus(msg, isErr) {
  const el = document.getElementById('statusBar');
  if (!msg) { el.classList.add('hidden'); return; }
  el.className = 'text-xs px-2 py-1 rounded border ' + (isErr ? 'border-red-500/50 text-red-400 bg-red-500/10' : 'border-emerald-500/40 text-emerald-400 bg-emerald-500/10');
  el.textContent = msg;
  if (!isErr) setTimeout(() => { if (el.textContent === msg) el.classList.add('hidden'); }, 4000);
}
function neckDays() { return Math.max(5, Math.min(60, parseInt(document.getElementById('cfg_neckDays').value) || 20)); }
function volRatio() { return Math.max(1, Math.min(5, parseFloat(document.getElementById('cfg_volRatio').value) || 1.2)); }

function toggleSettingsDrawer() { document.getElementById('settingsDrawer').classList.toggle('hidden'); }
function switchTheme(theme) {
  document.getElementById('bodyRoot').classList.toggle('theme-light', theme === 'light');
  try { localStorage.setItem('us_theme', theme); } catch (e) {}
  document.querySelectorAll('input[name=themeRadio]').forEach(r => r.checked = (r.value === theme));
  if (currentRawData) renderPlotlyChart(currentRawData);
}
function setFontSize(size) {
  ['normal','large','xlarge'].forEach(s => {
    const b = document.getElementById('btn_font_' + s);
    if (b) b.className = 'px-2.5 py-1 rounded bg-card border ' + (s === size ? 'border-[#38bdf8] font-bold' : 'border-theme');
  });
  document.documentElement.style.fontSize = size === 'large' ? '19.2px' : (size === 'xlarge' ? '22.4px' : '16px');
  try { localStorage.setItem('us_font', size); } catch (e) {}
}
function toggleOptions(force) {
  const d = document.getElementById('rightSidebar'), bd = document.getElementById('optBackdrop');
  const open = typeof force === 'boolean' ? force : !d.classList.contains('open');
  d.classList.toggle('open', open);
  bd.classList.toggle('show', open);
  if (open) { setTimeout(() => { try { Plotly.Plots.resize('payoffChart'); } catch (e) {} }, 300); }
}
function openChangeLogModal() { document.getElementById('changeLogModal').classList.remove('hidden'); }
function closeChangeLogModal() { document.getElementById('changeLogModal').classList.add('hidden'); }
function closeLineModal() { document.getElementById('lineModal').classList.add('hidden'); }

function setRadarMode(mode) {
  currentRadarMode = mode;
  document.getElementById('mode_long').className = 'py-1 rounded text-xs font-bold ' + (mode === 'LONG' ? 'bg-[#C0392B] text-white' : 'bg-input text-muted');
  document.getElementById('mode_short').className = 'py-1 rounded text-xs font-bold ' + (mode === 'SHORT' ? 'bg-[#1E7E5A] text-white' : 'bg-input text-muted');
  loadScreenerList();
  fetchStockData(currentSymbol);
}
function onParamChange() { loadScreenerList(); fetchStockData(currentSymbol); }
function onSectorChange() { currentSector = document.getElementById('sectorSelect').value; loadScreenerList(); }

let sectorLabels = {};
let uniRetry = null;
function sectorTag(k) { return (sectorLabels[k] || k || '').replace(/^\S+\s/, ''); }
function uniText(u) {
  if (!u) return '';
  let t = `股票池 ${u.total} 檔`;
  if (u.age_min != null) t += `｜更新於 ${u.age_min < 1 ? '剛剛' : u.age_min + ' 分鐘前'}`;
  if (u.refreshing) t += `｜更新中 ${u.progress}/${u.progress_total}`;
  return t;
}
function uniTip(u) { return u ? `來源：${u.source}${u.note ? '｜' + u.note : ''}${u.last_err ? '｜最近錯誤：' + u.last_err : ''}` : ''; }
async function loadUniverseMeta() {
  try {
    const d = await api('/api/universe');
    sectorLabels = {};
    const sel = document.getElementById('sectorSelect');
    const keep = sel.value || 'ALL';
    let html = `<option value="ALL">🌐 全部（${d.status.total} 檔）</option><option value="WATCH">⭐ 我的觀察名單（${d.watch_count} 檔）</option>`;
    d.sectors.forEach(s => { sectorLabels[s.key] = s.label; html += `<option value="${esc(s.key)}">${esc(s.label)}（${s.count}）</option>`; });
    sel.innerHTML = html;
    sel.value = [...sel.options].some(o => o.value === keep) ? keep : 'ALL';
    currentSector = sel.value;
  } catch (e) { /* 選單維持預設 */ }
}
function scheduleUniRetry(u) {
  if (uniRetry) { clearTimeout(uniRetry); uniRetry = null; }
  if (u && !u.ready) uniRetry = setTimeout(() => { loadUniverseMeta().then(loadScreenerList); }, 8000);
}

async function fetchIndicesData() {
  try {
    const data = await api('/api/market/indices');
    const f = o => {
      const up = o.chg >= 0, col = up ? 'text-[#C0392B]' : 'text-[#1E7E5A]';
      return `<span class="${col}">${Number(o.close).toLocaleString()} (${up ? '▲' : '▼'}${Math.abs(o.pct)}%)</span>`;
    };
    ['dji','ixic','sox','spx'].forEach(k => document.getElementById('idx_' + k).innerHTML = f(data[k]));
    document.getElementById('idx_vix').innerHTML = `<span class="text-amber-500">${data.vix.close} (${esc(data.vix.status)})</span>`;
    document.getElementById('sessionTag').textContent = `美股${data.session}｜美東 ${data.et_time}`;
  } catch (e) { setStatus('指數載入失敗：' + e.message, true); }
}

function setListTab(tab) {
  listTab = tab;
  [['break','tab_break'],['high','tab_high'],['low','tab_low']].forEach(([k, id]) => {
    document.getElementById(id).className = 'py-1.5 rounded ' + (k === tab ? 'bg-[#234E70] text-white' : 'bg-input text-muted');
  });
  document.getElementById('breakModeBox').classList.toggle('hidden', tab !== 'break');
  loadScreenerList();
}

async function loadWeek52List() {
  const note = document.getElementById('screenerNote');
  const data = await api(`/api/week52?kind=${listTab}&sector=${currentSector}`);
  universeData = data.list;
  let t = `已掃描 ${data.scanned} 檔`;
  if (data.universe && !data.universe.ready) t = `⏳ 股票池資料載入中（${data.universe.loaded}/${data.universe.total}），結果尚不完整，自動更新中`;
  if (data.failed && (!data.universe || data.universe.ready)) t += (data.universe && data.universe.refreshing) ? `，${data.failed} 檔尚待更新` : `，${data.failed} 檔取不到資料`;
  if (data.insufficient) t += `，${data.insufficient} 檔上市未滿一年(不納入)`;
  if (data.partial) t += '｜盤中：以當下最新價計算，收盤後才是定案';
  if (data.universe) t += '｜' + uniText(data.universe);
  note.textContent = t; note.title = uniTip(data.universe);
  scheduleUniRetry(data.universe);
  const box = document.getElementById('watchlist');
  box.innerHTML = '';
  if (!universeData.length) {
    box.innerHTML = `<div class="p-4 text-center text-sm text-muted">目前股票池內沒有接近 52 週${listTab === 'high' ? '新高' : '新低'}的標的</div>`;
    return;
  }
  universeData.forEach(item => {
    const up = (item.pct || 0) >= 0, hi = listTab === 'high';
    const badge = hi ? (item.close_new_high ? '收盤新高' : (item.new_high ? '盤中新高' : '接近新高')) : (item.close_new_low ? '收盤新低' : (item.new_low ? '盤中新低' : '接近新低'));
    const dist = hi ? `距高 ${item.from_high_pct}%` : `距低 +${item.from_low_pct}%`;
    const div = document.createElement('div');
    div.className = 'px-3 py-2 rounded-lg bg-input hover:opacity-80 transition border border-theme cursor-pointer';
    div.innerHTML = `
      <div class="font-bold text-sm flex justify-between">
        <span>${esc(item.symbol)} <span class="text-xs text-amber-500">[${esc(sectorTag(item.sector))}]</span></span>
        <span class="text-xs ${hi ? 'text-sky-400 border-sky-500/30' : 'text-amber-500 border-amber-500/30'} bg-card px-1.5 py-0.5 rounded border">${badge}</span>
      </div>
      <div class="text-xs text-muted flex justify-between mt-0.5">
        <span class="truncate max-w-[110px]">${item.name && item.name !== item.symbol ? esc(item.name) : '&nbsp;'}</span>
        <span style="color:${up ? '#C0392B' : '#1E7E5A'}">$${item.price} ${item.pct == null ? '' : '(' + (up ? '▲' : '▼') + Math.abs(item.pct) + '%)'}</span>
      </div>
      <div class="text-xs text-muted flex justify-between"><span>量比 ${item.v_ratio == null ? '—' : item.v_ratio + 'x'}</span><span>${dist}</span></div>
      <div class="text-xs text-muted">52週 ${item.low} – ${item.high}</div>`;
    div.onclick = () => fetchStockData(item.symbol);
    box.appendChild(div);
  });
}

async function loadScreenerList() {
  const note = document.getElementById('screenerNote');
  note.textContent = '掃描中...';
  if (listTab !== 'break') {
    try { await loadWeek52List(); } catch (e) { note.textContent = ''; setStatus('52週榜單載入失敗：' + e.message, true); }
    return;
  }
  try {
    const data = await api(`/api/screener?sector=${currentSector}&mode=${currentRadarMode}&neck_days=${neckDays()}&vol_ratio=${volRatio()}`);
    universeData = data.screener_list;
    let t = `已掃描 ${data.scanned} 檔`;
    if (data.universe && !data.universe.ready) t = `⏳ 股票池資料載入中（${data.universe.loaded}/${data.universe.total}），結果尚不完整，自動更新中`;
    if (data.failed && (!data.universe || data.universe.ready)) t += (data.universe && data.universe.refreshing) ? `，${data.failed} 檔尚待更新` : `，${data.failed} 檔取不到資料`;
    if (data.partial_volume) t += '｜盤中成交量尚未累積完整，量比會偏低';
    if (data.universe) t += '｜' + uniText(data.universe);
    note.textContent = t; note.title = uniTip(data.universe);
    scheduleUniRetry(data.universe);
    renderWatchlist();
  } catch (e) { note.textContent = ''; setStatus('篩選失敗：' + e.message, true); }
}

function renderWatchlist() {
  const box = document.getElementById('watchlist');
  box.innerHTML = '';
  if (!universeData.length) {
    box.innerHTML = `<div class="p-4 text-center text-sm text-muted">目前無符合「${currentRadarMode === 'LONG' ? '多方突破' : '空方破底'}」之標的</div>`;
    return;
  }
  universeData.forEach(item => {
    const up = item.pct >= 0;
    const div = document.createElement('div');
    div.className = 'px-3 py-2 rounded-lg bg-input hover:opacity-80 transition border border-theme cursor-pointer';
    div.innerHTML = `
      <div class="font-bold text-sm flex justify-between">
        <span>${esc(item.symbol)} <span class="text-xs text-amber-500">[${esc(sectorTag(item.sector))}]</span></span>
        <span class="text-xs text-emerald-500 bg-card px-1.5 py-0.5 rounded border border-emerald-500/20">${esc(item.score)}${item.first_day ? '·首日' : ''}</span>
      </div>
      <div class="text-xs text-muted flex justify-between mt-0.5">
        <span class="truncate max-w-[110px]">${item.name && item.name !== item.symbol ? esc(item.name) : '&nbsp;'}</span>
        <span style="color:${up ? '#C0392B' : '#1E7E5A'}">$${item.price} (${up ? '▲' : '▼'}${Math.abs(item.pct)}%)</span>
      </div>
      <div class="text-xs text-muted flex justify-between"><span>量比 ${item.v_ratio}x</span><span>距${currentRadarMode === 'LONG' ? '頸線' : '跌破線'} ${item.dist_neck_pct > 0 ? '+' : ''}${item.dist_neck_pct}%</span></div>`;
    div.onclick = () => fetchStockData(item.symbol);
    box.appendChild(div);
  });
}

function addCustomSymbol() {
  const s = document.getElementById('customSymbol').value.trim().toUpperCase();
  if (s) { fetchStockData(s); document.getElementById('customSymbol').value = ''; }
}

function toggleIndicator(key) {
  indState[key] = !indState[key];
  const b = document.getElementById('pill_' + key);
  b.classList.toggle('pill-active', indState[key]);
  b.classList.toggle('pill-inactive', !indState[key]);
  if (currentRawData) renderPlotlyChart(currentRawData);
}

function setPlotlyMode(mode) {
  const el = document.getElementById('plotlyChartContainer');
  if (el && el._fullLayout) Plotly.relayout(el, {dragmode: mode});
}
function clearAllShapes() {
  const el = document.getElementById('plotlyChartContainer');
  if (el && el._fullLayout) Plotly.relayout(el, {shapes: []});
}
function resetPlotlyAxes() {
  const el = document.getElementById('plotlyChartContainer');
  if (el && el._fullLayout) Plotly.relayout(el, {'xaxis.autorange': true, 'yaxis.autorange': true});
}

function renderPlotlyChart(data) {
  currentRawData = data;
  const el = document.getElementById('plotlyChartContainer');
  const sameChart = lastRender.symbol === data.symbol && lastRender.interval === data.interval;
  const keepShapes = (sameChart && el.layout && el.layout.shapes) ? el.layout.shapes : [];
  lastRender = {symbol: data.symbol, interval: data.interval};

  const d = data.dates, T = [];
  const light = document.getElementById('bodyRoot').classList.contains('theme-light');
  const paperBg = light ? '#ffffff' : '#121824', plotBg = light ? '#f8fafc' : '#0b0e14';
  const grid = light ? '#e2e8f0' : '#1e293b', txt = light ? '#475569' : '#94a3b8';

  T.push({x:d, open:data.opens, high:data.highs, low:data.lows, close:data.closes, type:'candlestick', name:'K線',
    increasing:{line:{color:'#C0392B'}, fillcolor:'#C0392B'}, decreasing:{line:{color:'#1E7E5A'}, fillcolor:'#1E7E5A'}, xaxis:'x', yaxis:'y'});
  const line = (y, color, name, w, dash) => ({x:d, y, type:'scatter', mode:'lines', connectgaps:false, line:{color, width:w||1.6, dash}, name, xaxis:'x', yaxis:'y'});
  if (indState.ma5) T.push(line(data.ma5, '#D97706', '5MA'));
  if (indState.ma10) T.push(line(data.ma10, '#2563EB', '10MA'));
  if (indState.ma20) T.push(line(data.ma20, '#7C3AED', '20MA(月線)', 2.2));
  if (indState.ma60) T.push(line(data.ma60, '#0D9488', '60MA(季線)'));
  if (indState.ma120) T.push(line(data.ma120, '#DB2777', '120MA(半年)'));
  if (indState.ma240) T.push(line(data.ma240, '#94a3b8', '240MA(年線)', 1.8, 'dash'));
  if (indState.bb) {
    T.push(line(data.bb_up, 'rgba(148,163,184,0.6)', '布林上軌', 1, 'dot'));
    T.push(line(data.bb_low, 'rgba(148,163,184,0.6)', '布林下軌', 1, 'dot'));
  }
  const neckName = `${data.neck_days}${data.neck_unit}${data.mode === 'LONG' ? '頸線' : '跌破線'}`;
  document.getElementById('neckPillText').textContent = neckName;
  if (indState.neck && data.neckline != null) T.push(line(Array(d.length).fill(data.neckline), '#B45309', neckName + ' $' + data.neckline, 2, 'dash'));

  const w = data.week52;
  if (indState.w52 && w) {
    T.push(line(Array(d.length).fill(w.high), '#0EA5E9', `52週高 $${w.high}（${w.high_date}）`, 1.4, 'dot'));
    T.push(line(Array(d.length).fill(w.low), '#F59E0B', `52週低 $${w.low}（${w.low_date}）`, 1.4, 'dot'));
  }

  const volColors = data.closes.map((c, i) => c >= data.opens[i] ? 'rgba(192,57,43,0.7)' : 'rgba(30,126,90,0.7)');
  T.push({x:d, y:data.volumes, type:'bar', marker:{color:volColors}, name:'成交量', xaxis:'x', yaxis:'y2'});
  T.push({x:d, y:data.vol_ma5, type:'scatter', mode:'lines', line:{color:'#B45309', width:1.6, dash:'dot'}, name:'5均量', xaxis:'x', yaxis:'y2'});
  T.push({x:d, y:data.k, type:'scatter', mode:'lines', line:{color:'#D97706', width:1.8}, name:'K(9)', xaxis:'x', yaxis:'y3'});
  T.push({x:d, y:data.d, type:'scatter', mode:'lines', line:{color:'#2563EB', width:1.8}, name:'D(9)', xaxis:'x', yaxis:'y3'});
  T.push({x:d, y:data.macd_hist, type:'bar', marker:{color:data.macd_hist.map(v => (v||0) >= 0 ? '#C0392B' : '#1E7E5A')}, name:'MACD柱', xaxis:'x', yaxis:'y4'});
  T.push({x:d, y:data.macd_dif, type:'scatter', mode:'lines', line:{color:'#D97706', width:1.5}, name:'DIF', xaxis:'x', yaxis:'y4'});
  T.push({x:d, y:data.macd_dea, type:'scatter', mode:'lines', line:{color:'#2563EB', width:1.5}, name:'DEA', xaxis:'x', yaxis:'y4'});

  const layout = {
    paper_bgcolor:paperBg, plot_bgcolor:plotBg, font:{color:txt, family:'sans-serif', size:13},
    margin:{l:10, r:55, t:15, b:34}, showlegend:false,
    xaxis:{type:'category', anchor:'y4', rangeslider:{visible:false}, gridcolor:grid, nticks:isTouch ? 4 : 8},
    yaxis:{domain:[0.52,1.0], gridcolor:grid, side:'right'},
    yaxis2:{domain:[0.36,0.49], gridcolor:grid, side:'right', title:'量'},
    yaxis3:{domain:[0.19,0.33], gridcolor:grid, range:[0,100], tickvals:[20,50,80], side:'right', title:'KD'},
    yaxis4:{domain:[0.0,0.16], gridcolor:grid, side:'right', title:'MACD'},
    hovermode:'x unified',
    dragmode: isTouch ? false : 'pan',
    shapes: keepShapes,
    newshape:{line:{color:'#facc15', width:2.5}}
  };
  Plotly.newPlot(el, T, layout, {responsive:true, displayModeBar:false, scrollZoom:!isTouch, edits:{shapePosition:true}});
}

async function fetchStockData(symbol) {
  currentSymbol = symbol;
  document.getElementById('chartTitle').innerText = `${symbol} 載入中...`;
  try {
    const data = await api(`/api/quote/${encodeURIComponent(symbol)}?interval=${currentInterval}&mode=${currentRadarMode}&neck_days=${neckDays()}`);
    currentSpot = data.spot; currentMA20 = data.ma20_val; currentNeckline = data.neckline;
    const iv = {'1d':'日K','1wk':'週K','1mo':'月K'}[currentInterval];
    document.getElementById('chartTitle').innerText = `${data.symbol}${data.name && data.name !== data.symbol ? ' ' + data.name : ''} [${data.exchange}] (${iv})`;
    const up = data.chg >= 0;
    const tag = document.getElementById('currentPriceTag');
    tag.style.color = up ? '#C0392B' : '#1E7E5A';
    tag.innerText = `收盤: $${data.spot} (${up ? '▲' : '▼'}${Math.abs(data.pct)}%) ${data.last_date}`;
    const m20 = data.ma20_val != null ? '$' + data.ma20_val : '資料不足';
    document.getElementById('spotDisplay').innerText = `$${data.spot} (20MA: ${m20})`;
    document.getElementById('stopLossDisplay').innerText = `${data.neckline != null ? '頸線 $' + data.neckline : '—'} ／ 月線 ${m20}`;
    const wt = document.getElementById('w52Tag'), w5 = data.week52;
    if (w5) {
      const flag = w5.close_new_high ? ' 🏔️收盤新高' : (w5.new_high ? ' 🏔️盤中新高' : (w5.new_low ? ' 🕳️創新低' : ''));
      wt.textContent = `52週 ${w5.low}–${w5.high}｜距高 ${w5.from_high_pct}%｜位置 ${w5.position_pct == null ? '—' : w5.position_pct + '%'}${flag}`;
    } else { wt.textContent = '52週：上市未滿一年，資料不足'; }
    wt.classList.remove('hidden');
    document.getElementById('qualityTag').innerText = data.quality_badge;
    renderPlotlyChart(data);
    loadOptionChain(data.symbol);
    loadEarnings(data.symbol);
  } catch (e) {
    document.getElementById('chartTitle').innerText = `${symbol} 載入失敗`;
    setStatus(`載入 ${symbol} 失敗：${e.message}`, true);
  }
}

async function loadEarnings(symbol) {
  nextEarnings = null;
  const tag = document.getElementById('earningsTag');
  tag.classList.add('hidden');
  try {
    const r = await api('/api/earnings/' + encodeURIComponent(symbol));
    if (symbol !== currentSymbol) return;
    nextEarnings = r.date;
    if (r.date) { tag.textContent = '📅 下次財報 ' + r.date; tag.classList.remove('hidden'); }
    checkEarningsVsExpiry();
  } catch (e) {}
}
function checkEarningsVsExpiry() {
  const exp = document.getElementById('expDateSelect').value;
  const note = document.getElementById('chainNote');
  if (nextEarnings && exp && nextEarnings <= exp) {
    note.textContent = `⚠️ 財報日 ${nextEarnings} 落在到期日 ${exp} 之前：IV 可能在財報後大幅下降(IV Crush)，股價也可能跳空，賣方策略請特別留意。`;
  }
}

/* ===================== 選擇權 ===================== */
const STRAT = {
  csp:      {legs:[{t:'put', side:-1, k:1}], list:'put'},
  cc:       {legs:[{t:'call', side:-1, k:1}], list:'call', stock:true},
  bull_put: {legs:[{t:'put', side:-1, k:1}, {t:'put', side:1, k:2}], list:'put', k2:-1, k2label:'買進保護腿（較低履約價）'},
  long_call:{legs:[{t:'call', side:1, k:1}], list:'call'},
  long_put: {legs:[{t:'put', side:1, k:1}], list:'put'},
  bull_call:{legs:[{t:'call', side:1, k:1}, {t:'call', side:-1, k:2}], list:'call', k2:1, k2label:'賣出腿（較高履約價）'},
  bear_put: {legs:[{t:'put', side:1, k:1}, {t:'put', side:-1, k:2}], list:'put', k2:-1, k2label:'賣出腿（較低履約價）'},
  straddle: {legs:[{t:'call', side:1, k:1}, {t:'put', side:1, k:1}], list:'call'}
};
function stratKey() { return document.getElementById('strategySelect').value; }

async function loadOptionChain(symbol) {
  const expSel = document.getElementById('expDateSelect');
  expSel.innerHTML = '<option value="">查詢到期日中...</option>';
  document.getElementById('strikeSelect').innerHTML = '<option value="">請先選到期日</option>';
  document.getElementById('chainNote').textContent = '';
  document.getElementById('retryChainBtn').classList.add('hidden');
  currentChain = null; chainCache = {};
  document.getElementById('chainSrc').textContent = '';
  document.getElementById('chainDte').textContent = '';
  try {
    const r = await api('/api/options/' + encodeURIComponent(symbol));
    if (symbol !== currentSymbol) return;
    currentExpirations = r.expirations || [];
    if (!currentExpirations.length) {
      expSel.innerHTML = '<option value="">⚠️ 無期權合約</option>';
      document.getElementById('chainNote').textContent = r.error || '此標的無期權合約';
      document.getElementById('retryChainBtn').classList.remove('hidden');
      return;
    }
    expSel.innerHTML = currentExpirations.map(e => `<option value="${esc(e)}">${esc(e)}</option>`).join('');
    document.getElementById('chainSrc').textContent = r.source_label ? '期權資料來源：' + r.source_label : '';
    // 預設選距今 ≥21 天的第一個到期日（避免一打開就選到明天到期的合約）；沒有就用第一個
    const dteOf = e => (new Date(e + 'T21:00:00Z') - Date.now()) / 86400000;
    const pick = currentExpirations.find(e => dteOf(e) >= 21) || currentExpirations[0];
    expSel.value = pick;
    onExpirationChange();
  } catch (e) {
    expSel.innerHTML = '<option value="">⚠️ 取得失敗</option>';
    document.getElementById('chainNote').textContent = e.message;
    document.getElementById('retryChainBtn').classList.remove('hidden');
  }
}

async function onExpirationChange() {
  const exp = document.getElementById('expDateSelect').value;
  const note = document.getElementById('chainNote');
  document.getElementById('retryChainBtn').classList.add('hidden');
  if (!exp) { if (currentSymbol) loadOptionChain(currentSymbol); return; }
  note.textContent = '載入期權鏈中...';
  try {
    const key = currentSymbol + '|' + exp;
    if (!chainCache[key]) chainCache[key] = await api(`/api/options/${encodeURIComponent(currentSymbol)}/chain?exp=${exp}&spot=${currentSpot}`);
    currentChain = chainCache[key];
    note.textContent = '';
    const dd = Math.max(0, Math.ceil((new Date(exp + 'T21:00:00Z') - Date.now()) / 86400000));
    document.getElementById('chainDte').textContent = dd <= 7 ? `⚠️ 距到期僅 ${dd} 天：Gamma 風險極高，股價小幅變動就會讓損益劇烈變化，且深度價內合約買賣價差可能很大。` : '';
    checkEarningsVsExpiry();
    populateStrikes();
  } catch (e) {
    note.textContent = e.message;
    document.getElementById('retryChainBtn').classList.remove('hidden');
  }
}
function onStrategyChange() { if (currentChain) populateStrikes(); }

function optLabel(o) {
  const q = (o.bid > 0 || o.ask > 0) ? `Bid ${o.bid} / Ask ${o.ask}` : `最後成交 ${o.last}（無即時報價）`;
  return `$${o.strike.toFixed(2)} | ${q} | IV ${o.impliedVolatility == null ? '—' : (o.impliedVolatility * 100).toFixed(0) + '%'} | OI ${o.openInterest.toLocaleString()}`;
}
function populateStrikes() {
  const S = STRAT[stratKey()];
  const list = (S.list === 'call' ? currentChain.calls : currentChain.puts) || [];
  const s1 = document.getElementById('strikeSelect');
  s1.innerHTML = list.map((o, i) => `<option value="${i}">${esc(optLabel(o))}</option>`).join('');
  document.getElementById('strike1Label').textContent = S.legs.length > 1 && S.k2 ? '主腿履約價（賣權/買權）' : '履約價 (Strike)';
  const wrap = document.getElementById('strike2Wrap');
  if (S.k2) {
    wrap.classList.remove('hidden');
    document.getElementById('strike2Label').textContent = S.k2label;
    document.getElementById('strike2Select').innerHTML = list.map((o, i) => `<option value="${i}">${esc(optLabel(o))}</option>`).join('');
  } else wrap.classList.add('hidden');
  autoAlignMA20();
}
function nearestIdx(list, target) {
  let bi = 0, bd = 1e12;
  list.forEach((o, i) => { const d = Math.abs(o.strike - target); if (d < bd) { bd = d; bi = i; } });
  return bi;
}
function autoAlignMA20() {
  if (!currentChain) return;
  const S = STRAT[stratKey()];
  const list = (S.list === 'call' ? currentChain.calls : currentChain.puts) || [];
  if (!list.length) return;
  let target = currentMA20 || currentSpot;
  if (stratKey() === 'straddle') target = currentSpot;   // 跨式取最接近現價(ATM)
  let i1;
  const k = stratKey();
  if (k === 'csp' || k === 'bull_put') {
    // 賣出賣權：履約價不應高於現價（否則是價內，不是「低接」）；取 min(月線, 現價) 以下最近的履約價
    const cap = Math.min(currentMA20 || currentSpot, currentSpot);
    const below = list.map((o, i) => ({i, k: o.strike})).filter(x => x.k <= cap);
    i1 = below.length ? below[below.length - 1].i : nearestIdx(list, cap);
  } else if (k === 'cc') {
    const flo = Math.max(currentMA20 || currentSpot, currentSpot);
    const above = list.map((o, i) => ({i, k: o.strike})).filter(x => x.k >= flo);
    i1 = above.length ? above[0].i : nearestIdx(list, flo);
  } else {
    i1 = nearestIdx(list, target);
  }
  document.getElementById('strikeSelect').selectedIndex = i1;
  if (S.k2) {
    const t2 = list[i1].strike * (1 + 0.05 * S.k2);
    let i2 = nearestIdx(list, t2);
    if (i2 === i1) i2 = Math.max(0, Math.min(list.length - 1, i1 + S.k2));
    document.getElementById('strike2Select').selectedIndex = i2;
  }
  recompute();
}

function legPrice(o, side) {
  // 買進用 Ask、賣出用 Bid；缺報價時退回最後成交價
  const p = side > 0 ? (o.ask > 0 ? o.ask : o.last) : (o.bid > 0 ? o.bid : o.last);
  return {price: p, approx: !(side > 0 ? o.ask > 0 : o.bid > 0)};
}
function buildLegs() {
  if (!currentChain) return null;
  const key = stratKey(), S = STRAT[key];
  const list = (S.list === 'call' ? currentChain.calls : currentChain.puts) || [];
  const i1 = parseInt(document.getElementById('strikeSelect').value);
  if (isNaN(i1) || !list[i1]) return null;
  const o1 = list[i1];
  let o2 = null;
  if (S.k2) {
    const i2 = parseInt(document.getElementById('strike2Select').value);
    if (isNaN(i2) || !list[i2] || i2 === i1) return {error: '兩腿履約價不能相同，請重新選擇第二腿'};
    o2 = list[i2];
  }
  const legs = S.legs.map(L => {
    let opt;
    if (L.k === 2) opt = o2;
    else if (L.t === (S.list)) opt = o1;
    else {   // 跨式的另一腿：同履約價的對向合約
      const other = (L.t === 'call' ? currentChain.calls : currentChain.puts) || [];
      opt = other[nearestIdx(other, o1.strike)];
    }
    const pp = legPrice(opt, L.side);
    return {t: L.t, side: L.side, strike: opt.strike, prem: pp.price, approx: pp.approx, iv: opt.impliedVolatility};
  });
  // 價差方向檢查
  if (key === 'bull_put' && legs[1].strike >= legs[0].strike) return {error: 'Bull Put：買進腿履約價必須低於賣出腿'};
  if (key === 'bear_put' && legs[1].strike >= legs[0].strike) return {error: 'Bear Put：賣出腿履約價必須低於買進腿'};
  if (key === 'bull_call' && legs[1].strike <= legs[0].strike) return {error: 'Bull Call：賣出腿履約價必須高於買進腿'};
  return {legs, stock: !!S.stock};
}
function payoffAt(P, built) {
  let v = 0;
  built.legs.forEach(L => {
    const intr = L.t === 'call' ? Math.max(P - L.strike, 0) : Math.max(L.strike - P, 0);
    v += L.side > 0 ? (intr - L.prem) : (L.prem - intr);
  });
  if (built.stock) v += (P - currentSpot);
  return v * 100;
}
function analyze(built) {
  const S = currentSpot, pts = [];
  for (let i = 0; i <= 600; i++) { const P = (S * 3) * i / 600; pts.push([P, payoffAt(P, built)]); }
  const vals = pts.map(p => p[1]);
  const slopeUp = payoffAt(S * 3.5, built) - payoffAt(S * 3, built);
  const maxP = slopeUp > 1 ? Infinity : Math.max(...vals), minP = Math.min(...vals);
  const be = [];
  for (let i = 1; i < pts.length; i++) {
    const a = pts[i - 1], b = pts[i];
    if ((a[1] < 0 && b[1] >= 0) || (a[1] > 0 && b[1] <= 0)) be.push(a[0] + (0 - a[1]) * (b[0] - a[0]) / (b[1] - a[1]));
  }
  return {maxP, minP, be};
}
function drawPayoff(built, res) {
  const S = currentSpot, xs = [], ys = [];
  for (let i = 0; i <= 160; i++) { const P = S * 0.6 + (S * 0.8) * i / 160; xs.push(P); ys.push(payoffAt(P, built)); }
  const light = document.getElementById('bodyRoot').classList.contains('theme-light');
  const txt = light ? '#475569' : '#94a3b8', grid = light ? '#e2e8f0' : '#1e293b';
  const shapes = [{type:'line', x0:S, x1:S, yref:'paper', y0:0, y1:1, line:{color:'#38bdf8', width:1, dash:'dot'}}];
  built.legs.forEach(L => shapes.push({type:'line', x0:L.strike, x1:L.strike, yref:'paper', y0:0, y1:1, line:{color:'#B45309', width:1, dash:'dash'}}));
  Plotly.newPlot('payoffChart', [
    {x:xs, y:ys, type:'scatter', mode:'lines', line:{color:'#38bdf8', width:2.5}, fill:'tozeroy', fillcolor:'rgba(56,189,248,0.12)',
     hovertemplate:'到期價 $%{x:.2f}<br>損益 %{y:,.0f}<extra></extra>'}
  ], {
    paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)', font:{color:txt, size:11}, margin:{l:48, r:8, t:6, b:24},
    xaxis:{gridcolor:grid, tickprefix:'$', fixedrange:true}, yaxis:{gridcolor:grid, zeroline:true, zerolinecolor:'#94a3b8', fixedrange:true}, shapes, showlegend:false, dragmode:false
  }, {responsive:true, displayModeBar:false, scrollZoom:false, doubleClick:false});
}
let lastBuilt = null, lastAnalysis = null;
async function recompute() {
  const built = buildLegs();
  const note = document.getElementById('chainNote');
  if (!built) return;
  if (built.error) { note.textContent = built.error; return; }
  checkEarningsVsExpiry();
  if (!note.textContent) note.textContent = '';
  lastBuilt = built;
  const res = analyze(built); lastAnalysis = res;
  const net = built.legs.reduce((a, L) => a + (L.side > 0 ? -L.prem : L.prem), 0) * 100;
  document.getElementById('netPremium').textContent = (net >= 0 ? '淨收入 ' : '淨支出 ') + '$' + Math.abs(net).toFixed(0) + (built.legs.some(L => L.approx) ? ' (含最後成交價估算)' : '');
  document.getElementById('payoffMax').textContent = res.maxP === Infinity ? '無上限' : fmt$(res.maxP);
  document.getElementById('payoffMin').textContent = fmt$(res.minP);
  document.getElementById('payoffBE').textContent = res.be.length ? res.be.map(b => '$' + b.toFixed(2)).join(' / ') : '無（不會損益兩平）';
  drawPayoff(built, res);

  // Greeks：各腿加總
  const exp = document.getElementById('expDateSelect').value;
  const days = Math.max(0.5, (new Date(exp + 'T21:00:00Z') - Date.now()) / 86400000);
  const T = days / 365;
  try {
    if (built.legs.some(L => !(L.iv > 0.01))) {
      ['Delta','Theta','Gamma','Vega'].forEach(k => document.getElementById('val' + k).innerText = '—');
      return;   // 有合約缺 IV：不用猜測值算 Greeks
    }
    const gs = await Promise.all(built.legs.map(L =>
      api(`/api/greeks?S=${currentSpot}&K=${L.strike}&T=${T}&sigma=${L.iv}&type=${L.t}`)));
    const sum = k => built.legs.reduce((a, L, i) => a + L.side * gs[i][k], 0) + (k === 'delta' && built.stock ? 1 : 0);
    ['delta','theta','gamma','vega'].forEach(k => document.getElementById('val' + k[0].toUpperCase() + k.slice(1)).innerText = sum(k).toFixed(4));
  } catch (e) {}
}

function calculateOptionStrategy() {
  if (!lastBuilt || !lastAnalysis) return;
  const key = stratKey(), L = lastBuilt.legs, a = lastAnalysis;
  const exp = document.getElementById('expDateSelect').value;
  const days = Math.max(1, Math.ceil((new Date(exp + 'T21:00:00Z') - Date.now()) / 86400000));
  const lines = [`標的 ${currentSymbol}｜到期 ${exp}（約 ${days} 天）`];
  const net = L.reduce((s, x) => s + (x.side > 0 ? -x.prem : x.prem), 0);
  if (key === 'csp') {
    const be = L[0].strike - L[0].prem;
    const buffer = (currentSpot - be) / currentSpot * 100;
    const ann = (L[0].prem / L[0].strike) * (365 / days) * 100;
    lines.push(`損益兩平 $${be.toFixed(2)}，距現價下跌空間 ${buffer.toFixed(1)}%（${buffer >= 8 ? '✅ 緩衝充足' : '⚠️ 緩衝偏窄(<8%)'}）`);
    lines.push(`年化報酬(以履約價為保證金) ≈ ${ann.toFixed(1)}%`);
    lines.push(`最壞情況(股價歸零)每口虧損 ${fmt$(a.minP)}；接股成本 $${be.toFixed(2)}，請確認願意持有。`);
  } else if (key === 'cc') {
    const ann = ((L[0].prem + Math.max(0, L[0].strike - currentSpot)) / currentSpot) * (365 / days) * 100;
    lines.push(`若被履約最大獲利 ${fmt$(a.maxP)}（含權利金與價差），年化 ≈ ${ann.toFixed(1)}%（假設持有到到期）`);
    lines.push(`下跌風險：現股下跌的損失只被權利金 $${L[0].prem.toFixed(2)} 小幅緩衝。`);
  } else {
    const rr = a.maxP === Infinity ? '無上限' : (a.minP < 0 ? (a.maxP / Math.abs(a.minP)).toFixed(2) + ' : 1' : '—');
    lines.push(`最大獲利 ${a.maxP === Infinity ? '無上限' : fmt$(a.maxP)}｜最大虧損 ${fmt$(a.minP)}｜報酬風險比 ${rr}`);
    lines.push(`損益兩平 ${a.be.length ? a.be.map(b => '$' + b.toFixed(2)).join(' / ') : '無'}（現價 $${currentSpot}）`);
  }
  if (L.some(x => !(x.iv > 0.01))) lines.push('⚠️ 有合約缺 IV 報價（可能盤後或無量），Greeks 顯示「—」，不以假設值替代。');
  if (nextEarnings && nextEarnings <= exp) lines.push(`⚠️ 財報日 ${nextEarnings} 在到期日前。`);
  const box = document.getElementById('riskBox');
  box.textContent = lines.join('\n');
  box.classList.remove('hidden');
}

async function sendCloudPush() {
  let pin = '';
  try { pin = sessionStorage.getItem('us_pin') || ''; } catch (e) {}
  if (!pin) { pin = prompt('請輸入管理員通行碼 (ADMIN_PUSH_PIN)') || ''; if (!pin) return; }
  let summary = '';
  if (lastBuilt && lastAnalysis) {
    const legsTxt = lastBuilt.legs.map(L => `${L.side > 0 ? '買' : '賣'} ${L.t === 'call' ? 'Call' : 'Put'} $${L.strike}@${L.prem}`).join(' + ');
    summary = `${legsTxt}｜到期 ${document.getElementById('expDateSelect').value}｜最大獲利 ${lastAnalysis.maxP === Infinity ? '無上限' : fmt$(lastAnalysis.maxP)} / 最大虧損 ${fmt$(lastAnalysis.minP)}`;
  }
  const result = document.getElementById('lineResult');
  try {
    const r = await api('/api/line/push', {
      method: 'POST',
      headers: {'Content-Type': 'application/json', 'X-Push-Pin': pin},
      body: JSON.stringify({symbol: currentSymbol, strategy: stratKey(), mode: currentRadarMode, neck_days: neckDays(), summary})
    });
    try { sessionStorage.setItem('us_pin', pin); } catch (e) {}
    result.textContent = r.status === 'success' ? '✅ 已成功送出 LINE 推播' : `❌ 推播失敗（LINE 狀態：${r.line_status}）`;
    result.style.color = r.status === 'success' ? '#06C755' : '#ef4444';
    document.getElementById('linePreviewText').innerText = r.preview_text;
    document.getElementById('lineModal').classList.remove('hidden');
  } catch (e) {
    if (/通行碼錯誤/.test(e.message)) { try { sessionStorage.removeItem('us_pin'); } catch (x) {} }
    setStatus('推播失敗：' + e.message, true);
  }
}

function changeInterval(intv) {
  currentInterval = intv;
  ['1d','1wk','1mo'].forEach(k => {
    document.getElementById('btn_' + k).className = 'px-2.5 py-0.5 rounded font-bold ' + (k === intv ? 'bg-[#38bdf8] text-slate-950' : 'text-muted');
  });
  fetchStockData(currentSymbol);
}

window.onload = () => {
  try {
    const t = localStorage.getItem('us_theme'); if (t) switchTheme(t);
    setFontSize(localStorage.getItem('us_font') || 'large');
  } catch (e) { setFontSize('large'); }
  if (isTouch) document.getElementById('touchHint').classList.remove('hidden');
  fetchIndicesData();
  loadUniverseMeta().then(loadScreenerList);
  fetchStockData(currentSymbol);
};
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def serve_index():
    return HTMLResponse(content=HTML_CONTENT, status_code=200)
