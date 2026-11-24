"""美股推播的「事件提醒」：總經事件（FOMC／CPI／非農）＋ 已入選推播個股的財報日、最新新聞。

資料來源（皆免費公開，2026-11-02 在 GitHub Actions 上實測）：
- FOMC：聯準會官網 https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm（每次執行即時抓）
- CPI／非農：美國勞工統計局 BLS 公布的發布行事曆。BLS 網站禁止程式自動抓取（回 403），
  所以把官方行事曆抄在下方 BLS_SCHEDULE（2026-11-02 依 bls.gov/schedule/news_release/cpi.htm、empsit.htm 核對），
  超出表格範圍的日期一律顯示「—」，需要每年更新一次。
- 財報日：Yahoo Finance（yfinance get_earnings_dates／calendar）
- 新聞：Google News RSS（英文原文標題＋來源；翻譯需要付費 API，所以保留原文）

原則：抓不到就顯示「—」，絕不填推測值；任何錯誤都不會中斷推播。
"""
import datetime
import email.utils
import html
import logging
import re
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from zoneinfo import ZoneInfo

import requests

log = logging.getLogger("us_scheduler")
ET = ZoneInfo("America/New_York")
TPE = ZoneInfo("Asia/Taipei")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
WEEKDAY_ZH = "一二三四五六日"

MACRO_AHEAD_DAYS = 7        # 總經事件：未來 7 天內
EARNINGS_AHEAD_DAYS = 14    # 財報：未來 14 天內
NEWS_HOURS = 36             # 新聞：最近 36 小時

# BLS 官方發布日（美東 08:30）。來源：bls.gov/schedule/news_release/cpi.htm、empsit.htm（2026-11-02 核對）
BLS_SCHEDULE = {
    "CPI": ["2026-11-10", "2026-12-10", "2027-01-13", "2027-02-11", "2027-03-10", "2027-04-13", "2027-05-12",
            "2027-06-10", "2027-07-14", "2027-08-11", "2027-09-14", "2027-10-14", "2027-11-10", "2027-12-10"],
    "非農就業": ["2026-11-06", "2026-12-04", "2027-01-08", "2027-02-05", "2027-03-05", "2027-04-02", "2027-05-07",
             "2027-06-04", "2027-07-02", "2027-08-06", "2027-09-03", "2027-10-08", "2027-11-05", "2027-12-03"],
}
BLS_COVERED_UNTIL = datetime.date(2027, 12, 31)

# 新聞搜尋用的英文公司名（ETF 不查個股新聞與財報）
EN_NAME = {
    "NVDA": "Nvidia", "TSM": "TSMC", "AMD": "AMD", "AVGO": "Broadcom", "QCOM": "Qualcomm", "MU": "Micron",
    "INTC": "Intel", "ARM": "Arm Holdings", "ASML": "ASML", "TXN": "Texas Instruments", "AAPL": "Apple",
    "MSFT": "Microsoft", "GOOGL": "Alphabet", "AMZN": "Amazon", "META": "Meta Platforms", "TSLA": "Tesla",
    "NFLX": "Netflix", "COST": "Costco", "ADBE": "Adobe", "COIN": "Coinbase", "CAT": "Caterpillar",
    "BA": "Boeing", "UNH": "UnitedHealth", "JNJ": "Johnson & Johnson", "MCD": "McDonald's", "HD": "Home Depot",
    "V": "Visa", "WMT": "Walmart", "IBM": "IBM", "GS": "Goldman Sachs", "PLTR": "Palantir", "JPM": "JPMorgan",
    "BAC": "Bank of America", "XOM": "Exxon Mobil", "CVX": "Chevron", "LLY": "Eli Lilly", "NKE": "Nike",
    "UBER": "Uber",
}
ALIASES = {"JNJ": ("j&j",), "XOM": ("exxon",), "META": ("meta",), "GOOGL": ("google",), "ARM": ("arm",),
           "JPM": ("jpmorgan chase", "jpmorgan"), "UNH": ("unitedhealth group",), "LLY": ("lilly",),
           "TSM": ("taiwan semiconductor",), "GS": ("goldman",), "BAC": ("bofa",), "MCD": ("mcdonald",)}
JUNK_TITLE = re.compile(r"Stock Price, News, Quote|Price, News & History|\b[A-Z]{1,5}\d{6}[CP]\d{8}\b|\d+\.\d{3} (call|put)\b", re.I)
ETFS = {"SPY", "QQQ", "SOXX", "DIA", "IWM"}

MONTHS = {m: i + 1 for i, m in enumerate(["January", "February", "March", "April", "May", "June", "July", "August",
                                          "September", "October", "November", "December"])}


def md_wd(d: datetime.date) -> str:
    return f"{d.month}/{d.day}（{WEEKDAY_ZH[d.weekday()]}）"


def _tpe(d: datetime.date, hh: int, mm: int) -> str:
    t = datetime.datetime(d.year, d.month, d.day, hh, mm, tzinfo=ET).astimezone(TPE)
    return f"台北 {t.month}/{t.day} {t:%H:%M}"


# ------------------------------------------------------------------------------
# FOMC（聯準會官網）
# ------------------------------------------------------------------------------
def fetch_fomc_dates():
    """回傳 [(決議日 date, 是否有經濟預測SEP)]；抓不到回 None。"""
    try:
        r = requests.get("https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm", headers=UA, timeout=20)
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}")
        t = r.text
        heads = [(m.start(), int(m.group(1))) for m in re.finditer(r"(\d{4}) FOMC Meetings", t)]
        if not heads:
            raise RuntimeError("找不到年度標題（可能改版）")
        pat = re.compile(r'fomc-meeting__month[^"]*"[^>]*>\s*(?:<strong>)?\s*([A-Za-z/]+)\s*(?:</strong>)?\s*</div>\s*'
                         r'<div class="fomc-meeting__date[^"]*"[^>]*>\s*([^<]+?)\s*</div>', re.S)
        out = []
        for m in pat.finditer(t):
            year = None
            for pos, y in heads:
                if pos < m.start():
                    year = y
            if year is None:
                continue
            mon_s, day_s = m.group(1), m.group(2)
            if re.search(r"[A-Za-z]", day_s.replace("*", "")):   # (unscheduled)、(notation vote) 等臨時會議不列
                continue
            days = re.findall(r"\d{1,2}", day_s)
            mon_name = mon_s.split("/")[-1]                       # Apr/May 30-1 → 決議日在後一個月
            mon = next((v for k, v in MONTHS.items() if k.startswith(mon_name[:3])), None)
            if not days or not mon:
                continue
            out.append((datetime.date(year, mon, int(days[-1])), "*" in day_s))
        if not out:
            raise RuntimeError("解析不到會議日期（可能改版）")
        return sorted(set(out))
    except Exception as e:
        log.warning("FOMC 行事曆抓取失敗：%s", e)
        return None


def macro_events(today: datetime.date = None):
    """未來 MACRO_AHEAD_DAYS 天內的總經事件：[(文字, 是否當天/隔天)]。"""
    today = today or datetime.datetime.now(ET).date()
    horizon = today + datetime.timedelta(days=MACRO_AHEAD_DAYS)
    items = []
    fomc = fetch_fomc_dates()
    if fomc is None:
        items.append((today, "🏛️ FOMC 利率決議 —（聯準會網站暫時抓不到）", False))
    else:
        for d, sep in fomc:
            if today <= d <= horizon:
                extra = "＋經濟預測" if sep else ""
                items.append((d, f"🏛️ FOMC 利率決議{extra} {md_wd(d)} 14:00 ET（{_tpe(d, 14, 0)}）", True))
    for name, dates in BLS_SCHEDULE.items():
        if horizon > BLS_COVERED_UNTIL:
            items.append((today, f"📊 {name} 公布日 —（行事曆需更新）", False))
            continue
        for ds in dates:
            d = datetime.date.fromisoformat(ds)
            if today <= d <= horizon:
                items.append((d, f"📊 {name} {md_wd(d)} 08:30 ET（{_tpe(d, 8, 30)}）", True))
    items.sort(key=lambda x: x[0])
    return [(txt, ok and (d - today).days <= 1) for d, txt, ok in items]


# ------------------------------------------------------------------------------
# 個股：財報日、新聞
# ------------------------------------------------------------------------------
def fetch_earnings(sym: str, today: datetime.date):
    """下一次財報（date, 盤前/盤後 or ''）；無資料回 ('none', None)；抓取失敗回 None。"""
    import yfinance as yf
    tk = yf.Ticker(sym)
    try:
        df = tk.get_earnings_dates(limit=6)
        if df is not None and len(df):
            fut = []
            for ts in df.index:
                ts_et = ts.tz_convert(ET) if getattr(ts, "tzinfo", None) else ts
                if ts_et.date() >= today:
                    fut.append(ts_et)
            if not fut:
                return ("none", None)
            nxt = min(fut)
            hm = nxt.hour * 100 + nxt.minute
            when = "盤後" if hm >= 1600 else ("盤前" if 0 < hm < 930 else "")
            return (nxt.date(), when)
    except Exception as e:
        log.info("%s get_earnings_dates 失敗，改用 calendar：%s", sym, e)
    try:
        cal = tk.calendar or {}
        ds = cal.get("Earnings Date") or []
        ds = [d for d in ds if isinstance(d, datetime.date) and d >= today]
        return (min(ds), "") if ds else ("none", None)
    except Exception as e:
        log.warning("%s 財報日抓取失敗：%s", sym, e)
        return None


def fetch_news(sym: str):
    """最近 NEWS_HOURS 小時內最新一則（標題, 來源, 時間ET）；沒有回 'none'；抓取失敗回 None。"""
    name = EN_NAME.get(sym, sym)
    q = f'"{name}" stock when:2d'
    url = "https://news.google.com/rss/search?" + urllib.parse.urlencode({"q": q, "hl": "en-US", "gl": "US", "ceid": "US:en"})
    try:
        r = requests.get(url, headers=UA, timeout=15)
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}")
        now = datetime.datetime.now(datetime.timezone.utc)
        best = None
        keys = {name.lower(), *ALIASES.get(sym, ())} | ({sym.lower()} if len(sym) >= 3 else set())
        for item in re.findall(r"<item>(.*?)</item>", r.text, re.S):
            t = re.search(r"<title>(.*?)</title>", item, re.S)
            p = re.search(r"<pubDate>(.*?)</pubDate>", item, re.S)
            s = re.search(r"<source[^>]*>(.*?)</source>", item, re.S)
            if not t or not p:
                continue
            title = html.unescape(re.sub(r"<!\[CDATA\[|\]\]>", "", t.group(1))).strip()
            src = html.unescape(s.group(1)).strip() if s else ""
            if src and title.endswith(" - " + src):
                title = title[: -len(" - " + src)].strip()
            if JUNK_TITLE.search(title):
                continue                      # 報價頁、選擇權合約頁之類不是新聞
            low = title.lower()
            if not any(re.search(r"(?<![a-z])" + re.escape(k) + r"(?![a-z])", low) for k in keys):
                continue                      # 標題沒提到公司 → 相關性太低，不列
            try:
                pub = email.utils.parsedate_to_datetime(p.group(1))
            except Exception:
                continue
            if (now - pub).total_seconds() > NEWS_HOURS * 3600:
                continue
            if best is None or pub > best[2]:
                best = (title, src, pub)
        if best is None:
            return "none"
        return (best[0], best[1], best[2].astimezone(ET))
    except Exception as e:
        log.warning("%s 新聞抓取失敗：%s", sym, e)
        return None


def stock_events(symbols, today: datetime.date = None):
    """{代號: [(文字, 是否緊急)]}；ETF 只回空清單。"""
    today = today or datetime.datetime.now(ET).date()
    syms = [s for s in dict.fromkeys(symbols) if s not in ETFS]
    out = {s: [] for s in symbols}
    if not syms:
        return out
    with ThreadPoolExecutor(max_workers=6) as ex:
        f_e = {s: ex.submit(fetch_earnings, s, today) for s in syms}
        f_n = {s: ex.submit(fetch_news, s) for s in syms}
        earn = {s: f.result() for s, f in f_e.items()}
        news = {s: f.result() for s, f in f_n.items()}
    for s in syms:
        lines = []
        e = earn.get(s)
        if e is None:
            lines.append(("📅 財報日 —", False))
        elif e[0] != "none" and (e[0] - today).days <= EARNINGS_AHEAD_DAYS:
            d, when = e
            days = (d - today).days
            tag = "今天" if days == 0 else f"{days}天後"
            lines.append((f"📅 財報 {md_wd(d)}{(' ' + when) if when else ''}（{tag}，財報前後跳空風險高）", days <= 3))
        n = news.get(s)
        if n is None:
            lines.append(("📰 新聞 —", False))
        elif n != "none":
            title, src, pub = n
            if len(title) > 90:
                title = title[:89] + "…"
            lines.append((f"📰 {pub.month}/{pub.day} {title}{f'（{src}）' if src else ''}", False))
        out[s] = lines
    return out
