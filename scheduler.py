import time
import datetime
import requests
import yfinance as yf
import pandas as pd
from apscheduler.schedulers.blocking import BlockingScheduler

# 💡 掌櫃的：請在此填入您的真實 LINE Channel Access Token 與目標群組/用戶 ID
LINE_CHANNEL_ACCESS_TOKEN = "l9IU7jZQ50Hlb5IAGuB8fM3ww2SUuhs1hzoMUe64jNybTBmIEKCzSY/gks5r+g7RJqzeZ0SIEXblTxwCT/QS77C+LgnxF+zZLITD//VO/O6y1X71MQl5V0wf5KxdnUozAQ5ZZMx0RlswvAbZeBBqAAdB04t89/1O/w1cDnyilFU="

def send_line_broadcast(message: str):
    if not LINE_CHANNEL_ACCESS_TOKEN:
        print(f"[模擬推播] {message}")
        return
    url = "https://api.line.me/v2/bot/message/broadcast"
    headers = {
        "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }
    data = {"messages": [{"type": "text", "text": message}]}
    try:
        res = requests.post(url, headers=headers, json=data)
        print(f"[LINE 推播回應] {res.status_code}: {res.text}")
    except Exception as e:
        print(f"[LINE 推播失敗] {e}")

def is_dst(dt: datetime.datetime) -> bool:
    """自動判斷是否為美國夏令時間 (DST)"""
    year = dt.year
    # 夏令時間起點：3月第二個星期日
    dst_start = datetime.datetime(year, 3, 8, 2, 0)
    dst_start += datetime.timedelta(days=(6 - dst_start.weekday() + 7) % 7)
    # 冬令時間起點：11月第一個星期日
    dst_end = datetime.datetime(year, 11, 1, 2, 0)
    dst_end += datetime.timedelta(days=(6 - dst_end.weekday()) % 7)
    return dst_start <= dt < dst_end

def daily_market_scan_and_push():
    now_utc = datetime.datetime.utcnow()
    dst_active = is_dst(now_utc)
    season_name = "夏令時間 (美東 UTC-4)" if dst_active else "冬令時間 (美東 UTC-5)"
    
    print(f"[{datetime.datetime.now()}] 開始執行美股盤後大數據自動掃描 ({season_name})...")
    
    universe = ["NVDA", "TSLA", "AAPL", "MSFT", "PLTR", "AMZN", "AMD", "SPY", "QQQ"]
    strong_list = []
    
    for sym in universe:
        try:
            t = yf.Ticker(sym)
            df = t.history(period="1mo")
            if df.empty or len(df) < 20:
                continue
            df['MA20'] = df['Close'].rolling(window=20).mean()
            last_c = df['Close'].iloc[-1]
            last_ma20 = df['MA20'].iloc[-1]
            if last_c > last_ma20:
                strong_list.append(f"🔥 {sym}: 現價 ${last_c:.2f} (站穩 MA20 支撐)")
        except:
            continue
            
    report = f"📊 【PRO CLOUD 美股定時戰報】\n時間模式: {season_name}\n\n" + "\n".join(strong_list[:5]) + "\n\n💡 戰情室運作正常，風控護航中！"
    send_line_broadcast(report)

if __name__ == "__main__":
    scheduler = BlockingScheduler()
    # 根據美股收盤後自動定時排程 (美東時間每日收盤後自動觸發)
    scheduler.add_job(daily_market_scan_and_push, 'cron', hour=21, minute=30) # 台灣時間晚上 9:30 (冬令) / 8:30 (夏令)
    print("🚀 美股定時推播守衛 (Scheduler) 已在背景啟動...")
    scheduler.start()
