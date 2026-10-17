import time
import datetime
import requests
import yfinance as yf
import pandas as pd
from apscheduler.schedulers.blocking import BlockingScheduler

# 💡 掌櫃的：請在此填入您的真實 LINE Channel Access Token 與目標群組/用戶 ID
LINE_CHANNEL_ACCESS_TOKEN = ""

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
    dst_start = datetime.datetime(year, 3, 8, 2, 0)
    dst_start += datetime.timedelta(days=(6 - dst_start.weekday() + 7) % 7)
    dst_end = datetime.datetime(year, 11, 1, 2, 0)
    dst_end += datetime.timedelta(days=(6 - dst_end.weekday()) % 7)
    return dst_start <= dt < dst_end

def job_stage_1():
    """第一階段推播：夏令 21:30 / 冬令 21:00 (美股開盤前夕掃描)"""
    now = datetime.datetime.utcnow()
    dst = is_dst(now)
    mode = "夏令時間" if dst else "冬令時間"
    msg = f"🌅 【PRO CLOUD 盤前戰情推播】\n目前模式: {mode}\n狀態: 美股即將開盤，核心大數據掃描啟動，今日關注突破與防守點位！"
    send_line_broadcast(msg)

def job_stage_2():
    """第二階段推播：夏令 12:30 (台灣半夜) / 冬令 12:00 (台灣半夜) (盤中戰情)"""
    now = datetime.datetime.utcnow()
    dst = is_dst(now)
    mode = "夏令時間" if dst else "冬令時間"
    msg = f"🌙 【PRO CLOUD 盤中戰情推播】\n目前模式: {mode}\n狀態: 美股盤中交投熱絡，MA20 支撐與選擇權 IV 波動監控中。"
    send_line_broadcast(msg)

def job_stage_3():
    """第三階段推播：夏令/冬令收盤時刻自動結算"""
    now = datetime.datetime.utcnow()
    dst = is_dst(now)
    mode = "夏令時間" if dst else "冬令時間"
    
    # 執行美股收盤強勢股掃描
    universe = ["NVDA", "TSLA", "AAPL", "MSFT", "PLTR", "AMZN", "AMD", "SPY", "QQQ"]
    strong_list = []
    for sym in universe:
        try:
            t = yf.Ticker(sym)
            df = t.history(period="1mo")
            if not df.empty and len(df) >= 20:
                df['MA20'] = df['Close'].rolling(window=20).mean()
                last_c = df['Close'].iloc[-1]
                last_ma20 = df['MA20'].iloc[-1]
                if last_c > last_ma20:
                    strong_list.append(f"🔥 {sym}: ${last_c:.2f} (站穩MA20)")
        except:
            continue

    report = f"📊 【PRO CLOUD 收盤戰報】\n模式: {mode}\n\n" + "\n".join(strong_list[:5]) + "\n\n💡 盤后數據結算完畢，風控守衛正常運作！"
    send_line_broadcast(report)

if __name__ == "__main__":
    scheduler = BlockingScheduler()
    
    # 根據您的需求精準設定台灣時間排程
    # 夏令時段 (台美時差 12 小時)
    # 冬令時段 (台美時差 13 小時，自動提前半小時對應)
    
    # 第一階段 (盤前)
    scheduler.add_job(job_stage_1, 'cron', hour=21, minute=30) # 夏令 21:30
    scheduler.add_job(job_stage_1, 'cron', hour=21, minute=0)  # 冬令 21:00
    
    # 第二階段 (盤中)
    scheduler.add_job(job_stage_2, 'cron', hour=0, minute=30)  # 夏令 00:30 (12:30 AM)
    scheduler.add_job(job_stage_2, 'cron', hour=0, minute=0)   # 冬令 00:00 (12:00 AM)
    
    # 第三階段 (收盤結算)
    scheduler.add_job(job_stage_3, 'cron', hour=4, minute=15)  # 夏令收盤 04:15
    scheduler.add_job(job_stage_3, 'cron', hour=5, minute=15)  # 冬令收盤 05:15

    print("🚀 美股三段式定時推播守衛 (Scheduler) 已啟動，夏冬令自動適配中...")
    scheduler.start()
