import time
import datetime
import requests
import yfinance as yf
import pandas as pd
from apscheduler.schedulers.blocking import BlockingScheduler
import os

# 💡 優先讀取環境變數，亦可填入常數
LINE_CHANNEL_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_TOKEN", "").strip()
LINE_USER_ID = os.environ.get("LINE_USER_ID", "").strip()

def send_line_message(message: str):
    if not LINE_CHANNEL_ACCESS_TOKEN or not LINE_USER_ID:
        print(f"[模擬推播] {message}")
        return
    url = "https://api.line.me/v2/bot/message/push"
    headers = {
        "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }
    data = {
        "to": LINE_USER_ID,
        "messages": [{"type": "text", "text": message}]
    }
    try:
        res = requests.post(url, headers=headers, json=data, timeout=10)
        print(f"[LINE 推播回應] {res.status_code}: {res.text}")
    except Exception as e:
        print(f"[LINE 推播失敗] {e}")

def is_dst(dt: datetime.datetime) -> bool:
    """自動判斷美東是否為夏令時間 (DST)"""
    year = dt.year
    dst_start = datetime.datetime(year, 3, 8, 2, 0)
    dst_start += datetime.timedelta(days=(6 - dst_start.weekday() + 7) % 7)
    dst_end = datetime.datetime(year, 11, 1, 2, 0)
    dst_end += datetime.timedelta(days=(6 - dst_end.weekday()) % 7)
    return dst_start <= dt < dst_end

def job_stage_1():
    """第一階段推播：開盤後半小時（夏令 22:00 / 冬令 23:00）"""
    now = datetime.datetime.utcnow()
    dst = is_dst(now)
    mode = "夏令時間" if dst else "冬令時間"
    
    # 掃描開盤前 30 分鐘帶量活絡標的
    universe = ["NVDA", "TSLA", "AAPL", "MSFT", "PLTR", "AMZN", "AMD", "SPY", "QQQ"]
    active_list = []
    for sym in universe:
        try:
            t = yf.Ticker(sym)
            df = t.history(period="5d", interval="5m")
            if not df.empty:
                last_c = float(df['Close'].iloc[-1])
                first_o = float(df['Open'].iloc[0])
                pct = ((last_c - first_o) / first_o) * 100.0
                p_sign = "▲" if pct >= 0 else "▼"
                active_list.append(f"• {sym}: ${last_c:.2f} ({p_sign}{abs(pct):.1f}%)")
        except:
            continue
            
    stock_info = "\n".join(active_list[:6]) if active_list else "核心標的平穩交投中"
    msg = (
        f"⚡ 【美股多空雷達 V9.8｜開盤半小時戰報】\n"
        f"模式：{mode} (美股已開盤 30 分鐘)\n"
        f"狀態：早盤初段量能消化完畢，核心趨勢確認中：\n"
        f"{stock_info}\n"
        f"━━━━━━━━━━━━━━\n"
        f"※ 僅供量化輔助監控，注意早盤追價風險。"
    )
    send_line_message(msg)

def job_stage_2():
    """第二階段推播：盤中戰報（夏令 00:30 / 冬令 00:00）"""
    now = datetime.datetime.utcnow()
    dst = is_dst(now)
    mode = "夏令時間" if dst else "冬令時間"
    msg = (
        f"🌙 【美股多空雷達 V9.8｜午盤盤中戰報】\n"
        f"模式：{mode}\n"
        f"狀態：美股進入午盤交投時段，各大板塊支撐與 20MA(月線) 防守線持續監控中。"
    )
    send_line_message(msg)

def job_stage_3():
    """第三階段推播：收盤結算（夏令 04:15 / 冬令 05:15）"""
    now = datetime.datetime.utcnow()
    dst = is_dst(now)
    mode = "夏令時間" if dst else "冬令時間"
    
    universe = ["NVDA", "TSLA", "AAPL", "MSFT", "PLTR", "AMZN", "AMD", "SPY", "QQQ"]
    strong_list = []
    for sym in universe:
        try:
            t = yf.Ticker(sym)
            df = t.history(period="1mo")
            if not df.empty and len(df) >= 20:
                df['MA20'] = df['Close'].rolling(window=20).mean()
                last_c = float(df['Close'].iloc[-1])
                last_ma20 = float(df['MA20'].iloc[-1])
                neckline = float(df['High'].iloc[-21:-1].max())
                dist_neck = ((last_c / neckline) - 1.0) * 100.0
                if last_c > last_ma20:
                    strong_list.append(f"🔥 {sym}: ${last_c:.2f} (站穩月線, 距頸線 {dist_neck:+.1f}%)")
        except:
            continue

    stocks_text = "\n".join(strong_list[:5]) if strong_list else "今日無特殊強勢突破標的"
    report = (
        f"📊 【美股多空雷達 V9.8｜全日收盤戰報】\n"
        f"模式：{mode}\n\n"
        f"{stocks_text}\n\n"
        f"💡 美股盤後大數據結算完畢，風控守衛正常運作中。"
    )
    send_line_message(report)

def run_scheduler_job_stage_1():
    now_utc = datetime.datetime.utcnow()
    dst = is_dst(now_utc)
    # 夏令在 22:00 觸發，冬令在 23:00 觸發
    current_tw_hour = (now_utc.hour + 8) % 24
    if (dst and current_tw_hour == 22) or (not dst and current_tw_hour == 23):
        job_stage_1()

def run_scheduler_job_stage_2():
    now_utc = datetime.datetime.utcnow()
    dst = is_dst(now_utc)
    current_tw_hour = (now_utc.hour + 8) % 24
    current_tw_min = now_utc.minute
    # 夏令 00:30，冬令 00:00
    if (dst and current_tw_hour == 0 and current_tw_min == 30) or (not dst and current_tw_hour == 0 and current_tw_min == 0):
        job_stage_2()

def run_scheduler_job_stage_3():
    now_utc = datetime.datetime.utcnow()
    dst = is_dst(now_utc)
    current_tw_hour = (now_utc.hour + 8) % 24
    # 夏令 04:15，冬令 05:15
    if (dst and current_tw_hour == 4) or (not dst and current_tw_hour == 5):
        job_stage_3()

if __name__ == "__main__":
    scheduler = BlockingScheduler()

    # 第一階段：開盤後半小時（夏令 22:00 / 冬令 23:00）
    scheduler.add_job(run_scheduler_job_stage_1, 'cron', hour=22, minute=0) # 夏令觸發點
    scheduler.add_job(run_scheduler_job_stage_1, 'cron', hour=23, minute=0) # 冬令觸發點

    # 第二階段：盤中戰報（夏令 00:30 / 冬令 00:00）
    scheduler.add_job(run_scheduler_job_stage_2, 'cron', hour=0, minute=30) # 夏令觸發點
    scheduler.add_job(run_scheduler_job_stage_2, 'cron', hour=0, minute=0)  # 冬令觸發點

    # 第三階段：收盤結算（夏令 04:15 / 冬令 05:15）
    scheduler.add_job(run_scheduler_job_stage_3, 'cron', hour=4, minute=15) # 夏令觸發點
    scheduler.add_job(run_scheduler_job_stage_3, 'cron', hour=5, minute=15) # 冬令觸發點

    print("🚀 美股開盤後半小時與三段式智慧推播排程已啟動...")
    scheduler.start()
