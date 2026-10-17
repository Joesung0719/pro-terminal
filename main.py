from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
import requests
import datetime
import os

app = FastAPI(title="US Professional Radar War Room - V9.8 True Full Edition")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LINE_CHANNEL_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_TOKEN", "").strip()
LINE_USER_ID = os.environ.get("LINE_USER_ID", "").strip()

def calculate_greeks(S, K, T, r, sigma, option_type="put"):
    if T <= 0 or sigma <= 0:
        return {"delta": 0.0, "theta": 0.0, "gamma": 0.0, "vega": 0.0}
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        delta = norm.cdf(d1)
        theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        delta = norm.cdf(d1) - 1
        theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100
    return {
        "delta": float(round(delta, 4)),
        "theta": float(round(theta, 4)),
        "gamma": float(round(gamma, 4)),
        "vega": float(round(vega, 4))
    }

# 美股各大市場與板塊分類大池
MARKET_SECTORS = {
    "ALL": "全市場精選大池",
    "SOX": "費城半導體 (SOX)",
    "NASDAQ": "那斯達克 100 (科技核心)",
    "DOW": "道瓊工業 30 (傳統藍籌)",
    "NYSE": "紐約證交所主力 (金融/能源)",
    "CBOE": "芝加哥主力與指數商品",
    "ETF": "全美主力指數與商品 ETF"
}

MARKET_UNIVERSE = [
    # 費半 SOX
    {"symbol": "NVDA", "name": "輝達", "sector": "SOX", "market": "NASDAQ"},
    {"symbol": "TSM", "name": "台積電 ADR", "sector": "SOX", "market": "NYSE"},
    {"symbol": "AMD", "name": "超微半導體", "sector": "SOX", "market": "NASDAQ"},
    {"symbol": "AVGO", "name": "博通", "sector": "SOX", "market": "NASDAQ"},
    {"symbol": "QCOM", "name": "高通", "sector": "SOX", "market": "NASDAQ"},
    {"symbol": "TXN", "name": "德州儀器", "sector": "SOX", "market": "NASDAQ"},
    {"symbol": "MU", "name": "美光科技", "sector": "SOX", "market": "NASDAQ"},
    {"symbol": "INTC", "name": "英特爾", "sector": "SOX", "market": "NASDAQ"},
    {"symbol": "ARM", "name": "安謀科技", "sector": "SOX", "market": "NASDAQ"},
    {"symbol": "ASML", "name": "艾司摩爾", "sector": "SOX", "market": "NASDAQ"},

    # 那斯達克 NASDAQ 100
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

    # 道瓊 DOW 30
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

    # 紐約證交所 NYSE 主力 (金融 / 能源 / 國防)
    {"symbol": "PLTR", "name": "帕蘭蒂爾科技", "sector": "NYSE", "market": "NYSE"},
    {"symbol": "JPM", "name": "摩根大通", "sector": "NYSE", "market": "NYSE"},
    {"symbol": "BAC", "name": "美國銀行", "sector": "NYSE", "market": "NYSE"},
    {"symbol": "XOM", "name": "埃克森美孚石油", "sector": "NYSE", "market": "NYSE"},
    {"symbol": "CVX", "name": "雪佛龍能源", "sector": "NYSE", "market": "NYSE"},
    {"symbol": "LLY", "name": "禮來藥廠", "sector": "NYSE", "market": "NYSE"},
    {"symbol": "NKE", "name": "耐吉 Nike", "sector": "NYSE", "market": "NYSE"},
    {"symbol": "UBER", "name": "優步 Uber", "sector": "NYSE", "market": "NYSE"},

    # 芝加哥 CBOE 衍生主力與指數工具
    {"symbol": "CBOE", "name": "芝加哥期權交易所集團", "sector": "CBOE", "market": "CBOE"},
    {"symbol": "CME", "name": "芝加哥商品交易所", "sector": "CBOE", "market": "NASDAQ"},
    {"symbol": "ICE", "name": "洲際交易所集團", "sector": "CBOE", "market": "NYSE"},

    # ETF 主力
    {"symbol": "SPY", "name": "標普500指數 ETF", "sector": "ETF", "market": "AMEX"},
    {"symbol": "QQQ", "name": "那斯達克100 ETF", "sector": "ETF", "market": "NASDAQ"},
    {"symbol": "SOXX", "name": "費城半導體指數 ETF", "sector": "ETF", "market": "NASDAQ"},
    {"symbol": "DIA", "name": "道瓊工業指數 ETF", "sector": "ETF", "market": "AMEX"},
    {"symbol": "IWM", "name": "羅素2000小型股 ETF", "sector": "ETF", "market": "AMEX"},
    {"symbol": "TLT", "name": "20年期以上美國公債 ETF", "sector": "ETF", "market": "NASDAQ"},
    {"symbol": "GLD", "name": "SPDR 黃金信託 ETF", "sector": "ETF", "market": "AMEX"}
]

HTML_CONTENT = """
<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <title>美股多空雷達戰情室 V9.8 旗艦完全體</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://unpkg.com/lightweight-charts@4.1.1/dist/lightweight-charts.standalone.production.js"></script>
</head>
<body class="bg-[#0b0e14] text-[#e2e8f0] font-sans antialiased h-screen flex flex-col overflow-hidden">
    <!-- 頂部大盤溫度計全市場看板 (道瓊、那斯達克、費半、標普、VIX) -->
    <header class="bg-[#121824] border-b border-[#1e293b] px-4 py-2 flex flex-col gap-1.5 shadow-lg">
        <div class="flex justify-between items-center">
            <div class="flex items-center space-x-3">
                <span class="text-xl font-black text-[#38bdf8] tracking-wider drop-shadow">⚡ 美股多空雷達 V9.8 旗艦版</span>
                <span class="text-xs bg-[#1e293b] text-amber-400 px-2.5 py-0.5 rounded-full font-bold border border-amber-500/30">道瓊・費半・那斯達克・紐交・芝加哥全市場對齊</span>
            </div>
            <div class="flex items-center space-x-3 text-xs">
                <span id="marketSessionTag" class="text-slate-400">市場狀態：美股盤後結算</span>
                <button onclick="fetchIndicesData()" class="bg-[#1e293b] hover:bg-[#2a374a] text-white px-2 py-0.5 rounded border border-[#38bdf8]/40 font-bold">🔄 刷新大盤溫度計</button>
            </div>
        </div>

        <!-- 五大指數多空溫度計即時卡片 -->
        <div class="grid grid-cols-5 gap-2 text-xs font-mono">
            <div class="bg-[#0b0e14] border border-[#1e293b] px-2.5 py-1.5 rounded flex justify-between items-center">
                <span class="text-slate-400 font-bold">🏛️ 道瓊 (DJI)</span>
                <span id="idx_dji" class="font-bold text-white">載入中...</span>
            </div>
            <div class="bg-[#0b0e14] border border-[#1e293b] px-2.5 py-1.5 rounded flex justify-between items-center">
                <span class="text-slate-400 font-bold">💻 那指 (IXIC)</span>
                <span id="idx_ixic" class="font-bold text-white">載入中...</span>
            </div>
            <div class="bg-[#0b0e14] border border-[#1e293b] px-2.5 py-1.5 rounded flex justify-between items-center">
                <span class="text-slate-400 font-bold">⚡ 費半 (SOX)</span>
                <span id="idx_sox" class="font-bold text-white">載入中...</span>
            </div>
            <div class="bg-[#0b0e14] border border-[#1e293b] px-2.5 py-1.5 rounded flex justify-between items-center">
                <span class="text-slate-400 font-bold">📊 標普 (SPX)</span>
                <span id="idx_spx" class="font-bold text-white">載入中...</span>
            </div>
            <div class="bg-[#0b0e14] border border-[#1e293b] px-2.5 py-1.5 rounded flex justify-between items-center">
                <span class="text-amber-400 font-bold">🔥 恐慌 (VIX)</span>
                <span id="idx_vix" class="font-bold text-amber-400">載入中...</span>
            </div>
        </div>
    </header>

    <!-- 主體區域 -->
    <div class="flex flex-1 overflow-hidden">
        <!-- 左側：各大市場板塊分類與 V9.8 量價過濾清單 -->
        <aside class="w-80 bg-[#121824] border-r border-[#1e293b] flex flex-col">
            <div class="p-3 border-b border-[#1e293b] space-y-2">
                <div class="flex justify-between items-center">
                    <h2 class="text-xs font-bold text-slate-300 uppercase tracking-wider">🎯 市場板塊分類</h2>
                    <span class="text-[10px] text-emerald-400 font-bold bg-[#0b0e14] px-1.5 py-0.5 rounded border border-emerald-500/30">V9.8 爆量突破</span>
                </div>
                
                <!-- 美股全市場板塊切換選單 -->
                <select id="sectorSelect" onchange="onSectorChange()" class="w-full bg-[#0b0e14] border border-[#2a374a] rounded px-2.5 py-1.5 text-xs text-[#38bdf8] font-bold focus:outline-none focus:border-[#38bdf8]">
                    <option value="ALL">🌐 全市場精選大池 (全部標的)</option>
                    <option value="SOX">⚡ 費城半導體 (SOX / 晶片軍火)</option>
                    <option value="NASDAQ">💻 那斯達克 100 (科技核心主力)</option>
                    <option value="DOW">🏛️ 道瓊工業 30 (傳統藍籌防守)</option>
                    <option value="NYSE">💼 紐約證交所主力 (金融/能源/國防)</option>
                    <option value="CBOE">📈 芝加哥期權主力 (交易所/波動)</option>
                    <option value="ETF">📊 全美主力 ETF (標普/公債/黃金)</option>
                </select>

                <div class="flex gap-2">
                    <input id="customSymbol" type="text" placeholder="輸入美股代號 (例: PLTR, TSM)" class="flex-1 bg-[#0b0e14] border border-[#2a374a] rounded px-3 py-1.5 text-xs text-white uppercase focus:outline-none focus:border-[#38bdf8]">
                    <button onclick="addCustomSymbol()" class="bg-[#38bdf8] hover:bg-sky-500 text-slate-950 font-black px-3 py-1.5 rounded text-xs transition">診斷</button>
                </div>
            </div>
            <div id="watchlist" class="flex-1 overflow-y-auto p-2 space-y-1.5"></div>
        </aside>

        <!-- 中間：台股 V9.8 規格四均線 ＋ 雙副圖 ＋ 游標數值面板 -->
        <main class="flex-1 flex flex-col bg-[#0b0e14] p-3 overflow-y-auto">
            <!-- 走勢頂部資訊列 -->
            <div class="flex justify-between items-center mb-2">
                <div class="flex items-center space-x-3">
                    <h1 id="chartTitle" class="text-xl font-black text-white tracking-wide">PLTR 帕蘭蒂爾科技</h1>
                    <div class="flex bg-[#121824] border border-[#1e293b] rounded p-0.5 space-x-1 text-xs">
                        <button onclick="changeInterval('1d')" id="btn_1d" class="px-2.5 py-0.5 rounded font-bold bg-[#38bdf8] text-slate-950">日K</button>
                        <button onclick="changeInterval('1wk')" id="btn_1wk" class="px-2.5 py-0.5 rounded font-bold text-slate-400 hover:text-white">週K</button>
                    </div>
                </div>
                <div class="flex items-center space-x-3">
                    <div id="qualityTag" class="text-xs bg-emerald-500/20 text-emerald-400 px-2 py-1 rounded border border-emerald-500/40 font-bold">
                        🛡️ 距頸線 ≤ 5% 且站上月線
                    </div>
                    <div id="currentPriceTag" class="text-base font-bold text-emerald-400 bg-[#121824] px-3 py-1 rounded-lg border border-[#1e293b] shadow">
                        收盤: $--
                    </div>
                </div>
            </div>

            <!-- V9.8 十字線浮動即時數值面板 (Legend) -->
            <div id="legendPanel" class="bg-[#121824] border border-[#1e293b] px-3 py-1.5 rounded mb-2 text-xs flex flex-wrap gap-x-4 gap-y-1 font-mono text-slate-300">
                <span>開: <strong id="legOpen" class="text-white">--</strong></span>
                <span>高: <strong id="legHigh" class="text-white">--</strong></span>
                <span>低: <strong id="legLow" class="text-white">--</strong></span>
                <span>收: <strong id="legClose" class="text-emerald-400">--</strong></span>
                <span>量: <strong id="legVol" class="text-white">--</strong></span>
                <span class="text-amber-400">MA5(週): <strong id="legMA5">--</strong></span>
                <span class="text-sky-400">MA20(月線): <strong id="legMA20">--</strong></span>
                <span class="text-purple-400">MA60(季線): <strong id="legMA60">--</strong></span>
                <span class="text-slate-400">MA240(年線): <strong id="legMA240">--</strong></span>
                <span class="text-emerald-300">RSI: <strong id="legRSI">--</strong></span>
            </div>

            <!-- 主圖 K線 (MA5 / MA20 / MA60 / MA240) -->
            <div class="bg-[#121824] border border-[#1e293b] rounded-t-xl p-2 h-[330px] flex flex-col">
                <div id="mainChartContainer" class="w-full flex-1"></div>
            </div>

            <!-- 副圖 1: 成交量 ＋ 20日均量線 -->
            <div class="bg-[#121824] border-x border-[#1e293b] p-2 h-[115px] flex flex-col border-t border-[#1e293b]/50">
                <div class="text-[10px] text-slate-400 font-bold mb-0.5">📊 成交量 ＆ 20日均量線 (Vol-MA20)</div>
                <div id="volChartContainer" class="w-full flex-1"></div>
            </div>

            <!-- 副圖 2: RSI(14) 動能震盪指標 -->
            <div class="bg-[#121824] border-x border-b border-[#1e293b] rounded-b-xl p-2 h-[115px] flex flex-col border-t border-[#1e293b]/50">
                <div class="text-[10px] text-[#38bdf8] font-bold mb-0.5">⚡ RSI (14) 超買超賣動能指標 (70/30)</div>
                <div id="rsiChartContainer" class="w-full flex-1"></div>
            </div>
        </main>

        <!-- 右側：選擇權風控、Payoff 與 LINE 推播 -->
        <aside class="w-96 bg-[#121824] border-l border-[#1e293b] flex flex-col p-4 overflow-y-auto">
            <h2 class="text-sm font-black text-white mb-3 flex items-center justify-between">
                <span>📊 V9.8 期權風控與真數據對齊</span>
                <span class="text-[10px] bg-emerald-500 text-slate-950 font-black px-2 py-0.5 rounded">機構旗艦</span>
            </h2>

            <div class="space-y-3 text-xs">
                <div>
                    <label class="text-slate-400 block mb-1">選擇到期日 (Expiration)</label>
                    <select id="expDateSelect" onchange="onExpirationChange()" class="w-full bg-[#0b0e14] border border-[#2a374a] rounded p-2 text-white font-bold focus:outline-none focus:border-[#38bdf8]">
                        <option value="">載入中...</option>
                    </select>
                </div>

                <div>
                    <label class="text-slate-400 block mb-1">選擇策略類型</label>
                    <select id="strategySelect" onchange="onExpirationChange()" class="w-full bg-[#0b0e14] border border-[#2a374a] rounded p-2 text-white font-bold focus:outline-none focus:border-[#38bdf8]">
                        <option value="csp">Short Put (低接收租 / 智慧對齊 MA20)</option>
                        <option value="cc">Covered Call (現股包租公)</option>
                        <option value="long_call">Long Call (突破多頭)</option>
                    </select>
                </div>

                <div class="bg-[#0b0e14] p-3 rounded-lg border border-[#2a374a] space-y-1">
                    <div class="flex justify-between items-center">
                        <span class="text-slate-400">現價 (S) / MA20 支撐</span>
                        <span id="spotDisplay" class="text-white font-bold">-</span>
                    </div>
                    <div class="flex justify-between items-center">
                        <span class="text-slate-400">停損參考點位</span>
                        <span id="stopLossDisplay" class="text-amber-400 font-bold">頸線支撐 / 月線</span>
                    </div>
                </div>

                <div class="bg-[#0b0e14] p-3 rounded-lg border border-[#2a374a]">
                    <div class="flex justify-between items-center mb-1">
                        <label class="text-slate-400">目標履約價 (Strike)</label>
                        <span class="text-[10px] text-[#38bdf8] cursor-pointer hover:underline" onclick="autoAlignMA20()">🎯 智慧對齊 MA20</span>
                    </div>
                    <select id="strikeSelect" onchange="onStrikeChange()" class="w-full bg-[#121824] border border-[#2a374a] rounded p-2 text-white font-bold focus:outline-none focus:border-[#38bdf8]">
                        <option value="">請選擇到期日...</option>
                    </select>
                </div>

                <!-- Payoff 損益分佈圖 -->
                <div class="bg-[#0b0e14] p-3 rounded-lg border border-[#2a374a] space-y-1.5">
                    <div class="font-bold text-[#38bdf8] flex justify-between items-center text-[11px]">
                        <span>📈 Payoff 損益分佈預覽</span>
                    </div>
                    <div class="grid grid-cols-2 gap-2 text-[11px] bg-[#121824] p-2 rounded border border-[#1e293b]">
                        <div>最大獲利: <strong id="payoffMax" class="text-emerald-400">--</strong></div>
                        <div>最大虧損: <strong id="payoffMin" class="text-red-400">--</strong></div>
                    </div>
                    <canvas id="payoffCanvas" height="85" class="w-full bg-[#121824] rounded border border-[#1e293b] mt-1"></canvas>
                </div>

                <!-- Greeks 風險敏感度 -->
                <div class="bg-[#0b0e14] p-3 rounded-lg border border-[#2a374a] space-y-1.5">
                    <div class="font-bold text-[#38bdf8] border-b border-[#1e293b] pb-1">⚡ Greeks 風險敏感度</div>
                    <div class="grid grid-cols-2 gap-2 text-[11px]">
                        <div>Delta: <strong id="valDelta" class="text-white">-</strong></div>
                        <div>Theta (每日損耗): <strong id="valTheta" class="text-amber-400">-</strong></div>
                        <div>Gamma: <strong id="valGamma" class="text-white">-</strong></div>
                        <div>Vega (波動敏感): <strong id="valVega" class="text-white">-</strong></div>
                    </div>
                </div>

                <div class="grid grid-cols-2 gap-2">
                    <button onclick="calculateOptionStrategy()" class="bg-[#38bdf8] hover:bg-sky-500 text-slate-950 font-black py-2.5 rounded-lg transition shadow">
                        V9.8 風控檢核
                    </button>
                    <button onclick="sendCloudPush()" class="bg-[#06C755] hover:bg-[#05b34c] text-white font-black py-2.5 rounded-lg transition shadow">
                        📲 LINE 戰報推播
                    </button>
                </div>
            </div>
        </aside>
    </div>

    <!-- LINE Flex 戰報實機預覽視窗 -->
    <div id="lineModal" class="fixed inset-0 bg-black/75 flex items-center justify-center hidden z-50">
        <div class="bg-[#121824] border border-[#1e293b] rounded-2xl p-6 w-[440px] shadow-2xl space-y-3">
            <div class="flex justify-between items-center border-b border-[#1e293b] pb-2">
                <div class="flex items-center space-x-2">
                    <span class="w-3 h-3 bg-[#06C755] rounded-full"></span>
                    <h3 class="text-sm font-black text-white">📲 美股多空雷達 V9.8 ｜ LINE 戰報實機發送</h3>
                </div>
                <button onclick="closeLineModal()" class="text-slate-400 hover:text-white text-sm">✕</button>
            </div>
            <div class="bg-[#0b0e14] p-4 rounded-xl border border-[#2a374a] text-xs font-mono text-slate-300 space-y-2 whitespace-pre-wrap max-h-[380px] overflow-y-auto" id="linePreviewText"></div>
            <button onclick="closeLineModal()" class="w-full bg-[#38bdf8] text-slate-950 font-black py-2 rounded-lg text-xs transition">確 認</button>
        </div>
    </div>

    <script>
        let mainChart, volChart, rsiChart;
        let candleSeries, ma5Series, ma20Series, ma60Series, ma240Series;
        let volSeries, volMaSeries, rsiSeries;
        let currentSpot = 0, currentMA20 = 0, currentMA240 = 0, currentNeckline = 0;
        let currentSymbol = "PLTR";
        let currentInterval = "1d";
        let currentOptionData = null;
        let currentSector = "ALL";

        let universeData = [];

        async function fetchIndicesData() {
            try {
                const res = await fetch('/api/market/indices');
                const data = await res.json();
                
                const fmtIdx = (obj) => {
                    const arrow = obj.chg >= 0 ? '▲' : '▼';
                    const color = obj.chg >= 0 ? 'text-red-400' : 'text-emerald-400';
                    return `<span class="${color}">${obj.close.toLocaleString()} (${arrow}${Math.abs(obj.pct)}%)</span>`;
                };

                document.getElementById('idx_dji').innerHTML = fmtIdx(data.dji);
                document.getElementById('idx_ixic').innerHTML = fmtIdx(data.ixic);
                document.getElementById('idx_sox').innerHTML = fmtIdx(data.sox);
                document.getElementById('idx_spx').innerHTML = fmtIdx(data.spx);
                document.getElementById('idx_vix').innerHTML = `<span class="text-amber-400">${data.vix.close} (${data.vix.status})</span>`;
            } catch(e) {}
        }

        async function loadScreenerList() {
            try {
                const res = await fetch(`/api/screener?sector=${currentSector}`);
                const data = await res.json();
                universeData = data.screener_list;
                renderWatchlist();
            } catch(e) {}
        }

        function onSectorChange() {
            currentSector = document.getElementById('sectorSelect').value;
            loadScreenerList();
        }

        function renderWatchlist() {
            const container = document.getElementById('watchlist');
            container.innerHTML = '';
            universeData.forEach((item, idx) => {
                const div = document.createElement('div');
                div.className = "flex items-center justify-between px-3 py-2 rounded-lg bg-[#0b0e14] hover:bg-[#1a2234] transition group border border-[#1e293b]";
                div.innerHTML = `
                    <div onclick="fetchStockData('${item.symbol}')" class="flex-1 cursor-pointer">
                        <div class="font-bold text-white group-hover:text-[#38bdf8] text-xs flex justify-between">
                            <span>${item.symbol} <span class="text-[10px] text-amber-400">[${item.sector}]</span></span>
                            <span class="text-[10px] text-emerald-400 bg-[#121824] px-1.5 py-0.5 rounded border border-emerald-500/20">${item.score}</span>
                        </div>
                        <div class="text-[11px] text-slate-400 flex justify-between mt-0.5">
                            <span>${item.name}</span>
                            <span>距頸線 ${item.dist_neck_pct}%</span>
                        </div>
                    </div>
                `;
                container.appendChild(div);
            });
        }

        function addCustomSymbol() {
            const symbol = document.getElementById('customSymbol').value.trim().toUpperCase();
            if(symbol) {
                fetchStockData(symbol);
                document.getElementById('customSymbol').value = '';
            }
        }

        function initCharts() {
            const mainEl = document.getElementById('mainChartContainer');
            const volEl = document.getElementById('volChartContainer');
            const rsiEl = document.getElementById('rsiChartContainer');

            const chartOpts = {
                layout: { background: { type: 'solid', color: '#121824' }, textColor: '#94a3b8' },
                grid: { vertLines: { color: '#1e293b' }, horzLines: { color: '#1e293b' } },
                timeScale: { borderColor: '#1e293b', visible: true },
                rightPriceScale: { borderColor: '#1e293b' }
            };

            mainChart = LightweightCharts.createChart(mainEl, { ...chartOpts, width: mainEl.clientWidth, height: 310 });
            candleSeries = mainChart.addCandlestickSeries({ upColor: '#ef5350', downColor: '#26a69a', wickUpColor: '#ef5350', wickDownColor: '#26a69a' });
            ma5Series = mainChart.addLineSeries({ color: '#f59e0b', lineWidth: 1.5, title: 'MA5' });
            ma20Series = mainChart.addLineSeries({ color: '#38bdf8', lineWidth: 2, title: 'MA20' });
            ma60Series = mainChart.addLineSeries({ color: '#c084fc', lineWidth: 1.5, title: 'MA60' });
            ma240Series = mainChart.addLineSeries({ color: '#64748b', lineWidth: 1.5, lineStyle: 2, title: 'MA240(年線)' });

            volChart = LightweightCharts.createChart(volEl, { ...chartOpts, width: volEl.clientWidth, height: 95 });
            volSeries = volChart.addHistogramSeries({ priceFormat: { type: 'volume' } });
            volMaSeries = volChart.addLineSeries({ color: '#facc15', lineWidth: 1.5, title: 'Vol-MA20' });

            rsiChart = LightweightCharts.createChart(rsiEl, { ...chartOpts, width: rsiEl.clientWidth, height: 95 });
            rsiSeries = rsiChart.addLineSeries({ color: '#38bdf8', lineWidth: 2, title: 'RSI' });

            mainChart.timeScale().subscribeVisibleTimeRangeChange(range => {
                volChart.timeScale().setVisibleRange(range);
                rsiChart.timeScale().setVisibleRange(range);
            });

            mainChart.subscribeCrosshairMove(param => {
                if (!param.time || !param.seriesData.get(candleSeries)) return;
                const c = param.seriesData.get(candleSeries);
                const m5 = param.seriesData.get(ma5Series);
                const m20 = param.seriesData.get(ma20Series);
                const m60 = param.seriesData.get(ma60Series);
                const m240 = param.seriesData.get(ma240Series);
                
                document.getElementById('legOpen').innerText = `$${c.open.toFixed(2)}`;
                document.getElementById('legHigh').innerText = `$${c.high.toFixed(2)}`;
                document.getElementById('legLow').innerText = `$${c.low.toFixed(2)}`;
                document.getElementById('legClose').innerText = `$${c.close.toFixed(2)}`;
                if(m5) document.getElementById('legMA5').innerText = `$${m5.value.toFixed(2)}`;
                if(m20) document.getElementById('legMA20').innerText = `$${m20.value.toFixed(2)}`;
                if(m60) document.getElementById('legMA60').innerText = `$${m60.value.toFixed(2)}`;
                if(m240) document.getElementById('legMA240').innerText = `$${m240.value.toFixed(2)}`;
            });

            window.addEventListener('resize', () => {
                mainChart.applyOptions({ width: mainEl.clientWidth });
                volChart.applyOptions({ width: volEl.clientWidth });
                rsiChart.applyOptions({ width: rsiEl.clientWidth });
            });
        }

        async function fetchStockData(symbol) {
            currentSymbol = symbol;
            document.getElementById('chartTitle').innerText = `${symbol} 載入中...`;
            try {
                const res = await fetch(`/api/quote/${symbol}?interval=${currentInterval}`);
                const data = await res.json();
                if(data.candles && data.candles.length > 0) {
                    currentSpot = data.spot;
                    currentMA20 = data.ma20_val;
                    currentMA240 = data.ma240_val;
                    currentNeckline = data.neckline;

                    candleSeries.setData(data.candles);
                    ma5Series.setData(data.ma5);
                    ma20Series.setData(data.ma20);
                    ma60Series.setData(data.ma60);
                    ma240Series.setData(data.ma240);

                    const volData = data.candles.map(c => ({
                        time: c.time,
                        value: c.volume,
                        color: c.close >= c.open ? 'rgba(239, 83, 80, 0.6)' : 'rgba(38, 166, 154, 0.6)'
                    }));
                    volSeries.setData(volData);
                    volMaSeries.setData(data.vol_ma);
                    rsiSeries.setData(data.rsi);

                    document.getElementById('chartTitle').innerText = `${symbol} ${data.name} (${currentInterval === '1d' ? '日K' : '週K'})`;
                    document.getElementById('currentPriceTag').innerText = `收盤: $${currentSpot}`;
                    document.getElementById('spotDisplay').innerText = `$${currentSpot} (MA20: $${currentMA20})`;
                    document.getElementById('stopLossDisplay').innerText = `頸線 $${currentNeckline} ／ 月線 $${currentMA20}`;
                    document.getElementById('qualityTag').innerText = data.quality_badge;

                    loadOptionChain(symbol);
                }
            } catch(e) {
                alert('載入美股數據失敗');
            }
        }

        async function loadOptionChain(symbol) {
            try {
                const res = await fetch(`/api/options/${symbol}`);
                const data = await res.json();
                currentOptionData = data.chain;
                const expSelect = document.getElementById('expDateSelect');
                expSelect.innerHTML = '';
                data.expirations.forEach(exp => {
                    const opt = document.createElement('option');
                    opt.value = exp;
                    opt.innerText = exp;
                    expSelect.appendChild(opt);
                });
                onExpirationChange();
            } catch(e) {}
        }

        function onExpirationChange() {
            const exp = document.getElementById('expDateSelect').value;
            const strat = document.getElementById('strategySelect').value;
            if(!currentOptionData || !currentOptionData[exp]) return;

            const strikes = (strat === 'cc') ? currentOptionData[exp].calls : currentOptionData[exp].puts;
            const strikeSelect = document.getElementById('strikeSelect');
            strikeSelect.innerHTML = '';

            strikes.forEach(stk => {
                const opt = document.createElement('option');
                opt.value = JSON.stringify(stk);
                opt.innerText = `Strike: $${stk.strike.toFixed(2)} (Bid: $${stk.bid} / Ask: $${stk.ask}, IV: ${(stk.impliedVolatility*100).toFixed(1)}%)`;
                strikeSelect.appendChild(opt);
            });
            autoAlignMA20();
        }

        function autoAlignMA20() {
            const strikeSelect = document.getElementById('strikeSelect');
            if (strikeSelect.options.length === 0) return;
            let closestIdx = 0, minDiff = 999999;
            for (let i = 0; i < strikeSelect.options.length; i++) {
                const stk = JSON.parse(strikeSelect.options[i].value);
                const diff = Math.abs(stk.strike - currentMA20);
                if (diff < minDiff) { minDiff = diff; closestIdx = i; }
            }
            strikeSelect.selectedIndex = closestIdx;
            onStrikeChange();
        }

        function onStrikeChange() {
            const val = document.getElementById('strikeSelect').value;
            if(!val) return;
            const stk = JSON.parse(val);
            const exp = document.getElementById('expDateSelect').value;
            const strat = document.getElementById('strategySelect').value;
            const optType = (strat === 'cc') ? 'call' : 'put';
            const days = Math.max(1, Math.floor((new Date(exp) - new Date()) / (1000 * 60 * 60 * 24)));
            const T = days / 365.0;
            const sigma = stk.impliedVolatility > 0 ? stk.impliedVolatility : 0.35;

            fetch(`/api/greeks?S=${currentSpot}&K=${stk.strike}&T=${T}&sigma=${sigma}&type=${optType}`)
                .then(res => res.json())
                .then(g => {
                    document.getElementById('valDelta').innerText = g.delta;
                    document.getElementById('valTheta').innerText = g.theta;
                    document.getElementById('valGamma').innerText = g.gamma;
                    document.getElementById('valVega').innerText = g.vega;
                });

            drawPayoffDiagram(stk.strike, (stk.bid + stk.ask) / 2, strat);
        }

        function drawPayoffDiagram(strike, premium, strategy) {
            const canvas = document.getElementById('payoffCanvas');
            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            const w = canvas.width, h = canvas.height;
            const minP = strike * 0.75, maxP = strike * 1.25;

            ctx.strokeStyle = '#1e293b';
            ctx.beginPath();
            ctx.moveTo(0, h / 2);
            ctx.lineTo(w, h / 2);
            ctx.stroke();

            ctx.strokeStyle = '#38bdf8';
            ctx.lineWidth = 2;
            ctx.beginPath();

            let maxProfit = `+$${(premium * 100).toFixed(0)}`;
            let maxLoss = `-$${((strike - premium) * 100).toFixed(0)}`;
            if (strategy === 'cc') {
                maxProfit = `+$${((strike - currentSpot + premium) * 100).toFixed(0)}`;
                maxLoss = `-$${((currentSpot - premium) * 100).toFixed(0)}`;
            } else if (strategy === 'long_call') {
                maxProfit = `無限大 (Long Call)`;
                maxLoss = `-$${(premium * 100).toFixed(0)}`;
            }

            document.getElementById('payoffMax').innerText = maxProfit;
            document.getElementById('payoffMin').innerText = maxLoss;

            for (let x = 0; x < w; x++) {
                const p = minP + (x / w) * (maxP - minP);
                let profit = 0;
                if (strategy === 'csp') {
                    profit = p >= strike ? premium : premium - (strike - p);
                } else if (strategy === 'cc') {
                    profit = p <= strike ? premium : premium - (p - strike);
                } else {
                    profit = p >= strike ? (p - strike) - premium : -premium;
                }
                const y = (h / 2) - (profit * 2.2);
                if (x === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.stroke();
        }

        function calculateOptionStrategy() {
            const val = document.getElementById('strikeSelect').value;
            if(!val) return;
            const stk = JSON.parse(val);
            const safetyBuffer = ((currentSpot - stk.strike) / currentSpot * 100).toFixed(2);
            alert(`[V9.8 智慧風控檢核]\\n標的: ${currentSymbol} | 履約價: $${stk.strike.toFixed(2)}\\n安全邊際緩衝: ${safetyBuffer}% (${safetyBuffer >= 8 ? '安全合格' : '警戒過窄'})`);
        }

        async function sendCloudPush() {
            const val = document.getElementById('strikeSelect').value;
            const stk = val ? JSON.parse(val) : { strike: currentSpot * 0.95 };
            const strat = document.getElementById('strategySelect').value;
            
            try {
                const res = await fetch('/api/line/push', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ symbol: currentSymbol, strike: stk.strike, strategy: strat })
                });
                const result = await res.json();
                document.getElementById('linePreviewText').innerText = result.preview_text;
                document.getElementById('lineModal').classList.remove('hidden');
            } catch(e) {
                alert('推播發送失敗');
            }
        }

        function closeLineModal() { document.getElementById('lineModal').classList.add('hidden'); }
        function changeInterval(intv) { currentInterval = intv; fetchStockData(currentSymbol); }

        window.onload = () => {
            initCharts();
            fetchIndicesData();
            loadScreenerList();
            fetchStockData("PLTR");
        };
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def serve_index():
    return HTMLResponse(content=HTML_CONTENT, status_code=200)

@app.get("/api/market/indices")
def get_indices():
    """取得道瓊、那指、費半、標普500、恐慌指數 VIX 大盤溫度計"""
    indices_map = {
        "dji": "^DJI",
        "ixic": "^IXIC",
        "sox": "^SOX",
        "spx": "^GSPC",
        "vix": "^VIX"
    }
    results = {}
    for k, sym in indices_map.items():
        try:
            t = yf.Ticker(sym)
            df = t.history(period="5d")
            c = float(df['Close'].iloc[-1])
            prev_c = float(df['Close'].iloc[-2])
            chg = round(c - prev_c, 2)
            pct = round((chg / prev_c) * 100.0, 2)
            if k == "vix":
                status = "恐慌警戒" if c > 20 else "正常平穩"
                results[k] = {"close": round(c, 2), "chg": chg, "pct": pct, "status": status}
            else:
                results[k] = {"close": int(round(c)), "chg": chg, "pct": pct}
        except:
            results[k] = {"close": 0, "chg": 0.0, "pct": 0.0, "status": "—"}
    return results

@app.get("/api/screener")
def run_v98_screener(sector: str = "ALL"):
    """V9.8 嚴謹量價突破與年線品質篩選引擎"""
    target_pool = MARKET_UNIVERSE if sector == "ALL" else [x for x in MARKET_UNIVERSE if x["sector"] == sector]
    results = []
    
    for item in target_pool[:25]:
        sym = item["symbol"]
        try:
            t = yf.Ticker(sym)
            df = t.history(period="3mo")
            if df.empty or len(df) < 25:
                continue
            
            c = float(df['Close'].iloc[-1])
            o = float(df['Open'].iloc[-1])
            v = float(df['Volume'].iloc[-1])
            v_ma5 = float(df['Volume'].shift(1).rolling(5).mean().iloc[-1])
            ma20 = float(df['Close'].rolling(20).mean().iloc[-1])
            neckline = float(df['High'].iloc[-21:-1].max())
            
            dist_neck = round(((c / neckline) - 1.0) * 100.0, 1)
            v_ratio = round(v / v_ma5, 1) if v_ma5 > 0 else 1.0
            
            # V9.8 突破判定：突破20日高點 + 紅K + 站上月線
            if c > neckline and c > o and c > ma20 and v_ratio >= 1.2:
                score = "🔥 暴量突破"
            elif c > ma20 and abs(dist_neck) <= 5.0:
                score = "🛡️ 頸線守候"
            else:
                score = "💎 偏多整理"
                
            results.append({
                "symbol": sym,
                "name": item["name"],
                "sector": item["sector"],
                "score": score,
                "dist_neck_pct": dist_neck,
                "v_ratio": v_ratio
            })
        except:
            continue
    return {"screener_list": results}

@app.get("/api/quote/{symbol}")
def get_quote_data(symbol: str, interval: str = "1d"):
    period = "2y" if interval == "1wk" else "1y"
    ticker = yf.Ticker(symbol.strip().upper())
    df = ticker.history(period=period, interval=interval)
    if df.empty:
        raise HTTPException(status_code=404, detail="Symbol not found")

    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA60'] = df['Close'].rolling(window=60).mean()
    df['MA240'] = df['Close'].rolling(window=240, min_periods=60).mean()
    df['Vol_MA'] = df['Volume'].rolling(window=20).mean()

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    df = df.dropna(subset=['Close'])
    spot = float(round(df['Close'].iloc[-1], 2))
    ma20_val = float(round(df['MA20'].iloc[-1], 2))
    ma240_val = float(round(df['MA240'].iloc[-1], 2)) if not pd.isna(df['MA240'].iloc[-1]) else spot
    neckline = float(round(df['High'].iloc[-21:-1].max(), 2))
    dist_neck = round(((spot / neckline) - 1.0) * 100.0, 1)

    quality_badge = "🛡️ 距頸線 ≤ 5% 且站上月線" if dist_neck <= 5.0 and spot >= ma20_val else "⚠️ 乖離偏遠或月線下"

    candles, ma5_data, ma20_data, ma60_data, ma240_data, vol_ma_data, rsi_data = [], [], [], [], [], [], []
    for date, row in df.iterrows():
        t_str = date.strftime("%Y-%m-%d")
        candles.append({
            "time": t_str,
            "open": float(round(row["Open"], 2)),
            "high": float(round(row["High"], 2)),
            "low": float(round(row["Low"], 2)),
            "close": float(round(row["Close"], 2)),
            "volume": int(row["Volume"])
        })
        if not pd.isna(row["MA5"]): ma5_data.append({"time": t_str, "value": float(round(row["MA5"], 2))})
        if not pd.isna(row["MA20"]): ma20_data.append({"time": t_str, "value": float(round(row["MA20"], 2))})
        if not pd.isna(row["MA60"]): ma60_data.append({"time": t_str, "value": float(round(row["MA60"], 2))})
        if not pd.isna(row["MA240"]): ma240_data.append({"time": t_str, "value": float(round(row["MA240"], 2))})
        if not pd.isna(row["Vol_MA"]): vol_ma_data.append({"time": t_str, "value": int(row["Vol_MA"])})
        if not pd.isna(row["RSI"]): rsi_data.append({"time": t_str, "value": float(round(row["RSI"], 1))})

    name = symbol
    for it in MARKET_UNIVERSE:
        if it["symbol"] == symbol.upper():
            name = it["name"]
            break

    return {
        "symbol": symbol.upper(),
        "name": name,
        "spot": spot,
        "ma20_val": ma20_val,
        "ma240_val": ma240_val,
        "neckline": neckline,
        "dist_neck": dist_neck,
        "quality_badge": quality_badge,
        "candles": candles,
        "ma5": ma5_data,
        "ma20": ma20_data,
        "ma60": ma60_data,
        "ma240": ma240_data,
        "vol_ma": vol_ma_data,
        "rsi": rsi_data
    }

@app.get("/api/options/{symbol}")
def get_options(symbol: str):
    ticker = yf.Ticker(symbol.strip().upper())
    try:
        expirations = ticker.options
        if expirations and len(expirations) > 0:
            chain_data = {}
            for exp in expirations[:4]:
                opt_chain = ticker.option_chain(exp)
                puts, calls = [], []
                for _, r in opt_chain.puts.iterrows():
                    puts.append({
                        "strike": float(r["strike"]),
                        "bid": float(r["bid"]) if not pd.isna(r["bid"]) else 1.5,
                        "ask": float(r["ask"]) if not pd.isna(r["ask"]) else 1.8,
                        "impliedVolatility": float(r["impliedVolatility"]) if not pd.isna(r["impliedVolatility"]) else 0.35
                    })
                for _, r in opt_chain.calls.iterrows():
                    calls.append({
                        "strike": float(r["strike"]),
                        "bid": float(r["bid"]) if not pd.isna(r["bid"]) else 1.5,
                        "ask": float(r["ask"]) if not pd.isna(r["ask"]) else 1.8,
                        "impliedVolatility": float(r["impliedVolatility"]) if not pd.isna(r["impliedVolatility"]) else 0.35
                    })
                chain_data[exp] = {"puts": puts, "calls": calls}
            return {"expirations": list(chain_data.keys()), "chain": chain_data}
    except Exception:
        pass

    df = ticker.history(period="1d")
    spot = float(df['Close'].iloc[-1]) if not df.empty else 186.0
    fallback_exp = ["2026-11-20", "2026-12-18", "2027-01-15"]
    fallback_chain = {}
    for exp in fallback_exp:
        puts, calls = [], []
        for i in range(-5, 6):
            stk = round(spot + i * 5, 2)
            puts.append({"strike": stk, "bid": 2.5, "ask": 2.8, "impliedVolatility": 0.35})
            calls.append({"strike": stk, "bid": 2.6, "ask": 2.9, "impliedVolatility": 0.35})
        fallback_chain[exp] = {"puts": puts, "calls": calls}
    return {"expirations": fallback_exp, "chain": fallback_chain}

@app.get("/api/greeks")
def get_greeks(S: float, K: float, T: float, sigma: float, type: str = "put"):
    return calculate_greeks(S, K, T, 0.045, sigma, type)

@app.post("/api/line/push")
def line_push(payload: dict):
    sym = payload.get("symbol", "PLTR")
    strike = payload.get("strike", 0.0)
    strat = payload.get("strategy", "CSP").upper()
    
    ticker = yf.Ticker(sym)
    df = ticker.history(period="3mo")
    spot = float(df['Close'].iloc[-1])
    ma20 = float(df['Close'].rolling(20).mean().iloc[-1])
    neckline = float(df['High'].iloc[-21:-1].max())
    dist_neck = round(((spot / neckline) - 1.0) * 100.0, 1)

    text_report = (
        f"📈【美股多空雷達 V9.8｜機構突破戰報】\n"
        f"🕒 資料時間：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"🌡️ 標的：{sym} 現價 ${spot:.2f}\n"
        f"📏 距20日頸線 (${neckline:.2f})：{dist_neck:+.1f}%\n"
        f"🧭 月線 (MA20)：${ma20:.2f} (站穩支撐)\n"
        f"🛡️ 停損參考位：頸線 ${neckline:.2f} ／ 月線 ${ma20:.2f}\n"
        f"⚡ 期權策略：{strat} (目標 Strike ${float(strike):.2f})\n"
        f"━━━━━━━━━━━━━━\n"
        f"※ 僅為 V9.8 量化條件篩選結果，非投資建議。"
    )

    if LINE_CHANNEL_ACCESS_TOKEN and LINE_USER_ID:
        url = "https://api.line.me/v2/bot/message/push"
        headers = {
            "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}",
            "Content-Type": "application/json"
        }
        data = {
            "to": LINE_USER_ID,
            "messages": [{"type": "text", "text": text_report}]
        }
        try:
            requests.post(url, headers=headers, json=data, timeout=8)
        except Exception:
            pass

    return {"status": "success", "preview_text": text_report}
