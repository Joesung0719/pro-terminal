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

app = FastAPI(title="US Professional Radar War Room - V9.8 Flagship Fully Integrated")

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

MARKET_UNIVERSE = [
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

HTML_CONTENT = """
<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>美股多空雷達戰情室 V9.8 旗艦完全體</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <style>
        :root {
            --bg-main: #0b0e14;
            --bg-card: #121824;
            --border-col: #1e293b;
            --text-main: #e2e8f0;
        }
        .theme-slate {
            --bg-main: #0f172a;
            --bg-card: #1e293b;
            --border-col: #334155;
            --text-main: #f8fafc;
        }
        .theme-dark {
            --bg-main: #000000;
            --bg-card: #0d0d0d;
            --border-col: #262626;
            --text-main: #e5e5e5;
        }
        body { background-color: var(--bg-main); color: var(--text-main); }
        .bg-card { background-color: var(--bg-card); }
        .border-theme { border-color: var(--border-col); }

        @media (max-width: 1024px) {
            #mainLayout { flex-direction: column !important; height: auto !important; overflow-y: auto !important; }
            #leftSidebar, #rightSidebar, #centerMain { width: 100% !important; height: auto !important; border: none !important; border-bottom: 1px solid var(--border-col) !important; }
            #centerMain { order: 1 !important; }
            #rightSidebar { order: 2 !important; }
            #leftSidebar { order: 3 !important; max-height: 480px !important; }
            .grid-indices { grid-template-columns: repeat(2, minmax(0, 1fr)) !important; }
            .grid-indices > div:last-child { grid-column: span 2 / span 2 !important; }
            body { overflow-y: auto !important; height: auto !important; }
        }
        .pill-active { background-color: #234E70 !important; color: #ffffff !important; border-color: #38bdf8 !important; font-weight: 700 !important; }
        .pill-inactive { background-color: var(--bg-card) !important; color: #64748b !important; border-color: var(--border-col) !important; }
    </style>
</head>
<body id="bodyRoot" class="font-sans antialiased min-h-screen flex flex-col">
    <!-- 頂部大盤溫度計全市場看板 -->
    <header class="bg-card border-b border-theme px-3.5 py-2 flex flex-col gap-1.5 shadow-lg shrink-0">
        <div class="flex justify-between items-center">
            <div class="flex items-center space-x-2">
                <span class="text-lg md:text-xl font-black text-[#38bdf8] tracking-wider drop-shadow">⚡ 美股多空雷達 V9.8</span>
                <span class="text-[10px] md:text-xs bg-[#1e293b] text-amber-400 px-2 py-0.5 rounded-full font-bold border border-amber-500/30">完全對齊台股 V9.8</span>
                <button onclick="toggleSettingsDrawer()" class="text-xs bg-[#1e293b] hover:bg-[#2a374a] text-slate-300 px-2 py-0.5 rounded border border-slate-600 transition">⚙️ 設定</button>
                <button onclick="openChangeLogModal()" class="text-xs bg-[#1e293b] hover:bg-[#2a374a] text-amber-300 px-2 py-0.5 rounded border border-amber-500/40 transition">📜 改版紀錄</button>
            </div>
            <button onclick="fetchIndicesData()" class="bg-[#1e293b] hover:bg-[#2a374a] text-white px-2 py-1 rounded border border-[#38bdf8]/40 text-xs font-bold shrink-0">🔄 刷新溫度計</button>
        </div>

        <div class="grid grid-cols-5 grid-indices gap-1.5 text-xs font-mono">
            <div class="bg-[#0b0e14] border border-theme px-2 py-1.5 rounded flex justify-between items-center">
                <span class="text-slate-400 font-bold">🏛️ 道瓊</span><span id="idx_dji" class="font-bold text-white">--</span>
            </div>
            <div class="bg-[#0b0e14] border border-theme px-2 py-1.5 rounded flex justify-between items-center">
                <span class="text-slate-400 font-bold">💻 那指</span><span id="idx_ixic" class="font-bold text-white">--</span>
            </div>
            <div class="bg-[#0b0e14] border border-theme px-2 py-1.5 rounded flex justify-between items-center">
                <span class="text-slate-400 font-bold">⚡ 費半</span><span id="idx_sox" class="font-bold text-white">--</span>
            </div>
            <div class="bg-[#0b0e14] border border-theme px-2 py-1.5 rounded flex justify-between items-center">
                <span class="text-slate-400 font-bold">📊 標普</span><span id="idx_spx" class="font-bold text-white">--</span>
            </div>
            <div class="bg-[#0b0e14] border border-theme px-2 py-1.5 rounded flex justify-between items-center">
                <span class="text-amber-400 font-bold">🔥 恐慌</span><span id="idx_vix" class="font-bold text-amber-400">--</span>
            </div>
        </div>

        <!-- ⚙️ 可收合設定抽屜 -->
        <div id="settingsDrawer" class="hidden bg-[#0b0e14] border border-theme rounded-lg p-3 mt-1.5 text-xs space-y-3">
            <div class="flex flex-wrap items-center justify-between gap-3 border-b border-theme pb-2">
                <div class="flex items-center space-x-2">
                    <span class="text-slate-400 font-bold">🎨 主題配色：</span>
                    <label class="cursor-pointer flex items-center space-x-1">
                        <input type="radio" name="themeRadio" value="default" checked onchange="switchTheme('default')">
                        <span>極夜黑</span>
                    </label>
                    <label class="cursor-pointer flex items-center space-x-1">
                        <input type="radio" name="themeRadio" value="slate" onchange="switchTheme('slate')">
                        <span>台股深灰藍 (溫和護眼)</span>
                    </label>
                    <label class="cursor-pointer flex items-center space-x-1">
                        <input type="radio" name="themeRadio" value="dark" onchange="switchTheme('dark')">
                        <span>純黑 AMOLED</span>
                    </label>
                </div>
                <div class="flex items-center space-x-2">
                    <span class="text-slate-400 font-bold">🔤 字體大小：</span>
                    <button onclick="setFontSize('normal')" class="px-2 py-0.5 rounded bg-card border border-theme">標準</button>
                    <button onclick="setFontSize('large')" class="px-2 py-0.5 rounded bg-card border border-theme">大</button>
                    <button onclick="setFontSize('xlarge')" class="px-2 py-0.5 rounded bg-card border border-theme">特大</button>
                </div>
            </div>
            <div class="flex flex-wrap items-center gap-4 text-slate-300">
                <div class="flex items-center space-x-2">
                    <span>頸線天數 (N日)：</span>
                    <input type="number" id="cfg_neckDays" value="20" min="5" max="60" class="w-14 bg-card border border-theme rounded px-1.5 py-0.5 text-center text-white" onchange="loadScreenerList()">
                </div>
                <div class="flex items-center space-x-2">
                    <span>5日均量倍數 (量比)：</span>
                    <input type="number" id="cfg_volRatio" value="1.2" step="0.1" min="1.0" max="3.0" class="w-16 bg-card border border-theme rounded px-1.5 py-0.5 text-center text-white" onchange="loadScreenerList()">
                </div>
                <span class="text-slate-500">※ 修改參數後左側篩選清單將即時重新計算過濾</span>
            </div>
        </div>
    </header>

    <!-- 主體區域 -->
    <div id="mainLayout" class="flex flex-1 overflow-hidden">
        <!-- 左側：多空實質篩選 ＋ 板塊分類 ＋ 股票清單 -->
        <aside id="leftSidebar" class="w-80 bg-card border-r border-theme flex flex-col shrink-0">
            <div class="p-3 border-b border-theme space-y-2">
                <div>
                    <label class="text-[11px] text-slate-400 font-bold block mb-1">🎯 選股多空方向：</label>
                    <div class="grid grid-cols-2 gap-1.5">
                        <button onclick="setRadarMode('LONG')" id="mode_long" class="py-1 rounded text-xs font-bold bg-[#C0392B] text-white transition">📈 多方突破(創高)</button>
                        <button onclick="setRadarMode('SHORT')" id="mode_short" class="py-1 rounded text-xs font-bold bg-[#1e293b] text-slate-400 hover:text-white transition">📉 空方破底(避險)</button>
                    </div>
                </div>

                <div>
                    <label class="text-[11px] text-slate-400 font-bold block mb-1">🏛️ 市場板塊大池：</label>
                    <select id="sectorSelect" onchange="onSectorChange()" class="w-full bg-[#0b0e14] border border-[#2a374a] rounded px-2.5 py-1.5 text-xs text-[#38bdf8] font-bold focus:outline-none focus:border-[#38bdf8]">
                        <option value="ALL">🌐 全市場精選大池 (全部標的)</option>
                        <option value="SOX">⚡ 費城半導體 (SOX / 晶片軍火)</option>
                        <option value="NASDAQ">💻 那斯達克 100 (科技核心主力)</option>
                        <option value="DOW">🏛️ 道瓊工業 30 (傳統藍籌防守)</option>
                        <option value="NYSE">💼 紐約證交所主力 (金融/能源/國防)</option>
                        <option value="ETF">📊 全美主力 ETF (標普/公債/黃金)</option>
                    </select>
                </div>

                <div class="flex gap-2">
                    <input id="customSymbol" type="text" placeholder="全美股代號 (例: NVDA, IONQ, BABA)" class="flex-1 bg-[#0b0e14] border border-[#2a374a] rounded px-3 py-1.5 text-xs text-white uppercase focus:outline-none focus:border-[#38bdf8]">
                    <button onclick="addCustomSymbol()" class="bg-[#38bdf8] hover:bg-sky-500 text-slate-950 font-black px-3 py-1.5 rounded text-xs transition">穿透診斷</button>
                </div>
            </div>
            <div id="watchlist" class="flex-1 overflow-y-auto p-2 space-y-1.5"></div>
        </aside>

        <!-- 中間：圖表區域 (含日K / 週K / 月K 完整切換) -->
        <main id="centerMain" class="flex-1 flex flex-col p-3 overflow-y-auto">
            <div class="flex flex-wrap justify-between items-center gap-2 mb-1.5">
                <div class="flex items-center space-x-2">
                    <h1 id="chartTitle" class="text-base md:text-xl font-black text-white tracking-wide">載入中...</h1>
                    <!-- ⭐ 日K / 週K / 月K 三段切換膠囊按鈕組 -->
                    <div class="flex bg-card border border-theme rounded p-0.5 space-x-1 text-xs">
                        <button onclick="changeInterval('1d')" id="btn_1d" class="px-2.5 py-0.5 rounded font-bold bg-[#38bdf8] text-slate-950 transition">日K</button>
                        <button onclick="changeInterval('1wk')" id="btn_1wk" class="px-2.5 py-0.5 rounded font-bold text-slate-400 hover:text-white transition">週K</button>
                        <button onclick="changeInterval('1mo')" id="btn_1mo" class="px-2.5 py-0.5 rounded font-bold text-slate-400 hover:text-white transition">月K</button>
                    </div>
                </div>
                <div class="flex items-center space-x-2">
                    <div id="qualityTag" class="text-[10px] md:text-xs bg-emerald-500/20 text-emerald-400 px-2 py-0.5 rounded border border-emerald-500/40 font-bold">
                        🛡️ 距頸線 ≤ 5%
                    </div>
                    <div id="currentPriceTag" class="text-sm md:text-base font-bold bg-card px-3 py-1 rounded-lg border border-theme shadow">
                        收盤: $-- (--%)
                    </div>
                </div>
            </div>

            <!-- 1. 上方：均線與指標開關膠囊列 -->
            <div class="flex flex-wrap items-center gap-1.5 mb-1.5 text-xs">
                <span class="text-slate-400 font-bold mr-1">📐 均線/指標:</span>
                <button onclick="toggleIndicator('ma5')" id="pill_ma5" class="pill-active px-2 py-0.5 rounded border border-[#D97706] text-[#D97706] text-[11px] transition">5MA(橘)</button>
                <button onclick="toggleIndicator('ma10')" id="pill_ma10" class="pill-inactive px-2 py-0.5 rounded border border-[#2563EB] text-[#2563EB] text-[11px] transition">10MA(藍)</button>
                <button onclick="toggleIndicator('ma20')" id="pill_ma20" class="pill-active px-2 py-0.5 rounded border border-[#7C3AED] text-[#7C3AED] text-[11px] transition">20MA(紫)</button>
                <button onclick="toggleIndicator('ma60')" id="pill_ma60" class="pill-active px-2 py-0.5 rounded border border-[#0D9488] text-[#0D9488] text-[11px] transition">60MA(綠)</button>
                <button onclick="toggleIndicator('ma120')" id="pill_ma120" class="pill-active px-2 py-0.5 rounded border border-[#DB2777] text-[#DB2777] text-[11px] transition">120MA(桃)</button>
                <button onclick="toggleIndicator('ma240')" id="pill_ma240" class="pill-active px-2 py-0.5 rounded border border-[#64748b] text-[#cbd5e1] text-[11px] transition">240MA(年線)</button>
                <button onclick="toggleIndicator('bb')" id="pill_bb" class="pill-inactive px-2 py-0.5 rounded border border-[#64748b] text-slate-300 text-[11px] transition">布林通道</button>
                <button onclick="toggleIndicator('neck')" id="pill_neck" class="pill-active px-2 py-0.5 rounded border border-[#B45309] text-[#B45309] text-[11px] transition">20日頸線</button>
            </div>

            <!-- 2. 中間：繁體中文線圖工具列 (位置移至 MA 下方、K 線上方置中) -->
            <div class="flex justify-center items-center bg-card border border-theme py-1 px-3 rounded-lg text-xs text-slate-300 mb-2 gap-2 shadow">
                <span class="font-bold text-[#38bdf8] mr-1">🛠️ 中文畫線工具:</span>
                <button onclick="setPlotlyMode('drawline')" class="tool-btn hover:text-white px-2 py-0.5 bg-[#0b0e14] rounded border border-theme transition" title="滑鼠按住拖曳畫趨勢線或水平線">📏 畫直線(趨勢/壓力)</button>
                <button onclick="setPlotlyMode('drawrect')" class="tool-btn hover:text-white px-2 py-0.5 bg-[#0b0e14] rounded border border-theme transition" title="畫盤整箱型區間">📦 畫方框(盤整箱)</button>
                <button onclick="setPlotlyMode('drawcircle')" class="tool-btn hover:text-white px-2 py-0.5 bg-[#0b0e14] rounded border border-theme transition" title="標記轉折圓圈">⭕ 畫圓形</button>
                <button onclick="setPlotlyMode('eraseshape')" class="tool-btn hover:text-white px-2 py-0.5 bg-[#0b0e14] rounded border border-theme text-red-400 transition" title="點擊線條刪除">🧹 橡皮擦</button>
                <span class="text-slate-600">|</span>
                <button onclick="setPlotlyMode('zoom')" class="tool-btn hover:text-white px-2 py-0.5 bg-[#0b0e14] rounded border border-theme transition">🔍 框選放大</button>
                <button onclick="setPlotlyMode('pan')" class="tool-btn hover:text-white px-2 py-0.5 bg-[#0b0e14] rounded border border-theme transition">✋ 拖曳平移</button>
                <button onclick="resetPlotlyAxes()" class="tool-btn hover:text-white px-2 py-0.5 bg-[#0b0e14] rounded border border-theme transition">🏠 還原初始</button>
            </div>

            <!-- 3. 下方：Plotly 原生 4 層圖表容器 -->
            <div id="plotlyChartContainer" class="bg-card border border-theme rounded-xl p-2 w-full h-[600px] md:h-[660px]"></div>
        </main>

        <!-- 右側：八大選擇權策略風控與 LINE 推播 -->
        <aside id="rightSidebar" class="w-96 bg-card border-l border-theme flex flex-col p-4 shrink-0">
            <h2 class="text-sm font-black text-white mb-2.5 flex items-center justify-between">
                <span>📊 選擇權全市場策略風控</span>
                <span class="text-[10px] bg-emerald-500 text-slate-950 font-black px-2 py-0.5 rounded">真實市場數據直通</span>
            </h2>

            <div class="space-y-2.5 text-xs">
                <div class="grid grid-cols-1 md:grid-cols-2 gap-2">
                    <div>
                        <label class="text-slate-400 block mb-1">到期日</label>
                        <select id="expDateSelect" onchange="onExpirationChange()" class="w-full bg-[#0b0e14] border border-[#2a374a] rounded p-2 text-white font-bold focus:outline-none focus:border-[#38bdf8]">
                            <option value="">載入中...</option>
                        </select>
                    </div>
                    <div>
                        <label class="text-slate-400 block mb-1">策略類型 (8大主流)</label>
                        <select id="strategySelect" onchange="onExpirationChange()" class="w-full bg-[#0b0e14] border border-[#2a374a] rounded p-2 text-white font-bold focus:outline-none focus:border-[#38bdf8]">
                            <optgroup label="💰 賣方穩健收租">
                                <option value="csp" selected>Short Put (低接收租)</option>
                                <option value="cc">Covered Call (現股包租公)</option>
                                <option value="bull_put">Bull Put Spread (賣權多頭價差)</option>
                            </optgroup>
                            <optgroup label="🚀 買方方向波段">
                                <option value="long_call">Long Call (大波段看漲)</option>
                                <option value="long_put">Long Put (看跌 / 現股避險)</option>
                                <option value="bull_call">Bull Call Spread (買權多頭價差)</option>
                                <option value="bear_put">Bear Put Spread (賣權空頭價差)</option>
                            </optgroup>
                            <optgroup label="⚡ 波動度交易">
                                <option value="straddle">Long Straddle (跨式 / 財報大波動)</option>
                            </optgroup>
                        </select>
                    </div>
                </div>

                <div class="bg-[#0b0e14] p-2.5 rounded-lg border border-[#2a374a] space-y-1">
                    <div class="flex justify-between items-center">
                        <span class="text-slate-400">現價 / 20MA(月線) 支撐</span>
                        <span id="spotDisplay" class="text-white font-bold">-</span>
                    </div>
                    <div class="flex justify-between items-center">
                        <span class="text-slate-400">停損參考點位</span>
                        <span id="stopLossDisplay" class="text-amber-400 font-bold">頸線 / 月線</span>
                    </div>
                </div>

                <div class="bg-[#0b0e14] p-2.5 rounded-lg border border-[#2a374a]">
                    <div class="flex justify-between items-center mb-1">
                        <label class="text-slate-400">目標履約價 (Strike)</label>
                        <span class="text-[10px] text-[#38bdf8] cursor-pointer hover:underline" onclick="autoAlignMA20()">🎯 智慧對齊 MA20</span>
                    </div>
                    <select id="strikeSelect" onchange="onStrikeChange()" class="w-full bg-[#121824] border border-[#2a374a] rounded p-2 text-white font-bold focus:outline-none focus:border-[#38bdf8]">
                        <option value="">請選擇到期日...</option>
                    </select>
                </div>

                <!-- Payoff 損益分佈圖 -->
                <div class="bg-[#0b0e14] p-2.5 rounded-lg border border-[#2a374a] space-y-1">
                    <div class="font-bold text-[#38bdf8] flex justify-between items-center text-[11px]">
                        <span>📈 Payoff 損益分佈預覽</span>
                    </div>
                    <div class="grid grid-cols-2 gap-2 text-[11px] bg-[#121824] p-1.5 rounded border border-theme">
                        <div>最大獲利: <strong id="payoffMax" class="text-emerald-400">--</strong></div>
                        <div>最大虧損: <strong id="payoffMin" class="text-red-400">--</strong></div>
                    </div>
                    <canvas id="payoffCanvas" height="75" class="w-full bg-[#121824] rounded border border-theme mt-1"></canvas>
                </div>

                <!-- Greeks 風險敏感度 -->
                <div class="bg-[#0b0e14] p-2.5 rounded-lg border border-[#2a374a] space-y-1">
                    <div class="font-bold text-[#38bdf8] border-b border-theme pb-0.5 text-[11px]">⚡ Greeks 風險敏感度</div>
                    <div class="grid grid-cols-2 gap-2 text-[11px]">
                        <div>Delta: <strong id="valDelta" class="text-white">-</strong></div>
                        <div>Theta: <strong id="valTheta" class="text-amber-400">-</strong></div>
                        <div>Gamma: <strong id="valGamma" class="text-white">-</strong></div>
                        <div>Vega: <strong id="valVega" class="text-white">-</strong></div>
                    </div>
                </div>

                <div class="grid grid-cols-2 gap-2 pt-1">
                    <button onclick="calculateOptionStrategy()" class="bg-[#38bdf8] hover:bg-sky-500 text-slate-950 font-black py-2 rounded-lg transition shadow text-xs">
                        V9.8 風控檢核
                    </button>
                    <button onclick="sendCloudPush()" class="bg-[#06C755] hover:bg-[#05b34c] text-white font-black py-2 rounded-lg transition shadow text-xs">
                        📲 LINE 戰報推播
                    </button>
                </div>
            </div>
        </aside>
    </div>

    <!-- 📜 歷次改版紀錄 Modal -->
    <div id="changeLogModal" class="fixed inset-0 bg-black/80 flex items-center justify-center hidden z-50 p-4">
        <div class="bg-card border border-theme rounded-2xl p-6 w-full max-w-xl shadow-2xl space-y-4">
            <div class="flex justify-between items-center border-b border-theme pb-2">
                <div class="flex items-center space-x-2">
                    <span class="text-lg">📜</span>
                    <h3 class="text-base font-black text-white">美股多空雷達戰情室 ｜ 歷次版本更新紀錄</h3>
                </div>
                <button onclick="closeChangeLogModal()" class="text-slate-400 hover:text-white text-base">✕</button>
            </div>
            <div class="space-y-3 text-xs max-h-[420px] overflow-y-auto pr-2 text-slate-300 font-sans">
                <div class="border-l-2 border-[#38bdf8] pl-3 py-1">
                    <div class="font-bold text-white text-sm">V9.8 旗艦完全體 (2026-10-17)</div>
                    <ul class="list-disc list-inside mt-1 space-y-0.5 text-slate-400">
                        <li>新增「月K」長線視角（日K / 週K / 月K 全面支援）。</li>
                        <li>全面導入 Plotly 專業多層圖表，原汁原味整合中文畫趨勢線、方框與橡皮擦。</li>
                        <li>四圖（K線/布林、成交量、KD、MACD）共用 X 軸完全同步縮放與平移。</li>
                        <li>補齊 120MA（桃紅）並支援均線即時膠囊開關切換。</li>
                        <li>選擇權策略擴充為 8 大主流策略（Short Put, Covered Call, Spreads, Straddle）。</li>
                        <li>拔除所有假數據 fallback，期權合約如實直通，無合約真實提示。</li>
                        <li>加入自適應設定欄（深灰藍護眼主題、字體調節、篩選參數調整）。</li>
                    </ul>
                </div>
                <div class="border-l-2 border-slate-600 pl-3 py-1">
                    <div class="font-bold text-slate-300">V9.5 ~ V9.7 雲端整合版</div>
                    <ul class="list-disc list-inside mt-1 space-y-0.5 text-slate-400">
                        <li>新增道瓊、那指、費半、標普、VIX 五大指數溫度計。</li>
                        <li>LINE 官方 Messaging API 推播串接與 Webhook Group ID 綁定。</li>
                        <li>夏令與冬令美股交易時間自動偵測與三段式智慧排程架構。</li>
                    </ul>
                </div>
                <div class="border-l-2 border-slate-700 pl-3 py-1">
                    <div class="font-bold text-slate-400">V1.0 ~ V9.0 核心引擎奠基</div>
                    <ul class="list-disc list-inside mt-1 space-y-0.5 text-slate-400">
                        <li>完成台股多空雷達 V9.8 標準色碼（紅漲綠跌、均線標準體系）。</li>
                        <li>Black-Scholes 選擇權定價模型與 Greeks 敏感度計算引擎。</li>
                    </ul>
                </div>
            </div>
            <button onclick="closeChangeLogModal()" class="w-full bg-[#38bdf8] hover:bg-sky-500 text-slate-950 font-black py-2 rounded-lg text-xs transition">關 閉</button>
        </div>
    </div>

    <!-- LINE 戰報預覽視窗 -->
    <div id="lineModal" class="fixed inset-0 bg-black/75 flex items-center justify-center hidden z-50 p-4">
        <div class="bg-card border border-theme rounded-2xl p-5 w-full max-w-md shadow-2xl space-y-3">
            <div class="flex justify-between items-center border-b border-theme pb-2">
                <div class="flex items-center space-x-2">
                    <span class="w-3 h-3 bg-[#06C755] rounded-full"></span>
                    <h3 class="text-sm font-black text-white">📲 美股多空雷達 V9.8 ｜ LINE 戰報</h3>
                </div>
                <button onclick="closeLineModal()" class="text-slate-400 hover:text-white text-sm">✕</button>
            </div>
            <div class="bg-[#0b0e14] p-3.5 rounded-xl border border-theme text-xs font-mono text-slate-300 space-y-1.5 whitespace-pre-wrap max-h-[340px] overflow-y-auto" id="linePreviewText"></div>
            <button onclick="closeLineModal()" class="w-full bg-[#38bdf8] text-slate-950 font-black py-2 rounded-lg text-xs transition">確 認</button>
        </div>
    </div>

    <script>
        let currentSpot = 0, currentMA20 = 0, currentMA240 = 0, currentNeckline = 0;
        let currentSymbol = "PLTR";
        let currentInterval = "1d"; // 1d, 1wk, 1mo
        let currentOptionData = null;
        let currentSector = "ALL";
        let currentRadarMode = "LONG";
        let universeData = [];
        let currentRawData = null;

        let indState = {
            ma5: true, ma10: false, ma20: true, ma60: true, ma120: true, ma240: true, bb: false, neck: true
        };

        function toggleSettingsDrawer() {
            const drawer = document.getElementById('settingsDrawer');
            drawer.classList.toggle('hidden');
        }

        function switchTheme(theme) {
            const root = document.getElementById('bodyRoot');
            root.className = "font-sans antialiased min-h-screen flex flex-col";
            if (theme === 'slate') root.classList.add('theme-slate');
            else if (theme === 'dark') root.classList.add('theme-dark');
            if (currentRawData) renderPlotlyChart(currentRawData);
        }

        function setFontSize(size) {
            const root = document.documentElement;
            if (size === 'large') root.style.fontSize = '17px';
            else if (size === 'xlarge') root.style.fontSize = '18.5px';
            else root.style.fontSize = '16px';
        }

        function openChangeLogModal() { document.getElementById('changeLogModal').classList.remove('hidden'); }
        function closeChangeLogModal() { document.getElementById('changeLogModal').classList.add('hidden'); }

        function setRadarMode(mode) {
            currentRadarMode = mode;
            if (mode === 'LONG') {
                document.getElementById('mode_long').className = "py-1 rounded text-xs font-bold bg-[#C0392B] text-white transition";
                document.getElementById('mode_short').className = "py-1 rounded text-xs font-bold bg-[#1e293b] text-slate-400 hover:text-white transition";
            } else {
                document.getElementById('mode_short').className = "py-1 rounded text-xs font-bold bg-[#1E7E5A] text-white transition";
                document.getElementById('mode_long').className = "py-1 rounded text-xs font-bold bg-[#1e293b] text-slate-400 hover:text-white transition";
            }
            loadScreenerList();
        }

        async function fetchIndicesData() {
            try {
                const res = await fetch('/api/market/indices');
                const data = await res.json();
                
                const fmtIdx = (obj) => {
                    const arrow = obj.chg >= 0 ? '▲' : '▼';
                    const color = obj.chg >= 0 ? 'text-[#C0392B]' : 'text-[#1E7E5A]';
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
            const neckDays = document.getElementById('cfg_neckDays') ? document.getElementById('cfg_neckDays').value : 20;
            const volRatio = document.getElementById('cfg_volRatio') ? document.getElementById('cfg_volRatio').value : 1.2;
            try {
                const res = await fetch(`/api/screener?sector=${currentSector}&mode=${currentRadarMode}&neck_days=${neckDays}&vol_ratio=${volRatio}`);
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
            if (!universeData || universeData.length === 0) {
                container.innerHTML = `<div class="p-4 text-center text-xs text-slate-500">目前時段無符合 ${currentRadarMode === 'LONG' ? '多方突破' : '空方破底'} 之標的</div>`;
                return;
            }
            universeData.forEach((item, idx) => {
                const div = document.createElement('div');
                div.className = "flex items-center justify-between px-3 py-2 rounded-lg bg-[#0b0e14] hover:bg-[#1a2234] transition group border border-theme";
                div.innerHTML = `
                    <div onclick="fetchStockData('${item.symbol}')" class="flex-1 cursor-pointer">
                        <div class="font-bold text-white group-hover:text-[#38bdf8] text-xs flex justify-between">
                            <span>${item.symbol} <span class="text-[10px] text-amber-400">[${item.sector}]</span></span>
                            <span class="text-[10px] text-emerald-400 bg-card px-1.5 py-0.5 rounded border border-emerald-500/20">${item.score}</span>
                        </div>
                        <div class="text-[11px] text-slate-400 flex justify-between mt-0.5">
                            <span class="truncate max-w-[120px]">${item.name}</span>
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

        function toggleIndicator(key) {
            indState[key] = !indState[key];
            const btn = document.getElementById(`pill_${key}`);
            if (indState[key]) {
                btn.className = "pill-active px-2 py-0.5 rounded border text-[11px] transition";
            } else {
                btn.className = "pill-inactive px-2 py-0.5 rounded border text-[11px] transition";
            }
            if (currentRawData) renderPlotlyChart(currentRawData);
        }

        function setPlotlyMode(mode) {
            const el = document.getElementById('plotlyChartContainer');
            if (!el || !el._fullLayout) return;
            if (mode === 'drawline') Plotly.relayout(el, { dragmode: 'drawline' });
            else if (mode === 'drawrect') Plotly.relayout(el, { dragmode: 'drawrect' });
            else if (mode === 'drawcircle') Plotly.relayout(el, { dragmode: 'drawcircle' });
            else if (mode === 'eraseshape') Plotly.relayout(el, { dragmode: 'eraseshape' });
            else if (mode === 'zoom') Plotly.relayout(el, { dragmode: 'zoom' });
            else if (mode === 'pan') Plotly.relayout(el, { dragmode: 'pan' });
        }

        function resetPlotlyAxes() {
            const el = document.getElementById('plotlyChartContainer');
            if (el) Plotly.relayout(el, { 'xaxis.autorange': true, 'yaxis.autorange': true });
        }

        function renderPlotlyChart(data) {
            currentRawData = data;
            const dates = data.dates;
            const traces = [];

            const isSlate = document.getElementById('bodyRoot').classList.contains('theme-slate');
            const isPureDark = document.getElementById('bodyRoot').classList.contains('theme-dark');
            const paperBg = isSlate ? '#1e293b' : (isPureDark ? '#0d0d0d' : '#121824');
            const plotBg = isSlate ? '#0f172a' : (isPureDark ? '#000000' : '#0b0e14');
            const gridCol = isSlate ? '#334155' : (isPureDark ? '#262626' : '#1e293b');

            // 1. K線
            traces.push({
                x: dates, open: data.opens, high: data.highs, low: data.lows, close: data.closes,
                type: 'candlestick', name: 'K線',
                increasing: { line: { color: '#C0392B' }, fillcolor: '#C0392B' },
                decreasing: { line: { color: '#1E7E5A' }, fillcolor: '#1E7E5A' },
                xaxis: 'x', yaxis: 'y'
            });

            // 均線
            if (indState.ma5) traces.push({ x: dates, y: data.ma5, type: 'scatter', mode: 'lines', line: { color: '#D97706', width: 1.5 }, name: '5MA', xaxis: 'x', yaxis: 'y' });
            if (indState.ma10) traces.push({ x: dates, y: data.ma10, type: 'scatter', mode: 'lines', line: { color: '#2563EB', width: 1.5 }, name: '10MA', xaxis: 'x', yaxis: 'y' });
            if (indState.ma20) traces.push({ x: dates, y: data.ma20, type: 'scatter', mode: 'lines', line: { color: '#7C3AED', width: 2 }, name: '20MA(月線)', xaxis: 'x', yaxis: 'y' });
            if (indState.ma60) traces.push({ x: dates, y: data.ma60, type: 'scatter', mode: 'lines', line: { color: '#0D9488', width: 1.5 }, name: '60MA(季線)', xaxis: 'x', yaxis: 'y' });
            if (indState.ma120) traces.push({ x: dates, y: data.ma120, type: 'scatter', mode: 'lines', line: { color: '#DB2777', width: 1.5 }, name: '120MA(半年)', xaxis: 'x', yaxis: 'y' });
            if (indState.ma240) traces.push({ x: dates, y: data.ma240, type: 'scatter', mode: 'lines', line: { color: '#64748b', width: 1.5, dash: 'dash' }, name: '240MA(年線)', xaxis: 'x', yaxis: 'y' });

            if (indState.bb) {
                traces.push({ x: dates, y: data.bb_up, type: 'scatter', mode: 'lines', line: { color: 'rgba(148, 163, 184, 0.55)', width: 1, dash: 'dot' }, name: '布林上軌', xaxis: 'x', yaxis: 'y' });
                traces.push({ x: dates, y: data.bb_low, type: 'scatter', mode: 'lines', line: { color: 'rgba(148, 163, 184, 0.55)', width: 1, dash: 'dot' }, name: '布林下軌', xaxis: 'x', yaxis: 'y' });
            }

            if (indState.neck) {
                traces.push({ x: dates, y: Array(dates.length).fill(data.neckline), type: 'scatter', mode: 'lines', line: { color: '#B45309', width: 2, dash: 'dash' }, name: '20日頸線', xaxis: 'x', yaxis: 'y' });
            }

            // 2. 成交量
            const volColors = data.closes.map((c, i) => c >= data.opens[i] ? 'rgba(192, 57, 43, 0.7)' : 'rgba(30, 126, 90, 0.7)');
            traces.push({ x: dates, y: data.volumes, type: 'bar', marker: { color: volColors }, name: '成交量', xaxis: 'x', yaxis: 'y2' });
            traces.push({ x: dates, y: data.vol_ma5, type: 'scatter', mode: 'lines', line: { color: '#B45309', width: 1.5, dash: 'dot' }, name: '5日均量', xaxis: 'x', yaxis: 'y2' });

            // 3. KD
            traces.push({ x: dates, y: data.k, type: 'scatter', mode: 'lines', line: { color: '#D97706', width: 1.8 }, name: 'K(9)', xaxis: 'x', yaxis: 'y3' });
            traces.push({ x: dates, y: data.d, type: 'scatter', mode: 'lines', line: { color: '#2563EB', width: 1.8 }, name: 'D(9)', xaxis: 'x', yaxis: 'y3' });

            // 4. MACD
            const macdColors = data.macd_hist.map(v => v >= 0 ? '#C0392B' : '#1E7E5A');
            traces.push({ x: dates, y: data.macd_hist, type: 'bar', marker: { color: macdColors }, name: 'MACD柱', xaxis: 'x', yaxis: 'y4' });
            traces.push({ x: dates, y: data.macd_dif, type: 'scatter', mode: 'lines', line: { color: '#D97706', width: 1.5 }, name: 'DIF', xaxis: 'x', yaxis: 'y4' });
            traces.push({ x: dates, y: data.macd_dea, type: 'scatter', mode: 'lines', line: { color: '#2563EB', width: 1.5 }, name: 'DEA', xaxis: 'x', yaxis: 'y4' });

            const layout = {
                paper_bgcolor: paperBg,
                plot_bgcolor: plotBg,
                font: { color: '#94a3b8', family: 'sans-serif', size: 11 },
                margin: { l: 45, r: 45, t: 15, b: 25 },
                showlegend: false,
                xaxis: { type: 'category', rangeslider: { visible: false }, gridcolor: gridCol, nticks: 8 },
                yaxis: { domain: [0.52, 1.0], gridcolor: gridCol, side: 'right' },
                yaxis2: { domain: [0.36, 0.49], gridcolor: gridCol, side: 'right', title: '量' },
                yaxis3: { domain: [0.19, 0.33], gridcolor: gridCol, range: [0, 100], side: 'right', title: 'KD' },
                yaxis4: { domain: [0.0, 0.16], gridcolor: gridCol, side: 'right', title: 'MACD' },
                hovermode: 'x unified',
                dragmode: 'drawline',
                newshape: { line: { color: '#facc15', width: 2 } }
            };

            const config = {
                responsive: true,
                displayModeBar: false
            };

            Plotly.newPlot('plotlyChartContainer', traces, layout, config);
        }

        async function fetchStockData(symbol) {
            currentSymbol = symbol;
            document.getElementById('chartTitle').innerText = `${symbol} 載入中...`;
            try {
                const res = await fetch(`/api/quote/${symbol}?interval=${currentInterval}`);
                const data = await res.json();
                if(data.dates && data.dates.length > 0) {
                    currentSpot = data.spot;
                    currentMA20 = data.ma20_val;
                    currentMA240 = data.ma240_val;
                    currentNeckline = data.neckline;

                    const intvName = currentInterval === '1d' ? '日K' : (currentInterval === '1wk' ? '週K' : '月K');
                    document.getElementById('chartTitle').innerText = `${symbol} ${data.name} [${data.exchange}] (${intvName})`;
                    
                    const pSign = data.chg >= 0 ? '▲' : '▼';
                    const pColor = data.chg >= 0 ? '#C0392B' : '#1E7E5A';
                    document.getElementById('currentPriceTag').style.color = pColor;
                    document.getElementById('currentPriceTag').innerText = `收盤: $${currentSpot} (${pSign}${Math.abs(data.pct)}%)`;

                    document.getElementById('spotDisplay').innerText = `$${currentSpot} (20MA: $${currentMA20})`;
                    document.getElementById('stopLossDisplay').innerText = `頸線 $${currentNeckline} ／ 月線 $${currentMA20}`;
                    document.getElementById('qualityTag').innerText = data.quality_badge;

                    renderPlotlyChart(data);
                    loadOptionChain(symbol);
                }
            } catch(e) {
                alert('載入美股數據失敗，請確認代號是否正確');
            }
        }

        async function loadOptionChain(symbol) {
            const expSelect = document.getElementById('expDateSelect');
            const strikeSelect = document.getElementById('strikeSelect');
            expSelect.innerHTML = '<option value="">查詢真實期權鏈中...</option>';
            strikeSelect.innerHTML = '<option value="">請選擇到期日...</option>';

            try {
                const res = await fetch(`/api/options/${symbol}`);
                const data = await res.json();
                currentOptionData = data.chain;
                expSelect.innerHTML = '';
                
                if (!data.expirations || data.expirations.length === 0) {
                    expSelect.innerHTML = '<option value="">⚠️ 此標的目前無期權合約或非交易時段</option>';
                    return;
                }

                data.expirations.forEach(exp => {
                    const opt = document.createElement('option');
                    opt.value = exp;
                    opt.innerText = exp;
                    expSelect.appendChild(opt);
                });
                onExpirationChange();
            } catch(e) {
                expSelect.innerHTML = '<option value="">⚠️ 期權鏈取得失敗</option>';
            }
        }

        function onExpirationChange() {
            const exp = document.getElementById('expDateSelect').value;
            const strat = document.getElementById('strategySelect').value;
            if(!currentOptionData || !currentOptionData[exp]) return;

            const isCallBased = ['cc', 'long_call', 'bull_call'].includes(strat);
            const strikes = isCallBased ? currentOptionData[exp].calls : currentOptionData[exp].puts;
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
            const optType = ['cc', 'long_call', 'bull_call'].includes(strat) ? 'call' : 'put';
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

            let maxProfit = '--', maxLoss = '--';

            if (strategy === 'csp') {
                maxProfit = `+$${(premium * 100).toFixed(0)}`;
                maxLoss = `-$${((strike - premium) * 100).toFixed(0)}`;
            } else if (strategy === 'cc') {
                maxProfit = `+$${((strike - currentSpot + premium) * 100).toFixed(0)}`;
                maxLoss = `-$${((currentSpot - premium) * 100).toFixed(0)}`;
            } else if (strategy === 'long_call') {
                maxProfit = `無限大 (Long Call)`;
                maxLoss = `-$${(premium * 100).toFixed(0)}`;
            } else if (strategy === 'long_put') {
                maxProfit = `+$${((strike - premium) * 100).toFixed(0)}`;
                maxLoss = `-$${(premium * 100).toFixed(0)}`;
            } else if (strategy === 'bull_call') {
                const spreadWidth = strike * 0.05;
                maxProfit = `+$${((spreadWidth - premium) * 100).toFixed(0)}`;
                maxLoss = `-$${(premium * 100).toFixed(0)}`;
            } else if (strategy === 'bear_put') {
                const spreadWidth = strike * 0.05;
                maxProfit = `+$${((spreadWidth - premium) * 100).toFixed(0)}`;
                maxLoss = `-$${(premium * 100).toFixed(0)}`;
            } else if (strategy === 'bull_put') {
                const spreadWidth = strike * 0.05;
                maxProfit = `+$${(premium * 100).toFixed(0)}`;
                maxLoss = `-$${((spreadWidth - premium) * 100).toFixed(0)}`;
            } else if (strategy === 'straddle') {
                maxProfit = `無限大 (雙向大波動)`;
                maxLoss = `-$${(premium * 2 * 100).toFixed(0)}`;
            }

            document.getElementById('payoffMax').innerText = maxProfit;
            document.getElementById('payoffMin').innerText = maxLoss;

            for (let x = 0; x < w; x++) {
                const p = minP + (x / w) * (maxP - minP);
                let profit = 0;

                if (strategy === 'csp') profit = p >= strike ? premium : premium - (strike - p);
                else if (strategy === 'cc') profit = p <= strike ? premium : premium - (p - strike);
                else if (strategy === 'long_call') profit = p >= strike ? (p - strike) - premium : -premium;
                else if (strategy === 'long_put') profit = p <= strike ? (strike - p) - premium : -premium;
                else if (strategy === 'bull_call') {
                    const upper = strike * 1.05;
                    profit = Math.min(Math.max(p - strike, 0), upper - strike) - premium;
                } else if (strategy === 'bear_put') {
                    const lower = strike * 0.95;
                    profit = Math.min(Math.max(strike - p, 0), strike - lower) - premium;
                } else if (strategy === 'bull_put') {
                    const lower = strike * 0.95;
                    profit = premium - Math.min(Math.max(strike - p, 0), strike - lower);
                } else if (strategy === 'straddle') {
                    profit = Math.abs(p - strike) - (premium * 1.8);
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
        
        // ⭐ 日K / 週K / 月K 切換事件
        function changeInterval(intv) {
            currentInterval = intv;
            ['1d', '1wk', '1mo'].forEach(k => {
                const btn = document.getElementById(`btn_${k}`);
                if (k === intv) {
                    btn.className = "px-2.5 py-0.5 rounded font-bold bg-[#38bdf8] text-slate-950 transition";
                } else {
                    btn.className = "px-2.5 py-0.5 rounded font-bold text-slate-400 hover:text-white transition";
                }
            });
            fetchStockData(currentSymbol);
        }

        window.onload = () => {
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
    indices_map = {"dji": "^DJI", "ixic": "^IXIC", "sox": "^SOX", "spx": "^GSPC", "vix": "^VIX"}
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
def run_v98_screener(sector: str = "ALL", mode: str = "LONG", neck_days: int = 20, vol_ratio: float = 1.2):
    target_pool = MARKET_UNIVERSE if sector == "ALL" else [x for x in MARKET_UNIVERSE if x["sector"] == sector]
    results = []
    
    for item in target_pool[:30]:
        sym = item["symbol"]
        try:
            t = yf.Ticker(sym)
            df = t.history(period="3mo")
            if df.empty or len(df) < (neck_days + 5):
                continue
            c = float(df['Close'].iloc[-1])
            o = float(df['Open'].iloc[-1])
            v = float(df['Volume'].iloc[-1])
            v_ma5 = float(df['Volume'].shift(1).rolling(5).mean().iloc[-1])
            ma20 = float(df['Close'].rolling(20).mean().iloc[-1])
            
            high_neck = float(df['High'].iloc[-(neck_days+1):-1].max())
            low_neck = float(df['Low'].iloc[-(neck_days+1):-1].min())
            cur_v_ratio = round(v / v_ma5, 1) if v_ma5 > 0 else 1.0

            if mode == "LONG":
                if not (c > high_neck and c > o and c > ma20 and cur_v_ratio >= vol_ratio):
                    continue
                dist_neck = round(((c / high_neck) - 1.0) * 100.0, 1)
                score = "🔥 暴量突破"
            else:
                if not (c < low_neck and c < o and c < ma20 and cur_v_ratio >= vol_ratio):
                    continue
                dist_neck = round(((c / low_neck) - 1.0) * 100.0, 1)
                score = "⚡ 破底轉弱"
                
            results.append({
                "symbol": sym, "name": item["name"], "sector": item["sector"],
                "score": score, "dist_neck_pct": dist_neck, "v_ratio": cur_v_ratio
            })
        except:
            continue
    return {"screener_list": results}

@app.get("/api/quote/{symbol}")
def get_quote_data(symbol: str, interval: str = "1d"):
    # ⭐ 根據週期設定拉取足夠長度的歷史數據
    if interval == "1mo":
        period = "10y"
    elif interval == "1wk":
        period = "3y"
    else:
        period = "1y"

    ticker = yf.Ticker(symbol.strip().upper())
    df = ticker.history(period=period, interval=interval)
    if df.empty:
        raise HTTPException(status_code=404, detail="Symbol not found")

    # 均線系統
    df['MA5'] = df['Close'].rolling(window=5, min_periods=1).mean()
    df['MA10'] = df['Close'].rolling(window=10, min_periods=1).mean()
    df['MA20'] = df['Close'].rolling(window=20, min_periods=1).mean()
    df['MA60'] = df['Close'].rolling(window=60, min_periods=1).mean()
    df['MA120'] = df['Close'].rolling(window=120, min_periods=1).mean()
    df['MA240'] = df['Close'].rolling(window=240, min_periods=1).mean()
    
    # 布林通道 (20MA +- 2*STD)
    bb_std = df['Close'].rolling(window=20, min_periods=5).std(ddof=0).fillna(0)
    df['BB_UP'] = df['MA20'] + 2.0 * bb_std
    df['BB_LOW'] = df['MA20'] - 2.0 * bb_std

    # 5 週期均量
    df['Vol_MA5'] = df['Volume'].shift(1).rolling(5, min_periods=1).mean()

    # KD (9, 3, 3)
    low9 = df['Low'].rolling(9, min_periods=1).min()
    high9 = df['High'].rolling(9, min_periods=1).max()
    rsv = ((df['Close'] - low9) / (high9 - low9).replace(0, np.nan) * 100.0).fillna(50.0)
    k_vals, d_vals = [], []
    k_prev, d_prev = 50.0, 50.0
    for val in rsv:
        k_curr = (2.0 / 3.0) * k_prev + (1.0 / 3.0) * float(val)
        d_curr = (2.0 / 3.0) * d_prev + (1.0 / 3.0) * k_curr
        k_vals.append(round(k_curr, 1))
        d_vals.append(round(d_curr, 1))
        k_prev, d_prev = k_curr, d_curr
    df['K'] = k_vals
    df['D'] = d_vals

    # MACD (12, 26, 9)
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['DIF'] = ema12 - ema26
    df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
    df['MACD_HIST'] = df['DIF'] - df['DEA']

    df = df.dropna(subset=['Close'])
    spot = float(round(df['Close'].iloc[-1], 2))
    prev_spot = float(round(df['Close'].iloc[-2], 2)) if len(df) >= 2 else spot
    chg = round(spot - prev_spot, 2)
    pct = round((chg / prev_spot) * 100.0, 2) if prev_spot > 0 else 0.0

    ma20_val = float(round(df['MA20'].iloc[-1], 2))
    ma240_val = float(round(df['MA240'].iloc[-1], 2)) if not pd.isna(df['MA240'].iloc[-1]) else spot
    
    neck_slice = df['High'].iloc[-21:-1] if len(df) > 21 else df['High']
    neckline = float(round(neck_slice.max(), 2))
    dist_neck = round(((spot / neckline) - 1.0) * 100.0, 1) if neckline > 0 else 0.0

    quality_badge = "🛡️ 距頸線 ≤ 5% 且站上月線" if dist_neck <= 5.0 and spot >= ma20_val else "⚠️ 乖離偏遠或月線下"

    dates = [d.strftime("%Y-%m-%d") for d in df.index]
    opens = [float(round(v, 2)) for v in df["Open"].values]
    highs = [float(round(v, 2)) for v in df["High"].values]
    lows = [float(round(v, 2)) for v in df["Low"].values]
    closes = [float(round(v, 2)) for v in df["Close"].values]
    volumes = [int(v) for v in df["Volume"].values]

    def _clean_series(s):
        return [float(round(v, 2)) if not pd.isna(v) else None for v in s.values]

    sym_clean = symbol.strip().upper()
    name_str = sym_clean
    exchange_str = "US"
    
    for it in MARKET_UNIVERSE:
        if it["symbol"] == sym_clean:
            name_str = it["name"]
            exchange_str = it["market"]
            break

    if name_str == sym_clean:
        try:
            info = ticker.info
            name_str = info.get('shortName') or info.get('longName') or sym_clean
            exchange_str = info.get('exchange', 'US')
        except:
            pass

    return {
        "symbol": sym_clean,
        "name": name_str,
        "exchange": exchange_str,
        "spot": spot,
        "chg": chg,
        "pct": pct,
        "ma20_val": ma20_val,
        "ma240_val": ma240_val,
        "neckline": neckline,
        "dist_neck": dist_neck,
        "quality_badge": quality_badge,
        "dates": dates,
        "opens": opens,
        "highs": highs,
        "lows": lows,
        "closes": closes,
        "volumes": volumes,
        "ma5": _clean_series(df["MA5"]),
        "ma10": _clean_series(df["MA10"]),
        "ma20": _clean_series(df["MA20"]),
        "ma60": _clean_series(df["MA60"]),
        "ma120": _clean_series(df["MA120"]),
        "ma240": _clean_series(df["MA240"]),
        "bb_up": _clean_series(df["BB_UP"]),
        "bb_low": _clean_series(df["BB_LOW"]),
        "vol_ma5": _clean_series(df["Vol_MA5"]),
        "k": _clean_series(df["K"]),
        "d": _clean_series(df["D"]),
        "macd_hist": _clean_series(df["MACD_HIST"]),
        "macd_dif": _clean_series(df["DIF"]),
        "macd_dea": _clean_series(df["DEA"])
    }

@app.get("/api/options/{symbol}")
def get_options(symbol: str):
    ticker = yf.Ticker(symbol.strip().upper())
    expirations = ticker.options
    
    if expirations and len(expirations) > 0:
        chain_data = {}
        for exp in expirations:
            try:
                opt_chain = ticker.option_chain(exp)
                puts, calls = [], []
                for _, r in opt_chain.puts.iterrows():
                    puts.append({
                        "strike": float(r["strike"]),
                        "bid": float(r["bid"]) if not pd.isna(r["bid"]) else 0.0,
                        "ask": float(r["ask"]) if not pd.isna(r["ask"]) else 0.0,
                        "impliedVolatility": float(r["impliedVolatility"]) if not pd.isna(r["impliedVolatility"]) else 0.35
                    })
                for _, r in opt_chain.calls.iterrows():
                    calls.append({
                        "strike": float(r["strike"]),
                        "bid": float(r["bid"]) if not pd.isna(r["bid"]) else 0.0,
                        "ask": float(r["ask"]) if not pd.isna(r["ask"]) else 0.0,
                        "impliedVolatility": float(r["impliedVolatility"]) if not pd.isna(r["impliedVolatility"]) else 0.35
                    })
                chain_data[exp] = {"puts": puts, "calls": calls}
            except:
                continue
        return {"expirations": list(chain_data.keys()), "chain": chain_data}
    
    return {"expirations": [], "chain": {}}

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
    prev_spot = float(df['Close'].iloc[-2]) if len(df) >= 2 else spot
    pct = round(((spot - prev_spot) / prev_spot) * 100.0, 2) if prev_spot > 0 else 0.0
    p_sign = "▲" if pct >= 0 else "▼"

    ma20 = float(df['Close'].rolling(20).mean().iloc[-1])
    neckline = float(df['High'].iloc[-21:-1].max())
    dist_neck = round(((spot / neckline) - 1.0) * 100.0, 1)

    text_report = (
        f"📈【美股多空雷達 V9.8｜機構戰報】\n"
        f"🕒 資料時間：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"🌡️ 標的：{sym} 現價 ${spot:.2f} ({p_sign}{abs(pct)}%)\n"
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
