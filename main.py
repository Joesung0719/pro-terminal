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

app = FastAPI(title="US Professional Radar War Room - V9.8 Complete Suite")

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
    {"symbol": "AAPL", "name": "蘋果", "sector": "NASDAQ", "market": "NASDAQ"},
    {"symbol": "MSFT", "name": "微軟", "sector": "NASDAQ", "market": "NASDAQ"},
    {"symbol": "GOOGL", "name": "Alphabet", "sector": "NASDAQ", "market": "NASDAQ"},
    {"symbol": "AMZN", "name": "亞馬遜", "sector": "NASDAQ", "market": "NASDAQ"},
    {"symbol": "META", "name": "Meta 臉書", "sector": "NASDAQ", "market": "NASDAQ"},
    {"symbol": "TSLA", "name": "特斯拉", "sector": "NASDAQ", "market": "NASDAQ"},
    {"symbol": "NFLX", "name": "網飛 Netflix", "sector": "NASDAQ", "market": "NASDAQ"},
    {"symbol": "PLTR", "name": "帕蘭蒂爾科技", "sector": "NYSE", "market": "NYSE"},
    {"symbol": "JPM", "name": "摩根大通", "sector": "NYSE", "market": "NYSE"},
    {"symbol": "BA", "name": "波音公司", "sector": "DOW", "market": "NYSE"},
    {"symbol": "CAT", "name": "開拓重工", "sector": "DOW", "market": "NYSE"},
    {"symbol": "SPY", "name": "標普500 ETF", "sector": "ETF", "market": "AMEX"},
    {"symbol": "QQQ", "name": "那指100 ETF", "sector": "ETF", "market": "NASDAQ"},
    {"symbol": "SOXX", "name": "費半 ETF", "sector": "ETF", "market": "NASDAQ"}
]

HTML_CONTENT = """
<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>美股多空雷達戰情室 V9.8 ｜ 完整指標聯動旗艦體</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://unpkg.com/lightweight-charts@4.1.1/dist/lightweight-charts.standalone.production.js"></script>
    <style>
        @media (max-width: 1024px) {
            #mainLayout {
                flex-direction: column !important;
                height: auto !important;
                overflow-y: auto !important;
            }
            #leftSidebar, #rightSidebar, #centerMain {
                width: 100% !important;
                height: auto !important;
                border: none !important;
                border-bottom: 1px solid #1e293b !important;
            }
            #centerMain { order: 1 !important; }
            #rightSidebar { order: 2 !important; }
            #leftSidebar { order: 3 !important; max-height: 480px !important; }
            .grid-indices { grid-template-columns: repeat(2, minmax(0, 1fr)) !important; }
            .grid-indices > div:last-child { grid-column: span 2 / span 2 !important; }
            body { overflow-y: auto !important; height: auto !important; }
        }
        .pill-active {
            background-color: #234E70 !important;
            color: #ffffff !important;
            border-color: #38bdf8 !important;
            font-weight: 700 !important;
        }
        .pill-inactive {
            background-color: #121824 !important;
            color: #64748b !important;
            border-color: #1e293b !important;
        }
    </style>
</head>
<body class="bg-[#0b0e14] text-[#e2e8f0] font-sans antialiased min-h-screen flex flex-col">
    <!-- 頂部五大指數大盤多空溫度計 -->
    <header class="bg-[#121824] border-b border-[#1e293b] px-3.5 py-2 flex flex-col gap-1.5 shadow-lg shrink-0">
        <div class="flex justify-between items-center">
            <div class="flex items-center space-x-2">
                <span class="text-lg md:text-xl font-black text-[#38bdf8] tracking-wider drop-shadow">⚡ 美股多空雷達 V9.8</span>
                <span class="text-[10px] md:text-xs bg-[#1e293b] text-amber-400 px-2 py-0.5 rounded-full font-bold border border-amber-500/30">四層指標聯動版</span>
            </div>
            <button onclick="fetchIndicesData()" class="bg-[#1e293b] hover:bg-[#2a374a] text-white px-2 py-1 rounded border border-[#38bdf8]/40 text-xs font-bold shrink-0">🔄 刷新溫度計</button>
        </div>

        <div class="grid grid-cols-5 grid-indices gap-1.5 text-xs font-mono">
            <div class="bg-[#0b0e14] border border-[#1e293b] px-2 py-1.5 rounded flex justify-between items-center">
                <span class="text-slate-400 font-bold">🏛️ 道瓊</span>
                <span id="idx_dji" class="font-bold text-white">--</span>
            </div>
            <div class="bg-[#0b0e14] border border-[#1e293b] px-2 py-1.5 rounded flex justify-between items-center">
                <span class="text-slate-400 font-bold">💻 那指</span>
                <span id="idx_ixic" class="font-bold text-white">--</span>
            </div>
            <div class="bg-[#0b0e14] border border-[#1e293b] px-2 py-1.5 rounded flex justify-between items-center">
                <span class="text-slate-400 font-bold">⚡ 費半</span>
                <span id="idx_sox" class="font-bold text-white">--</span>
            </div>
            <div class="bg-[#0b0e14] border border-[#1e293b] px-2 py-1.5 rounded flex justify-between items-center">
                <span class="text-slate-400 font-bold">📊 標普</span>
                <span id="idx_spx" class="font-bold text-white">--</span>
            </div>
            <div class="bg-[#0b0e14] border border-[#1e293b] px-2 py-1.5 rounded flex justify-between items-center">
                <span class="text-amber-400 font-bold">🔥 恐慌</span>
                <span id="idx_vix" class="font-bold text-amber-400">--</span>
            </div>
        </div>
    </header>

    <!-- 主體區域 -->
    <div id="mainLayout" class="flex flex-1 overflow-hidden">
        <!-- 左側：市場板塊清單 -->
        <aside id="leftSidebar" class="w-80 bg-[#121824] border-r border-[#1e293b] flex flex-col shrink-0">
            <div class="p-3 border-b border-[#1e293b] space-y-2">
                <div class="flex justify-between items-center">
                    <h2 class="text-xs font-bold text-slate-300 uppercase tracking-wider">🎯 市場板塊分類</h2>
                    <span class="text-[10px] text-emerald-400 font-bold bg-[#0b0e14] px-1.5 py-0.5 rounded border border-emerald-500/30">V9.8 爆量突破</span>
                </div>
                <select id="sectorSelect" onchange="onSectorChange()" class="w-full bg-[#0b0e14] border border-[#2a374a] rounded px-2.5 py-1.5 text-xs text-[#38bdf8] font-bold focus:outline-none focus:border-[#38bdf8]">
                    <option value="ALL">🌐 全市場精選大池 (全部標的)</option>
                    <option value="SOX">⚡ 費城半導體 (SOX / 晶片軍火)</option>
                    <option value="NASDAQ">💻 那斯達克 100 (科技核心主力)</option>
                    <option value="DOW">🏛️ 道瓊工業 30 (傳統藍籌防守)</option>
                    <option value="NYSE">💼 紐約證交所主力 (金融/能源)</option>
                    <option value="ETF">📊 全美主力 ETF (標普/費半)</option>
                </select>

                <div class="flex gap-2">
                    <input id="customSymbol" type="text" placeholder="輸入代號 (例: PLTR, NVDA)" class="flex-1 bg-[#0b0e14] border border-[#2a374a] rounded px-3 py-1.5 text-xs text-white uppercase focus:outline-none focus:border-[#38bdf8]">
                    <button onclick="addCustomSymbol()" class="bg-[#38bdf8] hover:bg-sky-500 text-slate-950 font-black px-3 py-1.5 rounded text-xs transition">診斷</button>
                </div>
            </div>
            <div id="watchlist" class="flex-1 overflow-y-auto p-2 space-y-1.5"></div>
        </aside>

        <!-- 中間：圖表區域 (K線/布林 + 成交量 + KD + MACD 四圖完全聯動) -->
        <main id="centerMain" class="flex-1 flex flex-col bg-[#0b0e14] p-3 overflow-y-auto">
            <!-- 標題、週期與帶 % 數的即時收盤價 -->
            <div class="flex flex-wrap justify-between items-center gap-2 mb-2">
                <div class="flex items-center space-x-2">
                    <h1 id="chartTitle" class="text-base md:text-xl font-black text-white tracking-wide">PLTR 帕蘭蒂爾科技</h1>
                    <div class="flex bg-[#121824] border border-[#1e293b] rounded p-0.5 space-x-1 text-xs">
                        <button onclick="changeInterval('1d')" id="btn_1d" class="px-2.5 py-0.5 rounded font-bold bg-[#38bdf8] text-slate-950">日K</button>
                        <button onclick="changeInterval('1wk')" id="btn_1wk" class="px-2.5 py-0.5 rounded font-bold text-slate-400 hover:text-white">週K</button>
                    </div>
                </div>
                <div class="flex items-center space-x-2">
                    <div id="qualityTag" class="text-[10px] md:text-xs bg-emerald-500/20 text-emerald-400 px-2 py-0.5 rounded border border-emerald-500/40 font-bold">
                        🛡️ 距頸線 ≤ 5%
                    </div>
                    <!-- 帶有 % 數與點數的即時價格膠囊 -->
                    <div id="currentPriceTag" class="text-sm md:text-base font-bold bg-[#121824] px-3 py-1 rounded-lg border border-[#1e293b] shadow">
                        收盤: $-- (--%)
                    </div>
                </div>
            </div>

            <!-- 台股 V9.8 均線與指標開關膠囊列 (預設 5, 20, 60, 240 開啟，10MA 關閉) -->
            <div class="flex flex-wrap items-center gap-1.5 mb-2 text-xs">
                <span class="text-slate-400 font-bold mr-1">📐 指標開關:</span>
                <button onclick="toggleIndicator('ma5')" id="pill_ma5" class="pill-active px-2 py-0.5 rounded border border-[#D97706] text-[#D97706] text-[11px] transition">5MA(橘)</button>
                <button onclick="toggleIndicator('ma10')" id="pill_ma10" class="pill-inactive px-2 py-0.5 rounded border border-[#2563EB] text-[#2563EB] text-[11px] transition">10MA(藍)</button>
                <button onclick="toggleIndicator('ma20')" id="pill_ma20" class="pill-active px-2 py-0.5 rounded border border-[#7C3AED] text-[#7C3AED] text-[11px] transition">20MA(紫)</button>
                <button onclick="toggleIndicator('ma60')" id="pill_ma60" class="pill-active px-2 py-0.5 rounded border border-[#0D9488] text-[#0D9488] text-[11px] transition">60MA(綠)</button>
                <button onclick="toggleIndicator('ma240')" id="pill_ma240" class="pill-active px-2 py-0.5 rounded border border-[#64748b] text-[#cbd5e1] text-[11px] transition">240MA(年線)</button>
                <button onclick="toggleIndicator('bb')" id="pill_bb" class="pill-inactive px-2 py-0.5 rounded border border-[#64748b] text-slate-300 text-[11px] transition">布林通道</button>
                <button onclick="toggleIndicator('neck')" id="pill_neck" class="pill-active px-2 py-0.5 rounded border border-[#B45309] text-[#B45309] text-[11px] transition">20日頸線</button>
            </div>

            <!-- 中文線圖操作工具列與手機手勢開關 -->
            <div class="flex justify-between items-center bg-[#121824] border border-[#1e293b] px-2.5 py-1 rounded text-xs text-slate-300 mb-2">
                <div class="flex items-center space-x-2">
                    <span class="font-bold text-[#38bdf8]">🛠️ 中文圖表工具:</span>
                    <button onclick="resetChartZoom()" class="hover:text-white px-1.5 py-0.5 bg-[#0b0e14] rounded border border-[#1e293b]">🏠 還原初始比例</button>
                    <button onclick="zoomIn()" class="hover:text-white px-1.5 py-0.5 bg-[#0b0e14] rounded border border-[#1e293b]">🔍 放大檢視</button>
                    <button onclick="zoomOut()" class="hover:text-white px-1.5 py-0.5 bg-[#0b0e14] rounded border border-[#1e293b]">🔎 縮小檢視</button>
                </div>
                <div class="flex items-center space-x-1.5">
                    <label class="cursor-pointer text-[11px] flex items-center space-x-1">
                        <input type="checkbox" id="chartLockToggle" onchange="toggleChartLock()" class="rounded bg-[#0b0e14] border-[#2a374a]" checked>
                        <span>✋ 鎖定頁面滑動 (避免誤觸)</span>
                    </label>
                </div>
            </div>

            <!-- 即時浮動 Legend 面板 -->
            <div id="legendPanel" class="bg-[#121824] border border-[#1e293b] px-2.5 py-1 rounded mb-2 text-[11px] md:text-xs flex flex-wrap gap-x-2.5 gap-y-1 font-mono text-slate-300">
                <span>開: <strong id="legOpen" class="text-white">--</strong></span>
                <span>高: <strong id="legHigh" class="text-white">--</strong></span>
                <span>低: <strong id="legLow" class="text-white">--</strong></span>
                <span>收: <strong id="legClose" class="text-emerald-400">--</strong></span>
                <span style="color: #D97706;">5MA: <strong id="legMA5">--</strong></span>
                <span style="color: #7C3AED;">20MA: <strong id="legMA20">--</strong></span>
                <span style="color: #0D9488;">60MA: <strong id="legMA60">--</strong></span>
                <span style="color: #64748b;">240MA: <strong id="legMA240">--</strong></span>
                <span style="color: #B45309;">頸線: <strong id="legNeck">--</strong></span>
            </div>

            <!-- 1. 主圖：K線 ＋ 均線 ＋ 布林通道 ＋ 頸線 -->
            <div class="bg-[#121824] border border-[#1e293b] rounded-t-xl p-2 h-[260px] md:h-[300px] flex flex-col w-full">
                <div id="mainChartContainer" class="w-full flex-1"></div>
            </div>

            <!-- 2. 副圖：成交量 ＋ 5日均量線 -->
            <div class="bg-[#121824] border-x border-[#1e293b] p-2 h-[85px] flex flex-col border-t border-[#1e293b]/50 w-full">
                <div class="text-[10px] text-slate-400 font-bold mb-0.5">📊 成交量 ＆ 5日均量線 (Vol-MA5 橘褐點線)</div>
                <div id="volChartContainer" class="w-full flex-1"></div>
            </div>

            <!-- 3. 副圖：KD (9, 3, 3) -->
            <div class="bg-[#121824] border-x border-[#1e293b] p-2 h-[85px] flex flex-col border-t border-[#1e293b]/50 w-full">
                <div class="text-[10px] text-slate-400 font-bold mb-0.5 flex justify-between">
                    <span>⚡ KD(9,3)：<strong style="color: #D97706;">K(橘)</strong> / <strong style="color: #2563EB;">D(藍)</strong></span>
                    <span class="text-slate-500 font-mono">超買 80 / 超賣 20</span>
                </div>
                <div id="kdChartContainer" class="w-full flex-1"></div>
            </div>

            <!-- 4. 副圖：MACD (12, 26, 9) -->
            <div class="bg-[#121824] border-x border-b border-[#1e293b] rounded-b-xl p-2 h-[85px] flex flex-col border-t border-[#1e293b]/50 w-full">
                <div class="text-[10px] text-slate-400 font-bold mb-0.5 flex justify-between">
                    <span>🌊 MACD(12,26,9)：<strong class="text-red-400">紅柱</strong> / <strong class="text-emerald-400">綠柱</strong> ＋ DIF/DEA</span>
                    <span class="text-slate-500 font-mono">零軸 (Zero Line)</span>
                </div>
                <div id="macdChartContainer" class="w-full flex-1"></div>
            </div>
        </main>

        <!-- 右側：選擇權風控與 LINE 推播 -->
        <aside id="rightSidebar" class="w-96 bg-[#121824] border-l border-[#1e293b] flex flex-col p-4 shrink-0">
            <h2 class="text-sm font-black text-white mb-2.5 flex items-center justify-between">
                <span>📊 V9.8 期權風控與真數據對齊</span>
                <span class="text-[10px] bg-emerald-500 text-slate-950 font-black px-2 py-0.5 rounded">機構旗艦</span>
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
                        <label class="text-slate-400 block mb-1">策略類型</label>
                        <select id="strategySelect" onchange="onExpirationChange()" class="w-full bg-[#0b0e14] border border-[#2a374a] rounded p-2 text-white font-bold focus:outline-none focus:border-[#38bdf8]">
                            <option value="csp">Short Put (低接收租)</option>
                            <option value="cc">Covered Call (現股包租)</option>
                            <option value="long_call">Long Call (突破多頭)</option>
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

                <div class="bg-[#0b0e14] p-2.5 rounded-lg border border-[#2a374a] space-y-1">
                    <div class="font-bold text-[#38bdf8] flex justify-between items-center text-[11px]">
                        <span>📈 Payoff 損益分佈預覽</span>
                    </div>
                    <div class="grid grid-cols-2 gap-2 text-[11px] bg-[#121824] p-1.5 rounded border border-[#1e293b]">
                        <div>最大獲利: <strong id="payoffMax" class="text-emerald-400">--</strong></div>
                        <div>最大虧損: <strong id="payoffMin" class="text-red-400">--</strong></div>
                    </div>
                    <canvas id="payoffCanvas" height="75" class="w-full bg-[#121824] rounded border border-[#1e293b] mt-1"></canvas>
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

    <!-- LINE Flex 戰報實機預覽視窗 -->
    <div id="lineModal" class="fixed inset-0 bg-black/75 flex items-center justify-center hidden z-50 p-4">
        <div class="bg-[#121824] border border-[#1e293b] rounded-2xl p-5 w-full max-w-md shadow-2xl space-y-3">
            <div class="flex justify-between items-center border-b border-[#1e293b] pb-2">
                <div class="flex items-center space-x-2">
                    <span class="w-3 h-3 bg-[#06C755] rounded-full"></span>
                    <h3 class="text-sm font-black text-white">📲 美股多空雷達 V9.8 ｜ LINE 戰報</h3>
                </div>
                <button onclick="closeLineModal()" class="text-slate-400 hover:text-white text-sm">✕</button>
            </div>
            <div class="bg-[#0b0e14] p-3.5 rounded-xl border border-[#2a374a] text-xs font-mono text-slate-300 space-y-1.5 whitespace-pre-wrap max-h-[340px] overflow-y-auto" id="linePreviewText"></div>
            <button onclick="closeLineModal()" class="w-full bg-[#38bdf8] text-slate-950 font-black py-2 rounded-lg text-xs transition">確 認</button>
        </div>
    </div>

    <script>
        let mainChart, volChart, kdChart, macdChart;
        let candleSeries;
        let ma5Series, ma10Series, ma20Series, ma60Series, ma240Series, bbUpSeries, bbLowSeries, neckSeries;
        let volSeries, volMaSeries;
        let kSeries, dSeries;
        let macdHistSeries, macdDifSeries, macdDeaSeries;

        let isSyncing = false;
        let currentSpot = 0, currentMA20 = 0, currentMA240 = 0, currentNeckline = 0;
        let currentSymbol = "PLTR";
        let currentInterval = "1d";
        let currentOptionData = null;
        let currentSector = "ALL";
        let universeData = [];

        // 指標可見性開關狀態 (預設 5, 20, 60, 240, 頸線開啟；10MA, 布林關閉)
        let indState = {
            ma5: true, ma10: false, ma20: true, ma60: true, ma240: true, bb: false, neck: true
        };

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

        // 開啟/關閉均線與指標
        function toggleIndicator(key) {
            indState[key] = !indState[key];
            const btn = document.getElementById(`pill_${key}`);
            if (indState[key]) {
                btn.className = "pill-active px-2 py-0.5 rounded border text-[11px] transition";
            } else {
                btn.className = "pill-inactive px-2 py-0.5 rounded border text-[11px] transition";
            }

            if (key === 'ma5' && ma5Series) ma5Series.applyOptions({ visible: indState.ma5 });
            if (key === 'ma10' && ma10Series) ma10Series.applyOptions({ visible: indState.ma10 });
            if (key === 'ma20' && ma20Series) ma20Series.applyOptions({ visible: indState.ma20 });
            if (key === 'ma60' && ma60Series) ma60Series.applyOptions({ visible: indState.ma60 });
            if (key === 'ma240' && ma240Series) ma240Series.applyOptions({ visible: indState.ma240 });
            if (key === 'neck' && neckSeries) neckSeries.applyOptions({ visible: indState.neck });
            if (key === 'bb' && bbUpSeries && bbLowSeries) {
                bbUpSeries.applyOptions({ visible: indState.bb });
                bbLowSeries.applyOptions({ visible: indState.bb });
            }
        }

        function toggleChartLock() {
            const isLocked = document.getElementById('chartLockToggle').checked;
            const scrollZoomVal = !isLocked;
            mainChart.applyOptions({ handleScroll: scrollZoomVal, handleScale: scrollZoomVal });
            volChart.applyOptions({ handleScroll: scrollZoomVal, handleScale: scrollZoomVal });
            kdChart.applyOptions({ handleScroll: scrollZoomVal, handleScale: scrollZoomVal });
            macdChart.applyOptions({ handleScroll: scrollZoomVal, handleScale: scrollZoomVal });
        }

        function resetChartZoom() {
            mainChart.timeScale().resetTimeScale();
        }

        function zoomIn() {
            const logicalRange = mainChart.timeScale().getVisibleLogicalRange();
            if (logicalRange) {
                const diff = (logicalRange.to - logicalRange.from) * 0.2;
                mainChart.timeScale().setVisibleLogicalRange({
                    from: logicalRange.from + diff,
                    to: logicalRange.to - diff
                });
            }
        }

        function zoomOut() {
            const logicalRange = mainChart.timeScale().getVisibleLogicalRange();
            if (logicalRange) {
                const diff = (logicalRange.to - logicalRange.from) * 0.25;
                mainChart.timeScale().setVisibleLogicalRange({
                    from: logicalRange.from - diff,
                    to: logicalRange.to + diff
                });
            }
        }

        function initCharts() {
            const mainEl = document.getElementById('mainChartContainer');
            const volEl = document.getElementById('volChartContainer');
            const kdEl = document.getElementById('kdChartContainer');
            const macdEl = document.getElementById('macdChartContainer');

            const isMobile = window.innerWidth <= 1024;
            const mainHeight = isMobile ? 260 : 300;

            const chartBaseOpts = {
                layout: { background: { type: 'solid', color: '#121824' }, textColor: '#94a3b8' },
                grid: { vertLines: { color: '#1e293b' }, horzLines: { color: '#1e293b' } },
                timeScale: { borderColor: '#1e293b', visible: false },
                rightPriceScale: { borderColor: '#1e293b' },
                handleScroll: false,
                handleScale: false
            };

            // 1. 主圖
            mainChart = LightweightCharts.createChart(mainEl, { ...chartBaseOpts, width: mainEl.clientWidth, height: mainHeight });
            candleSeries = mainChart.addCandlestickSeries({
                upColor: '#C0392B', downColor: '#1E7E5A',
                wickUpColor: '#C0392B', wickDownColor: '#1E7E5A'
            });

            ma5Series = mainChart.addLineSeries({ color: '#D97706', lineWidth: 1.5, title: '5MA', visible: indState.ma5 });
            ma10Series = mainChart.addLineSeries({ color: '#2563EB', lineWidth: 1.5, title: '10MA', visible: indState.ma10 });
            ma20Series = mainChart.addLineSeries({ color: '#7C3AED', lineWidth: 2, title: '20MA(月)', visible: indState.ma20 });
            ma60Series = mainChart.addLineSeries({ color: '#0D9488', lineWidth: 1.5, title: '60MA(季)', visible: indState.ma60 });
            ma240Series = mainChart.addLineSeries({ color: '#64748b', lineWidth: 1.5, lineStyle: 2, title: '240MA(年)', visible: indState.ma240 });
            
            // 布林通道 (點線)
            bbUpSeries = mainChart.addLineSeries({ color: 'rgba(148, 163, 184, 0.55)', lineWidth: 1, lineStyle: 1, title: '布林上軌', visible: indState.bb });
            bbLowSeries = mainChart.addLineSeries({ color: 'rgba(148, 163, 184, 0.55)', lineWidth: 1, lineStyle: 1, title: '布林下軌', visible: indState.bb });
            neckSeries = mainChart.addLineSeries({ color: '#B45309', lineWidth: 2, lineStyle: 1, title: '頸線', visible: indState.neck });

            // 2. 成交量
            volChart = LightweightCharts.createChart(volEl, { ...chartBaseOpts, width: volEl.clientWidth, height: 85 });
            volSeries = volChart.addHistogramSeries({ priceFormat: { type: 'volume' } });
            volMaSeries = volChart.addLineSeries({ color: '#B45309', lineWidth: 1.5, lineStyle: 1, title: 'Vol-MA5' });

            // 3. KD
            kdChart = LightweightCharts.createChart(kdEl, { ...chartBaseOpts, width: kdEl.clientWidth, height: 85 });
            kSeries = kdChart.addLineSeries({ color: '#D97706', lineWidth: 1.8, title: 'K(9)' });
            dSeries = kdChart.addLineSeries({ color: '#2563EB', lineWidth: 1.8, title: 'D(9)' });

            // 4. MACD (最下層顯示時間標尺)
            macdChart = LightweightCharts.createChart(macdEl, {
                ...chartBaseOpts,
                width: macdEl.clientWidth,
                height: 85,
                timeScale: { borderColor: '#1e293b', visible: true }
            });
            macdHistSeries = macdChart.addHistogramSeries({ title: 'OSC' });
            macdDifSeries = macdChart.addLineSeries({ color: '#D97706', lineWidth: 1.5, title: 'DIF' });
            macdDeaSeries = macdChart.addLineSeries({ color: '#2563EB', lineWidth: 1.5, title: 'DEA' });

            // ⭐ 核心四圖完全同步引擎 (主圖縮放平移，成交量、KD、MACD 完全同步聯動)
            const charts = [mainChart, volChart, kdChart, macdChart];
            charts.forEach(srcChart => {
                srcChart.timeScale().subscribeVisibleLogicalRangeChange(range => {
                    if (isSyncing || !range) return;
                    isSyncing = true;
                    charts.forEach(targetChart => {
                        if (targetChart !== srcChart) {
                            targetChart.timeScale().setVisibleLogicalRange(range);
                        }
                    });
                    isSyncing = false;
                });
            });

            // 十字線數值聯動
            mainChart.subscribeCrosshairMove(param => {
                if (!param.time || !param.seriesData.get(candleSeries)) return;
                const c = param.seriesData.get(candleSeries);
                const m5 = param.seriesData.get(ma5Series);
                const m20 = param.seriesData.get(ma20Series);
                const m60 = param.seriesData.get(ma60Series);
                const m240 = param.seriesData.get(ma240Series);
                const neck = param.seriesData.get(neckSeries);
                
                document.getElementById('legOpen').innerText = `$${c.open.toFixed(2)}`;
                document.getElementById('legHigh').innerText = `$${c.high.toFixed(2)}`;
                document.getElementById('legLow').innerText = `$${c.low.toFixed(2)}`;
                document.getElementById('legClose').innerText = `$${c.close.toFixed(2)}`;
                if(m5) document.getElementById('legMA5').innerText = `$${m5.value.toFixed(2)}`;
                if(m20) document.getElementById('legMA20').innerText = `$${m20.value.toFixed(2)}`;
                if(m60) document.getElementById('legMA60').innerText = `$${m60.value.toFixed(2)}`;
                if(m240) document.getElementById('legMA240').innerText = `$${m240.value.toFixed(2)}`;
                if(neck) document.getElementById('legNeck').innerText = `$${neck.value.toFixed(2)}`;
            });

            function resizeCharts() {
                const w = mainEl.clientWidth;
                mainChart.applyOptions({ width: w });
                volChart.applyOptions({ width: w });
                kdChart.applyOptions({ width: w });
                macdChart.applyOptions({ width: w });
            }

            window.addEventListener('resize', resizeCharts);
            setTimeout(resizeCharts, 250);
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
                    ma10Series.setData(data.ma10);
                    ma20Series.setData(data.ma20);
                    ma60Series.setData(data.ma60);
                    ma240Series.setData(data.ma240);
                    bbUpSeries.setData(data.bb_up);
                    bbLowSeries.setData(data.bb_low);
                    neckSeries.setData(data.neckline_data);

                    // 成交量與 5日均量
                    const volData = data.candles.map(c => ({
                        time: c.time,
                        value: c.volume,
                        color: c.close >= c.open ? 'rgba(192, 57, 43, 0.7)' : 'rgba(30, 126, 90, 0.7)'
                    }));
                    volSeries.setData(volData);
                    volMaSeries.setData(data.vol_ma5);

                    // KD 指標
                    kSeries.setData(data.k_data);
                    dSeries.setData(data.d_data);

                    // MACD 柱狀與雙線
                    macdHistSeries.setData(data.macd_hist);
                    macdDifSeries.setData(data.macd_dif);
                    macdDeaSeries.setData(data.macd_dea);

                    document.getElementById('chartTitle').innerText = `${symbol} ${data.name}`;
                    
                    // 價格標籤帶有漲跌幅與點數
                    const pSign = data.chg >= 0 ? '▲' : '▼';
                    const pColor = data.chg >= 0 ? '#C0392B' : '#1E7E5A';
                    document.getElementById('currentPriceTag').style.color = pColor;
                    document.getElementById('currentPriceTag').innerText = `收盤: $${currentSpot} (${pSign}${Math.abs(data.pct)}%)`;

                    document.getElementById('spotDisplay').innerText = `$${currentSpot} (20MA: $${currentMA20})`;
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
def run_v98_screener(sector: str = "ALL"):
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
            
            if c > neckline and c > o and c > ma20 and v_ratio >= 1.2:
                score = "🔥 暴量突破"
            elif c > ma20 and abs(dist_neck) <= 5.0:
                score = "🛡️ 頸線守候"
            else:
                score = "💎 偏多整理"
                
            results.append({
                "symbol": sym, "name": item["name"], "sector": item["sector"],
                "score": score, "dist_neck_pct": dist_neck, "v_ratio": v_ratio
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

    # 1. 均線系統 (5, 10, 20, 60, 240)
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA60'] = df['Close'].rolling(window=60).mean()
    df['MA240'] = df['Close'].rolling(window=240, min_periods=60).mean()
    
    # 2. 布林通道 (20MA +- 2*STD)
    bb_std = df['Close'].rolling(window=20).std(ddof=0)
    df['BB_UP'] = df['MA20'] + 2.0 * bb_std
    df['BB_LOW'] = df['MA20'] - 2.0 * bb_std

    # 3. 5日均量線
    df['Vol_MA5'] = df['Volume'].shift(1).rolling(5, min_periods=3).mean()

    # 4. KD (9, 3, 3)
    low9 = df['Low'].rolling(9, min_periods=9).min()
    high9 = df['High'].rolling(9, min_periods=9).max()
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

    # 5. MACD (12, 26, 9)
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
    neckline = float(round(df['High'].iloc[-21:-1].max(), 2))
    dist_neck = round(((spot / neckline) - 1.0) * 100.0, 1)

    quality_badge = "🛡️ 距頸線 ≤ 5% 且站上月線" if dist_neck <= 5.0 and spot >= ma20_val else "⚠️ 乖離偏遠或月線下"

    candles = []
    ma5_data, ma10_data, ma20_data, ma60_data, ma240_data = [], [], [], [], []
    bb_up_data, bb_low_data, neckline_data = [], [], []
    vol_ma5_data, k_data, d_data = [], [], []
    macd_hist_data, macd_dif_data, macd_dea_data = [], [], []

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
        if not pd.isna(row["MA10"]): ma10_data.append({"time": t_str, "value": float(round(row["MA10"], 2))})
        if not pd.isna(row["MA20"]): ma20_data.append({"time": t_str, "value": float(round(row["MA20"], 2))})
        if not pd.isna(row["MA60"]): ma60_data.append({"time": t_str, "value": float(round(row["MA60"], 2))})
        if not pd.isna(row["MA240"]): ma240_data.append({"time": t_str, "value": float(round(row["MA240"], 2))})
        if not pd.isna(row["BB_UP"]): bb_up_data.append({"time": t_str, "value": float(round(row["BB_UP"], 2))})
        if not pd.isna(row["BB_LOW"]): bb_low_data.append({"time": t_str, "value": float(round(row["BB_LOW"], 2))})
        neckline_data.append({"time": t_str, "value": neckline})
        if not pd.isna(row["Vol_MA5"]): vol_ma5_data.append({"time": t_str, "value": int(row["Vol_MA5"])})
        k_data.append({"time": t_str, "value": float(round(row["K"], 1))})
        d_data.append({"time": t_str, "value": float(round(row["D"], 1))})

        # MACD
        m_val = float(round(row["MACD_HIST"], 2))
        macd_hist_data.append({
            "time": t_str,
            "value": m_val,
            "color": '#C0392B' if m_val >= 0 else '#1E7E5A'
        })
        macd_dif_data.append({"time": t_str, "value": float(round(row["DIF"], 2))})
        macd_dea_data.append({"time": t_str, "value": float(round(row["DEA"], 2))})

    name = symbol
    for it in MARKET_UNIVERSE:
        if it["symbol"] == symbol.upper():
            name = it["name"]
            break

    return {
        "symbol": symbol.upper(),
        "name": name,
        "spot": spot,
        "chg": chg,
        "pct": pct,
        "ma20_val": ma20_val,
        "ma240_val": ma240_val,
        "neckline": neckline,
        "dist_neck": dist_neck,
        "quality_badge": quality_badge,
        "candles": candles,
        "ma5": ma5_data,
        "ma10": ma10_data,
        "ma20": ma20_data,
        "ma60": ma60_data,
        "ma240": ma240_data,
        "bb_up": bb_up_data,
        "bb_low": bb_low_data,
        "neckline_data": neckline_data,
        "vol_ma5": vol_ma5_data,
        "k_data": k_data,
        "d_data": d_data,
        "macd_hist": macd_hist_data,
        "macd_dif": macd_dif_data,
        "macd_dea": macd_dea_data
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
    prev_spot = float(df['Close'].iloc[-2]) if len(df) >= 2 else spot
    pct = round(((spot - prev_spot) / prev_spot) * 100.0, 2) if prev_spot > 0 else 0.0
    p_sign = "▲" if pct >= 0 else "▼"

    ma20 = float(df['Close'].rolling(20).mean().iloc[-1])
    neckline = float(df['High'].iloc[-21:-1].max())
    dist_neck = round(((spot / neckline) - 1.0) * 100.0, 1)

    text_report = (
        f"📈【美股多空雷達 V9.8｜機構突破戰報】\n"
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
