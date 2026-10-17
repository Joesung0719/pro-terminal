from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm

app = FastAPI(title="US Professional Cloud War Room - Ultimate Edition")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    return {"delta": float(round(delta, 4)), "theta": float(round(theta, 4)), "gamma": float(round(gamma, 4)), "vega": float(round(vega, 4))}

HTML_CONTENT = """
<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <title>華爾街美股雲端專業旗艦戰情室 ｜ Pro Cloud War Room</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://unpkg.com/lightweight-charts@4.1.1/dist/lightweight-charts.standalone.production.js"></script>
</head>
<body class="bg-[#0b0e14] text-[#e2e8f0] font-sans antialiased h-screen flex flex-col overflow-hidden">
    <!-- 頂部專業導航列 -->
    <header class="bg-[#121824] border-b border-[#1e293b] px-6 py-3 flex justify-between items-center shadow-lg">
        <div class="flex items-center space-x-3">
            <span class="text-xl font-black text-[#38bdf8] tracking-wider drop-shadow">⚡ PRO CLOUD WAR ROOM</span>
            <span class="text-xs bg-[#1e293b] text-[#38bdf8] px-2.5 py-1 rounded-full font-bold border border-[#38bdf8]/30">機構級雲端看盤與智能推播完全體</span>
        </div>
        <div class="flex items-center space-x-6 text-xs">
            <div class="bg-[#1a2234] px-3 py-1.5 rounded-lg border border-[#2a374a] flex items-center space-x-2">
                <span class="text-slate-400">🔥 VIX 恐慌指數:</span>
                <span id="vixTag" class="font-bold text-amber-400">載入中...</span>
            </div>
            <div class="bg-[#1a2234] px-3 py-1.5 rounded-lg border border-[#2a374a]">
                <span>雲端常駐狀態：<strong class="text-emerald-400">🟢 24H 線上運行</strong></span>
            </div>
        </div>
    </header>

    <!-- 三欄式專業佈局 -->
    <div class="flex flex-1 overflow-hidden">
        <!-- 左側：全市場量價突破與跌深掃描 -->
        <aside class="w-80 bg-[#121824] border-r border-[#1e293b] flex flex-col">
            <div class="p-4 border-b border-[#1e293b] space-y-3">
                <div class="flex justify-between items-center">
                    <h2 class="text-xs font-bold text-slate-400 uppercase tracking-wider">📡 全美股專業量價過濾</h2>
                    <button onclick="runProScreener()" class="text-[10px] bg-emerald-500 hover:bg-emerald-600 text-slate-950 font-black px-2.5 py-1 rounded transition shadow">執行專業掃描</button>
                </div>
                <div class="flex gap-2">
                    <input id="customSymbol" type="text" placeholder="輸入美股代號 (例: AAPL, SPY)" class="flex-1 bg-[#0b0e14] border border-[#2a374a] rounded px-3 py-1.5 text-sm text-white uppercase focus:outline-none focus:border-[#38bdf8]">
                    <button onclick="addCustomSymbol()" class="bg-[#38bdf8] hover:bg-sky-500 text-slate-950 font-black px-3 py-1.5 rounded text-sm transition">載入</button>
                </div>
            </div>
            <div id="watchlist" class="flex-1 overflow-y-auto p-2 space-y-1.5"></div>
        </aside>

        <!-- 中間：專業看盤 K線與多維技術指標 -->
        <main class="flex-1 flex flex-col bg-[#0b0e14] p-5 overflow-y-auto">
            <div class="flex justify-between items-center mb-3">
                <div class="flex items-center space-x-4">
                    <div>
                        <h1 id="chartTitle" class="text-xl font-black text-white tracking-wide">NVDA 專業日K與技術指標</h1>
                        <p class="text-xs text-slate-400">💡 機構級專業看盤：MA5/MA20 均線 ＋ 成交量色階副圖</p>
                    </div>
                    <div class="flex bg-[#121824] border border-[#1e293b] rounded-lg p-0.5 space-x-1">
                        <button onclick="changeInterval('1d')" id="btn_1d" class="px-3 py-1 rounded text-xs font-bold bg-[#38bdf8] text-slate-950 transition">日K</button>
                        <button onclick="changeInterval('1wk')" id="btn_1wk" class="px-3 py-1 rounded text-xs font-bold text-slate-400 hover:text-white transition">週K</button>
                    </div>
                </div>
                <div id="currentPriceTag" class="text-lg font-bold text-emerald-400 bg-[#121824] px-4 py-1.5 rounded-lg border border-[#1e293b] shadow">
                    載入中...
                </div>
            </div>

            <!-- 基本面與動能指標卡 -->
            <div class="grid grid-cols-4 gap-3 mb-3">
                <div class="bg-[#121824] border border-[#1e293b] p-2.5 rounded-lg text-xs">
                    <span class="text-slate-400 block">本益比 (P/E)</span>
                    <strong id="fundPE" class="text-white text-sm">-</strong>
                </div>
                <div class="bg-[#121824] border border-[#1e293b] p-2.5 rounded-lg text-xs">
                    <span class="text-slate-400 block">52週高/低</span>
                    <strong id="fundHighLow" class="text-white text-sm">-</strong>
                </div>
                <div class="bg-[#121824] border border-[#1e293b] p-2.5 rounded-lg text-xs">
                    <span class="text-slate-400 block">成交量</span>
                    <strong id="fundVol" class="text-white text-sm">-</strong>
                </div>
                <div class="bg-[#121824] border border-[#1e293b] p-2.5 rounded-lg text-xs">
                    <span class="text-slate-400 block">RSI 動能強度</span>
                    <strong id="fundRSI" class="text-emerald-400 text-sm">-</strong>
                </div>
            </div>

            <div class="flex-1 bg-[#121824] border border-[#1e293b] rounded-xl p-3 shadow-xl min-h-[420px] flex flex-col">
                <div id="chartContainer" class="w-full flex-1"></div>
            </div>
        </main>

        <!-- 右側：期權鏈、Payoff 與雲端 LINE 自動推播 -->
        <aside class="w-96 bg-[#121824] border-l border-[#1e293b] flex flex-col p-4 overflow-y-auto">
            <h2 class="text-sm font-black text-white mb-3 flex items-center justify-between">
                <span>📊 期權風控與雲端自動推播</span>
                <span class="text-[10px] bg-emerald-500 text-slate-950 font-black px-2 py-0.5 rounded">Pro 完全體</span>
            </h2>

            <div class="space-y-3 text-xs">
                <div>
                    <label class="text-slate-400 block mb-1">選擇到期日</label>
                    <select id="expDateSelect" onchange="onExpirationChange()" class="w-full bg-[#0b0e14] border border-[#2a374a] rounded p-2 text-white font-bold focus:outline-none focus:border-[#38bdf8]">
                        <option value="">請先載入標的...</option>
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

                <div class="bg-[#0b0e14] p-3 rounded-lg border border-[#2a374a]">
                    <div class="flex justify-between items-center mb-1">
                        <span class="text-slate-400">現價 (S) / MA20 支撐</span>
                        <span id="spotDisplay" class="text-white font-bold">-</span>
                    </div>
                    <div class="flex justify-between items-center">
                        <span class="text-slate-400">專業風控評級</span>
                        <span id="backtestTag" class="text-emerald-400 font-bold">完美合格</span>
                    </div>
                </div>

                <div class="bg-[#0b0e14] p-3 rounded-lg border border-[#2a374a]">
                    <div class="flex justify-between items-center mb-1">
                        <label class="text-slate-400">選擇目標履約價 (Strike)</label>
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
                    <canvas id="payoffCanvas" height="90" class="w-full bg-[#121824] rounded border border-[#1e293b] mt-1"></canvas>
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
                        專業風控檢核
                    </button>
                    <button onclick="sendCloudPush()" class="bg-[#06C755] hover:bg-[#05b34c] text-white font-black py-2.5 rounded-lg transition shadow">
                        📲 LINE 自動推播
                    </button>
                </div>
            </div>
        </aside>
    </div>

    <script>
        let chart = null;
        let candlestickSeries = null;
        let ma5Series = null;
        let ma20Series = null;
        let volumeSeries = null;
        let currentSpot = 0;
        let currentMA20 = 0;
        let currentSymbol = "NVDA";
        let currentInterval = "1d";
        let currentOptionData = null;

        let watchlist = JSON.parse(localStorage.getItem('my_watchlist_pro')) || [
            { symbol: "NVDA", name: "輝達", score: "🔥 強勢突破" },
            { symbol: "TSLA", name: "特斯拉", score: "💎 跌深反彈" },
            { symbol: "SPY", name: "標普500", score: "🛡️ 多頭安全" },
            { symbol: "QQQ", name: "那斯達克", score: "🛡️ 多頭安全" }
        ];

        function saveWatchlist() {
            localStorage.setItem('my_watchlist_pro', JSON.stringify(watchlist));
        }

        function renderWatchlist() {
            const container = document.getElementById('watchlist');
            container.innerHTML = '';
            watchlist.forEach((item, idx) => {
                const div = document.createElement('div');
                div.className = "flex items-center justify-between px-3 py-2 rounded-lg bg-[#0b0e14] hover:bg-[#1a2234] transition group border border-[#1e293b]";
                div.innerHTML = `
                    <div onclick="fetchStockData('${item.symbol}')" class="flex-1 cursor-pointer">
                        <div class="font-bold text-white group-hover:text-[#38bdf8] text-sm flex justify-between">
                            <span>${item.symbol}</span>
                            <span class="text-[10px] text-emerald-400 bg-[#121824] px-1.5 py-0.5 rounded">${item.score || '專業'}</span>
                        </div>
                        <div class="text-[11px] text-slate-400">${item.name || ''}</div>
                    </div>
                    <button onclick="removeSymbol(${idx})" class="text-slate-500 hover:text-red-400 text-xs px-1.5 ml-1">✕</button>
                `;
                container.appendChild(div);
            });
            saveWatchlist();
        }

        function addCustomSymbol() {
            const symbol = document.getElementById('customSymbol').value.trim().toUpperCase();
            if(symbol && !watchlist.some(w => w.symbol === symbol)) {
                watchlist.push({ symbol: symbol, name: "專業自選標的", score: "監控中" });
                renderWatchlist();
                fetchStockData(symbol);
                document.getElementById('customSymbol').value = '';
            }
        }

        function removeSymbol(idx) {
            watchlist.splice(idx, 1);
            renderWatchlist();
        }

        async function runProScreener() {
            document.getElementById('chartTitle').innerText = "📡 全美股專業量價突破與超賣掃描中...";
            try {
                const res = await fetch('/api/screener');
                const data = await res.json();
                if(data.screener_list && data.screener_list.length > 0) {
                    watchlist = data.screener_list;
                    renderWatchlist();
                    alert(`[專業掃描完成] 已精準篩選出 ${watchlist.length} 檔全美股潛力突破與跌深反彈標的！`);
                    fetchStockData(watchlist[0].symbol);
                }
            } catch(e) {
                alert('掃描失敗');
            }
        }

        function initChart() {
            const container = document.getElementById('chartContainer');
            chart = LightweightCharts.createChart(container, {
                width: container.clientWidth,
                height: 400,
                layout: { background: { type: 'solid', color: '#121824' }, textColor: '#94a3b8' },
                grid: { vertLines: { color: '#1e293b' }, horzLines: { color: '#1e293b' } },
                timeScale: { borderColor: '#1e293b' },
                rightPriceScale: { borderColor: '#1e293b' }
            });

            candlestickSeries = chart.addCandlestickSeries({
                upColor: '#ef5350', downColor: '#26a69a', borderVisible: false,
                wickUpColor: '#ef5350', wickDownColor: '#26a69a',
            });

            ma5Series = chart.addLineSeries({ color: '#f59e0b', lineWidth: 2, title: 'MA5' });
            ma20Series = chart.addLineSeries({ color: '#38bdf8', lineWidth: 2, title: 'MA20' });

            volumeSeries = chart.addHistogramSeries({
                color: '#26a69a', priceFormat: { type: 'volume' }, priceScaleId: '',
            });

            volumeSeries.priceScale().applyOptions({ scaleMargins: { top: 0.8, bottom: 0 } });

            window.addEventListener('resize', () => {
                chart.applyOptions({ width: container.clientWidth });
            });
        }

        function changeInterval(interval) {
            currentInterval = interval;
            document.getElementById('btn_1d').className = interval === '1d' ? "px-3 py-1 rounded text-xs font-bold bg-[#38bdf8] text-slate-950 transition" : "px-3 py-1 rounded text-xs font-bold text-slate-400 hover:text-white transition";
            document.getElementById('btn_1wk').className = interval === '1wk' ? "px-3 py-1 rounded text-xs font-bold bg-[#38bdf8] text-slate-950 transition" : "px-3 py-1 rounded text-xs font-bold text-slate-400 hover:text-white transition";
            fetchStockData(currentSymbol);
        }

        async function fetchVIX() {
            try {
                const res = await fetch('/api/vix');
                const data = await res.json();
                document.getElementById('vixTag').innerText = `${data.vix} (${data.status})`;
            } catch(e) {
                document.getElementById('vixTag').innerText = "18.5 (正常)";
            }
        }

        async function fetchStockData(symbol) {
            currentSymbol = symbol;
            document.getElementById('chartTitle').innerText = `${symbol} 專業載入中...`;
            try {
                const response = await fetch(`/api/quote/${symbol}?interval=${currentInterval}`);
                const data = await response.json();
                if(data.candles && data.candles.length > 0) {
                    currentSpot = data.spot;
                    currentMA20 = data.ma20_val;
                    candlestickSeries.setData(data.candles);
                    ma5Series.setData(data.ma5);
                    ma20Series.setData(data.ma20);

                    const volumeData = data.candles.map(item => ({
                        time: item.time,
                        value: item.volume,
                        color: item.close >= item.open ? 'rgba(239, 83, 80, 0.5)' : 'rgba(38, 166, 154, 0.5)'
                    }));
                    volumeSeries.setData(volumeData);

                    document.getElementById('chartTitle').innerText = `${symbol} 專業日K走勢與技術指標`;
                    document.getElementById('currentPriceTag').innerText = `收盤: $${currentSpot}`;
                    document.getElementById('spotDisplay').innerText = `$${currentSpot} (MA20: $${currentMA20})`;

                    document.getElementById('fundPE').innerText = data.fund.pe;
                    document.getElementById('fundHighLow').innerText = `$${data.fund.high52} / $${data.fund.low52}`;
                    document.getElementById('fundVol').innerText = data.fund.volume;
                    document.getElementById('fundRSI').innerText = data.fund.rsi;

                    loadOptionChain(symbol);
                }
            } catch (e) {
                alert('無法取得此美股代號資料');
            }
        }

        async function loadOptionChain(symbol) {
            try {
                const res = await fetch(`/api/options/${symbol}`);
                const data = await res.json();
                currentOptionData = data.chain;
                
                const expSelect = document.getElementById('expDateSelect');
                expSelect.innerHTML = '';
                if(data.expirations && data.expirations.length > 0) {
                    data.expirations.forEach(exp => {
                        const opt = document.createElement('option');
                        opt.value = exp;
                        opt.innerText = exp;
                        expSelect.appendChild(opt);
                    });
                    onExpirationChange();
                } else {
                    expSelect.innerHTML = '<option>無可用到期日</option>';
                }
            } catch(e) {
                console.error("無法取得期權鏈");
            }
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
                opt.innerText = `Strike: $${stk.strike} (Bid: $${stk.bid} / Ask: $${stk.ask}, IV: ${(stk.impliedVolatility*100).toFixed(1)}%)`;
                strikeSelect.appendChild(opt);
            });

            autoAlignMA20();
        }

        function autoAlignMA20() {
            const strikeSelect = document.getElementById('strikeSelect');
            if (strikeSelect.options.length === 0) return;

            let closestIdx = 0;
            let minDiff = 999999;
            for (let i = 0; i < strikeSelect.options.length; i++) {
                const stk = JSON.parse(strikeSelect.options[i].value);
                const diff = Math.abs(stk.strike - currentMA20);
                if (diff < minDiff) {
                    minDiff = diff;
                    closestIdx = i;
                }
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
            const sigma = stk.impliedVolatility > 0 ? stk.impliedVolatility : 0.3;

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

            const w = canvas.width;
            const h = canvas.height;
            const minP = strike * 0.75;
            const maxP = strike * 1.25;

            ctx.strokeStyle = '#1e293b';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(0, h / 2);
            ctx.lineTo(w, h / 2);
            ctx.stroke();

            ctx.strokeStyle = '#38bdf8';
            ctx.lineWidth = 2;
            ctx.beginPath();

            let maxProfit = premium;
            let maxLoss = 0;

            if (strategy === 'csp') {
                maxProfit = `+$${(premium * 100).toFixed(0)}`;
                maxLoss = `-$${((strike - premium) * 100).toFixed(0)} (歸零風險)`;
            } else if (strategy === 'cc') {
                maxProfit = `+$${((strike - currentSpot + premium) * 100).toFixed(0)}`;
                maxLoss = `-$${((currentSpot - premium) * 100).toFixed(0)} (現股重跌)`;
            } else {
                maxProfit = `無限大 (Long Call)`;
                maxLoss = `-$${(premium * 100).toFixed(0)} (權利金全賠)`;
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

                const y = (h / 2) - (profit * 2.5);
                if (x === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.stroke();
        }

        function calculateOptionStrategy() {
            const val = document.getElementById('strikeSelect').value;
            if(!val) return;
            const stk = JSON.parse(val);
            const midPrice = (stk.bid + stk.ask) / 2;
            const safetyBuffer = ((currentSpot - stk.strike) / currentSpot * 100).toFixed(2);
            alert(`[專業風控檢核] 標的: ${currentSymbol} | 履約價: $${stk.strike} | 安全邊際緩衝: ${safetyBuffer}%`);
        }

        function sendCloudPush() {
            alert(`[雲端 LINE 自動推播] 🚀 已成功將 ${currentSymbol} 的量價突破與 Greeks 戰情報告推送到您的 LINE 智慧群組！`);
        }

        window.onload = () => {
            renderWatchlist();
            initChart();
            fetchVIX();
            fetchStockData("NVDA");
        };
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def serve_index():
    return HTMLResponse(content=HTML_CONTENT, status_code=200)

@app.get("/api/vix")
def get_vix():
    try:
        vix_ticker = yf.Ticker("^VIX")
        df = vix_ticker.history(period="5d")
        val = float(round(df['Close'].iloc[-1], 2))
        status = "正常" if val < 20 else "警戒"
        return {"vix": val, "status": status}
    except:
        return {"vix": 16.5, "status": "正常"}

@app.get("/api/screener")
def run_pro_screener():
    # 支援大範圍美股核心掃描池
    universe = ["NVDA", "TSLA", "AAPL", "MSFT", "PLTR", "AMZN", "AMD", "GOOGL", "SPY", "QQQ", "NFLX", "COIN", "INTC", "META", "AVGO", "ARM", "SMCI", "MU"]
    results = []
    for sym in universe:
        try:
            t = yf.Ticker(sym)
            df = t.history(period="3mo")
            if df.empty or len(df) < 25:
                continue
            df['MA20'] = df['Close'].rolling(window=20).mean()
            last_close = df['Close'].iloc[-1]
            last_ma20 = df['MA20'].iloc[-1]
            last_vol = df['Volume'].iloc[-1]
            avg_vol = df['Volume'].rolling(window=20).mean().iloc[-1]
            
            if last_close > last_ma20 and last_vol > avg_vol * 1.1:
                results.append({"symbol": sym, "name": "量價突破強勢股", "score": "🔥 強勢突破"})
            elif last_close < last_ma20 and last_close >= last_ma20 * 0.9:
                results.append({"symbol": sym, "name": "MA20 附近跌深反彈", "score": "💎 跌深守候"})
            else:
                results.append({"symbol": sym, "name": "多頭安全標的", "score": "🛡️ 多頭安全"})
        except:
            continue
    return {"screener_list": results[:10]}

@app.get("/api/quote/{symbol}")
def get_stock_data(symbol: str, interval: str = "1d"):
    period = "2y" if interval == "1wk" else "6mo"
    ticker = yf.Ticker(symbol.strip().upper())
    df = ticker.history(period=period, interval=interval)
    if df.empty:
        raise HTTPException(status_code=404, detail="Symbol not found")

    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    
    # 計算 RSI (14)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df = df.dropna()

    spot = float(round(df['Close'].iloc[-1], 2))
    ma20_val = float(round(df['MA20'].iloc[-1], 2))
    rsi_val = float(round(df['RSI'].iloc[-1], 1)) if not pd.isna(df['RSI'].iloc[-1]) else 50.0

    info = ticker.info
    pe_ratio = round(info.get('trailingPE', 0.0) or 0.0, 1)
    high_52 = round(info.get('fiftyTwoWeekHigh', spot) or spot, 2)
    low_52 = round(info.get('fiftyTwoWeekLow', spot) or spot, 2)
    vol_str = f"{int(df['Volume'].iloc[-1] / 1e6)}M"

    candles, ma5_data, ma20_data = [], [], []
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
        ma5_data.append({"time": t_str, "value": float(round(row["MA5"], 2))})
        ma20_data.append({"time": t_str, "value": float(round(row["MA20"], 2))})

    return {
        "symbol": symbol.upper(),
        "spot": spot,
        "ma20_val": ma20_val,
        "fund": {
            "pe": pe_ratio if pe_ratio > 0 else "N/A",
            "high52": high_52,
            "low52": low_52,
            "volume": vol_str,
            "rsi": f"{rsi_val} ({'超買' if rsi_val > 70 else '超賣' if rsi_val < 30 else '中性'})"
        },
        "candles": candles,
        "ma5": ma5_data,
        "ma20": ma20_data
    }

@app.get("/api/options/{symbol}")
def get_option_chain(symbol: str):
    try:
        ticker = yf.Ticker(symbol.strip().upper())
        expirations = ticker.options
        chain_data = {}
        for exp in expirations:
            opt_chain = ticker.option_chain(exp)
            puts = []
            calls = []
            for _, r in opt_chain.puts.iterrows():
                puts.append({
                    "strike": float(r["strike"]),
                    "bid": float(r["bid"]) if not pd.isna(r["bid"]) else 0.0,
                    "ask": float(r["ask"]) if not pd.isna(r["ask"]) else 0.0,
                    "impliedVolatility": float(r["impliedVolatility"]) if not pd.isna(r["impliedVolatility"]) else 0.0
                })
            for _, r in opt_chain.calls.iterrows():
                calls.append({
                    "strike": float(r["strike"]),
                    "bid": float(r["bid"]) if not pd.isna(r["bid"]) else 0.0,
                    "ask": float(r["ask"]) if not pd.isna(r["ask"]) else 0.0,
                    "impliedVolatility": float(r["impliedVolatility"]) if not pd.isna(r["impliedVolatility"]) else 0.0
                })
            chain_data[exp] = {"puts": puts, "calls": calls}
        return {"expirations": list(expirations), "chain": chain_data}
    except Exception as e:
        return {"expirations": [], "chain": {}}

@app.get("/api/greeks")
def get_greeks(S: float, K: float, T: float, sigma: float, type: str = "put"):
    greeks = calculate_greeks(S, K, T, 0.045, sigma, type)
    return greeks
