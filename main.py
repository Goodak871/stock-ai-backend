"""
台股 AI 分析師 - FastAPI 後端
資料來源：TWSE API + FinMind + yfinance
啟動方式：uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
import asyncio

app = FastAPI(title="台股AI分析師", version="1.0.0")

@app.middleware("http")
async def cors_handler(request, call_next):
    if request.method == "OPTIONS":
        from fastapi.responses import Response
        response = Response()
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "*"
        return response
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)

FINMIND_TOKEN = ""  # 填入你的 Token（可留空，免費限額）
FINMIND_BASE = "https://api.finmindtrade.com/api/v4/data"
TWSE_BASE = "https://openapi.twse.com.tw/v1"

# ─── 工具函數 ─────────────────────────────────────────────────────────────────

def days_ago(n: int) -> str:
    return (datetime.now() - timedelta(days=n)).strftime("%Y-%m-%d")

def safe_float(v, dec=2):
    try:
        f = float(v)
        return round(f, dec) if not (np.isnan(f) or np.isinf(f)) else None
    except:
        return None

def sma(series: list, n: int) -> Optional[float]:
    if len(series) < n:
        return None
    return round(sum(series[-n:]) / n, 2)

def compute_rsi(closes: list, period=14) -> Optional[float]:
    if len(closes) < period + 1:
        return None
    deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
    gains = [d if d > 0 else 0 for d in deltas[-period:]]
    losses = [-d if d < 0 else 0 for d in deltas[-period:]]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)

def compute_macd(closes: list):
    if len(closes) < 26:
        return None, None, None
    def ema(data, n):
        k = 2 / (n + 1)
        e = data[0]
        for v in data[1:]:
            e = v * k + e * (1 - k)
        return e
    ema12 = ema(closes[-26:], 12)
    ema26 = ema(closes[-26:], 26)
    macd_val = round(ema12 - ema26, 3)
    # 用近9期MACD近似signal
    macd_series = []
    for i in range(max(0, len(closes)-35), len(closes)):
        sub = closes[max(0,i-25):i+1]
        if len(sub) >= 26:
            e12 = ema(sub[-26:], 12)
            e26 = ema(sub[-26:], 26)
            macd_series.append(e12 - e26)
    signal = round(ema(macd_series, 9), 3) if len(macd_series) >= 9 else macd_val * 0.85
    hist = round(macd_val - signal, 3)
    return macd_val, signal, hist

def compute_atr(highs, lows, closes, period=14):
    if len(closes) < period + 1:
        return None
    trs = []
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i-1]),
            abs(lows[i] - closes[i-1])
        )
        trs.append(tr)
    return round(sum(trs[-period:]) / period, 2)

# ─── TWSE API ─────────────────────────────────────────────────────────────────

async def fetch_twse_stock_info(stock_id: str) -> dict:
    """從TWSE取得股票基本資料"""
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            r = await client.get(f"{TWSE_BASE}/exchangeReport/STOCK_DAY_ALL")
            if r.status_code == 200:
                data = r.json()
                for item in data:
                    if item.get("Code") == stock_id:
                        return {
                            "name": item.get("Name", stock_id),
                            "close": safe_float(item.get("ClosingPrice", "").replace(",", "")),
                            "change": safe_float(item.get("Change", "").replace(",", "")),
                            "volume": item.get("TradeVolume", "0").replace(",", ""),
                        }
        except Exception as e:
            print(f"TWSE error: {e}")
    return {}

async def fetch_twse_daily(stock_id: str) -> dict:
    """今日即時報價（TWSE）"""
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            date_str = datetime.now().strftime("%Y%m%d")
            r = await client.get(
                f"{TWSE_BASE}/exchangeReport/STOCK_DAY",
                params={"response": "json", "date": date_str, "stockNo": stock_id}
            )
            if r.status_code == 200:
                j = r.json()
                if j.get("data"):
                    last = j["data"][-1]
                    return {
                        "date": last[0],
                        "open": safe_float(last[3].replace(",", "")),
                        "high": safe_float(last[4].replace(",", "")),
                        "low": safe_float(last[5].replace(",", "")),
                        "close": safe_float(last[6].replace(",", "")),
                        "volume": last[1].replace(",", ""),
                    }
        except Exception as e:
            print(f"TWSE daily error: {e}")
    return {}

async def fetch_twse_margin(stock_id: str) -> dict:
    """融資融券（TWSE）"""
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            r = await client.get(
                f"{TWSE_BASE}/marginTrading/MI_MARGN",
                params={"response": "json", "stockNo": stock_id}
            )
            if r.status_code == 200:
                j = r.json()
                items = j.get("selectMIMargnAFData", [])
                if items:
                    last = items[-1]
                    return {
                        "margin_balance": safe_float(last.get("financeBuyBalance", "0").replace(",", ""), 0),
                        "margin_chg": safe_float(last.get("financeBuyToday", "0").replace(",", ""), 0),
                        "short_balance": safe_float(last.get("borrowSellBalance", "0").replace(",", ""), 0),
                        "short_chg": safe_float(last.get("borrowSellToday", "0").replace(",", ""), 0),
                    }
        except Exception as e:
            print(f"TWSE margin error: {e}")
    return {}

# ─── FinMind API ─────────────────────────────────────────────────────────────

async def fetch_finmind(dataset: str, stock_id: str, start_date: str) -> list:
    params = {"dataset": dataset, "data_id": stock_id, "start_date": start_date}
    if FINMIND_TOKEN:
        params["token"] = FINMIND_TOKEN
    async with httpx.AsyncClient(timeout=15) as client:
        try:
            r = await client.get(FINMIND_BASE, params=params)
            j = r.json()
            return j.get("data", []) if j.get("status") == 200 else []
        except Exception as e:
            print(f"FinMind [{dataset}] error: {e}")
            return []

# ─── yfinance ────────────────────────────────────────────────────────────────

def fetch_yfinance(stock_id: str) -> dict:
    """用yfinance取得歷史K線與基本面"""
    ticker_id = f"{stock_id}.TW" if stock_id.isdigit() else stock_id
    try:
        tk = yf.Ticker(ticker_id)
        hist = tk.history(period="2y")
        info = tk.info

        if hist.empty:
            # 嘗試 TWO (上櫃)
            ticker_id = f"{stock_id}.TWO"
            tk = yf.Ticker(ticker_id)
            hist = tk.history(period="2y")
            info = tk.info

        if hist.empty:
            return {}

        closes = hist["Close"].tolist()
        highs = hist["High"].tolist()
        lows = hist["Low"].tolist()
        volumes = hist["Volume"].tolist()
        dates = [d.strftime("%Y-%m-%d") for d in hist.index]

        return {
            "closes": closes,
            "highs": highs,
            "lows": lows,
            "volumes": volumes,
            "dates": dates,
            "info": {
                "name": info.get("longName") or info.get("shortName"),
                "industry": info.get("industry"),
                "sector": info.get("sector"),
                "pe": safe_float(info.get("trailingPE")),
                "pb": safe_float(info.get("priceToBook")),
                "eps": safe_float(info.get("trailingEps")),
                "dividend_yield": safe_float(info.get("dividendYield", 0), 4),
                "market_cap": info.get("marketCap"),
                "52w_high": safe_float(info.get("fiftyTwoWeekHigh")),
                "52w_low": safe_float(info.get("fiftyTwoWeekLow")),
                "beta": safe_float(info.get("beta")),
                "roe": safe_float(info.get("returnOnEquity"), 4),
                "revenue_growth": safe_float(info.get("revenueGrowth"), 4),
                "gross_margin": safe_float(info.get("grossMargins"), 4),
                "operating_margin": safe_float(info.get("operatingMargins"), 4),
                "profit_margin": safe_float(info.get("profitMargins"), 4),
                "free_cashflow": info.get("freeCashflow"),
                "description": info.get("longBusinessSummary", "")[:200] if info.get("longBusinessSummary") else "",
            }
        }
    except Exception as e:
        print(f"yfinance error: {e}")
        return {}

# ─── 主分析端點 ───────────────────────────────────────────────────────────────

@app.get("/api/analyze/{stock_id}")
async def analyze_stock(stock_id: str):
    stock_id = stock_id.strip().upper()

    # 並行抓取所有資料
    twse_task = fetch_twse_stock_info(stock_id)
    margin_task = fetch_twse_margin(stock_id)
    inst_task = fetch_finmind("TaiwanStockInstitutionalInvestors", stock_id, days_ago(30))
    rev_task = fetch_finmind("TaiwanStockMonthRevenue", stock_id, days_ago(400))
    fin_task = fetch_finmind("TaiwanStockFinancialStatements", stock_id, days_ago(500))

    twse_info, margin_data, inst_data, rev_data, fin_data = await asyncio.gather(
        twse_task, margin_task, inst_task, rev_task, fin_task
    )

    # yfinance（同步，在執行緒中跑）
    loop = asyncio.get_event_loop()
    yf_data = await loop.run_in_executor(None, fetch_yfinance, stock_id)

    if not yf_data and not twse_info:
        raise HTTPException(status_code=404, detail=f"找不到股票 {stock_id} 的資料")

    # ── 技術指標計算 ──
    closes = yf_data.get("closes", [])
    highs = yf_data.get("highs", [])
    lows = yf_data.get("lows", [])
    volumes = yf_data.get("volumes", [])
    yf_info = yf_data.get("info", {})

    current_price = None
    if closes:
        current_price = round(closes[-1], 2)
    if twse_info.get("close"):
        current_price = twse_info["close"]

    if not current_price:
        raise HTTPException(status_code=404, detail="無法取得現價")

    # Moving averages
    ma5 = sma(closes, 5)
    ma20 = sma(closes, 20)
    ma60 = sma(closes, 60)
    ma120 = sma(closes, 120)
    ma240 = sma(closes, 240)

    # 52W
    c252 = closes[-252:] if len(closes) >= 252 else closes
    h52 = round(max(c252), 2) if c252 else None
    l52 = round(min(c252), 2) if c252 else None
    pos52 = round(((current_price - l52) / (h52 - l52)) * 100) if h52 and l52 and h52 != l52 else None

    # RSI / MACD / ATR
    rsi = compute_rsi(closes)
    macd_val, signal_val, macd_hist = compute_macd(closes)
    atr = compute_atr(highs, lows, closes) if highs and lows else None

    # Volume ratio
    vol5 = sma(volumes, 5)
    vol20 = sma(volumes, 20)
    vol_ratio = round(vol5 / vol20, 2) if vol5 and vol20 else None

    # Period returns
    def pct_chg(n):
        if len(closes) > n:
            return round((closes[-1] / closes[-n-1] - 1) * 100, 1)
        return None

    chg_1m = pct_chg(22)
    chg_3m = pct_chg(66)
    chg_1y = pct_chg(252)

    # Bias
    bias60 = round((current_price / ma60 - 1) * 100, 1) if ma60 else None

    # MA signals
    ma_warn = []
    if ma5 and current_price < ma5: ma_warn.append("MA5")
    if ma20 and current_price < ma20: ma_warn.append("MA20")
    ma_signal = f"⚠️ 跌破{'/'.join(ma_warn)}" if ma_warn else "✅ 站上均線"

    macd_cross = None
    if macd_val is not None and signal_val is not None:
        macd_cross = "黃金交叉↑" if macd_val > signal_val else "死亡交叉↓"

    rsi_label = None
    if rsi is not None:
        rsi_label = "超買⚠️" if rsi > 70 else ("超賣📉" if rsi < 30 else "正常")

    # ── 籌碼 ──
    inst_map = {}
    for r in inst_data:
        d = r.get("date", "")
        if d not in inst_map:
            inst_map[d] = {"fii": 0, "it": 0, "dealer": 0}
        name = r.get("name", "")
        val = (float(r.get("buy", 0)) - float(r.get("sell", 0))) / 1000  # 張
        if "外資" in name:
            inst_map[d]["fii"] += val
        elif "投信" in name:
            inst_map[d]["it"] += val
        elif "自營" in name:
            inst_map[d]["dealer"] += val

    recent_dates = sorted(inst_map.keys())[-5:]
    fii_net = sum(inst_map[d]["fii"] for d in recent_dates)
    it_net = sum(inst_map[d]["it"] for d in recent_dates)
    dealer_net = sum(inst_map[d]["dealer"] for d in recent_dates)
    total_net = round(fii_net + it_net + dealer_net)

    # Daily institutional for chart
    inst_daily = [
        {
            "date": d,
            "fii": round(inst_map[d]["fii"]),
            "it": round(inst_map[d]["it"]),
            "dealer": round(inst_map[d]["dealer"]),
        }
        for d in sorted(inst_map.keys())[-10:]
    ]

    # ── 營收 ──
    rev_sorted = sorted(rev_data, key=lambda x: x.get("date", ""))
    rev_recent = rev_sorted[-6:] if rev_sorted else []
    last_rev = rev_sorted[-1] if rev_sorted else None
    prev_year_rev = None
    if last_rev:
        last_yr = str(int(last_rev["date"][:4]) - 1)
        last_mo = last_rev["date"][5:7]
        prev_year_rev = next(
            (r for r in rev_sorted if r["date"][:4] == last_yr and r["date"][5:7] == last_mo), None
        )
    rev_yoy = None
    if last_rev and prev_year_rev:
        try:
            rev_yoy = round((float(last_rev["revenue"]) / float(prev_year_rev["revenue"]) - 1) * 100, 1)
        except:
            pass

    # ── 財報 ──
    def get_fin(type_name):
        items = [f for f in fin_data if f.get("type") == type_name]
        if not items:
            return None
        items.sort(key=lambda x: x.get("date", ""))
        return safe_float(items[-1].get("value"))

    opm = get_fin("OperatingIncomeMargin") or (yf_info.get("operating_margin", 0) * 100 if yf_info.get("operating_margin") else None)
    npm = get_fin("NetIncomeMargin") or (yf_info.get("profit_margin", 0) * 100 if yf_info.get("profit_margin") else None)
    eps = get_fin("EPS") or yf_info.get("eps")
    roe = get_fin("ROE") or (yf_info.get("roe", 0) * 100 if yf_info.get("roe") else None)
    gross_margin = get_fin("GrossMargin") or (yf_info.get("gross_margin", 0) * 100 if yf_info.get("gross_margin") else None)

    # PE
    pe = None
    if yf_info.get("pe"):
        pe = yf_info["pe"]
    elif eps and eps > 0:
        pe = round(current_price / eps, 1)

    # FCF
    fcf = yf_info.get("free_cashflow")
    fcf_b = round(fcf / 1e8, 1) if fcf else None  # 億

    # ── 多空評分 ──
    score = 0
    if ma5 and current_price > ma5: score += 1
    if ma20 and current_price > ma20: score += 1
    if ma60 and current_price > ma60: score += 1
    if rsi and 40 < rsi < 75: score += 1
    if macd_val and signal_val and macd_val > signal_val: score += 1
    if total_net > 0: score += 1
    if rev_yoy and rev_yoy > 0: score += 1
    if chg_1m and chg_1m > 0: score += 1

    if score >= 7: sentiment = "強勢偏多🔥"
    elif score >= 5: sentiment = "中性偏多📈"
    elif score >= 3: sentiment = "中性偏空📉"
    else: sentiment = "弱勢偏空❄️"

    sentiment_color = "#3fb950" if score >= 7 else "#79c0ff" if score >= 5 else "#f7a23e" if score >= 3 else "#f85149"

    # ── 融資融券 ──
    margin_balance = margin_data.get("margin_balance")
    margin_chg = margin_data.get("margin_chg")
    short_balance = margin_data.get("short_balance")
    short_chg = margin_data.get("short_chg")

    # Price history for chart (last 60 days)
    price_history = []
    if closes and yf_data.get("dates"):
        dates = yf_data["dates"]
        for i in range(max(0, len(closes)-60), len(closes)):
            price_history.append({
                "date": dates[i] if i < len(dates) else "",
                "close": round(closes[i], 2),
                "volume": int(volumes[i]) if i < len(volumes) else 0,
                "ma5": round(sma(closes[:i+1], 5) or closes[i], 2),
                "ma20": round(sma(closes[:i+1], 20) or closes[i], 2),
                "ma60": round(sma(closes[:i+1], 60) or closes[i], 2),
            })

    now_str = datetime.now().strftime("%Y/%m/%d %H:%M")

    return {
        "meta": {
            "stock_id": stock_id,
            "name": yf_info.get("name") or twse_info.get("name") or stock_id,
            "industry": yf_info.get("industry") or "—",
            "sector": yf_info.get("sector") or "—",
            "timestamp": now_str,
            "description": yf_info.get("description") or "",
        },
        "price": {
            "current": current_price,
            "change": twse_info.get("change"),
            "change_pct": round((twse_info.get("change", 0) / (current_price - twse_info.get("change", 0))) * 100, 2) if twse_info.get("change") else None,
            "open": twse_info.get("open"),
            "high": twse_info.get("high"),
            "low": twse_info.get("low"),
            "high_52w": h52,
            "low_52w": l52,
            "pos_52w": pos52,
        },
        "technical": {
            "ma5": ma5, "ma20": ma20, "ma60": ma60, "ma120": ma120, "ma240": ma240,
            "rsi": rsi, "rsi_label": rsi_label,
            "macd": macd_val, "signal": signal_val, "macd_hist": macd_hist, "macd_cross": macd_cross,
            "atr": atr,
            "vol_ratio": vol_ratio,
            "bias60": bias60,
            "ma_signal": ma_signal,
            "chg_1m": chg_1m, "chg_3m": chg_3m, "chg_1y": chg_1y,
        },
        "valuation": {
            "pe": pe, "pb": yf_info.get("pb"), "eps": eps,
            "roe": round(roe, 1) if roe else None,
            "dividend_yield": round((yf_info.get("dividend_yield") or 0) * 100, 2) if yf_info.get("dividend_yield") else None,
            "market_cap": yf_info.get("market_cap"),
            "beta": yf_info.get("beta"),
        },
        "profitability": {
            "gross_margin": round(gross_margin, 1) if gross_margin else None,
            "opm": round(opm, 1) if opm else None,
            "npm": round(npm, 1) if npm else None,
            "fcf_b": fcf_b,
            "revenue_growth": round((yf_info.get("revenue_growth") or 0) * 100, 1) if yf_info.get("revenue_growth") else None,
        },
        "chips": {
            "fii_net": round(fii_net), "it_net": round(it_net), "dealer_net": round(dealer_net),
            "total_net": total_net,
            "margin_balance": margin_balance, "margin_chg": margin_chg,
            "short_balance": short_balance, "short_chg": short_chg,
            "inst_daily": inst_daily,
        },
        "revenue": {
            "recent": [{"date": r.get("date",""), "revenue": int(r.get("revenue",0))} for r in rev_recent],
            "yoy": rev_yoy,
        },
        "summary": {
            "score": score,
            "max_score": 8,
            "sentiment": sentiment,
            "sentiment_color": sentiment_color,
        },
        "charts": {
            "price_history": price_history,
        }
    }


@app.get("/api/health")
async def health():
    return {"status": "ok", "time": datetime.now().isoformat()}
