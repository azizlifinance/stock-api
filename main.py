from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, date, timedelta, time as dtime
import yfinance as yf, pandas as pd, numpy as np, pytz

ET = pytz.timezone("America/New_York")
app = FastAPI()

# Allow your site to call this API
ALLOWED = ["https://www.azizlifinance.com", "https://azizlifinance.com"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED + ["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def last_close_on_or_before(ticker: str, target_d: date):
    start = (pd.Timestamp(target_d) - pd.tseries.offsets.BDay(20)).date()
    end = target_d + timedelta(days=2)
    df = yf.download(ticker, start=start, end=end, interval="1d",
                     progress=False, auto_adjust=False)
    if df.empty or "Close" not in df:
        raise HTTPException(404, "No daily data")
    dates = [ts.date() for ts in df.index]
    idx = None
    for i in range(len(dates)-1, -1, -1):
        if dates[i] <= target_d:
            idx = i; break
    if idx is None:
        raise HTTPException(404, f"No close on/before {target_d}")
    ts = df.index[idx]
    px = float(df["Close"].iloc[idx])
    # show official close time as 16:00 ET
    close_utc = ET.localize(datetime.combine(ts.date(), dtime(16,0))).astimezone(pytz.UTC)
    return px, close_utc, ts.date()

def previous_trading_day(ref_date: date) -> date:
    start = ref_date - timedelta(days=30)
    df = yf.download("SPY", start=start, end=ref_date + timedelta(days=1),
                     interval="1d", progress=False, auto_adjust=False)
    if df.empty:
        return ref_date - timedelta(days=1)
    dates = [ts.date() for ts in df.index if ts.date() < ref_date]
    return dates[-1] if dates else ref_date - timedelta(days=1)

@app.get("/health")
def health(): return {"ok": True}

@app.get("/price")
def price(ticker: str, date_str: str | None = None):
    t = ticker.upper().strip()
    if not t:
        raise HTTPException(400, "ticker required")
    d = datetime.fromisoformat(date_str).date() if date_str else datetime.now(ET).date()
    close, asof_utc, d_used = last_close_on_or_before(t, d)
    prev_day = previous_trading_day(d_used)
    prev_close, _, _ = last_close_on_or_before(t, prev_day)
    chg = close - prev_close
    pct = (chg / prev_close * 100.0) if prev_close else 0.0
    name = t
    try:
        info = yf.Ticker(t).get_info()
        name = info.get("longName") or info.get("shortName") or t
    except Exception:
        pass
    return {
        "ticker": t, "name": name,
        "price": round(close, 2),
        "change": round(chg, 2),
        "pct": round(pct, 2),
        "basis": "official daily close",
        "as_of": asof_utc.isoformat()
    }

RANGE = {
    "1D": ("1d","1m"), "5D": ("5d","5m"), "1M": ("1mo","30m"),
    "6M": ("6mo","1d"), "YTD": ("ytd","1d"), "1Y": ("1y","1d"),
    "5Y": ("5y","1wk"), "ALL": ("max","1mo")
}

@app.get("/history")
def history(ticker: str, range: str = "5D", normalize: bool = False):
    t = ticker.upper().strip()
    per, interval = RANGE.get(range.upper(), ("5d","5m"))
    df = yf.download(t, period=per, interval=interval, progress=False, auto_adjust=False)
    if df.empty or "Close" not in df:
        raise HTTPException(404, "No history")
    s = df["Close"].dropna()
    y = s.to_numpy(dtype=float)
    if normalize and len(y) > 0:
        y = (y / y[0] - 1.0) * 100.0
    x = [pd.Timestamp(ts).isoformat() for ts in s.index]
    return {"ticker": t, "x": x, "y": y.tolist(), "normalized": bool(normalize)}
