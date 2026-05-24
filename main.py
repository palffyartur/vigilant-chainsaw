"""
BTC/USDT 15m Prediction API
============================
FastAPI server that fetches live Binance data,
trains a LightGBM model, and returns next-candle predictions.

Endpoints:
  GET  /          → health check
  POST /predict   → run prediction, returns {direction, confidence, reasoning}
  GET  /predict   → same but no body needed (fetches own data)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import requests
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
import ta
import warnings
import time

warnings.filterwarnings("ignore")

app = FastAPI(title="BTC 15m Predictor", version="1.0.0")

# Allow all origins so your frontend can call this freely
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Cache: retrain at most once every 15 minutes ──────────────────────────────
_cache: dict = {"model": None, "features": None, "trained_at": 0, "ind": None}
CACHE_TTL = 60 * 15   # 15 minutes


# ── Pydantic models ───────────────────────────────────────────────────────────

class Candle(BaseModel):
    t: int          # open timestamp ms
    o: float
    h: float
    l: float
    c: float
    v: float
    buyV: Optional[float] = None

class PredictRequest(BaseModel):
    candles: Optional[List[Candle]] = None   # optional — server fetches if absent


class PredictResponse(BaseModel):
    direction: str          # "UP" | "DOWN"
    confidence: int         # 0–100
    prob_up: float
    prob_down: float
    reasoning: str
    indicators: dict
    model_accuracy: Optional[float] = None


# ── Data helpers ──────────────────────────────────────────────────────────────

def fetch_binance(limit: int = 2000) -> pd.DataFrame:
    url = "https://api.binance.com/api/v3/klines"
    r = requests.get(url, params={"symbol": "BTCUSDT", "interval": "15m", "limit": limit}, timeout=10)
    r.raise_for_status()
    raw = r.json()
    df = pd.DataFrame(raw, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","quote_volume","trades","taker_buy_base","taker_buy_quote","ignore"
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    for col in ["open","high","low","close","volume","taker_buy_base"]:
        df[col] = df[col].astype(float)
    return df.set_index("open_time").sort_index()


def candles_to_df(candles: List[Candle]) -> pd.DataFrame:
    rows = [{"open_time": pd.to_datetime(c.t, unit="ms"),
             "open": c.o, "high": c.h, "low": c.l,
             "close": c.c, "volume": c.v,
             "taker_buy_base": c.buyV or c.v * 0.5} for c in candles]
    df = pd.DataFrame(rows).set_index("open_time").sort_index()
    for col in ["open","high","low","close","volume","taker_buy_base"]:
        df[col] = df[col].astype(float)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]

    f["body"]       = (c - o) / o
    f["upper_wick"] = (h - c.clip(lower=o)) / (h - l + 1e-9)
    f["lower_wick"] = (c.clip(upper=o) - l) / (h - l + 1e-9)
    f["hl_range"]   = (h - l) / o

    for n in [1, 2, 3, 5, 8, 13]:
        f[f"ret_{n}"] = c.pct_change(n)

    f["vol_ratio"]    = v / v.rolling(20).mean()
    f["buy_ratio"]    = df["taker_buy_base"] / (v + 1e-9)
    f["vol_trend"]    = v.pct_change(5)

    for n in [8, 21, 55]:
        f[f"dist_ema{n}"] = (c - c.ewm(span=n).mean()) / c

    rsi_ind = ta.momentum.RSIIndicator(c, window=14)
    f["rsi"] = rsi_ind.rsi() / 100

    stoch = ta.momentum.StochasticOscillator(h, l, c, window=14, smooth_window=3)
    f["stoch_k"] = stoch.stoch() / 100
    f["stoch_d"] = stoch.stoch_signal() / 100

    macd = ta.trend.MACD(c, window_slow=26, window_fast=12, window_sign=9)
    f["macd"]       = macd.macd() / c
    f["macd_signal"]= macd.macd_signal() / c
    f["macd_diff"]  = macd.macd_diff() / c

    bb = ta.volatility.BollingerBands(c, window=20, window_dev=2)
    f["bb_pct"]   = bb.bollinger_pband()
    f["bb_width"] = bb.bollinger_wband() / c

    atr = ta.volatility.AverageTrueRange(h, l, c, window=14)
    f["atr"] = atr.average_true_range() / c

    obv = ta.volume.OnBalanceVolumeIndicator(c, v)
    f["obv_change"] = obv.on_balance_volume().pct_change(5)

    f["hour"]        = df.index.hour / 23
    f["day_of_week"] = df.index.dayofweek / 6
    f["ret_std_20"]  = c.pct_change().rolling(20).std()
    f["ret_skew_20"] = c.pct_change().rolling(20).skew()

    return f


def build_target(df: pd.DataFrame, threshold=0.0001) -> pd.Series:
    future = df["close"].shift(-1) / df["close"] - 1
    t = pd.Series(np.nan, index=df.index)
    t[future >  threshold] = 1
    t[future < -threshold] = 0
    return t


def train_model(df: pd.DataFrame):
    features = build_features(df)
    target   = build_target(df)
    data     = features.join(target.rename("target")).dropna()
    X = data.drop(columns=["target"])
    y = data["target"].astype(int)

    tscv = TimeSeriesSplit(n_splits=5)
    params = dict(
        objective="binary", metric="binary_logloss",
        n_estimators=500, learning_rate=0.03, num_leaves=31,
        min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, verbose=-1, n_jobs=-1,
    )
    accs = []
    model = None
    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        m = lgb.LGBMClassifier(**params)
        m.fit(X_tr, y_tr,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(50, verbose=False),
                         lgb.log_evaluation(period=-1)])
        preds = m.predict(X_val)
        accs.append((preds == y_val.values).mean())
        model = m

    last_features = features.dropna().iloc[[-1]]
    return model, last_features, float(np.mean(accs))


def compute_indicators_summary(df: pd.DataFrame) -> dict:
    c  = df["close"]
    h  = df["high"]
    l  = df["low"]
    v  = df["volume"]
    n  = len(df) - 2

    rsi_ser = ta.momentum.RSIIndicator(c, window=14).rsi()
    macd    = ta.trend.MACD(c, 26, 12, 9)
    bb      = ta.volatility.BollingerBands(c, window=20, window_dev=2)
    buy_p   = df["taker_buy_base"].iloc[n] / (v.iloc[n] + 1e-9) * 100
    vol_ma  = v.rolling(20).mean().iloc[n]

    return {
        "rsi":          round(float(rsi_ser.iloc[n]), 2),
        "macd_diff":    round(float(macd.macd_diff().iloc[n]), 4),
        "bb_pct":       round(float(bb.bollinger_pband().iloc[n]) * 100, 1),
        "buy_pressure": round(float(buy_p), 1),
        "vol_ratio":    round(float(v.iloc[n] / vol_ma), 2) if vol_ma else 1.0,
        "ret_1":        round(float((c.iloc[n] - c.iloc[n-1]) / c.iloc[n-1] * 100), 4),
        "ret_5":        round(float((c.iloc[n] - c.iloc[n-5]) / c.iloc[n-5] * 100), 4),
        "price":        round(float(c.iloc[n]), 2),
    }


def make_reasoning(ind: dict, direction: str, prob_up: float) -> str:
    parts = []
    if ind["rsi"] > 60:   parts.append(f"RSI overbought ({ind['rsi']})")
    elif ind["rsi"] < 40: parts.append(f"RSI oversold ({ind['rsi']})")
    if ind["macd_diff"] > 0:  parts.append("MACD bullish")
    elif ind["macd_diff"] < 0: parts.append("MACD bearish")
    if ind["buy_pressure"] > 55: parts.append(f"strong buy pressure ({ind['buy_pressure']}%)")
    elif ind["buy_pressure"] < 45: parts.append(f"sell pressure ({ind['buy_pressure']}%)")
    if ind["bb_pct"] > 80:  parts.append("near upper band")
    elif ind["bb_pct"] < 20: parts.append("near lower band")
    if not parts:
        parts.append("mixed signals")
    return f"{direction} signal: {', '.join(parts[:3])}."


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def health():
    return {"status": "ok", "service": "BTC 15m Predictor", "version": "1.0.0"}


@app.get("/predict", response_model=PredictResponse)
@app.post("/predict", response_model=PredictResponse)
async def predict(body: Optional[PredictRequest] = None):
    global _cache
    now = time.time()

    try:
        # Use cached model if fresh
        if _cache["model"] and (now - _cache["trained_at"]) < CACHE_TTL:
            model    = _cache["model"]
            features = _cache["features"]
            ind      = _cache["ind"]
            acc      = _cache.get("acc")
        else:
            # Fetch fresh data and retrain
            if body and body.candles and len(body.candles) >= 100:
                df = candles_to_df(body.candles)
            else:
                df = fetch_binance(limit=2000)

            model, features, acc = train_model(df)
            ind = compute_indicators_summary(df)
            _cache = {"model": model, "features": features,
                      "trained_at": now, "ind": ind, "acc": acc}

        # Predict
        prob_up   = float(model.predict_proba(features)[0][1])
        prob_down = 1.0 - prob_up
        direction = "UP" if prob_up > 0.5 else "DOWN"
        confidence = int(max(prob_up, prob_down) * 100)
        reasoning  = make_reasoning(ind, direction, prob_up)

        return PredictResponse(
            direction=direction,
            confidence=confidence,
            prob_up=round(prob_up, 4),
            prob_down=round(prob_down, 4),
            reasoning=reasoning,
            indicators=ind,
            model_accuracy=round(acc * 100, 1) if acc else None,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
