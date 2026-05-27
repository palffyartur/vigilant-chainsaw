"""
BTC Prediction API — v6
========================
Supports both 15m and 1h intervals via query param:
  GET /predict?interval=15m
  GET /predict?interval=1h  (default)

Each interval has its own cache so switching is instant
after first load.
"""

from fastapi import FastAPI, HTTPException, Query
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

app = FastAPI(title="BTC Predictor", version="6.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Separate cache per interval
_cache = {
    "15m": {"model": None, "features": None, "trained_at": 0, "ind": None, "acc": None},
    "1h":  {"model": None, "features": None, "trained_at": 0, "ind": None, "acc": None},
}

# Config per interval
INTERVAL_CONFIG = {
    "15m": {
        "binance_interval": "15m",
        "fetch_limit":      2000,
        "threshold":        0.00005,
        "cache_ttl":        60 * 5,      # retrain every 5 min
        "lookahead":        1,
    },
    "1h": {
        "binance_interval": "1h",
        "fetch_limit":      5000,
        "threshold":        0.0002,
        "cache_ttl":        60 * 60,     # retrain every 1 hour
        "lookahead":        1,
    },
}


class PredictResponse(BaseModel):
    interval: str
    direction: str
    confidence: int
    prob_up: float
    prob_down: float
    reasoning: str
    indicators: dict
    model_accuracy: Optional[float] = None


# ── Data fetch ────────────────────────────────────────────────────────────────

def fetch_binance(binance_interval: str, limit: int) -> pd.DataFrame:
    url = "https://api.binance.com/api/v3/klines"
    r = requests.get(url, params={
        "symbol": "BTCUSDT",
        "interval": binance_interval,
        "limit": limit
    }, timeout=15)
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


# ── Features ──────────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]

    f["body"]           = (c - o) / o
    f["upper_wick"]     = (h - c.clip(lower=o)) / (h - l + 1e-9)
    f["lower_wick"]     = (c.clip(upper=o) - l) / (h - l + 1e-9)
    f["hl_range"]       = (h - l) / o
    f["rel_size"]       = (h - l) / ((h - l).rolling(20).mean() + 1e-9)
    f["gap"]            = (o - c.shift(1)) / (c.shift(1) + 1e-9)
    f["trend_strength"] = abs(c - c.shift(20)) / ((h - l).rolling(20).mean() * 20 + 1e-9)

    for n in [1, 2, 3, 5, 8, 13, 24]:
        f[f"ret_{n}"]   = c.pct_change(n)

    f["vol_ratio"]      = v / v.rolling(20).mean()
    f["buy_ratio"]      = df["taker_buy_base"] / (v + 1e-9)
    f["vol_trend"]      = v.pct_change(5)

    for n in [8, 21, 55, 200]:
        f[f"dist_ema{n}"] = (c - c.ewm(span=n).mean()) / c

    f["rsi"]            = ta.momentum.RSIIndicator(c, window=14).rsi() / 100
    stoch = ta.momentum.StochasticOscillator(h, l, c, window=14, smooth_window=3)
    f["stoch_k"]        = stoch.stoch() / 100
    f["stoch_d"]        = stoch.stoch_signal() / 100
    macd = ta.trend.MACD(c, window_slow=26, window_fast=12, window_sign=9)
    f["macd"]           = macd.macd() / c
    f["macd_signal"]    = macd.macd_signal() / c
    f["macd_diff"]      = macd.macd_diff() / c

    bb = ta.volatility.BollingerBands(c, window=20, window_dev=2)
    f["bb_pct"]         = bb.bollinger_pband()
    f["bb_width"]       = bb.bollinger_wband() / c
    f["atr"]            = ta.volatility.AverageTrueRange(h, l, c, window=14).average_true_range() / c
    f["obv_change"]     = ta.volume.OnBalanceVolumeIndicator(c, v).on_balance_volume().pct_change(5)

    f["hour"]           = df.index.hour / 23
    f["day_of_week"]    = df.index.dayofweek / 6
    f["ret_std_20"]     = c.pct_change().rolling(20).std()
    f["ret_skew_20"]    = c.pct_change().rolling(20).skew()

    return f


# ── Target ────────────────────────────────────────────────────────────────────

def build_target(df: pd.DataFrame, threshold: float) -> pd.Series:
    future = df["close"].shift(-1) / df["close"] - 1
    t = pd.Series(np.nan, index=df.index)
    t[future >  threshold] = 1
    t[future < -threshold] = 0
    return t


# ── Train ─────────────────────────────────────────────────────────────────────

def train_model(df: pd.DataFrame, threshold: float, interval: str):
    features = build_features(df)
    target   = build_target(df, threshold)
    data     = features.join(target.rename("target")).dropna()

    X = data.drop(columns=["target"])
    y = data["target"].astype(int)

    print(f"[{interval}] Samples: {len(X)} | UP: {y.sum()} ({y.mean()*100:.1f}%) | DOWN: {(1-y).sum()}", flush=True)

    params = dict(
        objective="binary",
        metric="binary_logloss",
        learning_rate=0.01,
        num_leaves=31,
        min_child_samples=50,
        n_estimators=1000,
        is_unbalance=True,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.05,
        reg_lambda=0.5,
        verbose=-1,
        n_jobs=-1,
    )

    accs, model = [], None
    for fold, (train_idx, val_idx) in enumerate(TimeSeriesSplit(n_splits=5).split(X), 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        m = lgb.LGBMClassifier(**params)
        m.fit(X_tr, y_tr)
        preds = m.predict(X_val)
        acc = (preds == y_val.values).mean()
        accs.append(acc)
        print(f"[{interval}] Fold {fold} accuracy: {acc*100:.1f}%", flush=True)
        model = m

    mean_acc = float(np.mean(accs))
    print(f"[{interval}] CV ACCURACY: {mean_acc*100:.1f}%", flush=True)

    last_features = features.dropna().iloc[[-1]]
    return model, last_features, mean_acc


# ── Indicators ────────────────────────────────────────────────────────────────

def compute_indicators(df: pd.DataFrame) -> dict:
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    n = len(df) - 2
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


def make_reasoning(ind: dict, direction: str) -> str:
    parts = []
    if ind["rsi"] > 60:            parts.append(f"RSI overbought ({ind['rsi']})")
    elif ind["rsi"] < 40:          parts.append(f"RSI oversold ({ind['rsi']})")
    if ind["macd_diff"] > 0:       parts.append("MACD bullish")
    elif ind["macd_diff"] < 0:     parts.append("MACD bearish")
    if ind["buy_pressure"] > 55:   parts.append(f"strong buy pressure ({ind['buy_pressure']}%)")
    elif ind["buy_pressure"] < 45: parts.append(f"sell pressure ({ind['buy_pressure']}%)")
    if ind["bb_pct"] > 80:         parts.append("near upper band")
    elif ind["bb_pct"] < 20:       parts.append("near lower band")
    if not parts:                  parts.append("mixed signals")
    return f"{direction} signal: {', '.join(parts[:3])}."


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def health():
    return {"status": "ok", "service": "BTC Predictor", "version": "6.0.0",
            "supported_intervals": ["15m", "1h"]}


@app.get("/predict", response_model=PredictResponse)
@app.post("/predict", response_model=PredictResponse)
async def predict(interval: str = Query(default="1h", regex="^(15m|1h)$")):
    global _cache
    now = time.time()

    if interval not in INTERVAL_CONFIG:
        raise HTTPException(status_code=400, detail="interval must be '15m' or '1h'")

    cfg   = INTERVAL_CONFIG[interval]
    cache = _cache[interval]

    try:
        if cache["model"] and (now - cache["trained_at"]) < cfg["cache_ttl"]:
            model    = cache["model"]
            features = cache["features"]
            ind      = cache["ind"]
            acc      = cache["acc"]
        else:
            print(f"[{interval}] Training new model...", flush=True)
            df = fetch_binance(cfg["binance_interval"], cfg["fetch_limit"])
            model, features, acc = train_model(df, cfg["threshold"], interval)
            ind = compute_indicators(df)
            _cache[interval] = {
                "model": model, "features": features,
                "trained_at": now, "ind": ind, "acc": acc
            }

        prob_up    = float(model.predict_proba(features)[0][1])
        prob_down  = 1.0 - prob_up
        direction  = "UP" if prob_up > 0.5 else "DOWN"
        confidence = int(max(prob_up, prob_down) * 100)

        return PredictResponse(
            interval=interval,
            direction=direction,
            confidence=confidence,
            prob_up=round(prob_up, 4),
            prob_down=round(prob_down, 4),
            reasoning=make_reasoning(ind, direction),
            indicators=ind,
            model_accuracy=round(acc * 100, 1) if acc else None,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
