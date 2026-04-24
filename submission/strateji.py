from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


ASSETS = ("Varlik_A", "Varlik_B", "Varlik_C")
WARMUP_BARS = 20
FEATURE_COLUMNS = (
    "ret_1",
    "log_ret_1",
    "range_pct",
    "body_pct",
    "volume_change",
    "volatility_5",
    "volatility_20",
    "momentum_3",
    "momentum_10",
    "momentum_20",
    "sma_ratio_5_20",
    "sma_ratio_20_50",
    "ema_ratio_12_26",
    "rsi_14",
    "atr_14",
    "bb_z_20",
    "volume_z_20",
)


class Strategy:
    def fit(self, data):
        return None

    def predict(self, data):
        asset_frames = _split_assets(data)
        decisions = []
        for asset in ASSETS:
            frame = asset_frames.get(asset)
            if frame is None or len(frame) < WARMUP_BARS:
                decisions.append({"sinyal": 0, "oran": 0.0, "kaldirac": 2})
                continue

            features = _build_features(frame)
            row = features.iloc[-1] if not features.empty else pd.Series(0.0, index=FEATURE_COLUMNS)
            score, confidence = _score(row)
            direction = 1 if score > 0.09 else -1 if score < -0.09 else 0
            volatility = float(row.get("volatility_20", 1.0))
            decisions.append(_size(direction, confidence, volatility))

        return _cap_total(decisions)


def fit(data):
    _DEFAULT.fit(data)


def predict(data):
    return _DEFAULT.predict(data)


def _split_assets(data):
    if isinstance(data, Mapping):
        return {str(asset): frame.copy() for asset, frame in data.items()}

    if isinstance(data.columns, pd.MultiIndex):
        return {str(asset): data[asset].copy() for asset in data.columns.get_level_values(0).unique()}

    for column in ("asset", "symbol", "coin", "varlik", "Varlik"):
        if column in data.columns:
            return {str(asset): part.drop(columns=[column]).copy() for asset, part in data.groupby(column)}

    return {"Varlik_A": data.copy()}


def _build_features(frame):
    data = _normalize_ohlcv(frame)
    close = _nonzero(data["close"])
    high = data["high"]
    low = data["low"]
    open_ = _nonzero(data["open"])
    volume = data["volume"]

    features = pd.DataFrame(index=data.index)
    features["ret_1"] = close.pct_change()
    features["log_ret_1"] = np.log(close).diff()
    features["range_pct"] = (high - low) / close
    features["body_pct"] = (close - open_) / open_
    features["volume_change"] = volume.pct_change()
    features["volatility_5"] = features["log_ret_1"].rolling(5, min_periods=2).std()
    features["volatility_20"] = features["log_ret_1"].rolling(20, min_periods=5).std()
    features["momentum_3"] = close.pct_change(3)
    features["momentum_10"] = close.pct_change(10)
    features["momentum_20"] = close.pct_change(20)

    sma_5 = close.rolling(5, min_periods=1).mean()
    sma_20 = close.rolling(20, min_periods=1).mean()
    sma_50 = close.rolling(50, min_periods=1).mean()
    features["sma_ratio_5_20"] = sma_5 / _nonzero(sma_20) - 1.0
    features["sma_ratio_20_50"] = sma_20 / _nonzero(sma_50) - 1.0

    ema_12 = close.ewm(span=12, adjust=False, min_periods=1).mean()
    ema_26 = close.ewm(span=26, adjust=False, min_periods=1).mean()
    features["ema_ratio_12_26"] = ema_12 / _nonzero(ema_26) - 1.0
    features["rsi_14"] = _rsi(close, 14)
    features["atr_14"] = _atr(data, 14) / close
    features["bb_z_20"] = _zscore(close, 20)
    features["volume_z_20"] = _zscore(volume, 20)

    return features.reindex(columns=FEATURE_COLUMNS).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _normalize_ohlcv(frame):
    aliases = {
        "open": ("open", "Open", "OPEN", "acilis", "Acilis"),
        "high": ("high", "High", "HIGH", "yuksek", "Yuksek"),
        "low": ("low", "Low", "LOW", "dusuk", "Dusuk"),
        "close": ("close", "Close", "CLOSE", "kapanis", "Kapanis"),
        "volume": ("volume", "Volume", "VOLUME", "hacim", "Hacim"),
    }
    normalized = pd.DataFrame(index=frame.index.copy())
    for target, names in aliases.items():
        source = next((column for column in names if column in frame.columns), None)
        if source is None:
            normalized[target] = 0.0 if target == "volume" else np.nan
        else:
            normalized[target] = pd.to_numeric(frame[source], errors="coerce")
    return normalized.sort_index()


def _score(row):
    trend = (
        0.38 * np.tanh(row.get("sma_ratio_20_50", 0.0) * 18.0)
        + 0.30 * np.tanh(row.get("ema_ratio_12_26", 0.0) * 22.0)
        + 0.22 * np.tanh(row.get("momentum_20", 0.0) * 8.0)
        + 0.10 * np.tanh(row.get("momentum_10", 0.0) * 10.0)
    )
    bb_z = float(row.get("bb_z_20", 0.0))
    rsi = float(row.get("rsi_14", 50.0))
    short_momentum = float(row.get("momentum_3", 0.0))
    mean_reversion = (
        0.55 * -np.tanh(bb_z / 2.2)
        + 0.30 * np.clip((50.0 - rsi) / 35.0, -1.0, 1.0)
        - 0.15 * np.tanh(short_momentum * 25.0)
    )
    score = 0.62 * trend + 0.38 * mean_reversion
    return float(score), float(min(1.0, abs(score) / 0.75))


def _size(direction, confidence, volatility):
    confidence = min(max(float(confidence), 0.0), 1.0)
    volatility = max(float(volatility), 1e-6)
    if direction == 0 or confidence < 0.12:
        return {"sinyal": 0, "oran": 0.0, "kaldirac": 2}

    if confidence >= 0.90 and volatility <= 0.012:
        leverage = 10
    elif confidence >= 0.74 and volatility <= 0.020:
        leverage = 5
    elif confidence >= 0.52 and volatility <= 0.035:
        leverage = 3
    else:
        leverage = 2

    vol_haircut = min(1.25, max(0.20, 0.018 / volatility))
    allocation = min(0.42, max(0.0, 0.42 * (confidence**1.7) * vol_haircut))
    return {"sinyal": int(direction), "oran": float(round(allocation, 6)), "kaldirac": int(leverage)}


def _cap_total(decisions):
    total = sum(float(decision["oran"]) for decision in decisions)
    if total <= 1.0 or total <= 0.0:
        return decisions

    scale = 1.0 / total
    capped = []
    for decision in decisions:
        adjusted = dict(decision)
        adjusted["oran"] = float(round(float(adjusted["oran"]) * scale, 6))
        capped.append(adjusted)
    return capped


def _nonzero(series):
    return series.replace(0.0, np.nan).ffill().fillna(1.0)


def _rsi(close, window):
    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.ewm(alpha=1 / window, adjust=False, min_periods=1).mean()
    avg_loss = losses.ewm(alpha=1 / window, adjust=False, min_periods=1).mean()
    rs = avg_gain / _nonzero(avg_loss)
    return 100.0 - (100.0 / (1.0 + rs))


def _atr(data, window):
    previous_close = data["close"].shift(1)
    true_range = pd.concat(
        [
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / window, adjust=False, min_periods=1).mean()


def _zscore(series, window):
    min_periods = max(2, window // 4)
    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std()
    return (series - mean) / _nonzero(std)


_DEFAULT = Strategy()
