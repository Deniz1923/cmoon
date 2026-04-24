from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


OHLCV_ALIASES: dict[str, tuple[str, ...]] = {
    "open": ("open", "Open", "OPEN", "acilis", "Acilis"),
    "high": ("high", "High", "HIGH", "yuksek", "Yuksek"),
    "low": ("low", "Low", "LOW", "dusuk", "Dusuk"),
    "close": ("close", "Close", "CLOSE", "kapanis", "Kapanis"),
    "volume": ("volume", "Volume", "VOLUME", "hacim", "Hacim"),
}

ASSET_COLUMNS = ("asset", "symbol", "coin", "varlik", "Varlik")

FEATURE_COLUMNS: tuple[str, ...] = (
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


def split_assets(data: pd.DataFrame | Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Accept organizer-like data shapes and return one frame per asset."""
    if isinstance(data, Mapping):
        return {str(asset): frame.copy() for asset, frame in data.items()}

    if isinstance(data.columns, pd.MultiIndex):
        assets: dict[str, pd.DataFrame] = {}
        for asset in data.columns.get_level_values(0).unique():
            assets[str(asset)] = data[asset].copy()
        return assets

    for column in ASSET_COLUMNS:
        if column in data.columns:
            return {str(asset): part.drop(columns=[column]).copy() for asset, part in data.groupby(column)}

    return {"Varlik_A": data.copy()}


def normalize_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize common OHLCV column variants to open/high/low/close/volume."""
    normalized = pd.DataFrame(index=frame.index.copy())

    for target, aliases in OHLCV_ALIASES.items():
        source = next((column for column in aliases if column in frame.columns), None)
        if source is None:
            if target == "volume":
                normalized[target] = 0.0
                continue
            raise ValueError(f"Missing required OHLC column for {target!r}: {list(frame.columns)}")
        normalized[target] = pd.to_numeric(frame[source], errors="coerce")

    return normalized.sort_index()


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build causal features. Every row uses only current and prior candles."""
    data = normalize_ohlcv(frame)

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

    for window in (3, 10, 20):
        features[f"momentum_{window}"] = close.pct_change(window)

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

    features = features.replace([np.inf, -np.inf], np.nan)
    return features.reindex(columns=FEATURE_COLUMNS).fillna(0.0)


def latest_features(frame: pd.DataFrame) -> pd.Series:
    features = build_features(frame)
    if features.empty:
        return pd.Series(0.0, index=FEATURE_COLUMNS)
    return features.iloc[-1].reindex(FEATURE_COLUMNS).fillna(0.0)


def make_supervised_frame(
    frame: pd.DataFrame,
    *,
    horizon: int = 1,
    min_abs_return: float = 0.0,
) -> pd.DataFrame:
    """Attach future-return labels for research and training only."""
    data = normalize_ohlcv(frame)
    features = build_features(data)
    future_return = data["close"].pct_change(horizon).shift(-horizon)

    supervised = features.copy()
    supervised["future_return"] = future_return
    supervised["target"] = np.select(
        [future_return > min_abs_return, future_return < -min_abs_return],
        [1.0, -1.0],
        default=0.0,
    )
    return supervised.dropna(subset=["future_return"])


def _nonzero(series: pd.Series) -> pd.Series:
    return series.replace(0.0, np.nan).ffill().fillna(1.0)


def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)

    avg_gain = gains.ewm(alpha=1 / window, adjust=False, min_periods=1).mean()
    avg_loss = losses.ewm(alpha=1 / window, adjust=False, min_periods=1).mean()
    rs = avg_gain / _nonzero(avg_loss)
    return 100.0 - (100.0 / (1.0 + rs))


def _atr(data: pd.DataFrame, window: int) -> pd.Series:
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


def _zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=max(2, window // 4)).mean()
    std = series.rolling(window, min_periods=max(2, window // 4)).std()
    return (series - mean) / _nonzero(std)
