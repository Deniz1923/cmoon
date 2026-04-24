"""
feature_engineering.py — Dev A
Build leakage-free features from normalized OHLCV data.

Hard rule: every rolling calculation ends with .shift(1) so that the feature
at time t uses only data strictly before t.
Scalers are fit on train data only and returned as a FeatureTransformer object.
"""
import pickle
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from src.utils import atr as _atr


@dataclass
class FeatureTransformer:
    scaler: RobustScaler = field(default_factory=RobustScaler)
    feature_cols: list = field(default_factory=list)
    fitted: bool = False


def build_features(ohlcv_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Return a DataFrame of feat_* columns aligned to ohlcv_df's index."""
    df = ohlcv_df.copy()
    feat_cfg = config.get("features", {})
    features: dict = {}

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # ---- Price ----
    if "price" in feat_cfg:
        price_feats = feat_cfg["price"]
        if "returns" in price_feats:
            features["feat_ret_1"] = close.pct_change().shift(1)
            features["feat_ret_4"] = close.pct_change(4).shift(1)
            features["feat_ret_24"] = close.pct_change(24).shift(1)
        if "log_returns" in price_feats:
            log_r = np.log(close / close.shift(1))
            features["feat_log_ret_1"] = log_r.shift(1)
            features["feat_log_ret_24"] = log_r.rolling(24).sum().shift(1)

    # ---- Trend ----
    if "trend" in feat_cfg:
        trend_feats = feat_cfg["trend"]
        if "ema" in trend_feats:
            ema9 = close.ewm(span=9, adjust=False).mean()
            ema21 = close.ewm(span=21, adjust=False).mean()
            ema50 = close.ewm(span=50, adjust=False).mean()
            features["feat_ema_9_21_ratio"] = (ema9 / ema21 - 1).shift(1)
            features["feat_ema_21_50_ratio"] = (ema21 / ema50 - 1).shift(1)
            features["feat_price_ema21_dist"] = (close / ema21 - 1).shift(1)
        if "macd" in trend_feats:
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            features["feat_macd_norm"] = (macd_line / close).shift(1)
            features["feat_macd_hist_norm"] = ((macd_line - signal_line) / close).shift(1)

    # ---- Momentum ----
    if "momentum" in feat_cfg:
        mom_feats = feat_cfg["momentum"]
        if "rsi" in mom_feats:
            features["feat_rsi_14"] = _rsi(close, 14).shift(1) / 100.0
            features["feat_rsi_7"] = _rsi(close, 7).shift(1) / 100.0
        if "stochastic" in mom_feats:
            stoch_k, stoch_d = _stochastic(high, low, close, 14, 3)
            features["feat_stoch_k"] = stoch_k.shift(1) / 100.0
            features["feat_stoch_kd_diff"] = ((stoch_k - stoch_d) / 100.0).shift(1)

    # ---- Volatility ----
    if "volatility" in feat_cfg:
        vol_feats = feat_cfg["volatility"]
        if "atr" in vol_feats:
            atr = _atr(high, low, close, 14)
            features["feat_atr_norm"] = (atr / close).shift(1)
        if "realized_volatility" in vol_feats:
            log_r = np.log(close / close.shift(1))
            rv24 = log_r.rolling(24).std()
            rv168 = log_r.rolling(168).std()
            features["feat_rvol_24h"] = rv24.shift(1)
            features["feat_rvol_168h"] = rv168.shift(1)
            features["feat_rvol_ratio"] = (rv24 / rv168.replace(0, np.nan)).shift(1)
        if "bollinger_width" in vol_feats:
            bb_mid = close.rolling(20).mean()
            bb_std = close.rolling(20).std()
            features["feat_bb_width"] = (2 * bb_std / bb_mid).shift(1)
            features["feat_bb_position"] = (
                (close - bb_mid) / (2 * bb_std + 1e-9)
            ).shift(1)

    # ---- Volume ----
    if "volume" in feat_cfg:
        vol_feats = feat_cfg["volume"]
        vol_mean = volume.rolling(24).mean()
        vol_std = volume.rolling(24).std()
        features["feat_vol_zscore"] = ((volume - vol_mean) / (vol_std + 1e-9)).shift(1)
        if "obv" in vol_feats:
            obv = _obv(close, volume)
            obv_ema = obv.ewm(span=20, adjust=False).mean()
            features["feat_obv_trend"] = (obv / (obv_ema + 1e-9) - 1).shift(1)
        if "vwap_distance" in vol_feats:
            vwap = _rolling_vwap(close, high, low, volume, 24)
            features["feat_vwap_dist"] = (close / (vwap + 1e-9) - 1).shift(1)

    # ---- Futures-only ----
    if "futures_optional" in feat_cfg:
        futures_feats = feat_cfg["futures_optional"]
        if "funding_rate_zscore" in futures_feats and "funding_rate" in df.columns:
            fr = df["funding_rate"]
            fr_mean = fr.rolling(168).mean()
            fr_std = fr.rolling(168).std()
            features["feat_funding_zscore"] = ((fr - fr_mean) / (fr_std + 1e-9)).shift(1)
        if "open_interest_change" in futures_feats and "open_interest" in df.columns:
            features["feat_oi_change"] = df["open_interest"].pct_change().shift(1)

    return pd.DataFrame(features, index=df.index)


def fit_transformer(train_features: pd.DataFrame) -> FeatureTransformer:
    """Fit RobustScaler on train features only. Call once, reuse for val/holdout."""
    transformer = FeatureTransformer()
    transformer.feature_cols = train_features.columns.tolist()
    clean = train_features.dropna()
    transformer.scaler.fit(clean)
    transformer.fitted = True
    return transformer


def transform(features_df: pd.DataFrame, transformer: FeatureTransformer) -> pd.DataFrame:
    """Apply a fitted transformer to any split."""
    assert transformer.fitted, "Transformer must be fit before calling transform."
    cols = [c for c in transformer.feature_cols if c in features_df.columns]
    out = features_df.copy()
    mask = ~features_df[cols].isna().any(axis=1)
    out.loc[mask, cols] = transformer.scaler.transform(features_df.loc[mask, cols])
    return out


def save_transformer(transformer: FeatureTransformer, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(transformer, f)


def load_transformer(path: str) -> FeatureTransformer:
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    return 100 - (100 / (1 + gain / (loss + 1e-9)))


def _stochastic(high, low, close, k_period=14, d_period=3):
    low_k = low.rolling(k_period).min()
    high_k = high.rolling(k_period).max()
    k = 100 * (close - low_k) / (high_k - low_k + 1e-9)
    return k, k.rolling(d_period).mean()


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    return (np.sign(close.diff()).fillna(0) * volume).cumsum()


def _rolling_vwap(close, high, low, volume, window=24) -> pd.Series:
    tp = (high + low + close) / 3
    return (tp * volume).rolling(window).sum() / (volume.rolling(window).sum() + 1e-9)