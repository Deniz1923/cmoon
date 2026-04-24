"""
Feature matrix builder for ML models — Person 3's primary deliverable.

build_X_y() is the main entry point: it takes full coin_data and returns
(X, y, dates) ready for sklearn/XGBoost training.

Keep this file free of model code — just feature engineering.
"""
import numpy as np
import pandas as pd

from research.features import (
    ema, atr, atr_pct, rsi, bb_pct, bb_width,
    momentum, volume_ratio, lead_lag_signal, rolling_correlation, COINS,
)

# How many candles ahead to predict (target horizon)
# TODO: experiment with 1, 3, 5
TARGET_HORIZON = 3

# Minimum rows needed before the first valid feature row
MIN_ROWS = 55


def build_features_single(df: pd.DataFrame, leader_close: pd.Series | None = None) -> pd.DataFrame:
    """
    Build a feature DataFrame for one coin.

    leader_close: optional Close series of another coin to use as a lead-lag feature
                  (e.g. pass kapcoin Close when computing features for metucoin)

    Returns a DataFrame aligned with df's index. Rows with NaN are included —
    caller should drop them after aligning across all coins.

    TODO: add / remove features based on importance scores from train_models.py.
          Fewer, higher-quality features beat many weak ones.
    """
    close = df["Close"]
    feat = pd.DataFrame(index=df.index)

    # --- Returns ---
    # TODO: add these
    feat["ret_1"]  = close.pct_change(1)
    feat["ret_3"]  = close.pct_change(3)
    feat["ret_5"]  = close.pct_change(5)
    feat["ret_20"] = close.pct_change(20)

    # --- Trend ---
    # TODO: implement ema() in features.py first, then uncomment
    # feat["ema_diff_20_50"] = (ema(close, 20) - ema(close, 50)) / close

    # --- Momentum / oscillators ---
    # TODO: implement rsi(), bb_pct(), bb_width() in features.py first
    # feat["rsi_14"]    = rsi(close, 14)
    # feat["bb_pct_20"] = bb_pct(close, 20)
    # feat["bb_width"]  = bb_width(close, 20)
    # feat["mom_10"]    = momentum(close, 10)

    # --- Volatility ---
    # TODO: implement atr_pct() in features.py first
    # feat["atr_pct_14"] = atr_pct(df, 14)
    # feat["vol_ratio"]  = volume_ratio(df, 20)

    # --- Cross-coin (lead-lag) ---
    # TODO: implement lead_lag_signal() in features.py first
    # if leader_close is not None:
    #     feat["leader_ret_1"] = lead_lag_signal(leader_close, close, lag=1)
    #     feat["leader_ret_3"] = lead_lag_signal(leader_close, close, lag=3)

    return feat


def build_X_y(
    coin_data: dict[str, pd.DataFrame],
    target_coin: str,
    horizon: int = TARGET_HORIZON,
) -> tuple[np.ndarray, np.ndarray, pd.Index]:
    """
    Build (X, y, valid_index) for a single coin's model.

    target_coin: which coin to predict
    X: 2D float array, shape (n_samples, n_features)
    y: 1D int array, 1 if price rises over horizon candles, 0 if falls
    valid_index: original DataFrame index for the valid rows (for time-series split)

    TODO: consider a 3-class target (-1, 0, +1) with a dead-zone around 0
          to avoid trading on ambiguous signals.
    """
    df = coin_data[target_coin]

    # Use kapcoin as leader for non-kap coins (BTC leads alts historically)
    leader = coin_data["kapcoin-usd_train"]["Close"] if target_coin != "kapcoin-usd_train" else None

    feat = build_features_single(df, leader_close=leader)

    # Target: did price go up over the next `horizon` candles?
    future_return = df["Close"].pct_change(horizon).shift(-horizon)
    # TODO: consider a threshold (e.g. > 0.01) to filter noise
    y_raw = (future_return > 0).astype(int)

    # Align and drop NaN rows
    combined = pd.concat([feat, y_raw.rename("target")], axis=1)
    combined = combined.dropna()

    # Don't include the last `horizon` rows (no target available)
    combined = combined.iloc[:-horizon]

    X = combined.drop(columns=["target"]).values.astype(np.float32)
    y = combined["target"].values.astype(int)
    valid_index = combined.index

    return X, y, valid_index


def feature_names(coin_data: dict, target_coin: str) -> list[str]:
    """Returns column names in the same order as build_X_y() — useful for importance plots."""
    df = coin_data[target_coin]
    leader = coin_data["kapcoin-usd_train"]["Close"] if target_coin != "kapcoin-usd_train" else None
    feat = build_features_single(df, leader_close=leader)
    return list(feat.columns)
