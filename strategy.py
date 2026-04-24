"""
Final submission strategy.

Person 3's ML/ensemble plumbing is complete here. Person 2 only needs to
replace _rule_signal() with the rule-based trend/mean-reversion signal.
Until then, the strategy is intentionally runnable-flat.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from cnlib.base_strategy import BaseStrategy

COINS = ["kapcoin-usd_train", "metucoin-usd_train", "tamcoin-usd_train"]
RESULTS_DIR = Path(__file__).parent / "results"

MIN_FEATURE_ROWS = 55
TARGET_HORIZON = 3
MAX_ACTIVE_COINS = 2
MAX_TOTAL_ALLOCATION = 0.9

ML_CONFIDENCE_THRESHOLD = 0.60
ML_STRONG_THRESHOLD = 0.80
ATR_PERIOD = 14


def _flat(coin: str) -> dict:
    return {"coin": coin, "signal": 0, "allocation": 0.0, "leverage": 1}


class MyStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()
        self._open_signals: dict[str, int] = {coin: 0 for coin in COINS}
        self.models: dict[str, object] = {}
        self.model_feature_names: dict[str, list[str]] = {}
        self.model_metadata: dict[str, dict] = {}
        self._load_models()

    def predict(self, data: dict) -> list[dict]:
        candidates = []
        decisions = {coin: _flat(coin) for coin in COINS}

        for coin in COINS:
            df = data.get(coin)
            if df is None:
                continue

            rule_signal = self._rule_signal(coin, df, data)
            if rule_signal == 0:
                continue

            ml_prob_up = self._ml_prob(coin, data)
            candidate = self._candidate_decision(coin, df, rule_signal, ml_prob_up)
            if candidate is not None:
                candidates.append(candidate)

        candidates.sort(key=lambda item: item["confidence"], reverse=True)
        active = candidates[:MAX_ACTIVE_COINS]
        n_active = len(active)

        for candidate in active:
            strength = 1.0 if candidate["confidence"] >= ML_STRONG_THRESHOLD else 0.5
            decision = candidate["decision"]
            decision["allocation"] = _position_allocation(n_active, strength)
            decisions[decision["coin"]] = decision

        ordered = [decisions[coin] for coin in COINS]
        self._open_signals = {d["coin"]: d["signal"] for d in ordered}
        return ordered

    def _rule_signal(self, coin: str, df: pd.DataFrame, data: dict) -> int:
        """Person 2 hook: return +1 long, -1 short, or 0 flat."""
        if len(df) < MIN_FEATURE_ROWS:
            return 0

        close = df["Close"]
        current_bw = _bb_width(close).iloc[-1]

        if pd.isna(current_bw):
            return 0

        # Trending regime — EMA crossover
        if current_bw > 0.08:
            fast = _ema(close, 20)
            slow = _ema(close, 50)
            if pd.isna(fast.iloc[-1]) or pd.isna(slow.iloc[-1]):
                return 0
            return 1 if fast.iloc[-1] > slow.iloc[-1] else -1

        # Ranging regime — RSI + BB position
        if current_bw < 0.06:
            rsi_val = _rsi(close).iloc[-1]
            bb_pct_val = _bb_pct(close).iloc[-1]
            if pd.isna(rsi_val) or pd.isna(bb_pct_val):
                return 0
            if rsi_val < 35 and bb_pct_val < 0.2:
                return 1
            if rsi_val > 65 and bb_pct_val > 0.8:
                return -1
            return 0

        # Ambiguous regime — stay flat
        return 0

    def _load_models(self) -> None:
        for coin in COINS:
            path = _model_path(coin)
            if not path.exists():
                continue

            try:
                with open(path, "rb") as f:
                    payload = pickle.load(f)
            except (OSError, pickle.PickleError, AttributeError, ImportError, EOFError):
                continue

            if isinstance(payload, dict) and "estimator" in payload:
                self.models[coin] = payload["estimator"]
                self.model_feature_names[coin] = list(payload.get("feature_names") or [])
                self.model_metadata[coin] = {
                    key: value
                    for key, value in payload.items()
                    if key not in {"estimator"}
                }
            else:
                self.models[coin] = payload

    def _ml_prob(self, coin: str, data: dict) -> float | None:
        model = self.models.get(coin)
        if model is None:
            return None

        row = self._ml_feature_row(coin, data)
        if row is None:
            return None

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(row)
            if proba is None or len(proba) == 0:
                return None

            classes = list(getattr(model, "classes_", []))
            if 1 in classes:
                col = classes.index(1)
            elif len(proba[0]) >= 2:
                col = 1
            else:
                return None
            return float(proba[0][col])

        if hasattr(model, "predict"):
            return float(model.predict(row)[0])

        return None

    def _ml_feature_row(self, coin: str, data: dict) -> np.ndarray | None:
        df = data.get(coin)
        if df is None:
            return None

        leader = data["kapcoin-usd_train"]["Close"] if coin != "kapcoin-usd_train" else None
        features = _build_features_single(df, leader_close=leader)
        features = features.replace([np.inf, -np.inf], np.nan)
        if len(features.dropna()) <= MIN_FEATURE_ROWS:
            return None

        names = self.model_feature_names.get(coin) or list(features.columns)
        row = features.iloc[[-1]].reindex(columns=names)
        if row.isna().any(axis=None):
            return None
        return row.to_numpy(dtype=np.float32)

    def _feature_names(self, data: dict, coin: str) -> list[str]:
        df = data[coin]
        leader = data["kapcoin-usd_train"]["Close"] if coin != "kapcoin-usd_train" else None
        return list(_build_features_single(df, leader_close=leader).columns)

    def _candidate_decision(
        self,
        coin: str,
        df: pd.DataFrame,
        rule_signal: int,
        ml_prob_up: float | None,
    ) -> dict | None:
        if rule_signal not in {-1, 1} or ml_prob_up is None:
            return None

        if not 0.0 <= ml_prob_up <= 1.0:
            return None

        ml_signal = 1 if ml_prob_up > 0.5 else -1 if ml_prob_up < 0.5 else 0
        confidence = abs(ml_prob_up - 0.5) * 2.0
        if ml_signal != rule_signal or confidence < ML_CONFIDENCE_THRESHOLD:
            return None

        current_atr = _atr(df, ATR_PERIOD).iloc[-1]
        current_atr_pct = _atr_pct(df, ATR_PERIOD).iloc[-1]
        if pd.isna(current_atr) or pd.isna(current_atr_pct) or current_atr <= 0:
            return None

        entry = float(df["Close"].iloc[-1])
        decision = {
            "coin": coin,
            "signal": rule_signal,
            "allocation": 0.0,
            "leverage": _dynamic_leverage(float(current_atr_pct)),
            "stop_loss": _stop_loss_price(entry, rule_signal, float(current_atr)),
            "take_profit": _take_profit_price(entry, rule_signal, float(current_atr)),
        }
        return {"decision": decision, "confidence": confidence}


def _model_path(coin: str) -> Path:
    return RESULTS_DIR / f"model_{coin.replace('-', '_')}.pkl"


def _ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()


def _sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n).mean()


def _true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["Close"].shift(1)
    ranges = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs(),
    ], axis=1)
    return ranges.max(axis=1)


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return _true_range(df).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def _atr_pct(df: pd.DataFrame, n: int = 14) -> pd.Series:
    close = df["Close"].replace(0, np.nan)
    return _atr(df, n) / close


def _bb_bands(series: pd.Series, n: int = 20, k: float = 2.0):
    mid = _sma(series, n)
    std = series.rolling(n).std(ddof=0)
    upper = mid + k * std
    lower = mid - k * std
    return upper, mid, lower


def _bb_width(series: pd.Series, n: int = 20, k: float = 2.0) -> pd.Series:
    upper, mid, lower = _bb_bands(series, n, k)
    return (upper - lower) / mid.replace(0, np.nan)


def _bb_pct(series: pd.Series, n: int = 20, k: float = 2.0) -> pd.Series:
    upper, _, lower = _bb_bands(series, n, k)
    width = (upper - lower).replace(0, np.nan)
    return (series - lower) / width


def _rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.diff()
    gains = delta.clip(lower=0)
    losses = (-delta).clip(lower=0)
    avg_gain = gains.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    avg_loss = losses.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - (100 / (1 + rs))
    result = result.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    result = result.mask((avg_gain == 0) & (avg_loss > 0), 0.0)
    result = result.mask((avg_gain == 0) & (avg_loss == 0), 50.0)
    return result


def _momentum(series: pd.Series, n: int = 10) -> pd.Series:
    return series.pct_change(n)


def _volume_ratio(df: pd.DataFrame, n: int = 20) -> pd.Series:
    avg_volume = df["Volume"].rolling(n).mean().replace(0, np.nan)
    return df["Volume"] / avg_volume


def _rolling_correlation(a: pd.Series, b: pd.Series, n: int = 30) -> pd.Series:
    return a.pct_change().rolling(n).corr(b.pct_change())


def _lead_lag_signal(leader: pd.Series, follower: pd.Series, lag: int = 1) -> pd.Series:
    return leader.pct_change().shift(lag).reindex(follower.index)


def _build_features_single(
    df: pd.DataFrame,
    leader_close: pd.Series | None = None,
) -> pd.DataFrame:
    close = df["Close"]
    feat = pd.DataFrame(index=df.index)

    feat["ret_1"] = close.pct_change(1)
    feat["ret_3"] = close.pct_change(3)
    feat["ret_5"] = close.pct_change(5)
    feat["ret_10"] = close.pct_change(10)
    feat["ret_20"] = close.pct_change(20)

    ema_20 = _ema(close, 20)
    ema_50 = _ema(close, 50)
    feat["ema_diff_20_50"] = (ema_20 - ema_50) / close
    feat["close_vs_ema_20"] = (close - ema_20) / close
    feat["close_vs_ema_50"] = (close - ema_50) / close

    feat["rsi_14"] = _rsi(close, 14)
    feat["bb_pct_20"] = _bb_pct(close, 20)
    feat["bb_width_20"] = _bb_width(close, 20)
    feat["mom_10"] = _momentum(close, 10)

    feat["atr_pct_14"] = _atr_pct(df, 14)
    feat["vol_ratio_20"] = _volume_ratio(df, 20)

    if leader_close is not None:
        feat["leader_ret_1"] = _lead_lag_signal(leader_close, close, lag=1)
        feat["leader_ret_3"] = _lead_lag_signal(leader_close, close, lag=3)
        feat["leader_corr_30"] = _rolling_correlation(leader_close, close, n=30)

    return feat


def _dynamic_leverage(current_atr_pct: float) -> int:
    if current_atr_pct < 0.03:
        return 5
    if current_atr_pct < 0.06:
        return 3
    if current_atr_pct < 0.10:
        return 2
    return 1


def _stop_loss_price(
    entry: float,
    direction: int,
    current_atr: float,
    atr_multiplier: float = 2.0,
) -> float:
    return entry - direction * current_atr * atr_multiplier


def _take_profit_price(
    entry: float,
    direction: int,
    current_atr: float,
    risk_reward: float = 2.0,
    atr_multiplier: float = 2.0,
) -> float:
    return entry + direction * current_atr * atr_multiplier * risk_reward


def _position_allocation(
    n_active_coins: int,
    signal_strength: float = 1.0,
    max_total: float = MAX_TOTAL_ALLOCATION,
) -> float:
    if n_active_coins <= 0:
        return 0.0
    base = max_total / n_active_coins
    return round(min(base * signal_strength, base), 4)
