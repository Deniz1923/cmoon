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
from research.features import (
    atr as _atr,
    atr_pct as _atr_pct,
    bb_width as _bb_width,
    bb_pct as _bb_pct,
    ema as _ema,
    rsi as _rsi,
)
from research.ml_features import build_features_single as _build_features_single
from research.risk import (
    dynamic_leverage as _dynamic_leverage,
    stop_loss_price as _stop_loss_price,
    take_profit_price as _take_profit_price,
    position_allocation as _position_allocation,
)

COINS = ["kapcoin-usd_train", "metucoin-usd_train", "tamcoin-usd_train"]
RESULTS_DIR = Path(__file__).parent / "results"

MIN_FEATURE_ROWS = 55
TARGET_HORIZON = 3
MAX_ACTIVE_COINS = 2
MAX_TOTAL_ALLOCATION = 0.9
ATR_PERIOD = 14


def _flat(coin: str) -> dict:
    return {"coin": coin, "signal": 0, "allocation": 0.0, "leverage": 1}


class MyStrategy(BaseStrategy):
    # Confidence threshold: 0.55 selected via 4-fold walk-forward CV on
    # training window (candles 0-1099). Best avg return (+39.6%) with
    # lowest coefficient of variation and 21.5 trades/fold (not too sparse).
    ML_CONFIDENCE_THRESHOLD: float = 0.55
    ML_STRONG_THRESHOLD: float = 0.80

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
            strength = 1.0 if candidate["confidence"] >= self.__class__.ML_STRONG_THRESHOLD else 0.5
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
            except (OSError, pickle.PickleError, AttributeError, ImportError, EOFError) as exc:
                print(f"[MyStrategy] WARNING: failed to load model for {coin}: {type(exc).__name__}: {exc}")
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
        if ml_signal != rule_signal or confidence < self.__class__.ML_CONFIDENCE_THRESHOLD:
            return None

        current_atr = _atr(df, ATR_PERIOD).iloc[-1]
        current_atr_pct = _atr_pct(df, ATR_PERIOD).iloc[-1]
        if pd.isna(current_atr) or pd.isna(current_atr_pct) or current_atr <= 0:
            return None

        entry = float(df["Close"].iloc[-1])
        lev = _dynamic_leverage(float(current_atr_pct))
        decision = {
            "coin": coin,
            "signal": rule_signal,
            "allocation": 0.0,
            "leverage": lev,
            "stop_loss": _stop_loss_price(entry, rule_signal, float(current_atr), leverage=lev),
            "take_profit": _take_profit_price(entry, rule_signal, float(current_atr)),
        }
        return {"decision": decision, "confidence": confidence}


def _model_path(coin: str) -> Path:
    return RESULTS_DIR / f"model_{coin.replace('-', '_')}.pkl"
