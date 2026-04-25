"""
Final submission strategy.

Loads tracked ML model bundles from models/ and combines them with the
rule-based trend/mean-reversion signal.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from cnlib.base_strategy import BaseStrategy
from research.ensemble_model import EnsembleModel as _EnsembleModel  # noqa: F401 — pickle deserialization
from research.features import (
    atr as _atr,
    atr_pct as _atr_pct,
    bb_width as _bb_width,
    bb_pct as _bb_pct,
    ema as _ema,
    rsi as _rsi,
    volume_ratio as _volume_ratio,
)
from research.ml_features import build_features_single as _build_features_single
from research.risk import (
    dynamic_leverage as _dynamic_leverage,
    stop_loss_price as _stop_loss_price,
    take_profit_price as _take_profit_price,
    position_allocation as _position_allocation,
)

COINS = ["kapcoin-usd_train", "metucoin-usd_train", "tamcoin-usd_train"]
MODEL_DIR = Path(__file__).parent / "models"

MIN_FEATURE_ROWS = 55
TARGET_HORIZON = 3
MAX_ACTIVE_COINS = 2
MAX_TOTAL_ALLOCATION = 0.9
ATR_PERIOD = 14



def _flat(coin: str) -> dict:
    return {"coin": coin, "signal": 0, "allocation": 0.0, "leverage": 1}


class MyStrategy(BaseStrategy):
    # Lowered from 0.55 → 0.52: year-5 distributional shift will compress
    # model confidence; a 0.55 gate would filter out too many valid signals.
    ML_CONFIDENCE_THRESHOLD: float = 0.52
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
            try:
                candidate = self._candidate_decision(coin, df, rule_signal, ml_prob_up)
            except Exception as exc:
                print(f"[MyStrategy] WARNING: _candidate_decision failed for {coin}: {type(exc).__name__}: {exc}")
                candidate = None
            if candidate is not None:
                candidates.append(candidate)

        candidates.sort(key=lambda item: item["confidence"], reverse=True)
        active = candidates[:MAX_ACTIVE_COINS]
        n_active = len(active)

        if n_active == 1:
            active[0]["decision"]["allocation"] = MAX_TOTAL_ALLOCATION
            decisions[active[0]["decision"]["coin"]] = active[0]["decision"]
        elif n_active > 1:
            total_conf = sum(c["confidence"] for c in active)
            for candidate in active:
                weight = candidate["confidence"] / total_conf
                candidate["decision"]["allocation"] = round(
                    min(weight * MAX_TOTAL_ALLOCATION, MAX_TOTAL_ALLOCATION), 4
                )
                decisions[candidate["decision"]["coin"]] = candidate["decision"]

        # Order closes before opens so the engine frees cash before
        # attempting new entries (prevents failed opens from cash timing).
        ordered = [decisions[coin] for coin in COINS]
        ordered.sort(key=lambda d: (d["signal"] != 0, d["coin"]))
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

        # Trending regime — EMA position + RSI momentum confirmation
        if current_bw > 0.08:
            fast = _ema(close, 20)
            slow = _ema(close, 50)
            if pd.isna(fast.iloc[-1]) or pd.isna(slow.iloc[-1]):
                return 0
            ema_signal = 1 if fast.iloc[-1] > slow.iloc[-1] else -1

            # RSI momentum must agree with trend direction
            rsi_val = _rsi(close).iloc[-1]
            if not pd.isna(rsi_val):
                if ema_signal == 1 and rsi_val < 50:
                    return 0  # EMA bullish but RSI below midpoint — weakening trend
                if ema_signal == -1 and rsi_val > 50:
                    return 0  # EMA bearish but RSI above midpoint — weakening trend

            # Skip very low-volume candles (likely false breakouts)
            vol_r = _volume_ratio(df, 20).iloc[-1]
            if not pd.isna(vol_r) and vol_r < 0.5:
                return 0

            return ema_signal

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

        if ml_signal != rule_signal:
            # ML disagrees on direction — always exit or skip entry
            return None

        # Holding an existing position in the same direction: relax confidence gate
        # (don't prematurely close a trade just because ML became uncertain)
        # New entries still require full confidence threshold
        is_hold = (self._open_signals.get(coin, 0) == rule_signal)
        if not is_hold and confidence < self.__class__.ML_CONFIDENCE_THRESHOLD:
            return None

        current_atr = _atr(df, ATR_PERIOD).iloc[-1]
        current_atr_pct = _atr_pct(df, ATR_PERIOD).iloc[-1]
        if pd.isna(current_atr) or pd.isna(current_atr_pct) or current_atr <= 0:
            return None

        # Regime-aware risk/reward: let winners run in trends, take quick profits in ranges
        close = df["Close"]
        current_bw = _bb_width(close).iloc[-1]
        if not pd.isna(current_bw) and current_bw > 0.08:
            risk_reward = 3.0  # trending — hold for larger moves
        else:
            risk_reward = 1.5  # ranging — mean-revert targets are smaller

        entry = float(df["Close"].iloc[-1])
        if entry <= 0:
            return None

        lev = _dynamic_leverage(float(current_atr_pct))
        stop_loss = _stop_loss_price(entry, rule_signal, float(current_atr), leverage=lev)
        take_profit = _take_profit_price(entry, rule_signal, float(current_atr), risk_reward=risk_reward)

        # Sanity check: both prices must be positive (ATR too large relative to entry otherwise)
        if stop_loss <= 0 or take_profit <= 0:
            return None

        decision = {
            "coin": coin,
            "signal": rule_signal,
            "allocation": 0.0,
            "leverage": lev,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
        }
        return {"decision": decision, "confidence": confidence}


def _model_path(coin: str) -> Path:
    return MODEL_DIR / f"model_{coin.replace('-', '_')}.pkl"
