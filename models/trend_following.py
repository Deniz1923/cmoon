from __future__ import annotations

import numpy as np
import pandas as pd

from models.base_model import BaseSignalModel, RawSignal, bounded_confidence, direction_from_score


class TrendFollowingModel(BaseSignalModel):
    name = "trend_following"

    def __init__(self, threshold: float = 0.08) -> None:
        self.threshold = threshold

    def predict(self, features: pd.DataFrame, *, asset: str) -> RawSignal:
        if features.empty:
            return RawSignal(asset=asset, score=0.0, confidence=0.0, direction=0)

        row = features.iloc[-1]
        score = (
            0.38 * np.tanh(row.get("sma_ratio_20_50", 0.0) * 18.0)
            + 0.30 * np.tanh(row.get("ema_ratio_12_26", 0.0) * 22.0)
            + 0.22 * np.tanh(row.get("momentum_20", 0.0) * 8.0)
            + 0.10 * np.tanh(row.get("momentum_10", 0.0) * 10.0)
        )
        return RawSignal(
            asset=asset,
            score=float(score),
            confidence=bounded_confidence(float(score), scale=0.75),
            direction=direction_from_score(float(score), self.threshold),
            metadata={"model": self.name},
        )
