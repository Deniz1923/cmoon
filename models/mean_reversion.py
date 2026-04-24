from __future__ import annotations

import numpy as np
import pandas as pd

from models.base_model import BaseSignalModel, RawSignal, bounded_confidence, direction_from_score


class MeanReversionModel(BaseSignalModel):
    name = "mean_reversion"

    def __init__(self, threshold: float = 0.12) -> None:
        self.threshold = threshold

    def predict(self, features: pd.DataFrame, *, asset: str) -> RawSignal:
        if features.empty:
            return RawSignal(asset=asset, score=0.0, confidence=0.0, direction=0)

        row = features.iloc[-1]
        bb_z = float(row.get("bb_z_20", 0.0))
        rsi = float(row.get("rsi_14", 50.0))
        short_momentum = float(row.get("momentum_3", 0.0))

        stretch_score = -np.tanh(bb_z / 2.2)
        rsi_score = np.clip((50.0 - rsi) / 35.0, -1.0, 1.0)
        snapback_penalty = -0.15 * np.tanh(short_momentum * 25.0)
        score = 0.55 * stretch_score + 0.30 * rsi_score + snapback_penalty

        return RawSignal(
            asset=asset,
            score=float(score),
            confidence=bounded_confidence(float(score), scale=0.85),
            direction=direction_from_score(float(score), self.threshold),
            metadata={"model": self.name},
        )
