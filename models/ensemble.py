from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from models.base_model import BaseSignalModel, RawSignal, bounded_confidence, direction_from_score


class WeightedEnsemble(BaseSignalModel):
    name = "weighted_ensemble"

    def __init__(
        self,
        models: Iterable[BaseSignalModel],
        *,
        weights: Iterable[float] | None = None,
        threshold: float = 0.10,
    ) -> None:
        self.models = list(models)
        self.weights = list(weights) if weights is not None else [1.0] * len(self.models)
        if len(self.models) != len(self.weights):
            raise ValueError("models and weights must have the same length")
        self.threshold = threshold

    def train(self, asset_frames: dict[str, pd.DataFrame]) -> None:
        for model in self.models:
            model.train(asset_frames)

    def predict(self, features: pd.DataFrame, *, asset: str) -> RawSignal:
        if not self.models:
            return RawSignal(asset=asset, score=0.0, confidence=0.0, direction=0)

        weighted_score = 0.0
        total_weight = 0.0
        contributors: dict[str, float] = {}

        for model, weight in zip(self.models, self.weights, strict=True):
            signal = model.predict(features, asset=asset)
            weighted_score += signal.score * weight
            total_weight += abs(weight)
            contributors[model.name] = signal.score

        score = weighted_score / max(total_weight, 1e-12)
        return RawSignal(
            asset=asset,
            score=float(score),
            confidence=bounded_confidence(float(score), scale=0.75),
            direction=direction_from_score(float(score), self.threshold),
            metadata={"model": self.name, "contributors": contributors},
        )
