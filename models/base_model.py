from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class RawSignal:
    asset: str
    score: float
    confidence: float
    direction: int
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseSignalModel(ABC):
    name = "base"

    def train(self, asset_frames: dict[str, pd.DataFrame]) -> None:
        """Optional hook. Rule-based models can stay stateless."""

    @abstractmethod
    def predict(self, features: pd.DataFrame, *, asset: str) -> RawSignal:
        """Return a signed score from a causal feature window."""


def direction_from_score(score: float, threshold: float) -> int:
    if score > threshold:
        return 1
    if score < -threshold:
        return -1
    return 0


def bounded_confidence(score: float, scale: float = 1.0) -> float:
    return float(min(1.0, abs(score) / max(scale, 1e-12)))
