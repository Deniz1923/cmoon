from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Sequence

import pandas as pd

from model_types import Signal, StrategyState

if TYPE_CHECKING:
    from competition import CompetitionRules


class BaseStrategy(ABC):
    """
    Yarismacinin dolduracagi taban strateji sinifi.
    """

    name: str = "base-strategy"

    def __init__(self) -> None:
        self.rules: CompetitionRules | None = None

    def set_rules(self, rules: CompetitionRules) -> None:
        self.rules = rules

    def fit(self, train_data: dict[str, pd.DataFrame]) -> None:
        return None

    @abstractmethod
    def predict(
        self,
        current_window: dict[str, pd.DataFrame],
        state: StrategyState,
    ) -> Sequence[Signal | dict]:
        raise NotImplementedError
