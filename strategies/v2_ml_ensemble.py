from __future__ import annotations

from models.ensemble import WeightedEnsemble
from models.mean_reversion import MeanReversionModel
from models.ml_model import NumpyRidgeModel
from models.trend_following import TrendFollowingModel
from strategies.v3_final import CompetitionStrategy


class Strategy(CompetitionStrategy):
    def __init__(self) -> None:
        super().__init__(
            model=WeightedEnsemble(
                [TrendFollowingModel(), MeanReversionModel(), NumpyRidgeModel()],
                weights=[0.45, 0.25, 0.30],
                threshold=0.08,
            )
        )
