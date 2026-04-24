from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from config import DEFAULT_CONFIG
from features.engineer import build_features, split_assets
from models.base_model import BaseSignalModel
from models.ensemble import WeightedEnsemble
from models.mean_reversion import MeanReversionModel
from models.trend_following import TrendFollowingModel
from sizing.sizer import PositionDecision, RiskState, VolatilityAwareSizer, cap_total_exposure


class CompetitionStrategy:
    def __init__(
        self,
        *,
        model: BaseSignalModel | None = None,
        sizer: VolatilityAwareSizer | None = None,
        warmup_bars: int = DEFAULT_CONFIG.warmup_bars,
    ) -> None:
        self.model = model or WeightedEnsemble(
            [TrendFollowingModel(), MeanReversionModel()],
            weights=[0.62, 0.38],
            threshold=0.09,
        )
        self.sizer = sizer or VolatilityAwareSizer()
        self.risk_state = RiskState()
        self.assets = DEFAULT_CONFIG.assets
        self.warmup_bars = warmup_bars

    def fit(self, data: pd.DataFrame | Mapping[str, pd.DataFrame]) -> None:
        asset_frames = split_assets(data)
        self.model.train(asset_frames)

    def predict(self, data: pd.DataFrame | Mapping[str, pd.DataFrame]) -> list[dict[str, float | int]]:
        asset_frames = split_assets(data)
        decisions: list[PositionDecision] = []

        for asset in self.assets:
            frame = asset_frames.get(asset)
            if frame is None or len(frame) < self.warmup_bars:
                decisions.append(PositionDecision(asset, 0, 0.0, 2, 0.0))
                continue

            features = build_features(frame)
            signal = self.model.predict(features, asset=asset)
            volatility = float(features["volatility_20"].iloc[-1]) if not features.empty else 1.0
            decisions.append(
                self.sizer.size(signal, volatility=volatility, risk_state=self.risk_state)
            )

        decisions = cap_total_exposure(decisions, cap=1.0)
        return [decision.to_order_dict() for decision in decisions]


Strategy = CompetitionStrategy

_DEFAULT_STRATEGY = CompetitionStrategy()


def fit(data: pd.DataFrame | Mapping[str, pd.DataFrame]) -> None:
    _DEFAULT_STRATEGY.fit(data)


def predict(data: pd.DataFrame | Mapping[str, pd.DataFrame]) -> list[dict[str, float | int]]:
    return _DEFAULT_STRATEGY.predict(data)
