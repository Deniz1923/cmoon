from __future__ import annotations

from dataclasses import dataclass

from models.base_model import RawSignal


ALLOWED_LEVERAGE = (2, 3, 5, 10)


@dataclass(frozen=True)
class RiskState:
    equity: float = 1.0
    peak_equity: float = 1.0

    @property
    def drawdown(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return max(0.0, 1.0 - self.equity / self.peak_equity)


@dataclass(frozen=True)
class PositionDecision:
    asset: str
    sinyal: int
    oran: float
    kaldirac: int
    confidence: float = 0.0

    def to_order_dict(self, *, include_asset: bool = False) -> dict[str, float | int | str]:
        order: dict[str, float | int | str] = {
            "sinyal": int(self.sinyal),
            "oran": float(round(min(max(self.oran, 0.0), 1.0), 6)),
            "kaldirac": int(self.kaldirac),
        }
        if include_asset:
            order["asset"] = self.asset
        return order


class VolatilityAwareSizer:
    def __init__(
        self,
        *,
        max_asset_allocation: float = 0.42,
        target_volatility: float = 0.018,
        min_signal_confidence: float = 0.12,
    ) -> None:
        self.max_asset_allocation = max_asset_allocation
        self.target_volatility = target_volatility
        self.min_signal_confidence = min_signal_confidence

    def size(
        self,
        signal: RawSignal,
        *,
        volatility: float,
        risk_state: RiskState | None = None,
    ) -> PositionDecision:
        risk = risk_state or RiskState()
        confidence = min(max(signal.confidence, 0.0), 1.0)
        volatility = max(float(volatility), 1e-6)

        if signal.direction == 0 or confidence < self.min_signal_confidence:
            return PositionDecision(signal.asset, 0, 0.0, 2, confidence)

        leverage = self._choose_leverage(confidence, volatility, risk.drawdown)
        vol_haircut = min(1.25, max(0.20, self.target_volatility / volatility))
        drawdown_haircut = min(1.0, max(0.20, 1.0 - risk.drawdown * 3.0))
        allocation = self.max_asset_allocation * (confidence**1.7) * vol_haircut * drawdown_haircut
        allocation = min(self.max_asset_allocation, max(0.0, allocation))

        return PositionDecision(
            asset=signal.asset,
            sinyal=signal.direction,
            oran=allocation,
            kaldirac=leverage,
            confidence=confidence,
        )

    @staticmethod
    def _choose_leverage(confidence: float, volatility: float, drawdown: float) -> int:
        if confidence >= 0.90 and volatility <= 0.012 and drawdown <= 0.02:
            return 10
        if confidence >= 0.74 and volatility <= 0.020 and drawdown <= 0.06:
            return 5
        if confidence >= 0.52 and volatility <= 0.035 and drawdown <= 0.12:
            return 3
        return 2


def cap_total_exposure(decisions: list[PositionDecision], cap: float = 1.0) -> list[PositionDecision]:
    total = sum(max(0.0, decision.oran) for decision in decisions)
    if total <= cap or total <= 0.0:
        return decisions

    scale = cap / total
    return [
        PositionDecision(
            asset=decision.asset,
            sinyal=decision.sinyal,
            oran=decision.oran * scale,
            kaldirac=decision.kaldirac,
            confidence=decision.confidence,
        )
        for decision in decisions
    ]
