from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Signal:
    coin: str
    signal: int
    ratio: float
    leverage: int


@dataclass
class StrategyState:
    step: int
    equity: float
    peak_equity: float
    drawdown: float
    coins: tuple[str, ...]


@dataclass(frozen=True)
class BacktestConfig:
    """
    CompetitionRules uzerine override etmek icin opsiyonel runtime ayarlari.
    """

    initial_equity: float | None = None
    fee_rate: float | None = None
    min_history: int | None = None


@dataclass
class BacktestResult:
    final_equity: float
    total_return_pct: float
    max_drawdown_pct: float
    sharpe: float
    win_rate: float
    n_steps: int
    liquidation_count: int
    report_path: str | None = None
    equity_curve_path: str | None = None
    step_log_path: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)
