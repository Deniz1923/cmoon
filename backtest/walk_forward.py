from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import pandas as pd

from backtest.runner import BacktestResult, run_backtest


@dataclass(frozen=True)
class WalkForwardFold:
    name: str
    train_start: object
    train_end: object
    test_start: object
    test_end: object
    result: BacktestResult


def run_walk_forward(
    strategy_factory: Callable[[], object],
    data: Mapping[str, pd.DataFrame],
    *,
    train_size: int,
    test_size: int,
    step_size: int | None = None,
) -> list[WalkForwardFold]:
    timeline = sorted(set().union(*(frame.index for frame in data.values())))
    step = step_size or test_size
    folds: list[WalkForwardFold] = []

    start = 0
    fold_number = 1
    while start + train_size + test_size <= len(timeline):
        train_index = timeline[start : start + train_size]
        test_index = timeline[start + train_size : start + train_size + test_size]

        train_data = _slice_data(data, train_index[0], train_index[-1])
        test_data = _slice_data(data, test_index[0], test_index[-1])
        strategy = strategy_factory()
        if hasattr(strategy, "fit"):
            strategy.fit(train_data)
        result = run_backtest(strategy, test_data, fit_strategy=False)

        folds.append(
            WalkForwardFold(
                name=f"fold_{fold_number}",
                train_start=train_index[0],
                train_end=train_index[-1],
                test_start=test_index[0],
                test_end=test_index[-1],
                result=result,
            )
        )
        start += step
        fold_number += 1

    return folds


def _slice_data(data: Mapping[str, pd.DataFrame], start: object, end: object) -> dict[str, pd.DataFrame]:
    return {asset: frame.loc[(frame.index >= start) & (frame.index <= end)] for asset, frame in data.items()}
