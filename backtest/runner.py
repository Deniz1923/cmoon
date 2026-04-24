from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtest.metrics import MetricReport, equity_curve_metrics
from features.engineer import normalize_ohlcv


@dataclass(frozen=True)
class BacktestResult:
    equity: pd.Series
    orders: pd.DataFrame
    metrics: MetricReport


def run_backtest(
    strategy,
    data: Mapping[str, pd.DataFrame],
    *,
    initial_equity: float = 1.0,
    fee_bps: float = 2.0,
    periods_per_year: int = 365,
    fit_strategy: bool = True,
) -> BacktestResult:
    """Simple causal local engine for comparing ideas before organizer validation."""
    frames = {asset: normalize_ohlcv(frame) for asset, frame in data.items()}
    timeline = sorted(set().union(*(frame.index for frame in frames.values())))
    if len(timeline) < 2:
        equity = pd.Series([initial_equity], index=timeline or [0])
        return BacktestResult(equity, pd.DataFrame(), equity_curve_metrics(equity))

    if fit_strategy and hasattr(strategy, "fit"):
        strategy.fit(frames)

    equity_value = float(initial_equity)
    equity_points: list[tuple[object, float]] = []
    order_rows: list[dict[str, object]] = []

    for timestamp, next_timestamp in zip(timeline[:-1], timeline[1:], strict=False):
        histories = {
            asset: frame.loc[frame.index <= timestamp]
            for asset, frame in frames.items()
            if not frame.loc[frame.index <= timestamp].empty
        }
        if not histories:
            continue

        raw_orders = strategy.predict(histories)
        orders = _orders_by_asset(raw_orders, list(frames))

        portfolio_return = 0.0
        turnover_fee = 0.0
        for asset, order in orders.items():
            asset_return = _next_return(frames[asset], timestamp, next_timestamp)
            if asset_return is None:
                continue

            sinyal = int(order.get("sinyal", 0))
            oran = float(order.get("oran", 0.0))
            kaldirac = int(order.get("kaldirac", 2))
            position_return = sinyal * oran * kaldirac * asset_return
            portfolio_return += max(-0.95 * oran, position_return)
            turnover_fee += oran * kaldirac * fee_bps / 10_000

            order_rows.append(
                {
                    "timestamp": timestamp,
                    "asset": asset,
                    "sinyal": sinyal,
                    "oran": oran,
                    "kaldirac": kaldirac,
                    "asset_return": asset_return,
                }
            )

        equity_value *= max(0.01, 1.0 + portfolio_return - turnover_fee)
        equity_points.append((next_timestamp, equity_value))

    equity = pd.Series(
        [initial_equity] + [point[1] for point in equity_points],
        index=[timeline[0]] + [point[0] for point in equity_points],
        name="equity",
    )
    orders_frame = pd.DataFrame(order_rows)
    return BacktestResult(
        equity=equity,
        orders=orders_frame,
        metrics=equity_curve_metrics(equity, periods_per_year=periods_per_year),
    )


def _orders_by_asset(raw_orders: list[dict], assets: list[str]) -> dict[str, dict]:
    by_asset: dict[str, dict] = {}
    for index, asset in enumerate(assets):
        order = raw_orders[index] if index < len(raw_orders) else {}
        by_asset[asset] = order
    return by_asset


def _next_return(frame: pd.DataFrame, timestamp: object, next_timestamp: object) -> float | None:
    current = frame.loc[frame.index <= timestamp]
    future = frame.loc[frame.index <= next_timestamp]
    if current.empty or future.empty:
        return None

    current_close = float(current["close"].iloc[-1])
    next_close = float(future["close"].iloc[-1])
    if not np.isfinite(current_close) or current_close == 0.0:
        return None
    return next_close / current_close - 1.0
