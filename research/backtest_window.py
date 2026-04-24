"""
Bounded backtest helper for local research infrastructure.

cnlib.backtest.run() only supports start_candle. This module mirrors its
runtime semantics while adding an inclusive end_candle for walk-forward folds
and focused smoke tests.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from cnlib.backtest import BacktestResult
from cnlib.base_strategy import BaseStrategy
from cnlib.portfolio import Portfolio
from cnlib.validator import ValidationError, validate


def run_backtest_window(
    strategy: BaseStrategy,
    initial_capital: float = 3000.0,
    start_candle: int = 0,
    end_candle: int | None = None,
    data_dir: Path | None = None,
    silent: bool = False,
) -> BacktestResult:
    """
    Run a backtest over an inclusive candle window.

    Args mirror cnlib.backtest.run(), with end_candle added. If end_candle is
    None, the run continues through the last available candle.
    """
    strategy.get_data(data_dir)
    portfolio = Portfolio(initial_capital=initial_capital)

    full_data = _full_coin_data(strategy)
    first_coin = next(iter(full_data.values()))
    total_available = len(first_coin)
    start_candle, end_candle = _validate_window(start_candle, end_candle, total_available)

    portfolio_series: list[dict] = []
    trade_history: list[dict] = []
    failed_open_history: list[dict] = []
    total_trades = 0
    validation_errors = 0
    strategy_errors = 0
    failed_opens = 0

    if not silent:
        print(
            "Backtest starting: "
            f"{total_available} candles, start_candle={start_candle}, end_candle={end_candle}"
        )

    for i in range(start_candle, end_candle + 1):
        strategy.candle_index = i

        data = strategy._candle_data(i)
        prices = strategy.current_prices(i)
        highs = strategy.current_highs(i)
        lows = strategy.current_lows(i)
        portfolio.update_prices(prices, highs, lows)

        try:
            decisions: list[dict[str, Any]] = strategy.predict(data)
        except Exception as exc:
            if not silent:
                print(f"  [Candle {i}] predict() EXCEPTION ({type(exc).__name__}): {exc}")
            strategy_errors += 1
            _record(portfolio_series, i, portfolio, prices)
            continue

        try:
            validate(decisions)
        except ValidationError as exc:
            if not silent:
                print(f"  [Candle {i}] ValidationError: {exc}")
            validation_errors += 1
            _record(portfolio_series, i, portfolio, prices)
            continue

        turn = portfolio.update_positions(decisions, prices, highs, lows)
        total_trades += len(turn["opened"]) + len(turn["closed"])

        if turn["opened"] or turn["closed"] or turn["liquidated"]:
            trade_history.append({
                "candle_index": i,
                "timestamp": first_coin.iloc[i]["Date"],
                "opened": turn["opened"],
                "closed": turn["closed"],
                "liquidated": turn["liquidated"],
                "portfolio_value": round(turn["portfolio_value"], 2),
            })

        for fail in turn.get("failed_opens", []):
            failed_opens += 1
            failed_open_history.append({
                "candle_index": i,
                "timestamp": first_coin.iloc[i]["Date"],
                "coin": fail["coin"],
                "error": fail["error"],
            })

        _record(portfolio_series, i, portfolio, prices)

        if not silent and i % 100 == 0:
            print(f"  Candle {i:>4}/{end_candle}  Portfolio: ${portfolio.portfolio_value:,.2f}")

    summary = portfolio.summary()
    import inspect as _inspect
    _result_params = set(_inspect.signature(BacktestResult.__init__).parameters)
    _kwargs: dict = dict(
        initial_capital=initial_capital,
        final_portfolio_value=summary["portfolio_value"],
        net_pnl=summary["net_pnl"],
        return_pct=summary["return_pct"],
        total_candles=end_candle - start_candle + 1,
        total_trades=total_trades,
        total_liquidations=summary["total_liquidations"],
        total_liquidation_loss=summary["total_liquidation_loss"],
        validation_errors=validation_errors,
        strategy_errors=strategy_errors,
        portfolio_series=portfolio_series,
        trade_history=trade_history,
    )
    if "failed_opens" in _result_params:
        _kwargs["failed_opens"] = failed_opens
    if "failed_open_history" in _result_params:
        _kwargs["failed_open_history"] = failed_open_history
    return BacktestResult(**_kwargs)


def _full_coin_data(strategy: BaseStrategy) -> dict:
    """Return full loaded data across cnlib 0.1.3 and 0.1.4 style strategies."""
    full_data = getattr(strategy, "_full_data", None)
    if isinstance(full_data, dict) and full_data:
        return full_data
    coin_data = getattr(strategy, "coin_data", None)
    if isinstance(coin_data, dict) and coin_data:
        return coin_data
    raise ValueError("strategy.get_data() did not load any coin data")


def _validate_window(
    start_candle: int,
    end_candle: int | None,
    total_available: int,
) -> tuple[int, int]:
    if total_available <= 0:
        raise ValueError("cannot backtest an empty dataset")
    if not isinstance(start_candle, int) or isinstance(start_candle, bool):
        raise ValueError(f"start_candle must be an integer, got {start_candle!r}")
    if start_candle < 0:
        raise ValueError(f"start_candle must be >= 0, got {start_candle}")
    if start_candle >= total_available:
        raise ValueError(
            f"start_candle {start_candle} is out of range for {total_available} candles "
            f"(max {total_available - 1})"
        )

    if end_candle is None:
        end_candle = total_available - 1
    elif not isinstance(end_candle, int) or isinstance(end_candle, bool):
        raise ValueError(f"end_candle must be an integer or None, got {end_candle!r}")

    if end_candle < start_candle:
        raise ValueError(
            f"end_candle must be >= start_candle, got {end_candle} < {start_candle}"
        )
    if end_candle >= total_available:
        raise ValueError(
            f"end_candle {end_candle} is out of range for {total_available} candles "
            f"(max {total_available - 1})"
        )

    return start_candle, end_candle


def _record(
    series: list[dict],
    index: int,
    portfolio: Portfolio,
    prices: dict[str, float],
) -> None:
    series.append({
        "candle_index": index,
        "portfolio_value": round(portfolio.portfolio_value, 2),
        "cash": round(portfolio.cash, 2),
        **{f"{coin}_price": round(price, 4) for coin, price in prices.items()},
    })
