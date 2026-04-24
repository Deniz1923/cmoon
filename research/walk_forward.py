"""
Walk-forward validation — Person 1.

Use this to test any strategy on the holdout set WITHOUT peeking at it during
tuning. Only run on the test window once you're happy with train performance.

Also provides n-fold walk-forward to catch regime-specific overfitting.
"""
from dataclasses import dataclass
from typing import Callable

from cnlib.base_strategy import BaseStrategy
from research.backtest_window import run_backtest_window

# Sacred holdout — do not tune against this window
TRAIN_END  = 1100
TEST_START = 1100

StrategyFactory = Callable[[], BaseStrategy]
RetrainHook = Callable[[int], None]


@dataclass
class SplitResult:
    split_name: str
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    final_portfolio_value: float
    return_pct: float
    total_trades: int
    total_liquidations: int
    validation_errors: int
    strategy_errors: int

    def __repr__(self):
        return (
            f"[{self.split_name}] "
            f"train={self.train_start}-{self.train_end} "
            f"test={self.test_start}-{self.test_end} | "
            f"return={self.return_pct:+.2f}% "
            f"trades={self.total_trades} "
            f"liq={self.total_liquidations}"
        )


def holdout_test(
    strategy: BaseStrategy,
    initial_capital: float = 3000.0,
) -> SplitResult:
    """
    Run the strategy only on the holdout window (candles 1100–1569).

    Call this ONCE at the very end when you're done tuning.
    """
    result = run_backtest_window(
        strategy,
        initial_capital=initial_capital,
        start_candle=TEST_START,
        end_candle=None,
        silent=True,
    )
    result.print_summary()
    return SplitResult(
        split_name="HOLDOUT",
        train_start=0,
        train_end=TRAIN_END - 1,
        test_start=TEST_START,
        test_end=TEST_START + result.total_candles - 1,
        final_portfolio_value=result.final_portfolio_value,
        return_pct=result.return_pct,
        total_trades=result.total_trades,
        total_liquidations=result.total_liquidations,
        validation_errors=result.validation_errors,
        strategy_errors=result.strategy_errors,
    )


def walk_forward(
    strategy_cls: type[BaseStrategy] | StrategyFactory,
    n_splits: int = 4,
    initial_capital: float = 3000.0,
    retrain_hook: RetrainHook | None = None,
) -> list[SplitResult]:
    """
    Walk-forward cross-validation on the TRAIN window only (candles 0–1099).

    Splits the train window into n equal folds. Each fold runs only inside its
    test window, so future train/holdout candles are never executed.
    """
    if not isinstance(n_splits, int) or isinstance(n_splits, bool) or n_splits <= 0:
        raise ValueError(f"n_splits must be a positive integer, got {n_splits!r}")
    if n_splits >= TRAIN_END:
        raise ValueError(f"n_splits must be less than {TRAIN_END}, got {n_splits}")

    fold_size = TRAIN_END // (n_splits + 1)
    if fold_size <= 0:
        raise ValueError("n_splits creates empty walk-forward folds")

    results = []
    for i in range(n_splits):
        test_start = fold_size * (i + 1)
        test_end   = min(test_start + fold_size - 1, TRAIN_END - 1)
        train_end = test_start - 1

        if retrain_hook is not None:
            retrain_hook(train_end)

        strategy = strategy_cls()
        result = run_backtest_window(
            strategy,
            initial_capital=initial_capital,
            start_candle=test_start,
            end_candle=test_end,
            silent=True,
        )

        sr = SplitResult(
            split_name=f"fold_{i+1}",
            train_start=0,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            final_portfolio_value=result.final_portfolio_value,
            return_pct=result.return_pct,
            total_trades=result.total_trades,
            total_liquidations=result.total_liquidations,
            validation_errors=result.validation_errors,
            strategy_errors=result.strategy_errors,
        )
        results.append(sr)
        print(sr)

    if not results:
        raise RuntimeError("walk_forward produced no fold results")

    avg_return = sum(r.return_pct for r in results) / len(results)
    print(f"\nWalk-forward avg return: {avg_return:+.2f}%")
    print("(If individual folds vary wildly, you're overfitting to a specific regime)")
    return results


if __name__ == "__main__":
    # TODO: replace TrendStrategy with whichever strategy you're testing
    from research.trend_strategy import TrendStrategy

    print("=== Walk-forward (train window) ===")
    walk_forward(TrendStrategy, n_splits=4)

    print("\n=== Holdout test ===")
    holdout_test(TrendStrategy())
