"""
Walk-forward validation — Person 1.

Use this to test any strategy on the holdout set WITHOUT peeking at it during
tuning. Only run on the test window once you're happy with train performance.

Also provides n-fold walk-forward to catch regime-specific overfitting.
"""
from dataclasses import dataclass
from typing import Type

import pandas as pd

from cnlib import backtest
from cnlib.base_strategy import BaseStrategy

# Sacred holdout — do not tune against this window
TRAIN_END   = 1100
TEST_START  = 1100
TOTAL_CANDLES = 1570  # TODO: confirm this matches actual dataset length


@dataclass
class SplitResult:
    split_name: str
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    return_pct: float
    total_liquidations: int
    validation_errors: int

    def __repr__(self):
        return (
            f"[{self.split_name}] "
            f"train={self.train_start}-{self.train_end} "
            f"test={self.test_start}-{self.test_end} | "
            f"return={self.return_pct:+.2f}% "
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
    result = backtest.run(
        strategy,
        initial_capital=initial_capital,
        start_candle=TEST_START,
        silent=True,
    )
    result.print_summary()
    return SplitResult(
        split_name="HOLDOUT",
        train_start=0,
        train_end=TRAIN_END - 1,
        test_start=TEST_START,
        test_end=TOTAL_CANDLES - 1,
        return_pct=result.return_pct,
        total_liquidations=result.total_liquidations,
        validation_errors=result.validation_errors,
    )


def walk_forward(
    strategy_cls: Type[BaseStrategy],
    n_splits: int = 4,
    initial_capital: float = 3000.0,
) -> list[SplitResult]:
    """
    Walk-forward cross-validation on the TRAIN window only (0–1099).

    Splits the train window into n expanding folds. Each fold:
      - train on all prior candles
      - test on the next chunk

    This catches if your strategy only works in a specific regime.

    TODO: for ML strategies, you'll need to re-train the model inside each fold.
          Add a `retrain_hook` callback parameter if needed.
    """
    train_window = TRAIN_END
    fold_size = train_window // (n_splits + 1)

    results = []
    for i in range(n_splits):
        test_start = fold_size * (i + 1)
        test_end   = test_start + fold_size

        # TODO: for ML strategies — retrain model on candles 0..test_start before each fold

        strategy = strategy_cls()  # fresh instance per fold
        result = backtest.run(
            strategy,
            initial_capital=initial_capital,
            start_candle=test_start,
            silent=True,
        )

        sr = SplitResult(
            split_name=f"fold_{i+1}",
            train_start=0,
            train_end=test_start - 1,
            test_start=test_start,
            test_end=test_end,
            return_pct=result.return_pct,
            total_liquidations=result.total_liquidations,
            validation_errors=result.validation_errors,
        )
        results.append(sr)
        print(sr)

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
