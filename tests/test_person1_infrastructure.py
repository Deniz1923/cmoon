from contextlib import redirect_stdout
from io import StringIO
import math
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_index_equal, assert_series_equal

from cnlib.base_strategy import BaseStrategy, COINS
from cnlib.validator import validate
from research import features
from research.backtest_window import run_backtest_window
from research.ml_features import (
    build_X_y,
    build_features_single,
    feature_names,
)
from research.train_models import TRAIN_END_CANDLE, split_train_holdout
from research.walk_forward import TEST_START, holdout_test, walk_forward
from strategy import MyStrategy, _build_features_single as strategy_build_features_single


def make_ohlcv(
    close: list[float],
    high: list[float] | None = None,
    low: list[float] | None = None,
    volume: list[float] | None = None,
) -> pd.DataFrame:
    n = len(close)
    return pd.DataFrame({
        "Date": pd.date_range("2026-01-01", periods=n, freq="D"),
        "Open": close,
        "High": high if high is not None else close,
        "Low": low if low is not None else close,
        "Close": close,
        "Volume": volume if volume is not None else [100.0] * n,
    })


def make_coin_data(n: int = 8) -> dict[str, pd.DataFrame]:
    return {
        coin: make_ohlcv(
            [100.0 + i for i in range(n)],
            high=[101.0 + i for i in range(n)],
            low=[99.0 + i for i in range(n)],
            volume=[100.0 + i for i in range(n)],
        )
        for coin in COINS
    }


class DataBackedStrategy(BaseStrategy):
    def __init__(self, coin_data: dict[str, pd.DataFrame]):
        super().__init__()
        self._source_data = coin_data
        self.calls: list[int] = []

    def get_data(self, data_dir=None):
        self._full_data = {coin: df.copy() for coin, df in self._source_data.items()}
        return self.coin_data


class FlatStrategy(DataBackedStrategy):
    def predict(self, data):
        self.calls.append(self.candle_index)
        return [
            {"coin": coin, "signal": 0, "allocation": 0.0, "leverage": 1}
            for coin in COINS
        ]


class LongKapStrategy(DataBackedStrategy):
    def predict(self, data):
        self.calls.append(self.candle_index)
        return [
            {"coin": "kapcoin-usd_train", "signal": 1, "allocation": 0.5, "leverage": 2},
            {"coin": "metucoin-usd_train", "signal": 0, "allocation": 0.0, "leverage": 1},
            {"coin": "tamcoin-usd_train", "signal": 0, "allocation": 0.0, "leverage": 1},
        ]


class LiquidationOnlyStrategy(DataBackedStrategy):
    def predict(self, data):
        self.calls.append(self.candle_index)
        signal = 1 if self.candle_index == 0 else 0
        allocation = 0.5 if signal else 0.0
        return [
            {
                "coin": "kapcoin-usd_train",
                "signal": signal,
                "allocation": allocation,
                "leverage": 10 if signal else 1,
            },
            {"coin": "metucoin-usd_train", "signal": 0, "allocation": 0.0, "leverage": 1},
            {"coin": "tamcoin-usd_train", "signal": 0, "allocation": 0.0, "leverage": 1},
        ]


class TestFeatures(unittest.TestCase):
    def test_ema_sma_preserve_index_and_values(self):
        s = pd.Series([1.0, 2.0, 3.0], index=[10, 11, 12])

        assert_index_equal(features.ema(s, 2).index, s.index)
        self.assertAlmostEqual(features.ema(s, 2).iloc[-1], 2.5555555556)

        expected_sma = pd.Series([math.nan, 1.5, 2.5], index=s.index)
        assert_series_equal(features.sma(s, 2), expected_sma)

    def test_volatility_and_band_indicators(self):
        df = make_ohlcv(
            close=[10.0, 12.0, 11.0],
            high=[11.0, 13.0, 12.0],
            low=[9.0, 10.0, 10.0],
        )

        expected_tr = pd.Series([2.0, 3.0, 2.0])
        assert_series_equal(features.true_range(df), expected_tr)
        self.assertTrue(math.isnan(features.atr(df, 2).iloc[0]))
        self.assertAlmostEqual(features.atr(df, 2).iloc[1], 2.5)
        self.assertAlmostEqual(features.atr_pct(df, 2).iloc[1], 2.5 / 12.0)

        s = pd.Series([1.0, 2.0, 3.0])
        upper, mid, lower = features.bb_bands(s, 2)
        self.assertAlmostEqual(mid.iloc[-1], 2.5)
        self.assertAlmostEqual(upper.iloc[-1], 3.5)
        self.assertAlmostEqual(lower.iloc[-1], 1.5)
        self.assertAlmostEqual(features.bb_width(s, 2).iloc[-1], 2.0 / 2.5)
        self.assertAlmostEqual(features.bb_pct(s, 2).iloc[-1], 0.75)

    def test_momentum_oscillator_and_cross_coin_features(self):
        rising = pd.Series([1.0, 2.0, 3.0, 4.0])
        flat = pd.Series([1.0, 1.0, 1.0, 1.0])

        self.assertAlmostEqual(features.rsi(rising, 2).iloc[-1], 100.0)
        self.assertAlmostEqual(features.rsi(flat, 2).iloc[-1], 50.0)
        self.assertAlmostEqual(features.momentum(rising, 2).iloc[-1], 1.0)

        df = make_ohlcv([1.0, 1.0, 1.0], volume=[10.0, 20.0, 30.0])
        self.assertAlmostEqual(features.volume_ratio(df, 2).iloc[-1], 30.0 / 25.0)

        a = pd.Series([1.0, 2.0, 3.0, 4.0])
        b = pd.Series([2.0, 4.0, 6.0, 8.0])
        self.assertAlmostEqual(features.rolling_correlation(a, b, 2).iloc[-1], 1.0)

        lagged = features.lead_lag_signal(
            pd.Series([10.0, 15.0, 30.0]),
            pd.Series([5.0, 5.0, 5.0]),
            lag=1,
        )
        self.assertTrue(math.isnan(lagged.iloc[0]))
        self.assertTrue(math.isnan(lagged.iloc[1]))
        self.assertAlmostEqual(lagged.iloc[2], 0.5)

    def test_invalid_windows_and_short_inputs(self):
        s = pd.Series([1.0])
        df = make_ohlcv([1.0])

        for call in (
            lambda: features.ema(s, 0),
            lambda: features.sma(s, -1),
            lambda: features.atr(df, 0),
            lambda: features.bb_bands(s, 2, k=0),
            lambda: features.lead_lag_signal(s, s, lag=-1),
        ):
            with self.assertRaises(ValueError):
                call()

        self.assertEqual(len(features.sma(s, 3)), 1)
        self.assertTrue(math.isnan(features.sma(s, 3).iloc[0]))
        self.assertEqual(len(features.atr(df, 14)), 1)
        self.assertTrue(math.isnan(features.atr(df, 14).iloc[0]))

    def test_ohlcv_validation(self):
        with self.assertRaises(ValueError):
            features.validate_ohlcv(pd.DataFrame({"Close": [1.0]}))


class TestBacktestWindow(unittest.TestCase):
    def test_flat_strategy_bounded_window(self):
        strategy = FlatStrategy(make_coin_data(12))
        result = run_backtest_window(strategy, start_candle=2, end_candle=5, silent=True)

        self.assertEqual(result.total_candles, 4)
        self.assertEqual(strategy.calls, [2, 3, 4, 5])
        self.assertEqual(result.final_portfolio_value, 3000.0)
        self.assertEqual(result.total_trades, 0)
        self.assertEqual(result.validation_errors, 0)
        self.assertEqual(result.strategy_errors, 0)

    def test_long_strategy_opens_once_and_stops_at_end(self):
        strategy = LongKapStrategy(make_coin_data(10))
        result = run_backtest_window(strategy, start_candle=1, end_candle=3, silent=True)

        self.assertEqual(strategy.calls, [1, 2, 3])
        self.assertEqual(result.total_candles, 3)
        self.assertEqual(result.total_trades, 1)
        self.assertEqual(result.trade_history[0]["opened"], ["kapcoin-usd_train"])
        self.assertEqual(max(result.portfolio_dataframe()["candle_index"]), 3)

    def test_invalid_window_ranges_fail_clearly(self):
        strategy = FlatStrategy(make_coin_data(5))

        for kwargs in (
            {"start_candle": -1, "end_candle": 2},
            {"start_candle": 5, "end_candle": None},
            {"start_candle": 3, "end_candle": 2},
            {"start_candle": 0, "end_candle": 5},
        ):
            with self.assertRaises(ValueError):
                run_backtest_window(strategy, silent=True, **kwargs)

    def test_liquidation_only_turn_is_recorded(self):
        data = make_coin_data(2)
        data["kapcoin-usd_train"] = make_ohlcv(
            close=[100.0, 100.0],
            high=[100.0, 100.0],
            low=[100.0, 89.0],
        )
        strategy = LiquidationOnlyStrategy(data)
        result = run_backtest_window(strategy, start_candle=0, end_candle=1, silent=True)

        self.assertEqual(result.total_liquidations, 1)
        self.assertEqual(result.trade_history[-1]["opened"], [])
        self.assertEqual(result.trade_history[-1]["closed"], [])
        self.assertEqual(result.trade_history[-1]["liquidated"], ["kapcoin-usd_train"])


class TestRunnerAndWalkForward(unittest.TestCase):
    def test_strategy_registry_errors_are_friendly(self):
        import run

        self.assertIn("main", run.STRATEGIES)
        self.assertIsNotNone(run.get_strategy("main"))

        with self.assertRaises(ValueError):
            run.get_strategy("missing")

    def test_dataset_resolution_prefers_explicit_dir_and_supports_presets(self):
        import run

        explicit = Path("/tmp/custom-data")
        self.assertEqual(run.resolve_data_dir("cnlib", explicit), explicit)
        self.assertIsNone(run.resolve_data_dir("cnlib", None))

        with patch.object(run, "SYNTHETIC_DATASET_DIR", Path("/tmp/synthetic")):
            with patch.dict(run.DATASET_PRESETS, {"cnlib": None, "synthetic": Path("/tmp/synthetic")}, clear=True):
                with patch.object(Path, "exists", return_value=True):
                    self.assertEqual(
                        run.resolve_data_dir("synthetic", None),
                        Path("/tmp/synthetic"),
                    )

    def test_walk_forward_runs_only_train_fold_windows(self):
        data = make_coin_data(TEST_START + 20)
        strategies: list[FlatStrategy] = []

        def factory():
            strategy = FlatStrategy(data)
            strategies.append(strategy)
            return strategy

        hook_calls: list[int] = []
        with redirect_stdout(StringIO()):
            results = walk_forward(factory, n_splits=4, retrain_hook=hook_calls.append)

        self.assertEqual(len(results), 4)
        self.assertEqual(len(hook_calls), 4)
        self.assertTrue(all(r.test_end < TEST_START for r in results))
        self.assertTrue(all(max(strategy.calls) < TEST_START for strategy in strategies))

    def test_holdout_test_starts_at_configured_holdout(self):
        data = make_coin_data(TEST_START + 10)
        strategy = FlatStrategy(data)

        with redirect_stdout(StringIO()):
            result = holdout_test(strategy)

        self.assertEqual(result.test_start, TEST_START)
        self.assertEqual(result.test_end, TEST_START + 9)
        self.assertEqual(strategy.calls[0], TEST_START)
        self.assertEqual(strategy.calls[-1], TEST_START + 9)

    def test_cli_smoke_accepts_end_candle(self):
        project_root = Path(__file__).resolve().parent.parent
        completed = subprocess.run(
            [sys.executable, "-B", "run.py", "--silent", "--end", "5"],
            cwd=project_root,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("Total Candles", completed.stdout)
        self.assertIn("6", completed.stdout)


class FixedProbModel:
    classes_ = np.array([0, 1])

    def __init__(self, prob_up: float):
        self.prob_up = prob_up

    def predict_proba(self, X):
        return np.array([[1.0 - self.prob_up, self.prob_up] for _ in range(len(X))])


class AlwaysLongStrategy(MyStrategy):
    def _rule_signal(self, coin: str, df: pd.DataFrame, data: dict) -> int:
        return 1


class TestPerson3Ensemble(unittest.TestCase):
    def test_missing_model_artifacts_return_valid_flat_decisions(self):
        with TemporaryDirectory() as tmp:
            with patch("strategy.RESULTS_DIR", Path(tmp)):
                strategy = MyStrategy()

        decisions = strategy.predict(make_coin_data(140))

        validate(decisions)
        self.assertEqual([d["signal"] for d in decisions], [0, 0, 0])
        self.assertEqual([d["allocation"] for d in decisions], [0.0, 0.0, 0.0])

    def test_ensemble_caps_active_positions_and_total_allocation(self):
        strategy = AlwaysLongStrategy()
        strategy.models = {coin: FixedProbModel(0.99) for coin in COINS}
        strategy.model_feature_names = {}

        decisions = strategy.predict(make_coin_data(140))
        active = [d for d in decisions if d["signal"] != 0]

        validate(decisions)
        self.assertLessEqual(len(active), 2)
        self.assertLessEqual(sum(d["allocation"] for d in active), 0.9)
        self.assertTrue(all(d["allocation"] == 0.45 for d in active))

    def test_inference_feature_order_matches_training_features(self):
        data = make_coin_data(140)

        for coin in COINS:
            leader = data["kapcoin-usd_train"]["Close"] if coin != "kapcoin-usd_train" else None
            inferred_names = list(build_features_single(data[coin], leader_close=leader).columns)
            self.assertEqual(inferred_names, feature_names(data, coin))

    def test_inference_feature_values_match_training_features(self):
        data = make_coin_data(140)

        for coin in COINS:
            leader = data["kapcoin-usd_train"]["Close"] if coin != "kapcoin-usd_train" else None
            assert_frame_equal(
                strategy_build_features_single(data[coin], leader_close=leader),
                build_features_single(data[coin], leader_close=leader),
            )

    def test_inference_rejects_invalid_current_feature_row(self):
        data = make_coin_data(140)
        last = data["kapcoin-usd_train"].index[-1]
        data["kapcoin-usd_train"].loc[last, ["Open", "High", "Low", "Close"]] = 0.0

        with TemporaryDirectory() as tmp:
            with patch("strategy.RESULTS_DIR", Path(tmp)):
                strategy = MyStrategy()

        self.assertIsNone(strategy._ml_feature_row("kapcoin-usd_train", data))

    def test_build_x_y_skips_warmup_rows_even_for_short_inputs(self):
        X, y, valid_index = build_X_y(make_coin_data(60), "kapcoin-usd_train")

        self.assertEqual(X.shape[0], 0)
        self.assertEqual(len(y), 0)
        self.assertEqual(len(valid_index), 0)

    def test_train_holdout_split_uses_candle_index_not_row_count(self):
        X = np.arange(20, dtype=np.float32).reshape(5, 4)
        y = np.array([0, 1, 0, 1, 1])
        valid_index = pd.Index([100, TRAIN_END_CANDLE - 1, TRAIN_END_CANDLE, 1200, 1300])

        X_train, y_train, X_test, y_test, train_index, test_index = split_train_holdout(
            X,
            y,
            valid_index,
        )

        self.assertEqual(list(train_index), [100, TRAIN_END_CANDLE - 1])
        self.assertEqual(list(test_index), [TRAIN_END_CANDLE, 1200, 1300])
        self.assertEqual(len(X_train), 2)
        self.assertEqual(len(y_train), 2)
        self.assertEqual(len(X_test), 3)
        self.assertEqual(len(y_test), 3)


if __name__ == "__main__":
    unittest.main()
