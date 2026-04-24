"""
test_contracts.py
Contract tests for every module. These run fast (synthetic data, no ML training).
They verify schema, index invariants, leakage rules, and fill logic — not prediction quality.
"""
import numpy as np
import pandas as pd
import pytest

# ── Synthetic helpers ──────────────────────────────────────────────────────────

def _make_ohlcv(n: int = 500) -> pd.DataFrame:
    """Minimal synthetic OHLCV with UTC DatetimeIndex."""
    idx = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
    close = 30_000 + np.cumsum(np.random.randn(n) * 50)
    high = close + np.abs(np.random.randn(n) * 30)
    low = close - np.abs(np.random.randn(n) * 30)
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": np.random.rand(n) * 1e6},
        index=idx,
    )


def _make_config() -> dict:
    return {
        "paths": {
            "raw_data": "data/raw",
            "processed_data": "data/processed",
            "artifacts": "artifacts",
            "models": "artifacts/models",
            "predictions": "artifacts/predictions",
            "reports": "artifacts/reports",
        },
        "market": {"symbols": ["BTCUSDT"], "market_type": "spot", "bar_interval": "1h"},
        "data_source": {"provider": "csv"},
        "splits": {"train_years": 3, "validation_years": 1, "locked_holdout_months": 12, "embargo_days": 5},
        "features": {
            "price": ["returns", "log_returns"],
            "trend": ["ema", "macd"],
            "momentum": ["rsi", "stochastic"],
            "volatility": ["atr", "realized_volatility", "bollinger_width"],
            "volume": ["obv", "vwap_distance"],
        },
        "labels": {"method": "triple_barrier", "take_profit": 0.02, "stop_loss": 0.01, "max_horizon_bars": 24, "class_weight": "balanced"},
        "cross_validation": {"method": "purged_walk_forward", "folds": 3, "embargo_bars": 24},
        "model": {"baseline": "rf", "random_state": 42, "max_tuning_trials": 10},
        "signals": {"long_threshold": 0.60, "short_threshold": 0.40},
        "risk": {
            "target_annual_volatility": 0.15,
            "max_single_position_equity_fraction": 0.20,
            "max_leverage": 1.0,
            "stop_loss_atr_multiple": 1.0,
            "take_profit_atr_multiple": 2.0,
            "max_hold_bars": 24,
            "kill_switch_drawdown": 0.15,
        },
        "backtest": {
            "initial_equity": 100_000,
            "fill_policy": "next_bar_open",
            "fee_bps": 5,
            "slippage_bps": 2,
            "funding_enabled": False,
        },
        "reporting": {
            "experiment_log": "artifacts/experiment_log.csv",
            "equity_curve_plot": "artifacts/reports/equity_curve.png",
            "tear_sheet": "artifacts/reports/tear_sheet.png",
        },
    }


# ── data_loader contracts ──────────────────────────────────────────────────────

class TestDataLoader:
    def test_validate_returns_utc_index(self):
        from src.data_loader import _validate_and_normalize
        raw = _make_ohlcv(200)
        raw.index = raw.index.tz_localize(None)  # strip tz
        result = _validate_and_normalize(raw, "BTCUSDT")
        assert result.index.tz is not None, "Index must be tz-aware"
        assert str(result.index.tz) == "UTC"

    def test_validate_monotonic(self):
        from src.data_loader import _validate_and_normalize
        raw = _make_ohlcv(100)
        result = _validate_and_normalize(raw, "BTCUSDT")
        assert result.index.is_monotonic_increasing

    def test_validate_no_nans(self):
        from src.data_loader import _validate_and_normalize
        raw = _make_ohlcv(100)
        raw.loc[raw.index[5], "close"] = np.nan
        result = _validate_and_normalize(raw, "BTCUSDT")
        assert not result[["open", "high", "low", "close", "volume"]].isna().any().any()

    def test_get_splits_three_way(self):
        from src.data_loader import get_splits
        # Need enough data: 3 train + 1 val + 1 holdout years ≈ 5 years hourly
        ohlcv = _make_ohlcv(5 * 365 * 24)
        config = _make_config()
        train, val, holdout = get_splits(ohlcv, config)
        assert len(train) > 0
        assert len(val) > 0
        assert len(holdout) > 0

    def test_get_splits_no_overlap(self):
        from src.data_loader import get_splits
        ohlcv = _make_ohlcv(5 * 365 * 24)
        config = _make_config()
        train, val, holdout = get_splits(ohlcv, config)
        assert train.index[-1] < val.index[0], "Train must end before val starts (embargo)"
        assert val.index[-1] < holdout.index[0], "Val must end before holdout starts"

    def test_get_splits_holdout_is_last(self):
        from src.data_loader import get_splits
        ohlcv = _make_ohlcv(5 * 365 * 24)
        config = _make_config()
        train, val, holdout = get_splits(ohlcv, config)
        assert holdout.index[0] > val.index[-1], "Holdout must come after val chronologically"


# ── feature_engineering contracts ─────────────────────────────────────────────

class TestFeatureEngineering:
    def test_all_columns_have_feat_prefix(self):
        from src.feature_engineering import build_features
        ohlcv = _make_ohlcv(300)
        config = _make_config()
        feats = build_features(ohlcv, config)
        bad = [c for c in feats.columns if not c.startswith("feat_")]
        assert not bad, f"Non-feat_ columns found: {bad}"

    def test_index_aligned_to_input(self):
        from src.feature_engineering import build_features
        ohlcv = _make_ohlcv(300)
        config = _make_config()
        feats = build_features(ohlcv, config)
        assert feats.index.equals(ohlcv.index)

    def test_scaler_fit_on_train_only(self):
        from src.feature_engineering import build_features, fit_transformer, transform
        ohlcv = _make_ohlcv(400)
        config = _make_config()
        feats = build_features(ohlcv, config)
        train_feats = feats.iloc[:200]
        val_feats = feats.iloc[200:]

        transformer = fit_transformer(train_feats.dropna())
        assert transformer.fitted

        # transform val must not crash and must return same shape
        val_scaled = transform(val_feats, transformer)
        assert val_scaled.shape == val_feats.shape

    def test_no_leakage_shift_invariant(self):
        """
        If we remove the last bar from the OHLCV, every feature at time t-1
        should remain unchanged (features don't 'look ahead').
        """
        from src.feature_engineering import build_features
        ohlcv = _make_ohlcv(300)
        config = _make_config()
        feats_full = build_features(ohlcv, config)
        feats_trimmed = build_features(ohlcv.iloc[:-1], config)

        shared_idx = feats_full.index[:-1]
        diff = (feats_full.loc[shared_idx] - feats_trimmed.loc[shared_idx]).abs().max().max()
        assert diff < 1e-8, f"Feature values changed when last bar removed — possible leakage! max diff={diff}"


# ── labeling contracts ─────────────────────────────────────────────────────────

class TestLabeling:
    def test_labels_are_valid_classes(self):
        from src.labeling import make_labels
        ohlcv = _make_ohlcv(200)
        config = _make_config()
        labels = make_labels(ohlcv, config)
        assert set(labels["target"].unique()).issubset({-1, 0, 1})

    def test_labels_drop_last_horizon_rows(self):
        from src.labeling import make_labels
        ohlcv = _make_ohlcv(200)
        config = _make_config()
        max_h = config["labels"]["max_horizon_bars"]
        labels = make_labels(ohlcv, config)
        assert len(labels) <= len(ohlcv) - max_h

    def test_label_index_subset_of_ohlcv(self):
        from src.labeling import make_labels
        ohlcv = _make_ohlcv(200)
        config = _make_config()
        labels = make_labels(ohlcv, config)
        assert labels.index.isin(ohlcv.index).all()


# ── cv_splitter contracts ──────────────────────────────────────────────────────

class TestCVSplitter:
    def test_val_always_after_train(self):
        from src.cv_splitter import purged_walk_forward_cv
        df = _make_ohlcv(1000)
        config = _make_config()
        for train_idx, val_idx in purged_walk_forward_cv(df, n_folds=3, embargo_bars=24):
            assert train_idx.max() < val_idx.min(), "Train must end before val starts"

    def test_embargo_gap_respected(self):
        from src.cv_splitter import purged_walk_forward_cv
        df = _make_ohlcv(1000)
        embargo = 48
        for train_idx, val_idx in purged_walk_forward_cv(df, n_folds=3, embargo_bars=embargo):
            gap = val_idx.min() - train_idx.max()
            assert gap >= embargo, f"Embargo gap too small: {gap} < {embargo}"

    def test_no_random_kfold(self):
        """Folds must be chronological — val indices should be strictly increasing across folds."""
        from src.cv_splitter import purged_walk_forward_cv
        df = _make_ohlcv(1000)
        val_starts = [v.min() for _, v in purged_walk_forward_cv(df, n_folds=4, embargo_bars=24)]
        assert val_starts == sorted(val_starts), "Folds are not in chronological order"


# ── signal_generator contracts ─────────────────────────────────────────────────

class TestSignalGenerator:
    def _make_probs(self, n=100) -> pd.DataFrame:
        idx = pd.date_range("2021-01-01", periods=n, freq="h", tz="UTC")
        p_long = np.random.dirichlet([1, 1, 1], size=n)
        return pd.DataFrame(p_long, columns=[-1, 0, 1], index=idx)

    def test_signal_values(self):
        from src.signal_generator import generate_signals
        probs = self._make_probs()
        config = _make_config()
        sigs = generate_signals(probs, config)
        assert set(sigs["signal"].unique()).issubset({-1, 0, 1})

    def test_confidence_in_unit_interval(self):
        from src.signal_generator import generate_signals
        probs = self._make_probs()
        config = _make_config()
        sigs = generate_signals(probs, config)
        assert sigs["confidence"].between(0, 1).all()


# ── risk_manager contracts ─────────────────────────────────────────────────────

class TestRiskManager:
    def test_position_size_within_cap(self):
        from src.risk_manager import size_positions
        from src.signal_generator import generate_signals
        ohlcv = _make_ohlcv(200)
        config = _make_config()

        probs = pd.DataFrame(
            np.column_stack([np.zeros(len(ohlcv)), np.zeros(len(ohlcv)), np.ones(len(ohlcv))]),
            columns=[-1, 0, 1],
            index=ohlcv.index,
        )
        sigs = generate_signals(probs, config)
        equity = 100_000.0
        orders = size_positions(sigs, equity, ohlcv, config)

        max_frac = config["risk"]["max_single_position_equity_fraction"]
        max_leverage = config["risk"]["max_leverage"]
        max_pos_val = equity * max_frac * max_leverage

        long_orders = orders[orders["signal"] == 1]
        if len(long_orders) > 0:
            worst = (long_orders["position_size"].abs() * ohlcv.loc[long_orders.index, "close"]).max()
            assert worst <= max_pos_val * 1.01, f"Position size exceeds cap: {worst} > {max_pos_val}"


# ── backtest_engine contracts ──────────────────────────────────────────────────

class TestBacktestEngine:
    def test_fills_at_next_bar_open(self):
        """Orders at bar t should fill at bar t+1 open, not bar t."""
        from src.backtest_engine import run_backtest
        from src.risk_manager import size_positions
        from src.signal_generator import generate_signals
        ohlcv = _make_ohlcv(100)
        config = _make_config()

        # Force long signal everywhere
        probs = pd.DataFrame(
            np.column_stack([np.zeros(len(ohlcv)), np.zeros(len(ohlcv)), np.ones(len(ohlcv))]),
            columns=[-1, 0, 1],
            index=ohlcv.index,
        )
        sigs = generate_signals(probs, config)
        orders = size_positions(sigs, 100_000, ohlcv, config)
        trades, equity = run_backtest({"TEST": ohlcv}, {"TEST": orders}, config)

        if len(trades) > 0:
            for _, trade in trades.iterrows():
                entry_ts = trade["entry_ts"]
                entry_idx = ohlcv.index.get_loc(entry_ts)
                if entry_idx > 0:
                    expected_open = ohlcv.iloc[entry_idx]["open"]
                    slippage_factor = 1 + config["backtest"]["slippage_bps"] / 10_000
                    assert abs(trade["entry_price"] / (expected_open * slippage_factor) - 1) < 0.001

    def test_equity_never_nan(self):
        from src.backtest_engine import run_backtest
        from src.risk_manager import size_positions
        from src.signal_generator import generate_signals
        ohlcv = _make_ohlcv(200)
        config = _make_config()
        probs = pd.DataFrame(
            np.column_stack([np.zeros(len(ohlcv)), np.ones(len(ohlcv)), np.zeros(len(ohlcv))]),
            columns=[-1, 0, 1],
            index=ohlcv.index,
        )
        sigs = generate_signals(probs, config)
        orders = size_positions(sigs, 100_000, ohlcv, config)
        _, equity = run_backtest({"TEST": ohlcv}, {"TEST": orders}, config)
        assert not equity.isna().any(), "Equity curve contains NaN"

    def test_no_future_data_access(self):
        """
        Equity at bar t should be identical whether or not we have data after t.
        (Simulating that the engine does not look ahead.)
        """
        from src.backtest_engine import run_backtest
        from src.risk_manager import size_positions
        from src.signal_generator import generate_signals
        ohlcv = _make_ohlcv(200)
        config = _make_config()
        probs = pd.DataFrame(
            np.column_stack([np.zeros(len(ohlcv)), np.ones(len(ohlcv)), np.zeros(len(ohlcv))]),
            columns=[-1, 0, 1],
            index=ohlcv.index,
        )
        sigs = generate_signals(probs, config)
        orders = size_positions(sigs, 100_000, ohlcv, config)

        _, eq_full = run_backtest({"TEST": ohlcv}, {"TEST": orders}, config)
        _, eq_trim = run_backtest({"TEST": ohlcv.iloc[:-50]}, {"TEST": orders}, config)

        shared = eq_full.index.intersection(eq_trim.index)
        if len(shared) > 0:
            diff = (eq_full.loc[shared] - eq_trim.loc[shared]).abs().max()
            assert diff < 1.0, f"Equity diverges between full and trimmed run — possible look-ahead: {diff}"


# ── metrics contracts ──────────────────────────────────────────────────────────

class TestMetrics:
    def test_evaluate_returns_required_keys(self):
        from src.metrics import evaluate
        idx = pd.date_range("2021-01-01", periods=200, freq="h", tz="UTC")
        equity = pd.Series(100_000 + np.cumsum(np.random.randn(200) * 10), index=idx)
        trades = pd.DataFrame({"pnl": np.random.randn(20) * 100})
        result = evaluate(equity, trades)
        for key in ("sharpe", "sortino", "max_drawdown", "calmar", "win_rate", "trade_count"):
            assert key in result, f"Missing key: {key}"

    def test_max_drawdown_non_positive(self):
        from src.metrics import evaluate
        idx = pd.date_range("2021-01-01", periods=200, freq="h", tz="UTC")
        equity = pd.Series(100_000 + np.cumsum(np.random.randn(200) * 10), index=idx)
        result = evaluate(equity, pd.DataFrame())
        assert result["max_drawdown"] <= 0


# ── integration: no date slicing outside data_loader ──────────────────────────

class TestNoDateSlicingOutsideDataLoader:
    """
    Grep-based contract: no module outside data_loader.py may contain
    hard date-based index slices like df.index < or df[df.index >.
    """
    def test_no_hardcoded_date_slices(self):
        import re
        from pathlib import Path
        src_dir = Path("src")
        pattern = re.compile(r"df\.index\s*[<>]")
        violations = []
        for py_file in src_dir.glob("*.py"):
            if py_file.name == "data_loader.py":
                continue
            text = py_file.read_text()
            for lineno, line in enumerate(text.splitlines(), 1):
                if pattern.search(line):
                    violations.append(f"{py_file}:{lineno}: {line.strip()}")
        assert not violations, (
            "Date slicing found outside data_loader.py:\n" + "\n".join(violations)
        )