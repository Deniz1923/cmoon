# Hackathon Trading Bot Architecture

This document is the team contract for the hackathon build. It is intentionally implementation-free: it defines ownership, inputs, outputs, and overfitting guardrails.

## Guiding Principles

- One source of truth for time splits: only `src/data_loader.py` may define train, validation, embargo, and holdout boundaries.
- Stateless modules with explicit DataFrame contracts.
- Every market DataFrame uses a UTC-aware `DatetimeIndex`.
- Configuration lives in `config.yaml`; modules read configuration rather than hardcoding paths, dates, symbols, thresholds, or model settings.
- Tests focus on contracts first: schemas, index alignment, no forward-looking leakage, and realistic backtest fill rules.

## Module Contracts

### `src/data_loader.py`

Owner: Dev A

Purpose: Load raw market data, normalize it, validate chronology, and create time splits.

Inputs:
- `symbol`
- `market_type`
- `config`

Outputs:
- OHLCV DataFrame indexed by UTC-aware `DatetimeIndex`
- Required columns: `open`, `high`, `low`, `close`, `volume`
- Futures-only optional columns: `funding_rate`, `open_interest`

Invariants:
- No NaN rows in required OHLCV columns
- Strictly monotonic increasing index
- No duplicate timestamps
- No local test or hidden out-of-sample access

### `src/feature_engineering.py`

Owner: Dev A

Purpose: Build leakage-free features from normalized market data.

Inputs:
- OHLCV DataFrame from `data_loader`

Outputs:
- Feature DataFrame aligned to the same index
- All feature columns use the `feat_` prefix

Rules:
- Rolling calculations must use only information available strictly before time `t`.
- Scalers are fit on training data only, then reused for validation and holdout transforms.
- Feature families include price, trend, momentum, volatility, volume, and futures-only features when data exists.

### `src/labeling.py`

Owner: Dev A

Purpose: Build supervised targets aligned with the strategy's intended risk logic.

Inputs:
- OHLCV DataFrame
- Labeling section of `config.yaml`

Outputs:
- Label DataFrame or Series aligned to the feature index
- Classification labels use `-1`, `0`, and `1`
- Rows without enough future horizon are dropped

Preferred method:
- Triple-barrier labeling with take-profit, stop-loss, and max-horizon settings.

### `src/cv_splitter.py`

Owner: Dev B

Purpose: Provide purged walk-forward cross-validation splits.

Inputs:
- Training DataFrame or aligned index
- Fold count
- Embargo length

Outputs:
- Generator of train and validation index arrays

Rules:
- Validation folds must be chronologically forward of their corresponding training window.
- A purge gap at least as large as the label horizon separates training and validation.
- Random k-fold is not allowed for time-series model selection.

### `src/model_training.py`

Owner: Dev B

Purpose: Train models and produce out-of-fold predictions without owning time logic.

Inputs:
- Aligned feature matrix
- Aligned target
- Model section of `config.yaml`
- CV splitter output

Outputs:
- Fitted model artifact
- Out-of-fold predictions aligned to the feature index
- Fold-level experiment metrics

Rules:
- Never calls `data_loader.py` or `labeling.py`.
- Hyperparameter tuning uses training folds only.
- Supported baseline models: random forest first, XGBoost only after the baseline contract is green.

### `src/signal_generator.py`

Owner: Dev B

Purpose: Convert model outputs into trading intents.

Inputs:
- Probability or regression prediction DataFrame
- Signal thresholds from `config.yaml`

Outputs:
- Signal DataFrame with `signal` in `-1`, `0`, `1`
- `confidence` in `[0, 1]`

Rules:
- Thresholds come from config.
- Calibration, if used, is fit on training folds only.

### `src/risk_manager.py`

Owner: Dev C

Purpose: Convert signals into executable position instructions.

Inputs:
- Signals
- Portfolio state
- OHLCV data available up to the decision point
- Risk section of `config.yaml`

Outputs:
- Orders or target positions with position size, stop loss, take profit, and max hold bars

Baseline rules:
- Vol-targeted sizing
- Max single-position exposure cap
- ATR-scaled stop loss and take profit
- Drawdown kill switch

### `src/backtest_engine.py`

Owner: Dev C

Purpose: Simulate trades bar by bar under realistic execution assumptions.

Inputs:
- OHLCV DataFrame
- Orders or target positions
- Backtest section of `config.yaml`

Outputs:
- Trades DataFrame
- Equity curve Series

Rules:
- Fill at next bar open.
- Apply fees and slippage.
- Apply futures funding costs when enabled.
- Decision functions may receive only data available up to the current timestamp.

### `src/metrics.py`

Owner: Dev C

Purpose: Evaluate strategy behavior and produce reporting values.

Inputs:
- Equity curve
- Trades DataFrame

Outputs:
- Metrics dictionary with Sharpe, Sortino, max drawdown, Calmar, win rate, average trade, turnover, and trade count.
- Optional tear-sheet plot output.

### `src/pipeline.py`

Owner: Shared

Purpose: Wire all modules together for a complete experiment run.

Inputs:
- `config.yaml`

Outputs:
- Dataset artifact
- Model artifact
- Prediction artifact
- Backtest report
- Metrics summary

Rules:
- Pipeline coordinates modules but does not contain core business logic.
- Pipeline is the only integration entrypoint.

## Parallel Work Plan

### Dev A: Data Plumber

Owns raw rows through features and labels.

Deliverables:
- Normalized OHLCV loading contract
- Chronological split contract with embargo
- Initial feature set
- Initial triple-barrier or horizon label contract
- `artifacts/dataset_v1.parquet` once implementation begins

### Dev B: Modeler

Owns features through probabilities and signals.

Deliverables:
- Purged walk-forward CV
- Random forest baseline
- Out-of-fold probability artifact
- Fold-level experiment log
- Thresholded signal artifact

### Dev C: Execution and Evaluation

Owns probabilities through P&L.

Deliverables:
- Risk sizing contract
- Bar-by-bar backtest contract
- Metrics dictionary
- Dummy-signal backtest before ML is ready
- Equity curve and tear-sheet artifacts

## Roadmap

### Phase 1: MVP

Goal: produce one complete validation run.

Exit criterion:
- Pipeline run produces Sharpe, drawdown, and an equity curve plot.
- Any implausibly high Sharpe triggers a leakage audit before optimization.

### Phase 2: Optimization and Risk

Goal: improve validation performance only through purged CV.

Allowed work:
- Expand features with stable fold importance.
- Add XGBoost through the same training interface.
- Improve risk sizing, stop-loss, take-profit, and funding cost handling.
- Use simple model averaging only if each model has positive CV evidence.

Revert condition:
- Validation Sharpe improves while fold dispersion widens materially.

### Phase 3: Out-of-Sample Simulation

Goal: simulate the locked holdout once.

Required checks:
- Fees doubled
- Slippage doubled
- Stop-loss and take-profit jittered by plus or minus 20%
- Top feature ablation
- Monthly P&L review

Final rule:
- Freeze `config.yaml` and the final model before retraining on all available non-hidden data.

## Integration Guardrails

- Branch by ownership slice: `feat/A-features`, `feat/B-model`, `feat/C-risk`.
- Schema changes after the initial freeze require explicit team sign-off.
- Contract tests must cover schema, index monotonicity, timezone awareness, feature prefixing, label alignment, and no unauthorized date slicing.
- No module outside `data_loader.py` may hardcode date-based splits.
