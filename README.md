# cmoon Trading Bot

Hackathon scaffold for a leakage-aware trading bot. The repository is organized so three developers can work in parallel while sharing one set of data contracts, time splits, and experiment rules.

## Repository Layout

```text
cmoon/
|-- config.yaml
|-- data/
|   |-- raw/
|   `-- processed/
|-- docs/
|   `-- architecture.md
|-- src/
|   |-- data_loader.py
|   |-- feature_engineering.py
|   |-- labeling.py
|   |-- cv_splitter.py
|   |-- model_training.py
|   |-- signal_generator.py
|   |-- risk_manager.py
|   |-- backtest_engine.py
|   |-- metrics.py
|   `-- pipeline.py
|-- tests/
|-- notebooks/
`-- artifacts/
```

## Non-Negotiables

- `data_loader.py` is the only place that owns chronological train, validation, and holdout boundaries.
- Competition symbols are discovered from raw data by default; no module may hardcode a coin name.
- Every market DataFrame uses a UTC-aware `DatetimeIndex`.
- Combined datasets use `symbol + timestamp` as the unique row identity.
- Feature columns are prefixed with `feat_`.
- Label columns are aligned to the same index and must drop rows where the forward target cannot be known.
- Scalers, feature selectors, and model tuning are fit on training folds only.
- Backtests fill at `next_bar.open` and account for fees, slippage, and funding where applicable.
- The local locked holdout is touched exactly once.

## Developer Ownership

| Owner | Slice | Modules |
|---|---|---|
| Dev A | Raw rows to features and labels | `data_loader.py`, `feature_engineering.py`, `labeling.py` |
| Dev B | Features to probabilities | `cv_splitter.py`, `model_training.py`, `signal_generator.py` |
| Dev C | Probabilities to P&L | `risk_manager.py`, `backtest_engine.py`, `metrics.py` |
| Shared | Integration | `pipeline.py`, `config.yaml`, contract tests |

## Workflow

1. Freeze the `config.yaml` schema and DataFrame contracts before implementation starts.
2. Discover and validate the available coin universe from `data/raw/`.
3. Build each pipeline slice against synthetic or stubbed upstream data.
4. Merge only when contract tests pass.
5. Tune only through purged walk-forward CV.
6. Freeze the final config and model before the final retrain.

See [docs/architecture.md](docs/architecture.md) for the full module contract and hackathon plan.
