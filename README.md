# Cmoon Competition Workspace

This repo is set up as an internal competitor workspace. The organizer only needs
`submission/strateji.py` and an optional model artifact; everything else exists so the
team can research, backtest, harden, and ship without coupling every experiment together.

## Quick Start

```bash
uv sync
uv run python hello.py
uv run python scripts/smoke.py
```

Raw organizer files go in `data/raw/` as:

```text
Varlik_A.parquet
Varlik_B.parquet
Varlik_C.parquet
```

Then run:

```bash
uv run python scripts/build_features.py
uv run python scripts/run_backtest.py
```

## Team Lanes

Person 1 owns signal research:

- `research/`
- `features/`
- `models/`

Person 2 owns backtest and risk:

- `backtest/`
- `sizing/`
- `config.py`
- `tests/`

Person 3 owns integration and submission:

- `strategies/`
- `submission/`
- `scripts/`

The handoff contract is intentionally small:

- feature code returns a pandas frame with stable feature columns
- model code returns a `RawSignal`
- sizing code turns `RawSignal` into `sinyal`, `oran`, and `kaldirac`
- strategy code always returns exactly three order dictionaries

## Competition Rules Guardrails

- `predict()` only sees history up to the current candle.
- `predict()` handles the first candle with no crash.
- Total `oran` across all three assets is capped at `1.0`.
- `kaldirac` is always one of `2`, `3`, `5`, or `10`.
- Runtime submission code should import only standard library, `numpy`, and `pandas`.

## Dependency Policy

Runtime dependencies stay lean: `numpy`, `pandas`, and `pyarrow`.
Research-only packages belong in the `research` uv group:

```bash
uv sync --group research
```

Add heavyweight modeling libraries only after a strategy proves itself in walk-forward
testing. Ruin by dependency sprawl is less dramatic than ruin by 10x leverage, but it is
still ruin.
