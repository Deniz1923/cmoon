# cmoon — Code Night Trading Strategy

Leveraged long/short backtest strategy for the [Code Night](https://github.com/IYTE-Yazilim-Toplulugu/code-night-lib) algorithmic trading competition.

## Setup

```bash
git clone <repo-url>
cd cmoon
pip install cnlib
```

## Running a backtest

```bash
# Test the current submission strategy
python run.py

# Test individual research strategies
python run.py --strategy trend
python run.py --strategy meanrevert

# Options
python run.py --strategy trend --start 50 --plot --capital 3000
```

| Flag | Default | Description |
|---|---|---|
| `--strategy` | `main` | `main`, `trend`, `meanrevert`, `ensemble` |
| `--start` | `0` | Skip this many candles (warm-up) |
| `--capital` | `3000` | Starting capital |
| `--plot` | off | Save equity curve to `results/` (requires matplotlib) |
| `--silent` | off | Suppress per-candle progress output |

## Training ML models

ML models must be trained once before running the ensemble strategy:

```bash
python research/train_models.py
# → saves models to results/model_*.pkl
# → saves results/feature_importance.csv
```

## Project structure

```
cmoon/
├── strategy.py                   # Submission file — final ensemble
├── run.py                        # Backtest runner (see usage above)
│
├── research/
│   ├── features.py               # Shared indicator library (EMA, ATR, RSI, BB, …)
│   ├── risk.py                   # Leverage selection, stop/TP placement, sizing
│   ├── trend_strategy.py         # EMA crossover strategy (trending regime)
│   ├── mean_revert_strategy.py   # RSI + Bollinger Bands (ranging regime)
│   ├── ml_features.py            # Feature matrix builder for ML models
│   ├── train_models.py           # Train and save models to results/
│   ├── ensemble.py               # Combines trend signal + ML probability
│   └── walk_forward.py           # Validation: walk-forward + holdout test
│
└── results/                      # Generated — gitignored
    ├── model_*.pkl               # Trained model files
    ├── feature_importance.csv    # Feature importance from train_models.py
    └── *_equity.png              # Equity curve plots from --plot
```

## Strategy architecture

```
predict(data)
    │
    ├── features.py          compute EMA, ATR, RSI, BB width per coin
    │
    ├── trend_strategy.py    EMA crossover signal  (+1 / -1 / 0)
    │   └── risk.py          ATR-based leverage + stop loss price
    │
    ├── ml_features.py       build feature vector for current candle
    │   └── model.predict_proba()   P(price goes up over next N candles)
    │
    └── ensemble.py          only trade when trend + ML agree → final decision dict
```

### Regime switching

| BB Width | Regime | Active strategy |
|---|---|---|
| > 0.08 | Trending | `trend_strategy` — EMA crossover, leverage 3–5x |
| < 0.06 | Ranging | `mean_revert_strategy` — RSI + BB bands, leverage 2x max |
| Between | Ambiguous | Stay flat |

### Risk rules

- Stop loss set at position open via `stop_loss_price()` — never rely on `predict()` catching a reversal in time
- Leverage scales with ATR%: low volatility → 5x, high volatility → 1–2x
- Max 2 coins active simultaneously; total allocation ≤ 0.9
- 10x leverage is never used without a stop loss tighter than the liquidation threshold

### Liquidation thresholds (from engine)

| Leverage | Liquidation distance from entry |
|---|---|
| 10x | 10% |
| 5x | 20% |
| 3x | 33% |
| 2x | 50% |

Average daily range across all coins is ~8%, so **10x without a stop loss is effectively a coin flip on liquidation per candle.**

## Validation

Never tune parameters against the holdout set. Use walk-forward on the train window:

```python
from research.walk_forward import walk_forward, holdout_test
from research.trend_strategy import TrendStrategy

# Cross-validate on candles 0–1099 (train only)
walk_forward(TrendStrategy, n_splits=4)

# Run once at the very end
holdout_test(TrendStrategy())
```

| Window | Candles | Purpose |
|---|---|---|
| Train | 0 – 1099 | Parameter tuning, walk-forward CV |
| Holdout | 1100 – 1569 | Final evaluation only — do not peek |

## Team responsibilities

| Person | Owns | Files |
|---|---|---|
| Person 1 | Data & infrastructure | `features.py`, `walk_forward.py`, `run.py` |
| Person 2 | Quant strategies & risk | `risk.py`, `trend_strategy.py`, `mean_revert_strategy.py` |
| Person 3 | ML & ensemble | `ml_features.py`, `train_models.py`, `ensemble.py` |

Final `strategy.py` is assembled together by inlining the research pieces.
