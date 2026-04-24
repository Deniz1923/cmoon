# Research Notebooks

Use this folder for notebooks and throwaway analysis.

Suggested flow:

1. `01_eda.ipynb` - distributions, missing candles, correlations, regimes
2. `02_feature_engineering.ipynb` - predictive value of indicators
3. `03_signal_research.ipynb` - model and ensemble experiments
4. `04_leverage_sizing.ipynb` - drawdown and leverage stress tests

Do not import notebook-only code from production modules. Promote useful logic into
`features/`, `models/`, `sizing/`, or `backtest/`.
