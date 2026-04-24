"""
Compatibility runner for the Person 3 ML path.

The maintained ML workflow lives in research/train_models.py and strategy.py.
This file remains as a short entrypoint so old notes that mention `ml.py` do
not point at a broken cnlib 0.1.4 training script.
"""
from cnlib import backtest

from strategy import MyStrategy


if __name__ == "__main__":
    result = backtest.run(MyStrategy(), initial_capital=3000.0, silent=True)
    result.print_summary()
