"""
Final submission strategy — this is the only file that gets submitted.

All imports must work without the research/ folder (the judge runs just this file).
When ready to submit: inline everything from research/ that you need here,
or verify that the judge environment has the same packages installed.

Architecture:
  1. __init__: load pre-trained ML models (if using) + set up state
  2. predict(): compute indicators → trend signal → ML filter → combine → return
"""
from cnlib.base_strategy import BaseStrategy

COINS = ["kapcoin-usd_train", "metucoin-usd_train", "tamcoin-usd_train"]


class MyStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()

        # TODO (Person 1): initialize state to persist between candles
        # e.g. self._open_signals = {c: 0 for c in COINS}

        # TODO (Person 3): load pre-trained ML models here (NOT in predict)
        # from research.train_models import load_model
        # self.models = {coin: load_model(coin) for coin in COINS}
        # NOTE: train and save models first: python research/train_models.py

    def predict(self, data: dict) -> list[dict]:
        # TODO: replace placeholder with real ensemble logic once research/ is done
        #
        # Rough structure:
        #   decisions = []
        #   for coin in COINS:
        #       df = data[coin]
        #       if len(df) < MIN_CANDLES:
        #           decisions.append({"coin": coin, "signal": 0, "allocation": 0.0, "leverage": 1})
        #           continue
        #       trend_signal = self._trend_signal(coin, df)   # Person 2
        #       ml_prob      = self._ml_prob(coin, df)        # Person 3
        #       decision     = self._combine(coin, df, trend_signal, ml_prob)
        #       decisions.append(decision)
        #   return decisions

        return [
            {"coin": "kapcoin-usd_train",  "signal": 0, "allocation": 0.0, "leverage": 1},
            {"coin": "metucoin-usd_train", "signal": 0, "allocation": 0.0, "leverage": 1},
            {"coin": "tamcoin-usd_train",  "signal": 0, "allocation": 0.0, "leverage": 1},
        ]

    # TODO (Person 2): paste _trend_signal() logic here from trend_strategy.py
    # def _trend_signal(self, coin: str, df) -> int: ...

    # TODO (Person 3): paste _ml_prob() and _combine() logic here from ensemble.py
    # def _ml_prob(self, coin: str, df) -> float: ...
    # def _combine(self, coin, df, trend_signal, ml_prob) -> dict: ...