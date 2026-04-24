from __future__ import annotations

import pandas as pd

from indicators import rsi, sma
from model_types import Signal, StrategyState
from strategy_base import BaseStrategy


class MyStrategy(BaseStrategy):
    """
    Yarismaci tarafinda duzenlenecek ana dosya.
    """

    name = "my-ifelse-strategy"

    def fit(self, train_data: dict[str, pd.DataFrame]) -> None:
        self.close_means = {coin: float(df["close"].pct_change().dropna().mean()) for coin, df in train_data.items()}

    def predict(
        self,
        current_window: dict[str, pd.DataFrame],
        state: StrategyState,
    ) -> list[Signal]:
        proposals: list[tuple[str, int]] = []
        hold_coins: list[str] = []
        allowed_leverages = tuple(sorted((self.rules.allowed_leverages if self.rules else (2, 3, 5, 10))))
        default_lev = allowed_leverages[0]
        trade_lev = 3 if 3 in allowed_leverages else allowed_leverages[-1]

        for coin, df in current_window.items():
            if len(df) < 60:
                hold_coins.append(coin)
                continue

            close = df["close"]
            fast = sma(close, 20).iloc[-1]
            slow = sma(close, 50).iloc[-1]
            rsi_v = rsi(close, 14).iloc[-1]

            if pd.isna(fast) or pd.isna(slow) or pd.isna(rsi_v):
                hold_coins.append(coin)
                continue

            if fast > slow and rsi_v < 70:
                proposals.append((coin, 1))
            elif fast < slow and rsi_v > 30:
                proposals.append((coin, -1))
            else:
                hold_coins.append(coin)

        if not proposals:
            return [Signal(coin=coin, signal=0, ratio=0.0, leverage=default_lev) for coin in current_window]

        max_total_ratio = self.rules.max_total_ratio if self.rules else 1.0
        max_ratio_per_coin = self.rules.max_ratio_per_coin if self.rules else 1.0
        ratio = min(max_ratio_per_coin, max_total_ratio / len(proposals))

        decisions: list[Signal] = [Signal(coin=coin, signal=signal, ratio=ratio, leverage=trade_lev) for coin, signal in proposals]
        decisions.extend(Signal(coin=coin, signal=0, ratio=0.0, leverage=default_lev) for coin in hold_coins)

        return decisions
