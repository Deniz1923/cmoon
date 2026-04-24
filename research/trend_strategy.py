"""
Trend-following strategy — Person 2's primary deliverable.

Logic:
  1. Detect regime (trending vs ranging) via BB width
  2. In trending regime: follow EMA crossover signal (long or short)
  3. In ranging regime: stay flat (let mean_revert_strategy.py handle it)
  4. Size each position using ATR-based leverage + stop loss

This file is runnable standalone for quick iteration:
  python research/trend_strategy.py
"""
import pandas as pd
from cnlib.base_strategy import BaseStrategy
from cnlib import backtest

from research.features import (
    ema, atr, atr_pct, bb_width, COINS,
    # TODO: import anything else you need from features.py
)
from research.risk import dynamic_leverage, stop_loss_price, take_profit_price, position_allocation

# ---------------------------------------------------------------------------
# Tunable parameters — tweak these, validate with walk_forward.py
# ---------------------------------------------------------------------------

EMA_FAST = 20          # TODO: try 10, 15, 20
EMA_SLOW = 50          # TODO: try 30, 50, 100
ATR_PERIOD = 14
ATR_MULTIPLIER = 2.0   # stop loss distance in ATR units
RISK_REWARD = 2.0      # take profit = RISK_REWARD × stop distance
BB_TREND_THRESHOLD = 0.08   # TODO: calibrate from EDA; bb_width above this = trending
MIN_CANDLES = EMA_SLOW + 5  # warm-up: don't trade until we have enough history


class TrendStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()
        # Track what signal is currently open per coin so we know when to hold vs flip
        self._open_signals: dict[str, int] = {c: 0 for c in COINS}

    def predict(self, data: dict) -> list[dict]:
        decisions = []

        for coin in COINS:
            df = data[coin]

            # Not enough data for indicators yet — stay flat
            if len(df) < MIN_CANDLES:
                decisions.append({"coin": coin, "signal": 0, "allocation": 0.0, "leverage": 1})
                continue

            decision = self._decide(coin, df)
            decisions.append(decision)

        return decisions

    def _decide(self, coin: str, df: pd.DataFrame) -> dict:
        # TODO: implement trend logic
        #
        # Step 1: compute indicators on df["Close"]
        #   fast = ema(df["Close"], EMA_FAST)
        #   slow = ema(df["Close"], EMA_SLOW)
        #   current_atr = atr(df, ATR_PERIOD).iloc[-1]
        #   current_atr_pct = atr_pct(df, ATR_PERIOD).iloc[-1]
        #   current_bw = bb_width(df["Close"]).iloc[-1]
        #
        # Step 2: regime check
        #   if current_bw < BB_TREND_THRESHOLD → not trending → signal = 0
        #
        # Step 3: trend signal
        #   signal = +1 if fast[-1] > slow[-1] else -1
        #
        # Step 4: if signal matches self._open_signals[coin] → hold (re-state same signal, no TP/SL update)
        #          if signal differs → flip (new TP/SL will be set by open_position)
        #
        # Step 5: build return dict
        #   if signal == 0:
        #       return {"coin": coin, "signal": 0, "allocation": 0.0, "leverage": 1}
        #   else:
        #       lev = dynamic_leverage(current_atr_pct)
        #       entry = df["Close"].iloc[-1]
        #       sl = stop_loss_price(entry, signal, current_atr, ATR_MULTIPLIER)
        #       tp = take_profit_price(entry, signal, current_atr, RISK_REWARD, ATR_MULTIPLIER)
        #       alloc = position_allocation(n_active_coins=2)  # TODO: count actual active coins
        #       self._open_signals[coin] = signal
        #       return {"coin": coin, "signal": signal, "allocation": alloc, "leverage": lev,
        #               "stop_loss": sl, "take_profit": tp}

        raise NotImplementedError("_decide() not implemented yet")


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    result = backtest.run(TrendStrategy(), start_candle=MIN_CANDLES, silent=False)
    result.print_summary()

    # TODO: save equity curve to results/
    # df = result.portfolio_dataframe()
    # df.to_csv("results/trend_equity.csv", index=False)
