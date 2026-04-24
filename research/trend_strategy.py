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
        flat_decisions = []
        candidates = []

        for coin in COINS:
            df = data[coin]

            if len(df) < MIN_CANDLES:
                flat_decisions.append({"coin": coin, "signal": 0, "allocation": 0.0, "leverage": 1})
                continue

            decision = self._decide(coin, df)
            if decision["signal"] != 0:
                candidates.append(decision)
            else:
                flat_decisions.append(decision)

        # Max 2 active coins — set allocation based on actual active count
        active = candidates[:2]
        n_active = len(active)
        for dec in active:
            dec["allocation"] = position_allocation(n_active_coins=max(n_active, 1))

        # Coins that didn't get a slot go flat
        active_coins = {d["coin"] for d in active}
        for dec in candidates[2:]:
            dec["signal"] = 0
            dec["allocation"] = 0.0
            flat_decisions.append(dec)

        # Preserve COINS order
        decision_map = {d["coin"]: d for d in flat_decisions + active}
        return [decision_map[coin] for coin in COINS]

    def _decide(self, coin: str, df: pd.DataFrame) -> dict:
        fast = ema(df["Close"], EMA_FAST)
        slow = ema(df["Close"], EMA_SLOW)
        current_atr = atr(df, ATR_PERIOD).iloc[-1]
        current_atr_pct = atr_pct(df, ATR_PERIOD).iloc[-1]
        current_bw = bb_width(df["Close"]).iloc[-1]

        # Regime check — only trade in trending markets
        if pd.isna(current_bw) or current_bw < BB_TREND_THRESHOLD:
            return {"coin": coin, "signal": 0, "allocation": 0.0, "leverage": 1}

        # Guard against bad ATR values
        if pd.isna(current_atr) or current_atr <= 0 or pd.isna(current_atr_pct):
            return {"coin": coin, "signal": 0, "allocation": 0.0, "leverage": 1}

        # Trend signal from EMA crossover
        signal = 1 if fast.iloc[-1] > slow.iloc[-1] else -1

        lev = dynamic_leverage(float(current_atr_pct))
        entry = float(df["Close"].iloc[-1])
        if entry <= 0:
            return {"coin": coin, "signal": 0, "allocation": 0.0, "leverage": 1}
        sl = stop_loss_price(entry, signal, float(current_atr), ATR_MULTIPLIER)
        tp = take_profit_price(entry, signal, float(current_atr), RISK_REWARD, ATR_MULTIPLIER)
        if sl <= 0 or tp <= 0:
            return {"coin": coin, "signal": 0, "allocation": 0.0, "leverage": 1}
        alloc = position_allocation(n_active_coins=2)

        self._open_signals[coin] = signal
        return {
            "coin": coin,
            "signal": signal,
            "allocation": alloc,
            "leverage": lev,
            "stop_loss": sl,
            "take_profit": tp,
        }


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    result = backtest.run(TrendStrategy(), start_candle=MIN_CANDLES, silent=False)
    result.print_summary()

    # TODO: save equity curve to results/
    # df = result.portfolio_dataframe()
    # df.to_csv("results/trend_equity.csv", index=False)
