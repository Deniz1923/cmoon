"""
Mean-reversion strategy — fires in ranging (non-trending) markets.

Logic:
  1. Only trade when BB width is below threshold (ranging regime)
  2. Long when RSI oversold AND price near lower BB
  3. Short when RSI overbought AND price near upper BB
  4. Lower leverage (2x max) — mean reversion profits are smaller, survive chop

This file is runnable standalone:
  python research/mean_revert_strategy.py
"""
import pandas as pd
from cnlib.base_strategy import BaseStrategy
from cnlib import backtest

from research.features import (
    rsi, bb_bands, bb_pct, bb_width, atr, atr_pct, COINS,
)
from research.risk import dynamic_leverage, stop_loss_price, take_profit_price, position_allocation

# ---------------------------------------------------------------------------
# Tunable parameters
# ---------------------------------------------------------------------------

RSI_PERIOD = 14
RSI_OVERSOLD = 35        # TODO: try 25, 30, 35, 40
RSI_OVERBOUGHT = 65      # TODO: try 60, 65, 70, 75
BB_PERIOD = 20
BB_PCTB_LOW = 0.2        # price in lower 20% of bands → long candidate
BB_PCTB_HIGH = 0.8       # price in upper 80% of bands → short candidate
BB_RANGE_THRESHOLD = 0.06  # bb_width below this = ranging market
ATR_PERIOD = 14
ATR_MULTIPLIER = 1.5     # tighter stops for mean reversion (quicker reversals expected)
RISK_REWARD = 1.5        # 1.5:1 — mean reversion targets are smaller
MAX_LEVERAGE = 2         # hard cap; mean reversion is inherently counter-trend
MIN_CANDLES = BB_PERIOD + RSI_PERIOD + 5


class MeanRevertStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()
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

        # Mean reversion: one coin at a time, half allocation — counter-trend is higher risk
        active = candidates[:1]
        n_active = len(active)
        for dec in active:
            dec["allocation"] = position_allocation(n_active_coins=max(n_active, 1), signal_strength=0.5)

        for dec in candidates[1:]:
            dec["signal"] = 0
            dec["allocation"] = 0.0
            flat_decisions.append(dec)

        decision_map = {d["coin"]: d for d in flat_decisions + active}
        return [decision_map[coin] for coin in COINS]

    def _decide(self, coin: str, df: pd.DataFrame) -> dict:
        current_rsi = rsi(df["Close"], RSI_PERIOD).iloc[-1]
        current_bb_pct = bb_pct(df["Close"], BB_PERIOD).iloc[-1]
        current_bw = bb_width(df["Close"], BB_PERIOD).iloc[-1]
        current_atr = atr(df, ATR_PERIOD).iloc[-1]
        current_atr_pct = atr_pct(df, ATR_PERIOD).iloc[-1]

        # Regime gate — only trade in ranging markets
        if pd.isna(current_bw) or current_bw > BB_RANGE_THRESHOLD:
            self._open_signals[coin] = 0
            return {"coin": coin, "signal": 0, "allocation": 0.0, "leverage": 1}

        # Guard against bad indicator values
        if pd.isna(current_rsi) or pd.isna(current_bb_pct):
            self._open_signals[coin] = 0
            return {"coin": coin, "signal": 0, "allocation": 0.0, "leverage": 1}

        # Entry signals — both RSI and BB position must agree
        long_signal  = current_rsi < RSI_OVERSOLD  and current_bb_pct < BB_PCTB_LOW
        short_signal = current_rsi > RSI_OVERBOUGHT and current_bb_pct > BB_PCTB_HIGH

        if long_signal:
            signal = 1
        elif short_signal:
            signal = -1
        else:
            # Conditions no longer hold — close any open position
            self._open_signals[coin] = 0
            return {"coin": coin, "signal": 0, "allocation": 0.0, "leverage": 1}

        if pd.isna(current_atr) or current_atr <= 0 or pd.isna(current_atr_pct):
            self._open_signals[coin] = 0
            return {"coin": coin, "signal": 0, "allocation": 0.0, "leverage": 1}

        leverage = min(dynamic_leverage(float(current_atr_pct)), MAX_LEVERAGE)
        entry = float(df["Close"].iloc[-1])
        sl = stop_loss_price(entry, signal, float(current_atr), ATR_MULTIPLIER)
        tp = take_profit_price(entry, signal, float(current_atr), RISK_REWARD, ATR_MULTIPLIER)
        alloc = position_allocation(n_active_coins=1, signal_strength=0.5)  # 45% — counter-trend risk

        self._open_signals[coin] = signal
        return {
            "coin": coin,
            "signal": signal,
            "allocation": alloc,
            "leverage": leverage,
            "stop_loss": sl,
            "take_profit": tp,
        }


if __name__ == "__main__":
    result = backtest.run(MeanRevertStrategy(), start_candle=MIN_CANDLES, silent=False)
    result.print_summary()
