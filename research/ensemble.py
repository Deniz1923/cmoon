"""
Signal combiner — Person 3.

Merges outputs from the trend strategy and the ML model into a single
signal with dynamic confidence-weighted allocation.

The idea: only trade when both the rule-based signal and the ML model agree.
Disagreement = stay flat = avoid the trade. This kills win rate but
dramatically improves precision on the trades you do take.
"""
import numpy as np
import pandas as pd

from research.features import atr, atr_pct, COINS
from research.risk import dynamic_leverage, stop_loss_price, take_profit_price, position_allocation

# ---------------------------------------------------------------------------
# Thresholds — tune these via walk_forward.py, NOT against holdout
# ---------------------------------------------------------------------------

ML_CONFIDENCE_THRESHOLD = 0.60   # TODO: try 0.55, 0.60, 0.65, 0.70
                                  # Higher = fewer trades, higher precision
ML_STRONG_THRESHOLD = 0.70       # TODO: above this → full allocation
                                  # Below this but above base → half allocation

ATR_PERIOD = 14


def combine(
    coin: str,
    df: pd.DataFrame,
    trend_signal: int,
    ml_prob_up: float,
) -> dict:
    """
    Combine a rule-based signal and ML probability into a final decision.

    trend_signal : +1, -1, or 0 from TrendStrategy or MeanRevertStrategy
    ml_prob_up   : probability that price goes UP (from model.predict_proba)
                   maps to: > 0.5 = ML thinks bullish, < 0.5 = ML thinks bearish

    Returns a complete decision dict ready to include in predict()'s return list.

    TODO: consider adding a third input — current portfolio exposure —
          to avoid stacking too many correlated positions.
    """
    close = df["Close"].iloc[-1]

    # No rule-based signal → don't trade regardless of ML
    if trend_signal == 0:
        return {"coin": coin, "signal": 0, "allocation": 0.0, "leverage": 1}

    # ML direction: +1 if prob_up > 0.5, -1 if prob_up < 0.5
    ml_signal = 1 if ml_prob_up > 0.5 else -1
    ml_confidence = abs(ml_prob_up - 0.5) * 2  # rescale to [0, 1]

    # Agreement gate
    if ml_signal != trend_signal:
        # Signals disagree → stay flat
        return {"coin": coin, "signal": 0, "allocation": 0.0, "leverage": 1}

    if ml_confidence < ML_CONFIDENCE_THRESHOLD - 0.5:
        # ML is too uncertain even though direction agrees
        return {"coin": coin, "signal": 0, "allocation": 0.0, "leverage": 1}

    # TODO: compute ATR for dynamic sizing
    # current_atr = atr(df, ATR_PERIOD).iloc[-1]
    # current_atr_pct = atr_pct(df, ATR_PERIOD).iloc[-1]
    # lev = dynamic_leverage(current_atr_pct)
    # sl = stop_loss_price(close, trend_signal, current_atr)
    # tp = take_profit_price(close, trend_signal, current_atr)

    # Confidence-scaled allocation
    # TODO: scale alloc by ml_confidence — stronger signal = bigger position
    # if ml_confidence >= ML_STRONG_THRESHOLD - 0.5:
    #     alloc = position_allocation(n_active_coins=2)
    # else:
    #     alloc = position_allocation(n_active_coins=2) * 0.5

    # TODO: return the full dict with SL/TP
    # return {
    #     "coin": coin, "signal": trend_signal,
    #     "allocation": alloc, "leverage": lev,
    #     "stop_loss": sl, "take_profit": tp,
    # }

    raise NotImplementedError("combine() not fully implemented")


def flat(coin: str) -> dict:
    """Convenience — return a flat (no position) decision for a coin."""
    return {"coin": coin, "signal": 0, "allocation": 0.0, "leverage": 1}
