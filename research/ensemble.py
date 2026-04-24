"""
Signal combiner — Person 3.

Merges outputs from the trend strategy and the ML model into a single
signal with dynamic confidence-weighted allocation.

The idea: only trade when both the rule-based signal and the ML model agree.
Disagreement = stay flat = avoid the trade. This kills win rate but
dramatically improves precision on the trades you do take.
"""
import pandas as pd

from research.features import atr, atr_pct
from research.risk import dynamic_leverage, stop_loss_price, take_profit_price, position_allocation

# ---------------------------------------------------------------------------
# Thresholds — tune these via walk_forward.py, NOT against holdout
# ---------------------------------------------------------------------------

ML_CONFIDENCE_THRESHOLD = 0.60   # baseline walk-forward threshold
                                  # Higher = fewer trades, higher precision
ML_STRONG_THRESHOLD = 0.80       # above this -> full allocation
                                  # Below this but above base → half allocation

ATR_PERIOD = 14


def combine(
    coin: str,
    df: pd.DataFrame,
    trend_signal: int,
    ml_prob_up: float,
    n_active_coins: int = 2,
) -> dict:
    """
    Combine a rule-based signal and ML probability into a final decision.

    trend_signal : +1, -1, or 0 from TrendStrategy or MeanRevertStrategy
    ml_prob_up   : probability that price goes UP (from model.predict_proba)
                   maps to: > 0.5 = ML thinks bullish, < 0.5 = ML thinks bearish
    n_active_coins: number of positions being opened this candle, used for
                    allocation sizing.

    Returns a complete decision dict ready to include in predict()'s return list.

    Callers should rank agreed signals and pass no more than the intended
    number of active positions into the portfolio.
    """
    close = df["Close"].iloc[-1]

    # No rule-based signal → don't trade regardless of ML
    if trend_signal == 0:
        return {"coin": coin, "signal": 0, "allocation": 0.0, "leverage": 1}
    if not 0.0 <= ml_prob_up <= 1.0:
        return {"coin": coin, "signal": 0, "allocation": 0.0, "leverage": 1}

    # ML direction: +1 if prob_up > 0.5, -1 if prob_up < 0.5
    ml_signal = 1 if ml_prob_up > 0.5 else -1 if ml_prob_up < 0.5 else 0
    ml_confidence = abs(ml_prob_up - 0.5) * 2  # rescale to [0, 1]

    # Agreement gate
    if ml_signal != trend_signal:
        # Signals disagree → stay flat
        return {"coin": coin, "signal": 0, "allocation": 0.0, "leverage": 1}

    if ml_confidence < ML_CONFIDENCE_THRESHOLD:
        # ML is too uncertain even though direction agrees
        return {"coin": coin, "signal": 0, "allocation": 0.0, "leverage": 1}

    current_atr = atr(df, ATR_PERIOD).iloc[-1]
    current_atr_pct = atr_pct(df, ATR_PERIOD).iloc[-1]
    if pd.isna(current_atr) or pd.isna(current_atr_pct) or current_atr <= 0:
        return {"coin": coin, "signal": 0, "allocation": 0.0, "leverage": 1}

    lev = dynamic_leverage(float(current_atr_pct))
    sl = stop_loss_price(close, trend_signal, float(current_atr))
    tp = take_profit_price(close, trend_signal, float(current_atr))

    if ml_confidence >= ML_STRONG_THRESHOLD:
        alloc_strength = 1.0
    else:
        alloc_strength = 0.5
    alloc = position_allocation(n_active_coins=n_active_coins, signal_strength=alloc_strength)

    return {
        "coin": coin,
        "signal": trend_signal,
        "allocation": alloc,
        "leverage": lev,
        "stop_loss": sl,
        "take_profit": tp,
    }


def flat(coin: str) -> dict:
    """Convenience — return a flat (no position) decision for a coin."""
    return {"coin": coin, "signal": 0, "allocation": 0.0, "leverage": 1}
