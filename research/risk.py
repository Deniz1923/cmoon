"""
Risk management utilities — position sizing, leverage selection, stop placement.

Rules this module enforces:
  - 10x leverage is NEVER used without a stop_loss tighter than liquidation
  - Leverage scales down as ATR% rises
  - Stop loss always placed relative to ATR, not a fixed %
"""
import pandas as pd

from research.features import atr, atr_pct


# ---------------------------------------------------------------------------
# Leverage selection
# ---------------------------------------------------------------------------

LEVERAGE_CHOICES = [1, 2, 3, 5, 10]


def dynamic_leverage(current_atr_pct: float) -> int:
    """
    Pick leverage based on current normalized volatility (ATR / Close).

    Liquidation thresholds:
      10x → 10% move wipes position (normal daily range ~8% → too risky)
       5x → 20% move
       3x → 33% move
       2x → 50% move
       1x → no liquidation risk

    TODO: backtest these thresholds against EDA results.
          Plot ATR% distribution per coin, pick breakpoints that keep
          expected liquidation probability < 1% per candle.
    """
    # TODO: fill in thresholds after EDA
    # rough starting point:
    if current_atr_pct < 0.03:
        return 5
    elif current_atr_pct < 0.06:
        return 3
    elif current_atr_pct < 0.10:
        return 2
    else:
        return 1


# ---------------------------------------------------------------------------
# Stop loss placement
# ---------------------------------------------------------------------------

def stop_loss_price(
    entry: float,
    direction: int,
    current_atr: float,
    atr_multiplier: float = 2.0,
) -> float:
    """
    Place stop loss at entry ± (atr_multiplier × ATR).

    direction: +1 = long (stop below entry), -1 = short (stop above entry)

    IMPORTANT: stop must be tighter than liquidation price.
    At 5x leverage, liquidation is 20% away — a 2×ATR stop at ~16% is fine.
    At 3x leverage, liquidation is 33% away — even more room.

    TODO: experiment with atr_multiplier values (1.5, 2.0, 2.5, 3.0).
          Tighter stops = more frequent exits but smaller losses per trade.
          Use walk_forward.py to find the best value per coin.
    """
    # TODO: validate that the stop is tighter than liquidation price
    #       raise ValueError if not (prevents silently running without protection)
    return entry - direction * current_atr * atr_multiplier


def take_profit_price(
    entry: float,
    direction: int,
    current_atr: float,
    risk_reward: float = 2.0,
    atr_multiplier: float = 2.0,
) -> float:
    """
    Take profit at entry ± (risk_reward × atr_multiplier × ATR).

    Default 2:1 risk/reward: if stop is 2×ATR away, TP is 4×ATR away.

    TODO: experiment with risk_reward values. Higher = fewer wins but bigger wins.
          In trending markets, letting winners run (3:1 or 4:1) may work better.
          In ranging markets, 1.5:1 may be more reliable.
    """
    return entry + direction * current_atr * atr_multiplier * risk_reward


# ---------------------------------------------------------------------------
# Allocation sizing
# ---------------------------------------------------------------------------

def position_allocation(
    n_active_coins: int,
    signal_strength: float = 1.0,
    max_total: float = 0.9,
) -> float:
    """
    Compute per-coin allocation fraction given how many positions are active.

    n_active_coins: how many coins you plan to trade this candle (1, 2, or 3)
    signal_strength: optional confidence weight from ML model [0.0–1.0]
    max_total: never allocate more than this fraction of equity in total

    TODO: decide whether to allow all 3 coins simultaneously.
          3 leveraged positions = high correlation risk if they all move together.
          Recommended: max 2 active coins at once.
    """
    # TODO: incorporate signal_strength to size up on high-confidence signals
    base = max_total / max(n_active_coins, 1)
    return round(min(base * signal_strength, max_total / n_active_coins), 4)
