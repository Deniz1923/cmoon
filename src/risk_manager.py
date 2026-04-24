"""
risk_manager.py — Dev C
Convert trading signals into sized position orders with SL/TP.

Baseline rules:
- Vol-targeted position sizing: position size inversely scales with realized volatility.
- Max single-position equity fraction cap.
- ATR-scaled stop-loss and take-profit.
- Portfolio-level drawdown kill-switch check (see apply_kill_switch).
"""
import numpy as np
import pandas as pd


def size_positions(
    signals: pd.DataFrame,
    equity: float,
    ohlcv: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    Convert signals to position orders with vol-targeted sizing.

    Parameters
    ----------
    signals : DataFrame with 'signal' and 'confidence' columns
    equity  : current portfolio equity in base currency
    ohlcv   : OHLCV DataFrame available up to the decision timestamp
    config  : full config dict (reads config["risk"])

    Returns
    -------
    DataFrame indexed like signals with columns:
        signal, position_size, stop_loss_price, take_profit_price, max_hold_bars
    """
    cfg = config["risk"]
    target_vol = cfg.get("target_annual_volatility", 0.15)
    max_pos_frac = cfg.get("max_single_position_equity_fraction", 0.20)
    max_leverage = cfg.get("max_leverage", 1.0)
    sl_mult = cfg.get("stop_loss_atr_multiple", 1.0)
    tp_mult = cfg.get("take_profit_atr_multiple", 2.0)
    max_hold = cfg.get("max_hold_bars", 24)

    atr = _atr(ohlcv["high"], ohlcv["low"], ohlcv["close"], 14)
    rvol = _realized_vol_annual(ohlcv["close"], window=24)

    rows = []
    for ts in signals.index:
        sig = int(signals.loc[ts, "signal"])

        if ts not in ohlcv.index or sig == 0:
            rows.append({
                "signal": 0,
                "position_size": 0.0,
                "stop_loss_price": np.nan,
                "take_profit_price": np.nan,
                "max_hold_bars": max_hold,
            })
            continue

        price = float(ohlcv.loc[ts, "close"])
        atr_val = float(atr.get(ts, price * 0.01))
        vol_est = float(rvol.get(ts, target_vol))

        if vol_est <= 0 or price <= 0:
            pos_dollars = 0.0
        else:
            # Vol-targeting: scale so that 1 ATR move corresponds to target_vol / sqrt(365*24)
            hourly_target = target_vol / np.sqrt(365 * 24)
            pos_dollars = equity * hourly_target / (atr_val / price + 1e-9)

        # Cap by max fraction * leverage
        pos_dollars = min(pos_dollars, equity * max_pos_frac * max_leverage)
        pos_size = (pos_dollars / price) * sig  # negative for short

        if sig == 1:
            sl_price = price - sl_mult * atr_val
            tp_price = price + tp_mult * atr_val
        else:
            sl_price = price + sl_mult * atr_val
            tp_price = price - tp_mult * atr_val

        rows.append({
            "signal": sig,
            "position_size": pos_size,
            "stop_loss_price": sl_price,
            "take_profit_price": tp_price,
            "max_hold_bars": max_hold,
        })

    return pd.DataFrame(rows, index=signals.index)


def apply_kill_switch(equity_curve: pd.Series, config: dict) -> bool:
    """Return True if the drawdown kill-switch threshold was breached."""
    threshold = config["risk"].get("kill_switch_drawdown", 0.15)
    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak
    return bool((drawdown < -threshold).any())


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _atr(high, low, close, period=14) -> pd.Series:
    tr = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


def _realized_vol_annual(close: pd.Series, window: int = 24) -> pd.Series:
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(window).std() * np.sqrt(365 * 24)