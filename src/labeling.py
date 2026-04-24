"""
labeling.py — Dev A
Create supervised targets using triple-barrier labeling.

Label = +1 if take-profit hit first
Label = -1 if stop-loss hit first
Label =  0 if time barrier hit without touching either price barrier
Rows where the label cannot be computed (last max_horizon_bars rows) are dropped.
"""
import numpy as np
import pandas as pd


def make_labels(ohlcv_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Triple-barrier labeling.

    Parameters
    ----------
    ohlcv_df : OHLCV DataFrame with UTC DatetimeIndex
    config   : full config dict (reads config["labels"])

    Returns
    -------
    DataFrame with a single 'target' column, same index as ohlcv_df minus
    the final max_horizon_bars rows (which are dropped).
    """
    cfg = config["labels"]
    pt = cfg.get("take_profit", 0.02)
    sl = cfg.get("stop_loss", 0.01)
    max_h = cfg.get("max_horizon_bars", 24)

    close = ohlcv_df["close"].values
    high = ohlcv_df["high"].values
    low = ohlcv_df["low"].values
    n = len(close)

    labels = np.full(n, np.nan, dtype=float)

    for i in range(n - max_h):
        entry = close[i]
        tp_price = entry * (1.0 + pt)
        sl_price = entry * (1.0 - sl)
        label = 0

        for j in range(1, max_h + 1):
            h = high[i + j]
            l = low[i + j]
            tp_hit = h >= tp_price
            sl_hit = l <= sl_price

            if tp_hit and sl_hit:
                # Both barriers in same bar → conservative: treat as stop-loss
                label = -1
                break
            if tp_hit:
                label = 1
                break
            if sl_hit:
                label = -1
                break

        labels[i] = label

    result = pd.Series(labels, index=ohlcv_df.index, name="target")
    result = result.dropna().astype(int)
    return result.to_frame()