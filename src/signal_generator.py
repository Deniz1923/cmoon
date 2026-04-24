"""
signal_generator.py — Dev B
Convert model probability outputs to directional trading signals.

Thresholds come from config only — never hardcoded here.
Calibration, if used, must be fit on training folds only.
"""
import numpy as np
import pandas as pd


def generate_signals(probs_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Threshold model probabilities into {-1, 0, +1} signals.

    Parameters
    ----------
    probs_df : DataFrame with columns corresponding to class labels.
               Expected columns: -1, 0, 1 (integers or numeric).
    config   : full config dict (reads config["signals"])

    Returns
    -------
    DataFrame with columns:
        signal     : int in {-1, 0, 1}
        confidence : float in [0, 1]
    """
    sig_cfg = config["signals"]
    long_thresh = sig_cfg.get("long_threshold", 0.60)
    short_thresh = sig_cfg.get("short_threshold", 0.40)

    signal = pd.Series(0, index=probs_df.index, dtype=int, name="signal")
    confidence = pd.Series(0.0, index=probs_df.index, name="confidence")

    if 1 in probs_df.columns:
        long_mask = probs_df[1] >= long_thresh
        signal[long_mask] = 1
        confidence[long_mask] = probs_df.loc[long_mask, 1]

    if -1 in probs_df.columns:
        short_mask = probs_df[-1] >= short_thresh
        signal[short_mask] = -1
        confidence[short_mask] = probs_df.loc[short_mask, -1]

    # Resolve conflicts: if both long and short thresholds triggered → go flat
    if 1 in probs_df.columns and -1 in probs_df.columns:
        conflict = (probs_df[1] >= long_thresh) & (probs_df[-1] >= short_thresh)
        signal[conflict] = 0
        confidence[conflict] = 0.0

    return pd.DataFrame({"signal": signal, "confidence": confidence})