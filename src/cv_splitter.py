"""
cv_splitter.py — Dev B
Purged walk-forward cross-validation.

Standard k-fold leaks future information when time-series features/labels overlap.
This splitter ensures every validation fold is strictly after its training window,
with an embargo gap at least as large as the label horizon.
"""
from typing import Generator, Tuple

import numpy as np
import pandas as pd


def purged_walk_forward_cv(
    df: pd.DataFrame,
    n_folds: int,
    embargo_bars: int,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Generate (train_idx, val_idx) pairs for purged walk-forward CV.

    Each validation fold is one chronological chunk.
    The training window expands to include all data before the fold minus
    an embargo gap of `embargo_bars` bars.

    Parameters
    ----------
    df           : DataFrame whose integer positions form the index space
    n_folds      : number of CV folds
    embargo_bars : bars to remove from the end of each train window
                   (should equal at least the label horizon)

    Yields
    ------
    (train_idx, val_idx) as arrays of integer positions
    """
    n = len(df)
    fold_size = n // (n_folds + 1)

    for fold in range(n_folds):
        val_start = (fold + 1) * fold_size
        val_end = min(val_start + fold_size, n)
        train_end = val_start - embargo_bars

        if train_end <= 0 or val_start >= n:
            continue

        train_idx = np.arange(0, train_end)
        val_idx = np.arange(val_start, val_end)

        yield train_idx, val_idx