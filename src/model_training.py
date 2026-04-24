"""
model_training.py — Dev B
Fit ML models and produce out-of-fold predictions.

Rules:
- Never calls data_loader or labeling.
- Receives pre-split, pre-scaled data.
- Supports 'rf' and 'xgb' via a unified train() interface.
- Hyperparameter tuning uses only training folds.
"""
import pickle
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


def train(
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits,
    model_type: str = "rf",
    params: Optional[dict] = None,
    config: Optional[dict] = None,
) -> tuple:
    """
    Train a model with purged CV and return the final model + OOF predictions.

    Parameters
    ----------
    X          : feature DataFrame (pre-scaled, aligned to y)
    y          : target Series with values in {-1, 0, 1}
    cv_splits  : iterable of (train_idx, val_idx) from cv_splitter
    model_type : 'rf' or 'xgb'
    params     : dict of hyperparameter overrides
    config     : full config dict (used for class_weight, random_state)

    Returns
    -------
    (fitted_model, oof_preds_df)
    oof_preds_df has columns [-1, 0, 1] with probabilities for each class.
    """
    if params is None:
        params = {}

    classes = [-1, 0, 1]
    oof_preds = pd.DataFrame(np.nan, index=y.index, columns=classes, dtype=float)

    for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
        X_tr = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_tr = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        mask_tr = ~(X_tr.isna().any(axis=1) | y_tr.isna())
        mask_val = ~(X_val.isna().any(axis=1) | y_val.isna())

        if mask_tr.sum() == 0 or mask_val.sum() == 0:
            continue

        model = _build_model(model_type, params, config)
        model.fit(X_tr[mask_tr], y_tr[mask_tr])

        probs = model.predict_proba(X_val[mask_val])
        model_classes = list(model.classes_)

        for cls in classes:
            if cls in model_classes:
                col_idx = model_classes.index(cls)
                oof_preds.loc[X_val[mask_val].index, cls] = probs[:, col_idx]

        val_classes_present = sorted(y_val[mask_val].unique())
        if len(val_classes_present) > 1:
            fold_loss = log_loss(y_val[mask_val], probs, labels=model_classes)
            print(f"  Fold {fold_idx}: log_loss={fold_loss:.4f}  "
                  f"n_train={mask_tr.sum()}  n_val={mask_val.sum()}")

    # Final model fit on all non-NaN training data
    mask_all = ~(X.isna().any(axis=1) | y.isna())
    final_model = _build_model(model_type, params, config)
    final_model.fit(X[mask_all], y[mask_all])

    return final_model, oof_preds


def predict_proba(model, X: pd.DataFrame) -> pd.DataFrame:
    """Return class probability DataFrame aligned to X's index."""
    mask = ~X.isna().any(axis=1)
    probs = pd.DataFrame(np.nan, index=X.index, columns=model.classes_, dtype=float)
    if mask.sum() > 0:
        probs.loc[mask] = model.predict_proba(X[mask])
    return probs


def save_model(model, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _build_model(model_type: str, params: dict, config: Optional[dict]):
    model_cfg = (config or {}).get("model", {})
    label_cfg = (config or {}).get("labels", {})
    random_state = model_cfg.get("random_state", 42)
    class_weight = label_cfg.get("class_weight", "balanced")

    if model_type == "rf":
        defaults = dict(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=20,
            max_features="sqrt",
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=-1,
        )
        defaults.update(params)
        return RandomForestClassifier(**defaults)

    if model_type == "xgb":
        if not HAS_XGB:
            raise ImportError("xgboost not installed. Run: uv add xgboost")
        defaults = dict(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=20,
            random_state=random_state,
            eval_metric="mlogloss",
        )
        defaults.update(params)
        return xgb.XGBClassifier(**defaults)

    raise ValueError(f"Unknown model_type '{model_type}'. Choose 'rf' or 'xgb'.")