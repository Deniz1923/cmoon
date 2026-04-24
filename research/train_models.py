"""
Train and persist ML models - Person 3.

Run this script offline to produce saved model bundles.
The final strategy.py loads these at __init__ time, not in predict().
"""
from __future__ import annotations

import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from cnlib.base_strategy import BaseStrategy, COINS
from research.ml_features import TARGET_HORIZON, build_X_y, feature_names

warnings.filterwarnings(
    "ignore",
    message="`sklearn.utils.parallel.delayed` should be used with",
    category=UserWarning,
)

MODEL_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_FORMAT_VERSION = 1
TRAIN_END_CANDLE = 1100
PARAM_GRID = {
    "n_estimators": [100, 200, 500],
    "max_depth": [3, 5, 8, None],
    "min_samples_leaf": [5, 10, 20],
}


def load_all_data() -> dict[str, pd.DataFrame]:
    class _Loader(BaseStrategy):
        def predict(self, data):
            return []

    strategy = _Loader()
    strategy.get_data()
    full = getattr(strategy, "_full_data", None) or strategy.coin_data
    return {coin: df.copy() for coin, df in full.items()}


def split_train_holdout(
    X: np.ndarray,
    y: np.ndarray,
    valid_index: pd.Index,
    train_end_candle: int = TRAIN_END_CANDLE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.Index, pd.Index]:
    """
    Split by original candle index, not by post-feature row count.

    Training rows are purged by TARGET_HORIZON so labels never reach into the
    holdout window. Rows in the purge gap are excluded from both splits.
    """
    index = pd.Index(valid_index)
    if len(X) != len(y) or len(X) != len(index):
        raise ValueError(
            "X, y, and valid_index must have the same length "
            f"(got X={len(X)}, y={len(y)}, index={len(index)})"
        )
    if not pd.api.types.is_numeric_dtype(index.dtype):
        raise ValueError("valid_index must contain numeric candle indices")

    candle_index = index.to_numpy()
    train_mask = candle_index + TARGET_HORIZON < train_end_candle
    holdout_mask = candle_index >= train_end_candle
    return (
        X[train_mask],
        y[train_mask],
        X[holdout_mask],
        y[holdout_mask],
        index[train_mask],
        index[holdout_mask],
    )


def train_coin_model(
    coin: str,
    coin_data: dict[str, pd.DataFrame],
) -> tuple[object, np.ndarray, np.ndarray, dict]:
    """
    Train a model for one coin.

    Returns (fitted_model, X_test, y_test, metrics). The model bundle is
    assembled by save_model() so load_model() can remain estimator-compatible.
    """
    X, y, valid_index = build_X_y(coin_data, coin)
    X_train, y_train, X_test, y_test, train_index, test_index = split_train_holdout(
        X,
        y,
        valid_index,
    )

    if len(X_train) < 3:
        raise ValueError(f"{coin}: not enough train samples before candle {TRAIN_END_CANDLE}")

    print(f"\n{coin}")
    print(f"  Train samples: {len(X_train)}, Holdout samples: {len(X_test)}")
    print(f"  Train candles: {int(train_index.min())}-{int(train_index.max())}")
    if len(test_index):
        print(f"  Holdout candles: {int(test_index.min())}-{int(test_index.max())}")
    print(f"  Train class balance: {y_train.mean():.2%} positive")

    n_splits = min(4, max(len(X_train) // 150, 2))
    cv = TimeSeriesSplit(n_splits=n_splits)
    search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
        param_grid=PARAM_GRID,
        scoring="f1",
        cv=cv,
        n_jobs=1,
        refit=True,
    )
    search.fit(X_train, y_train)
    model = search.best_estimator_

    train_pred = model.predict(X_train)
    train_prob = _positive_prob(model, X_train)
    train_metrics = evaluate_predictions(y_train, train_pred, train_prob)

    holdout_metrics = {}
    if len(X_test):
        test_pred = model.predict(X_test)
        test_prob = _positive_prob(model, X_test)
        holdout_metrics = evaluate_predictions(y_test, test_pred, test_prob)

    metrics = {
        "cv_best_params": search.best_params_,
        "cv_best_f1": float(search.best_score_),
        "train": train_metrics,
        "holdout": holdout_metrics,
        "train_samples": int(len(X_train)),
        "holdout_samples": int(len(X_test)),
        "train_index_start": int(train_index.min()),
        "train_index_end": int(train_index.max()),
        "holdout_index_start": int(test_index.min()) if len(test_index) else None,
        "holdout_index_end": int(test_index.max()) if len(test_index) else None,
    }

    print(f"  Best params: {search.best_params_}")
    print(f"  Best CV F1: {search.best_score_:.4f}")
    _print_metrics("Train metrics", train_metrics)
    if holdout_metrics:
        _print_metrics("Holdout metrics", holdout_metrics)

    return model, X_test, y_test, metrics


def _positive_prob(model: object, X: np.ndarray) -> np.ndarray:
    proba = model.predict_proba(X)
    classes = list(getattr(model, "classes_", []))
    if 1 in classes:
        return proba[:, classes.index(1)]
    if proba.shape[1] >= 2:
        return proba[:, 1]
    return np.zeros(len(X), dtype=float)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": np.nan,
    }
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    return metrics


def _print_metrics(label: str, metrics: dict) -> None:
    print(
        f"  {label}:"
        f" acc={metrics['accuracy']:.2%}"
        f" precision={metrics['precision']:.2%}"
        f" recall={metrics['recall']:.2%}"
        f" f1={metrics['f1']:.2%}"
        f" auc={metrics['roc_auc']:.4f}"
    )


def model_path(coin: str) -> Path:
    return MODEL_DIR / f"model_{coin.replace('-', '_')}.pkl"


def save_model(
    model: object,
    coin: str,
    *,
    names: list[str] | None = None,
    metrics: dict | None = None,
    horizon: int = TARGET_HORIZON,
    train_end_candle: int = TRAIN_END_CANDLE,
) -> Path:
    bundle = {
        "format_version": MODEL_FORMAT_VERSION,
        "coin": coin,
        "estimator": model,
        "feature_names": list(names or []),
        "horizon": horizon,
        "train_end_candle": train_end_candle,
        "metrics": metrics or {},
    }
    path = model_path(coin)
    path.parent.mkdir(exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"  Saved model bundle -> {path}")
    return path


def load_model_bundle(coin: str) -> dict:
    with open(model_path(coin), "rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, dict) and "estimator" in payload:
        return payload
    return {
        "format_version": 0,
        "coin": coin,
        "estimator": payload,
        "feature_names": [],
        "horizon": None,
        "train_end_candle": None,
        "metrics": {},
    }


def load_model(coin: str) -> object:
    """Backward-compatible estimator loader used by older strategy code."""
    return load_model_bundle(coin)["estimator"]


def save_feature_importance(models: dict[str, object], coin_data: dict) -> None:
    rows = []
    for coin, model in models.items():
        names = feature_names(coin_data, coin)
        importances = _importance_values(model, len(names))
        for name, imp in zip(names, importances):
            rows.append({"coin": coin, "feature": name, "importance": imp})

    df = pd.DataFrame(rows, columns=["coin", "feature", "importance"])
    if not df.empty:
        df = df.sort_values("importance", ascending=False, na_position="last")

    path = RESULTS_DIR / "feature_importance.csv"
    path.parent.mkdir(exist_ok=True)
    df.to_csv(path, index=False)
    print(f"\nFeature importance -> {path}")
    if not df.empty:
        print(df.groupby("feature")["importance"].mean().sort_values(ascending=False).to_string())


def _importance_values(model: object, n_features: int) -> np.ndarray:
    if hasattr(model, "feature_importances_"):
        return np.asarray(model.feature_importances_, dtype=float)
    if hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float)
        return np.abs(coef).reshape(-1, n_features).mean(axis=0)
    return np.full(n_features, np.nan)


if __name__ == "__main__":
    print("Loading data...")
    coin_data = load_all_data()

    models = {}
    for coin in COINS:
        model, X_test, y_test, metrics = train_coin_model(coin, coin_data)
        names = feature_names(coin_data, coin)
        save_model(model, coin, names=names, metrics=metrics)
        models[coin] = model

    save_feature_importance(models, coin_data)
    print("\nDone. Model bundles are ready for strategy.py.")
