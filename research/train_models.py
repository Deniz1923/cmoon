"""
Train and persist ML models — Person 3.

Run this script once offline to produce saved model files.
The final strategy.py loads these at __init__ time (not in predict()).

Usage:
  python research/train_models.py

Outputs:
  results/model_kapcoin.pkl
  results/model_metucoin.pkl
  results/model_tamcoin.pkl
  results/feature_importance.csv   ← use this to prune bad features
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

# TODO: choose your model. Options:
#   from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
#   from xgboost import XGBClassifier   (pip install xgboost)
#   from lightgbm import LGBMClassifier (pip install lightgbm)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from cnlib.base_strategy import BaseStrategy, COINS
from research.ml_features import build_X_y, feature_names

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Train/test split — NEVER tune hyperparameters against TEST set
TRAIN_END_CANDLE = 1100   # candles 0–1099 = train, 1100–1569 = test (holdout)
PARAM_GRID = {
    "n_estimators": [100, 200, 500],
    "max_depth": [3, 5, 8, None],
    "min_samples_leaf": [5, 10, 20],
}


def load_all_data() -> dict[str, pd.DataFrame]:
    class _Loader(BaseStrategy):
        def predict(self, data):
            return []
    s = _Loader()
    s.get_data()
    return {coin: df.copy() for coin, df in s._full_data.items()}


def train_coin_model(
    coin: str,
    coin_data: dict[str, pd.DataFrame],
) -> tuple[object, np.ndarray, np.ndarray, dict]:
    """
    Train a model for one coin.
    Returns (fitted_model, X_test, y_test) — caller evaluates test performance.
    """
    X, y, valid_index = build_X_y(coin_data, coin)

    # Time-series split — no shuffling
    split_at = min(TRAIN_END_CANDLE, len(valid_index))
    X_train, y_train = X[:split_at], y[:split_at]
    X_test, y_test = X[split_at:], y[split_at:]

    print(f"\n{coin}")
    print(f"  Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"  Train class balance: {y_train.mean():.2%} positive")

    n_splits = min(4, max(len(X_train) // 150, 2))
    cv = TimeSeriesSplit(n_splits=n_splits)
    search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
        param_grid=PARAM_GRID,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    model = search.best_estimator_
    print(f"  Best params: {search.best_params_}")
    print(f"  Best CV F1: {search.best_score_:.4f}")

    train_pred = model.predict(X_train)
    train_prob = model.predict_proba(X_train)[:, 1]
    train_metrics = evaluate_predictions(y_train, train_pred, train_prob)
    print(
        "  Train metrics:"
        f" acc={train_metrics['accuracy']:.2%}"
        f" precision={train_metrics['precision']:.2%}"
        f" recall={train_metrics['recall']:.2%}"
        f" f1={train_metrics['f1']:.2%}"
        f" auc={train_metrics['roc_auc']:.4f}"
    )

    metrics = {}
    if len(X_test):
        test_pred = model.predict(X_test)
        test_prob = model.predict_proba(X_test)[:, 1]
        metrics = evaluate_predictions(y_test, test_pred, test_prob)
        print(
            "  Holdout metrics:"
            f" acc={metrics['accuracy']:.2%}"
            f" precision={metrics['precision']:.2%}"
            f" recall={metrics['recall']:.2%}"
            f" f1={metrics['f1']:.2%}"
            f" auc={metrics['roc_auc']:.4f}"
        )

    return model, X_test, y_test, metrics


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": np.nan,
    }
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    return metrics


def save_model(model, coin: str) -> Path:
    path = RESULTS_DIR / f"model_{coin.replace('-', '_')}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"  Saved → {path}")
    return path


def load_model(coin: str) -> object:
    path = RESULTS_DIR / f"model_{coin.replace('-', '_')}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def save_feature_importance(models: dict[str, object], coin_data: dict) -> None:
    """Save feature importance to CSV — use this to prune weak features."""
    rows = []
    for coin, model in models.items():
        names = feature_names(coin_data, coin)
        # TODO: handle models without feature_importances_ (e.g. logistic regression)
        for name, imp in zip(names, model.feature_importances_):
            rows.append({"coin": coin, "feature": name, "importance": imp})
    df = pd.DataFrame(rows).sort_values("importance", ascending=False)
    path = RESULTS_DIR / "feature_importance.csv"
    df.to_csv(path, index=False)
    print(f"\nFeature importance → {path}")
    print(df.groupby("feature")["importance"].mean().sort_values(ascending=False).to_string())


if __name__ == "__main__":
    print("Loading data...")
    coin_data = load_all_data()

    models = {}
    for coin in COINS:
        model, X_test, y_test, metrics = train_coin_model(coin, coin_data)
        save_model(model, coin)
        models[coin] = model

    save_feature_importance(models, coin_data)

    print("\nDone. Load models in strategy.py with train_models.load_model(coin).")
