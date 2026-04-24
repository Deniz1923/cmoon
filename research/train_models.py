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

from cnlib.base_strategy import BaseStrategy, COINS
from research.ml_features import build_X_y, feature_names

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Train/test split — NEVER tune hyperparameters against TEST set
TRAIN_END_CANDLE = 1100   # candles 0–1099 = train, 1100–1569 = test (holdout)


def load_all_data() -> dict[str, pd.DataFrame]:
    class _Loader(BaseStrategy):
        def predict(self, data):
            return []
    s = _Loader()
    s.get_data()
    return s.coin_data


def train_coin_model(
    coin: str,
    coin_data: dict[str, pd.DataFrame],
) -> tuple[object, np.ndarray, np.ndarray]:
    """
    Train a model for one coin.
    Returns (fitted_model, X_test, y_test) — caller evaluates test performance.
    """
    X, y, valid_index = build_X_y(coin_data, coin)

    # Time-series split — no shuffling
    train_mask = valid_index < valid_index[min(TRAIN_END_CANDLE, len(valid_index) - 1)]
    X_train, y_train = X[train_mask],  y[train_mask]
    X_test,  y_test  = X[~train_mask], y[~train_mask]

    print(f"\n{coin}")
    print(f"  Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"  Train class balance: {y_train.mean():.2%} positive")

    # TODO: tune hyperparameters via TimeSeriesSplit cross-validation on train set only
    #       from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
    model = RandomForestClassifier(
        n_estimators=200,    # TODO: try 100, 200, 500
        max_depth=5,         # TODO: try 3, 5, 8, None — deeper = more overfit risk
        min_samples_leaf=20, # TODO: higher = more regularization
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    train_acc = (model.predict(X_train) == y_train).mean()
    test_acc  = (model.predict(X_test)  == y_test).mean()
    print(f"  Train accuracy: {train_acc:.2%}  Test accuracy: {test_acc:.2%}")
    # TODO: also print precision/recall/F1 — accuracy is misleading on imbalanced data
    #       from sklearn.metrics import classification_report
    #       print(classification_report(y_test, model.predict(X_test)))

    return model, X_test, y_test


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
        model, X_test, y_test = train_coin_model(coin, coin_data)
        save_model(model, coin)
        models[coin] = model

    save_feature_importance(models, coin_data)

    print("\nDone. Load models in strategy.py with train_models.load_model(coin).")
