"""
pipeline.py — Shared integration entrypoint
Wires all modules end-to-end for a reproducible experiment run.

Usage:
    python -m src.pipeline --config config.yaml
    python -m src.pipeline --config config.yaml --split holdout   # ⚠ one-time only
"""
import argparse
import csv
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from src import (
    backtest_engine,
    cv_splitter,
    data_loader,
    feature_engineering,
    labeling,
    metrics,
    model_training,
    risk_manager,
    signal_generator,
)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _ensure_dirs(config: dict) -> None:
    for key in ("artifacts", "reports", "models", "predictions", "processed_data"):
        path = config["paths"].get(key)
        if path:
            Path(path).mkdir(parents=True, exist_ok=True)


def _log_experiment(config: dict, result: dict) -> None:
    log_path = config["reporting"]["experiment_log"]
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    row = {"run_ts": datetime.now(timezone.utc).isoformat(), **result}
    write_header = not Path(log_path).exists()
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_pipeline(config_path: str = "config.yaml", eval_split: str = "val") -> dict:
    """
    Run a full pipeline experiment.

    Parameters
    ----------
    config_path : path to config.yaml
    eval_split  : 'val' for normal development, 'holdout' for final OOS simulation
                  (holdout should be used exactly once)
    """
    config = load_config(config_path)
    _ensure_dirs(config)

    if eval_split == "holdout":
        print("=" * 60)
        print("  WARNING: Running on the LOCKED HOLDOUT.")
        print("  Use this exactly once. No re-tuning after this run.")
        print("=" * 60)

    symbol = config["market"]["symbols"][0]
    market_type = config["market"]["market_type"]

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print(f"\n[1/7] Loading {symbol} ({market_type}) ...")
    ohlcv = data_loader.load_ohlcv(symbol, market_type, config)
    print(f"      {len(ohlcv)} bars  |  {ohlcv.index[0]}  →  {ohlcv.index[-1]}")

    train_df, val_df, holdout_df = data_loader.get_splits(ohlcv, config)
    print(f"      train={len(train_df)}  val={len(val_df)}  holdout={len(holdout_df)}")

    eval_df = val_df if eval_split == "val" else holdout_df

    # ── 2. Features ───────────────────────────────────────────────────────────
    print("\n[2/7] Building features ...")
    train_feats = feature_engineering.build_features(train_df, config)
    eval_feats = feature_engineering.build_features(eval_df, config)

    transformer = feature_engineering.fit_transformer(train_feats.dropna())
    train_scaled = feature_engineering.transform(train_feats, transformer)
    eval_scaled = feature_engineering.transform(eval_feats, transformer)

    transformer_path = os.path.join(config["paths"]["models"], "transformer.pkl")
    feature_engineering.save_transformer(transformer, transformer_path)
    print(f"      Transformer saved → {transformer_path}")
    print(f"      Feature columns: {len(transformer.feature_cols)}")

    # ── 3. Labels ─────────────────────────────────────────────────────────────
    print("\n[3/7] Building labels ...")
    train_labels = labeling.make_labels(train_df, config)
    eval_labels = labeling.make_labels(eval_df, config)

    # Align features and labels
    train_idx = train_scaled.index.intersection(train_labels.index)
    X_train = train_scaled.loc[train_idx].dropna()
    y_train = train_labels.loc[X_train.index, "target"]

    eval_idx = eval_scaled.index.intersection(eval_labels.index)
    X_eval = eval_scaled.loc[eval_idx].dropna()
    y_eval = eval_labels.loc[X_eval.index, "target"]

    dist = y_train.value_counts().sort_index().to_dict()
    print(f"      Label distribution (train): {dist}")
    print(f"      X_train={len(X_train)}  X_eval={len(X_eval)}")

    # ── 4. Train model ────────────────────────────────────────────────────────
    cv_cfg = config["cross_validation"]
    cv_splits = list(
        cv_splitter.purged_walk_forward_cv(
            X_train,
            n_folds=cv_cfg["folds"],
            embargo_bars=cv_cfg["embargo_bars"],
        )
    )

    model_type = config["model"]["baseline"]
    print(f"\n[4/7] Training {model_type} with {len(cv_splits)} CV folds ...")
    model, oof_preds = model_training.train(
        X_train, y_train, cv_splits, model_type=model_type, config=config
    )

    model_path = os.path.join(config["paths"]["models"], f"model_{model_type}.pkl")
    model_training.save_model(model, model_path)
    print(f"      Model saved → {model_path}")

    # Save dataset artifact
    dataset_path = os.path.join(config["paths"]["processed_data"], "dataset_v1.parquet")
    ds = X_train.copy()
    ds["target"] = y_train
    ds.to_parquet(dataset_path)
    print(f"      Dataset artifact → {dataset_path}")

    # ── 5. Signals ────────────────────────────────────────────────────────────
    print(f"\n[5/7] Generating signals on '{eval_split}' set ...")
    eval_probs = model_training.predict_proba(model, X_eval)
    signals = signal_generator.generate_signals(eval_probs, config)
    sig_counts = signals["signal"].value_counts().sort_index().to_dict()
    print(f"      Signal distribution: {sig_counts}")

    preds_path = os.path.join(config["paths"]["predictions"], f"signals_{eval_split}.parquet")
    signals.to_parquet(preds_path)

    # ── 6. Risk sizing + backtest ──────────────────────────────────────────────
    print("\n[6/7] Sizing positions and running backtest ...")
    initial_equity = config["backtest"]["initial_equity"]
    orders = risk_manager.size_positions(signals, initial_equity, eval_df, config)

    trades, equity_curve = backtest_engine.run_backtest(eval_df, orders, config)
    print(f"      Trades: {len(trades)}")

    # ── 7. Metrics + reporting ─────────────────────────────────────────────────
    print("\n[7/7] Computing metrics ...")
    result = metrics.evaluate(equity_curve, trades)
    result["model_type"] = model_type
    result["eval_split"] = eval_split
    result["n_train"] = len(X_train)
    result["n_eval"] = len(X_eval)
    result["n_features"] = len(transformer.feature_cols)

    print("\n" + "=" * 40)
    print("  RESULTS")
    print("=" * 40)
    for k, v in result.items():
        print(f"  {k:<25} {v}")

    if result.get("sharpe", 0) > 5:
        print("\n  ⚠  Sharpe > 5 — possible look-ahead leakage. Audit before proceeding.")

    if risk_manager.apply_kill_switch(equity_curve, config):
        print("  ⚠  Kill-switch would have triggered during this period.")

    plot_path = config["reporting"]["equity_curve_plot"]
    metrics.plot_equity_curve(equity_curve, trades, plot_path, result)

    _log_experiment(config, result)
    print(f"\n  Experiment logged → {config['reporting']['experiment_log']}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="cmoon trading pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--split",
        default="val",
        choices=["val", "holdout"],
        help="Evaluation split (use 'holdout' only once for final OOS simulation)",
    )
    args = parser.parse_args()
    run_pipeline(config_path=args.config, eval_split=args.split)