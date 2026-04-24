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

    market_type = config["market"]["market_type"]

    # ── 1. Discover universe ───────────────────────────────────────────────────
    print("\n[1/7] Discovering coin universe ...")
    symbols = data_loader.discover_universe(config)
    if not symbols:
        raise RuntimeError(
            "No valid symbols discovered. Add data files to data/raw/ "
            "or set universe_mode=explicit in config.yaml."
        )
    print(f"      {len(symbols)} symbol(s): {symbols}")

    # ── 2. Load data + build per-symbol features and labels ───────────────────
    print("\n[2/7] Loading data and building features / labels ...")
    train_feat_parts: list = []
    train_label_parts: list = []
    eval_feats_by_sym: dict = {}
    eval_ohlcv_by_sym: dict = {}

    for sym in symbols:
        ohlcv = data_loader.load_ohlcv(sym, market_type, config)
        train_df, val_df, holdout_df = data_loader.get_splits(ohlcv, config)
        eval_df = val_df if eval_split == "val" else holdout_df

        # Per-symbol features
        train_feats = feature_engineering.build_features(train_df, config)
        eval_feats = feature_engineering.build_features(eval_df, config)

        # Per-symbol labels
        train_labels = labeling.make_labels(train_df, config)
        eval_labels = labeling.make_labels(eval_df, config)

        # Align train features ↔ labels
        t_idx = train_feats.index.intersection(train_labels.index)
        X_tr = train_feats.loc[t_idx].dropna().copy()
        y_tr = train_labels.loc[X_tr.index, "target"]
        X_tr["symbol"] = sym

        # Align eval features ↔ labels (labels used for signal eval only, not model input)
        e_idx = eval_feats.index.intersection(eval_labels.index)
        X_ev = eval_feats.loc[e_idx].dropna().copy()
        X_ev["symbol"] = sym

        train_feat_parts.append(X_tr)
        train_label_parts.append(y_tr)
        eval_feats_by_sym[sym] = X_ev
        eval_ohlcv_by_sym[sym] = eval_df

        print(f"      {sym}: train={len(X_tr)}  eval={len(X_ev)}")

    X_train_all = pd.concat(train_feat_parts).sort_index()
    y_train_all = pd.concat(train_label_parts).sort_index()

    # ── 3. Fit transformer on combined train features ──────────────────────────
    print("\n[3/7] Fitting feature transformer ...")
    feat_cols = [c for c in X_train_all.columns if c != "symbol"]
    transformer = feature_engineering.fit_transformer(X_train_all[feat_cols].dropna())

    transformer_path = os.path.join(config["paths"]["models"], "transformer.pkl")
    feature_engineering.save_transformer(transformer, transformer_path)
    print(f"      Transformer saved → {transformer_path}")
    print(f"      Feature columns: {len(transformer.feature_cols)}")

    X_train_scaled = feature_engineering.transform(X_train_all[feat_cols], transformer)
    X_train_scaled = X_train_scaled.copy()
    X_train_scaled["symbol"] = X_train_all["symbol"]

    eval_scaled_by_sym: dict = {}
    for sym, X_ev in eval_feats_by_sym.items():
        ev_feat_cols = [c for c in X_ev.columns if c != "symbol"]
        X_ev_scaled = feature_engineering.transform(X_ev[ev_feat_cols], transformer).copy()
        X_ev_scaled["symbol"] = sym
        eval_scaled_by_sym[sym] = X_ev_scaled

    # Feature matrix for model (no symbol column)
    X_train_model = X_train_scaled.drop(columns=["symbol"])
    y_train_model = y_train_all.reindex(X_train_model.index).dropna()
    X_train_model = X_train_model.loc[y_train_model.index]

    dist = y_train_model.value_counts().sort_index().to_dict()
    print(f"      Label distribution (train): {dist}")
    print(f"      X_train={len(X_train_model)}  features={len(transformer.feature_cols)}")

    # Save dataset artifact (features + target + symbol)
    ds = X_train_model.copy()
    ds["target"] = y_train_model
    ds["symbol"] = X_train_scaled.loc[ds.index, "symbol"]
    dataset_path = os.path.join(config["paths"]["processed_data"], "dataset_v1.parquet")
    ds.to_parquet(dataset_path)
    print(f"      Dataset artifact → {dataset_path}")

    # ── 4. CV splits + optional Optuna tuning + model training ────────────────
    cv_cfg = config["cross_validation"]
    cv_splits = list(
        cv_splitter.purged_walk_forward_cv(
            X_train_model,
            n_folds=cv_cfg["folds"],
            embargo_bars=cv_cfg["embargo_bars"],
        )
    )

    model_type = config["model"]["baseline"]
    best_params: dict = {}
    n_trials = config["model"].get("max_tuning_trials", 0)
    if n_trials > 0:
        print(f"\n[4a/7] Optuna tuning {model_type} ({n_trials} trials) ...")
        best_params = model_training.tune_hyperparams(
            X_train_model, y_train_model, cv_splits,
            model_type=model_type, config=config,
        )
        print(f"       Best params: {best_params}")

    print(f"\n[4/7] Training {model_type} with {len(cv_splits)} CV folds ...")
    model, oof_preds = model_training.train(
        X_train_model, y_train_model, cv_splits,
        model_type=model_type, params=best_params, config=config,
    )

    model_path = os.path.join(config["paths"]["models"], f"model_{model_type}.pkl")
    model_training.save_model(model, model_path)
    print(f"      Model saved → {model_path}")

    # ── 5. Signals + risk sizing per symbol ───────────────────────────────────
    print(f"\n[5/7] Generating signals on '{eval_split}' set ...")
    initial_equity = config["backtest"]["initial_equity"]
    orders_by_sym: dict = {}

    for sym in symbols:
        X_ev = eval_scaled_by_sym[sym].drop(columns=["symbol"])
        eval_probs = model_training.predict_proba(model, X_ev)
        sigs = signal_generator.generate_signals(eval_probs, config)
        sigs = sigs.copy()
        sigs["symbol"] = sym

        orders = risk_manager.size_positions(sigs, initial_equity, eval_ohlcv_by_sym[sym], config)
        orders = orders.copy()
        orders["symbol"] = sym
        orders_by_sym[sym] = orders

    sig_counts = (
        pd.concat([orders_by_sym[s]["signal"] for s in symbols])
        .value_counts()
        .sort_index()
        .to_dict()
    )
    print(f"      Signal distribution: {sig_counts}")

    all_signals = pd.concat(
        [eval_scaled_by_sym[s][[]].assign(
            signal=orders_by_sym[s]["signal"],
            symbol=s,
        ) for s in symbols]
    )
    preds_path = os.path.join(config["paths"]["predictions"], f"signals_{eval_split}.parquet")
    all_signals.to_parquet(preds_path)

    # ── 6. Multi-symbol backtest ───────────────────────────────────────────────
    print("\n[6/7] Running multi-symbol backtest ...")
    trades, equity_curve = backtest_engine.run_backtest(
        eval_ohlcv_by_sym, orders_by_sym, config
    )
    print(f"      Trades: {len(trades)}")

    # ── 7. Metrics + reporting ─────────────────────────────────────────────────
    print("\n[7/7] Computing metrics ...")
    result = metrics.evaluate(equity_curve, trades)
    result["model_type"] = model_type
    result["eval_split"] = eval_split
    result["n_symbols"] = len(symbols)
    result["n_train"] = len(X_train_model)
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
