"""
Offline calibration: sweep ML confidence threshold on holdout (1100-1569)
and report ATR% percentiles to validate leverage breakpoints.

Run: python research/calibrate_threshold.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from research.train_models import load_all_data, load_model_bundle, split_train_holdout, _positive_prob, TRAIN_END_CANDLE
from research.ml_features import build_X_y, TARGET_HORIZON
from research.features import atr_pct
from cnlib.base_strategy import COINS


def main() -> None:
    print("Loading data and models...")
    coin_data = load_all_data()

    # ------------------------------------------------------------------ #
    # 1. Per-coin threshold sweep on holdout                               #
    # ------------------------------------------------------------------ #
    thresholds = np.arange(0.02, 0.72, 0.02)   # confidence scale [0,1]
    # confidence = |prob - 0.5| * 2  →  prob = 0.5 + conf/2
    # conf=0.04 → prob=0.52, conf=0.10 → prob=0.55, conf=0.52 → prob=0.76

    all_probs: list[np.ndarray] = []
    all_y: list[np.ndarray] = []

    print("\n── Per-coin holdout calibration ──")
    for coin in COINS:
        bundle = load_model_bundle(coin)
        model = bundle["estimator"]

        feature_names = bundle.get("feature_names") or []
        X, y, valid_index = build_X_y(coin_data, coin)
        _, y_train, X_test, y_test, _, test_idx = split_train_holdout(X, y, valid_index)

        if len(X_test) == 0:
            print(f"{coin}: no holdout samples, skipping")
            continue

        # Reindex to match training feature set (same as strategy.py:238)
        if feature_names:
            import pandas as pd
            from research.ml_features import build_features_single
            from research.features import atr_pct as _atr_pct_feat
            leader_close = coin_data["kapcoin-usd_train"]["Close"] if coin != "kapcoin-usd_train" else None
            feat_df = build_features_single(coin_data[coin], leader_close=leader_close)
            feat_df = feat_df.replace([float("inf"), float("-inf")], float("nan"))
            feat_df = feat_df.reindex(columns=feature_names)
            X_test = feat_df.loc[test_idx].dropna(how="any").values.astype("float32")
            y_test = y_test[:len(X_test)]

        probs = _positive_prob(model, X_test)   # P(up)
        # Convert to confidence score used in strategy
        conf = np.abs(probs - 0.5) * 2.0

        all_probs.append(probs)
        all_y.append(y_test)

        print(f"\n  {coin}  (n={len(y_test)}, base rate={y_test.mean():.2%})")
        print(f"  {'conf_thresh':>11}  {'prob_equiv':>10}  {'signals':>8}  {'precision':>10}  {'recall':>8}  {'f1':>8}")
        for t in thresholds:
            mask = conf >= t
            n_sig = mask.sum()
            if n_sig == 0:
                continue
            # Only evaluate directional precision: did ML-direction match outcome?
            # signal_dir: +1 if prob>0.5, -1 if prob<0.5
            signal_dir = np.where(probs > 0.5, 1, -1)[mask]
            actual_dir = np.where(y_test[mask] == 1, 1, -1)
            correct = (signal_dir == actual_dir).sum()
            precision = correct / n_sig
            recall = correct / (y_test == (probs > 0.5).astype(int)).sum() if (y_test == (probs > 0.5).astype(int)).sum() > 0 else 0.0
            # Simpler: precision only (most important for entry gate)
            prob_equiv = 0.5 + t / 2
            print(f"  {t:>11.2f}  {prob_equiv:>10.3f}  {n_sig:>8d}  {precision:>10.2%}  {'—':>8}  {'—':>8}")

    # ------------------------------------------------------------------ #
    # 2. Combined across all coins — find best threshold                   #
    # ------------------------------------------------------------------ #
    if all_probs:
        probs_all = np.concatenate(all_probs)
        y_all = np.concatenate(all_y)
        conf_all = np.abs(probs_all - 0.5) * 2.0

        print("\n\n── Combined (all coins) threshold sweep ──")
        print(f"  {'conf_thresh':>11}  {'prob_equiv':>10}  {'signals':>8}  {'precision':>10}  {'pct_filtered':>12}")
        best_t, best_score = 0.04, 0.0
        for t in thresholds:
            mask = conf_all >= t
            n_sig = mask.sum()
            if n_sig < 5:
                break
            signal_dir = np.where(probs_all > 0.5, 1, -1)[mask]
            actual_dir = np.where(y_all[mask] == 1, 1, -1)
            precision = (signal_dir == actual_dir).mean()
            pct_filtered = 1.0 - n_sig / len(y_all)
            prob_equiv = 0.5 + t / 2
            marker = " ◄" if precision > best_score and n_sig >= 20 else ""
            if precision > best_score and n_sig >= 20:
                best_score = precision
                best_t = t
            print(f"  {t:>11.2f}  {prob_equiv:>10.3f}  {n_sig:>8d}  {precision:>10.2%}  {pct_filtered:>12.1%}{marker}")

        print(f"\n  ► Recommended conf threshold: {best_t:.2f}  (prob equiv: {0.5+best_t/2:.3f})")
        print(f"    Best directional precision on holdout: {best_score:.2%}")

    # ------------------------------------------------------------------ #
    # 3. ATR% distribution on holdout — validate leverage breakpoints      #
    # ------------------------------------------------------------------ #
    print("\n\n── ATR%(14) distribution on holdout (candles 1100-1569) ──")
    for coin in COINS:
        df = coin_data[coin]
        holdout_df = df[df.index >= TRAIN_END_CANDLE]
        atr_series = atr_pct(holdout_df, 14).dropna()
        if atr_series.empty:
            continue
        pcts = np.percentile(atr_series, [25, 50, 75, 90, 95, 99])
        print(f"\n  {coin}")
        print(f"    p25={pcts[0]:.2%}  p50={pcts[1]:.2%}  p75={pcts[2]:.2%}"
              f"  p90={pcts[3]:.2%}  p95={pcts[4]:.2%}  p99={pcts[5]:.2%}")
        # Show what % of candles each leverage tier would apply
        vals = atr_series.values
        n5x = (vals < 0.075).mean()
        n3x = ((vals >= 0.075) & (vals < 0.100)).mean()
        n2x = ((vals >= 0.100) & (vals < 0.124)).mean()
        n1x = (vals >= 0.124).mean()
        print(f"    5x: {n5x:.1%}  3x: {n3x:.1%}  2x: {n2x:.1%}  1x: {n1x:.1%}  (current breakpoints)")

        # Max single-candle return — compare to liquidation thresholds
        max_ret = df["Close"].pct_change().abs().loc[df.index >= TRAIN_END_CANDLE].max()
        print(f"    Max |candle return| on holdout: {max_ret:.2%}  (5x liquidation at 20%)")


if __name__ == "__main__":
    main()
