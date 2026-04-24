"""
Parameter tuning via data analysis and walk-forward CV.

Three phases — all analysis is on the TRAINING window (candles 0-1099).
The holdout (1100-1569) is never touched.

Run:
    uv run python research/tune_params.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from research.features import atr as _atr, atr_pct as _atr_pct, ema, bb_width
from research.train_models import load_all_data
from research.walk_forward import walk_forward, TRAIN_END

TRAIN_SLICE = slice(0, TRAIN_END)   # candles 0-1099, never touch holdout


# ---------------------------------------------------------------------------
# Phase A: Leverage thresholds — safety constraint analysis
# ---------------------------------------------------------------------------

def analyse_leverage(coin_data: dict[str, pd.DataFrame]) -> None:
    print("=" * 60)
    print("PHASE A: Leverage threshold analysis")
    print("=" * 60)
    print()

    all_returns: list[float] = []
    all_atr_pcts: list[float] = []

    for coin, df in coin_data.items():
        df = df.iloc[TRAIN_SLICE].copy()
        ret = df["Close"].pct_change().abs().dropna()
        all_returns.extend(ret.tolist())

        ap = _atr_pct(df, 14).dropna()
        all_atr_pcts.extend(ap.tolist())

    returns = np.array(all_returns)
    atr_pcts = np.array(all_atr_pcts)

    print("Per-candle absolute return distribution (all coins, train window):")
    for p in [50, 75, 90, 95, 99, 99.9]:
        print(f"  p{p:5.1f}: {np.percentile(returns, p)*100:.2f}%")

    print()
    print("ATR%(14) distribution:")
    for p in [25, 50, 75, 90, 95]:
        print(f"  p{p:2d}: {np.percentile(atr_pcts, p)*100:.2f}%")

    print()
    print("Liquidation probability at each leverage level:")
    print("  (P that a single candle wipes the position)")
    for lev, label in [(5, "5x → liq at 20% move"),
                       (3, "3x → liq at 33% move"),
                       (2, "2x → liq at 50% move"),
                       (1, "1x → no liquidation")]:
        threshold = 1.0 / lev
        liq_prob = (returns > threshold).mean() * 100
        safe = "✓ SAFE" if liq_prob < 0.5 else "✗ RISKY"
        print(f"  {label:28s}  liq_prob={liq_prob:.3f}%  {safe}")

    print()
    print("Recommended ATR% breakpoints for leverage selection:")
    p50 = np.percentile(atr_pcts, 50)
    p75 = np.percentile(atr_pcts, 75)
    p90 = np.percentile(atr_pcts, 90)
    print(f"  ATR% < {p50*100:.1f}% (p50)  → 5x")
    print(f"  ATR% < {p75*100:.1f}% (p75)  → 3x")
    print(f"  ATR% < {p90*100:.1f}% (p90)  → 2x")
    print(f"  ATR% >= {p90*100:.1f}%        → 1x")
    print()
    print(f"  [store these values: p50={p50:.4f}, p75={p75:.4f}, p90={p90:.4f}]")
    print()
    return p50, p75, p90


# ---------------------------------------------------------------------------
# Phase B: ATR multiplier — stop-noise analysis
# ---------------------------------------------------------------------------

def analyse_atr_multiplier(coin_data: dict[str, pd.DataFrame]) -> None:
    print("=" * 60)
    print("PHASE B: ATR multiplier (stop-loss noise analysis)")
    print("=" * 60)
    print()
    print("How often does the candle's adverse extreme exceed k * ATR?")
    print("(Long signal: how often does Low go below entry - k*ATR?)")
    print("(Short signal: how often does High go above entry + k*ATR?)")
    print()

    long_adverse: list[float] = []
    short_adverse: list[float] = []

    for coin, df in coin_data.items():
        df = df.iloc[TRAIN_SLICE].copy()
        close = df["Close"]
        atr_ser = _atr(df, 14)
        bw = bb_width(close)
        fast = ema(close, 20)
        slow = ema(close, 50)

        for i in range(50, len(df) - 1):
            bw_i = bw.iloc[i]
            atr_i = atr_ser.iloc[i]
            if pd.isna(bw_i) or pd.isna(atr_i) or atr_i <= 0:
                continue

            entry = close.iloc[i]

            # Trending regime
            if bw_i > 0.08:
                f, s = fast.iloc[i], slow.iloc[i]
                if pd.isna(f) or pd.isna(s):
                    continue
                if f > s:  # long signal
                    low_next = df["Low"].iloc[i + 1]
                    adverse = (entry - low_next) / entry
                    long_adverse.append(adverse / (atr_i / entry))  # in ATR units
                else:  # short signal
                    high_next = df["High"].iloc[i + 1]
                    adverse = (high_next - entry) / entry
                    short_adverse.append(adverse / (atr_i / entry))

    all_adverse = np.array(long_adverse + short_adverse)
    if len(all_adverse) == 0:
        print("  No signal candles found.\n")
        return

    print(f"  Samples (signal candles): {len(all_adverse)}")
    print()
    print(f"  {'Multiplier':>12}  {'Stop-out prob':>14}  {'Recommendation'}")
    print(f"  {'-'*12}  {'-'*14}  {'-'*20}")
    best_k = 2.0
    for k in [1.0, 1.5, 2.0, 2.5, 3.0]:
        prob = (all_adverse > k).mean() * 100
        note = ""
        if 8 <= prob <= 18:
            note = "<-- sweet spot (8-18%)"
            best_k = k
        print(f"  k = {k:7.1f}  {prob:12.1f}%  {note}")

    print()
    print(f"  Recommended atr_multiplier: {best_k}")
    print()
    return best_k


# ---------------------------------------------------------------------------
# Phase C: ML confidence threshold — walk-forward CV sweep
# ---------------------------------------------------------------------------

def analyse_confidence_threshold() -> None:
    print("=" * 60)
    print("PHASE C: ML confidence threshold sweep (walk-forward CV)")
    print("=" * 60)
    print()
    print("Sweeping ML_CONFIDENCE_THRESHOLD on 4-fold walk-forward")
    print("(train window only — holdout never touched)")
    print()

    from strategy import MyStrategy

    thresholds = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    summary: list[dict] = []

    for threshold in thresholds:
        class TunedStrategy(MyStrategy):
            ML_CONFIDENCE_THRESHOLD = threshold  # noqa: F821

        TunedStrategy.__name__ = f"MyStrategy(conf={threshold})"

        results = walk_forward(TunedStrategy, n_splits=4)

        returns = [r.return_pct for r in results]
        trades = [r.total_trades for r in results]
        liqs = sum(r.total_liquidations for r in results)
        avg_ret = np.mean(returns)
        std_ret = np.std(returns)
        avg_trades = np.mean(trades)

        summary.append({
            "threshold": threshold,
            "avg_return": avg_ret,
            "std_return": std_ret,
            "avg_trades_per_fold": avg_trades,
            "total_liquidations": liqs,
        })
        print()

    print("=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    print(f"  {'Threshold':>10}  {'Avg Ret%':>9}  {'Std Ret%':>9}  {'Trades/fold':>12}  {'Liqs':>5}")
    print(f"  {'-'*10}  {'-'*9}  {'-'*9}  {'-'*12}  {'-'*5}")

    best = None
    for row in summary:
        sparse = row["avg_trades_per_fold"] < 3
        flag = "  (too sparse)" if sparse else ""
        print(
            f"  {row['threshold']:>10.2f}  "
            f"{row['avg_return']:>+8.2f}%  "
            f"{row['std_return']:>8.2f}%  "
            f"{row['avg_trades_per_fold']:>11.1f}  "
            f"{row['total_liquidations']:>5}{flag}"
        )
        if not sparse and (best is None or row["avg_return"] > best["avg_return"]):
            best = row

    print()
    if best:
        print(f"  Recommended ML_CONFIDENCE_THRESHOLD: {best['threshold']}")
        print(f"  (avg return {best['avg_return']:+.2f}%, std {best['std_return']:.2f}%, "
              f"{best['avg_trades_per_fold']:.1f} trades/fold)")
    else:
        print("  All thresholds produced too few trades — keep default 0.60")
    print()
    return best["threshold"] if best else 0.60


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print()
    print("Loading training data...")
    raw = load_all_data()
    coin_data = {coin: df for coin, df in raw.items()}
    print(f"  Loaded {len(coin_data)} coins, slicing to train window (0-{TRAIN_END})")
    print()

    p50, p75, p90 = analyse_leverage(coin_data)
    best_k = analyse_atr_multiplier(coin_data)
    best_threshold = analyse_confidence_threshold()

    print("=" * 60)
    print("FINAL RECOMMENDATIONS")
    print("=" * 60)
    print()
    print("research/risk.py  dynamic_leverage() breakpoints:")
    print(f"  ATR% < {p50*100:.1f}%  → 5x")
    print(f"  ATR% < {p75*100:.1f}%  → 3x")
    print(f"  ATR% < {p90*100:.1f}%  → 2x")
    print(f"  otherwise            → 1x")
    print()
    print(f"research/risk.py  stop_loss_price(atr_multiplier=...): {best_k}")
    print()
    print(f"strategy.py  MyStrategy.ML_CONFIDENCE_THRESHOLD: {best_threshold}")
    print()
    print("Apply these manually to the respective files after reviewing the analysis above.")
