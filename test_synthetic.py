"""
Test strategies against synthetic_data/ files.

Maps synthetic parquet files to the 3 expected coin names and runs
all available strategies across different market regime scenarios.

Usage:
  python test_synthetic.py
  python test_synthetic.py --strategy trend
  python test_synthetic.py --scenario all_trending
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SYNTHETIC_DIR = ROOT / "synthetic_data"

# cnlib COINS: kapcoin / metucoin / tamcoin
COIN_NAMES = ["kapcoin-usd_train", "metucoin-usd_train", "tamcoin-usd_train"]

# Scenarios: name → 3 synthetic files (must share same row count)
SCENARIOS: dict[str, list[str]] = {
    "trending":         ["btc_like_momentum.parquet",  "btc_like_breakout.parquet", "btc_like_choppy.parquet"],
    "inverse":          ["btc_like_inverse.parquet",   "btc_like_choppy.parquet",   "btc_like_momentum.parquet"],
    "chaos":            ["random_walk_chaos.parquet",  "btc_like_choppy.parquet",   "btc_like_inverse.parquet"],
    "all_choppy":       ["btc_like_choppy.parquet",    "btc_like_choppy.parquet",   "btc_like_choppy.parquet"],
    "all_momentum":     ["btc_like_momentum.parquet",  "btc_like_momentum.parquet", "btc_like_momentum.parquet"],
    "all_inverse":      ["btc_like_inverse.parquet",   "btc_like_inverse.parquet",  "btc_like_inverse.parquet"],
}


def run_scenario(scenario_name: str, files: list[str], strategy_name: str) -> None:
    from research.backtest_window import run_backtest_window

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for coin_name, src_file in zip(COIN_NAMES, files):
            shutil.copy(SYNTHETIC_DIR / src_file, tmp / f"{coin_name}.parquet")

        if strategy_name == "main":
            from strategy import MyStrategy
            strategy = MyStrategy()
        elif strategy_name == "trend":
            from research.trend_strategy import TrendStrategy
            strategy = TrendStrategy()
        elif strategy_name == "meanrevert":
            from research.mean_revert_strategy import MeanRevertStrategy
            strategy = MeanRevertStrategy()
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        result = run_backtest_window(
            strategy,
            initial_capital=3000.0,
            data_dir=tmp,
            silent=True,
        )

    print(f"\n{'─'*60}")
    print(f"  Scenario : {scenario_name}")
    print(f"  Files    : {files[0]} / {files[1]} / {files[2]}")
    print(f"  Strategy : {strategy_name}")
    result.print_summary()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategy",
        default="all",
        choices=["all", "main", "trend", "meanrevert"],
    )
    parser.add_argument(
        "--scenario",
        default="all",
        choices=["all"] + list(SCENARIOS),
    )
    args = parser.parse_args()

    strategies = ["trend", "meanrevert", "main"] if args.strategy == "all" else [args.strategy]
    scenarios = SCENARIOS if args.scenario == "all" else {args.scenario: SCENARIOS[args.scenario]}

    print("=" * 60)
    print("  SYNTHETIC DATA BACKTEST")
    print("=" * 60)

    for scenario_name, files in scenarios.items():
        for strat in strategies:
            try:
                run_scenario(scenario_name, files, strat)
            except Exception as exc:
                print(f"\n[FAILED] {scenario_name} / {strat}: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
