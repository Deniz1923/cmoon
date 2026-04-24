"""
Local backtest runner — run this to test any strategy quickly.

Usage:
  .venv/bin/python run.py                         # runs the current strategy.py
  .venv/bin/python run.py --strategy trend        # runs TrendStrategy
  .venv/bin/python run.py --strategy meanrevert

Add --plot to show the equity curve (requires matplotlib).
"""
import argparse
import sys
from pathlib import Path
from typing import Callable

from cnlib.base_strategy import BaseStrategy
from research.backtest_window import run_backtest_window

ROOT_DIR = Path(__file__).resolve().parent
SYNTHETIC_DATASET_DIR = ROOT_DIR / "synthetic_cnlib_data"

# ---------------------------------------------------------------------------
# Strategy registry — add new strategies here as they're built
# ---------------------------------------------------------------------------

StrategyFactory = Callable[[], BaseStrategy]


def _main_strategy() -> BaseStrategy:
    from strategy import MyStrategy
    return MyStrategy()


def _trend_strategy() -> BaseStrategy:
    from research.trend_strategy import TrendStrategy
    return TrendStrategy()


def _mean_revert_strategy() -> BaseStrategy:
    from research.mean_revert_strategy import MeanRevertStrategy
    return MeanRevertStrategy()


STRATEGIES: dict[str, StrategyFactory] = {
    "main": _main_strategy,
    "trend": _trend_strategy,
    "meanrevert": _mean_revert_strategy,
}

DATASET_PRESETS: dict[str, Path | None] = {
    "cnlib": None,
    "synthetic": SYNTHETIC_DATASET_DIR,
}


def get_strategy(name: str):
    if name not in STRATEGIES:
        available = ", ".join(STRATEGIES)
        raise ValueError(f"Unknown strategy '{name}'. Available: {available}")
    return STRATEGIES[name]()


def resolve_data_dir(dataset: str, data_dir: Path | None) -> Path | None:
    if data_dir is not None:
        return data_dir

    resolved = DATASET_PRESETS.get(dataset)
    if dataset not in DATASET_PRESETS:
        available = ", ".join(sorted(DATASET_PRESETS))
        raise ValueError(f"Unknown dataset '{dataset}'. Available: {available}")
    if resolved is None:
        return None
    if not resolved.exists():
        raise FileNotFoundError(
            f"Dataset preset '{dataset}' is not available at {resolved}. "
            "Run `python research/generate_synthetic_datasets.py` first."
        )
    return resolved


def plot_equity(result, strategy_name: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("pip install matplotlib to enable plots")
        return

    df = result.portfolio_dataframe()
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Equity curve
    axes[0].plot(df["candle_index"], df["portfolio_value"], label="Portfolio Value")
    axes[0].axhline(3000, color="gray", linestyle="--", label="Initial Capital")
    axes[0].set_ylabel("Portfolio ($)")
    axes[0].set_title(f"{strategy_name} — Equity Curve")
    axes[0].legend()

    # Coin prices (normalized to 100)
    price_cols = [c for c in df.columns if c.endswith("_price")]
    for col in price_cols:
        norm = df[col] / df[col].iloc[0] * 100
        axes[1].plot(df["candle_index"], norm, label=col.replace("_price", ""), alpha=0.7)
    axes[1].set_ylabel("Price (normalized to 100)")
    axes[1].set_xlabel("Candle Index")
    axes[1].legend()

    plt.tight_layout()

    # Save
    out = Path("results") / f"{strategy_name}_equity.png"
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=150)
    print(f"Plot saved → {out}")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="main", choices=sorted(STRATEGIES), help="Strategy to run")
    parser.add_argument("--capital",  type=float, default=3000.0)
    parser.add_argument("--start",    type=int,   default=0, help="start_candle (warm-up skip)")
    parser.add_argument("--end",      type=int,   default=None, help="inclusive end_candle")
    parser.add_argument("--data-dir", type=Path,  default=None, help="directory containing coin parquet files")
    parser.add_argument(
        "--dataset",
        default="cnlib",
        choices=sorted(DATASET_PRESETS),
        help="dataset preset to use when --data-dir is not provided",
    )
    parser.add_argument("--list-strategies", action="store_true", help="show available strategies and exit")
    parser.add_argument("--list-datasets", action="store_true", help="show dataset presets and exit")
    parser.add_argument("--plot",     action="store_true")
    parser.add_argument("--silent",   action="store_true")
    args = parser.parse_args()

    if args.list_strategies:
        for name in STRATEGIES:
            print(f"{name}\tready")
        return

    if args.list_datasets:
        for name, path in DATASET_PRESETS.items():
            location = "bundled cnlib data" if path is None else str(path)
            status = "ready" if path is None or path.exists() else "missing"
            print(f"{name}\t{status}\t{location}")
        return

    try:
        strategy = get_strategy(args.strategy)
        data_dir = resolve_data_dir(args.dataset, args.data_dir)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        sys.exit(2)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        sys.exit(2)

    print(f"Running strategy: {args.strategy}")
    if args.data_dir is not None:
        print(f"Dataset: custom ({args.data_dir})")
    else:
        print(f"Dataset: {args.dataset}")
    result = run_backtest_window(
        strategy,
        initial_capital=args.capital,
        start_candle=args.start,
        end_candle=args.end,
        data_dir=data_dir,
        silent=args.silent,
    )
    result.print_summary()

    if args.plot:
        plot_equity(result, args.strategy)


if __name__ == "__main__":
    main()
