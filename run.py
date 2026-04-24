"""
Local backtest runner — run this to test any strategy quickly.

Usage:
  python run.py                          # runs the current strategy.py
  python run.py --strategy trend        # runs TrendStrategy
  python run.py --strategy meanrevert
  python run.py --strategy ensemble

Add --plot to show the equity curve (requires matplotlib).
"""
import argparse
import sys
from pathlib import Path

from cnlib import backtest

# ---------------------------------------------------------------------------
# Strategy registry — add new strategies here as they're built
# ---------------------------------------------------------------------------

def get_strategy(name: str):
    if name == "main":
        from strategy import MyStrategy
        return MyStrategy()
    elif name == "trend":
        from research.trend_strategy import TrendStrategy
        return TrendStrategy()
    elif name == "meanrevert":
        from research.mean_revert_strategy import MeanRevertStrategy
        return MeanRevertStrategy()
    elif name == "ensemble":
        # TODO: import FinalStrategy from strategy.py once it's built
        raise NotImplementedError("Ensemble strategy not ready yet")
    else:
        print(f"Unknown strategy: {name}")
        print("Available: main, trend, meanrevert, ensemble")
        sys.exit(1)


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
    parser.add_argument("--strategy", default="main", help="Strategy to run")
    parser.add_argument("--capital",  type=float, default=3000.0)
    parser.add_argument("--start",    type=int,   default=0, help="start_candle (warm-up skip)")
    parser.add_argument("--plot",     action="store_true")
    parser.add_argument("--silent",   action="store_true")
    args = parser.parse_args()

    strategy = get_strategy(args.strategy)

    print(f"Running strategy: {args.strategy}")
    result = backtest.run(
        strategy,
        initial_capital=args.capital,
        start_candle=args.start,
        silent=args.silent,
    )
    result.print_summary()

    if args.plot:
        plot_equity(result, args.strategy)


if __name__ == "__main__":
    main()
