from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from cnlib.base_strategy import BaseStrategy, COINS

matplotlib.use("Agg")


class _Loader(BaseStrategy):
    def predict(self, data):
        return []


def load_coin_data() -> dict[str, pd.DataFrame]:
    loader = _Loader()
    loader.get_data()
    return {coin: df.copy() for coin, df in loader._full_data.items()}


def main() -> None:
    coin_data = load_coin_data()

    fig, axes = plt.subplots(len(COINS), 1, figsize=(14, 10), sharex=True)
    fig.suptitle("Coin Closing Prices Over Time", fontsize=16)

    for ax, coin in zip(axes, COINS):
        df = coin_data[coin].copy()
        df["Date"] = pd.to_datetime(df["Date"])

        ax.plot(df["Date"], df["Close"], linewidth=1.5)
        ax.set_title(coin)
        ax.set_ylabel("Close Price")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Date")
    fig.autofmt_xdate()
    plt.tight_layout()

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / "coin_prices.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
