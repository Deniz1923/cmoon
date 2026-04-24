from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from competition import CompetitionRules, load_rules


def _simulate_ohlcv(close: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    open_ = np.concatenate(([close[0]], close[:-1]))
    intraday_spread = np.abs(rng.normal(0.003, 0.0015, len(close)))
    high = np.maximum(open_, close) * (1 + intraday_spread)
    low = np.minimum(open_, close) * (1 - intraday_spread)
    volume = rng.integers(10_000, 80_000, len(close)).astype(float)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def generate_correlated_prices(coins: tuple[str, ...], n: int, seed: int = 42) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    n_coins = len(coins)

    # Tek market faktoru + idiosyncratic noise ile korelasyonlu seri uretir.
    market = rng.normal(0.0003, 0.02, size=n)
    beta = rng.uniform(0.5, 1.2, size=n_coins)
    idio_scale = rng.uniform(0.01, 0.03, size=n_coins)
    drift = rng.normal(0.00015, 0.00015, size=n_coins)
    idio = rng.normal(0.0, idio_scale, size=(n, n_coins))
    returns = market[:, None] * beta[None, :] + idio + drift[None, :]
    returns = np.clip(returns, -0.35, 0.35)

    start_prices = rng.uniform(300.0, 1500.0, size=n_coins)
    price_paths = np.exp(np.cumsum(returns, axis=0)) * start_prices

    out = {}
    for i, coin in enumerate(coins):
        out[coin] = _simulate_ohlcv(price_paths[:, i], rng)
    return out


def save_split(data: dict[str, pd.DataFrame], split_dir: Path, n_rows: int) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    for coin, df in data.items():
        out = df.iloc[:n_rows].copy().reset_index(drop=True)
        out.index.name = "time_index"
        out.to_parquet(split_dir / f"{coin}.parquet", index=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rules dosyasina gore dummy parquet veri uret.")
    p.add_argument("--rules", default="configs/competition.json")
    p.add_argument("--train-bars", type=int, default=1200)
    p.add_argument("--validation-bars", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    train_dir = root / "data" / "train"
    validation_dir = root / "data" / "validation"

    rules: CompetitionRules = load_rules(args.rules)
    n_train = args.train_bars
    n_val = args.validation_bars
    total = n_train + n_val
    if total < 30:
        raise ValueError("Toplam bar sayisi en az 30 olmali.")

    data = generate_correlated_prices(rules.coins, total, seed=args.seed)
    save_split({k: v.iloc[:n_train] for k, v in data.items()}, train_dir, n_train)
    save_split({k: v.iloc[n_train:] for k, v in data.items()}, validation_dir, n_val)

    print("Dummy parquet verileri olusturuldu:")
    print(f"- Coins: {rules.coins}")
    print(f"- {train_dir}")
    print(f"- {validation_dir}")


if __name__ == "__main__":
    main()
