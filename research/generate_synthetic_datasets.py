from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = ROOT / ".venv" / "Lib" / "site-packages" / "cnlib" / "data" / "kapcoin-usd_train.parquet"
OUTPUT_DIR = ROOT / "synthetic_data"
CNLIB_COMPAT_DIR = ROOT / "synthetic_cnlib_data"
CNLIB_NAMES = [
    "kapcoin-usd_train.parquet",
    "metucoin-usd_train.parquet",
    "tamcoin-usd_train.parquet",
]


@dataclass(frozen=True)
class SeriesSpec:
    name: str
    seed: int
    start_price: float
    drift: float
    volatility: float
    volume_base: float
    volume_scale: float
    shock_scale: float
    base_scale: float = 1.0
    inverse_of: str | None = None
    fully_random: bool = False


def _load_dates() -> pd.Series:
    template = pd.read_parquet(TEMPLATE_PATH)
    return template["Date"].astype(str)


def _gaussian_bump(length: int, center: float, width: float, amplitude: float) -> np.ndarray:
    idx = np.arange(length, dtype=np.float64)
    mu = center * (length - 1)
    sigma = max(width * length, 1.0)
    return amplitude * np.exp(-0.5 * ((idx - mu) / sigma) ** 2)


def _btc_like_regime(length: int) -> np.ndarray:
    # Approximate a BTC-like path: long-run uptrend, late drawdown, then a recent rally.
    baseline = np.linspace(0.00025, 0.00055, length)
    drawdown = _gaussian_bump(length, center=0.82, width=0.06, amplitude=-0.0045)
    rebound = _gaussian_bump(length, center=0.94, width=0.04, amplitude=0.0065)
    return baseline + drawdown + rebound


def _random_regime(length: int, rng: np.random.Generator) -> np.ndarray:
    steps = rng.normal(0.0, 0.0009, length)
    local = np.convolve(steps, np.ones(15) / 15, mode="same")
    return local


def _build_close_series(length: int, spec: SeriesSpec, base_returns: np.ndarray | None = None) -> np.ndarray:
    rng = np.random.default_rng(spec.seed)
    if spec.fully_random:
        regime = _random_regime(length, rng)
        noise = rng.normal(0.0, spec.volatility * 1.5, length)
    elif base_returns is not None and spec.inverse_of:
        regime = -spec.base_scale * base_returns
        noise = rng.normal(0.0, spec.volatility, length)
    elif base_returns is not None:
        regime = spec.base_scale * base_returns
        noise = rng.normal(0.0, spec.volatility * 0.65, length)
    else:
        regime = _btc_like_regime(length)
        noise = rng.normal(0.0, spec.volatility, length)

    cyclical = 0.0012 * np.sin(np.linspace(0, 14 * np.pi, length))
    shocks = rng.normal(0.0, spec.shock_scale, length) * (rng.random(length) < 0.045)
    log_returns = spec.drift + regime + cyclical + noise + shocks

    if spec.inverse_of:
        log_returns = log_returns - np.mean(log_returns) * 0.35

    close = np.empty(length, dtype=np.float64)
    close[0] = spec.start_price
    for i in range(1, length):
        close[i] = max(1000.0, close[i - 1] * np.exp(log_returns[i]))
    return close


def _build_ohlcv(dates: pd.Series, close: np.ndarray, spec: SeriesSpec) -> pd.DataFrame:
    rng = np.random.default_rng(spec.seed + 10_000)
    length = len(close)

    open_ = np.empty(length, dtype=np.float64)
    high = np.empty(length, dtype=np.float64)
    low = np.empty(length, dtype=np.float64)

    open_[0] = close[0] * (1 + rng.normal(0.0, 0.004))
    for i in range(1, length):
        overnight_gap = rng.normal(0.0, 0.006)
        open_[i] = max(1000.0, close[i - 1] * (1 + overnight_gap))

    intraday_scale = np.clip(np.abs(close - open_) / np.maximum(open_, 1.0), 0.003, 0.08)
    wick_up = rng.uniform(0.002, 0.018, length) + intraday_scale * rng.uniform(0.3, 1.1, length)
    wick_down = rng.uniform(0.002, 0.018, length) + intraday_scale * rng.uniform(0.3, 1.1, length)

    high = np.maximum(open_, close) * (1 + wick_up)
    low = np.minimum(open_, close) * np.maximum(0.75, 1 - wick_down)

    realized_move = np.abs(np.diff(np.r_[close[0], close])) / np.maximum(close, 1.0)
    volume = spec.volume_base * (1 + realized_move * spec.volume_scale + rng.lognormal(mean=-0.15, sigma=0.45, size=length))

    return pd.DataFrame(
        {
            "Date": dates,
            "Close": close.astype(np.float64),
            "High": high.astype(np.float64),
            "Low": low.astype(np.float64),
            "Open": open_.astype(np.float64),
            "Volume": volume.astype(np.float64),
        }
    )


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    CNLIB_COMPAT_DIR.mkdir(exist_ok=True)
    dates = _load_dates()
    length = len(dates)
    ref_rng = np.random.default_rng(20260425)
    shared_base_returns = (
        _btc_like_regime(length)
        + 0.0010 * np.sin(np.linspace(0, 10 * np.pi, length))
        + ref_rng.normal(0.0, 0.014, length)
        + ref_rng.normal(0.0, 0.040, length) * (ref_rng.random(length) < 0.035)
    )

    specs = [
        SeriesSpec(
            name="btc_like_momentum",
            seed=7,
            start_price=43800.0,
            drift=0.00012,
            volatility=0.014,
            volume_base=1.5e10,
            volume_scale=40.0,
            shock_scale=0.045,
            base_scale=0.95,
        ),
        SeriesSpec(
            name="btc_like_breakout",
            seed=19,
            start_price=47200.0,
            drift=0.00018,
            volatility=0.016,
            volume_base=1.8e10,
            volume_scale=45.0,
            shock_scale=0.050,
            base_scale=1.08,
        ),
        SeriesSpec(
            name="btc_like_choppy",
            seed=29,
            start_price=40100.0,
            drift=0.00008,
            volatility=0.020,
            volume_base=1.3e10,
            volume_scale=50.0,
            shock_scale=0.060,
            base_scale=0.82,
        ),
        SeriesSpec(
            name="btc_like_inverse",
            seed=41,
            start_price=45500.0,
            drift=-0.00008,
            volatility=0.015,
            volume_base=1.4e10,
            volume_scale=42.0,
            shock_scale=0.040,
            base_scale=0.90,
            inverse_of="btc_like_momentum",
        ),
        SeriesSpec(
            name="random_walk_chaos",
            seed=101,
            start_price=12000.0,
            drift=0.0,
            volatility=0.030,
            volume_base=8.0e9,
            volume_scale=65.0,
            shock_scale=0.090,
            fully_random=True,
        ),
    ]

    generated_closes: dict[str, np.ndarray] = {}
    generated_paths: dict[str, Path] = {}
    for spec in specs:
        base_returns = None if spec.fully_random else shared_base_returns
        if spec.inverse_of:
            ref = generated_closes[spec.inverse_of]
            base_returns = np.diff(np.log(ref), prepend=np.log(ref[0]))

        close = _build_close_series(length, spec, base_returns=base_returns)
        generated_closes[spec.name] = close

        df = _build_ohlcv(dates, close, spec)
        out_path = OUTPUT_DIR / f"{spec.name}.parquet"
        df.to_parquet(out_path, index=False)
        generated_paths[spec.name] = out_path

        first_close = float(df["Close"].iloc[0])
        last_close = float(df["Close"].iloc[-1])
        print(
            f"{out_path.name}: rows={len(df)} start={first_close:.2f} "
            f"end={last_close:.2f} change={(last_close / first_close - 1) * 100:.2f}%"
        )

    compat_mapping = {
        "kapcoin-usd_train.parquet": generated_paths["btc_like_momentum"],
        "metucoin-usd_train.parquet": generated_paths["btc_like_breakout"],
        "tamcoin-usd_train.parquet": generated_paths["btc_like_inverse"],
    }
    for target_name in CNLIB_NAMES:
        source = compat_mapping[target_name]
        shutil.copy2(source, CNLIB_COMPAT_DIR / target_name)
    print(f"cnlib-compatible synthetic dataset written to: {CNLIB_COMPAT_DIR}")


if __name__ == "__main__":
    main()
