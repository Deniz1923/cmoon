from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.runner import run_backtest
from strategies.v3_final import CompetitionStrategy
from submission import strateji


def main() -> None:
    data = make_synthetic_data()

    strategy = CompetitionStrategy()
    one_row_orders = strategy.predict({asset: frame.iloc[:1] for asset, frame in data.items()})
    _assert_contract(one_row_orders)

    result = run_backtest(strategy, data)
    _assert_contract(strategy.predict(data))
    _assert_contract(strateji.predict({asset: frame.iloc[:1] for asset, frame in data.items()}))

    print("smoke_ok=true")
    print(f"orders_first_row={one_row_orders}")
    print(f"final_equity={result.metrics.final_equity:.4f}")
    print(f"max_drawdown={result.metrics.max_drawdown:.2%}")


def make_synthetic_data(rows: int = 240) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(42)
    index = pd.RangeIndex(rows)
    data: dict[str, pd.DataFrame] = {}

    for asset_index, asset in enumerate(("Varlik_A", "Varlik_B", "Varlik_C")):
        drift = [0.0004, -0.0001, 0.0002][asset_index]
        shocks = rng.normal(drift, 0.015 + asset_index * 0.004, size=rows)
        close = 100 * np.exp(np.cumsum(shocks))
        open_ = np.roll(close, 1)
        open_[0] = close[0]
        spread = np.abs(rng.normal(0.006, 0.003, size=rows))
        high = np.maximum(open_, close) * (1 + spread)
        low = np.minimum(open_, close) * (1 - spread)
        volume = rng.lognormal(mean=12.0, sigma=0.4, size=rows)
        data[asset] = pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            },
            index=index,
        )

    return data


def _assert_contract(orders: list[dict]) -> None:
    assert len(orders) == 3, orders
    assert sum(float(order["oran"]) for order in orders) <= 1.000001, orders
    for order in orders:
        assert set(order) == {"sinyal", "oran", "kaldirac"}, order
        assert order["sinyal"] in {-1, 0, 1}, order
        assert 0.0 <= float(order["oran"]) <= 1.0, order
        assert order["kaldirac"] in {2, 3, 5, 10}, order


if __name__ == "__main__":
    main()
