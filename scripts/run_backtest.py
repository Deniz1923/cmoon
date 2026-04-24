from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.runner import run_backtest
from strategies.v3_final import CompetitionStrategy


RAW_DIR = ROOT / "data" / "raw"


def main() -> None:
    data = {
        path.stem: pd.read_parquet(path)
        for path in sorted(RAW_DIR.glob("*.parquet"))
        if path.stem in {"Varlik_A", "Varlik_B", "Varlik_C"}
    }
    if len(data) != 3:
        raise SystemExit("Expected Varlik_A/B/C parquet files under data/raw")

    result = run_backtest(CompetitionStrategy(), data)
    metrics = result.metrics
    print(f"final_equity={metrics.final_equity:.4f}")
    print(f"total_return={metrics.total_return:.2%}")
    print(f"sharpe={metrics.sharpe:.3f}")
    print(f"max_drawdown={metrics.max_drawdown:.2%}")
    print(f"hit_rate={metrics.hit_rate:.2%}")


if __name__ == "__main__":
    main()
