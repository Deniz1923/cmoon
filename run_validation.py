from __future__ import annotations

import argparse
import json

from constants import DEFAULT_RULES_PATH
from model_types import BacktestConfig
from runner import run_local_validation


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Yerel validation backtest calistir.")
    p.add_argument("--strategy-file", default="strategy.py")
    p.add_argument("--class-name", default="MyStrategy")
    p.add_argument("--rules", default=DEFAULT_RULES_PATH)
    p.add_argument("--train-dir", default="data/train")
    p.add_argument("--validation-dir", default="data/validation")
    p.add_argument("--out-dir", default="outputs")
    p.add_argument("--initial-equity", type=float, default=None)
    p.add_argument("--fee-rate", type=float, default=None)
    p.add_argument("--min-history", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = BacktestConfig(
        initial_equity=args.initial_equity,
        fee_rate=args.fee_rate,
        min_history=args.min_history,
    )
    result = run_local_validation(
        strategy_file=args.strategy_file,
        class_name=args.class_name,
        train_dir=args.train_dir,
        validation_dir=args.validation_dir,
        out_dir=args.out_dir,
        rules_path=args.rules,
        config=cfg,
    )

    print("\n=== Validation Sonucu ===")
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
