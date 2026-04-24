from __future__ import annotations

from pathlib import Path

from backtest import BacktestEngine
from competition import CompetitionRules, load_rules
from data_loader import load_split
from evaluation import build_backtest_result
from model_types import BacktestConfig, BacktestResult
from output_writer import save_outputs
from strategy_loader import load_strategy_class


def run_local_validation(
    strategy_file: str | Path,
    class_name: str,
    train_dir: str | Path,
    validation_dir: str | Path,
    out_dir: str | Path,
    rules: CompetitionRules | None = None,
    rules_path: str | Path | None = None,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    resolved_rules = rules or load_rules(rules_path)

    strategy_cls = load_strategy_class(strategy_file, class_name)
    strategy = strategy_cls()
    strategy.set_rules(resolved_rules)

    train_data = load_split(train_dir, coins=resolved_rules.coins)
    validation_data = load_split(validation_dir, coins=resolved_rules.coins)

    strategy.fit(train_data)
    engine = BacktestEngine(rules=resolved_rules, config=config)
    equity_curve, liquidation_count = engine.run(strategy=strategy, data=validation_data)
    result = build_backtest_result(equity_curve, liquidation_count)
    return save_outputs(
        result=result,
        equity_curve=equity_curve,
        out_dir=out_dir,
        rules=resolved_rules,
        effective_config={
            "initial_equity": engine.initial_equity,
            "fee_rate": engine.fee_rate,
            "min_history": engine.min_history,
        },
    )
