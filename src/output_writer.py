from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from competition import CompetitionRules
from model_types import BacktestResult


def save_outputs(
    result: BacktestResult,
    equity_curve: pd.DataFrame,
    out_dir: str | Path,
    rules: CompetitionRules,
    effective_config: dict[str, float | int],
) -> BacktestResult:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report_path = out_dir / "report.json"
    equity_curve_path = out_dir / "equity_curve.csv"
    step_log_path = out_dir / "step_log.csv"

    result.report_path = str(report_path)
    result.equity_curve_path = str(equity_curve_path)
    result.step_log_path = str(step_log_path)

    payload = asdict(result)
    payload["rules"] = rules.to_dict()
    payload["effective_config"] = effective_config
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    equity_curve[["equity"]].to_csv(equity_curve_path)
    equity_curve.to_csv(step_log_path)
    return result
