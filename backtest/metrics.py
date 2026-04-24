from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MetricReport:
    final_equity: float
    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe: float
    max_drawdown: float
    calmar: float
    hit_rate: float


def equity_curve_metrics(equity: pd.Series, *, periods_per_year: int = 365) -> MetricReport:
    if equity.empty:
        return MetricReport(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    clean = equity.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 2:
        final_equity = float(clean.iloc[-1]) if len(clean) else 1.0
        return MetricReport(final_equity, final_equity - 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    returns = clean.pct_change().dropna()
    total_return = float(clean.iloc[-1] / clean.iloc[0] - 1.0)
    years = max(len(returns) / periods_per_year, 1 / periods_per_year)
    annualized_return = float((1.0 + total_return) ** (1.0 / years) - 1.0)
    annualized_volatility = float(returns.std(ddof=0) * np.sqrt(periods_per_year))
    sharpe = float(annualized_return / annualized_volatility) if annualized_volatility > 0 else 0.0
    drawdown = max_drawdown(clean)
    calmar = float(annualized_return / abs(drawdown)) if drawdown < 0 else 0.0
    hit_rate = float((returns > 0).mean()) if len(returns) else 0.0

    return MetricReport(
        final_equity=float(clean.iloc[-1]),
        total_return=total_return,
        annualized_return=annualized_return,
        annualized_volatility=annualized_volatility,
        sharpe=sharpe,
        max_drawdown=drawdown,
        calmar=calmar,
        hit_rate=hit_rate,
    )


def max_drawdown(equity: pd.Series) -> float:
    running_peak = equity.cummax()
    drawdown = equity / running_peak - 1.0
    return float(drawdown.min())
