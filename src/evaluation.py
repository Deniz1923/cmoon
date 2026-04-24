from __future__ import annotations

import numpy as np
import pandas as pd

from model_types import BacktestResult


def build_backtest_result(equity_curve: pd.DataFrame, liquidation_count: int) -> BacktestResult:
    equity = equity_curve["equity"]
    returns = equity.pct_change().replace([np.inf, -np.inf], np.nan).dropna()

    total_return_pct = (equity.iloc[-1] / equity.iloc[0] - 1.0) * 100
    cummax = equity.cummax()
    drawdowns = equity / cummax - 1.0
    max_drawdown_pct = drawdowns.min() * 100

    if len(returns) > 1 and returns.std(ddof=1) > 0:
        sharpe = float((returns.mean() / returns.std(ddof=1)) * np.sqrt(252))
    else:
        sharpe = 0.0

    win_rate = float((equity_curve["pnl"] > 0).mean()) if len(equity_curve) > 0 else 0.0

    return BacktestResult(
        final_equity=float(equity.iloc[-1]),
        total_return_pct=float(total_return_pct),
        max_drawdown_pct=float(max_drawdown_pct),
        sharpe=sharpe,
        win_rate=win_rate,
        n_steps=int(len(equity_curve)),
        liquidation_count=int(liquidation_count),
    )
