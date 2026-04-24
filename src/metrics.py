"""
metrics.py — Dev C
Compute strategy performance statistics and produce a tear-sheet plot.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


def evaluate(equity: pd.Series, trades: pd.DataFrame) -> dict:
    """
    Compute performance metrics from an equity curve and trades log.

    Assumes hourly bars (annualization factor = sqrt(365 * 24)).
    """
    if len(equity) < 2:
        return {}

    returns = equity.pct_change().dropna()
    annual_factor = np.sqrt(365 * 24)

    mean_ret = returns.mean()
    std_ret = returns.std()

    sharpe = (mean_ret / (std_ret + 1e-9)) * annual_factor if std_ret > 0 else 0.0

    downside = returns[returns < 0].std()
    sortino = (mean_ret / (downside + 1e-9)) * annual_factor if downside > 0 else 0.0

    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    max_dd = float(drawdown.min())

    years = max((equity.index[-1] - equity.index[0]).days / 365.0, 1 / 365)
    total_return = (equity.iloc[-1] / equity.iloc[0]) - 1.0
    annual_return = float((1.0 + total_return) ** (1.0 / years) - 1.0)
    calmar = annual_return / abs(max_dd + 1e-9) if max_dd < 0 else 0.0

    if len(trades) > 0 and "pnl" in trades.columns:
        win_rate = float((trades["pnl"] > 0).mean())
        avg_trade = float(trades["pnl"].mean())
        trade_count = int(len(trades))
    else:
        win_rate = avg_trade = 0.0
        trade_count = 0

    return {
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "max_drawdown": round(max_dd, 4),
        "calmar": round(calmar, 4),
        "annual_return": round(annual_return, 4),
        "total_return": round(total_return, 4),
        "win_rate": round(win_rate, 4),
        "avg_trade_pnl": round(avg_trade, 2),
        "trade_count": trade_count,
        "turnover_per_year": round(trade_count / years, 1),
    }


def plot_equity_curve(
    equity: pd.Series,
    trades: pd.DataFrame,
    output_path: str,
    metrics_dict: Optional[dict] = None,
) -> None:
    """Save a 3-panel tear-sheet: equity curve, drawdown, and monthly returns."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("[metrics] matplotlib not installed — skipping plot. Run: uv add matplotlib")
        return

    from pathlib import Path
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 1], hspace=0.35)

    # Panel 1 — equity curve
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(equity.index, equity.values, color="royalblue", linewidth=1.4, label="Equity")
    ax1.set_title("Equity Curve", fontsize=13)
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Panel 2 — drawdown
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    peak = equity.cummax()
    dd = (equity - peak) / peak * 100
    ax2.fill_between(dd.index, dd.values, 0, color="crimson", alpha=0.45, label="Drawdown")
    ax2.set_ylabel("Drawdown (%)")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Panel 3 — monthly returns bar chart
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    monthly = equity.resample("ME").last().pct_change().dropna() * 100
    colors = ["seagreen" if r >= 0 else "crimson" for r in monthly.values]
    ax3.bar(monthly.index, monthly.values, color=colors, width=20, alpha=0.75)
    ax3.axhline(0, color="black", linewidth=0.8)
    ax3.set_ylabel("Monthly Return (%)")
    ax3.grid(True, alpha=0.3)

    # Metrics footer
    if metrics_dict:
        footer = (
            f"Sharpe: {metrics_dict.get('sharpe', '—')}  |  "
            f"Sortino: {metrics_dict.get('sortino', '—')}  |  "
            f"Max DD: {metrics_dict.get('max_drawdown', 0):.1%}  |  "
            f"Win Rate: {metrics_dict.get('win_rate', 0):.1%}  |  "
            f"Trades: {metrics_dict.get('trade_count', '—')}"
        )
        fig.text(0.5, 0.005, footer, ha="center", fontsize=9, style="italic", color="#333333")

    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[metrics] Tear-sheet saved to {output_path}")