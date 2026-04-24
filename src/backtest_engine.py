"""
backtest_engine.py — Dev C
Bar-by-bar trade simulator with realistic execution assumptions.

Rules:
- Fill at next bar open only.
- Apply fees and slippage on every fill.
- Apply futures funding costs when enabled.
- Decision functions may receive only data up to the current timestamp.
- Kill-switch check after each trade close.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


def run_backtest(
    ohlcv: pd.DataFrame,
    orders: pd.DataFrame,
    config: dict,
) -> tuple:
    """
    Simulate trades bar by bar.

    Parameters
    ----------
    ohlcv   : full OHLCV DataFrame for the evaluation period
    orders  : DataFrame indexed like ohlcv with columns:
              signal, position_size, stop_loss_price, take_profit_price, max_hold_bars
    config  : full config dict

    Returns
    -------
    (trades_df, equity_curve)
    trades_df    : one row per closed trade
    equity_curve : pd.Series of portfolio value at each bar
    """
    bt_cfg = config["backtest"]
    initial_equity = bt_cfg.get("initial_equity", 100_000)
    fee_bps = bt_cfg.get("fee_bps", 5) / 10_000
    slippage_bps = bt_cfg.get("slippage_bps", 2) / 10_000
    funding_enabled = bt_cfg.get("funding_enabled", False)
    max_dd_threshold = config["risk"].get("kill_switch_drawdown", 0.15)

    equity = float(initial_equity)
    equity_curve: dict = {}
    trades: list = []
    active = None  # type: Optional[dict]
    kill_triggered = False

    timestamps = ohlcv.index.tolist()

    for i, ts in enumerate(timestamps):
        equity_curve[ts] = equity

        if i == 0:
            continue

        bar = ohlcv.loc[ts]
        prev_ts = timestamps[i - 1]

        if kill_triggered:
            continue

        # ---- Manage active position ----
        if active is not None:
            active["bars_held"] += 1
            direction = active["direction"]
            exit_price = None
            close_reason = None

            if direction == 1:
                if bar["low"] <= active["sl"]:
                    exit_price, close_reason = active["sl"], "sl"
                elif bar["high"] >= active["tp"]:
                    exit_price, close_reason = active["tp"], "tp"
            elif direction == -1:
                if bar["high"] >= active["sl"]:
                    exit_price, close_reason = active["sl"], "sl"
                elif bar["low"] <= active["tp"]:
                    exit_price, close_reason = active["tp"], "tp"

            if close_reason is None and active["bars_held"] >= active["max_hold"]:
                exit_price, close_reason = bar["open"], "time"

            if close_reason is not None:
                # Apply slippage on exit (adverse direction)
                slip_adj = slippage_bps * direction
                exit_price_net = exit_price * (1 - slip_adj)

                size = abs(active["size"])
                pnl = direction * size * (exit_price_net - active["entry_price"])
                exit_fee = size * exit_price_net * fee_bps

                # Funding cost for futures
                if funding_enabled and "funding_rate" in ohlcv.columns:
                    funding_sum = ohlcv.loc[active["entry_ts"]:ts, "funding_rate"].sum()
                    pnl -= size * active["entry_price"] * abs(funding_sum)

                net_pnl = pnl - exit_fee
                equity += net_pnl

                trades.append({
                    "entry_ts": active["entry_ts"],
                    "exit_ts": ts,
                    "direction": direction,
                    "entry_price": active["entry_price"],
                    "exit_price": exit_price_net,
                    "size": size,
                    "pnl": net_pnl,
                    "close_reason": close_reason,
                })
                active = None
                equity_curve[ts] = equity

                # Check kill-switch
                eq_series = pd.Series(equity_curve)
                peak = eq_series.cummax().iloc[-1]
                if peak > 0 and (equity - peak) / peak < -max_dd_threshold:
                    kill_triggered = True
                    print(f"[backtest] Kill switch at {ts}: "
                          f"drawdown={(equity - peak) / peak:.2%}")

        # ---- Open new position ----
        if active is None and not kill_triggered and prev_ts in orders.index:
            order = orders.loc[prev_ts]
            sig = int(order["signal"])
            pos_size = order["position_size"]

            if sig != 0 and not np.isnan(pos_size) and abs(pos_size) > 0:
                fill_price = bar["open"] * (1 + slippage_bps * sig)
                entry_fee = abs(pos_size) * fill_price * fee_bps
                equity -= entry_fee

                active = {
                    "direction": sig,
                    "entry_price": fill_price,
                    "size": pos_size,
                    "sl": float(order["stop_loss_price"]),
                    "tp": float(order["take_profit_price"]),
                    "max_hold": int(order["max_hold_bars"]),
                    "bars_held": 0,
                    "entry_ts": ts,
                }

        equity_curve[ts] = equity

    # Close any open position at end of data
    if active is not None:
        last_bar = ohlcv.iloc[-1]
        direction = active["direction"]
        exit_price = float(last_bar["close"]) * (1 - slippage_bps * direction)
        size = abs(active["size"])
        pnl = direction * size * (exit_price - active["entry_price"])
        exit_fee = size * exit_price * fee_bps
        equity += pnl - exit_fee
        last_ts = ohlcv.index[-1]
        trades.append({
            "entry_ts": active["entry_ts"],
            "exit_ts": last_ts,
            "direction": direction,
            "entry_price": active["entry_price"],
            "exit_price": exit_price,
            "size": size,
            "pnl": pnl - exit_fee,
            "close_reason": "end_of_data",
        })
        equity_curve[last_ts] = equity

    _trade_cols = [
        "entry_ts", "exit_ts", "direction", "entry_price",
        "exit_price", "size", "pnl", "close_reason",
    ]
    trades_df = pd.DataFrame(trades, columns=_trade_cols) if trades else pd.DataFrame(columns=_trade_cols)
    equity_series = pd.Series(equity_curve)

    return trades_df, equity_series