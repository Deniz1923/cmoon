"""
backtest_engine.py — Dev C
Bar-by-bar trade simulator with realistic execution assumptions.

Rules:
- Fill at next bar open only.
- Apply fees and slippage on every fill.
- Apply futures funding costs when enabled.
- Decision functions may receive only data up to the current timestamp.
- Kill-switch check after each trade close.
- One open position per symbol at a time; up to max_open_positions symbols simultaneously.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def run_backtest(
    ohlcv_by_symbol: dict,
    orders_by_symbol: dict,
    config: dict,
) -> tuple:
    """
    Multi-symbol bar-by-bar backtest.

    Parameters
    ----------
    ohlcv_by_symbol   : dict mapping symbol → OHLCV DataFrame
    orders_by_symbol  : dict mapping symbol → orders DataFrame (signal, position_size,
                        stop_loss_price, take_profit_price, max_hold_bars)
    config            : full config dict

    Returns
    -------
    (trades_df, equity_curve)
    trades_df    : one row per closed trade, includes 'symbol' column
    equity_curve : pd.Series of portfolio value indexed by UTC timestamp
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
    active: dict = {}   # symbol → position dict; one position per symbol
    kill_triggered = False

    # Global clock = union of all symbol timestamps
    all_timestamps = sorted(
        set().union(*[set(df.index) for df in ohlcv_by_symbol.values()])
    )

    for i, ts in enumerate(all_timestamps):
        equity_curve[ts] = equity

        if i == 0:
            continue

        if kill_triggered:
            continue

        prev_ts = all_timestamps[i - 1]

        for sym, ohlcv in ohlcv_by_symbol.items():
            if ts not in ohlcv.index:
                continue

            bar = ohlcv.loc[ts]
            orders = orders_by_symbol.get(sym)

            # ---- Manage active position for this symbol ----
            pos: Optional[dict] = active.get(sym)
            if pos is not None:
                pos["bars_held"] += 1
                direction = pos["direction"]
                exit_price = None
                close_reason = None

                if direction == 1:
                    if bar["low"] <= pos["sl"]:
                        exit_price, close_reason = pos["sl"], "sl"
                    elif bar["high"] >= pos["tp"]:
                        exit_price, close_reason = pos["tp"], "tp"
                elif direction == -1:
                    if bar["high"] >= pos["sl"]:
                        exit_price, close_reason = pos["sl"], "sl"
                    elif bar["low"] <= pos["tp"]:
                        exit_price, close_reason = pos["tp"], "tp"

                if close_reason is None and pos["bars_held"] >= pos["max_hold"]:
                    exit_price, close_reason = bar["open"], "time"

                if close_reason is not None:
                    slip_adj = slippage_bps * direction
                    exit_price_net = exit_price * (1 - slip_adj)

                    size = abs(pos["size"])
                    pnl = direction * size * (exit_price_net - pos["entry_price"])
                    exit_fee = size * exit_price_net * fee_bps

                    if funding_enabled and "funding_rate" in ohlcv.columns:
                        funding_sum = ohlcv.loc[pos["entry_ts"]:ts, "funding_rate"].sum()
                        pnl -= size * pos["entry_price"] * abs(funding_sum)

                    net_pnl = pnl - exit_fee
                    equity += net_pnl

                    trades.append({
                        "symbol": sym,
                        "entry_ts": pos["entry_ts"],
                        "exit_ts": ts,
                        "direction": direction,
                        "entry_price": pos["entry_price"],
                        "exit_price": exit_price_net,
                        "size": size,
                        "pnl": net_pnl,
                        "close_reason": close_reason,
                    })
                    del active[sym]
                    equity_curve[ts] = equity

                    # Check kill-switch
                    eq_series = pd.Series(equity_curve)
                    peak = eq_series.cummax().iloc[-1]
                    if peak > 0 and (equity - peak) / peak < -max_dd_threshold:
                        kill_triggered = True
                        print(
                            f"[backtest] Kill switch at {ts}: "
                            f"drawdown={(equity - peak) / peak:.2%}"
                        )

            # ---- Open new position for this symbol ----
            if (
                sym not in active
                and not kill_triggered
                and orders is not None
                and prev_ts in orders.index
            ):
                order = orders.loc[prev_ts]
                sig = int(order["signal"])
                pos_size = order["position_size"]

                if sig != 0 and not np.isnan(pos_size) and abs(pos_size) > 0:
                    fill_price = bar["open"] * (1 + slippage_bps * sig)
                    entry_fee = abs(pos_size) * fill_price * fee_bps
                    equity -= entry_fee

                    active[sym] = {
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

    # Close any open positions at end of data
    for sym, pos in list(active.items()):
        ohlcv = ohlcv_by_symbol[sym]
        last_bar = ohlcv.iloc[-1]
        direction = pos["direction"]
        exit_price = float(last_bar["close"]) * (1 - slippage_bps * direction)
        size = abs(pos["size"])
        pnl = direction * size * (exit_price - pos["entry_price"])
        exit_fee = size * exit_price * fee_bps
        equity += pnl - exit_fee
        trades.append({
            "symbol": sym,
            "entry_ts": pos["entry_ts"],
            "exit_ts": ohlcv.index[-1],
            "direction": direction,
            "entry_price": pos["entry_price"],
            "exit_price": exit_price,
            "size": size,
            "pnl": pnl - exit_fee,
            "close_reason": "end_of_data",
        })

    if all_timestamps:
        equity_curve[all_timestamps[-1]] = equity

    _trade_cols = [
        "symbol", "entry_ts", "exit_ts", "direction", "entry_price",
        "exit_price", "size", "pnl", "close_reason",
    ]
    trades_df = (
        pd.DataFrame(trades, columns=_trade_cols)
        if trades
        else pd.DataFrame(columns=_trade_cols)
    )
    equity_series = pd.Series(equity_curve).sort_index()

    return trades_df, equity_series
