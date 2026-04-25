"""
Bounded backtest helper for local research infrastructure.

cnlib.backtest.run() only supports start_candle. This module mirrors its
runtime semantics while adding an inclusive end_candle for walk-forward folds
and focused smoke tests.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from cnlib.backtest import BacktestResult
from cnlib.base_strategy import BaseStrategy
from cnlib.portfolio import Portfolio
from cnlib.validator import ValidationError, validate


def run_backtest_window(
    strategy: BaseStrategy,
    initial_capital: float = 3000.0,
    start_candle: int = 0,
    end_candle: int | None = None,
    data_dir: Path | None = None,
    silent: bool = False,
    verbose: bool = False,
) -> BacktestResult:
    """
    Run a backtest over an inclusive candle window.

    Args mirror cnlib.backtest.run(), with end_candle added. If end_candle is
    None, the run continues through the last available candle.
    """
    if verbose:
        setattr(strategy, "_verbose", True)

    strategy.get_data(data_dir)
    portfolio = Portfolio(initial_capital=initial_capital)

    full_data = _full_coin_data(strategy)
    first_coin = next(iter(full_data.values()))
    total_available = len(first_coin)
    start_candle, end_candle = _validate_window(start_candle, end_candle, total_available)

    portfolio_series: list[dict] = []
    trade_history: list[dict] = []
    failed_open_history: list[dict] = []
    total_trades = 0
    validation_errors = 0
    strategy_errors = 0
    failed_opens = 0

    if not silent:
        print(
            "Backtest starting: "
            f"{total_available} candles, start_candle={start_candle}, end_candle={end_candle}"
        )

    for i in range(start_candle, end_candle + 1):
        strategy.candle_index = i

        data = strategy._candle_data(i)
        prices = strategy.current_prices(i)
        highs = strategy.current_highs(i)
        lows = strategy.current_lows(i)
        portfolio.update_prices(prices, highs, lows)

        try:
            decisions: list[dict[str, Any]] = strategy.predict(data)
        except Exception as exc:
            if not silent:
                print(f"  [Candle {i}] predict() EXCEPTION ({type(exc).__name__}): {exc}")
            strategy_errors += 1
            _record(portfolio_series, i, portfolio, prices)
            continue

        try:
            validate(decisions)
        except ValidationError as exc:
            if not silent:
                print(f"  [Candle {i}] ValidationError: {exc}")
            validation_errors += 1
            _record(portfolio_series, i, portfolio, prices)
            continue

        positions_before = dict(portfolio.positions) if verbose else {}
        turn = portfolio.update_positions(decisions, prices, highs, lows)
        total_trades += len(turn["opened"]) + len(turn["closed"])

        if verbose:
            _print_verbose_candle(
                i, prices, highs, lows,
                positions_before,
                getattr(strategy, "_verbose_log", {}),
                decisions,
                turn,
                portfolio,
            )

        if turn["opened"] or turn["closed"] or turn["liquidated"]:
            trade_history.append({
                "candle_index": i,
                "timestamp": first_coin.iloc[i]["Date"],
                "opened": turn["opened"],
                "closed": turn["closed"],
                "liquidated": turn["liquidated"],
                "portfolio_value": round(turn["portfolio_value"], 2),
            })

        for fail in turn.get("failed_opens", []):
            failed_opens += 1
            failed_open_history.append({
                "candle_index": i,
                "timestamp": first_coin.iloc[i]["Date"],
                "coin": fail["coin"],
                "error": fail["error"],
            })

        _record(portfolio_series, i, portfolio, prices)

        if not silent and i % 100 == 0:
            print(f"  Candle {i:>4}/{end_candle}  Portfolio: ${portfolio.portfolio_value:,.2f}")

    summary = portfolio.summary()
    import inspect as _inspect
    _result_params = set(_inspect.signature(BacktestResult.__init__).parameters)
    _kwargs: dict = dict(
        initial_capital=initial_capital,
        final_portfolio_value=summary["portfolio_value"],
        net_pnl=summary["net_pnl"],
        return_pct=summary["return_pct"],
        total_candles=end_candle - start_candle + 1,
        total_trades=total_trades,
        total_liquidations=summary["total_liquidations"],
        total_liquidation_loss=summary["total_liquidation_loss"],
        validation_errors=validation_errors,
        strategy_errors=strategy_errors,
        portfolio_series=portfolio_series,
        trade_history=trade_history,
    )
    if "failed_opens" in _result_params:
        _kwargs["failed_opens"] = failed_opens
    if "failed_open_history" in _result_params:
        _kwargs["failed_open_history"] = failed_open_history
    return BacktestResult(**_kwargs)


def _full_coin_data(strategy: BaseStrategy) -> dict:
    """Return full loaded data across cnlib 0.1.3 and 0.1.4 style strategies."""
    full_data = getattr(strategy, "_full_data", None)
    if isinstance(full_data, dict) and full_data:
        return full_data
    coin_data = getattr(strategy, "coin_data", None)
    if isinstance(coin_data, dict) and coin_data:
        return coin_data
    raise ValueError("strategy.get_data() did not load any coin data")


def _validate_window(
    start_candle: int,
    end_candle: int | None,
    total_available: int,
) -> tuple[int, int]:
    if total_available <= 0:
        raise ValueError("cannot backtest an empty dataset")
    if not isinstance(start_candle, int) or isinstance(start_candle, bool):
        raise ValueError(f"start_candle must be an integer, got {start_candle!r}")
    if start_candle < 0:
        raise ValueError(f"start_candle must be >= 0, got {start_candle}")
    if start_candle >= total_available:
        raise ValueError(
            f"start_candle {start_candle} is out of range for {total_available} candles "
            f"(max {total_available - 1})"
        )

    if end_candle is None:
        end_candle = total_available - 1
    elif not isinstance(end_candle, int) or isinstance(end_candle, bool):
        raise ValueError(f"end_candle must be an integer or None, got {end_candle!r}")

    if end_candle < start_candle:
        raise ValueError(
            f"end_candle must be >= start_candle, got {end_candle} < {start_candle}"
        )
    if end_candle >= total_available:
        raise ValueError(
            f"end_candle {end_candle} is out of range for {total_available} candles "
            f"(max {total_available - 1})"
        )

    return start_candle, end_candle


def _record(
    series: list[dict],
    index: int,
    portfolio: Portfolio,
    prices: dict[str, float],
) -> None:
    series.append({
        "candle_index": index,
        "portfolio_value": round(portfolio.portfolio_value, 2),
        "cash": round(portfolio.cash, 2),
        **{f"{coin}_price": round(price, 4) for coin, price in prices.items()},
    })


_COIN_SHORT = {
    "kapcoin-usd_train": "kapcoin",
    "metucoin-usd_train": "metucoin",
    "tamcoin-usd_train": "tamcoin",
}
_SEP = "─" * 62


def _short(coin: str) -> str:
    return _COIN_SHORT.get(coin, coin)


def _print_verbose_candle(
    candle_idx: int,
    prices: dict[str, float],
    highs: dict[str, float],
    lows: dict[str, float],
    positions_before: dict,
    verbose_log: dict,
    decisions: list[dict],
    turn: dict,
    portfolio: Portfolio,
) -> None:
    print(_SEP)
    print(f"CANDLE {candle_idx}")
    print(_SEP)

    # ── Prices ──────────────────────────────────────────────────────────────
    print("\nPRICES")
    for coin, close in prices.items():
        h = highs.get(coin, close)
        l = lows.get(coin, close)
        print(f"  {_short(coin):<12}  H={h:<10.4f} L={l:<10.4f} C={close:.4f}")

    # ── Open positions at start of candle ───────────────────────────────────
    print("\nOPEN POSITIONS (start of candle)")
    if positions_before:
        for coin, pos in positions_before.items():
            close = prices.get(coin, pos.entry_price)
            pnl = pos.pnl(close)
            pnl_pct = (pnl / pos.capital * 100) if pos.capital > 0 else 0.0
            direction = "LONG" if pos.direction == 1 else "SHORT"
            liq = pos.liquidation_price
            sign = "+" if pnl >= 0 else ""
            tp_str = f"  TP={pos.take_profit:.4f}" if pos.take_profit is not None else ""
            sl_str = f"  SL={pos.stop_loss:.4f}" if pos.stop_loss is not None else ""
            print(
                f"  {_short(coin):<12}  {direction} ×{pos.leverage}"
                f"  entry={pos.entry_price:.4f}  capital=${pos.capital:.2f}"
                f"  liq={liq:.4f}{tp_str}{sl_str}"
            )
            print(f"               unrealised P&L: {sign}${pnl:.2f} ({sign}{pnl_pct:.1f}%)")
    else:
        print("  (no open positions)")

    # ── Per-coin signal reasoning ────────────────────────────────────────────
    print("\nSIGNALS")
    for coin in list(prices.keys()):
        log = verbose_log.get(coin, {})
        print(f"\n  {_short(coin)}")

        if not log.get("has_data", True):
            print("    (no data this candle)")
            continue

        if log.get("insufficient_data"):
            print("    Rule:  insufficient data — FLAT")
            continue

        # Rule signal
        regime = log.get("regime")
        bb_w = log.get("bb_width")
        rule_sig = log.get("rule_signal", 0)
        bb_str = f"BB_w={bb_w:.3f}" if bb_w is not None else "BB_w=?"

        if regime == "trending":
            ef = log.get("ema_fast")
            es = log.get("ema_slow")
            rsi = log.get("rsi")
            vol_r = log.get("vol_ratio")
            ema_cmp = ">" if (ef and es and ef > es) else "<"
            ema_str = f"EMA20{ema_cmp}EMA50" if (ef is not None and es is not None) else "EMA=?"
            rsi_str = f"RSI={rsi:.1f}" if rsi is not None else "RSI=?"
            vol_str = f"vol_ratio={vol_r:.2f}" if vol_r is not None else ""
            rule_desc = f"trending ({bb_str})  {ema_str}  {rsi_str}  {vol_str}"
        elif regime == "ranging":
            rsi = log.get("rsi")
            bp = log.get("bb_pct")
            rsi_str = f"RSI={rsi:.1f}" if rsi is not None else "RSI=?"
            bp_str = f"BB_pct={bp:.2f}" if bp is not None else "BB_pct=?"
            rule_desc = f"ranging ({bb_str})  {rsi_str}  {bp_str}"
        elif regime == "ambiguous":
            rule_desc = f"ambiguous regime ({bb_str})"
        else:
            rule_desc = f"regime=? ({bb_str})"

        sig_word = {1: "LONG", -1: "SHORT", 0: "FLAT"}.get(rule_sig, "?")
        print(f"    Rule:  {rule_desc}  →  {sig_word}")

        if rule_sig == 0:
            continue

        # ML
        ml_prob = log.get("ml_prob_up")
        if ml_prob is None:
            reason = "model not loaded" if not log.get("ml_available", False) else "insufficient features"
            print(f"    ML:    {reason}  →  SKIP")
            continue

        ml_sig = log.get("ml_signal", 0)
        conf = log.get("confidence", 0.0)
        ml_dir = {1: "LONG", -1: "SHORT", 0: "FLAT"}.get(ml_sig, "?")
        ml_agrees = log.get("ml_agrees", False)
        agree_str = "agrees" if ml_agrees else "DISAGREES — SKIP"
        print(f"    ML:    prob_up={ml_prob:.3f}  conf={conf:.1%}  {ml_dir}  {agree_str}")

        if not ml_agrees:
            continue

        # Confidence gate
        is_hold = log.get("is_hold", False)
        min_conf = log.get("min_conf", 0.52)
        entry_type = "hold" if is_hold else "new entry"
        if conf >= min_conf:
            print(f"    Gate:  {entry_type}  conf={conf:.1%} ≥ min={min_conf:.0%}  →  PASS")
        else:
            print(f"    Gate:  {entry_type}  conf={conf:.1%} < min={min_conf:.0%}  →  SKIP (low confidence)")
            continue

        # Risk params (only present when candidate was produced)
        atr = log.get("atr")
        atr_pct = log.get("atr_pct")
        rr = log.get("risk_reward")
        lev = log.get("leverage")
        sl = log.get("stop_loss")
        tp = log.get("take_profit")
        if atr is not None:
            atr_str = f"ATR={atr:.4f} ({atr_pct:.2%})" if atr_pct is not None else f"ATR={atr:.4f}"
            rr_str = f"RR={rr:.1f}×" if rr is not None else ""
            lev_str = f"lev={lev}×" if lev is not None else ""
            tp_str = f"TP={tp:.4f}" if tp is not None else ""
            sl_str = f"SL={sl:.4f}" if sl is not None else ""
            print(f"    Risk:  {atr_str}  {rr_str}  {lev_str}  {tp_str}  {sl_str}")

    # ── Final decisions ──────────────────────────────────────────────────────
    print("\nFINAL DECISIONS")
    for d in decisions:
        coin = d["coin"]
        sig = d["signal"]
        if sig == 0:
            print(f"  {_short(coin):<12}  FLAT")
        else:
            direction = "LONG" if sig == 1 else "SHORT"
            alloc = d.get("allocation", 0.0)
            lev = d.get("leverage", 1)
            tp = d.get("take_profit")
            sl = d.get("stop_loss")
            tp_str = f"  TP={tp:.4f}" if tp is not None else ""
            sl_str = f"  SL={sl:.4f}" if sl is not None else ""
            print(f"  {_short(coin):<12}  {direction}  alloc={alloc:.1%}  lev={lev}×{tp_str}{sl_str}")

    # ── Execution results ────────────────────────────────────────────────────
    print("\nEXECUTION")
    liquidated = turn.get("liquidated", [])
    tp_sl_closed = turn.get("tp_sl_closed", [])
    all_closed = turn.get("closed", [])
    opened = turn.get("opened", [])
    failed = turn.get("failed_opens", [])
    signal_closed = [c for c in all_closed if c not in tp_sl_closed and c not in liquidated]

    def _coins(lst: list) -> str:
        return "  ".join(_short(c) for c in lst) if lst else "—"

    if liquidated:
        print(f"  LIQUIDATED:   {_coins(liquidated)}")
        for c in liquidated:
            pos = positions_before.get(c)
            if pos:
                print(f"    {_short(c)}: entire capital ${pos.capital:.2f} lost"
                      f"  (liq price {pos.liquidation_price:.4f})")
    else:
        print(f"  Liquidated:   —")

    if tp_sl_closed:
        print(f"  TP/SL hit:    {_coins(tp_sl_closed)}")
        for c in tp_sl_closed:
            pos = positions_before.get(c)
            if pos:
                close_px = prices.get(c, pos.entry_price)
                pnl = pos.pnl(close_px)
                sign = "+" if pnl >= 0 else ""
                triggered = "TP" if (
                    (pos.direction == 1 and close_px >= (pos.take_profit or 0))
                    or (pos.direction == -1 and close_px <= (pos.take_profit or float("inf")))
                ) else "SL"
                print(f"    {_short(c)}: {triggered} triggered  realised {sign}${pnl:.2f}")
    else:
        print(f"  TP/SL hit:    —")

    print(f"  Closed:       {_coins(signal_closed)}")
    print(f"  Opened:       {_coins(opened)}")

    if failed:
        print(f"  Failed opens: {len(failed)}")
        for f in failed:
            print(f"    {_short(f['coin'])}: {f['error']}")
    else:
        print(f"  Failed opens: —")

    # ── End-of-candle state ──────────────────────────────────────────────────
    print(f"\nCANDLE END")
    print(
        f"  Portfolio: ${portfolio.portfolio_value:,.2f}"
        f"  Cash: ${portfolio.cash:,.2f}"
        f"  Open positions: {len(portfolio.positions)}/3"
    )
    for coin, pos in portfolio.positions.items():
        close = prices.get(coin, pos.entry_price)
        pnl = pos.pnl(close)
        pnl_pct = (pnl / pos.capital * 100) if pos.capital > 0 else 0.0
        direction = "LONG" if pos.direction == 1 else "SHORT"
        sign = "+" if pnl >= 0 else ""
        print(
            f"  {_short(coin):<12}  {direction} ×{pos.leverage}"
            f"  entry={pos.entry_price:.4f}  close={close:.4f}"
            f"  unrealised {sign}${pnl:.2f} ({sign}{pnl_pct:.1f}%)"
        )
    print()
