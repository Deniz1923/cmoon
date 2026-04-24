from __future__ import annotations

import pandas as pd

from competition import CompetitionRules
from model_types import BacktestConfig, StrategyState
from strategy_base import BaseStrategy
from validation import validate_signals


class BacktestEngine:
    def __init__(self, rules: CompetitionRules, config: BacktestConfig | None = None):
        self.rules = rules
        self.config = config or BacktestConfig()

    @property
    def initial_equity(self) -> float:
        return self.config.initial_equity if self.config.initial_equity is not None else self.rules.initial_equity

    @property
    def fee_rate(self) -> float:
        return self.config.fee_rate if self.config.fee_rate is not None else self.rules.fee_rate

    @property
    def min_history(self) -> int:
        return self.config.min_history if self.config.min_history is not None else self.rules.min_history

    def run(
        self,
        strategy: BaseStrategy,
        data: dict[str, pd.DataFrame],
    ) -> tuple[pd.DataFrame, int]:
        self._validate_data(data)

        n = len(next(iter(data.values())))
        min_history = max(self.min_history, 2)
        if n <= min_history + 1:
            raise ValueError(f"Veri yetersiz: n={n}, min_history={min_history}")

        equity = self.initial_equity
        peak_equity = equity
        liquidation_count = 0
        step_rows: list[dict] = [{"t": 0, "equity": equity, "pnl": 0.0, "fees": 0.0, "drawdown": 0.0}]

        for t in range(min_history, n - 1):
            current_window = {coin: df.iloc[: t + 1].copy() for coin, df in data.items()}
            state = StrategyState(
                step=t,
                equity=equity,
                peak_equity=peak_equity,
                drawdown=(equity / peak_equity - 1.0) if peak_equity > 0 else 0.0,
                coins=self.rules.coins,
            )

            raw_signals = strategy.predict(current_window, state)
            signals = validate_signals(raw_signals, self.rules)

            pnl, fees, liq_hits = self._step_pnl(
                signals=signals,
                current_bars={coin: df.iloc[t] for coin, df in data.items()},
                next_bars={coin: df.iloc[t + 1] for coin, df in data.items()},
                equity=equity,
            )
            liquidation_count += liq_hits

            equity = max(0.0, equity + pnl - fees)
            peak_equity = max(peak_equity, equity)
            drawdown = (equity / peak_equity - 1.0) if peak_equity > 0 else -1.0

            row = {
                "t": t + 1,
                "equity": equity,
                "pnl": pnl,
                "fees": fees,
                "drawdown": drawdown,
                "liquidations_so_far": liquidation_count,
            }
            for sig in signals:
                row[f"{sig.coin}_signal"] = sig.signal
                row[f"{sig.coin}_ratio"] = sig.ratio
                row[f"{sig.coin}_lev"] = sig.leverage
            step_rows.append(row)

            if equity <= 0:
                break

        equity_curve = pd.DataFrame(step_rows).set_index("t")
        return equity_curve, liquidation_count

    def _step_pnl(
        self,
        signals,
        current_bars: dict[str, pd.Series],
        next_bars: dict[str, pd.Series],
        equity: float,
    ) -> tuple[float, float, int]:
        pnl = 0.0
        fees = 0.0
        liq_hits = 0

        for sig in signals:
            if sig.signal == 0 or sig.ratio == 0:
                continue

            entry = float(current_bars[sig.coin]["close"])
            nxt = next_bars[sig.coin]
            close_next = float(nxt["close"])
            high_next = float(nxt["high"])
            low_next = float(nxt["low"])

            allocated_equity = equity * sig.ratio
            if allocated_equity <= 0:
                continue

            notional = allocated_equity * sig.leverage
            fees += notional * self.fee_rate * 2

            if sig.signal == 1:
                liq_price = entry * (1.0 - 1.0 / sig.leverage)
                liquidated = low_next <= liq_price
                position_return = -1.0 if liquidated else sig.leverage * ((close_next - entry) / entry)
            else:
                liq_price = entry * (1.0 + 1.0 / sig.leverage)
                liquidated = high_next >= liq_price
                position_return = -1.0 if liquidated else sig.leverage * ((entry - close_next) / entry)

            if liquidated:
                liq_hits += 1
            position_return = max(-1.0, position_return)
            pnl += allocated_equity * position_return

        return pnl, fees, liq_hits

    def _validate_data(self, data: dict[str, pd.DataFrame]) -> None:
        if set(data.keys()) != set(self.rules.coins):
            raise ValueError(f"Data coin seti {self.rules.coins} olmali.")

        lengths = {coin: len(df) for coin, df in data.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"Coin uzunluklari esit olmali: {lengths}")

        required = {"open", "high", "low", "close", "volume"}
        for coin, df in data.items():
            missing = required - set(df.columns)
            if missing:
                raise ValueError(f"{coin} eksik kolon(lar): {sorted(missing)}")
