from __future__ import annotations

from typing import Iterable

from competition import CompetitionRules
from model_types import Signal


def _as_signal(item: Signal | dict) -> Signal:
    if isinstance(item, Signal):
        return item
    if not isinstance(item, dict):
        raise TypeError("Signal elemani dict veya Signal olmali.")

    required = {"coin", "signal", "ratio", "leverage"}
    missing = required - set(item.keys())
    if missing:
        raise ValueError(f"Signal dict eksik alan(lar): {sorted(missing)}")

    return Signal(
        coin=str(item["coin"]),
        signal=int(item["signal"]),
        ratio=float(item["ratio"]),
        leverage=int(item["leverage"]),
    )


def validate_signals(
    raw_signals: Iterable[Signal | dict],
    rules: CompetitionRules,
) -> list[Signal]:
    signals = [_as_signal(s) for s in raw_signals]
    if not signals:
        return [Signal(coin=c, signal=0, ratio=0.0, leverage=rules.default_leverage) for c in rules.coins]

    seen: set[str] = set()
    normalized: list[Signal] = []
    ratio_sum = 0.0

    for sig in signals:
        if sig.coin not in rules.coins:
            raise ValueError(f"Izinli olmayan coin: {sig.coin}")
        if sig.coin in seen:
            raise ValueError(f"Ayni coin icin birden fazla sinyal: {sig.coin}")
        seen.add(sig.coin)

        if sig.signal not in (-1, 0, 1):
            raise ValueError(f"signal -1/0/1 olmali: {sig.coin}")
        if sig.signal == 1 and not rules.allow_long:
            raise ValueError("Bu yarismada long kapali.")
        if sig.signal == -1 and not rules.allow_short:
            raise ValueError("Bu yarismada short kapali.")
        if sig.leverage not in rules.allowed_leverages:
            raise ValueError(
                f"leverage izinli degil ({sig.leverage}). Izinliler: {list(rules.allowed_leverages)}"
            )
        if not (0.0 <= sig.ratio <= rules.max_ratio_per_coin):
            raise ValueError(
                f"ratio 0..{rules.max_ratio_per_coin:.2f} araliginda olmali: {sig.coin}"
            )
        if sig.signal == 0 and sig.ratio != 0:
            raise ValueError(f"signal=0 iken ratio 0 olmali: {sig.coin}")
        if sig.signal != 0 and sig.ratio == 0:
            raise ValueError(f"signal!=0 iken ratio 0 olamaz: {sig.coin}")

        ratio_sum += sig.ratio
        normalized.append(sig)

    if ratio_sum > rules.max_total_ratio + 1e-9:
        raise ValueError(
            f"Toplam ratio {rules.max_total_ratio:.2f}'i gecemez, gelen: {ratio_sum:.4f}"
        )

    missing_coins = [c for c in rules.coins if c not in seen]
    normalized.extend(
        [Signal(coin=c, signal=0, ratio=0.0, leverage=rules.default_leverage) for c in missing_coins]
    )
    return sorted(normalized, key=lambda s: s.coin)
