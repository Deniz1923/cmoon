from __future__ import annotations

from dataclasses import dataclass


ASSETS = ("Varlik_A", "Varlik_B", "Varlik_C")
ALLOWED_LEVERAGE = (2, 3, 5, 10)


@dataclass(frozen=True)
class CompetitionConfig:
    assets: tuple[str, ...] = ASSETS
    warmup_bars: int = 20
    max_total_allocation: float = 1.0
    max_asset_allocation: float = 0.42
    target_volatility: float = 0.018
    fee_bps: float = 2.0
    seed: int = 42


DEFAULT_CONFIG = CompetitionConfig()
