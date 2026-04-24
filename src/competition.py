from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CompetitionRules:
    """
    Yarismadaki kurallari ve backtest varsayimlarini tanimlar.
    """

    coins: tuple[str, ...] = ("Varlik_A", "Varlik_B", "Varlik_C")
    allowed_leverages: tuple[int, ...] = (2, 3, 5, 10)
    default_leverage: int = 2
    allow_long: bool = True
    allow_short: bool = True
    max_total_ratio: float = 1.0
    max_ratio_per_coin: float = 1.0
    initial_equity: float = 3000.0
    fee_rate: float = 0.0004
    min_history: int = 50

    def __post_init__(self) -> None:
        if not self.coins:
            raise ValueError("coins bos olamaz.")
        if len(set(self.coins)) != len(self.coins):
            raise ValueError("coins icinde tekrar eden isim olamaz.")
        if not self.allowed_leverages:
            raise ValueError("allowed_leverages bos olamaz.")
        if self.default_leverage not in self.allowed_leverages:
            raise ValueError("default_leverage, allowed_leverages icinde olmali.")
        if self.max_total_ratio <= 0:
            raise ValueError("max_total_ratio pozitif olmali.")
        if self.max_ratio_per_coin <= 0:
            raise ValueError("max_ratio_per_coin pozitif olmali.")
        if self.initial_equity <= 0:
            raise ValueError("initial_equity pozitif olmali.")
        if self.fee_rate < 0:
            raise ValueError("fee_rate negatif olamaz.")
        if self.min_history < 1:
            raise ValueError("min_history en az 1 olmali.")
        if not (self.allow_long or self.allow_short):
            raise ValueError("En az bir yon aktif olmali (long veya short).")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _tuple_of_str(values: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{field_name} bos olmayan list olmali.")
    out = tuple(str(v) for v in values)
    if any(not x for x in out):
        raise ValueError(f"{field_name} bos string iceremez.")
    return out


def _tuple_of_int(values: Any, field_name: str) -> tuple[int, ...]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{field_name} bos olmayan list olmali.")
    return tuple(int(v) for v in values)


def rules_from_dict(data: dict[str, Any]) -> CompetitionRules:
    if not isinstance(data, dict):
        raise TypeError("Kurallar JSON objesi olmali.")

    payload = {
        "coins": _tuple_of_str(data.get("coins", ["Varlik_A", "Varlik_B", "Varlik_C"]), "coins"),
        "allowed_leverages": _tuple_of_int(data.get("allowed_leverages", [2, 3, 5, 10]), "allowed_leverages"),
        "default_leverage": int(data.get("default_leverage", 2)),
        "allow_long": bool(data.get("allow_long", True)),
        "allow_short": bool(data.get("allow_short", True)),
        "max_total_ratio": float(data.get("max_total_ratio", 1.0)),
        "max_ratio_per_coin": float(data.get("max_ratio_per_coin", 1.0)),
        "initial_equity": float(data.get("initial_equity", 3000.0)),
        "fee_rate": float(data.get("fee_rate", 0.0004)),
        "min_history": int(data.get("min_history", 50)),
    }
    return CompetitionRules(**payload)


def load_rules(path: str | Path | None = None) -> CompetitionRules:
    """
    path yoksa varsayilan kurallari dondurur.
    path varsa JSON dosyasini okuyup kurallari olusturur.
    """
    if path is None:
        return CompetitionRules()

    fp = Path(path)
    if not fp.exists():
        raise FileNotFoundError(f"Kurallar dosyasi bulunamadi: {fp}")
    with fp.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return rules_from_dict(data)
