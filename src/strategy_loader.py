from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Type

from strategy_base import BaseStrategy


def load_strategy_class(strategy_file: str | Path, class_name: str) -> Type[BaseStrategy]:
    strategy_file = Path(strategy_file)
    if not strategy_file.exists():
        raise FileNotFoundError(f"Strategy dosyasi yok: {strategy_file}")

    spec = importlib.util.spec_from_file_location("user_strategy_module", str(strategy_file))
    if spec is None or spec.loader is None:
        raise ImportError(f"Import edilemedi: {strategy_file}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cls = getattr(module, class_name, None)
    if cls is None:
        raise AttributeError(f"{class_name} bulunamadi: {strategy_file}")
    if not issubclass(cls, BaseStrategy):
        raise TypeError(f"{class_name}, BaseStrategy'den turemeli.")
    return cls
