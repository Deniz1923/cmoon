from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).lower() for c in out.columns]

    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
        out = out.set_index("timestamp")
    elif "time_index" in out.columns:
        out = out.set_index("time_index")

    missing = [c for c in REQUIRED_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"Eksik OHLCV kolon(lar)i: {missing}")

    out = out[list(REQUIRED_COLUMNS)].copy()
    return out.sort_index()


def discover_coins(split_dir: str | Path) -> tuple[str, ...]:
    split_dir = Path(split_dir)
    return tuple(sorted(p.stem for p in split_dir.glob("*.parquet")))


def load_split(
    split_dir: str | Path,
    coins: Sequence[str] | None = None,
    strict: bool = True,
) -> dict[str, pd.DataFrame]:
    split_dir = Path(split_dir)
    if not split_dir.exists():
        raise FileNotFoundError(f"Klasor bulunamadi: {split_dir}")

    coin_list = tuple(coins) if coins else discover_coins(split_dir)
    if not coin_list:
        raise ValueError(f"{split_dir} altinda parquet coin dosyasi bulunamadi.")

    raw: dict[str, pd.DataFrame] = {}
    missing_files: list[str] = []
    for coin in coin_list:
        fp = split_dir / f"{coin}.parquet"
        if not fp.exists():
            if strict:
                missing_files.append(str(fp))
            continue
        raw[coin] = _normalize_ohlcv(pd.read_parquet(fp))

    if missing_files:
        raise FileNotFoundError("Eksik coin parquet dosyasi:\n- " + "\n- ".join(missing_files))
    if not raw:
        raise ValueError("Yuklenebilen coin verisi yok.")

    common_index = None
    for df in raw.values():
        if common_index is None:
            common_index = df.index
        else:
            common_index = common_index.intersection(df.index)

    if common_index is None or len(common_index) < 3:
        raise ValueError("Coinler arasi ortak index yetersiz.")

    return {coin: df.loc[common_index].copy() for coin, df in raw.items()}
