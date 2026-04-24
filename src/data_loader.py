"""
data_loader.py — Dev A
Load raw market data and enforce chronological splits.
Only this module may define or enforce train/val/holdout boundaries.
"""
import os
from pathlib import Path

import pandas as pd
import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_ohlcv(symbol: str, market_type: str, config: dict) -> pd.DataFrame:
    """Load OHLCV data from the configured source (csv or ccxt)."""
    source = config.get("data_source", {})
    provider = source.get("provider", "csv")

    if provider == "csv":
        return _load_from_csv(symbol, market_type, config)
    elif provider in ("ccxt", "binance"):
        return _load_from_ccxt(symbol, market_type, config)
    else:
        raise ValueError(f"Unknown provider '{provider}'. Choose 'csv' or 'ccxt'.")


def get_splits(df: pd.DataFrame, config: dict):
    """
    Carve train, validation, and locked holdout from df.

    Order of operations:
      1. Holdout is carved from the END first — never touched during training.
      2. Remaining data is split into train and val with an embargo gap.

    Returns
    -------
    (train_df, val_df, holdout_df)
    """
    splits_cfg = config["splits"]
    embargo_days = splits_cfg.get("embargo_days", 5)
    holdout_months = splits_cfg.get("locked_holdout_months", 12)
    val_years = splits_cfg.get("validation_years", 1)

    # Step 1 — carve holdout from the end
    holdout_start = df.index[-1] - pd.DateOffset(months=holdout_months)
    holdout_df = df.loc[df.index > holdout_start].copy()
    remaining = df.loc[df.index <= holdout_start].copy()

    # Step 2 — split remaining into train / val with embargo
    val_days = int(val_years * 365)
    val_start = remaining.index[-1] - pd.Timedelta(days=val_days)
    embargo_end = val_start + pd.Timedelta(days=embargo_days)

    train_df = remaining.loc[remaining.index < val_start].copy()
    val_df = remaining.loc[remaining.index >= embargo_end].copy()

    assert len(train_df) > 0, "Train split is empty — check split config."
    assert len(val_df) > 0, "Validation split is empty — check split config."
    assert len(holdout_df) > 0, "Holdout split is empty — check split config."
    assert train_df.index[-1] < val_df.index[0], "Train/val overlap after embargo."

    return train_df, val_df, holdout_df


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _load_from_csv(symbol: str, market_type: str, config: dict) -> pd.DataFrame:
    raw_dir = Path(config["paths"]["raw_data"])
    safe_symbol = symbol.replace("/", "_")
    candidates = [
        raw_dir / f"{safe_symbol}_{market_type}.csv",
        raw_dir / f"{safe_symbol}.csv",
        raw_dir / f"{symbol}.csv",
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            return _validate_and_normalize(df, symbol)
    raise FileNotFoundError(
        f"No CSV found for {symbol} in {raw_dir}. Tried: {[str(c) for c in candidates]}"
    )


def _load_from_ccxt(symbol: str, market_type: str, config: dict) -> pd.DataFrame:
    try:
        import ccxt
    except ImportError:
        raise ImportError("ccxt not installed. Run: uv add ccxt")

    source = config["data_source"]
    exchange_id = source.get("exchange", "binance")
    exchange_class = getattr(ccxt, exchange_id)

    api_key = os.environ.get(source.get("api_key_env", ""), "")
    secret = os.environ.get(source.get("secret_env", ""), "")

    exchange = exchange_class({"apiKey": api_key, "secret": secret, "enableRateLimit": True})
    timeframe = config["market"]["bar_interval"]
    limit = 1000

    all_bars: list = []
    since = None
    while True:
        bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        if not bars:
            break
        all_bars.extend(bars)
        if len(bars) < limit:
            break
        since = bars[-1][0] + 1

    df = pd.DataFrame(all_bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp")
    return _validate_and_normalize(df, symbol)


def _validate_and_normalize(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} for {symbol}")

    # Ensure UTC timezone-aware index
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    df = df.sort_index()
    df = df.dropna(subset=required)
    df = df[~df.index.duplicated(keep="first")]

    assert df.index.is_monotonic_increasing, f"Index not monotonic for {symbol}"
    assert not df[required].isna().any().any(), f"NaN values in OHLCV for {symbol}"

    extras = [c for c in df.columns if c not in required]
    return df[required + extras]