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


def discover_universe(config: dict) -> list:
    """
    Return the list of valid trading symbols.

    universe_mode=explicit → returns config["market"]["symbols"].
    universe_mode=auto     → scans raw_data dir, filters by quality rules.
    """
    market_cfg = config["market"]
    universe_mode = market_cfg.get("universe_mode", "auto")

    if universe_mode == "explicit":
        syms = list(market_cfg.get("symbols", []))
        if not syms:
            raise ValueError("universe_mode=explicit but market.symbols is empty in config.")
        return syms

    raw_dir = Path(config["paths"]["raw_data"])
    file_pattern = market_cfg.get("raw_file_pattern", "*.parquet")
    symbol_source = market_cfg.get("symbol_source", "filename")
    market_type = market_cfg.get("market_type", "spot")
    min_bars = market_cfg.get("min_history_bars", 1000)
    max_missing_frac = market_cfg.get("max_missing_bar_fraction", 0.05)

    _interval_to_freq = {
        "1m": "min", "5m": "5min", "15m": "15min", "30m": "30min",
        "1h": "h", "4h": "4h", "1d": "D",
    }
    bar_interval = market_cfg.get("bar_interval", "1h")
    freq = _interval_to_freq.get(bar_interval, "h")

    valid: list = []
    for fpath in sorted(raw_dir.glob(file_pattern)):
        if symbol_source != "filename":
            continue
        symbol = _symbol_from_path(fpath, market_type)
        try:
            df = load_ohlcv(symbol, market_type, config)
        except Exception as exc:
            print(f"[discover] Skip {fpath.name}: {exc}")
            continue

        if len(df) < min_bars:
            print(f"[discover] Skip {symbol}: {len(df)} bars < min_history_bars={min_bars}")
            continue

        expected = len(pd.date_range(df.index[0], df.index[-1], freq=freq))
        missing_frac = max(0.0, 1.0 - len(df) / expected)
        if missing_frac > max_missing_frac:
            print(f"[discover] Skip {symbol}: missing={missing_frac:.1%} > {max_missing_frac:.1%}")
            continue

        valid.append(symbol)
        print(f"[discover] Accept {symbol}: {len(df)} bars, missing={missing_frac:.1%}")

    if not valid:
        print(f"[discover] No valid symbols in {raw_dir} matching '{file_pattern}'")
    return valid


def load_ohlcv(symbol: str, market_type: str, config: dict) -> pd.DataFrame:
    """Load OHLCV data from the configured source (file or ccxt)."""
    source = config.get("data_source", {})
    provider = source.get("provider", "csv")

    if provider == "csv":
        return _load_from_file(symbol, market_type, config)
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

def _symbol_from_path(fpath: Path, market_type: str) -> str:
    """Infer symbol name from a data file path."""
    stem = fpath.stem
    suffix = f"_{market_type}"
    return stem[: -len(suffix)] if stem.endswith(suffix) else stem


def _load_from_file(symbol: str, market_type: str, config: dict) -> pd.DataFrame:
    """Load from local file — supports both CSV and Parquet."""
    raw_dir = Path(config["paths"]["raw_data"])
    safe_symbol = symbol.replace("/", "_")
    ts_col = config.get("market", {}).get("timestamp_column", "timestamp")

    candidates = [
        raw_dir / f"{safe_symbol}_{market_type}.parquet",
        raw_dir / f"{safe_symbol}.parquet",
        raw_dir / f"{safe_symbol}_{market_type}.csv",
        raw_dir / f"{safe_symbol}.csv",
        raw_dir / f"{symbol}.csv",
    ]
    for path in candidates:
        if path.exists():
            if path.suffix == ".parquet":
                df = pd.read_parquet(path)
                if ts_col in df.columns:
                    df = df.set_index(ts_col)
            else:
                df = pd.read_csv(path, index_col=0, parse_dates=True)
            return _validate_and_normalize(df, symbol)

    raise FileNotFoundError(
        f"No data file found for {symbol} in {raw_dir}. "
        f"Tried: {[str(c) for c in candidates]}"
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
