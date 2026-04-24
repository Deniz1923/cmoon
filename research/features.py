"""
Shared indicator library — import from here, never reimplement elsewhere.

All functions take a pd.Series or pd.DataFrame and return a pd.Series.
Every function must handle the case where len(series) < window gracefully
(return NaN, not raise).
"""
import numpy as np
import pandas as pd


COINS = ["kapcoin-usd_train", "metucoin-usd_train", "tamcoin-usd_train"]


# ---------------------------------------------------------------------------
# Trend indicators
# ---------------------------------------------------------------------------

def ema(series: pd.Series, n: int) -> pd.Series:
    # TODO: implement exponential moving average
    # hint: series.ewm(span=n, adjust=False).mean()
    raise NotImplementedError


def sma(series: pd.Series, n: int) -> pd.Series:
    # TODO: implement simple moving average
    raise NotImplementedError


def ema_cross_signal(series: pd.Series, fast: int = 20, slow: int = 50) -> pd.Series:
    """Returns +1 when fast EMA crosses above slow, -1 when crosses below, 0 otherwise."""
    # TODO: compute fast/slow EMAs, detect crossovers
    # hint: np.sign(fast_ema - slow_ema) != np.sign(fast_ema.shift(1) - slow_ema.shift(1))
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Volatility indicators
# ---------------------------------------------------------------------------

def true_range(df: pd.DataFrame) -> pd.Series:
    """Single-candle True Range: max(H-L, |H-prev_C|, |L-prev_C|)."""
    # TODO: implement TR
    # hint: three components, take element-wise max
    raise NotImplementedError


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Average True Range over n periods."""
    # TODO: use true_range() + rolling mean (or EMA)
    raise NotImplementedError


def atr_pct(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """ATR as a fraction of Close — normalized volatility, comparable across coins."""
    # TODO: atr(df, n) / df["Close"]
    raise NotImplementedError


def bb_bands(series: pd.Series, n: int = 20, k: float = 2.0):
    """Returns (upper, mid, lower) Bollinger Bands."""
    # TODO: mid = sma(n), upper = mid + k*std, lower = mid - k*std
    raise NotImplementedError


def bb_width(series: pd.Series, n: int = 20, k: float = 2.0) -> pd.Series:
    """(upper - lower) / mid — measures band squeeze. High = trending, low = ranging."""
    # TODO: use bb_bands()
    raise NotImplementedError


def bb_pct(series: pd.Series, n: int = 20, k: float = 2.0) -> pd.Series:
    """(Close - lower) / (upper - lower) — position within bands [0, 1]."""
    # TODO: use bb_bands()
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Momentum / oscillators
# ---------------------------------------------------------------------------

def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    # TODO: classic Wilder RSI
    # hint: compute gains/losses, use ewm(alpha=1/n) for rolling average
    raise NotImplementedError


def momentum(series: pd.Series, n: int = 10) -> pd.Series:
    """Rate of change: (close / close[n] - 1)."""
    # TODO: series.pct_change(n)
    raise NotImplementedError


def volume_ratio(df: pd.DataFrame, n: int = 20) -> pd.Series:
    """Current volume vs n-period average — spike = breakout confirmation."""
    # TODO: df["Volume"] / df["Volume"].rolling(n).mean()
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Cross-coin features
# ---------------------------------------------------------------------------

def rolling_correlation(a: pd.Series, b: pd.Series, n: int = 30) -> pd.Series:
    """Rolling Pearson correlation between two return series."""
    # TODO: a.pct_change().rolling(n).corr(b.pct_change())
    raise NotImplementedError


def lead_lag_signal(leader: pd.Series, follower: pd.Series, lag: int = 1) -> pd.Series:
    """
    Returns the leader's return lagged by `lag` candles.
    Use as a feature: if kap went up yesterday, does met go up today?
    """
    # TODO: leader.pct_change().shift(lag)
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Regime detection
# ---------------------------------------------------------------------------

def is_trending(df: pd.DataFrame, bb_threshold: float = 0.08) -> pd.Series:
    """
    Boolean series: True when market is in a trending regime.
    Uses BB width — wide bands = trending, narrow = ranging.

    TODO: tune bb_threshold against EDA results (see research/eda.ipynb)
    """
    # TODO: bb_width(df["Close"]) > bb_threshold
    raise NotImplementedError
