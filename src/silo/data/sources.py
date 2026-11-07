"""Free data sources. Network sources are optional; synthetic data needs none."""

from __future__ import annotations

import io
import urllib.request

import numpy as np
import pandas as pd


def fetch_yfinance(symbol: str, start: str = "2010-01-01") -> pd.DataFrame:
    """Daily bars from Yahoo Finance via the optional ``yfinance`` package."""
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError("install the data extra: pip install 'silo[data]'") from exc
    df = yf.download(symbol, start=start, auto_adjust=True, progress=False, multi_level_index=False)
    if df.empty:
        raise RuntimeError(f"yfinance returned no data for {symbol}")
    return df


def fetch_stooq(symbol: str) -> pd.DataFrame:
    """Daily bars from Stooq's free CSV endpoint (US tickers get a .us suffix)."""
    code = symbol.lower() if "." in symbol else f"{symbol.lower()}.us"
    url = f"https://stooq.com/q/d/l/?s={code}&i=d"
    with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310 - fixed https host
        text = resp.read().decode()
    df = pd.read_csv(io.StringIO(text), parse_dates=["Date"], index_col="Date")
    if df.empty:
        raise RuntimeError(f"stooq returned no data for {symbol}")
    return df


def synthetic_bars(
    n_days: int = 1500,
    seed: int = 0,
    start: str = "2015-01-02",
    drift: float = 0.0003,
    vol: float = 0.012,
    start_price: float = 100.0,
) -> pd.DataFrame:
    """Random-walk daily bars for offline tests. Contains no real signal."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n_days)
    rets = rng.normal(drift, vol, n_days)
    close = start_price * np.exp(np.cumsum(rets))
    gap = rng.normal(0, vol / 3, n_days)
    open_ = np.r_[start_price, close[:-1]] * np.exp(gap)
    hi = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, vol / 2, n_days)))
    lo = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, vol / 2, n_days)))
    volume = rng.integers(1_000_000, 5_000_000, n_days).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": hi, "low": lo, "close": close, "volume": volume}, index=dates
    )


SOURCES = {"yfinance": fetch_yfinance, "stooq": fetch_stooq}
