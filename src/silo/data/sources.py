"""Free data sources. Network sources are optional; synthetic data needs none.

Prices are kept split-adjusted but NOT dividend-adjusted. Dividend
adjustment rewrites past prices using dividends paid later, which leaks the
future into a backtest; instead dividends are stored in their own column and
credited as cash on the ex-date by the simulated broker.
"""

from __future__ import annotations

import io
import urllib.request

import numpy as np
import pandas as pd

from silo.data.vault import normalize_bars
from silo.markets import market_for


def drop_unfinished(df: pd.DataFrame, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Remove bars that are not final yet (today's bar during trading hours).

    Yahoo returns a partial bar for a session still trading: a US ETF during
    New York hours, or today's UTC day for crypto. Such a row would later be
    replaced, so it is never stored.
    """
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    return df[df["available_at"] <= now]


def crypto_volume_to_units(df: pd.DataFrame) -> pd.DataFrame:
    """Yahoo reports crypto volume in dollars traded; Silo stores units (coins),
    like shares, so close x volume is value traded everywhere."""
    out = df.copy()
    out["volume"] = out["volume"] / out["close"]
    return out


def fetch_yfinance(symbol: str, start: str = "2010-01-01") -> pd.DataFrame:
    """Daily bars from Yahoo Finance, normalized (pence converted for ``.L``)."""
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError("install the data extra: pip install 'silo[data]'") from exc
    df = yf.download(
        symbol, start=start, auto_adjust=False, actions=True, progress=False, multi_level_index=False
    )
    if df.empty:
        raise RuntimeError(f"yfinance returned no data for {symbol}")
    market = market_for(symbol)
    pence = market.code == "UK"
    try:  # Yahoo reports GBp for most London listings; trust it when available
        currency = yf.Ticker(symbol).fast_info.get("currency")
        if currency:
            pence = currency == "GBp"
    except Exception:  # noqa: BLE001 - metadata is best-effort
        pass
    out = normalize_bars(df.drop(columns=["Adj Close"], errors="ignore"), market, pence=pence)
    if market.code == "CRYPTO":
        out = crypto_volume_to_units(out)
    return drop_unfinished(out)


def fetch_stooq(symbol: str) -> pd.DataFrame:
    """Daily bars from Stooq's free CSV endpoint (.us / .uk suffixes)."""
    market = market_for(symbol)
    if market.code == "CRYPTO":
        raise RuntimeError("Stooq has no crypto daily bars; use --source yfinance")
    base = symbol.lower()[:-2] if market.code == "UK" else symbol.lower()
    code = base if "." in base else f"{base}.{'uk' if market.code == 'UK' else 'us'}"
    url = f"https://stooq.com/q/d/l/?s={code}&i=d"
    with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310 - fixed https host
        text = resp.read().decode()
    df = pd.read_csv(io.StringIO(text), parse_dates=["Date"], index_col="Date")
    if df.empty:
        raise RuntimeError(f"stooq returned no data for {symbol}")
    return drop_unfinished(normalize_bars(df, market, pence=market.code == "UK"))


def synthetic_bars(
    n_days: int = 1500,
    seed: int = 0,
    start: str = "2015-01-02",
    drift: float = 0.0003,
    vol: float = 0.012,
    start_price: float = 100.0,
    every_day: bool = False,
) -> pd.DataFrame:
    """Random-walk daily bars for offline tests. Contains no real signal.

    ``every_day`` gives a seven-day calendar, like crypto."""
    rng = np.random.default_rng(seed)
    dates = (pd.date_range if every_day else pd.bdate_range)(start=start, periods=n_days)
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
