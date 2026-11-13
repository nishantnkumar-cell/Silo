"""Market conventions: closing time, currency and default trading costs.

Silo's UK focus means London Stock Exchange rules are first-class:
* the closing auction ends about 16:35 London time (DST handled by tz);
* Yahoo quotes ``.L`` tickers in pence (GBp), which Silo converts to pounds;
* buys of UK shares pay 0.5% stamp duty (SDRT); sells do not.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Market:
    code: str
    tz: str
    close_time: str  # local time after which the daily bar counts as public
    currency: str


US = Market("US", "America/New_York", "16:05", "USD")
UK = Market("UK", "Europe/London", "16:35", "GBP")
MARKETS = {"US": US, "UK": UK}


def market_for(symbol: str) -> Market:
    """Infer the market from a Yahoo-style ticker suffix."""
    return UK if symbol.upper().endswith(".L") else US


def market_for_symbols(symbols: list[str]) -> Market:
    found = {market_for(s).code for s in symbols}
    if len(found) > 1:
        raise ValueError(f"one book cannot mix markets {sorted(found)}: costs and currencies differ")
    return MARKETS[found.pop()]


def close_timestamps(dates: pd.DatetimeIndex, market: Market) -> pd.DatetimeIndex:
    """UTC moment each daily bar becomes public: the local close on its date."""
    idx = pd.DatetimeIndex(dates)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    local = idx.normalize() + pd.Timedelta(market.close_time + ":00")
    return local.tz_localize(market.tz, ambiguous="NaT", nonexistent="shift_forward").tz_convert("UTC")
