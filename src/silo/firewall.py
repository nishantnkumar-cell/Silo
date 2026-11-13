"""The time firewall: agents and strategies only see data that was public.

A :class:`PointInTimeView` wraps the vault's tables and a simulated clock.
Every read is filtered by ``available_at <= clock``. Asking directly for
anything newer raises :class:`FutureDataError`, so a leak fails loudly
instead of quietly inflating a backtest.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


class FutureDataError(LookupError):
    """Raised when code asks for data that was not yet public at the clock."""


class PreparedTables(Mapping):
    """Tables split once into data and an ``available_at`` array, for fast views."""

    def __init__(self, tables: Mapping[str, pd.DataFrame]) -> None:
        self._raw = dict(tables)
        self._data: dict[str, pd.DataFrame] = {}
        self._avail: dict[str, np.ndarray] = {}
        for name, df in self._raw.items():
            if "available_at" not in df.columns:
                raise ValueError(f"table {name!r} has no available_at column")
            avail = pd.DatetimeIndex(df["available_at"])
            if not avail.is_monotonic_increasing:
                raise ValueError(f"table {name!r} available_at is not sorted")
            self._avail[name] = avail.tz_convert("UTC").as_unit("ns").asi8
            self._data[name] = df.drop(columns="available_at")

    def __getitem__(self, name: str) -> pd.DataFrame:
        return self._raw[name]

    def __iter__(self):
        return iter(self._raw)

    def __len__(self) -> int:
        return len(self._raw)

    def cut(self, name: str, clock: pd.Timestamp) -> int:
        return int(np.searchsorted(self._avail[name], clock.as_unit("ns").value, side="right"))


class PointInTimeView:
    def __init__(self, tables: Mapping[str, pd.DataFrame], clock: pd.Timestamp) -> None:
        self._t = tables if isinstance(tables, PreparedTables) else PreparedTables(tables)
        self.clock = _utc(clock)

    @property
    def symbols(self) -> list[str]:
        return list(self._t)

    def bars(self, symbol: str, lookback: int | None = None) -> pd.DataFrame:
        """Rows of ``symbol`` public at the clock (a copy), newest last."""
        cut = self._t.cut(symbol, self.clock)
        lo = 0 if lookback is None else max(0, cut - lookback)
        return self._t._data[symbol].iloc[lo:cut].copy()

    def close(self, symbol: str, lookback: int | None = None) -> pd.Series:
        return self.bars(symbol, lookback)["close"]

    def latest(self, symbol: str) -> pd.Series | None:
        rows = self.bars(symbol, lookback=1)
        return None if rows.empty else rows.iloc[-1]

    def bar_on(self, symbol: str, date: str | pd.Timestamp) -> pd.Series:
        """One bar by date; refuses dates whose bar was not yet public."""
        df = self._t[symbol]
        ts = pd.Timestamp(date).normalize()
        if ts not in df.index:
            raise KeyError(f"{symbol} has no bar on {ts.date()}")
        row = df.loc[ts]
        if row["available_at"] > self.clock:
            raise FutureDataError(
                f"{symbol} bar for {ts.date()} is public at {row['available_at']}, clock is {self.clock}"
            )
        return row.drop("available_at")


def _utc(ts: pd.Timestamp | str) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
