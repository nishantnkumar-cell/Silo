"""The time firewall: agents and strategies only see data that was public.

A :class:`PointInTimeView` wraps the vault's tables and a simulated clock.
Every read is filtered by ``available_at <= clock``. Asking directly for
anything newer raises :class:`FutureDataError`, so a leak fails loudly
instead of quietly inflating a backtest.
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd


class FutureDataError(LookupError):
    """Raised when code asks for data that was not yet public at the clock."""


class PointInTimeView:
    def __init__(self, tables: Mapping[str, pd.DataFrame], clock: pd.Timestamp) -> None:
        for name, df in tables.items():
            if "available_at" not in df.columns:
                raise ValueError(f"table {name!r} has no available_at column")
        self._tables = tables
        self.clock = _utc(clock)

    @property
    def symbols(self) -> list[str]:
        return list(self._tables)

    def bars(self, symbol: str, lookback: int | None = None) -> pd.DataFrame:
        """Rows of ``symbol`` public at the clock (a copy), newest last."""
        df = self._tables[symbol]
        cut = int(df["available_at"].searchsorted(self.clock, side="right"))
        out = df.iloc[:cut]
        if lookback is not None:
            out = out.iloc[-lookback:]
        return out.drop(columns="available_at").copy()

    def close(self, symbol: str, lookback: int | None = None) -> pd.Series:
        return self.bars(symbol, lookback)["close"]

    def latest(self, symbol: str) -> pd.Series | None:
        rows = self.bars(symbol, lookback=1)
        return None if rows.empty else rows.iloc[-1]

    def bar_on(self, symbol: str, date: str | pd.Timestamp) -> pd.Series:
        """One bar by date; refuses dates whose bar was not yet public."""
        df = self._tables[symbol]
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
