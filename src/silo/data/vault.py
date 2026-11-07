"""Local cache of market data where every row records when it became public.

Each stored table has an ``available_at`` column (UTC). The time firewall
uses it to decide what an agent may see at a given simulated moment.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

BAR_COLUMNS = ["open", "high", "low", "close", "volume"]

# US equities close at 16:00 New York time, which is 21:00 UTC in winter and
# 20:00 UTC in summer. Using 21:00 UTC is conservative: a daily bar is never
# treated as public before the market has actually closed.
DAILY_CLOSE_UTC = pd.Timedelta(hours=21)


def daily_available_at(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """When a daily bar dated ``d`` becomes public: ``d`` at 21:00 UTC."""
    idx = pd.DatetimeIndex(dates)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return (idx.normalize() + DAILY_CLOSE_UTC).tz_localize("UTC")


def normalize_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case columns, index by naive date, add ``available_at``."""
    out = df.rename(columns=str.lower).copy()
    missing = [c for c in BAR_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"bars are missing columns: {missing}")
    out = out[BAR_COLUMNS].astype(float)
    idx = pd.DatetimeIndex(out.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    out.index = idx.normalize()
    out.index.name = "date"
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out["available_at"] = daily_available_at(out.index)
    return out


class DataVault:
    """Parquet-backed store of daily bars, one file per symbol."""

    def __init__(self, root: str | Path = "data") -> None:
        self.root = Path(root)
        (self.root / "bars").mkdir(parents=True, exist_ok=True)

    def _path(self, symbol: str) -> Path:
        safe = symbol.upper().replace("/", "_").replace("^", "_")
        return self.root / "bars" / f"{safe}.parquet"

    def put_bars(self, symbol: str, bars: pd.DataFrame) -> pd.DataFrame:
        """Merge ``bars`` into the cache and return the stored table."""
        new = normalize_bars(bars)
        path = self._path(symbol)
        if path.exists():
            old = pd.read_parquet(path)
            new = pd.concat([old, new])
            new = new[~new.index.duplicated(keep="last")].sort_index()
        new.to_parquet(path)
        return new

    def get_bars(self, symbol: str) -> pd.DataFrame:
        path = self._path(symbol)
        if not path.exists():
            raise KeyError(f"no cached bars for {symbol}; run `silo fetch {symbol}` first")
        return pd.read_parquet(path)

    def symbols(self) -> list[str]:
        return sorted(p.stem for p in (self.root / "bars").glob("*.parquet"))
