"""Local cache of market data where every row records when it became public.

Each stored table has an ``available_at`` column (UTC). The time firewall
uses it to decide what an agent may see at a given simulated moment.

Every fetch is kept as its own dated snapshot and never overwritten, so a
later vendor revision (a re-adjusted price, a corrected row) can always be
compared with what was known before. Reads merge snapshots, newest wins.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from silo.markets import US, Market, close_timestamps, market_for

BAR_COLUMNS = ["open", "high", "low", "close", "volume"]
PRICE_COLUMNS = ["open", "high", "low", "close"]
OPTIONAL_COLUMNS = ["dividend"]

# Kept for callers of the Phase 0 API: US close as a fixed, conservative 21:00 UTC.
DAILY_CLOSE_UTC = pd.Timedelta(hours=21)


def daily_available_at(dates: pd.DatetimeIndex, market: Market | None = None) -> pd.DatetimeIndex:
    """When a daily bar dated ``d`` becomes public.

    With no market, 21:00 UTC (a conservative US close). With a market, its
    local closing time converted to UTC, so London is 16:35 in winter and
    15:35 UTC in summer.
    """
    if market is not None:
        return close_timestamps(dates, market)
    idx = pd.DatetimeIndex(dates)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return (idx.normalize() + DAILY_CLOSE_UTC).tz_localize("UTC")


def fix_pence_glitches(df: pd.DataFrame, window: int = 5) -> tuple[pd.DataFrame, int]:
    """Repair isolated rows that are 100x off (pounds mixed into a pence series).

    Yahoo sometimes reports a London row in GBP among GBp rows, or the
    reverse. A row whose close is ~1/100 or ~100x its neighbours' median is
    rescaled. Returns the repaired frame and how many rows were fixed.
    """
    out = df.copy()
    med = out["close"].rolling(window, center=True, min_periods=2).median()
    ratio = out["close"] / med
    low = ratio.between(0.005, 0.02)
    high = ratio.between(50, 200)
    out.loc[low, PRICE_COLUMNS] *= 100
    out.loc[high, PRICE_COLUMNS] /= 100
    return out, int(low.sum() + high.sum())


def normalize_bars(df: pd.DataFrame, market: Market | None = None, pence: bool = False) -> pd.DataFrame:
    """Lower-case columns, index by naive date, convert pence, add ``available_at``."""
    out = df.rename(columns=lambda c: str(c).lower().replace(" ", "_")).copy()
    out = out.rename(columns={"dividends": "dividend"})
    missing = [c for c in BAR_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"bars are missing columns: {missing}")
    cols = BAR_COLUMNS + [c for c in OPTIONAL_COLUMNS if c in out.columns]
    out = out[cols].astype(float)
    idx = pd.DatetimeIndex(out.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    out.index = idx.normalize()
    out.index.name = "date"
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out = out[out["close"].notna() & (out["close"] > 0)]
    if pence:
        out, _ = fix_pence_glitches(out)
        out[PRICE_COLUMNS] = out[PRICE_COLUMNS] / 100.0
        if "dividend" in out.columns:
            out["dividend"] = out["dividend"] / 100.0
    if "dividend" in out.columns:
        out["dividend"] = out["dividend"].fillna(0.0)
    out["available_at"] = daily_available_at(out.index, market)
    out = out[out["available_at"].notna()]
    return out


class DataVault:
    """Parquet-backed store of daily bars: dated snapshots, one folder per symbol."""

    def __init__(self, root: str | Path = "data") -> None:
        self.root = Path(root)
        (self.root / "bars").mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _safe(symbol: str) -> str:
        return symbol.upper().replace("/", "_").replace("^", "_")

    def _dir(self, symbol: str) -> Path:
        return self.root / "bars" / self._safe(symbol)

    def snapshots(self, symbol: str) -> list[Path]:
        files = sorted(self._dir(symbol).glob("*.parquet")) if self._dir(symbol).exists() else []
        legacy = self.root / "bars" / f"{self._safe(symbol)}.parquet"
        return ([legacy] if legacy.exists() else []) + files

    def put_bars(self, symbol: str, bars: pd.DataFrame, fetched_at: datetime | None = None) -> pd.DataFrame:
        """Store ``bars`` (already normalized, or raw OHLCV) as a new snapshot."""
        new = bars if "available_at" in bars.columns else normalize_bars(bars, market_for(symbol), pence=False)
        stamp = (fetched_at or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%S%fZ")
        d = self._dir(symbol)
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{stamp}.parquet"
        if path.exists():
            raise FileExistsError(f"snapshot {path} already exists; snapshots are never overwritten")
        new.to_parquet(path)
        return self.get_bars(symbol)

    def get_bars(self, symbol: str) -> pd.DataFrame:
        snaps = self.snapshots(symbol)
        if not snaps:
            raise KeyError(f"no cached bars for {symbol}; run `silo fetch {symbol}` first")
        merged = pd.concat([pd.read_parquet(p) for p in snaps])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        if "dividend" in merged.columns:
            merged["dividend"] = merged["dividend"].fillna(0.0)
        return merged

    def revisions(self, symbol: str, column: str = "close", tol: float = 1e-6) -> pd.DataFrame:
        """Dates whose value changed between snapshots (vendor revisions)."""
        frames = [pd.read_parquet(p)[column].rename(p.stem) for p in self.snapshots(symbol)]
        if len(frames) < 2:
            return pd.DataFrame()
        wide = pd.concat(frames, axis=1)
        spread = wide.max(axis=1) - wide.min(axis=1)
        return wide[spread > tol * wide.abs().max(axis=1).replace(0, np.nan)]

    # -- context tables: macro series, news events (names start with "ctx:") --

    def _ctx_dir(self, name: str) -> Path:
        return self.root / "context" / name.removeprefix("ctx:").replace(":", "__").replace("/", "_")

    def put_context(self, name: str, df: pd.DataFrame, fetched_at: datetime | None = None) -> pd.DataFrame:
        """Store a context table as a new dated snapshot (never overwritten).

        ``df`` needs an ``available_at`` UTC column. Rows are keyed by the
        index; on reads the newest snapshot wins for a repeated key.
        """
        if not name.startswith("ctx:"):
            raise ValueError(f"context table names start with 'ctx:' (got {name!r})")
        if "available_at" not in df.columns:
            raise ValueError(f"context table {name!r} needs an available_at column")
        stamp = (fetched_at or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%S%fZ")
        d = self._ctx_dir(name)
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{stamp}.parquet"
        if path.exists():
            raise FileExistsError(f"snapshot {path} already exists; snapshots are never overwritten")
        df.to_parquet(path)
        (d / "name.txt").write_text(name)
        return self.get_context(name)

    def get_context(self, name: str) -> pd.DataFrame:
        d = self._ctx_dir(name)
        snaps = sorted(d.glob("*.parquet")) if d.exists() else []
        if not snaps:
            raise KeyError(f"no cached context table {name}")
        merged = pd.concat([pd.read_parquet(p) for p in snaps])
        merged = merged[~merged.index.duplicated(keep="last")]
        avail = pd.to_datetime(merged["available_at"], utc=True)
        merged = merged.assign(available_at=avail)
        return merged.iloc[np.argsort(avail.to_numpy(), kind="stable")]

    def context_names(self) -> list[str]:
        root = self.root / "context"
        if not root.exists():
            return []
        return sorted((p / "name.txt").read_text() for p in root.iterdir() if (p / "name.txt").exists())

    def tables(self, symbols: list[str], context: bool = True) -> dict[str, pd.DataFrame]:
        """Bars for ``symbols`` plus every cached context table (macro, news)."""
        out = {s: self.get_bars(s) for s in symbols}
        if context:
            out.update({n: self.get_context(n) for n in self.context_names()})
        return out

    def symbols(self) -> list[str]:
        bars = self.root / "bars"
        names = {p.name for p in bars.iterdir() if p.is_dir()} | {p.stem for p in bars.glob("*.parquet")}
        return sorted(names)
