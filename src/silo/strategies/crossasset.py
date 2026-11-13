"""Plain-rule versions of the cross-market signals, so the lab can test them without a model.

* :class:`SectorMomentum` - hold the strongest names in the strongest
  sectors (industry momentum, Moskowitz and Grinblatt 1999).
* :class:`MacroTilt` - the 12-1 momentum book, but only as invested as the
  cross-market risk signals allow (time-series momentum across assets,
  Moskowitz, Ooi and Pedersen 2012). With no macro data cached it falls back
  to the UK market's own trend.

If these can't beat the random control, the same numbers are unlikely to
help an AI contestant either.
"""

from __future__ import annotations

import numpy as np

from silo.agents import crossasset
from silo.strategies.base import Strategy
from silo.strategies.classic import Momentum12_1
from silo.universe.sectors import sector_of


class _Monthly(Strategy):
    def reset(self) -> None:
        self.month = None

    def new_month(self, view) -> bool:
        month = view.clock.tz_convert(None).to_period("M")
        if month == self.month:
            return False
        self.month = month
        return True


class SectorMomentum(_Monthly):
    """Monthly: the best names (6-month return, skipping a month) in the best few sectors."""

    name = "sector_mom"
    param_grid = {"n_sectors": [2, 3], "per_sector": [2, 3]}

    def __init__(self, symbols, n_sectors: int = 3, per_sector: int = 3, lookback: int = 126, skip: int = 21,
                 sectors: dict[str, str] | None = None):
        super().__init__(symbols)
        self.n_sectors, self.per_sector, self.lookback, self.skip = n_sectors, per_sector, lookback, skip
        self.sectors = sectors

    def decide(self, view):
        rets = {}
        for s in self.symbols:
            c = view.close(s, lookback=self.lookback + 1)
            if len(c) > self.lookback:
                rets[s] = float(c.iloc[-1 - self.skip] / c.iloc[0] - 1)
        # trade once enough names have history; a late listing must not hold the book in cash
        if len(rets) < min(len(self.symbols), self.n_sectors * self.per_sector) or not self.new_month(view):
            return None
        groups: dict[str, list[str]] = {}
        for s in rets:
            groups.setdefault(sector_of(s, self.sectors), []).append(s)
        strength = {g: float(np.mean([rets[s] for s in names])) for g, names in groups.items()}
        best = [g for g in sorted(strength, key=lambda g: -strength[g])[: self.n_sectors] if strength[g] > 0]
        picks = []
        for g in best:
            picks += [s for s in sorted(groups[g], key=lambda s: -rets[s])[: self.per_sector] if rets[s] > 0]
        slots = self.n_sectors * self.per_sector
        return {s: 1.0 / slots for s in picks}


class MacroTilt(Momentum12_1):
    """12-1 momentum, scaled by the share of cross-market risk signals that are "on"."""

    name = "macro_tilt"
    param_grid = {"lookback": [126, 252], "floor": [0.0, 0.5]}

    def __init__(self, symbols, lookback: int = 252, skip: int = 21, top_k: int = 3, floor: float = 0.0):
        super().__init__(symbols, lookback, skip, top_k)
        self.floor = floor

    def decide(self, view):
        weights = super().decide(view)
        if not weights:
            return weights
        on, total = crossasset.risk_on(view, self.symbols)
        scale = max(self.floor, on / total) if total else 1.0
        return {s: w * scale for s, w in weights.items()}
