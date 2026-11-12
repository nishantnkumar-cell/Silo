"""Random-entry strategy: no information, used as a luck control."""

from __future__ import annotations

import numpy as np

from silo.firewall import PointInTimeView
from silo.strategies.base import Strategy


class RandomEntry(Strategy):
    """Rebalances on random days into a random subset of symbols.

    Uses no market data, only the clock. Randomness is drawn per date from
    ``(seed, date)``, so a run is reproducible and unaffected by data that
    comes later.
    """

    name = "random"

    def __init__(self, symbols, decide_prob: float = 0.05, exposure: float = 0.5, seed: int = 0):
        super().__init__(symbols)
        self.decide_prob, self.exposure, self.seed = decide_prob, exposure, seed

    def decide(self, view: PointInTimeView) -> dict[str, float] | None:
        day = int(view.clock.tz_convert(None).normalize().value // 86_400_000_000_000)
        rng = np.random.default_rng([self.seed, day])
        if rng.random() >= self.decide_prob:
            return None
        n = len(self.symbols)
        return {s: 1.0 / n for s in self.symbols if rng.random() < self.exposure}
