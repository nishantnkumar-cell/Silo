"""No-AI twins of the AI contestants, so the model's own contribution can be seen.

An AI desk only ever chooses from a shortlist that a rule built (the 30
shares with the strongest six-month return). So "the desk beat random picks
from the whole market" says little: the shortlist rule may have done the
work. These two controls see exactly the same shortlist and size positions
the same way, but use no model:

* :class:`ShortlistTop` - the top ``top_k`` of the shortlist by six-month
  return. If the AI can't beat this, it adds nothing to the rule.
* :class:`ShortlistRandom` - ``top_k`` names drawn at random from the
  shortlist, keeping most holdings from week to week as a desk does. If the
  AI can't beat this, its choices within the shortlist are noise.
"""

from __future__ import annotations

import numpy as np

from silo.agents.brief import build_brief
from silo.strategies.base import Strategy


def inverse_vol_weights(picks: list[str], rows: dict[str, list[float]], top_k: int) -> dict[str, float]:
    """The AI contestants' sizing: inverse volatility, partly in cash when few names are picked."""
    picks = list(dict.fromkeys(picks))[:top_k]
    if not picks:
        return {}
    inv = {s: 1 / max(rows[s][4], 1.0) for s in picks}
    total = sum(inv.values())
    exposure = min(1.0, len(picks) / top_k)
    return {s: exposure * v / total for s, v in inv.items()}


class ShortlistTop(Strategy):
    name = "shortlist_top"
    param_grid: dict[str, list] = {}

    def __init__(self, symbols, shortlist: int = 30, top_k: int = 12):
        super().__init__(symbols)
        self.shortlist, self.top_k = shortlist, top_k

    def decide(self, view):
        brief = build_brief(view, self.symbols, anonymize=False, limit=self.shortlist)
        if not brief.rows:
            return {}
        picks = sorted(brief.rows, key=lambda s: (-brief.rows[s][2], s))[: self.top_k]
        return inverse_vol_weights(picks, brief.rows, self.top_k)


class ShortlistRandom(Strategy):
    name = "shortlist_random"
    param_grid: dict[str, list] = {}

    def __init__(self, symbols, shortlist: int = 30, top_k: int = 12, keep: float = 0.8, seed: int = 0):
        super().__init__(symbols)
        self.shortlist, self.top_k, self.keep, self.seed = shortlist, top_k, keep, seed

    def reset(self) -> None:
        self.held: list[str] = []

    def decide(self, view):
        brief = build_brief(view, self.symbols, anonymize=False, limit=self.shortlist)
        if not brief.rows:
            return {}
        day = int(view.clock.tz_convert(None).value // 86_400_000_000_000)
        rng = np.random.default_rng([self.seed, day])
        names = sorted(brief.rows)
        stay = [s for s in getattr(self, "held", []) if s in brief.rows and rng.random() < self.keep]
        pool = [s for s in names if s not in stay]
        n_new = min(len(pool), self.top_k - len(stay))
        fresh = list(rng.choice(pool, size=n_new, replace=False)) if n_new > 0 else []
        self.held = stay + [str(s) for s in fresh]
        return inverse_vol_weights(self.held, brief.rows, self.top_k)
