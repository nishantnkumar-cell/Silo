"""Silo's own agent: combine contestants by how well they have been doing.

The idea from the study: no single design wins in every market, so run
several side by side and lean on whichever is working, without ever
trusting one blindly.

Every rebalance, :class:`Blend`

1. asks each member for its target weights (a member that returns ``None``
   keeps its last ones);
2. scores each member on what its *previous* targets would have earned since
   the last rebalance (price change plus dividends, from the firewall view),
   kept as an exponentially weighted Sharpe-like ratio, so recent weeks
   count most;
3. weights members in proportion to their positive scores (equal weights
   while warming up);
4. applies a skeptic layer:
   * if no member has a positive score, it halves exposure: nothing is
     working, so hold more cash;
   * if fewer than 40% of the universe trade above their 200-day average
     (a falling market), it halves exposure again.

Members are scored on their hypothetical targets whether or not the blend
followed them, so a member that was ignored can earn its way back in. All
state comes from past views only, so the blend passes the truncation test.
"""

from __future__ import annotations

import numpy as np

from silo.strategies.base import Strategy


class Blend(Strategy):
    name = "blend"
    param_grid = {"halflife": [8, 26]}
    default_members = ("uk_trend", "momentum", "donchian")

    def __init__(self, symbols, members=None, halflife: float = 13, warmup: int = 8,
                 breadth_floor: float = 0.4, member_params: dict | None = None):
        super().__init__(symbols)
        from silo.strategies import REGISTRY  # late import: the registry includes this class

        names = list(members or self.default_members)
        params = member_params or {}
        self.members = {n: REGISTRY[n](symbols, **params.get(n, {})) for n in names}
        self.uses_llm = any(m.uses_llm for m in self.members.values())
        self.halflife, self.warmup, self.breadth_floor = halflife, warmup, breadth_floor
        self.reset()

    @property
    def model_cutoff(self):
        cuts = [getattr(m, "model_cutoff", None) for m in self.members.values() if m.uses_llm]
        cuts = [c for c in cuts if c]
        return max(cuts) if cuts else None

    def reset(self):
        for m in getattr(self, "members", {}).values():
            m.reset()
        self.last_targets = {n: {} for n in getattr(self, "members", {})}
        self.mean = {n: 0.0 for n in self.last_targets}
        self.var = {n: 0.0 for n in self.last_targets}
        self.n_obs = 0
        self.last_prices: dict[str, float] = {}
        self._last_date = None
        self.last_alpha: dict[str, float] = {}

    def _prices(self, view):
        out, divs = {}, {}
        for s in self.symbols:
            bars = view.bars(s, lookback=260)
            if len(bars):
                out[s] = float(bars["close"].iloc[-1])
                divs[s] = bars
        return out, divs

    def _member_return(self, targets, prices, bars) -> float:
        r = 0.0
        for s, w in targets.items():
            if s in prices and s in self.last_prices and self.last_prices[s] > 0:
                b = bars[s]
                paid = float(b.loc[b.index > self._last_date, "dividend"].sum()) if "dividend" in b else 0.0
                r += w * ((prices[s] + paid) / self.last_prices[s] - 1)
        return r

    def decide(self, view):
        prices, bars = self._prices(view)
        if not prices:
            return {}
        if self.last_prices:
            a = 1 - 0.5 ** (1 / self.halflife)
            for n, t in self.last_targets.items():
                r = self._member_return(t, prices, bars)
                d = r - self.mean[n]
                self.mean[n] += a * d
                self.var[n] = (1 - a) * (self.var[n] + a * d * d)
            self.n_obs += 1

        for n, m in self.members.items():
            w = m.decide(view)
            if w is not None:
                self.last_targets[n] = dict(w)
        self.last_prices = prices
        self._last_date = max(b.index[-1] for b in bars.values())

        if self.n_obs < self.warmup:
            alpha = {n: 1 / len(self.members) for n in self.members}
            exposure = 1.0
        else:
            score = {n: self.mean[n] / np.sqrt(self.var[n]) if self.var[n] > 0 else 0.0 for n in self.members}
            pos = {n: s for n, s in score.items() if s > 0}
            if pos:
                total = sum(pos.values())
                alpha, exposure = {n: s / total for n, s in pos.items()}, 1.0
            else:
                alpha, exposure = {n: 1 / len(self.members) for n in self.members}, 0.5
        breadth = self._breadth(bars)
        if breadth is not None and breadth < self.breadth_floor:
            exposure *= 0.5
        self.last_alpha = alpha

        out: dict[str, float] = {}
        for n, a in alpha.items():
            for s, w in self.last_targets[n].items():
                out[s] = out.get(s, 0.0) + exposure * a * w
        return {s: w for s, w in out.items() if w > 1e-6}

    @staticmethod
    def _breadth(bars) -> float | None:
        flags = [float(b["close"].iloc[-1] > b["close"].iloc[-200:].mean()) for b in bars.values() if len(b) >= 200]
        return float(np.mean(flags)) if flags else None


class SiloAgent(Blend):
    """The blend with the AI contestants on board: rules plus the two desks."""

    name = "silo_agent"
    uses_llm = True
    param_grid: dict[str, list] = {}
    default_members = ("uk_trend", "momentum", "desk", "panel")
