"""Daily trade management for any book: stop-loss, sell half at 1:2, trail the rest.

The project's risk rules say every position has a stop that is acted on the
day it is hit, half is sold once the trade has made twice its risk, and the
rest follows a trailing stop. The Core Strategy books have this built in.
:class:`Managed` wraps any other contestant with the same rules:

* **Stop.** When a name enters the book, its stop is ``stop_atr`` average
  daily ranges below the entry close. That distance is the trade's risk
  ("1R"). A daily close at or below the stop sells the whole position at the
  next open.
* **Half off at 1:2.** The first time the day's high reaches entry plus
  ``partial_at`` x risk, ``partial_frac`` of the position is sold and the
  stop moves up to the entry price.
* **Trail.** After that the stop follows ``trail_atr`` average daily ranges
  below the highest close since entry, and only ever moves up.
* A stopped-out name is not bought back for ``cooloff`` trading days, or
  until the base strategy has dropped it.

A momentum book has no price target, so the "at least 1:4 reward to risk"
entry test can't be applied to it: there is no reward to measure. The exits
above are the part of the rules that carries over. Whether they help a
momentum book is an open question the lab answers: compare
``momentum_managed`` with ``momentum``.

Checks use daily closes and fill at the next open, like every Silo order.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from silo.strategies.base import Strategy


@dataclass
class Position:
    entry: float
    stop: float
    risk: float
    high: float
    opened: object
    checked: object
    partial_done: bool = False


def avg_range(bars, n: int) -> float:
    h, lo, c = (bars[k].to_numpy(dtype=float) for k in ("high", "low", "close"))
    if len(c) < 2:
        return float("nan")
    prev = np.r_[c[0], c[:-1]]
    tr = np.maximum(h, prev) - np.minimum(lo, prev)
    return float(np.mean(tr[-n:]))


class Managed(Strategy):
    name = "managed"
    base_name = "momentum"
    base_defaults: dict = {}
    param_grid = {"stop_atr": [2.0, 3.0], "trail_atr": [3.0, 4.0]}

    def __init__(self, symbols, base: str | None = None, base_params: dict | None = None, stop_atr: float = 3.0,
                 partial_at: float = 2.0, partial_frac: float = 0.5, trail_atr: float = 3.0, atr_days: int = 20,
                 cooloff: int = 20):
        super().__init__(symbols)
        from silo.strategies import REGISTRY

        base = base or self.base_name
        if REGISTRY[base] is type(self) or issubclass(REGISTRY[base], Managed):
            raise ValueError("a managed book needs an unmanaged base strategy")
        self.base = REGISTRY[base](symbols, **{**self.base_defaults, **(base_params or {})})
        self.uses_llm = self.base.uses_llm
        self.members = {base: self.base, **getattr(self.base, "members", {})}  # so a model gets attached
        self.stop_atr, self.partial_at, self.partial_frac = stop_atr, partial_at, partial_frac
        self.trail_atr, self.atr_days, self.cooloff = trail_atr, atr_days, cooloff
        self.reset()

    @property
    def model_cutoff(self):
        return getattr(self.base, "model_cutoff", None)

    def reset(self) -> None:
        if hasattr(self, "base"):
            self.base.reset()
        self.book: dict[str, Position] = {}
        self.stopped: dict[str, int] = {}  # symbol -> decisions left to sit out
        self.log: list[dict] = []
        self.last_notes = None

    def manage(self, view):
        out: dict[str, float] = {}
        for s, p in list(self.book.items()):
            bars = view.bars(s, lookback=self.atr_days + 1)
            if not len(bars):
                continue
            recent = bars[bars.index > p.checked]
            if not len(recent):
                continue
            p.checked = bars.index[-1]
            price = float(bars["close"].iloc[-1])
            if float(recent["close"].min()) <= p.stop:
                self.log.append({"date": str(p.checked), "symbol": s, "action": "exit",
                                 "why": "trailing stop" if p.partial_done else "stop", "price": price})
                del self.book[s]
                self.stopped[s] = self.cooloff
                out[s] = 0.0
                continue
            p.high = max(p.high, float(recent["close"].max()))
            if not p.partial_done and p.risk > 0 and float(recent["high"].max()) >= p.entry + self.partial_at * p.risk:
                p.partial_done = True
                p.stop = max(p.stop, p.entry)
                self.log.append({"date": str(p.checked), "symbol": s, "action": "partial", "price": price})
                out[s] = 1.0 - self.partial_frac
            if p.partial_done:
                rng = avg_range(bars, self.atr_days)
                if np.isfinite(rng):
                    p.stop = max(p.stop, p.high - self.trail_atr * rng)
        for s in list(self.stopped):  # one trading day of the cool-off has passed
            self.stopped[s] -= 1
            if self.stopped[s] <= 0:
                del self.stopped[s]
        return out or None

    def decide(self, view):
        weights = self.base.decide(view)
        self.last_notes = getattr(self.base, "last_notes", None)
        if weights is None:
            return None
        for s in list(self.stopped):
            if weights.get(s, 0.0) <= 0:
                del self.stopped[s]  # the base dropped it: a later signal is a new trade
        out = {}
        for s, w in weights.items():
            if w <= 0 or s in self.stopped:
                continue
            p = self.book.get(s)
            if p is None:
                bars = view.bars(s, lookback=self.atr_days + 1)
                rng = avg_range(bars, self.atr_days) if len(bars) else float("nan")
                if not len(bars) or not np.isfinite(rng) or rng <= 0:
                    out[s] = w
                    continue
                entry = float(bars["close"].iloc[-1])
                risk = self.stop_atr * rng
                self.book[s] = p = Position(entry, entry - risk, risk, entry, bars.index[-1], bars.index[-1])
            out[s] = w * (1.0 - self.partial_frac) if p.partial_done else w
        for s in list(self.book):
            if s not in out:
                del self.book[s]
        return out


class MomentumManaged(Managed):
    name = "momentum_managed"
    base_name = "momentum"
    base_defaults = {"top_k": 15}


class SectorMomentumManaged(Managed):
    name = "sector_mom_managed"
    base_name = "sector_mom"
