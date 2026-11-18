"""Matched random twins: no-skill books shaped like the book they are judging.

A strategy that holds 9 names and trades monthly cannot be judged against a
random book that holds 120 names: they differ in concentration, turnover,
fees and time in cash before any skill comes into it. A *twin* copies the
real book's shape and nothing else. On each day the real book decided, the
twin

* holds the same number of names with the same share of the book invested;
* keeps the same fraction of its previous names (so turnover and fees match);
* draws every new name at random from the shares that pass the liquidity
  rules at that moment (through the time firewall).

It then runs through the same engine, costs and rules. Where the real book
ranks among its twins is the luck check: a book that beats fewer than 95% of
them has shown shape, not skill.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from silo.liquidity import Liquidity
from silo.strategies.base import Strategy


def shape_of(decisions: list[dict]) -> dict[str, tuple[int, float, float]]:
    """clock -> (names held, fraction invested, fraction of the previous names kept)."""
    out, prev = {}, set()
    for d in decisions:
        names = {s for s, w in d["weights"].items() if w > 1e-4}
        kept = len(names & prev) / len(prev) if prev else 0.0
        out[str(d["clock"])] = (len(names), float(sum(d["weights"][s] for s in names)), kept)
        prev = names
    return out


class Twin(Strategy):
    name = "twin"

    def __init__(self, symbols, shape: dict[str, tuple[int, float, float]], seed: int = 0,
                 liquidity: Liquidity | None = None, min_history: int = 210):
        super().__init__(symbols)
        self.shape, self.seed, self.liquidity, self.min_history = shape, seed, liquidity, min_history

    def reset(self) -> None:
        self.held: list[str] = []

    def _eligible(self, view) -> list[str]:
        out = []
        for s in self.symbols:
            if len(view.bars(s, lookback=self.min_history)) < self.min_history:
                continue
            if self.liquidity is not None and not self.liquidity.check(view, s)[0]:
                continue
            out.append(s)
        return out

    def decide(self, view):
        spec = self.shape.get(str(view.clock))
        if spec is None:
            return None
        n, exposure, kept = spec
        if n == 0:
            self.held = []
            return {}
        day = int(view.clock.tz_convert(None).value // 86_400_000_000_000)
        rng = np.random.default_rng([self.seed, day])
        n_keep = min(n, int(round(kept * len(self.held))))
        stay = [str(s) for s in rng.choice(self.held, size=n_keep, replace=False)] if n_keep else []
        pool = [s for s in self._eligible(view) if s not in self.held]
        n_new = min(len(pool), n - len(stay))
        fresh = [str(s) for s in rng.choice(pool, size=n_new, replace=False)] if n_new > 0 else []
        self.held = stay + fresh
        if not self.held:
            return {}
        return {s: exposure / n for s in self.held}


@dataclass
class TwinResult:
    n: int
    sharpe_percentile: float  # share of twins the real book beat on Sharpe
    return_percentile: float
    median_return: float
    median_sharpe: float
    median_fees: float

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def run_twins(reference_decisions, reference_equity: pd.Series, symbols, data, days, costs, constraints,
              n: int = 20) -> TwinResult | None:
    """Run ``n`` twins of a finished book over the same ``days`` and rank the book among them."""
    from silo.engine import Session
    from silo.lab.stats import sharpe

    shape = shape_of(reference_decisions)
    if not shape or n <= 0 or len(reference_equity) < 3:
        return None
    liq = constraints.liquidity if constraints is not None else None
    rets, srs, fees = [], [], []
    for seed in range(n):
        session = Session.start(Twin(symbols, shape, seed, liq), 100_000.0,
                                costs, constraints)
        for d in days:
            session.step(d, data)
        res = session.result()
        rets.append(float(res.equity.iloc[-1] / res.equity.iloc[0] - 1))
        srs.append(sharpe(res.equity.pct_change()))
        fees.append(res.fees_paid)
    real_ret = float(reference_equity.iloc[-1] / reference_equity.iloc[0] - 1)
    real_sr = sharpe(reference_equity.pct_change())
    ann = float(np.sqrt(252))
    return TwinResult(n, float(np.mean([real_sr > x for x in srs])), float(np.mean([real_ret > x for x in rets])),
                      float(np.median(rets)), float(np.median(srs)) * ann, float(np.median(fees)))
