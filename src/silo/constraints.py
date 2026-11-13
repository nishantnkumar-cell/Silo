"""Position-trading rules enforced in code, not left to a strategy's goodwill.

Silo targets holding periods of weeks to months, so by default a strategy is
only consulted at rebalance points (weekly), recently opened positions cannot
be cut before a minimum holding period, and each rebalance can only trade a
capped share of the portfolio.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from silo.liquidity import Liquidity


@dataclass(frozen=True)
class Constraints:
    rebalance: str = "daily"  # "daily", "weekly" or "monthly"
    min_hold_days: int = 0  # trading days a new position must be held
    max_turnover: float | None = None  # max sum |delta weight| per rebalance
    max_weight: float | None = None  # cap on any one position
    liquidity: Liquidity | None = None  # see silo.liquidity
    rebalance_band: float = 0.0  # skip resizing a held name by less than this weight (cuts fee churn)


    @classmethod
    def position_trading(cls, **overrides) -> Constraints:
        """Weeks-to-months defaults: weekly decisions, 20-day minimum hold,
        at most 10% in one name, and the liquidity rules on."""
        base = dict(rebalance="weekly", min_hold_days=20, max_turnover=0.5, max_weight=0.1,
                    liquidity=Liquidity(), rebalance_band=0.02)
        base.update(overrides)
        return cls(**base)

    @classmethod
    def daily_trading(cls, **overrides) -> Constraints:
        """Daily-bar trading for a separate silver/crypto book (not intraday):
        decide at each daily close, fill at the next open, hold at least one
        full day, at most 50% in one name, and the liquidity rules on."""
        base = dict(rebalance="daily", min_hold_days=1, max_turnover=None, max_weight=0.5,
                    liquidity=Liquidity())
        base.update(overrides)
        return cls(**base)


def is_rebalance_day(date: pd.Timestamp, previous: pd.Timestamp | None, rule: str) -> bool:
    """True on the first trading day of each new period. Uses no future data."""
    if rule == "daily" or previous is None:
        return True
    if rule == "weekly":
        return date.isocalendar()[:2] != previous.isocalendar()[:2]
    if rule == "monthly":
        return (date.year, date.month) != (previous.year, previous.month)
    raise ValueError(f"unknown rebalance rule {rule!r}")


def apply_constraints(
    target: dict[str, float],
    current: dict[str, float],
    held_days: dict[str, int],
    c: Constraints,
) -> dict[str, float]:
    """Adjust target weights to respect the holding, size and turnover rules."""
    w = {s: max(0.0, v) for s, v in target.items()}
    if c.max_weight is not None:
        w = {s: min(v, c.max_weight) for s, v in w.items()}

    locked = {s for s, d in held_days.items() if d < c.min_hold_days and current.get(s, 0) > 0}
    for s in locked:  # a young position may grow but not shrink
        w[s] = max(w.get(s, 0.0), current[s])

    if c.rebalance_band:  # a small resize of a name we keep isn't worth a fee
        for s in list(w):
            if current.get(s, 0) > 0 and w[s] > 0 and abs(w[s] - current[s]) < c.rebalance_band:
                w[s] = current[s]

    total = sum(w.values())
    if total > 1.0:
        free = [s for s in w if s not in locked]
        excess = total - 1.0
        free_sum = sum(w[s] for s in free)
        if free_sum > 0:
            scale = max(0.0, (free_sum - excess) / free_sum)
            for s in free:
                w[s] *= scale
        total = sum(w.values())
        if total > 1.0:  # young positions alone are over budget: limit how much they grow
            base = sum(current[s] for s in locked) + sum(w[s] for s in free)
            growth = sum(w[s] - current[s] for s in locked)
            k = max(0.0, (1.0 - base) / growth) if growth > 0 else 0.0
            for s in locked:
                w[s] = current[s] + k * (w[s] - current[s])
        total = sum(w.values())
        if total > 1.0:  # drift alone left current weights a hair over 1
            w = {s: v / total for s, v in w.items()}

    if c.max_turnover is not None:
        names = set(w) | set(current)
        turnover = sum(abs(w.get(s, 0.0) - current.get(s, 0.0)) for s in names)
        if turnover > c.max_turnover > 0:
            k = c.max_turnover / turnover
            w = {s: current.get(s, 0.0) + k * (w.get(s, 0.0) - current.get(s, 0.0)) for s in names}
            for s in locked:
                w[s] = max(w[s], current[s])
    return {s: v for s, v in w.items() if v > 1e-9}
