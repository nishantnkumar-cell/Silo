"""Random-entry control: a strategy with no information, matched in activity.

If a real strategy cannot beat most random strategies that trade as often and
stay invested as much, its result is indistinguishable from luck.
"""

from __future__ import annotations

import pandas as pd

from silo.engine import BacktestResult, run_backtest
from silo.strategies.random_entry import RandomEntry


def activity_profile(result: BacktestResult, n_symbols: int) -> tuple[float, float]:
    """(decisions per day, average fraction invested) of a backtest."""
    days = max(1, len(result.equity))
    decide_prob = min(1.0, max(1, len(result.decisions)) / days)
    if not result.decisions:
        return decide_prob, 0.0
    w = pd.Series(
        {d["date"]: sum(d["weights"].values()) for d in result.decisions}, dtype=float
    ).reindex(result.equity.index).ffill().fillna(0.0)
    return decide_prob, float(w.mean())


def random_control(
    reference: BacktestResult,
    symbols: list[str],
    tables: dict[str, pd.DataFrame],
    n_runs: int = 30,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    **kwargs,
) -> list[BacktestResult]:
    decide_prob, exposure = activity_profile(reference, len(symbols))
    return [
        run_backtest(RandomEntry(symbols, decide_prob, exposure, seed), tables, start=start, end=end, **kwargs)
        for seed in range(n_runs)
    ]
