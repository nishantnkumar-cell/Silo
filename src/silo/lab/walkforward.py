"""Walk-forward evaluation: pick parameters on the past, score on what follows.

The research period is split into an initial training window and ``n_folds``
test windows. For each fold, every parameter set in the strategy's grid is
backtested on the training window before the fold; the best by Sharpe is then
run, untouched, on the test window. Stitching the test windows gives an
out-of-sample equity curve no parameter choice ever saw.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import pandas as pd

from silo.engine import run_backtest
from silo.lab.stats import sharpe


@dataclass
class Fold:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    params: dict
    train_sharpe: float
    test_sharpe: float


@dataclass
class WalkForwardResult:
    strategy: str
    folds: list[Fold]
    oos_returns: pd.Series
    trial_sharpes: list[float] = field(default_factory=list)
    oos_decisions: list[dict] = field(default_factory=list)
    oos_fills: pd.DataFrame | None = None

    @property
    def n_trials(self) -> int:
        return len(self.trial_sharpes)

    def equity(self, cash: float = 100_000.0) -> pd.Series:
        return cash * (1 + self.oos_returns).cumprod()


def param_sets(grid: dict[str, list]) -> list[dict]:
    if not grid:
        return [{}]
    keys = sorted(grid)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(grid[k] for k in keys))]


def fold_bounds(dates: list[pd.Timestamp], n_folds: int, train_frac: float, max_train: int | None):
    first_test = int(len(dates) * train_frac)
    size = (len(dates) - first_test) // n_folds
    if size < 20:
        raise ValueError("not enough data for that many folds")
    for i in range(n_folds):
        t0 = first_test + i * size
        t1 = len(dates) - 1 if i == n_folds - 1 else t0 + size - 1
        tr0 = 0 if max_train is None else max(0, t0 - max_train)
        yield dates[tr0], dates[t0 - 1], dates[t0], dates[t1]


def select_params(strategy_cls, symbols, tables, start, end, grid=None, **kwargs):
    """Best parameter set by Sharpe on [start, end], plus every trial's Sharpe."""
    grid = getattr(strategy_cls, "param_grid", {}) if grid is None else grid
    scored = []
    for params in param_sets(grid):
        res = run_backtest(strategy_cls(symbols, **params), tables, start=start, end=end, **kwargs)
        scored.append((sharpe(res.equity.pct_change()), params))
    best_sr, best = max(scored, key=lambda x: x[0])
    return best, best_sr, [sr for sr, _ in scored]


def walk_forward(
    strategy_cls,
    symbols: list[str],
    tables: dict[str, pd.DataFrame],
    end: str | pd.Timestamp | None = None,
    n_folds: int = 4,
    train_frac: float = 0.4,
    max_train: int | None = 756,
    grid: dict[str, list] | None = None,
    **kwargs,
) -> WalkForwardResult:
    """Walk-forward over data up to ``end`` (exclusive of any holdout)."""
    grid = getattr(strategy_cls, "param_grid", {}) if grid is None else grid
    dates = sorted(set().union(*(tables[s].index for s in symbols)))
    if end is not None:
        dates = [d for d in dates if d < pd.Timestamp(end)]
    folds, pieces, trials, decisions, fills = [], [], [], [], []
    for tr0, tr1, te0, te1 in fold_bounds(dates, n_folds, train_frac, max_train):
        best, best_sr, fold_trials = select_params(strategy_cls, symbols, tables, tr0, tr1, grid, **kwargs)
        trials.extend(fold_trials)
        test = run_backtest(strategy_cls(symbols, **best), tables, start=te0, end=te1, **kwargs)
        rets = test.equity.pct_change().fillna(0.0)
        pieces.append(rets)
        decisions.extend(test.decisions)
        fills.append(test.fills)
        folds.append(Fold(tr0, tr1, te0, te1, best, best_sr, sharpe(rets)))
    all_fills = pd.concat([f for f in fills if not f.empty]) if any(not f.empty for f in fills) else None
    return WalkForwardResult(strategy_cls.name, folds, pd.concat(pieces), trials, decisions, all_fills)
