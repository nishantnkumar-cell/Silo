"""Does learning from past trades actually help? Test the whole loop, walk-forward.

For each fold: run the agent on the training window, build its trade
journal, review it (statistics, and optionally an AI reviewer), then trade
the *next* window with the lessons from that review. The lessons never see
the window they trade. Stitching the test windows gives an out-of-sample
record of "the agent that learns", which is compared with the same agent
never learning, and with buy-and-hold.

This is the honest answer to "is it improving on what works?". A loop that
only looks good on the trades it learned from is overfitting.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from silo.engine import run_backtest
from silo.lab.stats import deflated_sharpe, sharpe
from silo.lab.walkforward import fold_bounds
from silo.learning.gate import ANNUAL, skipped_counts
from silo.learning.journal import build_journal, summary
from silo.learning.lessons import Learned
from silo.learning.review import ai_review, mine
from silo.strategies import REGISTRY
from silo.strategies.classic import BuyAndHold


@dataclass
class LearningFold:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    trades: int
    splits_tried: int
    lessons: list[dict]
    fired: int
    base_sharpe: float
    learned_sharpe: float
    bench_sharpe: float


@dataclass
class LearningWalkForward:
    base: str
    folds: list[LearningFold]
    base_returns: pd.Series
    learned_returns: pd.Series
    bench_returns: pd.Series
    trial_sharpes: list[float] = field(default_factory=list)

    def metrics(self) -> dict:
        diff = (self.learned_returns - self.base_returns).to_numpy()
        return {
            "base_sharpe": sharpe(self.base_returns) * ANNUAL,
            "learned_sharpe": sharpe(self.learned_returns) * ANNUAL,
            "bench_sharpe": sharpe(self.bench_returns) * ANNUAL,
            "base_return": float((1 + self.base_returns).prod() - 1),
            "learned_return": float((1 + self.learned_returns).prod() - 1),
            "bench_return": float((1 + self.bench_returns).prod() - 1),
            "deflated": float(deflated_sharpe(diff, self.trial_sharpes)) if np.any(diff) else 0.0,
            "folds_improved": int(sum(f.learned_sharpe > f.base_sharpe for f in self.folds)),
            "folds": len(self.folds),
        }

    def to_dict(self) -> dict:
        return {"base": self.base, "metrics": self.metrics(),
                "folds": [{**f.__dict__, **{k: str(pd.Timestamp(getattr(f, k)).date()) for k in
                                            ("train_start", "train_end", "test_start", "test_end")}}
                          for f in self.folds]}


def learning_walk_forward(base: str, symbols, tables, end=None, n_folds: int = 4, train_frac: float = 0.4,
                          max_train: int | None = 756, base_params: dict | None = None, client=None,
                          max_lessons: int = 2, core: bool | None = None, costs=None, constraints=None,
                          progress=None) -> LearningWalkForward:
    """Walk-forward test of learn-then-trade. ``end`` excludes the holdout. ``client`` adds the AI reviewer."""
    params = base_params or {}
    core = base.startswith("core") if core is None else core
    dates = sorted(set().union(*(tables[s].index for s in symbols)))
    if end is not None:
        dates = [d for d in dates if d < pd.Timestamp(end)]
    kw = {"costs": costs, "constraints": constraints}
    folds, pb, pl, pm, trials = [], [], [], [], []
    for tr0, tr1, te0, te1 in fold_bounds(dates, n_folds, train_frac, max_train):
        train = run_backtest(REGISTRY[base](symbols, **params), tables, start=tr0, end=tr1, **kw)
        trades = build_journal(train, tables, symbols, core=core)
        cands, tried = mine(trades)
        lessons = ai_review(trades, client, cands, max_lessons) if client is not None else \
            [c.lesson for c in cands[:max_lessons]]
        b = run_backtest(REGISTRY[base](symbols, **params), tables, start=te0, end=te1, **kw)
        lr = run_backtest(Learned(symbols, base, params, lessons), tables, start=te0, end=te1, **kw)
        bh = run_backtest(BuyAndHold(symbols), tables, start=te0, end=te1, **kw)
        rb, rl, rm = (x.equity.pct_change().fillna(0.0) for x in (b, lr, bh))
        pb.append(rb)
        pl.append(rl)
        pm.append(rm)
        trials.append(sharpe((rl - rb).to_numpy()))
        fired, _ = skipped_counts(lr.decisions)
        folds.append(LearningFold(tr0, tr1, te0, te1, summary(trades).get("closed", 0), tried,
                                  [x.to_dict() for x in lessons], fired, sharpe(rb) * ANNUAL,
                                  sharpe(rl) * ANNUAL, sharpe(rm) * ANNUAL))
        if progress:
            progress(f"  fold {len(folds)}: learned {len(lessons)} lesson(s) from {folds[-1].trades} trades; "
                     f"test Sharpe {folds[-1].learned_sharpe:.2f} learned vs {folds[-1].base_sharpe:.2f} unchanged")
    return LearningWalkForward(base, folds, pd.concat(pb), pd.concat(pl), pd.concat(pm), trials)
