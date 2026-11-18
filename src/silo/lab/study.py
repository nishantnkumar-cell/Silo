"""One honest study of one strategy: walk-forward, controls, luck checks, holdout."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from silo.engine import BacktestResult, run_backtest
from silo.lab.controls import random_control
from silo.lab.holdout import HoldoutLog
from silo.lab.stats import bootstrap_sharpe_ci, deflated_sharpe, sharpe
from silo.lab.walkforward import WalkForwardResult, select_params, walk_forward
from silo.metrics import periods_per_year, summarize
from silo.strategies.classic import BuyAndHold

ANNUAL = np.sqrt(252)


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


@dataclass
class LabResult:
    strategy: str
    symbols: list[str]
    wf: WalkForwardResult
    benchmark: BacktestResult
    randoms: list[BacktestResult]
    holdout_start: pd.Timestamp
    dsr: float
    sharpe_ci: tuple[float, float]
    random_percentile: float
    holdout: dict | None = None
    model_cutoff: str | None = None
    annual: float = float(ANNUAL)  # sqrt(periods per year): 252 weekday bars, 365 for crypto
    checks: list[Check] = field(default_factory=list)

    @property
    def oos_equity(self) -> pd.Series:
        return self.wf.equity()

    def oos_metrics(self) -> dict[str, float]:
        return summarize(self.oos_equity, self.wf.oos_fills)

    def benchmark_metrics(self) -> dict[str, float]:
        return summarize(self.benchmark.equity, self.benchmark.fills)

    @property
    def interesting(self) -> bool:
        return bool(self.checks) and all(c.passed for c in self.checks)


def _span(dates, frac):
    return dates[int(len(dates) * (1 - frac))]


def run_study(
    strategy_cls,
    symbols: list[str],
    tables: dict[str, pd.DataFrame],
    holdout_frac: float = 0.2,
    n_folds: int = 4,
    n_random: int = 30,
    open_holdout: bool = False,
    holdout_log: HoldoutLog | None = None,
    model_cutoff: str | None = None,
    trial_log=None,
    **kwargs,
) -> LabResult:
    dates = sorted(set().union(*(tables[s].index for s in symbols)))
    holdout_start = _span(dates, holdout_frac)

    wf = walk_forward(strategy_cls, symbols, tables, end=holdout_start, n_folds=n_folds, **kwargs)
    oos = wf.oos_returns
    oos_start, oos_end = oos.index[0], oos.index[-1]

    bench = run_backtest(BuyAndHold(symbols), tables, start=oos_start, end=oos_end, **kwargs)
    reference = BacktestResult(wf.strategy, wf.equity(), pd.DataFrame(), wf.oos_decisions)
    randoms = random_control(reference, symbols, tables, n_random, start=oos_start, end=oos_end, **kwargs)

    ann = float(np.sqrt(periods_per_year(oos.index)))
    sr = sharpe(oos)
    random_srs = [sharpe(r.equity.pct_change()) for r in randoms]
    pct = float(np.mean([sr > x for x in random_srs])) if random_srs else float("nan")
    bench_sr = sharpe(bench.equity.pct_change())
    # Luck test: count this study's parameter trials and every earlier attempt on overlapping dates.
    trials = list(wf.trial_sharpes)
    if trial_log is not None:
        key = trial_log.key(wf.strategy, {}, len(symbols), oos_start, oos_end)
        trials += trial_log.overlapping(oos_start, oos_end, exclude_key=key)
        trial_log.record("lab", wf.strategy, {}, len(symbols), oos_start, oos_end, sr)
    n_counted = len(trials)
    result = LabResult(
        strategy=wf.strategy,
        symbols=list(symbols),
        wf=wf,
        benchmark=bench,
        randoms=randoms,
        holdout_start=holdout_start,
        dsr=deflated_sharpe(oos, trials),
        sharpe_ci=bootstrap_sharpe_ci(oos, periods=periods_per_year(oos.index)),
        random_percentile=pct,
        model_cutoff=model_cutoff,
        annual=ann,
    )
    result.checks = [
        Check(
            "Beats buy-and-hold, out of sample, after costs",
            sr > bench_sr,
            f"Sharpe {sr * ann:.2f} vs {bench_sr * ann:.2f}",
        ),
        Check(
            "Survives the deflated Sharpe test",
            result.dsr >= 0.95,
            f"DSR {result.dsr:.2f} after {wf.n_trials} parameter trials"
            + (f" and {n_counted - wf.n_trials} earlier attempts on these dates" if n_counted > wf.n_trials else "")
            + " (needs 0.95)",
        ),
        Check(
            "Beats 95% of random strategies with the same activity",
            pct >= 0.95,
            f"beats {pct:.0%} of {len(randoms)} random runs",
        ),
    ]

    if model_cutoff:
        clean = model_cutoff != "unknown" and oos_start > pd.Timestamp(model_cutoff) + pd.offsets.MonthEnd(0)
        result.checks.append(Check(
            "Tested only after the model's training cutoff",
            bool(clean),
            f"model cutoff {model_cutoff}; out-of-sample starts {oos_start.date()}"
            + ("" if clean else ": the model may remember what happened, so treat this as contaminated"),
        ))

    if open_holdout:
        result.holdout = _open_holdout(strategy_cls, symbols, tables, holdout_start, holdout_log, **kwargs)
        h = result.holdout
        result.checks.append(
            Check(
                "Beats buy-and-hold on the locked holdout",
                h["sharpe"] > h["benchmark_sharpe"] and not h["reopened"],
                f"Sharpe {h['sharpe'] * ann:.2f} vs {h['benchmark_sharpe'] * ann:.2f}"
                + (f"; holdout already opened {h['previous_openings']} time(s)" if h["reopened"] else ""),
            )
        )
    return result


def _open_holdout(strategy_cls, symbols, tables, holdout_start, log, max_train: int = 756, **kwargs) -> dict:
    dates = sorted(set().union(*(tables[s].index for s in symbols)))
    research = [d for d in dates if d < holdout_start]
    train_start = research[max(0, len(research) - max_train)]
    params, _, _ = select_params(strategy_cls, symbols, tables, train_start, research[-1], **kwargs)
    run = run_backtest(strategy_cls(symbols, **params), tables, start=holdout_start, **kwargs)
    bench = run_backtest(BuyAndHold(symbols), tables, start=holdout_start, **kwargs)
    log = log or HoldoutLog()
    key = log.key(strategy_cls.name, params, symbols, str(holdout_start.date()))
    before = log.record(key, strategy=strategy_cls.name, params=params, symbols=symbols)
    return {
        "params": params,
        "equity": run.equity,
        "benchmark_equity": bench.equity,
        "sharpe": sharpe(run.equity.pct_change()),
        "benchmark_sharpe": sharpe(bench.equity.pct_change()),
        "metrics": summarize(run.equity, run.fills),
        "benchmark_metrics": summarize(bench.equity, bench.fills),
        "reopened": before > 0,
        "previous_openings": before,
    }
