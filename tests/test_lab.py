import numpy as np
import pandas as pd
import pytest

from silo.cli import main
from silo.engine import run_backtest
from silo.lab import HoldoutLog, run_study
from silo.lab.controls import activity_profile
from silo.lab.report import render
from silo.lab.stats import bootstrap_sharpe_ci, deflated_sharpe, expected_max_sharpe, probabilistic_sharpe
from silo.lab.walkforward import walk_forward
from silo.strategies import RandomEntry, SmaCross


def test_psr_rises_with_true_edge():
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 0.01, 1000)
    edge = rng.normal(0.002, 0.01, 1000)
    assert probabilistic_sharpe(edge) > 0.99
    assert 0.05 < probabilistic_sharpe(noise - noise.mean()) < 0.95


def test_deflated_sharpe_penalises_many_trials():
    rng = np.random.default_rng(2)
    r = rng.normal(0.0006, 0.01, 750)
    few = deflated_sharpe(r, list(rng.normal(0, 0.02, 2)))
    many = deflated_sharpe(r, list(rng.normal(0, 0.02, 200)))
    assert many < few
    assert expected_max_sharpe(list(rng.normal(0, 0.02, 200))) > expected_max_sharpe(list(rng.normal(0, 0.02, 5)))


def test_bootstrap_interval_contains_sample_sharpe():
    rng = np.random.default_rng(3)
    r = rng.normal(0.0005, 0.01, 1000)
    lo, hi = bootstrap_sharpe_ci(r, n_boot=300)
    sample = r.mean() / r.std(ddof=1) * np.sqrt(252)
    assert lo < sample < hi


def test_walk_forward_never_tests_on_training_data(tables):
    wf = walk_forward(SmaCross, list(tables), tables, n_folds=3, grid={"fast": [10, 20], "slow": [50]})
    assert len(wf.folds) == 3 and wf.n_trials == 6
    for f in wf.folds:
        assert f.train_end < f.test_start
    for a, b in zip(wf.folds, wf.folds[1:]):
        assert a.test_end < b.test_start
    assert wf.oos_returns.index.is_unique


def test_random_entry_matches_activity(tables):
    symbols = list(tables)
    ref = run_backtest(SmaCross(symbols, 20, 50), tables)
    p, exposure = activity_profile(ref, len(symbols))
    rnd = run_backtest(RandomEntry(symbols, p, exposure, seed=7), tables)
    p2, exposure2 = activity_profile(rnd, len(symbols))
    assert p2 == pytest.approx(p, rel=0.6)
    assert exposure2 == pytest.approx(exposure, abs=0.25)


def test_holdout_log_flags_reopening(tmp_path, tables):
    log = HoldoutLog(tmp_path / "log.jsonl")
    kw = dict(n_folds=2, n_random=3, open_holdout=True, holdout_log=log)
    first = run_study(SmaCross, list(tables), tables, **kw)
    second = run_study(SmaCross, list(tables), tables, **kw)
    assert first.holdout["reopened"] is False
    assert second.holdout["reopened"] is True
    assert not second.checks[-1].passed


def test_study_on_noise_is_not_interesting_and_report_renders(tables):
    result = run_study(SmaCross, list(tables), tables, n_folds=2, n_random=5)
    assert not result.interesting  # random walks hold no edge to find
    page = render(result)
    assert "not financial advice" in page and "chart-oos" in page
    assert result.holdout_start > result.oos_equity.index[-1]


def test_cli_lab_demo(tmp_path, capsys):
    assert main(["lab", "--demo", "--strategy", "sma_cross", "--folds", "2", "--random", "3", "--out", str(tmp_path)]) == 0
    assert (tmp_path / "sma_cross.html").exists()
