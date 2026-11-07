import pytest

from silo.engine import run_backtest
from silo.strategies import REGISTRY, SmaCross, VolTarget


@pytest.mark.parametrize("name", list(REGISTRY))
def test_strategies_run_and_report(tables, name):
    result = run_backtest(REGISTRY[name](list(tables)), tables)
    m = result.metrics()
    assert m["days"] == len(tables["AAA"])
    assert -1 < m["total_return"] < 10
    assert result.equity.iloc[0] == pytest.approx(100_000)


def test_vol_target_wrapper_runs(tables):
    result = run_backtest(VolTarget(SmaCross(list(tables), 20, 50)), tables)
    assert result.metrics()["max_drawdown"] <= 0


def test_costs_reduce_returns(tables):
    from silo.broker import CostModel

    cheap = run_backtest(REGISTRY["bollinger"](list(tables)), tables, costs=CostModel(0, 0, 0, 0))
    dear = run_backtest(REGISTRY["bollinger"](list(tables)), tables, costs=CostModel(0.01, 5, 20, 20))
    assert dear.equity.iloc[-1] < cheap.equity.iloc[-1]
