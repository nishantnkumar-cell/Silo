import pandas as pd
import pytest

from silo.engine import run_backtest
from silo.firewall import FutureDataError, PointInTimeView
from silo.strategies import REGISTRY
from silo.strategies.base import Strategy


def test_view_hides_future_rows(tables):
    df = tables["AAA"]
    day = df.index[300]
    view = PointInTimeView(tables, df.loc[day, "available_at"])
    assert view.bars("AAA").index[-1] == day
    with pytest.raises(FutureDataError):
        view.bar_on("AAA", df.index[301])


def test_bar_not_public_before_close(tables):
    df = tables["AAA"]
    day = df.index[300]
    morning = pd.Timestamp(day).tz_localize("UTC") + pd.Timedelta(hours=14)
    view = PointInTimeView(tables, morning)
    assert view.bars("AAA").index[-1] == df.index[299]


class Spy(Strategy):
    """Records the newest date it ever sees, per decision."""

    name = "spy"

    def reset(self):
        self.seen = []

    def decide(self, view):
        newest = max(view.bars(s).index[-1] for s in self.symbols)
        self.seen.append((view.clock, newest))
        return None


def test_strategy_never_sees_past_its_clock(tables):
    spy = Spy(list(tables))
    run_backtest(spy, tables)
    for clock, newest in spy.seen:
        assert newest + pd.Timedelta(hours=21) <= clock.tz_convert(None)


@pytest.mark.parametrize("name", list(REGISTRY))
def test_truncation_invariance(tables, name):
    """Decisions before day T must not change when data after T is deleted.

    If they did, the strategy or engine would be reading the future.
    """
    symbols = list(tables)
    cut = tables["AAA"].index[500]
    truncated = {s: df[df.index <= cut] for s, df in tables.items()}
    full = run_backtest(REGISTRY[name](symbols), tables)
    part = run_backtest(REGISTRY[name](symbols), truncated)
    before = [d for d in full.decisions if d["date"] <= cut]
    assert before == part.decisions
    pd.testing.assert_series_equal(full.equity[full.equity.index <= cut], part.equity)
