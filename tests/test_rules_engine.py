import numpy as np
import pandas as pd
import pytest

from silo.constraints import Constraints
from silo.data.sources import synthetic_bars
from silo.data.vault import normalize_bars
from silo.engine import run_backtest
from silo.firewall import PointInTimeView
from silo.strategies import REGISTRY
from silo.strategies.rules_engine import RulesEngine


def trending(n=500, seed=0, drift=0.0015, vol=0.01):
    return normalize_bars(synthetic_bars(n_days=n, seed=seed, drift=drift, vol=vol))


def view_at(tables, i):
    some = next(iter(tables.values()))
    return PointInTimeView(tables, some["available_at"].iloc[i])


@pytest.fixture
def up_tables():
    return {f"S{i}": trending(seed=i) for i in range(6)}


def test_registered_and_risk_bounds():
    assert REGISTRY["engine"] is RulesEngine
    RulesEngine(["A"], risk_per_trade=0.05)  # 5% is the most a user may choose
    for bad in (0.0, 0.051, 0.1):
        with pytest.raises(ValueError):
            RulesEngine(["A"], risk_per_trade=bad)


def test_size_is_risk_over_stop_distance(up_tables):
    eng = RulesEngine(list(up_tables), risk_per_trade=0.01, max_weight=1.0, max_open_risk=1.0, jump_limit=None,
                      max_names=3)  # total stays under 100%, so nothing is scaled down
    w = eng.decide(view_at(up_tables, 400))
    assert w
    for s, weight in w.items():
        p = eng.book[s]
        dist = p.risk / p.entry
        assert weight == pytest.approx(0.01 / dist)
        assert p.stop == pytest.approx(p.entry - p.risk)


def test_weight_cap_and_open_risk_cap(up_tables):
    capped = RulesEngine(list(up_tables), risk_per_trade=0.05, max_weight=0.10, max_open_risk=1.0, jump_limit=None)
    w = capped.decide(view_at(up_tables, 400))
    assert w and max(w.values()) <= 0.10 + 1e-12

    tight = RulesEngine(list(up_tables), risk_per_trade=0.01, max_weight=1.0, max_open_risk=0.025, jump_limit=None)
    w = tight.decide(view_at(up_tables, 400))
    assert len(w) == 2  # two trades at 1% risk fit under a 2.5% cap, the third does not
    assert "R10" in next(iter(tight.last_notes["skipped"].values()))


def test_market_filter_blocks_new_buys(up_tables):
    falling = trending(seed=99, drift=-0.003)
    tables = dict(up_tables, MKT=falling)
    on = RulesEngine(list(tables), market="MKT", jump_limit=None)
    assert on.decide(view_at(tables, 400)) == {}
    assert "no new buys" in on.last_notes["market"]
    off = RulesEngine(list(tables), market="MKT", market_filter=False, jump_limit=None)
    w = off.decide(view_at(tables, 400))
    assert w and "MKT" not in w  # the market line is never traded


def test_jump_filter_excludes_gappers(up_tables):
    jumpy = trending(seed=1).copy()
    i = jumpy.index[380]
    jumpy.loc[jumpy.index >= i, ["open", "high", "low", "close"]] *= 1.3
    tables = dict(up_tables, S1=jumpy)
    eng = RulesEngine(list(tables), max_weight=1.0, max_open_risk=1.0)
    w = eng.decide(view_at(tables, 400))
    assert "S1" not in w


def test_stop_sells_and_blocks_rebuy(up_tables):
    eng = RulesEngine(list(up_tables), max_open_risk=1.0, jump_limit=None)
    eng.decide(view_at(up_tables, 400))
    s = next(iter(eng.book))
    eng.book[s].stop = 10**9  # any close is now below the stop
    out = eng.manage(view_at(up_tables, 401))
    assert out[s] == 0.0 and s not in eng.book
    assert eng.blocked[s][1].startswith("R8")
    assert any(e["rule"] == "R13 stop" for e in eng.log)


def test_partial_at_2r_then_trail(up_tables):
    eng = RulesEngine(list(up_tables), max_open_risk=1.0, jump_limit=None)
    eng.decide(view_at(up_tables, 400))
    s = next(iter(eng.book))
    p = eng.book[s]
    p.risk = 1e-6  # the next high is far above entry + 2R
    before = eng.weight[s]
    out = eng.manage(view_at(up_tables, 401))
    assert out[s] == pytest.approx(0.5)
    assert p.partial_done and p.stop >= p.entry
    assert eng.weight[s] == pytest.approx(before * 0.5)


def test_same_data_same_trades(up_tables):
    runs = [run_backtest(RulesEngine(list(up_tables)), up_tables, constraints=Constraints.position_trading(liquidity=None))
            for _ in range(2)]
    pd.testing.assert_frame_equal(runs[0].fills, runs[1].fills)
    assert np.allclose(runs[0].equity.to_numpy(), runs[1].equity.to_numpy())
    assert len(runs[0].fills)  # it actually traded
