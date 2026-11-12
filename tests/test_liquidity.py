import pandas as pd
import pytest

from silo.broker import CostModel, SimBroker
from silo.constraints import Constraints
from silo.data.sources import synthetic_bars
from silo.data.vault import normalize_bars
from silo.engine import run_backtest
from silo.firewall import PointInTimeView, PreparedTables
from silo.lab.regimes import label_regimes, regime_table
from silo.liquidity import Liquidity, apply_liquidity
from silo.markets import UK
from silo.strategies.base import Strategy


class EqualWeight(Strategy):
    name = "equal_weight_test"

    def decide(self, view):
        return {s: 1 / len(self.symbols) for s in self.symbols}
from silo.universe.uk import UK_CANDIDATES, screen


def _tables():
    good = normalize_bars(synthetic_bars(n_days=400, seed=1, start_price=2500), UK, pence=True)  # ~£25
    penny = normalize_bars(synthetic_bars(n_days=400, seed=2, start_price=40), UK, pence=True)  # ~40p
    thin = synthetic_bars(n_days=400, seed=3, start_price=1500)
    thin["volume"] = 2_000.0  # ~£30k a day
    thin = normalize_bars(thin, UK, pence=True)
    return {"GOOD.L": good, "PENNY.L": penny, "THIN.L": thin}


def _view(tables, i=300):
    clock = tables["GOOD.L"]["available_at"].iloc[i]
    return PointInTimeView(PreparedTables(tables), clock)


def test_gate_rejects_penny_and_thin_names():
    view, rules = _view(_tables()), Liquidity()
    assert rules.check(view, "GOOD.L")[0]
    ok, _, why = rules.check(view, "PENNY.L")
    assert not ok and "penny" in why
    ok, _, why = rules.check(view, "THIN.L")
    assert not ok and "illiquid" in why


def test_ineligible_names_can_be_held_or_trimmed_never_added():
    view = _view(_tables())
    w, blocked = apply_liquidity({"GOOD.L": 0.3, "PENNY.L": 0.3, "THIN.L": 0.05},
                                 {"THIN.L": 0.1}, view, 100_000, Liquidity())
    assert "PENNY.L" not in w and "PENNY.L" in blocked
    assert w["THIN.L"] == pytest.approx(0.05)  # trimming is allowed
    w, _ = apply_liquidity({"THIN.L": 0.5}, {"THIN.L": 0.1}, view, 100_000, Liquidity())
    assert w["THIN.L"] == pytest.approx(0.1)  # adding is not


def test_position_capped_to_what_can_be_sold_in_two_days():
    view = _view(_tables())
    ok, cap, _ = Liquidity().check(view, "GOOD.L")
    huge = cap * 20  # a book so big the cap binds
    w, blocked = apply_liquidity({"GOOD.L": 0.5}, {}, view, huge, Liquidity())
    assert w["GOOD.L"] * huge == pytest.approx(cap)
    assert "capped" in blocked["GOOD.L"]


def test_fills_capped_by_daily_volume_rest_waits():
    b = SimBroker(cash=1e6, costs=CostModel(0, 0, 0, 0, 0), max_participation=0.1)
    b.target_weights({"X.L": 0.5}, {"X.L": 10.0}, pd.Timestamp("2025-01-02"))
    b.fill_at_open({"X.L": 10.0}, pd.Timestamp("2025-01-03"), {"X.L": 20_000})
    assert b.positions["X.L"] == pytest.approx(2_000)
    assert b.pending and b.pending[0].qty == pytest.approx(48_000)
    b.fill_at_open({"X.L": 10.0}, pd.Timestamp("2025-01-06"), {"X.L": 20_000})
    assert b.positions["X.L"] == pytest.approx(4_000)


def test_engine_never_buys_penny_or_thin_stocks():
    tables = _tables()
    res = run_backtest(EqualWeight(list(tables)), tables, costs=CostModel.uk(),
                       constraints=Constraints.position_trading())
    bought = set(res.fills.loc[res.fills["qty"] > 0, "symbol"])
    assert bought == {"GOOD.L"}
    assert any("PENNY.L" in d.get("liquidity", {}) for d in res.decisions)


def test_screen_rejects_penny_stocks():
    out = screen(_tables())
    assert not out.loc["PENNY.L", "passes"] and "penny" in out.loc["PENNY.L", "reason"]


def test_uk_candidates_unique_and_wider():
    assert len(UK_CANDIDATES) == len(set(UK_CANDIDATES)) >= 120
    assert all(s.endswith(".L") for s in UK_CANDIDATES)


def test_regime_labels_use_only_the_past():
    eq = _tables()["GOOD.L"]["close"]
    full = label_regimes(eq)
    cut = label_regimes(eq.iloc[:300])
    pd.testing.assert_frame_equal(full.iloc[:300], cut)
    rows = regime_table(eq * 1.01, eq)
    assert {"rising market", "calm volatility"} & {r["state"] for r in rows}
