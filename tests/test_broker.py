import pandas as pd
import pytest

from silo.broker import CostModel, SimBroker


def test_fill_at_next_open_with_costs():
    costs = CostModel(commission_per_share=0.01, min_commission=1.0, spread_bps=10, slippage_bps=5)
    b = SimBroker(cash=10_000, costs=costs)
    b.target_weights({"X": 0.5}, {"X": 100.0}, pd.Timestamp("2024-01-02"))
    assert b.positions == {}  # nothing fills at decision time
    b.fill_at_open({"X": 102.0}, pd.Timestamp("2024-01-03"))
    fill = b.fills[0]
    assert fill["price"] == pytest.approx(102.0 * (1 + 10e-4))  # half spread 5bp + slippage 5bp
    assert fill["qty"] == pytest.approx(50.0)
    assert b.cash == pytest.approx(10_000 - 50 * fill["price"] - 1.0)


def test_rejects_leverage_and_shorts():
    b = SimBroker()
    with pytest.raises(ValueError):
        b.target_weights({"X": 0.7, "Y": 0.6}, {"X": 1.0, "Y": 1.0}, pd.Timestamp("2024-01-02"))
    with pytest.raises(ValueError):
        b.target_weights({"X": -0.5}, {"X": 1.0}, pd.Timestamp("2024-01-02"))


def test_buys_scaled_to_cash():
    b = SimBroker(cash=1_000)
    b.target_weights({"X": 1.0}, {"X": 10.0}, pd.Timestamp("2024-01-02"))
    b.fill_at_open({"X": 20.0}, pd.Timestamp("2024-01-03"))  # gapped up: can't afford full size
    assert b.cash >= 0
    assert b.positions["X"] < 100
