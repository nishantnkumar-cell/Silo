from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from silo.broker import CostModel, SimBroker
from silo.constraints import Constraints, apply_constraints, is_rebalance_day
from silo.data.sources import synthetic_bars
from silo.data.vault import DataVault, normalize_bars
from silo.engine import run_backtest
from silo.markets import UK, market_for_symbols
from silo.metrics import holding_stats
from silo.paper import PaperBook
from silo.strategies import SmaCross, TrendQuality
from silo.universe import screen


def uk_tables(n=4, n_days=900):
    syms = [f"T{i}.L" for i in range(n)]
    return syms, {s: normalize_bars(synthetic_bars(n_days=n_days, seed=i, start_price=2500), UK, pence=True)
                  for i, s in enumerate(syms)}


def test_stamp_duty_on_buys_only():
    b = SimBroker(cash=10_000, costs=CostModel.uk(spread_bps=0, slippage_bps=0))
    b.target_weights({"X.L": 0.5}, {"X.L": 10.0}, pd.Timestamp("2025-01-02"))
    b.fill_at_open({"X.L": 10.0}, pd.Timestamp("2025-01-03"))
    buy = b.fills[-1]
    assert buy["fee"] == pytest.approx(5.0 + 0.005 * buy["qty"] * 10.0)
    b.target_weights({}, {"X.L": 10.0}, pd.Timestamp("2025-01-03"))
    b.fill_at_open({"X.L": 10.0}, pd.Timestamp("2025-01-06"))
    assert b.fills[-1]["fee"] == pytest.approx(5.0)


def test_dividends_credited_to_holders():
    b = SimBroker(cash=1_000)
    b.positions["X.L"] = 100.0
    b.credit_dividends({"X.L": 0.12, "Y.L": 1.0}, pd.Timestamp("2025-03-01"))
    assert b.cash == pytest.approx(1_012.0)


def test_cannot_mix_markets():
    with pytest.raises(ValueError):
        market_for_symbols(["AZN.L", "AAPL"])


def test_weekly_rebalance_days():
    days = pd.bdate_range("2025-01-06", periods=10)
    flags = [is_rebalance_day(d, p, "weekly") for d, p in zip(days, [None, *days[:-1]])]
    assert flags == [True, False, False, False, False, True, False, False, False, False]


def test_constraints_hold_cap_and_turnover():
    c = Constraints(min_hold_days=20, max_turnover=0.3, max_weight=0.25)
    out = apply_constraints({"A": 0.6, "C": 0.4}, {"A": 0.2, "B": 0.3}, {"A": 30, "B": 5}, c)
    assert out["B"] >= 0.3 - 1e-9  # young position not cut
    assert all(v <= 0.3 + 1e-9 for v in out.values())
    turnover = sum(abs(out.get(s, 0) - {"A": 0.2, "B": 0.3}.get(s, 0)) for s in "ABC")
    assert turnover <= 0.3 + 1e-9
    assert sum(out.values()) <= 1 + 1e-9


def test_position_trading_backtest_trades_rarely():
    syms, tables = uk_tables()
    daily = run_backtest(SmaCross(syms, 5, 20), tables, costs=CostModel.uk())
    slow = run_backtest(SmaCross(syms, 5, 20), tables, costs=CostModel.uk(),
                        constraints=Constraints.position_trading())
    assert len(slow.fills) < len(daily.fills)
    assert {d["date"].isocalendar()[:2] for d in slow.decisions}.__len__() == len(slow.decisions)
    m = slow.metrics()
    assert m["avg_hold_days"] >= 20


def test_holding_stats():
    fills = pd.DataFrame({
        "date": pd.to_datetime(["2025-01-01", "2025-02-01", "2025-03-01"]),
        "symbol": ["A", "A", "B"], "qty": [10.0, -10.0, 5.0],
    })
    trips, hold = holding_stats(fills, pd.Timestamp("2025-03-11"))
    assert trips == 1 and hold == pytest.approx((31 + 10) / 2)


def test_screen_flags_volatility_and_liquidity():
    syms, tables = uk_tables(n=3, n_days=400)
    calm = normalize_bars(synthetic_bars(n_days=400, seed=9, vol=0.003, start_price=2500), UK, pence=True)
    tables["CALM.L"] = calm
    out = screen(tables, min_value_traded=1e6)
    assert not out.loc["CALM.L", "passes"]
    assert "volatility" in out.loc["CALM.L", "reason"]


def test_paper_book_is_forward_only_and_matches_backtest(tmp_path):
    syms, full = uk_tables(n_days=700)
    vault = DataVault(tmp_path / "data")
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for s, df in full.items():
        vault.put_bars(s, df.iloc[:500], fetched_at=t0)
    book = PaperBook("uk", tmp_path / "paper")
    book.create("uk_trend", syms, vault, {"lookback": 126, "top_k": 2})
    assert book.run(vault) == []  # nothing new yet: no backfilled history

    for k, cut in enumerate([560, 640, 700], start=1):  # new bars arrive over time
        for s, df in full.items():
            vault.put_bars(s, df.iloc[:cut], fetched_at=t0 + timedelta(days=k))
        book.run(vault)
    assert book.run(vault) == []  # rerunning the same day does nothing

    start = full[syms[0]].index[500]
    bt = run_backtest(TrendQuality(syms, lookback=126, top_k=2), full, start=start,
                      costs=CostModel.uk(), constraints=Constraints.position_trading())
    st = book.status()
    assert st["days"] == 200
    assert st["equity"] == pytest.approx(bt.equity.iloc[-1])
    lines = (tmp_path / "paper" / "uk" / "journal.jsonl").read_text().splitlines()
    assert len(lines) == 1 + 200
