import re

import numpy as np
import pandas as pd
import pytest

from silo.agents import crossasset
from silo.agents.informed import InformedDesk, InformedPanel
from silo.agents.llm import LLMClient
from silo.data.macro import DRIVERS, boe_stamp, from_yahoo_frame, parse_boe_csv, synthetic_macro, table_name
from silo.data.sources import synthetic_bars
from silo.data.vault import DataVault, normalize_bars
from silo.engine import MarketData, run_backtest
from silo.firewall import PointInTimeView, PreparedTables
from silo.markets import UK
from silo.strategies import REGISTRY
from silo.universe.sectors import SECTORS


def uk_tables(n=6, n_days=700):
    syms = [f"T{i}.L" for i in range(n)]
    t = {s: normalize_bars(synthetic_bars(n_days=n_days, seed=i, start_price=2500), UK, pence=True)
         for i, s in enumerate(syms)}
    return syms, t | synthetic_macro(t[syms[0]].index, seed=1)


def test_context_tables_are_point_in_time_and_never_traded():
    syms, t = uk_tables()
    day = t[syms[0]].index[400]
    view = PointInTimeView(PreparedTables(t), t[syms[0]].loc[day, "available_at"])
    assert view.symbols == syms
    oil = view.context(table_name("oil"))
    assert oil.index[-1] == t[syms[0]].index[399]  # stamped 23:00 UTC: today's value not public at the UK close
    gilt = view.context(table_name("gilt10"))
    assert gilt.index[-1] < day
    assert view.context("ctx:missing").empty
    data = MarketData(t)
    assert set(data.rows) == set(syms)
    assert data.dates == sorted(t[syms[0]].index)


def test_vault_context_snapshots(tmp_path):
    vault = DataVault(tmp_path)
    syms, t = uk_tables(n=2, n_days=300)
    oil = t[table_name("oil")]
    vault.put_bars(syms[0], t[syms[0]])
    vault.put_context(table_name("oil"), oil.iloc[:200])
    revised = oil.iloc[150:].copy()
    revised["value"] *= 2
    got = vault.put_context(table_name("oil"), revised)
    assert len(got) == len(oil)
    assert got["value"].iloc[0] == oil["value"].iloc[0] and got["value"].iloc[-1] == revised["value"].iloc[-1]
    assert got["available_at"].is_monotonic_increasing
    assert vault.context_names() == [table_name("oil")]
    tables = vault.tables([syms[0]])
    assert set(tables) == {syms[0], table_name("oil")}
    assert len(list((tmp_path / "context").rglob("*.parquet"))) == 2
    with pytest.raises(ValueError):
        vault.put_context("oil", oil)


def test_boe_csv_and_stamps():
    text = "DATE,IUDMNPY\n02 Jan 2024,3.61\n03 Jan 2024,3.70\n05 Jan 2024,\n"
    df = parse_boe_csv(text, "IUDMNPY")
    assert list(df["value"]) == [3.61, 3.70]
    assert df["available_at"].iloc[0] == pd.Timestamp("2024-01-03 12:00", tz="UTC")
    assert boe_stamp(pd.DatetimeIndex(["2024-01-05"]))[0] == pd.Timestamp("2024-01-08 12:00", tz="UTC")
    y = from_yahoo_frame(pd.DataFrame({"Close": [1.27, 1.28]}, index=pd.to_datetime(["2024-01-02", "2024-01-03"])))
    assert y["available_at"].iloc[0] == pd.Timestamp("2024-01-02 23:00", tz="UTC")


def test_sensitivities_recover_a_known_oil_beta():
    dates = pd.bdate_range("2020-01-01", periods=400)
    rng = np.random.default_rng(3)
    oil_r = rng.normal(0, 0.02, len(dates))
    oil = pd.Series(100 * np.exp(np.cumsum(oil_r)), index=dates)
    tables = {table_name("oil"): pd.DataFrame({"value": oil, "available_at": dates.tz_localize("UTC")})}
    for i in range(4):
        r = (1.5 if i == 0 else 0.0) * oil_r + rng.normal(0, 0.01, len(dates))
        c = 100 * np.exp(np.cumsum(r))
        tables[f"S{i}"] = pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1e7,
                                        "available_at": dates.tz_localize("UTC") + pd.Timedelta(hours=1)},
                                       index=dates)
    view = PointInTimeView(tables, dates[-1].tz_localize("UTC") + pd.Timedelta(hours=2))
    betas = crossasset.sensitivities(view, [f"S{i}" for i in range(4)])
    assert betas["S0"]["oil"] == pytest.approx(1.5 * 3 / 4, abs=0.35)  # market absorbs a share of the oil effect
    assert abs(betas["S1"]["oil"]) < 0.4


def test_sector_numbers():
    dates = pd.bdate_range("2020-01-01", periods=200)
    tables = {}
    for s, g in {"A": 0.002, "B": 0.0, "C": -0.001}.items():
        c = 100 * np.exp(g * np.arange(len(dates)))
        tables[s] = pd.DataFrame({"close": c, "available_at": dates.tz_localize("UTC")}, index=dates)
    view = PointInTimeView(tables, dates[-1].tz_localize("UTC"))
    sec = crossasset.sector_numbers(view, ["A", "B", "C"], {"A": "x", "B": "x", "C": "y"})
    assert sec["A"][0] > 0 > sec["B"][0]
    assert sec["C"] == (0.0, pytest.approx(sec["C"][1]))
    assert sec["A"][1] > 0 > sec["C"][1]


class Recorder(LLMClient):
    def __init__(self):
        super().__init__(model="recorder")
        self.prompts = []

    def _ask(self, system, prompt):
        self.prompts.append((system, prompt))
        return "{}"


@pytest.mark.parametrize("cls", [InformedPanel, InformedDesk])
def test_informed_briefs_stay_anonymous(cls):
    syms, t = uk_tables()
    client = Recorder()
    strat = cls(syms, client=client)
    view = PointInTimeView(PreparedTables(t), t[syms[0]]["available_at"].iloc[600])
    strat.decide(view)
    assert client.prompts
    text = "\n".join(p for _, p in client.prompts)
    assert "Cross-market" in text and "vs own sector 6-month %" in text
    assert not re.search(r"\b(19|20)\d\d\b", text)
    assert not any(s in text or s[:-2] in text for s in syms)
    assert not any(g in text for g in set(SECTORS.values()) - {"other"})


def test_rulebot_cross_market_voice_reads_its_columns(monkeypatch):
    syms, t = uk_tables()
    strat = InformedPanel(syms, sectors={s: ("a" if i < 3 else "b") for i, s in enumerate(syms)})
    view = PointInTimeView(PreparedTables(t), t[syms[0]]["available_at"].iloc[600])
    assert isinstance(strat.decide(view), dict)


@pytest.mark.parametrize("name", ["sector_mom", "macro_tilt", "panel_macro", "desk_macro"])
def test_truncation_invariance_with_context(name):
    """Same check as tests/test_firewall.py, with macro tables present."""
    syms, t = uk_tables(n=4)
    cut = t[syms[0]].index[500]
    truncated = {s: df[df.index <= cut] for s, df in t.items()}
    full = run_backtest(REGISTRY[name](syms), t)
    part = run_backtest(REGISTRY[name](syms), truncated)
    before = [d for d in full.decisions if d["date"] <= cut]
    assert before == part.decisions
    assert part.decisions or name in ("sector_mom",)
    pd.testing.assert_series_equal(full.equity[full.equity.index <= cut], part.equity)


def test_macro_tilt_scales_with_risk_signals():
    syms, t = uk_tables(n=4)
    res = run_backtest(REGISTRY["macro_tilt"](syms), t)
    assert res.decisions
    assert all(sum(d["proposed"].values()) <= 1 + 1e-9 for d in res.decisions)


def test_drivers_have_labels_without_dates():
    for d in DRIVERS.values():
        assert not re.search(r"\d{4}", d.label)
