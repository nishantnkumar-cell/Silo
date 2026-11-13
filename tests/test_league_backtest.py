import json

import pandas as pd

from silo.broker import CostModel
from silo.constraints import Constraints
from silo.data.sources import synthetic_bars
from silo.data.vault import normalize_bars
from silo.lab.league_backtest import data_check, run_league_backtest, write
from silo.markets import UK


def uk(n=8, n_days=900):
    syms = [f"T{i}.L" for i in range(n)]
    return syms, {s: normalize_bars(synthetic_bars(n_days=n_days, seed=i, start_price=2500), UK, pence=True)
                  for i, s in enumerate(syms)}


def test_league_backtest_compares_everyone_and_reuses_model_answers(tmp_path, monkeypatch):
    monkeypatch.setenv("SILO_LLM_LOG", str(tmp_path / "llm"))
    syms, t = uk()
    names = ["uk_trend", "blend", "random", "desk", "panel", "silo_agent"]
    entries, bench = run_league_backtest(names, syms, t, costs=CostModel.uk(),
                                         constraints=Constraints.position_trading(), progress=None)
    assert [e.name for e in entries] == names
    assert all(len(e.equity) == len(bench.equity) for e in entries)
    by = {e.name: e for e in entries}
    assert by["desk"].llm_calls > 0 and by["desk"].status == "rulebot"
    assert by["silo_agent"].llm_calls == 0 and by["silo_agent"].cache_hits > 0  # replayed desk and panel
    assert by["uk_trend"].status == "rules"
    assert any(r["state"].endswith("market") for r in by["uk_trend"].regimes)
    page, data = write(entries, bench, tmp_path / "out")
    text = page.read_text()
    assert "Buy-and-hold" in text and "Research only" in text and "silo_agent" in text
    assert json.loads(data.read_text())["contestants"][0]["name"] == "uk_trend"


def test_data_check_flags_flat_and_stale_series():
    syms, t = uk(n=3, n_days=400)
    flat = t["T1.L"].copy()
    flat.iloc[-60:, flat.columns.get_loc("close")] = flat["close"].iloc[-61]
    t["T1.L"] = flat
    t["T2.L"] = t["T2.L"].iloc[:-30]
    out = data_check(t)
    assert out.loc["T1.L", "suspect"] and out.loc["T1.L", "longest_flat_run"] >= 59
    assert out.loc["T2.L", "suspect"] and out.loc["T2.L", "days_behind"] >= 29
    assert not out.loc["T0.L", "suspect"]
