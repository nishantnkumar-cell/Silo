"""Tests for the changes that came out of the November 2026 project review."""

import json
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import pytest

from silo.agents.contestants import DebateDesk, StylePanel
from silo.agents.controls import ShortlistRandom, ShortlistTop
from silo.agents.llm import RuleBot
from silo.agents.v2 import DeskV2, PanelV2, ranked
from silo.broker import CostModel
from silo.constraints import Constraints
from silo.data.macro import to_context
from silo.data.sources import synthetic_bars
from silo.data.vault import normalize_bars
from silo.engine import BANK_RATE, MarketData, run_backtest
from silo.firewall import PointInTimeView
from silo.lab.league_backtest import run_league_backtest, write
from silo.lab.signal import SIGNALS, event_study, pick_spread, rank_ic
from silo.lab.study import run_study
from silo.lab.trials import TrialLog
from silo.lab.twins import Twin, shape_of
from silo.learning.journal import build_journal
from silo.liquidity import Liquidity
from silo.markets import UK
from silo.strategies import REGISTRY
from silo.strategies.managed import Managed

POSITION = dict(costs=CostModel.uk(), constraints=Constraints.position_trading())


def uk(n=8, n_days=700, **kw):
    syms = [f"T{i}.L" for i in range(n)]
    return syms, {s: normalize_bars(synthetic_bars(n_days=n_days, seed=i, start_price=2500, **kw), UK, pence=True)
                  for i, s in enumerate(syms)}


@dataclass
class Flaky(RuleBot):
    """A model that answers normally, then talks nonsense for a while."""

    bad_from: int = 40
    bad_for: int = 12
    n: int = 0

    def _ask(self, system, prompt):
        self.n += 1
        if self.bad_from <= self.n < self.bad_from + self.bad_for:
            return "Sorry, I can't help with that."
        return super()._ask(system, prompt)


def invested(res):
    return pd.Series({d["date"]: sum(d["weights"].values()) for d in res.decisions})


@pytest.mark.parametrize("cls", [DebateDesk, StylePanel, DeskV2, PanelV2])
def test_unusable_model_reply_holds_the_book_instead_of_selling(cls, tmp_path):
    syms, t = uk(n=12)
    good = run_backtest(cls(syms, client=RuleBot(log_dir=tmp_path / "a")), t, **POSITION)
    strat = cls(syms, client=Flaky(log_dir=tmp_path / "b"))
    bad = run_backtest(strat, t, **POSITION)
    assert strat.bad_replies > 0
    assert len(bad.decisions) < len(good.decisions)  # skipped decisions leave no order at all
    held = invested(bad)
    base = invested(good)
    floor = base[base.gt(0).idxmax():].min()
    assert held[held.gt(0).idxmax():].min() > min(0.3, floor - 0.1)  # a bad answer never dumps the book to cash


def test_explicit_empty_pick_list_still_means_cash(tmp_path):
    class Nothing(RuleBot):
        def _ask(self, system, prompt):
            return json.dumps({"picks": [], "why": "nothing looks good"}) if "trader" in system.lower() \
                else super()._ask(system, prompt)

    syms, t = uk(n=6)
    res = run_backtest(DebateDesk(syms, client=Nothing(log_dir=tmp_path)), t, **POSITION)
    assert res.fills.empty


def test_sector_momentum_trades_before_a_late_listing_has_history():
    syms, t = uk(n=10)
    t = dict(t)
    t["T9.L"] = t["T9.L"].iloc[-60:]  # listed two months ago
    res = run_backtest(REGISTRY["sector_mom"](syms, sectors={s: f"g{i % 3}" for i, s in enumerate(syms)}), t,
                       **POSITION)
    assert not res.fills.empty
    assert res.fills["date"].min() < t["T9.L"].index[0]
    assert "T9.L" not in set(res.fills["symbol"])


def test_stale_prices_fail_the_liquidity_check():
    syms, t = uk(n=1)
    df = t[syms[0]].copy()
    stale = df.copy()
    stale.loc[stale.index[-40:], ["open", "high", "low", "close"]] = float(df["close"].iloc[-41])
    clock = df["available_at"].iloc[-1]
    assert Liquidity().check(PointInTimeView({syms[0]: df}, clock), syms[0])[0]
    ok, _, why = Liquidity().check(PointInTimeView({syms[0]: stale}, clock), syms[0])
    assert not ok and "stale" in why


def test_twin_copies_the_shape_of_the_book_and_nothing_else():
    syms, t = uk(n=20)
    real = run_backtest(REGISTRY["momentum"](syms, top_k=5), t, **POSITION)
    shape = shape_of(real.decisions)
    twin = run_backtest(Twin(syms, shape, seed=1, liquidity=Liquidity()), t, **POSITION)
    assert [d["date"] for d in twin.decisions] == [d["date"] for d in real.decisions]
    for a, b in zip(real.decisions[3:], twin.decisions[3:]):
        assert abs(len(a["weights"]) - len(b["weights"])) <= 2
    other = run_backtest(Twin(syms, shape, seed=2, liquidity=Liquidity()), t, **POSITION)
    assert twin.decisions != other.decisions  # different seeds pick different names
    # no future data: cutting the tables changes nothing before the cut
    cut = t[syms[0]].index[500]
    part = run_backtest(Twin(syms, shape, seed=1, liquidity=Liquidity()),
                        {s: df[df.index <= cut] for s, df in t.items()}, **POSITION)
    assert [d for d in twin.decisions if d["date"] <= cut] == part.decisions


def test_shortlist_controls_choose_only_from_the_shortlist():
    syms, t = uk(n=20)
    top = run_backtest(ShortlistTop(syms, shortlist=8, top_k=4), t, **POSITION)
    rnd = run_backtest(ShortlistRandom(syms, shortlist=8, top_k=4), t, **POSITION)
    assert not top.fills.empty and not rnd.fills.empty
    assert max(len(d["proposed"]) for d in top.decisions) <= 4
    assert max(len(d["proposed"]) for d in rnd.decisions) <= 4
    assert top.decisions != rnd.decisions


def planted(n=40, n_days=600):
    """Shares whose long-run drift differs, so past return really does predict future return."""
    syms = [f"P{i:02d}.L" for i in range(n)]
    drifts = np.linspace(-0.002, 0.002, n)
    return syms, {s: normalize_bars(synthetic_bars(n_days=n_days, seed=i, start_price=2500, drift=float(drifts[i]),
                                                   vol=0.008), UK, pence=True) for i, s in enumerate(syms)}


def test_rank_ic_finds_a_planted_signal_and_not_noise():
    syms, t = planted()
    real = rank_ic(SIGNALS["momentum_6_1"], t, syms, horizon=20)
    assert real["weeks"] > 30 and real["mean_ic"] > 0.2 and real["t"] > 3
    nsyms, nt = uk(n=40, n_days=600)
    noise = rank_ic(SIGNALS["momentum_6_1"], nt, nsyms, horizon=20)
    assert abs(noise["mean_ic"]) < 0.1


def test_pick_spread_and_event_study_reward_a_real_signal():
    syms, t = planted()
    res = run_backtest(REGISTRY["momentum"](syms, lookback=126, top_k=8), t, **POSITION)
    spread = pick_spread(res.decisions, t, syms, horizon=20)
    assert spread["decisions"] > 5 and spread["mean_spread"] > 0 and spread["hit_rate"] > 0.7
    trades = build_journal(res, t, syms)
    rows = event_study(trades, t, syms, horizons=(20,), by="ret_6m", cuts=[0.0])
    overall = next(r for r in rows if r["group"] == "all entries")
    assert overall["entries"] >= 8 and overall["mean_excess"] > 0


def test_ranked_brief_shows_percentiles_and_no_traded_value():
    syms, t = uk(n=10)
    desk = DeskV2(syms, client=RuleBot())
    view = PointInTimeView(t, t[syms[0]]["available_at"].iloc[-1])
    brief = desk.brief(view)
    text = brief.table()
    assert "traded" not in text and "percentile" in text and "+" not in text.split("\n", 2)[2]
    cells = [int(x) for line in text.splitlines()[2:] for x in line.split(" | ")[1:]]
    assert min(cells) == 0 and max(cells) == 100
    assert ranked(brief).rows is brief.rows  # sizing still uses the real numbers


def test_managed_book_sells_on_its_stop_and_sits_out():
    syms, t = uk(n=6, n_days=500)
    crash = t[syms[0]].copy()
    k = 400
    for c in ("open", "high", "low", "close"):
        crash.loc[crash.index[k:], c] = crash[c].iloc[k:] * 0.6  # a 40% gap down that never recovers
    t = {**t, syms[0]: crash}

    class Always(REGISTRY["buy_and_hold"]):
        name = "always"

        def decide(self, view):
            return {s: 0.1 for s in self.symbols}

    REGISTRY["always"] = Always
    try:
        cons = Constraints.position_trading()
        plain = run_backtest(Always(syms), t, costs=CostModel.uk(), constraints=cons)
        strat = Managed(syms, base="always")
        res = run_backtest(strat, t, costs=CostModel.uk(), constraints=cons)
    finally:
        del REGISTRY["always"]
    sells = res.fills[(res.fills["symbol"] == syms[0]) & (res.fills["qty"] < 0)]
    assert len(sells) and sells["date"].min() <= crash.index[k + 2]  # out within two days of the gap
    assert any(e["action"] == "exit" and e["symbol"] == syms[0] for e in strat.log)
    back = res.fills[(res.fills["symbol"] == syms[0]) & (res.fills["qty"] > 0) & (res.fills["date"] > crash.index[k])]
    assert back.empty or back["date"].min() > crash.index[k + 15]  # the cool-off holds
    assert plain.fills[(plain.fills["symbol"] == syms[0]) & (plain.fills["qty"] < 0)].empty


def test_managed_takes_half_off_at_twice_the_risk():
    syms, t = uk(n=4, n_days=500, drift=0.004, vol=0.006)  # steady risers
    strat = REGISTRY["momentum_managed"](syms, base_params={"top_k": 2, "lookback": 60})
    res = run_backtest(strat, t, **POSITION)
    assert any(e["action"] == "partial" for e in strat.log)
    assert any(d.get("exits") and any(abs(k - 0.5) < 1e-9 for k in d["exits"].values()) for d in res.decisions)


def test_momentum_keep_zone_trades_less():
    syms, t = uk(n=30, n_days=900)
    plain = run_backtest(REGISTRY["momentum"](syms, top_k=6), t, **POSITION)
    keep = run_backtest(REGISTRY["momentum_keep"](syms, top_k=6), t, **POSITION)
    assert keep.fees_paid < plain.fees_paid


def test_cash_interest_is_off_by_default_and_uses_bank_rate_when_on():
    syms, t = uk(n=3)
    idle = REGISTRY["random"](syms, decide_prob=0.0)  # never trades: the whole book is cash
    off = run_backtest(idle, t, costs=CostModel.uk())
    assert off.equity.iloc[-1] == off.equity.iloc[0] and off.interest == 0
    fixed = run_backtest(idle, t, costs=replace(CostModel.uk(), cash_interest=True, cash_rate=5.0))
    years = (fixed.equity.index[-1] - fixed.equity.index[0]).days / 365
    assert fixed.equity.iloc[-1] / fixed.equity.iloc[0] - 1 == pytest.approx(1.05 ** years - 1, rel=0.03)
    dates = t[syms[0]].index
    rate = pd.Series(2.0, index=dates)
    stamps = (dates + pd.Timedelta(hours=12)).tz_localize("UTC")
    with_rate = run_backtest(idle, {**t, BANK_RATE: to_context(rate, stamps)},
                             costs=replace(CostModel.uk(), cash_interest=True, cash_rate=5.0))
    assert 0 < with_rate.interest < fixed.interest  # the cached 2% series wins over the 5% fallback
    assert MarketData({**t, BANK_RATE: to_context(rate, stamps)}).cash_rate(stamps[0] - pd.Timedelta(days=1), 9.0) == 9.0


def test_trial_log_counts_earlier_attempts_once(tmp_path):
    log = TrialLog(tmp_path / "trials.jsonl")
    log.record("lab", "a", {}, 10, "2020-01-01", "2021-01-01", 0.05)
    log.record("lab", "a", {}, 10, "2020-01-01", "2021-01-01", 0.06)  # a rerun, not a new attempt
    log.record("league", "b", {"k": 1}, 10, "2020-06-01", "2022-01-01", -0.01)
    log.record("lab", "c", {}, 10, "2015-01-01", "2016-01-01", 0.2)  # other dates
    assert sorted(log.overlapping("2020-03-01", "2020-09-01")) == [-0.01, 0.06]
    assert len(log.path.read_text().splitlines()) == 4  # append-only


def test_lab_luck_test_gets_harder_with_earlier_attempts(tmp_path, tables):
    syms = list(tables)
    alone = run_study(REGISTRY["sma_cross"], syms, tables, n_random=3)
    log = TrialLog(tmp_path / "trials.jsonl")
    oos = alone.wf.oos_returns.index
    for i, sr in enumerate([0.15, -0.1, 0.2, 0.05, -0.2, 0.12]):
        log.record("league", f"x{i}", {}, len(syms), oos[0], oos[-1], sr)
    crowded = run_study(REGISTRY["sma_cross"], syms, tables, n_random=3, trial_log=log)
    assert crowded.dsr <= alone.dsr
    assert "earlier attempts" in next(c.detail for c in crowded.checks if "deflated" in c.name)
    assert any(e["strategy"] == "sma_cross" for e in log.entries())


def test_league_report_shows_twins_picks_and_tracker(tmp_path):
    syms, t = uk(n=24, n_days=700)
    t["ISF.L"] = normalize_bars(synthetic_bars(n_days=700, seed=99, start_price=800), UK, pence=True)
    entries, bench = run_league_backtest(["momentum", "shortlist_top", "random"], syms, t, twins=3,
                                         tracker="ISF.L", progress=None, **POSITION)
    by = {e.name: e for e in entries}
    assert by["momentum"].extra["twins"]["n"] == 3
    assert 0 <= by["momentum"].extra["twins"]["sharpe_percentile"] <= 1
    assert "twins" not in by["random"].extra
    assert by["momentum"].extra["picks"]["decisions"] > 0
    assert bench.tracker is not None and len(bench.tracker.equity) > 100
    page, data = write(entries, bench, tmp_path)
    text = page.read_text()
    assert "Beat twins" in text and "Picks vs the rest" in text and "tracker ISF.L" in text
    saved = json.loads(data.read_text())
    assert saved["tracker"]["name"] == "tracker ISF.L" and "twins" in saved["contestants"][0]
