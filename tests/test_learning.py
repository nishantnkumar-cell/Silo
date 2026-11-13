import json
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from silo.agents.contestants import DebateDesk
from silo.agents.llm import LLMClient, RuleBot
from silo.broker import CostModel
from silo.cli import main
from silo.constraints import Constraints
from silo.data.sources import synthetic_bars
from silo.data.vault import DataVault, normalize_bars
from silo.engine import run_backtest
from silo.firewall import PointInTimeView, PreparedTables
from silo.learning.gate import Ledger, run_gate
from silo.learning.journal import Trade, anonymised, build_journal, scrub
from silo.learning.lessons import Learned, Lesson
from silo.learning.review import ai_review, mine, review_prompt
from silo.learning.walkforward import learning_walk_forward
from silo.markets import UK
from silo.paper import PaperBook
from silo.strategies import REGISTRY
from silo.strategies.base import Strategy
from silo.strategies.classic import Momentum12_1

RULES = {"costs": CostModel.uk(), "constraints": Constraints.position_trading()}


def uk(n=8, n_days=900):
    syms = [f"T{i}.L" for i in range(n)]
    return syms, {s: normalize_bars(synthetic_bars(n_days=n_days, seed=i, start_price=2500), UK, pence=True)
                  for i, s in enumerate(syms)}


def test_journal_accounts_for_every_pound():
    """Profit over all trades (open ones marked at the last close) equals the change in equity."""
    syms, t = uk()
    res = run_backtest(Momentum12_1(syms, top_k=4), t, **RULES)
    trades = build_journal(res, t, syms)
    assert trades and any(not x.open for x in trades)
    pnl = sum(x.proceeds + x.dividends - x.fees - x.cost for x in trades)
    assert pnl == pytest.approx(res.equity.iloc[-1] - 100_000, rel=1e-6, abs=1.0)
    x = next(x for x in trades if not x.open)
    assert x.hold_days > 0 and x.opened <= x.closed and "ret_12m" in x.features and "breadth" in x.features
    assert x.excess == pytest.approx(x.ret - x.market_ret)


def test_entry_features_are_point_in_time():
    """Features at entry must not change when later data is deleted."""
    syms, t = uk()
    res = run_backtest(Momentum12_1(syms, top_k=4), t, **RULES)
    first = build_journal(res, t, syms)[0]
    cut = {s: df[df.index <= first.opened] for s, df in t.items()}
    view = PointInTimeView(PreparedTables(cut), first.decided_at)
    from silo.learning.lessons import breadth, features_at

    assert features_at(view, first.symbol, market=breadth(view, syms)) == first.features


def test_what_the_reviewer_sees_has_no_names_or_dates():
    syms, t = uk()
    res = run_backtest(Momentum12_1(syms, top_k=4), t, **RULES)
    trades = build_journal(res, t, syms)
    trades[0].why = "Bought T3.L in March 2021 because it looked strong"
    text = review_prompt(trades, mine(trades)[0]) + json.dumps(anonymised(trades))
    assert not any(s in text for s in syms)
    assert not re.search(r"\b(19|20)\d\d\b", text)
    assert "March" not in text
    assert scrub("BARC.L rallied in Jan 2020") == "[stock] rallied in [month] [year]"


def test_lessons_validate_and_describe():
    with pytest.raises(ValueError):
        Lesson("ticker", ">", 1)  # only known features: a lesson can't name a stock
    with pytest.raises(ValueError):
        Lesson("vol", "=", 1)
    a = Lesson("ret_12m", ">", 60, "avoid", "ran too far")
    assert a.blocks({"ret_12m": 70}) and not a.blocks({"ret_12m": 10}) and not a.blocks({})
    only = Lesson("weekly_trend", ">", 0, "only")
    assert only.blocks({"weekly_trend": -1}) and not only.blocks({"weekly_trend": 1})
    assert Lesson.from_dict(a.to_dict()) == a and a.id == Lesson.from_dict(a.to_dict()).id
    assert a.describe().startswith("Don't buy when 12-month return % is above 60")
    assert not Lesson(action="advice", text="be patient").blocks({"vol": 99})


def _trade(f, excess):
    ts = pd.Timestamp("2020-01-01")
    return Trade("x", "S", ts, ts, ts, False, 20, 100.0, 100.0, 0.0, 0.0, excess, 0.0, f)


def test_miner_finds_a_planted_pattern_and_counts_its_tries():
    rng = np.random.default_rng(1)
    trades = [_trade({"ret_12m": float(v), "vol": float(rng.uniform(10, 40))},
                     (-0.08 if v > 50 else 0.02) + rng.normal(0, 0.03))
              for v in rng.uniform(-20, 100, 120)]
    cands, tried = mine(trades)
    assert cands[0].lesson.feature == "ret_12m" and cands[0].lesson.op == ">" and cands[0].t < -2
    assert tried >= 8
    assert mine(trades[:10]) == ([], 0)  # too few trades: no lessons


class Scripted(LLMClient):
    def __init__(self, reply):
        super().__init__(model="scripted")
        self.reply, self.prompts = reply, []

    def _ask(self, system, prompt):
        self.prompts.append((system, prompt))
        return self.reply


def test_ai_review_keeps_only_well_formed_lessons():
    trades = [_trade({"vol": float(v)}, 0.0) for v in range(40)]
    client = Scripted(json.dumps({"lessons": [
        {"feature": "vol", "op": ">", "value": 30, "action": "avoid", "text": "too jumpy"},
        {"feature": "company_name", "op": ">", "value": 1},  # not a feature
        {"feature": "vol", "op": ">", "value": "lots"},  # not a number
        {"action": "advice", "text": "hold cash when nothing qualifies"},
        "nonsense",
        {"feature": "ret_1m", "op": "<", "value": -10, "action": "only"},
        {"feature": "ret_3m", "op": "<", "value": -10, "action": "avoid"},
    ]}))
    out = ai_review(trades, client, [])
    assert [x.action for x in out] == ["avoid", "advice", "only"]  # at most three, bad ones dropped
    assert "Reply with JSON only" in client.prompts[0][0]
    rb = ai_review([_trade({"vol": float(v)}, -0.1 if v > 20 else 0.05) for v in range(40)], RuleBot())
    assert rb and rb[0].feature == "vol"


class Stub(Strategy):
    """Wants AAA, then (with the bear warning about it) still AAA, then nothing, then AAA again."""

    name = "stub"
    plan = [({"AAA": 0.5}, []), ({"AAA": 0.5}, ["AAA"]), ({}, ["AAA"]), ({"AAA": 0.5}, ["AAA"])]

    def reset(self):
        self.i = 0

    def decide(self, view):
        w, bear = self.plan[min(self.i, len(self.plan) - 1)]
        self.i += 1
        self.last_notes = {"bull": [], "bear": bear}
        return dict(w)


def test_lessons_only_stop_new_entries(monkeypatch, tables):
    monkeypatch.setitem(REGISTRY, "stub", Stub)
    agent = Learned(list(tables), "stub", lessons=[Lesson("bear_warned", ">", 0.5, "avoid")])
    view = PointInTimeView(PreparedTables(tables), tables["AAA"]["available_at"].iloc[300])
    assert agent.decide(view) == {"AAA": 0.5}  # no warning: bought
    assert agent.decide(view) == {"AAA": 0.5}  # warned, but already held: never force-sold
    assert agent.decide(view) == {}
    assert agent.decide(view) == {}  # a new entry the bear warned about: skipped, weight kept in cash
    assert agent.last_notes["skipped"] == {"AAA": [agent.lessons[0].id]}


def test_learned_without_lessons_is_the_base_agent():
    syms, t = uk()
    a = run_backtest(REGISTRY["uk_trend"](syms), t, **RULES)
    b = run_backtest(Learned(syms, "uk_trend"), t, **RULES)
    pd.testing.assert_series_equal(a.equity, b.equity)
    blocked = run_backtest(Learned(syms, "uk_trend", lessons=[{"feature": "vol", "op": ">", "value": 0}]), t, **RULES)
    def buys(r):
        return int((r.fills["qty"] > 0).sum())

    assert buys(blocked) < buys(a) / 2  # only buys made before there was enough history to judge remain


@pytest.mark.parametrize("kw", [{"lessons": [Lesson("ret_6m", ">", 20), Lesson("breadth", "<", 40)]},
                                {"lessons": [Lesson("zone_score", "<", 5)], "base": "core_zones"},
                                {"random_skip": (0.5, 3)}])
def test_learned_truncation_invariance(tables, kw):
    symbols = list(tables)
    kw = {"base": "momentum", **kw}
    cut = tables["AAA"].index[500]
    truncated = {s: df[df.index <= cut] for s, df in tables.items()}
    full = run_backtest(Learned(symbols, **kw), tables)
    part = run_backtest(Learned(symbols, **kw), truncated)
    assert [d for d in full.decisions if d["date"] <= cut] == part.decisions
    pd.testing.assert_series_equal(full.equity[full.equity.index <= cut], part.equity)


def test_ai_agents_get_lessons_in_their_prompt_only_when_there_are_some():
    syms, t = uk(n=5)
    view = PointInTimeView(PreparedTables(t), t[syms[0]]["available_at"].iloc[400])
    plain, guided = Scripted("{}"), Scripted("{}")
    DebateDesk(syms, client=plain).decide(view)
    agent = Learned(syms, "desk", lessons=[Lesson("vol", ">", 60, text="too jumpy")])
    agent.base.client = guided
    agent.decide(view)
    assert all("Lessons from" not in s for s, _ in plain.prompts)  # cached answers stay valid
    assert all("volatility % is above 60. too jumpy" in s for s, _ in guided.prompts)
    assert [p for _, p in plain.prompts] == [p for _, p in guided.prompts]


def test_desk_records_its_reasons_in_the_journal():
    syms, t = uk(n=6)
    res = run_backtest(DebateDesk(syms, client=RuleBot()), t, **RULES)
    noted = [d for d in res.decisions if "notes" in d]
    assert noted and {"bull", "bear", "why"} <= set(noted[0]["notes"])
    trades = build_journal(res, t, syms)
    assert all("bear_warned" in x.features and x.why for x in trades)


def test_gate_tests_only_on_unseen_data_and_never_the_holdout(tmp_path):
    syms, t = uk(n_days=1100)
    d = sorted(t[syms[0]].index)
    lesson = Lesson("ret_3m", ">", 5)
    ledger = Ledger(tmp_path / "ledger.jsonl")
    kw = dict(base_params={"top_k": 4}, ledger=ledger, n_random=4, **RULES)
    with pytest.raises(ValueError, match="later data"):
        run_gate("momentum", [lesson], syms, t, learned_until=d[600], test_start=d[500], test_end=d[800], **kw)
    with pytest.raises(ValueError, match="holdout"):
        run_gate("momentum", [lesson], syms, t, learned_until=d[600], test_start=d[601], test_end=d[1000],
                 holdout_start=d[900], **kw)
    ledger.propose([lesson], "momentum", d[0].date(), d[600].date(), "statistics")
    g = run_gate("momentum", [lesson], syms, t, learned_until=d[600], test_start=d[601], test_end=d[899],
                 holdout_start=d[900], **kw)
    assert g.new_entries > 0 and g.n_trials == 1
    names = [c.name for c in g.checks]
    assert any("random" in n for n in names) and any("buy-and-hold" in n for n in names)
    again = run_gate("momentum", [lesson], syms, t, learned_until=d[600], test_start=d[650], test_end=d[899],
                     holdout_start=d[900], **kw)
    assert again.n_trials == 2  # every retry on the same data raises the bar
    status = ledger.lessons("momentum")[lesson.id]["status"]
    assert status == ("accepted" if again.passed else "rejected")
    assert len(ledger.events()) == 3  # append-only: one proposal, two gate runs


def test_gate_rejects_a_lesson_that_never_acts(tmp_path):
    syms, t = uk(n_days=1000)
    d = sorted(t[syms[0]].index)
    g = run_gate("momentum", [Lesson("vol", ">", 10_000)], syms, t, learned_until=d[500], test_start=d[501],
                 test_end=d[900], ledger=Ledger(tmp_path / "l.jsonl"), n_random=2, **RULES)
    assert not g.passed and g.fired == 0
    assert Ledger(tmp_path / "l.jsonl").accepted("momentum") == []


def test_learning_walk_forward_never_trades_what_it_learned_from():
    syms, t = uk(n_days=1200)
    wf = learning_walk_forward("momentum", syms, t, n_folds=3, base_params={"top_k": 4}, **RULES)
    assert len(wf.folds) == 3
    for f in wf.folds:
        assert f.train_end < f.test_start
    m = wf.metrics()
    assert {"learned_sharpe", "base_sharpe", "bench_sharpe", "deflated"} <= set(m)
    assert len(wf.learned_returns) == len(wf.base_returns) == len(wf.bench_returns)


def test_learn_cli_review_gate_list_params(tmp_path, capsys):
    out = str(tmp_path / "learning")
    assert main(["learn", "review", "--demo", "--strategy", "momentum", "--params", '{"top_k": 4}', "--out", out]) == 0
    text = capsys.readouterr().out
    assert "journal:" in text and (tmp_path / "learning" / "journal-momentum" / "trades.csv").exists()
    ids = [e["id"] for e in Ledger(tmp_path / "learning" / "ledger.jsonl").events()]
    if ids:
        assert main(["learn", "gate", *ids, "--demo", "--strategy", "momentum", "--params", '{"top_k": 4}',
                     "--random", "2", "--out", out]) == 0
        assert "buy-and-hold" in capsys.readouterr().out
    assert main(["learn", "list", "--out", out]) == 0
    assert main(["learn", "params", "--strategy", "momentum", "--out", out]) == 0
    params = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert params["base"] == "momentum" and isinstance(params["lessons"], list)


def test_paper_book_can_run_a_learned_agent(tmp_path):
    syms, t = uk(n=5, n_days=700)
    vault = DataVault(tmp_path / "data")
    when = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for s, df in t.items():
        vault.put_bars(s, df.iloc[:600], fetched_at=when)
    book = PaperBook("learn", tmp_path / "paper")
    book.create("learned", syms, vault, {"base": "uk_trend", "lessons": [Lesson("ret_1m", ">", 15).to_dict()]})
    for s, df in t.items():
        vault.put_bars(s, df, fetched_at=when.replace(day=2))
    assert len(book.run(vault)) == 100
