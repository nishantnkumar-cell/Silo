import json
import re
from datetime import datetime, timedelta, timezone

import pytest

from silo.agents.blend import Blend
from silo.agents.brief import build_brief
from silo.agents.contestants import DebateDesk, StylePanel
from silo.agents.llm import LLMClient, LLMError, OllamaClient, RuleBot, client_from_spec, parse_json
from silo.broker import CostModel
from silo.constraints import Constraints
from silo.data.sources import synthetic_bars
from silo.data.vault import DataVault, normalize_bars
from silo.engine import run_backtest
from silo.firewall import PointInTimeView, PreparedTables
from silo.lab.study import run_study
from silo.league import create_league, fetch_for_books, run_all
from silo.markets import UK


def uk(n=8, n_days=700):
    syms = [f"T{i}.L" for i in range(n)]
    return syms, {s: normalize_bars(synthetic_bars(n_days=n_days, seed=i, start_price=2500), UK, pence=True)
                  for i, s in enumerate(syms)}


class Scripted(LLMClient):
    """Returns canned replies by role keyword, and records prompts."""

    def __init__(self, replies):
        super().__init__(model="scripted")
        self.replies, self.prompts = replies, []

    def _ask(self, system, prompt):
        self.prompts.append((system, prompt))
        for key, reply in self.replies.items():
            if key in system.lower():
                return reply
        return "{}"


def _view(tables, i=400):
    first = next(iter(tables.values()))
    return PointInTimeView(PreparedTables(tables), first["available_at"].iloc[i])


def test_brief_hides_tickers_and_dates():
    syms, t = uk()
    view = _view(t)
    b = build_brief(view, syms)
    text = b.market + b.table()
    assert not any(s in text or s[:-2] in text for s in syms)
    assert not re.search(r"\b(19|20)\d\d\b", text)  # no years anywhere
    assert sorted(b.to_symbols(list(b.alias.values()))) == sorted(b.rows)
    assert build_brief(view, syms).alias == b.alias  # stable labels


def test_desk_follows_trader_and_risk_veto():
    syms, t = uk()
    view = _view(t)
    b = build_brief(view, syms)
    a, c, d = (b.alias[s] for s in list(b.rows)[:3])
    client = Scripted({
        "bull": json.dumps({"argument": "x", "names": [a]}),
        "bear": json.dumps({"argument": "y", "names": [d]}),
        "trader": json.dumps({"picks": [a, c, d, "ZZZ"]}),
        "risk manager": json.dumps({"veto": [d]}),
    })
    w = DebateDesk(syms, client=client, top_k=4).decide(view)
    held = {b.alias[s] for s in w}
    assert held == {a, c}
    assert sum(w.values()) == pytest.approx(0.5)  # 2 of 4 slots used: half in cash
    assert any("Bull case: x" in p for _, p in client.prompts)


def test_panel_combines_style_scores():
    syms, t = uk()
    view = _view(t)
    b = build_brief(view, syms)
    a, c = (b.alias[s] for s in list(b.rows)[:2])
    client = Scripted({k: json.dumps({"scores": {a: 1, c: -1}}) for k in ("trend", "contrarian", "income")}
                      | {"defensive": json.dumps({"scores": {a: 0.5, c: "bad"}})})
    w = StylePanel(syms, client=client).decide(view)
    assert {b.alias[s] for s in w} == {a}


def test_bad_model_reply_means_no_trades():
    syms, t = uk()
    assert DebateDesk(syms, client=Scripted({})).decide(_view(t)) == {}
    assert parse_json("Sure! {\"picks\": [\"A\"]} hope that helps") == {"picks": ["A"]}
    assert parse_json("no json") == {}


def test_calls_are_logged_and_cached(tmp_path):
    bot = RuleBot(log_dir=tmp_path)
    bot.complete("s", "p", role="r")
    bot.complete("s", "p", role="r")
    assert (bot.calls, bot.cache_hits) == (1, 1)
    again = RuleBot(log_dir=tmp_path)  # a new run replays from disk
    again.complete("s", "p")
    assert again.calls == 0
    lines = [json.loads(x) for x in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert lines[0]["prompt"] == "p" and lines[1]["cached"] and "prompt" not in lines[1]


def test_ollama_must_be_local_and_needs_a_model(monkeypatch):
    with pytest.raises(LLMError):
        OllamaClient(model="qwen2.5:7b", host="http://example.com:11434")
    assert OllamaClient(model="qwen2.5:7b").cutoff == "2024-09"
    assert OllamaClient(model="mystery").cutoff == "unknown"
    monkeypatch.delenv("SILO_LLM")
    with pytest.raises(LLMError, match="SILO_LLM"):
        client_from_spec()
    monkeypatch.setenv("OLLAMA_HOST", "0.0.0.0:11434")
    assert client_from_spec("ollama:qwen2.5:7b").host == "http://localhost:11434"


def test_blend_leans_on_the_member_that_works():
    syms, t = uk(n_days=900)
    blend = Blend(syms, members=["uk_trend", "random"], warmup=4)
    run_backtest(blend, t, costs=CostModel.uk(), constraints=Constraints.position_trading())
    assert set(blend.last_alpha) <= {"uk_trend", "random"}
    assert blend.n_obs > 50


def test_blend_cuts_exposure_when_breadth_collapses():
    syms, t = uk()
    falling = {s: df.assign(close=df["close"] * (1 - 0.6 * (df.index >= df.index[350]))) for s, df in t.items()}
    view = _view(falling, 400)
    b = Blend(syms, members=["buy_and_hold"])
    assert sum(b.decide(view).values()) == pytest.approx(0.5)


def test_ai_lab_flags_training_cutoff_contamination():
    syms, t = uk(n=5, n_days=900)
    res = run_study(DebateDesk, syms, t, n_random=2, n_folds=2, model_cutoff="2030-01",
                    costs=CostModel.uk(), constraints=Constraints.position_trading())
    check = [c for c in res.checks if "cutoff" in c.name][0]
    assert not check.passed and "contaminated" in check.detail


def test_league_runs_every_book_and_ranks_them(tmp_path):
    syms, full = uk(n=6, n_days=700)
    vault = DataVault(tmp_path / "data")
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for s, df in full.items():
        vault.put_bars(s, df.iloc[:600], fetched_at=t0)
    made = create_league(syms, vault, tmp_path / "paper", with_ai=True)
    assert len(made) == 7
    cfg = json.loads((tmp_path / "paper" / "league-desk" / "config.json").read_text())
    assert cfg["model"]["model"] == "rulebot"
    assert create_league(syms, vault, tmp_path / "paper", with_ai=True) == []  # never overwrites
    calls = iter(range(100))
    failed = fetch_for_books(vault, tmp_path / "paper",
                             lambda s: full[s].iloc[: 640 + 0 * next(calls)], log=lambda *_: None)
    assert failed == []
    board = run_all(vault, tmp_path / "paper")
    assert len(board) == 7 and all(r["days"] == 40 for r in board)
    assert board == sorted(board, key=lambda r: -r["excess"])
    assert (tmp_path / "paper" / "league-desk" / "llm" / "calls.jsonl").exists()
