"""The gate: a lesson goes live only if it holds on data it was not drawn from.

Learning from your own trades is the fastest way to fool yourself. Look at
enough past trades and some pattern will always seem to explain the losers.
So a proposed lesson has to pass, on a later test window that the lesson's
trades never touched (and never the locked holdout):

1. **It acts.** The lesson must skip at least ``min_fired`` entries in the
   test window, or there is no evidence either way.
2. **It helps.** The agent with the lesson has a higher Sharpe ratio than
   the same agent without it, after costs.
3. **It isn't luck.** It beats 95% of *random* filters that skip the same
   share of entries. Skipping trades changes exposure and costs on its own;
   the lesson has to do better than a coin flip that skips as much.
4. **It survives repeated tries.** The daily difference in returns must pass
   a deflated Sharpe test that counts every gate run on an overlapping
   window for this agent (from the ledger), so retrying lessons until one
   passes doesn't work.
5. For AI agents, the test window starts after the model's training cutoff.

Buy-and-hold is reported next to every result. Every proposal and gate run
is appended to ``ledger.jsonl``, which is never rewritten.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from silo.engine import run_backtest
from silo.lab.stats import deflated_sharpe, sharpe
from silo.learning.lessons import Learned, Lesson, as_lessons
from silo.strategies import REGISTRY
from silo.strategies.classic import BuyAndHold

ANNUAL = np.sqrt(252)
DISCLAIMER = "Research only. Simulated results, not financial advice. Silo never trades real money."


class Ledger:
    """Append-only record of proposed lessons and gate results."""

    def __init__(self, path: str | Path = "learning/ledger.jsonl") -> None:
        self.path = Path(path)

    def _append(self, entry: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        entry = {"at": datetime.now(timezone.utc).isoformat(), **entry}
        with self.path.open("a") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def events(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    def propose(self, lessons, base: str, learn_start, learn_end, source: str, evidence: dict | None = None) -> list[str]:
        ids = []
        for x in as_lessons(lessons):
            self._append({"kind": "proposed", "id": x.id, "lesson": x.to_dict(), "base": base,
                          "learn_start": str(learn_start), "learn_end": str(learn_end), "source": source,
                          "evidence": (evidence or {}).get(x.id)})
            ids.append(x.id)
        return ids

    def record_gate(self, result: GateResult) -> None:
        self._append({"kind": "gated", **result.to_dict()})

    def lessons(self, base: str | None = None) -> dict[str, dict]:
        """id -> {lesson, base, learn_end, status, gates}. Status is the latest gate's verdict."""
        out: dict[str, dict] = {}
        for e in self.events():
            if e["kind"] == "proposed" and (base is None or e["base"] == base):
                rec = out.setdefault(e["id"], {"lesson": e["lesson"], "base": e["base"], "gates": [],
                                               "status": "proposed", "learn_end": e["learn_end"],
                                               "source": e["source"]})
                rec["learn_end"] = max(rec["learn_end"], e["learn_end"])
            elif e["kind"] == "gated":
                for i in e["lesson_ids"]:
                    if i in out and out[i]["base"] == e["base"]:
                        out[i]["gates"].append(e)
                        out[i]["status"] = "accepted" if e["passed"] else "rejected"
        return out

    def accepted(self, base: str) -> list[Lesson]:
        return [Lesson.from_dict(r["lesson"]) for r in self.lessons(base).values() if r["status"] == "accepted"]

    def prior_trials(self, base: str, start, end) -> list[float]:
        """Difference Sharpes of earlier gate runs for ``base`` whose test window overlaps this one."""
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        return [g["diff_sharpe"] for g in self.events()
                if g["kind"] == "gated" and g["base"] == base
                and pd.Timestamp(g["test_start"]) <= e and pd.Timestamp(g["test_end"]) >= s]


@dataclass
class Check:
    name: str
    passed: bool | None  # None: not applicable
    detail: str


@dataclass
class GateResult:
    base: str
    lesson_ids: list[str]
    lessons: list[dict]
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    base_sharpe: float
    variant_sharpe: float
    bench_sharpe: float
    base_return: float
    variant_return: float
    bench_return: float
    fired: int
    new_entries: int
    random_percentile: float
    diff_sharpe: float
    deflated: float
    n_trials: int
    checks: list[Check] = field(default_factory=list)
    variant_equity: pd.Series | None = None
    base_equity: pd.Series | None = None
    bench_equity: pd.Series | None = None

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks if c.passed is not None)

    def to_dict(self) -> dict:
        return {
            "base": self.base, "lesson_ids": self.lesson_ids, "lessons": self.lessons,
            "test_start": str(pd.Timestamp(self.test_start).date()), "test_end": str(pd.Timestamp(self.test_end).date()),
            "base_sharpe": self.base_sharpe, "variant_sharpe": self.variant_sharpe, "bench_sharpe": self.bench_sharpe,
            "base_return": self.base_return, "variant_return": self.variant_return, "bench_return": self.bench_return,
            "fired": self.fired, "new_entries": self.new_entries, "random_percentile": self.random_percentile,
            "diff_sharpe": self.diff_sharpe, "deflated": self.deflated, "n_trials": self.n_trials,
            "checks": [c.__dict__ for c in self.checks], "passed": self.passed,
        }


def skipped_counts(decisions: list[dict]) -> tuple[int, int]:
    """(entries skipped by lessons, new entries considered) across a run's decisions."""
    fired = sum(len((d.get("notes") or {}).get("skipped", {})) for d in decisions)
    new = sum(len((d.get("notes") or {}).get("new_entries", [])) for d in decisions)
    return fired, new


def _ret(eq: pd.Series) -> float:
    return float(eq.iloc[-1] / eq.iloc[0] - 1) if len(eq) > 1 else 0.0


def run_gate(base: str, lessons, symbols: list[str], tables: dict[str, pd.DataFrame], *, learned_until,
             test_start, test_end, holdout_start=None, base_params: dict | None = None, n_random: int = 20,
             min_fired: int = 10, guidance: bool = True, ledger: Ledger | None = None,
             costs=None, constraints=None) -> GateResult:
    """Test ``lessons`` on ``base`` over [test_start, test_end]; record the verdict in ``ledger``."""
    lessons = as_lessons(lessons)
    if not lessons:
        raise ValueError("no lessons to test")
    t0, t1 = pd.Timestamp(test_start), pd.Timestamp(test_end)
    if t0 <= pd.Timestamp(learned_until):
        raise ValueError(f"test window starts {t0.date()}, but the lessons were drawn from trades up to "
                         f"{pd.Timestamp(learned_until).date()}; test on later data only")
    if holdout_start is not None and t1 >= pd.Timestamp(holdout_start):
        raise ValueError(f"test window ends {t1.date()}, inside the locked holdout (from "
                         f"{pd.Timestamp(holdout_start).date()}); the gate never uses the holdout")
    kw = {"costs": costs, "constraints": constraints, "start": t0, "end": t1}
    params = base_params or {}
    base_run = run_backtest(REGISTRY[base](symbols, **params), tables, **kw)
    variant = Learned(symbols, base, params, lessons, guidance=guidance)
    var_run = run_backtest(variant, tables, **kw)
    bench = run_backtest(BuyAndHold(symbols), tables, **kw)
    fired, new = skipped_counts(var_run.decisions)
    rules = [x for x in lessons if x.action != "advice"]

    r_base = base_run.equity.pct_change().fillna(0.0)
    r_var = var_run.equity.pct_change().fillna(0.0)
    diff = (r_var - r_base).to_numpy()
    sr_b, sr_v = sharpe(r_base), sharpe(r_var)
    d_sr = sharpe(diff)
    ledger = ledger or Ledger()
    trials = ledger.prior_trials(base, t0, t1) + [d_sr]
    dsr = deflated_sharpe(diff, trials) if np.any(diff) else 0.0

    pct = float("nan")
    if rules and new and fired:
        p = fired / new
        rand = [sharpe(run_backtest(Learned(symbols, base, params, random_skip=(p, seed)), tables, **kw)
                       .equity.pct_change()) for seed in range(n_random)]
        pct = float(np.mean([sr_v > x for x in rand]))

    checks = []
    if rules:
        checks.append(Check("The lesson acts on the test data", fired >= min_fired,
                            f"skipped {fired} of {new} new entries (needs {min_fired})"))
        checks.append(Check("Beats random filters that skip as many entries", (pct >= 0.95) if fired else False,
                            f"beats {pct:.0%} of {n_random} random filters (needs 95%)" if fired else "nothing skipped"))
    checks.append(Check("Beats the unchanged agent on data the lesson wasn't drawn from", sr_v > sr_b,
                        f"Sharpe {sr_v * ANNUAL:.2f} with the lesson vs {sr_b * ANNUAL:.2f} without"))
    checks.append(Check("Survives the deflated Sharpe test", dsr >= 0.95,
                        f"{dsr:.2f} on the daily difference, counting {len(trials)} gate run(s) on this window "
                        "(needs 0.95)"))
    strat = REGISTRY[base](symbols, **params)
    if strat.uses_llm:
        cutoff = getattr(variant, "model_cutoff", None)
        clean = cutoff not in (None, "unknown") and t0 > pd.Timestamp(cutoff) + pd.offsets.MonthEnd(0)
        checks.append(Check("Tested only after the model's training cutoff",
                            bool(clean) if cutoff is not None else None,
                            f"model cutoff {cutoff}; test starts {t0.date()}" if cutoff else "offline stand-in"))
    checks.append(Check("Reported next to buy-and-hold", None,
                        f"buy-and-hold Sharpe {sharpe(bench.equity.pct_change()) * ANNUAL:.2f}, "
                        f"return {_ret(bench.equity):.1%}"))
    result = GateResult(
        base, [x.id for x in lessons], [x.to_dict() for x in lessons], t0, t1, sr_b, sr_v,
        sharpe(bench.equity.pct_change()), _ret(base_run.equity), _ret(var_run.equity), _ret(bench.equity),
        fired, new, pct, d_sr, float(dsr), len(trials), checks, var_run.equity, base_run.equity, bench.equity,
    )
    ledger.record_gate(result)
    return result
