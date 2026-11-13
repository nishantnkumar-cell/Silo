"""Turn a trade journal into proposed lessons.

Two reviewers, both working only from closed trades:

* :func:`mine` - a statistical miner. For every feature it tries a few cut
  points (the journal's own quintiles) and asks whether trades on one side
  did clearly worse than the market than the rest (Welch t-test on excess
  return). It reports how many splits it tried, because trying many and
  keeping the best is exactly how overfitting happens; the gate accounts for
  that by testing on other data.
* :func:`ai_review` - an AI reviewer. It is shown an anonymised summary
  (feature quintiles, a sample of trades, the miner's candidates) with no
  names and no dates, and must answer with at most three lessons in Silo's
  fixed rule format, or plain advice for AI agents. Anything outside the
  format is dropped.

Neither reviewer can put a lesson live. Proposals go to the ledger and only
the gate (:mod:`silo.learning.gate`) can accept them.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np

from silo.learning.journal import Trade, anonymised, summary
from silo.learning.lessons import FEATURES, Lesson

REVIEWER = (
    "You review the closed trades of a long-only share portfolio held for weeks to months. "
    "Stocks and dates are hidden on purpose; do not guess them. Your job is to find what kept "
    "working and what kept failing, and turn it into at most three short lessons. Prefer no lesson "
    "to a weak one: every lesson will be tested on later data and rejected if it does not hold. "
    "Reply with JSON only."
)


@dataclass
class Candidate:
    lesson: Lesson
    n_side: int
    n_rest: int
    side_excess: float
    rest_excess: float
    side_win_rate: float
    t: float

    def to_dict(self) -> dict:
        return {"lesson": self.lesson.to_dict(), "n_side": self.n_side, "n_rest": self.n_rest,
                "side_excess": self.side_excess, "rest_excess": self.rest_excess,
                "side_win_rate": self.side_win_rate, "t": self.t}


def _welch(a: np.ndarray, b: np.ndarray) -> float:
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    if va + vb <= 0:
        return 0.0
    return float((a.mean() - b.mean()) / math.sqrt(va + vb))


def mine(trades: list[Trade], min_n: int = 15, min_t: float = 2.0, top: int = 3) -> tuple[list[Candidate], int]:
    """Entry conditions whose trades lagged the market. Returns (candidates, splits tried)."""
    closed = [t for t in trades if not t.open]
    if len(closed) < 2 * min_n:
        return [], 0
    ex = np.array([t.excess for t in closed])
    names = sorted({k for t in closed for k in t.features if k in FEATURES})
    found, tried = [], 0
    for f in names:
        x = np.array([t.features.get(f, np.nan) for t in closed], dtype=float)
        ok = ~np.isnan(x)
        if ok.sum() < 2 * min_n:
            continue
        cuts = sorted(set(np.round(np.quantile(x[ok], [0.2, 0.4, 0.6, 0.8]), 2)))
        for c in cuts:
            for op in (">", "<"):
                side = ok & ((x > c) if op == ">" else (x < c))
                rest = ok & ~side
                if side.sum() < min_n or rest.sum() < min_n:
                    continue
                tried += 1
                t = _welch(ex[side], ex[rest])
                if t <= -min_t:
                    found.append(Candidate(
                        Lesson(f, op, float(c), "avoid",
                               f"In past trades these did {100 * ex[side].mean():+.1f}% vs the market per trade, "
                               f"against {100 * ex[rest].mean():+.1f}% for the rest ({int(side.sum())} trades)."),
                        int(side.sum()), int(rest.sum()), float(ex[side].mean()), float(ex[rest].mean()),
                        float(np.mean(ex[side] > 0)), t))
    best: dict[str, Candidate] = {}
    for c in sorted(found, key=lambda c: c.t):
        best.setdefault(c.lesson.feature, c)  # one cut per feature: the strongest
    return sorted(best.values(), key=lambda c: c.t)[:top], tried


def quintile_table(trades: list[Trade]) -> str:
    """For each feature: trades by quintile of its value, with average result vs the market."""
    closed = [t for t in trades if not t.open]
    lines = ["feature | quintile range | trades | avg vs market % | win rate vs market %"]
    for f in sorted({k for t in closed for k in t.features if k in FEATURES}):
        pairs = [(t.features[f], t.excess) for t in closed if f in t.features]
        if len(pairs) < 10:
            continue
        pairs.sort()
        for chunk in np.array_split(np.array(pairs), 5):
            if len(chunk):
                lines.append(f"{f} | {chunk[0, 0]:.1f} to {chunk[-1, 0]:.1f} | {len(chunk)} | "
                             f"{100 * chunk[:, 1].mean():+.1f} | {100 * np.mean(chunk[:, 1] > 0):.0f}")
    return "\n".join(lines)


def review_prompt(trades: list[Trade], candidates: list[Candidate], sample: int = 25) -> str:
    s = summary(trades)
    feats = "\n".join(f"- {k}: {v}" for k, v in FEATURES.items())
    rows = anonymised(trades)[:sample]
    cand = "\n".join("CANDIDATE " + json.dumps({k: v for k, v in c.lesson.to_dict().items() if k != "id"})
                     for c in candidates) or "(the statistics found no clear candidate)"
    return (
        f"Overall: {s.get('closed', 0)} closed trades, won against the market {100 * s.get('win_rate_vs_market', 0):.0f}% "
        f"of the time, average {100 * s.get('avg_excess', 0):+.1f}% vs the market per trade, held "
        f"{s.get('avg_hold_days', 0):.0f} trading days on average.\n\n"
        f"Features (measured on the day each trade was decided):\n{feats}\n\n"
        f"Results by feature quintile:\n{quintile_table(trades)}\n\n"
        f"A sample of trades (random order):\n" + "\n".join(json.dumps(r) for r in rows) + "\n\n"
        f"Statistical candidates (each lags the market clearly in these trades):\n{cand}\n\n"
        "Reply as {\"lessons\": [{\"feature\": \"<feature name>\", \"op\": \">\" or \"<\", \"value\": <number>, "
        "\"action\": \"avoid\" or \"only\", \"text\": \"<1 sentence why>\"}]}. "
        "You may instead give {\"action\": \"advice\", \"text\": \"<1 sentence>\"} for a lesson that is not one "
        "number. At most three lessons; an empty list is a fine answer."
    )


def ai_review(trades: list[Trade], client, candidates: list[Candidate] | None = None, max_lessons: int = 3) -> list[Lesson]:
    """Ask a model for lessons; keep only well-formed ones in Silo's rule format."""
    from silo.agents.llm import parse_json

    if candidates is None:
        candidates, _ = mine(trades)
    reply = parse_json(client.complete(REVIEWER, review_prompt(trades, candidates), role="reviewer"))
    out = []
    items = reply.get("lessons", [])
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        try:
            lesson = Lesson.from_dict(item)
        except (TypeError, ValueError):
            continue
        if lesson.action == "advice" and not lesson.text.strip():
            continue
        out.append(lesson)
        if len(out) >= max_lessons:
            break
    return out
