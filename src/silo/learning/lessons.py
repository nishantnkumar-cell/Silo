"""Lessons: small, checkable rules learned from past trades, and an agent that follows them.

A lesson is one condition on a number the agent could see when it bought,
plus what to do about it::

    {"feature": "ret_12m", "op": ">", "value": 60, "action": "avoid",
     "text": "Stocks that already rose over 60% in a year kept lagging the market."}

* ``avoid``: don't open a new position when the condition holds.
* ``only``: open new positions only when the condition holds.
* ``advice``: no condition; the text is only given to AI agents as guidance.

Lessons act on *new* entries only. A name the agent already holds is never
force-sold by a lesson, so following lessons does not add churn and fees
(the blend's whipsaw problem).

Features are the numbers in the AI brief (returns, volatility, distance
from the 200-day average, drawdown, yield, liquidity), market breadth, the
Core Strategy zone readings, and, for AI desks, whether the bull backed or
the bear warned about the name. No names or dates: a lesson can't say
"avoid Rolls-Royce" or "avoid 2022".
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np

from silo.agents.brief import COLUMNS, stock_numbers
from silo.firewall import PointInTimeView
from silo.strategies.base import Strategy

FEATURES = {
    "ret_1m": "1-month return %",
    "ret_3m": "3-month return %",
    "ret_6m": "6-month return %",
    "ret_12m": "12-month return %",
    "vol": "volatility %",
    "vs_200d": "% above the 200-day average",
    "drawdown_1y": "1-year drawdown %",
    "yield": "dividend yield %",
    "traded_m": "daily value traded (millions)",
    "breadth": "% of the universe above its 200-day average",
    "zone_score": "Core Strategy zone score /10",
    "in_zone": "in a demand zone now (1 yes)",
    "above_zone": "% above the demand zone",
    "weekly_trend": "weekly trend (+1 up, 0 sideways, -1 down)",
    "curve_pos": "position in the monthly range %",
    "reward_risk": "reward to risk",
    "bull_backed": "the bull researcher backed it (1 yes)",
    "bear_warned": "the bear researcher warned about it (1 yes)",
}
CORE_FEATURES = ("zone_score", "in_zone", "above_zone", "weekly_trend", "curve_pos", "reward_risk")
NOTE_FEATURES = ("bull_backed", "bear_warned")
ACTIONS = ("avoid", "only", "advice")


def breadth(view: PointInTimeView, symbols) -> float | None:
    flags = []
    for s in symbols:
        c = view.bars(s, lookback=200)["close"]
        if len(c) >= 200:
            flags.append(float(c.iloc[-1] > c.mean()))
    return 100 * float(np.mean(flags)) if flags else None


def features_at(view: PointInTimeView, symbol: str, *, core: bool = False, notes: dict | None = None,
                market: float | None = None) -> dict[str, float]:
    """Every feature Silo can compute for ``symbol`` at the view's clock (point-in-time)."""
    out: dict[str, float] = {}
    nums = stock_numbers(view, symbol)
    if nums is not None:
        out.update({k: round(float(v), 3) for k, v in zip(COLUMNS, nums)})
    if market is not None:
        out["breadth"] = round(market, 1)
    if core:
        from silo.agents.rulebook import core_numbers

        out.update({k: round(float(v), 3) for k, v in zip(CORE_FEATURES, core_numbers(view, symbol))})
    if notes and ("bull" in notes or "bear" in notes):
        out["bull_backed"] = float(symbol in notes.get("bull", []))
        out["bear_warned"] = float(symbol in notes.get("bear", []))
    return out


@dataclass(frozen=True)
class Lesson:
    feature: str = ""
    op: str = ">"
    value: float = 0.0
    action: str = "avoid"
    text: str = ""

    def __post_init__(self):
        object.__setattr__(self, "value", float(self.value))
        if self.action not in ACTIONS:
            raise ValueError(f"lesson action must be one of {ACTIONS}")
        if self.action != "advice":
            if self.feature not in FEATURES and self.feature != "random":
                raise ValueError(f"unknown lesson feature {self.feature!r}")
            if self.op not in ("<", ">"):
                raise ValueError("lesson op must be < or >")
            if not np.isfinite(self.value):
                raise ValueError("lesson value must be a number")

    @property
    def id(self) -> str:
        blob = json.dumps([self.feature, self.op, round(self.value, 6), self.action,
                           self.text if self.action == "advice" else ""])
        return hashlib.sha256(blob.encode()).hexdigest()[:10]

    @classmethod
    def from_dict(cls, d: dict) -> Lesson:
        return cls(str(d.get("feature", "")), str(d.get("op", ">")), float(d.get("value", 0.0)),
                   str(d.get("action", "avoid")), str(d.get("text", "")))

    def to_dict(self) -> dict:
        return {**asdict(self), "id": self.id}

    def condition(self) -> str:
        if self.action == "advice":
            return "advice only"
        word = "above" if self.op == ">" else "below"
        return f"{FEATURES.get(self.feature, self.feature)} is {word} {self.value:g}"

    def describe(self) -> str:
        if self.action == "advice":
            return self.text
        rule = {"avoid": "Don't buy when", "only": "Only buy when"}[self.action] + f" {self.condition()}."
        return f"{rule} {self.text}".strip()

    @property
    def needs_core(self) -> bool:
        return self.feature in CORE_FEATURES

    def holds(self, feats: dict[str, float]) -> bool | None:
        """Whether the condition is true; None when the feature is unknown here."""
        if self.feature not in feats:
            return None
        x = feats[self.feature]
        return x > self.value if self.op == ">" else x < self.value

    def blocks(self, feats: dict[str, float]) -> bool:
        """True when this lesson says not to open the position."""
        if self.action == "advice":
            return False
        h = self.holds(feats)
        if h is None:
            return False  # can't tell: let the agent decide
        return h if self.action == "avoid" else not h


def _random_block(symbol: str, clock, p: float, seed: int) -> bool:
    """A deterministic coin flip per (symbol, decision): the random-filter control."""
    h = hashlib.sha256(f"{seed}|{symbol}|{clock}".encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF < p


def as_lessons(items) -> list[Lesson]:
    return [x if isinstance(x, Lesson) else Lesson.from_dict(x) for x in (items or [])]


class Learned(Strategy):
    """Any agent in the registry, following lessons learned from its past trades.

    Without lessons it behaves exactly like ``base``. With lessons, each new
    entry the base agent wants is checked against them at the decision clock
    and skipped if a lesson blocks it; the skipped weight stays in cash.
    With ``guidance`` on and an AI base agent, the lessons' text is also
    added to the agent's prompts.

    ``random_skip`` (probability, seed) replaces the lessons with coin flips;
    the gate uses it as a control.
    """

    name = "learned"
    param_grid: dict[str, list] = {}

    def __init__(self, symbols, base: str = "uk_trend", base_params: dict | None = None, lessons=(),
                 guidance: bool = True, random_skip: tuple[float, int] | None = None):
        super().__init__(symbols)
        from silo.agents.contestants import LLMStrategy
        from silo.strategies import REGISTRY

        if base == self.name:
            raise ValueError("a learned agent needs a different base agent")
        self.base_name = base
        self.base = REGISTRY[base](symbols, **(base_params or {}))
        self.lessons = as_lessons(lessons)
        self.random_skip = tuple(random_skip) if random_skip else None
        self.uses_llm = self.base.uses_llm
        self.members = {base: self.base, **getattr(self.base, "members", {})}  # so the model gets attached
        self.guidance = bool(guidance)
        if self.guidance and isinstance(self.base, LLMStrategy):
            self.base.guidance = "\n".join(f"- {x.describe()}" for x in self.lessons)
        self._core = any(x.needs_core for x in self.lessons)
        self._market = any(x.feature == "breadth" for x in self.lessons)
        self.reset()

    @property
    def model_cutoff(self):
        return getattr(self.base, "model_cutoff", None)

    def reset(self):
        if hasattr(self, "base"):
            self.base.reset()
        self.held: set[str] = set()
        self.last_notes = None

    def _blocked(self, view, symbol, notes, market) -> list[str]:
        if self.random_skip:
            p, seed = self.random_skip
            return ["random"] if _random_block(symbol, view.clock, p, seed) else []
        rules = [x for x in self.lessons if x.action != "advice"]
        if not rules:
            return []
        feats = features_at(view, symbol, core=self._core, notes=notes, market=market)
        return [x.id for x in rules if x.blocks(feats)]

    def decide(self, view):
        weights = self.base.decide(view)
        notes = getattr(self.base, "last_notes", None)
        if weights is None:
            self.last_notes = notes
            return None
        market = breadth(view, self.symbols) if self._market else None
        keep, skipped, new = {}, {}, []
        for s, w in weights.items():
            if w > 0 and s not in self.held:
                new.append(s)
                why = self._blocked(view, s, notes, market)
                if why:
                    skipped[s] = why
                    continue
            keep[s] = w
        self.held = {s for s, w in keep.items() if w > 0}
        self.last_notes = dict(notes or {})
        if skipped or self.lessons or self.random_skip:
            self.last_notes["skipped"] = skipped
            self.last_notes["new_entries"] = sorted(new)
        return keep
