"""Outside AI trading designs, re-built to run inside Silo's rules.

These follow the *shape* of two popular open-source projects, using only
data Silo can serve point-in-time (prices, volumes, dividends):

* :class:`DebateDesk` - after TradingAgents (TauricResearch): a bull and a
  bear researcher argue, a trader decides, a risk manager can veto. The
  original's news, sentiment and fundamentals analysts are left out until
  Silo has point-in-time sources for them; the brief plays the technical
  analyst.
* :class:`StylePanel` - after ai-hedge-fund (virattt): several investors
  with different styles score every stock, and a portfolio manager combines
  the scores. The original's famous-investor personas are replaced with
  plain styles (trend, contrarian, income, defensive) because value and
  quality styles need fundamentals we can't yet serve without look-ahead.

Both see a :class:`~silo.agents.brief.Brief` (tickers and dates hidden by
default), decide on rebalance days only, and go through the same liquidity
and position-trading rules as every other strategy. Every model call is
logged with the model's training cutoff.
"""

from __future__ import annotations

import json

import numpy as np

from silo.agents.brief import build_brief
from silo.agents.llm import LLMClient, client_from_spec, parse_json
from silo.strategies.base import Strategy

RULES = (
    "You manage a long-only UK share portfolio held for weeks to months. "
    "You only know the numbers given. Stocks are labelled with letters and the date is hidden "
    "on purpose; do not guess which companies or years they are. Reply with JSON only."
)


class LLMStrategy(Strategy):
    uses_llm = True
    param_grid: dict[str, list] = {}  # each extra trial costs model calls; keep the lab small
    # Lessons from past trades that passed the learning gate (see silo.learning). Empty by
    # default, and then prompts are byte-for-byte unchanged so cached answers stay valid.
    guidance: str = ""
    last_notes: dict | None = None
    bad_replies: int = 0  # decisions skipped because the model's answer could not be used

    def __init__(self, symbols, client: LLMClient | None = None, anonymize: bool = True,
                 shortlist: int = 30, top_k: int = 12):
        super().__init__(symbols)
        self.client, self.anonymize, self.shortlist, self.top_k = client, anonymize, shortlist, top_k

    @property
    def llm(self) -> LLMClient:
        if self.client is None:
            self.client = client_from_spec()
        return self.client

    @property
    def model_cutoff(self) -> str | None:
        return self.llm.cutoff

    def ask(self, role: str, system: str, prompt: str, clock) -> dict:
        system = f"{RULES}\n\n{system}"
        if self.guidance:
            system += f"\n\nLessons from your own past trades, each checked on later data before use:\n{self.guidance}"
        return parse_json(self.llm.complete(system, prompt, role=f"{self.name}:{role}", clock=clock))

    def brief(self, view):
        """The numbers this agent is shown. Subclasses may add columns."""
        return build_brief(view, self.symbols, self.anonymize, limit=self.shortlist)

    def weights_for(self, picks: list[str], brief) -> dict[str, float]:
        picks = list(dict.fromkeys(picks))[: self.top_k]
        if not picks:
            return {}
        inv = {s: 1 / max(brief.rows[s][4], 1.0) for s in picks}  # inverse volatility
        total = sum(inv.values())
        exposure = min(1.0, len(picks) / self.top_k)  # few convincing names: keep some cash
        return {s: exposure * v / total for s, v in inv.items()}


class DebateDesk(LLMStrategy):
    """Bull vs bear, then a trader, then a risk manager. Four calls a decision."""

    name = "desk"

    def decide(self, view):
        self.last_notes = None
        brief = self.brief(view)
        if not brief.rows:
            return {}
        table = f"{brief.market}\n\n{brief.table()}"
        clock = view.clock
        bull = self.ask("bull", "You are the bull researcher. Make the strongest case for the stocks most "
                        "likely to rise over the next 1 to 3 months.",
                        f"{table}\n\nReply as {{\"argument\": \"<3 sentences>\", \"names\": [up to 8 ids]}}", clock)
        bear = self.ask("bear", "You are the bear researcher. Make the strongest case against the stocks most "
                        "likely to fall or disappoint over the next 1 to 3 months.",
                        f"{table}\n\nReply as {{\"argument\": \"<3 sentences>\", \"names\": [up to 8 ids]}}", clock)
        debate = (f"Bull case: {bull.get('argument', '')} Favours: {bull.get('names', [])}\n"
                  f"Bear case: {bear.get('argument', '')} Warns about: {bear.get('names', [])}")
        trade = self.ask("trader", "You are the trader. Weigh both researchers and choose what to hold.",
                         f"{table}\n\n{debate}\n\nReply as {{\"picks\": [up to {self.top_k} ids], "
                         "\"why\": \"<2 sentences>\"}. Fewer picks is fine if few look good.", clock)
        raw = trade.get("picks")
        picks = brief.to_symbols(_ids(raw))
        if not isinstance(raw, list) or (raw and not picks):
            # Not an answer (no JSON, wrong shape, or names that aren't in the table): leave the
            # book as it is. Only an explicit empty list means "hold nothing".
            self.bad_replies += 1
            return None
        self.last_notes = {"bull": brief.to_symbols(_ids(bull.get("names"))),
                           "bear": brief.to_symbols(_ids(bear.get("names"))),
                           "why": str(trade.get("why", ""))[:400]}
        if not picks:
            return {}
        risk = self.ask("risk", "You are the risk manager. Remove any pick that is too risky to hold for "
                        "weeks: falling knives, extreme volatility, or a crowded single bet.",
                        f"{brief.market}\n\n{brief.table(picks)}\n\nReply as {{\"veto\": [ids to drop], "
                        "\"why\": \"<1 sentence>\"}", clock)
        vetoed = set(brief.to_symbols(risk.get("veto", [])))
        self.last_notes["veto"] = sorted(vetoed)
        return self.weights_for([s for s in picks if s not in vetoed], brief)


def _ids(x) -> list:
    return x if isinstance(x, list) else []


STYLES = {
    "trend": "You are a trend follower. You buy stocks in strong, steady uptrends and avoid downtrends.",
    "contrarian": "You are a contrarian. You buy sound stocks after sharp falls and avoid stocks that "
                  "have run up too far.",
    "income": "You are an income investor. You prefer steady dividend payers with moderate volatility.",
    "defensive": "You are a defensive, risk-first investor. You prefer low volatility and shallow drawdowns.",
}


class StylePanel(LLMStrategy):
    """Four investing styles score every stock; a portfolio manager combines them."""

    name = "panel"

    def __init__(self, symbols, client=None, anonymize=True, shortlist=30, top_k=12, styles=None):
        super().__init__(symbols, client, anonymize, shortlist, top_k)
        self.styles = list(styles or STYLES)

    def decide(self, view):
        brief = self.brief(view)
        if not brief.rows:
            return {}
        table = f"{brief.market}\n\n{brief.table()}"
        totals = {s: [] for s in brief.rows}
        for style in self.styles:
            reply = self.ask(style, STYLES[style],
                             f"{table}\n\nScore every id from -1 (avoid) to 1 (buy). Reply as "
                             "{\"scores\": {\"<id>\": <score>, ...}, \"why\": \"<1 sentence>\"}", view.clock)
            scores = reply.get("scores", {})
            if not isinstance(scores, dict):
                continue
            for label, sc in scores.items():
                sym = brief.to_symbols([label])
                if sym:
                    try:
                        totals[sym[0]].append(float(np.clip(float(sc), -1, 1)))
                    except (TypeError, ValueError):
                        pass
        mean = {s: float(np.mean(v)) for s, v in totals.items() if v}
        if not mean:  # no style gave a usable answer: leave the book as it is
            self.bad_replies += 1
            return None
        picks = [s for s in sorted(mean, key=lambda s: -mean[s]) if mean[s] > 0]
        return self.weights_for(picks, brief)


def describe(strategy: LLMStrategy) -> str:
    return json.dumps({"strategy": strategy.name, "model": strategy.llm.model, "cutoff": strategy.llm.cutoff,
                       "anonymize": strategy.anonymize})
