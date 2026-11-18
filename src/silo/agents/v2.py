"""Second versions of the AI desk and panel, built from what the first versions' logs showed.

The first live answers from the local model (November 2026) had four faults:
it picked shares for being heavily traded (a column that only repeated the
liquidity filter), it misread raw decimals (volatility printed as "+51.2"),
it named three shares when asked for up to twelve (leaving the book mostly in
cash), and the panel's scores were all-or-nothing. So here:

* the table shows each number as a percentile within the list (0 lowest,
  100 highest), which small models compare far more reliably than decimals;
* the traded-value column is gone: every share shown already passes the
  liquidity rules;
* the trader must rank exactly ``top_k`` names; how much to invest is not
  the model's call;
* panel members name their best and worst few instead of scoring all thirty.

``desk`` and ``panel`` are unchanged, so their cached answers stay valid and
the two versions can be compared like for like.
"""

from __future__ import annotations

import numpy as np

from silo.agents.brief import Brief
from silo.agents.contestants import STYLES, DebateDesk, StylePanel, _ids

RANK_HEADER = ("id | 1-month return | 3-month return | 6-month return | 12-month return | volatility | "
               "distance above 200-day average | drawdown from 1-year high (100 = smallest fall) | dividend yield")
RANK_NOTE = ("Every number is a percentile within this list: 0 is the lowest value in the list, 100 the highest. "
             "All of these shares are liquid enough to trade.")


class RankedBrief(Brief):
    """The same rows as a :class:`Brief`, shown as percentiles within the list."""

    @property
    def header(self) -> str:
        return RANK_HEADER + "".join(f" | {h}" for h in self.extra)

    def _percentiles(self) -> dict[str, list[int]]:
        names = list(self.rows)
        raw = np.array([self.rows[s][:8] for s in names], dtype=float)
        if len(names) == 1:
            return {names[0]: [50] * 8}
        order = raw.argsort(axis=0).argsort(axis=0)
        pct = np.rint(100 * order / (len(names) - 1)).astype(int)
        return {s: list(pct[i]) for i, s in enumerate(names)}

    def table(self, symbols: list[str] | None = None) -> str:
        pct = self._percentiles()  # always ranked within the whole list, also when only some rows are shown
        lines = [RANK_NOTE, self.header]
        for s in symbols or list(self.rows):
            extra = "".join(f" | {x:+.1f}" for x in self.rows[s][9:])
            lines.append(f"{self.alias[s]} | " + " | ".join(str(x) for x in pct[s]) + extra)
        return "\n".join(lines)


def ranked(brief: Brief) -> RankedBrief:
    return RankedBrief(brief.rows, brief.alias, brief.market, brief.extra)


class DeskV2(DebateDesk):
    """Bull, bear, trader, risk manager, with the repaired brief and a fixed number of picks."""

    name = "desk_v2"

    def brief(self, view):
        return ranked(super().brief(view))

    def decide(self, view):
        self.last_notes = None
        brief = self.brief(view)
        if not brief.rows:
            return {}
        k = min(self.top_k, len(brief.rows))
        table = f"{brief.market}\n\n{brief.table()}"
        clock = view.clock
        bull = self.ask("bull", "You are the bull researcher. Name the six stocks most likely to rise over the next "
                        "1 to 3 months and say which numbers make you think so.",
                        f"{table}\n\nReply as {{\"argument\": \"<3 sentences>\", \"names\": [exactly 6 ids]}}", clock)
        bear = self.ask("bear", "You are the bear researcher. Name the six stocks most likely to fall or disappoint "
                        "over the next 1 to 3 months and say which numbers make you think so.",
                        f"{table}\n\nReply as {{\"argument\": \"<3 sentences>\", \"names\": [exactly 6 ids]}}", clock)
        debate = (f"Bull case: {bull.get('argument', '')} Favours: {bull.get('names', [])}\n"
                  f"Bear case: {bear.get('argument', '')} Warns about: {bear.get('names', [])}")
        trade = self.ask("trader", "You are the trader. The book always holds a fixed number of stocks; your job is "
                         "only to choose which. Weigh both researchers, but check their claims against the table.",
                         f"{table}\n\n{debate}\n\nReply as {{\"picks\": [exactly {k} different ids, best first], "
                         "\"why\": \"<2 sentences>\"}", clock)
        raw = trade.get("picks")
        picks = brief.to_symbols(_ids(raw))
        if not isinstance(raw, list) or not picks:  # not an answer: leave the book as it is
            self.bad_replies += 1
            return None
        self.last_notes = {"bull": brief.to_symbols(_ids(bull.get("names"))),
                           "bear": brief.to_symbols(_ids(bear.get("names"))),
                           "why": str(trade.get("why", ""))[:400]}
        risk = self.ask("risk", "You are the risk manager. You may remove at most three picks that are too risky "
                        "to hold for weeks: the most volatile, or those that have fallen furthest from their high.",
                        f"{brief.market}\n\n{brief.table(picks)}\n\nReply as {{\"veto\": [up to 3 ids], "
                        "\"why\": \"<1 sentence>\"}", clock)
        vetoed = set(brief.to_symbols(_ids(risk.get("veto")))[:3])
        self.last_notes["veto"] = sorted(vetoed)
        return self.weights_for([s for s in picks if s not in vetoed], brief)


class PanelV2(StylePanel):
    """Four styles each name their best and worst eight; a name's score is its net votes.

    The book holds the ``top_k`` names with the most net votes. Names no style mentioned
    fill any gap in order of six-month return; names with more votes against than for are never held.
    """

    name = "panel_v2"

    def brief(self, view):
        return ranked(super().brief(view))

    def decide(self, view):
        brief = self.brief(view)
        if not brief.rows:
            return {}
        n = max(1, min(8, len(brief.rows) // 3))
        table = f"{brief.market}\n\n{brief.table()}"
        votes = {s: 0 for s in brief.rows}
        answered = 0
        for style in self.styles:
            reply = self.ask(style, STYLES[style],
                             f"{table}\n\nName the {n} ids you most want to hold and the {n} you most want to avoid. "
                             f"Reply as {{\"favours\": [{n} ids], \"avoids\": [{n} ids], \"why\": \"<1 sentence>\"}}",
                             view.clock)
            up, down = brief.to_symbols(_ids(reply.get("favours"))), brief.to_symbols(_ids(reply.get("avoids")))
            if not up and not down:
                continue
            answered += 1
            for s in dict.fromkeys(up):
                votes[s] += 1
            for s in dict.fromkeys(down):
                votes[s] -= 1
        if not answered:  # no style gave a usable answer: leave the book as it is
            self.bad_replies += 1
            return None
        # Most votes first; names nobody mentioned fill any gap by six-month return; net-negative names are out.
        picks = [s for s in sorted(votes, key=lambda s: (-votes[s], -brief.rows[s][2], s)) if votes[s] >= 0]
        return self.weights_for(picks, brief)
