"""The Core Strategy rulebook as guidance for AI contestants.

A short, paraphrased version of the supply-and-demand rules in
``docs/core-strategy-rulebook.md``, written for a model that only sees a
table of numbers. Each stock row gains six readings computed by
:mod:`silo.strategies.core_zones` (point-in-time, no names or dates), so the
model can apply the rules instead of guessing at chart shapes.

:class:`CoreDesk` is the bull/bear/trader/risk desk with this rulebook in
every prompt. The original ``desk`` is left untouched so its results (and
cached answers) stay comparable.
"""

from __future__ import annotations

import math

from silo.agents.brief import build_brief
from silo.agents.contestants import DebateDesk
from silo.strategies.core_zones import analyze

RULEBOOK = """Follow this rulebook (supply and demand, held for weeks to months, long only):
1. Prices turn where buyers and sellers were badly out of balance. A demand zone is a short pause followed by a strong rally; buy when price comes back down into a fresh demand zone, not after a rally.
2. Never buy after a big run-up into supply: that is the novice mistake. Buy cheap (low in the yearly range), sell expensive.
3. Score each setup out of 10: strength of the move away (2), few days in the pause (1), zone untested (2), weekly trend up (2; sideways 1; down 0), low in the monthly range (1), room to the next supply at least 5x the risk (2; 3x = 1).
4. Score 8.5+: buy at the zone. 7-8: buy only once price closes back above the zone. Below 7: no trade.
5. In a weekly downtrend only buy very low in the range with 5x reward-to-risk. High in the range only buy in an uptrend with 5x.
6. Every trade has a stop just below the zone and a target before the next supply; reward must be at least 3x the risk. Risk at most 2% of the book per trade and 6% across all trades. Never add to a losing position.
7. When nothing qualifies, holding cash is correct."""

CORE_COLUMNS = ("zone score /10 | in zone now (1 yes) | % above zone | weekly trend (+1 up, 0 sideways, -1 down) | "
                "position in monthly range % | reward:risk")


def core_numbers(view, symbol) -> list[float]:
    """Six readings from the best active demand zone, for the brief. Neutral values when none exists."""
    a = analyze(view.bars(symbol, lookback=2520))
    if a is None or a.best is None:
        tr = 0.0 if a is None else float(a.trend.direction)
        return [0.0, 0.0, 0.0, tr, 50.0, 0.0]
    st = a.best
    z = st.zone
    rr = st.reward_risk if math.isfinite(st.reward_risk) else 9.9
    above = 100 * (a.price / z.proximal - 1)
    in_zone = 1.0 if a.price <= z.proximal or z.in_zone_now else 0.0
    return [st.score, in_zone, above, float(a.trend.direction), 100 * min(max(st.curve_location, 0), 1.5),
            min(rr, 9.9)]


class CoreDesk(DebateDesk):
    """The debate desk, told to follow the Core Strategy rulebook and shown the zone readings."""

    name = "core_desk"

    def ask(self, role, system, prompt, clock):
        return super().ask(role, f"{RULEBOOK}\n\n{system}", prompt, clock)

    def brief(self, view):
        """Every liquid stock with its zone readings, shortlisted by zone score (not by recent gains)."""
        b = build_brief(view, self.symbols, self.anonymize, extra=(CORE_COLUMNS, core_numbers))
        if self.shortlist and len(b.rows) > self.shortlist:
            keep = sorted(b.rows, key=lambda s: (-b.rows[s][9], s))[: self.shortlist]
            b.rows = {s: b.rows[s] for s in keep}
            b.alias = {s: b.alias[s] for s in keep}
        return b
