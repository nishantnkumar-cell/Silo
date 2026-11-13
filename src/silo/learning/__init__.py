"""The learning loop: agents learn from their own past trades, under guard.

How it fits together (see ``docs/learning-loop.md``):

1. :mod:`silo.learning.journal` turns a run (backtest or paper) into a trade
   journal: every position from open to close, the numbers the agent saw when
   it bought, its stated reasons (for AI desks: which names the bull backed,
   which the bear warned about, the trader's "why"), and the outcome after
   costs and dividends, next to the market over the same weeks.
2. :mod:`silo.learning.review` reads the journal and proposes *lessons*: a
   statistical miner looks for entry conditions that kept losing to the
   market, and an optional AI reviewer reads an anonymised summary (no
   names, no dates) and writes up to three lessons in a fixed rule format.
3. :mod:`silo.learning.lessons` applies lessons to any agent: as filters on
   new entries (every agent) and, for AI agents, as extra guidance in the
   prompt.
4. :mod:`silo.learning.gate` decides whether a lesson goes live. It must
   beat the unchanged agent on data the lesson was *not* drawn from, beat
   random filters that skip as many trades, and survive a deflated Sharpe
   test that counts every gate run on that period. The locked holdout is
   never used. Every proposal and gate result is appended to a ledger.
5. :mod:`silo.learning.walkforward` tests the loop itself: learn on the past,
   trade the next period with what was learned, repeat, and compare with the
   agent that never learns.

Research only. Silo never trades real money.
"""

_EXPORTS = {
    "GateResult": "gate", "Ledger": "gate", "run_gate": "gate",
    "Trade": "journal", "build_journal": "journal",
    "FEATURES": "lessons", "Learned": "lessons", "Lesson": "lessons",
    "ai_review": "review", "mine": "review",
    "learning_walk_forward": "walkforward",
}
__all__ = ["FEATURES", "GateResult", "Learned", "Ledger", "Lesson", "Trade", "ai_review", "build_journal",
           "learning_walk_forward", "mine", "run_gate"]


def __getattr__(name):  # lazy: the strategy registry imports silo.learning.lessons early
    if name in _EXPORTS:
        import importlib

        return getattr(importlib.import_module(f"silo.learning.{_EXPORTS[name]}"), name)
    raise AttributeError(name)
