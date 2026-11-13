"""Plain-rule versions of the news signals, so the lab can test them without a model.

* :class:`NewsMomentum` - momentum counted only on news days (moves on news
  tend to continue; moves without news tend to reverse).
* :class:`ResultsDrift` - hold stocks after a strong results-day reaction
  or an "ahead of expectations" update, for about three months.
* :class:`DirectorBuys` - hold stocks where directors bought shares in the
  last 90 days, unless the company has warned on profits.

With no announcements stored, news days fall back to heavy-volume days and
the other two simply stay in cash.
"""

from __future__ import annotations

from silo.agents import news
from silo.strategies.base import Strategy


class _Periodic(Strategy):
    period = "W"

    def reset(self) -> None:
        self.last = None

    def due(self, view) -> bool:
        p = view.clock.tz_convert(None).to_period(self.period)
        if p == self.last:
            return False
        self.last = p
        return True

    def ready(self, view, bars: int) -> bool:
        return all(len(view.bars(s, lookback=bars)) >= bars for s in self.symbols)


class NewsMomentum(_Periodic):
    """Monthly: the top names by six-month return earned on news days."""

    name = "news_mom"
    period = "M"
    param_grid = {"top_k": [3, 5]}

    def __init__(self, symbols, top_k: int = 3):
        super().__init__(symbols)
        self.top_k = top_k

    def decide(self, view):
        if not self.ready(view, news.WINDOW + 1) or not self.due(view):
            return None
        cols = news.columns(view, self.symbols)
        score = cols["news_mom"]
        winners = [s for s in sorted(score, key=lambda s: -score[s])[: self.top_k] if score[s] > 0]
        return {s: 1.0 / self.top_k for s in winners}


class ResultsDrift(_Periodic):
    """Weekly: names whose latest results went well, held while the news is fresh."""

    name = "results_drift"
    param_grid = {"min_move": [2.0, 5.0], "hold": [40, 60]}

    def __init__(self, symbols, min_move: float = 3.0, hold: int = 60, top_k: int = 8):
        super().__init__(symbols)
        self.min_move, self.hold, self.top_k = min_move, hold, top_k

    def decide(self, view):
        if not self.ready(view, 30) or not self.due(view):
            return None
        c = news.columns(view, self.symbols)
        good = [s for s in self.symbols
                if c["days_since_results"][s] <= self.hold and not c["warning"][s]
                and (c["results_move"][s] >= self.min_move or c["expect"][s] > 0)]
        good.sort(key=lambda s: -(c["results_move"][s] + 5 * c["expect"][s]))
        return {s: 1.0 / self.top_k for s in good[: self.top_k]}


class DirectorBuys(_Periodic):
    """Weekly: names with director purchases in the last 90 days and no profit warning."""

    name = "director_buys"
    param_grid = {"top_k": [5, 8]}

    def __init__(self, symbols, top_k: int = 8):
        super().__init__(symbols)
        self.top_k = top_k

    def decide(self, view):
        if not self.ready(view, 30) or not self.due(view):
            return None
        c = news.columns(view, self.symbols)
        picks = [s for s in self.symbols if c["director_buys"][s] >= 1 and not c["warning"][s]]
        picks.sort(key=lambda s: -c["director_buys"][s])
        return {s: 1.0 / self.top_k for s in picks[: self.top_k]}
