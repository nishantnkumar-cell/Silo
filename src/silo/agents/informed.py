"""Panel and desk contestants that also hear from context agents.

A *context agent* turns non-price data into numbers for the brief, plus one
line about the market, one investing voice for the panel, one analyst for
the desk, and a note for the risk manager. There are two: the cross-market
agent (:mod:`silo.agents.crossasset`) and the news agent
(:mod:`silo.agents.news`).

The deciding model never sees raw text, sector names, series levels or
dates: only extra numeric columns next to the anonymised ones it already
gets. The original ``panel`` and ``desk`` are left untouched so their cached
backtests stay valid; these are separate contestants so the lab can compare
"panel" with "panel + cross-market" on the same history.
"""

from __future__ import annotations

import numpy as np

from silo.agents import crossasset
from silo.agents.brief import build_brief
from silo.agents.contestants import STYLES, DebateDesk, StylePanel


class CrossAssetAgent:
    name = "macro"
    style = ("cross-market", "You are a cross-market analyst. You favour stocks in strong sectors, laggards in "
             "strong sectors that may catch up, and stocks whose sensitivities to the pound, oil, metals and gilt "
             "yields line up with where those markets are heading. You avoid books that hinge on one market.")
    analyst = ("cross-market analyst", "You are the cross-market analyst. Use the sector and macro columns and the "
               "cross-market line to say which stocks the wider market picture helps or hurts.")

    def __init__(self, sectors: dict[str, str] | None = None):
        self.sectors = sectors

    def prepare(self, view, symbols):
        self._readings = crossasset.driver_readings(view)
        self._betas = crossasset.sensitivities(view, symbols)
        cols = crossasset.columns(view, symbols, self.sectors, self._readings, self._betas)
        return ([(header, cols[key]) for key, header in crossasset.COLUMNS],
                crossasset.market_line(view, symbols, self._readings))

    def risk_note(self, picks):
        return crossasset.exposure_line(picks, self._betas)


class NewsAgent:
    name = "news"
    style = ("event-driven", "You are an event-driven investor. You favour stocks that rose on their latest "
             "results or said they are ahead of expectations, whose gains came on news days rather than quiet "
             "days, and where directors have been buying. You avoid stocks with a recent profit warning.")
    analyst = ("news analyst", "You are the news analyst. Use the results, expectations, warning, director "
               "buying and news-day columns to say which stocks recent company news supports or undermines.")

    def __init__(self, sectors=None):
        pass

    def prepare(self, view, symbols):
        from silo.agents import news

        cols = news.columns(view, symbols)
        self._cols = cols
        return [(header, cols[key]) for key, header in news.COLUMNS], news.market_line(view, symbols, cols)

    def risk_note(self, picks):
        warned = sum(1 for s in picks if self._cols["warning"].get(s, 0) > 0)
        stale = sum(1 for s in picks if self._cols["days_since_results"].get(s, 0) >= 120)
        return (f"News check: {warned} of these picks had a profit warning in the last 90 days; "
                f"{stale} have not reported results for 120 trading days or more.")


AGENTS = {"macro": CrossAssetAgent, "news": NewsAgent}


class _Informed:
    """Shared set-up: build the usual brief, then let each context agent add to it."""

    sources: tuple[str, ...] = ()

    def _agents(self):
        if not hasattr(self, "_ctx"):
            self._ctx = [AGENTS[s](getattr(self, "sectors", None)) for s in self.sources]
        return self._ctx

    def informed_brief(self, view):
        brief = build_brief(view, self.symbols, self.anonymize, limit=self.shortlist)
        if not brief.rows:
            return brief, ""
        lines = []
        for agent in self._agents():
            # numbers are computed over the whole universe, so sector and market averages don't
            # depend on the shortlist; only the shortlisted rows are shown
            cols, line = agent.prepare(view, self.symbols)
            brief.add_columns(cols)
            lines.append(line)
        return brief, "\n".join(lines)


class InformedPanel(_Informed, StylePanel):
    """The style panel plus one voice per context agent."""

    sources = ("macro",)
    name = "panel_macro"

    def __init__(self, symbols, client=None, anonymize=True, shortlist=30, top_k=12, sectors=None):
        super().__init__(symbols, client, anonymize, shortlist, top_k)
        self.sectors = sectors

    def decide(self, view):
        brief, context = self.informed_brief(view)
        if not brief.rows:
            return {}
        table = f"{brief.market}\n{context}\n\n{brief.table()}"
        voices = [(k, STYLES[k]) for k in self.styles] + [a.style for a in self._agents()]
        totals = {s: [] for s in brief.rows}
        for role, system in voices:
            reply = self.ask(role, system,
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
        if not mean:  # no voice gave a usable answer: leave the book as it is
            self.bad_replies += 1
            return None
        picks = [s for s in sorted(mean, key=lambda s: -mean[s]) if mean[s] > 0]
        return self.weights_for(picks, brief)


class InformedDesk(_Informed, DebateDesk):
    """The debate desk with an analyst per context agent briefing both researchers first."""

    sources = ("macro",)
    name = "desk_macro"

    def __init__(self, symbols, client=None, anonymize=True, shortlist=30, top_k=12, sectors=None):
        super().__init__(symbols, client, anonymize, shortlist, top_k)
        self.sectors = sectors

    def decide(self, view):
        brief, context = self.informed_brief(view)
        if not brief.rows:
            return {}
        clock = view.clock
        table = f"{brief.market}\n{context}\n\n{brief.table()}"
        notes = []
        for agent in self._agents():
            role, system = agent.analyst
            r = self.ask(agent.name, system, f"{table}\n\nReply as {{\"note\": \"<3 sentences>\", "
                         "\"favours\": [up to 5 ids], \"avoids\": [up to 5 ids]}", clock)
            notes.append(f"{role.capitalize()}: {r.get('note', '')} Favours: {r.get('favours', [])}. "
                         f"Avoids: {r.get('avoids', [])}.")
        briefing = "\n".join(notes)
        bull = self.ask("bull", "You are the bull researcher. Make the strongest case for the stocks most "
                        "likely to rise over the next 1 to 3 months.",
                        f"{table}\n\n{briefing}\n\nReply as {{\"argument\": \"<3 sentences>\", "
                        "\"names\": [up to 8 ids]}", clock)
        bear = self.ask("bear", "You are the bear researcher. Make the strongest case against the stocks most "
                        "likely to fall or disappoint over the next 1 to 3 months.",
                        f"{table}\n\n{briefing}\n\nReply as {{\"argument\": \"<3 sentences>\", "
                        "\"names\": [up to 8 ids]}", clock)
        debate = (f"Bull case: {bull.get('argument', '')} Favours: {bull.get('names', [])}\n"
                  f"Bear case: {bear.get('argument', '')} Warns about: {bear.get('names', [])}")
        trade = self.ask("trader", "You are the trader. Weigh the analysts and both researchers and choose what "
                         "to hold.", f"{table}\n\n{briefing}\n{debate}\n\nReply as {{\"picks\": [up to "
                         f"{self.top_k} ids], \"why\": \"<2 sentences>\"}}. Fewer picks is fine if few look good.",
                         clock)
        raw = trade.get("picks")
        picks = brief.to_symbols(raw if isinstance(raw, list) else [])
        if not isinstance(raw, list) or (raw and not picks):  # not an answer: leave the book as it is
            self.bad_replies += 1
            return None
        if not picks:
            return {}
        risk_notes = "\n".join(a.risk_note(picks) for a in self._agents())
        risk = self.ask("risk", "You are the risk manager. Remove any pick that is too risky to hold for "
                        "weeks: falling knives, extreme volatility, or a crowded single bet, including a book "
                        "that depends on one market such as the pound or oil.",
                        f"{brief.market}\n{context}\n\n{brief.table(picks)}\n\n{risk_notes}\n\nReply as "
                        "{\"veto\": [ids to drop], \"why\": \"<1 sentence>\"}", clock)
        vetoed = set(brief.to_symbols(risk.get("veto", [])))
        return self.weights_for([s for s in picks if s not in vetoed], brief)


class NewsPanel(InformedPanel):
    sources, name = ("news",), "panel_news"


class NewsDesk(InformedDesk):
    sources, name = ("news",), "desk_news"


class FullPanel(InformedPanel):
    sources, name = ("macro", "news"), "panel_full"


class FullDesk(InformedDesk):
    sources, name = ("macro", "news"), "desk_full"
