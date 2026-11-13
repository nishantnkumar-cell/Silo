"""Strategy interface shared by classic rules and, later, AI agents."""

from __future__ import annotations

from silo.firewall import PointInTimeView


class Strategy:
    """Return target weights from what the firewall lets you see.

    ``decide`` is called once per trading day at the close. Return a dict of
    symbol -> weight (0..1, summing to at most 1) to rebalance at the next
    open, or ``None`` to leave the book unchanged.
    """

    name = "strategy"
    uses_llm = False
    # Parameter values the lab may choose between in walk-forward tests.
    param_grid: dict[str, list] = {}

    def __init__(self, symbols: list[str]) -> None:
        if not symbols:
            raise ValueError("a strategy needs at least one symbol")
        self.symbols = list(symbols)

    def reset(self) -> None:
        """Clear any state before a new run."""

    def decide(self, view: PointInTimeView) -> dict[str, float] | None:
        raise NotImplementedError

    def manage(self, view: PointInTimeView) -> dict[str, float] | None:
        """Check open trades at a daily close, between rebalances too.

        Return ``{symbol: keep}`` for positions to cut at the next open, where
        ``keep`` is the fraction of the holding to keep (0 sells it all, 0.5
        sells half). Only sells are honoured, and they override the minimum
        hold, because a stop-loss must always be acted on. ``None`` (the
        default) means the strategy does not manage trades day to day.
        """
        return None
