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

    def __init__(self, symbols: list[str]) -> None:
        if not symbols:
            raise ValueError("a strategy needs at least one symbol")
        self.symbols = list(symbols)

    def reset(self) -> None:
        """Clear any state before a new run."""

    def decide(self, view: PointInTimeView) -> dict[str, float] | None:
        raise NotImplementedError
