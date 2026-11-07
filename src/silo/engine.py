"""Day-by-day backtester that routes every read through the time firewall.

Each trading day:
1. orders decided yesterday fill at today's open (simulated broker);
2. the clock moves to today's close and equity is marked;
3. the strategy sees a :class:`PointInTimeView` at that clock and returns
   target weights, which become orders for tomorrow's open.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from silo.broker import CostModel, SimBroker
from silo.firewall import PointInTimeView
from silo.metrics import summarize
from silo.strategies.base import Strategy


@dataclass
class BacktestResult:
    strategy: str
    equity: pd.Series
    fills: pd.DataFrame
    decisions: list[dict] = field(default_factory=list)
    fees_paid: float = 0.0

    def metrics(self) -> dict[str, float]:
        return summarize(self.equity, self.fills)


def run_backtest(
    strategy: Strategy,
    tables: dict[str, pd.DataFrame],
    start: str | None = None,
    end: str | None = None,
    cash: float = 100_000.0,
    costs: CostModel | None = None,
) -> BacktestResult:
    tables = {s: tables[s] for s in strategy.symbols}
    dates = sorted(set().union(*(df.index for df in tables.values())))
    dates = [d for d in dates if (start is None or d >= pd.Timestamp(start)) and (end is None or d <= pd.Timestamp(end))]
    broker = SimBroker(cash=cash, costs=costs or CostModel())
    strategy.reset()
    last_close: dict[str, float] = {}
    equity, decisions = {}, []

    for date in dates:
        today = {s: df.loc[date] for s, df in tables.items() if date in df.index}
        broker.fill_at_open({s: float(r["open"]) for s, r in today.items()}, date)
        last_close.update({s: float(r["close"]) for s, r in today.items()})
        clock = max(r["available_at"] for r in today.values())
        equity[date] = broker.equity(last_close)

        view = PointInTimeView(tables, clock)
        weights = strategy.decide(view)
        if weights is not None:
            if strategy.uses_llm:
                broker.charge(broker.costs.llm_cost_per_decision)
            if all(s in last_close for s in weights):
                broker.target_weights(weights, last_close, clock)
                decisions.append({"date": date, "clock": clock, "weights": dict(weights)})

    fills = pd.DataFrame(broker.fills, columns=["date", "decided_at", "symbol", "qty", "price", "fee"])
    return BacktestResult(strategy.name, pd.Series(equity, name="equity"), fills, decisions, broker.fees_paid)


def buy_and_hold(tables: dict[str, pd.DataFrame], symbols: list[str], **kwargs) -> BacktestResult:
    from silo.strategies.classic import BuyAndHold

    return run_backtest(BuyAndHold(symbols), tables, **kwargs)
