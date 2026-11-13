"""Day-by-day simulation that routes every read through the time firewall.

The same :class:`Session` drives backtests and the forward paper loop. Each
trading day:
1. dividends go ex: shares held at the prior close receive cash;
2. orders decided at the last rebalance fill at today's open;
3. the clock moves to today's close and equity is marked;
4. on a rebalance day the strategy sees a :class:`PointInTimeView` at that
   clock; its target weights pass through the position-trading constraints
   and become orders for the next open. Liquidity rules (no penny stocks,
   enough daily trading, positions small enough to sell) apply first.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from silo.broker import CostModel, SimBroker
from silo.constraints import Constraints, apply_constraints, is_rebalance_day
from silo.firewall import PointInTimeView, PreparedTables
from silo.liquidity import apply_liquidity
from silo.metrics import summarize
from silo.strategies.base import Strategy

FILL_COLUMNS = ["date", "decided_at", "symbol", "qty", "price", "fee"]


@dataclass
class BacktestResult:
    strategy: str
    equity: pd.Series
    fills: pd.DataFrame
    decisions: list[dict] = field(default_factory=list)
    fees_paid: float = 0.0
    dividends: float = 0.0

    def metrics(self) -> dict[str, float]:
        return summarize(self.equity, self.fills)


class MarketData:
    """Tables prepared once for fast point-in-time reads and daily lookups."""

    def __init__(self, tables: dict[str, pd.DataFrame]) -> None:
        self.tables = tables
        self.prepared = PreparedTables(tables)
        self.rows = {}
        for s, df in tables.items():
            div = df["dividend"].to_numpy() if "dividend" in df.columns else [0.0] * len(df)
            vol = df["volume"].to_numpy() if "volume" in df.columns else [float("nan")] * len(df)
            self.rows[s] = dict(
                zip(df.index, zip(df["open"].to_numpy(), df["close"].to_numpy(), df["available_at"], div, vol))
            )
        self.dates = sorted(set().union(*(df.index for df in tables.values())))


@dataclass
class Session:
    strategy: Strategy
    broker: SimBroker
    constraints: Constraints = field(default_factory=Constraints)
    last_date: pd.Timestamp | None = None
    last_close: dict[str, float] = field(default_factory=dict)
    held_days: dict[str, int] = field(default_factory=dict)
    equity: dict[pd.Timestamp, float] = field(default_factory=dict)
    decisions: list[dict] = field(default_factory=list)

    @classmethod
    def start(cls, strategy, cash=100_000.0, costs=None, constraints=None) -> Session:
        strategy.reset()
        constraints = constraints or Constraints()
        liq = constraints.liquidity
        broker = SimBroker(cash=cash, costs=costs or CostModel(),
                           max_participation=liq.max_participation if liq else None)
        return cls(strategy, broker, constraints)

    def step(self, date: pd.Timestamp, data: MarketData) -> dict:
        """Advance one trading day. Returns what happened, for journals."""
        today = {s: r[date] for s, r in data.rows.items() if date in r}
        n_fills = len(self.broker.fills)
        n_divs = len(self.broker.dividends)
        self.broker.credit_dividends({s: float(r[3]) for s, r in today.items() if r[3]}, date)
        volumes = {s: float(r[4]) for s, r in today.items()} if self.constraints.liquidity else None
        self.broker.fill_at_open({s: float(r[0]) for s, r in today.items()}, date, volumes)
        self.last_close.update({s: float(r[1]) for s, r in today.items()})
        for s in list(self.held_days):
            if not self.broker.positions.get(s):
                del self.held_days[s]
        for s, q in self.broker.positions.items():
            if q:
                self.held_days[s] = self.held_days.get(s, -1) + 1
        clock = max(r[2] for r in today.values())
        self.equity[date] = self.broker.equity(self.last_close)

        decision = None
        if is_rebalance_day(date, self.last_date, self.constraints.rebalance):
            view = PointInTimeView(data.prepared, clock)
            weights = self.strategy.decide(view)
            if weights is not None:
                if self.strategy.uses_llm:
                    self.broker.charge(self.broker.costs.llm_cost_per_decision)
                weights = {s: w for s, w in weights.items() if s in self.last_close}  # no price yet: can't trade
                current = self.broker.weights(self.last_close)
                blocked = {}
                target = weights
                if self.constraints.liquidity:
                    target, blocked = apply_liquidity(
                        weights, current, view, self.equity[date], self.constraints.liquidity
                    )
                final = apply_constraints(target, current, self.held_days, self.constraints)
                self.broker.target_weights(final, self.last_close, clock)
                decision = {"date": date, "clock": clock, "weights": dict(final), "proposed": dict(weights)}
                if blocked:
                    decision["liquidity"] = blocked
                self.decisions.append(decision)
        self.last_date = date
        return {
            "date": date,
            "equity": self.equity[date],
            "fills": self.broker.fills[n_fills:],
            "dividends": self.broker.dividends[n_divs:],
            "decision": decision,
        }

    def result(self) -> BacktestResult:
        fills = pd.DataFrame(self.broker.fills, columns=FILL_COLUMNS)
        divs = sum(d["qty"] * d["per_share"] - d.get("withheld", 0.0) for d in self.broker.dividends)
        return BacktestResult(
            self.strategy.name, pd.Series(self.equity, name="equity", dtype=float), fills,
            self.decisions, self.broker.fees_paid, divs,
        )


def run_backtest(
    strategy: Strategy,
    tables: dict[str, pd.DataFrame],
    start: str | None = None,
    end: str | None = None,
    cash: float = 100_000.0,
    costs: CostModel | None = None,
    constraints: Constraints | None = None,
) -> BacktestResult:
    data = MarketData({s: tables[s] for s in strategy.symbols})
    lo = None if start is None else pd.Timestamp(start)
    hi = None if end is None else pd.Timestamp(end)
    session = Session.start(strategy, cash, costs, constraints)
    for date in data.dates:
        if (lo is None or date >= lo) and (hi is None or date <= hi):
            session.step(date, data)
    return session.result()


def buy_and_hold(tables: dict[str, pd.DataFrame], symbols: list[str], **kwargs) -> BacktestResult:
    from silo.strategies.classic import BuyAndHold

    return run_backtest(BuyAndHold(symbols), tables, **kwargs)
