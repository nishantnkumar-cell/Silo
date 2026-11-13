"""Day-by-day simulation that routes every read through the time firewall.

The same :class:`Session` drives backtests and the forward paper loop. Each
trading day:
1. dividends go ex: shares held at the prior close receive cash;
2. orders decided at the last rebalance fill at today's open;
3. the clock moves to today's close and equity is marked;
4. if the strategy manages open trades (``Strategy.manage``), it checks
   them at every close: stop-loss and profit-taking sells go to the next
   open even between rebalances and inside the minimum hold;
5. on a rebalance day the strategy sees a :class:`PointInTimeView` at that
   clock; its target weights pass through the position-trading constraints
   and become orders for the next open. Liquidity rules (no penny stocks,
   enough daily trading, positions small enough to sell) apply first.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from silo.broker import CostModel, Order, SimBroker
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
        rebalance = is_rebalance_day(date, self.last_date, self.constraints.rebalance)
        exits: dict[str, float] = {}
        view = None
        if self._checks_daily() or (rebalance and self._has_manage()):
            view = PointInTimeView(data.prepared, clock)
            exits = {s: min(max(float(k), 0.0), 1.0) for s, k in (self.strategy.manage(view) or {}).items()
                     if self.broker.positions.get(s) and s in self.last_close}
        if rebalance:
            view = view or PointInTimeView(data.prepared, clock)
            weights = self.strategy.decide(view)
            if weights is None and exits:
                final = self._apply_exits(self.broker.weights(self.last_close), exits)
                self._queue_exits(exits, clock)
                decision = {"date": date, "clock": clock, "weights": final, "proposed": {}, "exits": dict(exits)}
                self.decisions.append(decision)
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
                if exits:  # stops and profit-taking override the minimum hold
                    final = self._apply_exits(final, exits, current)
                self.broker.target_weights(final, self.last_close, clock)
                decision = {"date": date, "clock": clock, "weights": dict(final), "proposed": dict(weights)}
                if exits:
                    decision["exits"] = dict(exits)
                if blocked:
                    decision["liquidity"] = blocked
                self.decisions.append(decision)
        elif exits:
            final = self._apply_exits(self.broker.weights(self.last_close), exits)
            self._queue_exits(exits, clock)
            decision = {"date": date, "clock": clock, "weights": final, "proposed": {}, "exits": dict(exits)}
            self.decisions.append(decision)
        self.last_date = date
        return {
            "date": date,
            "equity": self.equity[date],
            "fills": self.broker.fills[n_fills:],
            "dividends": self.broker.dividends[n_divs:],
            "decision": decision,
        }

    def _has_manage(self) -> bool:
        return type(self.strategy).manage is not Strategy.manage

    def _checks_daily(self) -> bool:
        return self.constraints.daily_exit_checks and self._has_manage()

    @staticmethod
    def _apply_exits(weights: dict[str, float], exits: dict[str, float],
                     current: dict[str, float] | None = None) -> dict[str, float]:
        """Cut each exited name to ``keep`` times what is held now (never more than the plan)."""
        current = weights if current is None else current
        out = dict(weights)
        for s, keep in exits.items():
            out[s] = min(out.get(s, 0.0), keep * current.get(s, 0.0))
        return {s: w for s, w in out.items() if w > 1e-9}

    def _queue_exits(self, exits: dict[str, float], clock: pd.Timestamp) -> None:
        """Sell orders for the next open, outside a rebalance: only the exited names trade."""
        self.broker.pending = [o for o in self.broker.pending if o.symbol not in exits]
        for s, keep in exits.items():
            qty = self.broker.positions.get(s, 0.0)
            sell = qty * (1.0 - keep)
            if sell > 0:
                self.broker.pending.append(Order(s, -sell, clock))

    def result(self) -> BacktestResult:
        fills = pd.DataFrame(self.broker.fills, columns=FILL_COLUMNS)
        divs = sum(d["qty"] * d["per_share"] for d in self.broker.dividends)
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
