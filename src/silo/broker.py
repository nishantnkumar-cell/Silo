"""Simulated broker. The only execution venue Silo ships with.

Orders are decided at a day's close and filled at the next day's open, with
commission, half-spread and slippage charged on every fill. Long-only and
unlevered: buys are scaled down to the cash available.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class CostModel:
    commission_per_share: float = 0.005
    min_commission: float = 1.0
    spread_bps: float = 2.0  # full quoted spread; we pay half on each fill
    slippage_bps: float = 3.0
    llm_cost_per_decision: float = 0.0  # set when an LLM agent makes decisions

    def fill_price(self, open_price: float, side: int) -> float:
        bump = (self.spread_bps / 2 + self.slippage_bps) / 1e4
        return open_price * (1 + side * bump)

    def commission(self, qty: float) -> float:
        return max(self.min_commission, self.commission_per_share * abs(qty))


@dataclass
class Order:
    symbol: str
    qty: float  # positive buys, negative sells
    decided_at: pd.Timestamp


@dataclass
class SimBroker:
    cash: float = 100_000.0
    costs: CostModel = field(default_factory=CostModel)
    min_trade_value: float = 50.0
    positions: dict[str, float] = field(default_factory=dict)
    pending: list[Order] = field(default_factory=list)
    fills: list[dict] = field(default_factory=list)
    fees_paid: float = 0.0

    def equity(self, prices: dict[str, float]) -> float:
        return self.cash + sum(q * prices[s] for s, q in self.positions.items() if q)

    def target_weights(self, weights: dict[str, float], prices: dict[str, float], when: pd.Timestamp) -> None:
        """Queue orders that move the book toward ``weights`` at the next open."""
        if any(w < 0 for w in weights.values()):
            raise ValueError("Silo's simulated broker is long-only")
        total = sum(weights.values())
        if total > 1.0 + 1e-9:
            raise ValueError(f"weights sum to {total:.3f}; leverage is not allowed")
        equity = self.equity(prices)
        self.pending.clear()
        for symbol in sorted(set(weights) | set(self.positions)):
            target_qty = weights.get(symbol, 0.0) * equity / prices[symbol]
            delta = target_qty - self.positions.get(symbol, 0.0)
            if abs(delta) * prices[symbol] >= self.min_trade_value or (target_qty == 0 and delta):
                self.pending.append(Order(symbol, delta, when))

    def fill_at_open(self, opens: dict[str, float], date: pd.Timestamp) -> None:
        """Fill queued orders at ``date``'s open. Sells go first to free cash."""
        orders, self.pending = sorted(self.pending, key=lambda o: o.qty), []
        for order in orders:
            if order.symbol not in opens or pd.isna(opens[order.symbol]):
                continue  # no bar today: the order lapses
            side = 1 if order.qty > 0 else -1
            price = self.costs.fill_price(opens[order.symbol], side)
            qty = order.qty
            if side > 0:
                affordable = (self.cash - self.costs.commission(qty)) / price
                qty = max(0.0, min(qty, affordable))
                if qty * price < self.min_trade_value:
                    continue
            fee = self.costs.commission(qty)
            self.cash -= qty * price + fee
            self.fees_paid += fee
            new_qty = self.positions.get(order.symbol, 0.0) + qty
            self.positions[order.symbol] = 0.0 if abs(new_qty) < 1e-9 else new_qty
            self.fills.append(
                {
                    "date": date,
                    "decided_at": order.decided_at,
                    "symbol": order.symbol,
                    "qty": qty,
                    "price": price,
                    "fee": fee,
                }
            )

    def charge(self, amount: float) -> None:
        """Debit a non-trade cost, such as LLM usage for a decision."""
        self.cash -= amount
        self.fees_paid += amount
