"""Simulated broker. The only execution venue Silo ships with.

Orders are decided at a day's close and filled at the next day's open, with
commission, half-spread, slippage, percentage fees (crypto, FX) and (in the
UK) stamp duty charged on every fill. Long-only and unlevered: buys are scaled down to the cash available.
Dividends are credited as cash on the ex-date to shares held at the prior close.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class CostModel:
    commission_per_share: float = 0.005
    min_commission: float = 1.0
    commission_flat: float = 0.0  # if set, a flat fee per fill replaces the per-share fee
    spread_bps: float = 2.0  # full quoted spread; we pay half on each fill
    slippage_bps: float = 3.0
    stamp_duty_bps: float = 0.0  # charged on buys only (UK: 50bp)
    commission_bps: float = 0.0  # percentage fee on traded value, both sides (crypto exchanges)
    fx_bps: float = 0.0  # currency conversion on traded value, both sides (pounds into dollars)
    llm_cost_per_decision: float = 0.0  # set when an LLM agent makes decisions
    cash_interest: bool = False  # pay interest on idle cash (off by default; turn on to compare with a tracker)
    cash_rate: float = 0.0  # % a year, used when no Bank Rate series is cached (ctx:macro:bank_rate)

    @classmethod
    def uk(cls, **overrides) -> CostModel:
        """Typical retail costs for FTSE 350 shares.

        0.5% stamp duty (SDRT) on purchases, a flat dealing fee, and wider
        spreads than US large caps. AIM shares and ETFs are exempt from stamp
        duty; pass ``stamp_duty_bps=0`` for those books.
        """
        base = dict(
            commission_per_share=0.0,
            min_commission=0.0,
            commission_flat=5.0,
            spread_bps=20.0,
            slippage_bps=10.0,
            stamp_duty_bps=50.0,
        )
        base.update(overrides)
        return cls(**base)

    @classmethod
    def us_from_uk(cls, **overrides) -> CostModel:
        """US-listed ETFs bought with pounds: US costs plus a 0.15% FX fee each way
        (Trading 212's published rate), so a round trip pays about 0.4%."""
        base = dict(fx_bps=15.0)
        base.update(overrides)
        return cls(**base)

    @classmethod
    def crypto(cls, **overrides) -> CostModel:
        """Retail crypto on a mainstream exchange: 0.25% taker fee each way, a
        10bp spread and 10bp slippage, so a round trip costs about 0.8%.
        Apps with a built-in markup (often 1% to 1.5%) are dearer still."""
        base = dict(commission_per_share=0.0, min_commission=0.0, commission_bps=25.0,
                    spread_bps=10.0, slippage_bps=10.0)
        base.update(overrides)
        return cls(**base)

    @classmethod
    def for_market(cls, code: str) -> CostModel:
        return {"UK": cls.uk, "CRYPTO": cls.crypto}.get(code, cls)()

    def scaled(self, k: float) -> CostModel:
        """Every cost multiplied by ``k``: a stress test for cost assumptions."""
        from dataclasses import replace

        return replace(self, commission_per_share=self.commission_per_share * k,
                       min_commission=self.min_commission * k, commission_flat=self.commission_flat * k,
                       spread_bps=self.spread_bps * k, slippage_bps=self.slippage_bps * k,
                       stamp_duty_bps=self.stamp_duty_bps * k, commission_bps=self.commission_bps * k,
                       fx_bps=self.fx_bps * k)

    def fill_price(self, open_price: float, side: int) -> float:
        bump = (self.spread_bps / 2 + self.slippage_bps) / 1e4
        return open_price * (1 + side * bump)

    def commission(self, qty: float) -> float:
        if self.commission_flat:
            return self.commission_flat
        return max(self.min_commission, self.commission_per_share * abs(qty))

    def fees(self, qty: float, price: float) -> float:
        """All fees on one fill: commission, percentage fees, and stamp duty on buys."""
        stamp = self.stamp_duty_bps / 1e4 * qty * price if qty > 0 else 0.0
        pct = (self.commission_bps + self.fx_bps) / 1e4 * abs(qty) * price
        return self.commission(qty) + stamp + pct

    @property
    def buy_fee_rate(self) -> float:
        """Fees proportional to value on a buy, as a fraction."""
        return (self.stamp_duty_bps + self.commission_bps + self.fx_bps) / 1e4


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
    max_participation: float | None = None  # cap each fill at this share of the day's volume
    positions: dict[str, float] = field(default_factory=dict)
    opened_at: dict[str, pd.Timestamp] = field(default_factory=dict)
    pending: list[Order] = field(default_factory=list)
    fills: list[dict] = field(default_factory=list)
    dividends: list[dict] = field(default_factory=list)
    fees_paid: float = 0.0
    interest_earned: float = 0.0

    def equity(self, prices: dict[str, float]) -> float:
        return self.cash + sum(q * prices[s] for s, q in self.positions.items() if q)

    def weights(self, prices: dict[str, float]) -> dict[str, float]:
        eq = self.equity(prices)
        return {s: q * prices[s] / eq for s, q in self.positions.items() if q and eq > 0}

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

    def credit_dividends(self, per_share: dict[str, float], date: pd.Timestamp) -> None:
        """Pay cash dividends on shares held going into the ex-date."""
        for symbol, amount in per_share.items():
            qty = self.positions.get(symbol, 0.0)
            if qty and amount:
                self.cash += qty * amount
                self.dividends.append({"date": date, "symbol": symbol, "qty": qty, "per_share": amount})

    def fill_at_open(
        self, opens: dict[str, float], date: pd.Timestamp, volumes: dict[str, float] | None = None
    ) -> None:
        """Fill queued orders at ``date``'s open. Sells go first to free cash.

        With ``max_participation`` set and ``volumes`` given, a fill is capped
        at that share of the day's volume and the rest waits for the next open.
        """
        orders, self.pending = sorted(self.pending, key=lambda o: o.qty), []
        for order in orders:
            if order.symbol not in opens or pd.isna(opens[order.symbol]):
                continue  # no bar today: the order lapses
            side = 1 if order.qty > 0 else -1
            price = self.costs.fill_price(opens[order.symbol], side)
            qty = order.qty
            if self.max_participation is not None and volumes is not None:
                room = self.max_participation * float(volumes.get(order.symbol) or 0.0)
                if abs(qty) > room:
                    rest = qty - side * room
                    qty = side * room
                    self.pending.append(Order(order.symbol, rest, order.decided_at))
                    if not room:
                        continue
            if side > 0:
                per_share = price * (1 + self.costs.buy_fee_rate)
                affordable = (self.cash - self.costs.commission(qty)) / per_share
                qty = max(0.0, min(qty, affordable))
                if qty * price < self.min_trade_value:
                    continue
            fee = self.costs.fees(qty, price)
            self.cash -= qty * price + fee
            self.fees_paid += fee
            before = self.positions.get(order.symbol, 0.0)
            new_qty = before + qty
            self.positions[order.symbol] = 0.0 if abs(new_qty) < 1e-9 else new_qty
            if before <= 0 < self.positions[order.symbol]:
                self.opened_at[order.symbol] = date
            elif self.positions[order.symbol] == 0:
                self.opened_at.pop(order.symbol, None)
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

    def credit_interest(self, amount: float) -> None:
        """Interest on idle cash (see ``CostModel.cash_interest``)."""
        if amount > 0:
            self.cash += amount
            self.interest_earned += amount

    def charge(self, amount: float) -> None:
        """Debit a non-trade cost, such as LLM usage for a decision."""
        self.cash -= amount
        self.fees_paid += amount
