"""Liquidity rules: never buy something you could get stuck in.

Three hard rules, enforced in the engine for every strategy (not just in a
universe screen a strategy could ignore):

1. **No penny stocks.** The last close must be at least ``min_price`` in the
   listing currency (pounds for London after pence conversion, dollars in
   the US).
2. **Enough trading.** The median daily value traded (close x volume) over
   the last ``window`` days must be at least ``min_value_traded``.
3. **Exit in a day or two.** A position may be no bigger than
   ``max_participation`` of that median daily value times
   ``max_days_to_exit``, so it could be sold without being most of the
   market for days.

All three are checked through the time firewall, at the decision clock.
A held name that stops passing may be held or trimmed, never added to.

The simulated broker also caps each fill at ``max_participation`` of the
fill day's volume; the rest of the order waits for the next open. (Using the
fill day's own volume for the cap is a common simplification: it limits the
size of a trade and never informs a decision.)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from silo.firewall import PointInTimeView


@dataclass(frozen=True)
class Liquidity:
    min_price: float = 1.0
    min_value_traded: float = 5e6
    window: int = 60
    max_participation: float = 0.10
    max_days_to_exit: float = 2.0

    @classmethod
    def for_market(cls, code: str) -> Liquidity:
        """Floors in each market's own units. UK: £1 a share and £5m a day."""
        return {"UK": cls(), "US": cls(min_price=5.0, min_value_traded=2e7)}.get(code, cls())

    def check(self, view: PointInTimeView, symbol: str) -> tuple[bool, float, str]:
        """(passes, largest position value allowed, reason if not)."""
        bars = view.bars(symbol, lookback=self.window)
        if len(bars) < self.window or "volume" not in bars:
            return False, 0.0, f"fewer than {self.window} days of volume"
        price = float(bars["close"].iloc[-1])
        value = float(np.nanmedian((bars["close"] * bars["volume"]).to_numpy()))
        cap = self.max_participation * value * self.max_days_to_exit
        if not price >= self.min_price:
            return False, cap, f"penny stock: price {price:.2f} below {self.min_price:.2f}"
        if not value >= self.min_value_traded:
            return False, cap, f"illiquid: {value / 1e6:.1f}m a day traded, need {self.min_value_traded / 1e6:.1f}m"
        return True, cap, ""


def apply_liquidity(
    target: dict[str, float],
    current: dict[str, float],
    view: PointInTimeView,
    equity: float,
    rules: Liquidity,
) -> tuple[dict[str, float], dict[str, str]]:
    """Clip target weights to the liquidity rules. Returns weights and reasons."""
    out, blocked = {}, {}
    for s in sorted(set(target) | set(current)):
        want = target.get(s, 0.0)
        ok, cap_value, reason = rules.check(view, s)
        if not ok:
            if want > current.get(s, 0.0) + 1e-12:
                blocked[s] = reason
            want = min(want, current.get(s, 0.0))  # hold or trim, never add
        if equity > 0 and want * equity > cap_value:
            cap_w = cap_value / equity
            if want > max(cap_w, current.get(s, 0.0)):
                blocked.setdefault(s, f"capped at {cap_w:.1%} of the book to stay sellable")
            want = min(want, max(cap_w, current.get(s, 0.0)))  # never forced to sell by the cap
        if want > 1e-9:
            out[s] = want
    return out, blocked
