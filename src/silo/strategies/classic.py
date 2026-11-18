"""Classic public strategy families, used as baselines AI agents must beat.

Each is a plain re-implementation of a widely published rule (the kind
shared as TradingView scripts), not a copy of any particular script.
Signals are computed only from the point-in-time view.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from silo.firewall import PointInTimeView
from silo.strategies.base import Strategy


def _equal(active: list[str], n: int) -> dict[str, float]:
    return {s: 1.0 / n for s in active}


class _PerSymbolSwitch(Strategy):
    """Holds 1/n of equity in each symbol whose signal is on."""

    lookback = 1

    def reset(self) -> None:
        self.state = {s: False for s in self.symbols}

    def signal(self, close: pd.Series, bars: pd.DataFrame, holding: bool) -> bool:
        raise NotImplementedError

    def decide(self, view: PointInTimeView) -> dict[str, float] | None:
        changed = False
        for s in self.symbols:
            bars = view.bars(s, lookback=self.lookback + 1)
            if len(bars) < self.lookback + 1:
                continue
            on = bool(self.signal(bars["close"], bars, self.state[s]))
            if on != self.state[s]:
                self.state[s], changed = on, True
        if not changed:
            return None
        return _equal([s for s in self.symbols if self.state[s]], len(self.symbols))


class BuyAndHold(Strategy):
    """Buy every symbol in equal weights once, then hold.

    It buys the symbols that are trading at the start, once each has
    ``warmup`` bars of history, so the purchase is not refused by the
    liquidity rules (which need 60 days of volume). A run that starts with that
    history already in place buys on its first day.
    """

    name = "buy_and_hold"

    def __init__(self, symbols, warmup: int = 60):
        super().__init__(symbols)
        self.warmup = warmup

    def reset(self) -> None:
        self.done = False

    def decide(self, view):
        """Equal weights in every symbol that has a price at the start, then hold.

        Symbols that list later are not added: that is what "buy and hold" means.
        """
        if self.done:
            return None
        live = [s for s in self.symbols if len(view.bars(s, lookback=1))]
        if not live or any(len(view.bars(s, lookback=self.warmup)) < self.warmup for s in live):
            return None
        self.done = True
        return _equal(live, len(live))


class SmaCross(_PerSymbolSwitch):
    """Long while the fast moving average is above the slow one (50/200)."""

    name = "sma_cross"
    param_grid = {"fast": [20, 50], "slow": [100, 200]}

    def __init__(self, symbols, fast: int = 50, slow: int = 200):
        super().__init__(symbols)
        self.fast, self.slow, self.lookback = fast, slow, slow

    def signal(self, close, bars, holding):
        return close.iloc[-self.fast :].mean() > close.iloc[-self.slow :].mean()


class DonchianBreakout(_PerSymbolSwitch):
    """Turtle-style: enter on a 20-day high, exit on a 10-day low."""

    name = "donchian"
    param_grid = {"entry": [20, 55], "exit": [10, 20]}

    def __init__(self, symbols, entry: int = 20, exit: int = 10):
        super().__init__(symbols)
        self.entry, self.exit, self.lookback = entry, exit, max(entry, exit)

    def signal(self, close, bars, holding):
        prior = bars.iloc[:-1]
        if not holding:
            return close.iloc[-1] > prior["high"].iloc[-self.entry :].max()
        return not close.iloc[-1] < prior["low"].iloc[-self.exit :].min()


def rsi(close: pd.Series, length: int) -> float:
    """Wilder's RSI of the last value."""
    delta = close.diff().dropna()
    gain = delta.clip(lower=0).ewm(alpha=1 / length, adjust=False).mean().iloc[-1]
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / length, adjust=False).mean().iloc[-1]
    if loss == 0:
        return 100.0
    return 100 - 100 / (1 + gain / loss)


class Rsi2MeanReversion(_PerSymbolSwitch):
    """Buy short-term dips (RSI(2) < 10) in an uptrend (above 200-day MA)."""

    name = "rsi2"
    param_grid = {"entry": [5, 10], "exit": [60, 70]}

    def __init__(self, symbols, length: int = 2, entry: float = 10, exit: float = 70, trend: int = 200):
        super().__init__(symbols)
        self.length, self.entry, self.exit, self.trend = length, entry, exit, trend
        self.lookback = trend

    def signal(self, close, bars, holding):
        r = rsi(close.iloc[-50:], self.length)
        if not holding:
            return r < self.entry and close.iloc[-1] > close.mean()
        return r < self.exit


class BollingerFade(_PerSymbolSwitch):
    """Buy a close below the lower band, sell back at the middle band."""

    name = "bollinger"
    param_grid = {"length": [20, 40], "width": [1.5, 2.0]}

    def __init__(self, symbols, length: int = 20, width: float = 2.0):
        super().__init__(symbols)
        self.length, self.width, self.lookback = length, width, length

    def signal(self, close, bars, holding):
        window = close.iloc[-self.length :]
        mid, sd = window.mean(), window.std(ddof=0)
        if not holding:
            return close.iloc[-1] < mid - self.width * sd
        return close.iloc[-1] < mid


class Momentum12_1(Strategy):
    """Monthly: hold the top-k symbols by 12-month return, skipping the last month.

    ``buffer`` above 1 keeps a holding while it still ranks in the top
    ``buffer`` x ``top_k``, so a name that slips from 15th to 17th isn't sold
    and bought back: the same idea, far fewer fees.
    """

    name = "momentum"
    param_grid = {"lookback": [126, 252], "top_k": [3, 15]}

    def __init__(self, symbols, lookback: int = 252, skip: int = 21, top_k: int = 3, buffer: float = 1.0):
        super().__init__(symbols)
        self.lookback, self.skip, self.top_k, self.buffer = lookback, skip, top_k, buffer

    def reset(self) -> None:
        self.month = None
        self.held: list[str] = []

    def decide(self, view):
        month = view.clock.tz_convert(None).to_period("M")
        if month == self.month:
            return None
        scores = {}
        for s in self.symbols:
            c = view.close(s, lookback=self.lookback + 1)
            if len(c) > self.lookback:
                scores[s] = c.iloc[-1 - self.skip] / c.iloc[0] - 1
        if len(scores) < self.top_k:
            return None  # wait until enough symbols have a full lookback
        self.month = month
        ranked = [s for s, v in sorted(scores.items(), key=lambda kv: -kv[1]) if v > 0]
        if self.buffer > 1:
            zone = set(ranked[: int(self.top_k * self.buffer)])
            winners = [s for s in self.held if s in zone][: self.top_k]
            winners += [s for s in ranked if s not in winners][: self.top_k - len(winners)]
        else:
            winners = ranked[: self.top_k]
        self.held = winners
        return {s: 1.0 / self.top_k for s in winners}


class MomentumKeep(Momentum12_1):
    """12-1 momentum that keeps a holding while it stays in the top 1.5 x k (lower turnover)."""

    name = "momentum_keep"
    param_grid = {"lookback": [126, 252], "buffer": [1.5, 2.0]}

    def __init__(self, symbols, lookback: int = 252, skip: int = 21, top_k: int = 15, buffer: float = 1.5):
        super().__init__(symbols, lookback, skip, top_k, buffer)


class VolTarget(Strategy):
    """Wrap a strategy and scale its book toward an annual volatility target."""

    def __init__(self, inner: Strategy, target: float = 0.10, window: int = 20):
        super().__init__(inner.symbols)
        self.inner, self.target, self.window = inner, target, window
        self.name = f"{inner.name}+vol{int(target * 100)}"
        self.uses_llm = inner.uses_llm

    def reset(self) -> None:
        self.inner.reset()
        self.base: dict[str, float] = {}

    def decide(self, view):
        w = self.inner.decide(view)
        if w is not None:
            self.base = w
        if not self.base:
            return w
        rets = pd.DataFrame({s: view.close(s, self.window + 1).pct_change() for s in self.base}).dropna()
        if len(rets) < self.window:
            return w
        port = rets.mul(pd.Series(self.base)).sum(axis=1)
        realized = port.std(ddof=1) * np.sqrt(252)
        scale = min(1.0, self.target / realized) if realized > 0 else 1.0
        return {s: v * scale for s, v in self.base.items()}


class TrendQuality(Strategy):
    """Position-trading rule for a screened universe, held for weeks to months.

    Ranks names that are above their 200-day average by risk-adjusted
    6-month momentum (return divided by volatility, skipping the latest
    week), holds the top ``top_k``, and sizes them by inverse volatility so a
    jumpy stock does not dominate the book. Decides on rebalance days only.
    """

    name = "uk_trend"
    param_grid = {"lookback": [126, 252], "top_k": [15, 25]}
    buffer = 1.5  # a holding stays while it ranks in the top buffer x top_k (cuts churn and fees)

    def __init__(self, symbols, lookback: int = 126, skip: int = 5, top_k: int = 20, trend: int = 200,
                 buffer: float = 1.5):
        super().__init__(symbols)
        self.lookback, self.skip, self.top_k, self.trend = lookback, skip, top_k, trend
        self.buffer = buffer

    def reset(self) -> None:
        self.held: list[str] = []

    def decide(self, view):
        need = max(self.lookback, self.trend) + 1
        scores, vols = {}, {}
        for s in self.symbols:
            c = view.close(s, lookback=need)
            if len(c) < need:
                continue
            if c.iloc[-1] <= c.iloc[-self.trend :].mean():
                continue
            window = c.iloc[-self.lookback - 1 : len(c) - self.skip]
            r = window.pct_change().dropna()
            vol = r.std(ddof=1) * np.sqrt(252)
            if vol <= 0:
                continue
            ret = window.iloc[-1] / window.iloc[0] - 1
            if ret > 0:
                scores[s], vols[s] = ret / vol, vol
        if not scores:
            self.held = []
            return {}
        ranked = sorted(scores, key=lambda s: -scores[s])
        keep_zone = set(ranked[: int(self.top_k * self.buffer)])
        winners = [s for s in getattr(self, "held", []) if s in keep_zone][: self.top_k]
        winners += [s for s in ranked if s not in winners][: self.top_k - len(winners)]
        self.held = winners
        inv = {s: 1 / vols[s] for s in winners}
        total = sum(inv.values())
        slots = len(winners) / self.top_k  # stay partly in cash when few names qualify
        return {s: slots * v / total for s, v in inv.items()}


class TrendVolTarget(Strategy):
    """Daily time-series momentum with volatility targeting.

    A common rule for commodities and crypto: hold a symbol while its return
    over ``lookback`` days is positive, sized so each holding runs at about
    ``target_vol`` annualised volatility (never more than 1/n of the book).
    It decides every day but only trades when a signal flips or a size has
    drifted more than ``band`` from its target, so costs stay contained.
    """

    name = "tsmom"
    param_grid = {"lookback": [20, 60, 120], "target_vol": [0.2, 0.4]}

    def __init__(self, symbols, lookback: int = 60, target_vol: float = 0.3, vol_window: int = 20,
                 band: float = 0.25):
        super().__init__(symbols)
        self.lookback, self.target_vol, self.vol_window, self.band = lookback, target_vol, vol_window, band

    def reset(self) -> None:
        self.current: dict[str, float] = {}

    def decide(self, view):
        from silo.metrics import periods_per_year

        n = len(self.symbols)
        target = {}
        for s in self.symbols:
            c = view.close(s, lookback=max(self.lookback, self.vol_window) + 1)
            if len(c) <= max(self.lookback, self.vol_window):
                continue
            if c.iloc[-1] <= c.iloc[-1 - self.lookback]:
                continue
            r = c.pct_change().dropna().iloc[-self.vol_window :]
            vol = r.std(ddof=1) * np.sqrt(periods_per_year(c.index))
            if vol > 0:
                target[s] = min(1.0, self.target_vol / vol) / n
        same_names = set(target) == set(self.current)
        drift = any(abs(target[s] - self.current[s]) > self.band * self.current[s] for s in target) if same_names else True
        if same_names and not drift:
            return None
        self.current = target
        return dict(target)
