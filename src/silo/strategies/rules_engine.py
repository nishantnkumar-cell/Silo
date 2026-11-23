"""The rules engine: one deterministic strategy that picks, sizes and exits every trade.

Silo's direction from 2026-11-23: written rules decide; AI only summarises and
agrees or flags. This module is the "rules decide" half. The same data always
gives the same decisions, and every entry and exit records the rule that fired
in ``last_notes`` (written to the journal by the engine).

Rules, numbered as in the direction doc (research only, simulated money):

Universe (weekly)
  R1-R4  Rank liquid names by 12-1 momentum, only names above their 200-day
         average and with no one-day move bigger than ``jump_limit`` in the last
         ``jump_days`` (Clenow's gap filter). The top ``shortlist`` are buyable.
         Liquidity rules are applied by the engine afterwards, as for any book.
Entry (weekly)
  R5     Market filter: no new buys while the market (``market`` symbol, or an
         equal-weight index of the universe) closes below its ``market_ma``-day
         average. Held names are not sold by this rule.
  R6     Buy shortlist names in rank order until ``max_names`` are held.
  R7     Room for 1:4: skip a name whose nearest fresh daily supply zone is
         closer than ``room_r`` times the stop distance (``None`` turns it off).
  R8     Skip a name stopped out in the last ``cooloff`` trading days.
Sizing
  R9     Risk ``risk_per_trade`` of the book per trade (default 1%, at most 5%):
         weight = risk / stop distance as a fraction of price.
  R10    At most ``max_weight`` in one name, and new buys stop once the open
         risk of the book (sum of weight x distance to stop) would pass
         ``max_open_risk``.
  R10b   Optional volatility scaling: all weights x min(1, ``vol_target`` /
         market volatility over 20 days). Off when ``vol_target`` is None.
Exits (every daily close, fill next open)
  R12    Initial stop ``stop_atr`` average daily ranges below the entry close (1R).
  R13    A close at or below the stop sells the whole position.
  R14    The first time the high reaches entry + ``partial_at`` R, sell
         ``partial_frac`` and move the stop to entry (``partial_frac=0`` = off).
  R15    Then trail the stop ``trail_atr`` ranges below the highest close.
  R16    Weekly: sell a holding that drops out of the top ``keep_rank``.
  C6     Sell a holding after a profit warning (or "below expectations"
         results) and block it for ``warning_block`` trading days. Needs the
         news context table; with none stored the rule does nothing.

Rules C3/C7 (blocking buys or cutting risk before FOMC and Bank of England
decisions) were dropped after the 2026-11-23 research: holding cash in the two
days before FOMC cut the US market's return from 10.5% to 7.1% a year.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from silo.firewall import PointInTimeView
from silo.strategies.base import Strategy
from silo.strategies.managed import Position, avg_range

MAX_RISK_PER_TRADE = 0.05  # the user may choose up to 5%; the engine refuses more


class RulesEngine(Strategy):
    name = "engine"
    param_grid = {"stop_atr": [3.0, 4.0, 5.0], "partial_frac": [0.0, 0.5]}

    def __init__(self, symbols, risk_per_trade: float = 0.01, max_open_risk: float = 0.10,
                 max_weight: float = 0.10, max_names: int = 15, shortlist: int = 30, keep_rank: int = 60,
                 lookback: int = 252, skip: int = 21, trend: int = 200,
                 market: str | None = None, market_ma: int = 200, market_filter: bool = True,
                 vol_target: float | None = None, jump_limit: float | None = 0.15, jump_days: int = 90,
                 room_r: float | None = None, stop_atr: float = 3.0, partial_at: float = 2.0,
                 partial_frac: float = 0.5, trail_atr: float = 3.0, atr_days: int = 20, cooloff: int = 20,
                 warning_exit: bool = True, warning_block: int = 60):
        super().__init__(symbols)
        if not 0 < risk_per_trade <= MAX_RISK_PER_TRADE:
            raise ValueError(f"risk_per_trade must be above 0 and at most {MAX_RISK_PER_TRADE:.0%}")
        if not 0 < max_weight <= 1 or not 0 < max_open_risk <= 1:
            raise ValueError("max_weight and max_open_risk must be between 0 and 1")
        if not 0 <= partial_frac < 1:
            raise ValueError("partial_frac must be in [0, 1)")
        self.risk_per_trade, self.max_open_risk, self.max_weight = risk_per_trade, max_open_risk, max_weight
        self.max_names, self.shortlist, self.keep_rank = max_names, shortlist, max(keep_rank, shortlist)
        self.lookback, self.skip, self.trend = lookback, skip, trend
        self.market, self.market_ma, self.market_filter = market, market_ma, market_filter
        self.vol_target, self.jump_limit, self.jump_days, self.room_r = vol_target, jump_limit, jump_days, room_r
        self.stop_atr, self.partial_at, self.partial_frac = stop_atr, partial_at, partial_frac
        self.trail_atr, self.atr_days, self.cooloff = trail_atr, atr_days, cooloff
        self.warning_exit, self.warning_block = warning_exit, warning_block
        self.tradable = [s for s in self.symbols if s != market]
        self.reset()

    def reset(self) -> None:
        self.book: dict[str, Position] = {}
        self.weight: dict[str, float] = {}  # target weight of each holding before volatility scaling
        self.blocked: dict[str, tuple[int, str]] = {}  # symbol -> (trading days left, rule)
        self.week = None
        self.last_notes = None
        self.log: list[dict] = []

    # ----- daily exits (R12-R15)
    def manage(self, view: PointInTimeView):
        out: dict[str, float] = {}
        for s, p in list(self.book.items()):
            bars = view.bars(s, lookback=self.atr_days + 1)
            recent = bars[bars.index > p.checked] if len(bars) else bars
            if not len(recent):
                continue
            p.checked = bars.index[-1]
            price = float(bars["close"].iloc[-1])
            if float(recent["close"].min()) <= p.stop:
                rule = "R15 trailing stop" if p.partial_done else "R13 stop"
                self._exit(s, p.checked, price, rule)
                self.blocked[s] = (self.cooloff, "R8 cool-off")
                out[s] = 0.0
                continue
            p.high = max(p.high, float(recent["close"].max()))
            if (self.partial_frac > 0 and not p.partial_done and p.risk > 0
                    and float(recent["high"].max()) >= p.entry + self.partial_at * p.risk):
                p.partial_done = True
                p.stop = max(p.stop, p.entry)
                self.weight[s] *= 1.0 - self.partial_frac
                self.log.append({"date": str(p.checked), "symbol": s, "action": "partial",
                                 "rule": f"R14 sell {self.partial_frac:.0%} at {self.partial_at:g}R", "price": price})
                out[s] = 1.0 - self.partial_frac
            if p.partial_done or self.partial_frac == 0:
                rng = avg_range(bars, self.atr_days)
                if np.isfinite(rng) and p.high > p.entry:
                    p.stop = max(p.stop, p.high - self.trail_atr * rng)
        for s in list(self.blocked):  # one trading day has passed
            days, rule = self.blocked[s]
            if days <= 1:
                del self.blocked[s]
            else:
                self.blocked[s] = (days - 1, rule)
        return out or None

    # ----- weekly decisions (R1-R11, R16, C6)
    def decide(self, view: PointInTimeView):
        week = view.clock.tz_convert(None).to_period("W")
        if week == self.week:
            return None
        ranked, info = self._rank(view)
        if not info:
            return None  # not enough history yet
        self.week = week
        notes: dict[str, object] = {}
        warned = self._warnings(view) if self.warning_exit else set()

        rank = {s: i + 1 for i, s in enumerate(ranked)}
        for s in list(self.book):
            why = None
            if s in warned:
                why = "C6 profit warning"
                self.blocked[s] = (self.warning_block, "C6 profit warning")
            elif rank.get(s, 10**9) > self.keep_rank:
                why = f"R16 out of top {self.keep_rank}"
            if why:
                last = info.get(s, {}).get("price", float("nan"))
                self._exit(s, view.clock, last, why)
        risk_on, mkt_vol = self._market(view)
        notes["market"] = "R5 risk on" if risk_on else "R5 below average: no new buys"

        bought, skipped = [], {}
        if risk_on or not self.market_filter:
            open_risk = sum(self._open_risk(s, info) for s in self.book)
            for s in ranked[: self.shortlist]:
                if len(self.book) >= self.max_names:
                    break
                if s in self.book:
                    continue
                if s in self.blocked:
                    skipped[s] = self.blocked[s][1]
                    continue
                if s in warned:
                    skipped[s] = "C6 profit warning"
                    continue
                d = info[s]["dist"]
                w = min(self.risk_per_trade / d, self.max_weight)
                if open_risk + w * d > self.max_open_risk + 1e-12:
                    skipped[s] = f"R10 open risk cap {self.max_open_risk:.0%}"
                    break
                if self.room_r is not None and not self._room(view, s, info[s]):
                    skipped[s] = f"R7 supply closer than {self.room_r:g}R"
                    continue
                price = info[s]["price"]
                risk = d * price
                self.book[s] = Position(price, price - risk, risk, price, info[s]["date"], info[s]["date"])
                self.weight[s] = w
                open_risk += w * d
                bought.append(s)
                self.log.append({"date": str(info[s]["date"]), "symbol": s, "action": "buy",
                                 "rule": f"R6 rank {rank[s]}", "weight": round(w, 4), "stop": round(price - risk, 4)})
        scale = 1.0
        if self.vol_target is not None and mkt_vol and np.isfinite(mkt_vol) and mkt_vol > 0:
            scale = min(1.0, self.vol_target / mkt_vol)
            notes["vol_scale"] = round(scale, 3)
        weights = {s: self.weight[s] * scale for s in self.book}
        total = sum(weights.values())
        if total > 1.0:
            weights = {s: w / total for s, w in weights.items()}
        if bought:
            notes["bought"] = {s: f"R6 rank {rank[s]}" for s in bought}
        if skipped:
            notes["skipped"] = skipped
        exits = [e for e in self.log if e["action"] == "exit" and e["date"] == str(view.clock)]
        if exits:
            notes["sold"] = {e["symbol"]: e["rule"] for e in exits}
        self.last_notes = notes
        return weights

    # ----- helpers
    def _rank(self, view):
        need = max(self.lookback, self.trend) + 1
        scores, info = {}, {}
        for s in self.tradable:
            bars = view.bars(s, lookback=max(need, self.jump_days + 1))
            if len(bars) < need:
                continue
            c = bars["close"]
            price = float(c.iloc[-1])
            if price <= 0:
                continue
            rng = avg_range(bars, self.atr_days)
            if not np.isfinite(rng) or rng <= 0:
                continue
            info[s] = {"price": price, "dist": self.stop_atr * rng / price, "date": bars.index[-1]}
            if price <= float(c.iloc[-self.trend:].mean()):
                continue  # R2: only names in an uptrend
            if self.jump_limit is not None:
                moves = c.iloc[-self.jump_days - 1:].pct_change().abs()
                if float(moves.max()) > self.jump_limit:
                    continue  # R3: a one-day jump this big is news, not trend
            mom = float(c.iloc[-1 - self.skip] / c.iloc[-1 - self.lookback] - 1)
            if mom > 0:
                scores[s] = mom
        ranked = sorted(scores, key=lambda s: (-scores[s], s))
        return ranked, info

    def _market(self, view) -> tuple[bool, float]:
        if self.market and self.market in view.symbols:
            c = view.close(self.market, lookback=max(self.market_ma, 21) + 1)
        else:
            from silo.agents.crossasset import market_index

            c = market_index(view, self.tradable, lookback=max(self.market_ma, 21) + 1)
        if len(c) < self.market_ma:
            return True, float("nan")  # too little history to judge: the filter stays out of the way
        on = float(c.iloc[-1]) > float(c.iloc[-self.market_ma:].mean())
        r = np.log(c.iloc[-21:].astype(float)).diff().dropna()
        vol = float(r.std(ddof=1) * np.sqrt(252)) if len(r) > 2 else float("nan")
        return on, vol

    def _open_risk(self, s, info) -> float:
        p = self.book[s]
        price = info.get(s, {}).get("price", p.entry)
        return self.weight.get(s, 0.0) * max(0.0, (price - p.stop) / price) if price > 0 else 0.0

    def _room(self, view, s, row) -> bool:
        from silo.strategies.core_zones import Bars, find_zones

        bars = view.bars(s, lookback=260)
        price, risk = row["price"], row["dist"] * row["price"]
        above = [z.proximal for z in find_zones(Bars.from_frame(bars))
                 if z.kind == "supply" and not z.used_up and z.proximal > price]
        return not above or min(above) - price >= self.room_r * risk

    def _warnings(self, view) -> set[str]:
        if not view.context_names("ctx:news"):
            return set()
        from silo.agents import news

        names = sorted(set(self.book) | set(self.tradable))
        cols = news.columns(view, names)
        return {s for s, v in cols["warning"].items() if v}

    def _exit(self, s, when, price, rule) -> None:
        self.log.append({"date": str(when), "symbol": s, "action": "exit", "rule": rule, "price": price})
        self.book.pop(s, None)
        self.weight.pop(s, None)
