"""Supply and demand zones, scored with odds enhancers, as testable rules.

This turns the supply-and-demand method from a classroom trading manual the
user studied into code, so Silo can check whether it holds up on UK shares
held for weeks to months. The rules are paraphrased in
``docs/core-strategy-rulebook.md``; where the manual relies on a trader's eye
the choices made here are listed there as "interpretations".

The method in one paragraph: prices turn where buyers and sellers were badly
out of balance. On a chart that shows up as a short pause (a *base* of small
candles) followed by a strong move away (a *leg*). The base is a *zone*:
demand if price left it upwards, supply if downwards. A zone is scored out of
10 on six *odds enhancers*: how strongly price left (strength, 2), how short
the pause was (time, 1), whether price has been back since (freshness, 2),
whether the zone agrees with the weekly trend (trend, 2), where it sits in the
monthly range (curve, 1) and how far it is to the next opposing zone compared
with the risk (profit zone, 2). Buy at a high-scoring demand zone, put the
stop just beyond the far edge, and take profit before the next supply zone.

Time frames follow the manual's settings for trades lasting weeks to months:
monthly bars for the big range ("curve"), weekly bars for the trend, daily
bars for entry and exit zones.

Everything is computed from the point-in-time view, so the strategy passes
the truncation-invariance test like every other strategy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from silo.firewall import PointInTimeView
from silo.strategies.base import Strategy

# --------------------------------------------------------------------------- candles


@dataclass
class Bars:
    """OHLC arrays for one time frame, oldest first."""

    o: np.ndarray
    h: np.ndarray
    l: np.ndarray  # noqa: E741 - low
    c: np.ndarray
    index: pd.Index

    def __len__(self) -> int:
        return len(self.c)

    @classmethod
    def from_frame(cls, df: pd.DataFrame) -> Bars:
        return cls(df["open"].to_numpy(float), df["high"].to_numpy(float), df["low"].to_numpy(float),
                   df["close"].to_numpy(float), df.index)


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Daily bars to weekly ("W") or monthly ("M") bars. The last period may be partial."""
    if df.empty:
        return df[["open", "high", "low", "close"]].iloc[:0]
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_convert(None)
    if rule == "W":  # Monday of the week, as day number
        key = (idx.normalize().to_numpy("datetime64[D]").astype(np.int64) + 3) // 7
    elif rule == "M":
        key = idx.year.to_numpy() * 12 + idx.month.to_numpy()
    else:
        raise ValueError(f"unknown rule {rule!r}")
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    ends = np.r_[starts[1:], len(key)] - 1
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))  # noqa: E741
    return pd.DataFrame({"open": o[starts], "high": np.maximum.reduceat(h, starts),
                         "low": np.minimum.reduceat(l, starts), "close": c[ends]}, index=df.index[ends])


def candle_kinds(b: Bars, base_body: float = 0.5) -> np.ndarray:
    """+1 rally leg, -1 drop leg, 0 base candle.

    A *base* candle is a small-bodied, neutral candle (body at most half its
    range). A *leg* candle has a body bigger than half its range. A gap counts
    toward the body: the manual colours in the space between the prior close
    and the gapping candle's close, so the "body" runs from the prior close
    when that is larger.
    """
    n = len(b)
    prev_c = np.r_[b.o[0], b.c[:-1]]
    body_own = np.abs(b.c - b.o)
    body_gap = np.abs(b.c - prev_c)
    use_gap = body_gap > body_own
    start = np.where(use_gap, prev_c, b.o)
    body = np.abs(b.c - start)
    hi = np.maximum(b.h, np.where(use_gap, prev_c, b.h))
    lo = np.minimum(b.l, np.where(use_gap, prev_c, b.l))
    rng = np.maximum(hi - lo, 1e-12)
    kind = np.zeros(n, dtype=int)
    leg = body > base_body * rng
    kind[leg & (b.c > start)] = 1
    kind[leg & (b.c < start)] = -1
    return kind


def atr(b: Bars, n: int = 14) -> float:
    if len(b) < 2:
        return float(np.nan)
    prev = np.r_[b.c[0], b.c[:-1]]
    tr = np.maximum(b.h, prev) - np.minimum(b.l, prev)
    return float(np.mean(tr[-n:]))


# --------------------------------------------------------------------------- zones


@dataclass
class Zone:
    kind: str  # "demand" | "supply"
    formation: str  # DBR, RBR, RBD, DBD
    proximal: float  # edge nearest price: entries go here
    distal: float  # far edge: the stop goes beyond it
    base_start: int  # bar positions in the time frame it was found on
    base_end: int
    leg_out_end: int
    n_base: int
    move_out: float  # (move-out extreme - proximal) / zone height
    breakout: bool  # the move out crossed the previous opposing swing
    penetration: float = 0.0  # deepest return into the zone before the current visit, 0..1
    violated: bool = False  # price closed beyond the distal line: zone disqualified
    in_zone_now: bool = False  # the latest bar traded into the zone

    @property
    def height(self) -> float:
        return abs(self.proximal - self.distal)

    # --- odds enhancers 1-3: zone structure
    @property
    def strength(self) -> float:
        """2 if price left strongly (>= 2x the zone height) *and* broke a prior swing, 1 if one, 0 if neither."""
        return float((self.move_out >= 2.0) + bool(self.breakout))

    @property
    def time(self) -> float:
        """1 for 1-3 basing candles, 0.5 for 4-6, 0 for more."""
        return 1.0 if self.n_base <= 3 else 0.5 if self.n_base <= 6 else 0.0

    @property
    def freshness(self) -> float:
        """2 if never tested, 1 if tested up to half way, 0 if deeper (but not through the distal line)."""
        if self.penetration <= 0:
            return 2.0
        return 1.0 if self.penetration <= 0.5 else 0.0

    @property
    def used_up(self) -> bool:
        """Price has already traded through the far line (a wick, not a close): the orders are gone.

        Not a violation in the manual's sense, but such a zone no longer counts
        as the opposing zone that caps a trade's profit.
        """
        return self.violated or self.penetration >= 1.0

    @property
    def structure(self) -> float:
        return self.strength + self.time + self.freshness


def swings(b: Bars, k: int = 2) -> tuple[list[int], list[int]]:
    """Swing highs and lows: a bar higher (lower) than the k bars each side.

    A swing is only known k bars after it happens; callers only see bars up to
    the clock, so the last k bars can never be swings yet (no look-ahead).
    """
    n = len(b)
    if n < 2 * k + 1:
        return [], []
    from numpy.lib.stride_tricks import sliding_window_view as win

    wh, wl = win(b.h, 2 * k + 1), win(b.l, 2 * k + 1)
    mid_h, mid_l = b.h[k:n - k], b.l[k:n - k]
    is_h = (mid_h >= wh.max(axis=1)) & (wh[:, :k] < mid_h[:, None]).all(axis=1)
    is_l = (mid_l <= wl.min(axis=1)) & (wl[:, :k] > mid_l[:, None]).all(axis=1)
    return list(np.flatnonzero(is_h) + k), list(np.flatnonzero(is_l) + k)


def find_zones(b: Bars, max_base: int = 10, kinds: np.ndarray | None = None) -> list[Zone]:
    """Every leg-base-leg formation in the bars, scored for structure and freshness.

    Going left to right: a leg candle (leg-in), 1 to ``max_base`` base
    candles, then a leg candle (leg-out). The leg-out's direction makes it
    demand (up) or supply (down); the leg-in's makes it a continuation (RBR,
    DBD) or a reversal (DBR, RBD). Line placement follows the manual's
    preferred setting: proximal at the base's highest body (demand) or lowest
    body (supply); distal at the formation's extreme low (high), including the
    leg-in for reversals.
    """
    n = len(b)
    if n < 4:
        return []
    kind = candle_kinds(b) if kinds is None else kinds
    sw_hi, sw_lo = swings(b)
    body_hi = np.maximum(b.o, b.c)
    body_lo = np.minimum(b.o, b.c)
    zones: list[Zone] = []
    i = 1
    while i < n - 1:
        if kind[i] != 0 or kind[i - 1] == 0:
            i += 1
            continue
        j = i
        while j + 1 < n and kind[j + 1] == 0 and j - i + 1 < max_base:
            j += 1
        if j + 1 >= n or kind[j + 1] == 0:  # no leg-out (yet), or base too long
            i = j + 1
            continue
        leg_in, out = kind[i - 1], kind[j + 1]
        # move out: consecutive closes in the leg-out's direction
        k = j + 1
        while k + 1 < n and (b.c[k + 1] - b.c[k]) * out > 0:
            k += 1
        if out > 0:
            prox = body_hi[i:j + 1].max()
            lo_from = i - 1 if leg_in < 0 else i
            dist = b.l[lo_from:k + 1].min()
            extreme = b.h[j + 1:k + 1].max()
            if not (extreme > b.h[i - 1:j + 1].max()):  # must make a new high: an explosive rally
                i = j + 1
                continue
            prior = [s for s in sw_hi if s < i - 1]
            breakout = bool(prior) and extreme > b.h[prior[-1]] and b.h[prior[-1]] > prox
            form = "DBR" if leg_in < 0 else "RBR"
            kind_name = "demand"
        else:
            prox = body_lo[i:j + 1].min()
            hi_from = i - 1 if leg_in > 0 else i
            dist = b.h[hi_from:k + 1].max()
            extreme = b.l[j + 1:k + 1].min()
            if not (extreme < b.l[i - 1:j + 1].min()):
                i = j + 1
                continue
            prior = [s for s in sw_lo if s < i - 1]
            breakout = bool(prior) and extreme < b.l[prior[-1]] and b.l[prior[-1]] < prox
            form = "RBD" if leg_in > 0 else "DBD"
            kind_name = "supply"
        height = abs(prox - dist)
        if height <= 0:
            i = j + 1
            continue
        move = abs(extreme - prox) / height
        z = Zone(kind_name, form, float(prox), float(dist), i, j, k, j - i + 1, float(move), bool(breakout))
        _freshness(z, b, k + 1)
        zones.append(z)
        i = j + 1
    return zones


def _freshness(z: Zone, b: Bars, start: int) -> None:
    """How deep price came back into the zone after leaving it, before the current visit.

    A close beyond the distal line violates the zone. The current visit (the
    run of latest bars that trade into the zone) is the trade being planned,
    so it does not count against freshness.
    """
    n = len(b)
    if start >= n:
        return
    if z.kind == "demand":
        inside = b.l[start:] <= z.proximal
        if (b.c[start:] < z.distal).any():
            z.violated = True
            return
    else:
        inside = b.h[start:] >= z.proximal
        if (b.c[start:] > z.distal).any():
            z.violated = True
            return
    m = len(inside)
    cur = m
    while cur > 0 and inside[cur - 1]:
        cur -= 1
    z.in_zone_now = cur < m
    before = slice(start, start + cur)
    if cur == 0:
        return
    if z.kind == "demand":
        deepest = b.l[before].min()
        z.penetration = float(max(0.0, (z.proximal - deepest) / z.height))
    else:
        deepest = b.h[before].max()
        z.penetration = float(max(0.0, (deepest - z.proximal) / z.height))


# --------------------------------------------------------------------------- trend


@dataclass
class Trend:
    direction: int  # +1 up, 0 sideways, -1 down
    controlling: float | None  # controlling swing low (uptrend) or high (downtrend)
    last_swing_high: float | None
    last_swing_low: float | None

    @property
    def label(self) -> str:
        return {1: "up", 0: "sideways", -1: "down"}[self.direction]


def trend(b: Bars, k: int = 2) -> Trend:
    """Higher highs and higher lows (up), lower highs and lower lows (down), else sideways.

    Walks the bars in order with the manual's rules: an uptrend starts when
    price breaks above the last swing high after making a higher (or equal)
    low; each new break higher makes the latest swing low the *controlling*
    low; a close below the controlling low ends the uptrend (back to a range).
    Downtrends mirror this. Swings need ``k`` bars either side, so they are
    only used once confirmed.
    """
    sw_hi, sw_lo = swings(b, k)
    state, ctrl = 0, None
    hi_i = lo_i = 0  # how many swings are confirmed by bar t
    for t in range(len(b)):
        while hi_i < len(sw_hi) and sw_hi[hi_i] + k <= t:
            hi_i += 1
        while lo_i < len(sw_lo) and sw_lo[lo_i] + k <= t:
            lo_i += 1
        if not hi_i or not lo_i:
            continue
        last_hi, last_lo = b.h[sw_hi[hi_i - 1]], b.l[sw_lo[lo_i - 1]]
        close = b.c[t]
        if state == 1 and close < ctrl:
            state, ctrl = 0, None
        elif state == -1 and close > ctrl:
            state, ctrl = 0, None
        if close > last_hi:
            if state == 1:
                ctrl = last_lo
            elif lo_i >= 2 and last_lo >= b.l[sw_lo[lo_i - 2]]:
                state, ctrl = 1, last_lo
        elif close < last_lo:
            if state == -1:
                ctrl = last_hi
            elif hi_i >= 2 and last_hi <= b.h[sw_hi[hi_i - 2]]:
                state, ctrl = -1, last_hi
    last_h = float(b.h[sw_hi[-1]]) if sw_hi else None
    last_l = float(b.l[sw_lo[-1]]) if sw_lo else None
    return Trend(state, None if ctrl is None else float(ctrl), last_h, last_l)


# --------------------------------------------------------------------------- curve


@dataclass
class Curve:
    low: float  # distal of the nearest higher-time-frame demand below price
    high: float  # distal of the nearest higher-time-frame supply above price
    known: bool  # False when there was not enough history to find both edges

    def location(self, price: float) -> float:
        """0 = bottom of the range (wholesale), 1 = top (retail)."""
        if self.high <= self.low:
            return 0.5
        return float((price - self.low) / (self.high - self.low))


def curve(htf: Bars, price: float, min_bars: int = 24) -> Curve:
    """The higher-time-frame range: nearest quality demand below and supply above, distal to distal."""
    if len(htf) < min_bars:
        return Curve(float(np.nanmin(htf.l)) if len(htf) else price, float(np.nanmax(htf.h)) if len(htf) else price,
                     False)
    zones = [z for z in find_zones(htf) if not z.violated and z.strength >= 1]
    below = [z for z in zones if z.kind == "demand" and z.proximal <= price]
    above = [z for z in zones if z.kind == "supply" and z.proximal >= price]
    low = max((z.distal for z in below), default=float(np.min(htf.l)))
    high = min((z.distal for z in above), default=float(np.max(htf.h)))
    return Curve(low, high, bool(below and above))


# --------------------------------------------------------------------------- scoring


@dataclass
class Setup:
    """A scored long setup at one demand zone, with its stop, entry and target (S.E.T.)."""

    zone: Zone
    strength: float
    time: float
    freshness: float
    trend: float
    curve: float
    profit_zone: float
    trend_label: str
    curve_location: float
    entry: float
    stop: float
    target: float | None
    reward_risk: float
    decision: str  # matrix outcome: "long", "long_if_5to1" or "no_action"
    notes: list[str] = field(default_factory=list)

    @property
    def score(self) -> float:
        return self.strength + self.time + self.freshness + self.trend + self.curve + self.profit_zone

    @property
    def entry_type(self) -> str:
        """Manual's guidance: 8.5+ proximal (limit) entry, 7-8 wait for confirmation, below 7 no trade."""
        if self.score >= 8.5:
            return "proximal"
        if self.score >= 7.0:
            return "confirmation"
        return "none"

    @property
    def allowed(self) -> bool:
        if self.decision == "no_action":
            return False
        if self.decision == "long_if_5to1" and self.reward_risk < 5:
            return False
        return self.entry_type != "none"


def decision_matrix(curve_loc: float, trend_dir: int) -> str:
    """The manual's curve x trend table, for demand zones (Silo is long-only).

    Low on the curve: long in any trend (against a downtrend only with 5:1).
    Middle: long in a sideways or up trend; no action in a downtrend.
    High: long only in an uptrend and only with 5:1; otherwise no action.
    """
    third = 0 if curve_loc < 1 / 3 else 1 if curve_loc <= 2 / 3 else 2
    if third == 0:
        return "long_if_5to1" if trend_dir < 0 else "long"
    if third == 1:
        return "no_action" if trend_dir < 0 else "long"
    return "long_if_5to1" if trend_dir > 0 else "no_action"


def score_demand(z: Zone, ltf: Bars, zones: list[Zone], tr: Trend, cv: Curve, htf_supply: list[Zone],
                 stop_buffer: float, price: float) -> Setup:
    """Score one demand zone as a long setup, out of 10."""
    notes = []
    # 4. trend (weekly): with the trend 2, sideways 1, against 0; with-trend zones must sit above the controlling low
    if tr.direction > 0:
        t_score = 2.0 if tr.controlling is None or z.distal >= tr.controlling * 0.98 else 1.0
    else:
        t_score = 1.0 if tr.direction == 0 else 0.0
    # 5. curve (monthly): low third 1, middle 0.5, high 0
    loc = cv.location(z.proximal)
    c_score = 1.0 if loc < 1 / 3 else 0.5 if loc <= 2 / 3 else 0.0
    if not cv.known:
        notes.append("curve edge estimated from the price range (not enough monthly zones)")
    # 6. profit zone: distance to the nearest opposing supply over the risk
    entry = z.proximal
    stop = z.distal - stop_buffer
    risk = entry - stop
    above = [s.proximal for s in zones if s.kind == "supply" and not s.used_up and s.proximal > max(entry, price)]
    target_zone = min(above) if above else None
    if target_zone is None:
        htf_above = [s.proximal for s in htf_supply if s.proximal > max(entry, price)]
        target_zone = min(htf_above) if htf_above else None
        if target_zone is not None:
            notes.append("target from the monthly supply zone (no daily supply above)")
    if target_zone is None:
        rr = float("inf")
        p_score = 2.0
        notes.append("no supply overhead: profit zone open")
    else:
        rr = (target_zone - entry) / risk if risk > 0 else 0.0
        p_score = 2.0 if rr >= 5 else 1.0 if rr >= 3 else 0.0
    return Setup(z, z.strength, z.time, z.freshness, t_score, c_score, p_score, tr.label, loc, entry, stop,
                 target_zone, rr, decision_matrix(loc, tr.direction), notes)


@dataclass
class Analysis:
    """What the method sees in one stock at one moment."""

    setups: list[Setup]  # active demand zones below or around price, best first
    supply_above: list[Zone]
    trend: Trend
    curve: Curve
    atr: float
    price: float

    @property
    def best(self) -> Setup | None:
        return self.setups[0] if self.setups else None


def analyze(daily: pd.DataFrame, ltf_lookback: int = 260, itf_lookback: int = 780, htf_lookback: int = 2520,
            stop_atr: float = 0.10, htf_cache: dict | None = None, cache_key=None) -> Analysis | None:
    """Run the 6-step plan's analysis on one stock's daily bars (already point-in-time)."""
    if len(daily) < 120:
        return None
    ltf = Bars.from_frame(daily.iloc[-ltf_lookback:])
    price = float(ltf.c[-1])
    a = atr(ltf, 14)
    if not np.isfinite(a) or a <= 0:
        return None
    itf = Bars.from_frame(resample(daily.iloc[-itf_lookback:], "W"))
    tr = trend(itf)
    # monthly zones from completed months only, cached per month (they cannot change mid-month)
    month = pd.Timestamp(daily.index[-1]).to_period("M")
    key = (cache_key, month)
    if htf_cache is not None and key in htf_cache:
        htf_done = htf_cache[key]
    else:
        m = resample(daily.iloc[-htf_lookback:], "M")
        htf_done = Bars.from_frame(m.iloc[:-1]) if len(m) > 1 else Bars.from_frame(m.iloc[:0])
        if htf_cache is not None:
            htf_cache[key] = htf_done
    cv = curve(htf_done, price)
    htf_supply = [z for z in find_zones(htf_done) if z.kind == "supply" and not z.used_up] if len(htf_done) else []
    zones = find_zones(ltf)
    buffer = stop_atr * a
    demand = [z for z in zones if z.kind == "demand" and not z.violated and z.proximal <= price * 1.10]
    setups = [score_demand(z, ltf, zones, tr, cv, htf_supply, buffer, price) for z in demand]
    setups.sort(key=lambda s: (-s.score, -s.zone.proximal))
    supply = sorted((z for z in zones if z.kind == "supply" and not z.used_up and z.proximal > price),
                    key=lambda z: z.proximal)
    return Analysis(setups, supply, tr, cv, a, price)


# --------------------------------------------------------------------------- strategy


@dataclass
class Trade:
    symbol: str
    entry: float
    stop: float
    target: float | None
    units: float  # shadow-book units, for weights
    risk: float  # fraction of the book lost if the stop is hit
    zone_proximal: float
    opened: pd.Timestamp
    take_at: float | None = None  # managed exits: sell part of the trade here (entry + partial_at x risk)
    partial_done: bool = False
    checked: pd.Timestamp | None = None  # last close the trade was checked at


class CoreZones(Strategy):
    """Long-only supply-and-demand trader for UK shares held for weeks to months.

    Each week, for every stock: score the active daily demand zones (out of 10)
    and buy when price has come back into a zone that is allowed by the
    decision matrix and scores well enough (8.5+ buys at the zone; 7-8 waits
    for price to close back above the zone first). The stop sits 10% of a
    day's average range below the zone; the target is 75% of the way to the
    next supply zone, and must be at least 3x the risk. Each trade risks at
    most 2% of the book and all open trades together at most 6%. With
    ``exit="trail"`` there is no fixed target: the stop is raised under each
    new demand zone that forms (the manual's trailing stop) and profits run.
    With ``exit="managed"`` (see :class:`CoreManaged`) open trades are checked
    at every daily close through :meth:`manage`.

    Silo cannot place limit or stop orders: it decides at the close and fills
    at the next open, and by default only decides weekly. So entries, stops
    and targets are acted on at the next decision, which costs some slippage
    against the manual's intent. The report says so.
    """

    name = "core_zones"
    param_grid = {"min_score": [7.0, 8.5], "exit": ["target", "trail"]}

    def __init__(self, symbols, min_score: float = 7.0, exit: str = "target", risk_per_trade: float = 0.02,
                 max_open_risk: float = 0.06, max_weight: float = 0.10, target_frac: float = 0.75,
                 min_reward_risk: float = 3.0, stop_atr: float = 0.10, chase: float = 0.5,
                 partial_at: float = 2.0, partial_frac: float = 0.5, breakeven: bool = True):
        super().__init__(symbols)
        if exit not in ("target", "trail", "managed"):
            raise ValueError("exit must be 'target', 'trail' or 'managed'")
        self.partial_at, self.partial_frac, self.breakeven = partial_at, partial_frac, breakeven
        self.min_score, self.exit = min_score, exit
        self.risk_per_trade, self.max_open_risk, self.max_weight = risk_per_trade, max_open_risk, max_weight
        self.target_frac, self.min_reward_risk, self.stop_atr, self.chase = (
            target_frac, min_reward_risk, stop_atr, chase)

    def reset(self) -> None:
        self.trades: dict[str, Trade] = {}
        self.cash = 1.0  # shadow book in "value units", to turn held units into weights
        self.last_seen: pd.Timestamp | None = None
        self._htf: dict = {}
        self.log: list[dict] = []
        self.exited: dict[str, pd.Timestamp] = {}

    # -- helpers
    def _value(self, prices: dict[str, float]) -> float:
        return self.cash + sum(t.units * prices.get(s, t.entry) for s, t in self.trades.items())

    def _since(self, bars: pd.DataFrame) -> pd.DataFrame:
        if self.last_seen is None:
            return bars.iloc[-5:]
        return bars[bars.index > self.last_seen]

    def _trail(self, t: Trade, df: pd.DataFrame) -> None:
        """Raise the stop under the newest demand zone that formed after the trade opened."""
        if len(df) <= 120:
            return
        a = analyze(df, stop_atr=self.stop_atr, htf_cache=self._htf, cache_key=t.symbol)
        if a is None:
            return
        new = [z for z in find_zones(Bars.from_frame(df.iloc[-260:]))
               if z.kind == "demand" and not z.violated and z.strength >= 1
               and df.index[-260:][z.base_start] > t.opened]
        if new:
            t.stop = max(t.stop, max(z.distal for z in new) - self.stop_atr * a.atr)

    def manage(self, view: PointInTimeView):
        """Daily check of open trades (``exit="managed"`` only).

        At each close: a close at or below the stop sells the whole trade; the
        first time the high reaches ``partial_at`` x the risk (1:2 by default),
        ``partial_frac`` of it is sold and the stop moves up to the entry
        (``breakeven``); after that the rest trails under new demand zones.
        Orders go to the next open, like every Silo order.
        """
        if self.exit != "managed" or not self.trades:
            return None
        out: dict[str, float] = {}
        for s, t in list(self.trades.items()):
            df = view.bars(s, lookback=2520)
            if not len(df):
                continue
            recent = df[df.index > t.checked] if t.checked is not None else df.iloc[-1:]
            if not len(recent):
                continue
            now = df.index[-1]
            price = float(df["close"].iloc[-1])
            t.checked = now
            if float(recent["close"].min()) <= t.stop:
                self.cash += t.units * price
                why = "trailing stop" if t.partial_done else "stop"
                self.log.append({"date": str(now), "symbol": s, "action": "exit", "why": why, "price": price,
                                 "r": (price - t.entry) / (t.entry - t.stop) if t.entry > t.stop else 0.0})
                del self.trades[s]
                self.exited[s] = now
                out[s] = 0.0
                continue
            if not t.partial_done and t.take_at is not None and float(recent["high"].max()) >= t.take_at:
                sold = t.units * self.partial_frac
                self.cash += sold * price
                t.units -= sold
                t.partial_done = True
                t.risk = 0.0 if self.breakeven else t.risk * (1 - self.partial_frac)
                if self.breakeven:
                    t.stop = max(t.stop, t.entry)
                self.log.append({"date": str(now), "symbol": s, "action": "partial", "why": "1:%g reached"
                                 % self.partial_at, "price": price, "sold_frac": self.partial_frac})
                out[s] = 1.0 - self.partial_frac
            if t.partial_done:
                self._trail(t, df)
        return out or None

    def decide(self, view: PointInTimeView):
        changed = False
        prices: dict[str, float] = {}
        frames: dict[str, pd.DataFrame] = {}
        for s in self.symbols:
            df = view.bars(s, lookback=2520)
            if len(df):
                frames[s] = df
                prices[s] = float(df["close"].iloc[-1])
        newest = max((f.index[-1] for f in frames.values()), default=None)

        # 1. exits: a stop or target touched since the last decision (managed trades exit in manage())
        for s, t in list(self.trades.items()):
            df = frames.get(s)
            if df is None or self.exit == "managed":
                continue
            recent = self._since(df)
            if self.exit == "trail":
                self._trail(t, df)
            hit_stop = len(recent) and float(recent["close"].min()) <= t.stop
            hit_target = t.target is not None and len(recent) and float(recent["high"].max()) >= t.target
            if hit_stop or hit_target:
                self.cash += t.units * prices[s]
                self.log.append({"date": str(newest), "symbol": s, "action": "exit",
                                 "why": "stop" if hit_stop else "target", "price": prices[s]})
                del self.trades[s]
                self.exited[s] = newest
                changed = True

        # 2. entries: price back at a qualifying demand zone
        open_risk = sum(t.risk for t in self.trades.values())
        candidates = []
        for s, df in frames.items():
            if s in self.trades or len(df) < 260:
                continue
            if self.last_seen is not None and self.exited.get(s, self.last_seen) > self.last_seen:
                continue  # sold since the last decision: don't buy straight back
            a = analyze(df, stop_atr=self.stop_atr, htf_cache=self._htf, cache_key=s)
            if a is None:
                continue
            recent = self._since(df)
            for st in a.setups:
                z = st.zone
                if not st.allowed or st.score < self.min_score:
                    continue
                touched = len(recent) and float(recent["low"].min()) <= z.proximal
                if not touched:
                    continue
                price = prices[s]
                if price <= st.stop or price > z.proximal + self.chase * z.height:
                    continue  # through the stop already, or bounced too far to chase
                if st.entry_type == "confirmation" and price <= z.proximal:
                    continue  # medium score: wait for a close back above the zone
                entry = price  # Silo fills at the next open; plan from the last close
                risk = entry - st.stop
                if risk <= 0:
                    continue
                if self.exit == "managed":
                    if st.target is not None:  # the room to the next supply must pay min_reward_risk
                        room = z.proximal + self.target_frac * (st.target - z.proximal)
                        if (room - entry) / risk < self.min_reward_risk:
                            continue
                    target = None
                elif self.exit == "target":
                    if st.target is None:
                        target = entry + self.min_reward_risk * risk * 2
                    else:
                        target = z.proximal + self.target_frac * (st.target - z.proximal)
                    if (target - entry) / risk < self.min_reward_risk:
                        continue
                else:
                    target = None
                candidates.append((st.score, s, entry, st.stop, target, z.proximal))
                break
        candidates.sort(key=lambda c: (-c[0], c[1]))
        value = self._value(prices)
        for score, s, entry, stop, target, prox in candidates:
            stop_frac = (entry - stop) / entry
            weight = min(self.max_weight, self.risk_per_trade / stop_frac)
            risk = weight * stop_frac
            if open_risk + risk > self.max_open_risk + 1e-12:
                room = self.max_open_risk - open_risk
                if room <= 0.25 * risk:
                    continue
                weight, risk = room / stop_frac, room
            spend = weight * value
            if spend > self.cash + 1e-12:
                spend = self.cash
                weight = spend / value
                risk = weight * stop_frac
                if weight < 0.01:
                    continue
            self.cash -= spend
            take = entry + self.partial_at * (entry - stop) if self.exit == "managed" else None
            self.trades[s] = Trade(s, entry, stop, target, spend / entry, risk, prox, newest, take, False, newest)
            open_risk += risk
            self.log.append({"date": str(newest), "symbol": s, "action": "buy", "score": score,
                             "entry": entry, "stop": stop, "target": target})
            changed = True

        self.last_seen = newest
        if not changed:
            return None
        value = self._value(prices)
        return {s: t.units * prices[s] / value for s, t in self.trades.items()}


class CoreZonesTrail(CoreZones):
    """The same entries, but no fixed target: trail the stop under each new demand zone and let profits run."""

    name = "core_trail"
    param_grid = {"min_score": [7.0, 8.5]}

    def __init__(self, symbols, **kwargs):
        kwargs.setdefault("exit", "trail")
        super().__init__(symbols, **kwargs)


class CoreManaged(CoreZones):
    """Core Strategy entries with Nishant's trade management, checked every day.

    * Only setups with room for at least 1:4 reward-to-risk to the next supply.
    * A daily close at or below the stop sells the whole position (even inside
      the 20-day minimum hold: stops always win).
    * At 1:2, half is sold and the stop moves to the entry price.
    * The other half trails under each new demand zone until it is stopped out.
    """

    name = "core_managed"
    param_grid = {"min_score": [7.0, 8.5]}

    def __init__(self, symbols, **kwargs):
        kwargs.setdefault("exit", "managed")
        kwargs.setdefault("min_reward_risk", 4.0)
        super().__init__(symbols, **kwargs)
