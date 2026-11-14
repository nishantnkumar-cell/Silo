"""Score the signal, not only the book.

A book's equity curve is one path: two years of weekly decisions is about a
hundred data points, too few to tell skill from luck. These reports use every
share on every decision day instead, so they reach a verdict much sooner.

* :func:`pick_spread` - for any finished run: after each decision, how did
  the names the strategy *asked for* do over the next weeks, against the
  average of everything it could have chosen? Works for rules and AI alike.
* :func:`rank_ic` - for a scoring rule: each week, rank the shares by the
  score and measure how well that rank lines up with what they did next
  (Spearman rank correlation, the "information coefficient").
* :func:`event_study` - for a trade journal: the return after each entry
  over a fixed number of days, less the market over the same days, grouped
  by a feature seen at entry (for example the Core Strategy zone score).

Scores are computed through the time firewall. Forward returns are read
from the full tables: they are the exam marks, never shown to a strategy.
Returns run from the open after the decision, like Silo's fills, and ignore
costs and dividends.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from silo.firewall import PointInTimeView, PreparedTables, is_context


def _forward(tables, symbols, horizon: int) -> pd.DataFrame:
    """date x symbol: return from the next bar's open to the open ``horizon`` bars after that."""
    opens = pd.DataFrame({s: tables[s]["open"] for s in symbols if not is_context(s)}).sort_index()
    return opens.shift(-1 - horizon) / opens.shift(-1) - 1


def _tstat(x: np.ndarray, step: float, horizon: int) -> float:
    """t of the mean, with the sample shrunk for overlapping forward windows."""
    x = x[~np.isnan(x)]
    if len(x) < 3 or x.std(ddof=1) == 0:
        return float("nan")
    n_eff = max(2.0, len(x) * min(1.0, step / horizon))
    return float(x.mean() / (x.std(ddof=1) / math.sqrt(n_eff)))


def _step(dates) -> float:
    d = pd.DatetimeIndex(sorted(dates))
    return float(np.median(np.diff(d.values).astype("timedelta64[D]").astype(float))) * 5 / 7 if len(d) > 1 else 1.0


def pick_spread(decisions: list[dict], tables: dict[str, pd.DataFrame], symbols: list[str],
                horizon: int = 20) -> dict:
    """Forward return of the proposed names less the average of all names, per decision."""
    fwd = _forward(tables, symbols, horizon)
    rows = {}
    for d in decisions:
        picks = [s for s, w in (d.get("proposed") or {}).items() if w > 0 and s in fwd.columns]
        if not picks or d["date"] not in fwd.index:
            continue
        r = fwd.loc[d["date"]].dropna()
        got = r.reindex(picks).dropna()
        if len(got) and len(r) > len(got):
            rows[d["date"]] = float(got.mean() - r.mean())
    x = np.array(list(rows.values()), dtype=float)
    if not len(x):
        return {"decisions": 0, "horizon": horizon}
    return {"decisions": int(len(x)), "horizon": horizon, "mean_spread": float(x.mean()),
            "hit_rate": float(np.mean(x > 0)), "t": _tstat(x, _step(rows), horizon)}


# --- scoring rules for rank_ic: (view, symbol) -> score or None -------------------------------

def _mom(lookback: int, skip: int):
    def score(view, s):
        c = view.close(s, lookback=lookback + 1)
        return float(c.iloc[-1 - skip] / c.iloc[0] - 1) if len(c) > lookback else None
    return score


def _risk_adj_mom(view, s, lookback: int = 126, skip: int = 5):
    c = view.close(s, lookback=lookback + 1)
    if len(c) <= lookback:
        return None
    w = c.iloc[: len(c) - skip]
    vol = w.pct_change().std(ddof=1)
    return float((w.iloc[-1] / w.iloc[0] - 1) / (vol * math.sqrt(252))) if vol and vol > 0 else None


def _low_vol(view, s):
    c = view.close(s, lookback=64)
    return -float(np.log(c).diff().std(ddof=1)) if len(c) >= 64 else None


def _vs_high(view, s):
    c = view.close(s, lookback=252)
    return float(c.iloc[-1] / c.max() - 1) if len(c) >= 252 else None


def _reversal(view, s):
    c = view.close(s, lookback=22)
    return -float(c.iloc[-1] / c.iloc[0] - 1) if len(c) >= 22 else None


def _zone_score(view, s):
    from silo.strategies.core_zones import analyze

    df = view.bars(s, lookback=2520)
    if len(df) < 260:
        return None
    a = analyze(df)
    return float(a.best.score) if a is not None and a.best is not None else 0.0


SIGNALS = {
    "momentum_12_1": _mom(252, 21),
    "momentum_6_1": _mom(126, 21),
    "trend_quality": _risk_adj_mom,
    "near_52w_high": _vs_high,
    "low_volatility": _low_vol,
    "one_month_reversal": _reversal,
    "zone_score": _zone_score,  # slow: scans supply and demand zones for every share
}


def rank_ic(score, tables: dict[str, pd.DataFrame], symbols: list[str], horizon: int = 20, every: int = 5,
            start=None, end=None, min_names: int = 20, liquidity=None) -> dict:
    """Mean weekly rank correlation between ``score`` and the next ``horizon`` days' return."""
    symbols = [s for s in symbols if not is_context(s)]
    prepared = PreparedTables({s: tables[s] for s in symbols})
    fwd = _forward(tables, symbols, horizon)
    dates = [d for d in fwd.index if (start is None or d >= pd.Timestamp(start))
             and (end is None or d <= pd.Timestamp(end))][::every]
    ics, spreads, used = [], [], []
    for d in dates:
        clock = max(tables[s].loc[d, "available_at"] for s in symbols if d in tables[s].index)
        view = PointInTimeView(prepared, clock)
        sc = {}
        for s in symbols:
            if liquidity is not None and not liquidity.check(view, s)[0]:
                continue
            v = score(view, s)
            if v is not None and np.isfinite(v):
                sc[s] = v
        r = fwd.loc[d].reindex(list(sc)).dropna()
        if len(r) < min_names:
            continue
        x = pd.Series(sc).reindex(r.index)
        if x.nunique() < 3:
            continue
        ics.append(float(x.rank().corr(r.rank())))
        k = max(1, len(r) // 5)
        order = x.sort_values()
        spreads.append(float(r[order.index[-k:]].mean() - r[order.index[:k]].mean()))
        used.append(d)
    ic = np.array(ics, dtype=float)
    if not len(ic):
        return {"weeks": 0, "horizon": horizon}
    return {"weeks": int(len(ic)), "horizon": horizon, "mean_ic": float(np.nanmean(ic)),
            "ic_positive": float(np.nanmean(ic > 0)), "t": _tstat(ic, every, horizon),
            "top_minus_bottom": float(np.mean(spreads)), "first": str(used[0].date()), "last": str(used[-1].date())}


def event_study(trades, tables: dict[str, pd.DataFrame], symbols: list[str], horizons=(20, 60),
                by: str | None = None, cuts: list[float] | None = None) -> list[dict]:
    """Return after each entry less the market's, over fixed horizons, grouped by an entry feature.

    ``trades`` are journal trades (:func:`silo.learning.journal.build_journal`). Each entry
    is measured from the open it filled at, whatever the strategy did afterwards, so the
    answer is about the entry signal and not about stops, targets or position size.
    """
    symbols = [s for s in symbols if not is_context(s)]
    opens = pd.DataFrame({s: tables[s]["open"] for s in symbols}).sort_index()
    rows = []
    for h in horizons:
        fwd = opens.shift(-h) / opens - 1
        market = fwd.mean(axis=1)
        groups: dict[str, list[float]] = {}
        dates: dict[str, list] = {}
        for t in trades:
            if t.opened not in fwd.index or t.symbol not in fwd.columns:
                continue
            x = fwd.at[t.opened, t.symbol] - market.at[t.opened]
            if not np.isfinite(x):
                continue
            label = "all entries"
            keys = [label]
            if by is not None and by in t.features and cuts:
                v = t.features[by]
                cs = sorted(cuts)
                lo = max([c for c in cs if v >= c], default=None)
                hi = min([c for c in cs if v < c], default=None)
                keys.append(f"{by} below {cs[0]:g}" if lo is None else
                            f"{by} {lo:g} or more" if hi is None else f"{by} {lo:g} to {hi:g}")
            for k in keys:
                groups.setdefault(k, []).append(float(x))
                dates.setdefault(k, []).append(t.opened)
        for k, xs in groups.items():
            a = np.array(xs)
            per_day = pd.Series(a, index=pd.DatetimeIndex(dates[k])).groupby(level=0).mean()  # same-day entries move together
            rows.append({"group": k, "horizon": h, "entries": int(len(a)), "mean_excess": float(a.mean()),
                         "beat_market": float(np.mean(a > 0)),
                         "t": _tstat(per_day.to_numpy(), _step(per_day.index), h)})
    return sorted(rows, key=lambda r: (r["horizon"], r["group"]))
