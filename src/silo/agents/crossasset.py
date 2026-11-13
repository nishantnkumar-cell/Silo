"""Cross-market numbers for the brief: sector strength, macro sensitivities, regime.

What the research supports (see docs/news-and-cross-asset-research.md):
* sector momentum and laggards catching up with their sector are well
  evidenced, so the brief gets sector-relative columns;
* the pound, oil, metals, gilt yields and US shares mostly explain risk and
  regime rather than forecast single stocks, so they appear as each stock's
  sensitivities, a market line and an exposure check for the risk manager.

Everything is computed from a :class:`PointInTimeView`. Macro moves are
shown as percentiles of each series' own history, never as levels, because
"oil fell 60% in three months" dates itself and would break the brief's
anonymity. Sector names are never shown to a model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from silo.data.macro import BETA_DRIVERS, DRIVERS, table_name
from silo.firewall import PointInTimeView
from silo.universe.sectors import sector_of

MOVE = 63  # three months of trading days
TREND = 252
WEEKS = 52

COLUMNS = [
    ("sector_rel_6m", "vs own sector 6-month %"),
    ("sector_6m", "sector vs market 6-month %"),
    ("macro_tailwind", "macro-implied 3-month %"),
]


def driver_series(view: PointInTimeView, name: str) -> pd.Series:
    df = view.context(table_name(name))
    if df.empty:
        return pd.Series(dtype=float)
    return df["value"].astype(float)


def _change(values: np.ndarray, n: int, kind: str) -> np.ndarray:
    """n-day change: log change for prices, points for levels (yields)."""
    if kind == "level":
        return values[n:] - values[:-n]
    return np.log(values[n:] / values[:-n])


def driver_readings(view: PointInTimeView) -> dict[str, dict]:
    """Each driver's 3-month change, its percentile in its own history, and trend."""
    out = {}
    for name, d in DRIVERS.items():
        s = driver_series(view, name)
        v = s.to_numpy(dtype=float)
        if len(v) < TREND + MOVE:
            continue
        moves = _change(v, MOVE, d.kind)
        out[name] = {
            "move": float(moves[-1]),
            "pct": float((moves < moves[-1]).mean() * 100),
            "up": bool(v[-1] > v[-TREND:].mean()),
        }
    return out


def market_index(view: PointInTimeView, symbols: list[str], lookback: int = TREND + 10) -> pd.Series:
    """Equal-weight index of the universe from daily log returns (a UK market proxy)."""
    rets = []
    for s in symbols:
        c = view.close(s, lookback=lookback)
        if len(c) > 1:
            rets.append(np.log(c).diff().rename(s))
    if not rets:
        return pd.Series(dtype=float)
    mean = pd.concat(rets, axis=1).mean(axis=1).fillna(0.0)
    return np.exp(mean.cumsum())


def risk_on(view: PointInTimeView, symbols: list[str], readings: dict | None = None) -> tuple[int, int]:
    """How many risk signals point "on": UK market and risk-sensitive drivers in trend."""
    readings = driver_readings(view) if readings is None else readings
    on = total = 0
    idx = market_index(view, symbols)
    if len(idx) >= TREND:
        total += 1
        on += int(idx.iloc[-1] > idx.iloc[-TREND:].mean())
    for name, r in readings.items():
        sign = DRIVERS[name].risk_on
        if sign:
            total += 1
            on += int(r["up"] if sign > 0 else not r["up"])
    return on, total


def _weekly(series: pd.Series, kind: str) -> pd.Series:
    if series.empty:
        return series
    w = series.resample("W-FRI").last().dropna()
    return w.diff() if kind == "level" else np.log(w).diff()


def sensitivities(view: PointInTimeView, symbols: list[str]) -> dict[str, dict[str, float]]:
    """Each stock's weekly sensitivity (beta) to the UK market and each macro driver.

    One regression per stock over the last year of weekly returns, with the
    equal-weight UK market alongside the drivers, so "oil beta" means oil's
    effect beyond the market's.
    """
    lookback = WEEKS * 5 + 15
    closes = {s: view.close(s, lookback=lookback) for s in symbols}
    closes = {s: c for s, c in closes.items() if len(c) > 30}
    if not closes:
        return {}
    stocks = pd.DataFrame({s: _weekly(c, "price") for s, c in closes.items()})
    factors = {"market": stocks.mean(axis=1)}
    for name in BETA_DRIVERS:
        s = driver_series(view, name)
        if len(s) > 30:
            factors[name] = _weekly(s.iloc[-lookback - 10:], DRIVERS[name].kind)
    X = pd.DataFrame(factors).reindex(stocks.index)
    out = {}
    for s in stocks.columns:
        frame = pd.concat([stocks[s].rename("y"), X], axis=1).dropna().iloc[-WEEKS:]
        if len(frame) < 26:
            continue
        A = np.column_stack([np.ones(len(frame)), frame[list(X.columns)].to_numpy()])
        coef = np.linalg.lstsq(A, frame["y"].to_numpy(), rcond=None)[0]
        out[s] = dict(zip(X.columns, (float(c) for c in coef[1:])))
    return out


def sector_numbers(view: PointInTimeView, symbols: list[str], sectors: dict[str, str] | None = None,
                   lookback: int = 126) -> dict[str, tuple[float, float]]:
    """(stock vs its sector peers, sector vs the whole universe), 6-month % returns."""
    rets = {}
    for s in symbols:
        c = view.close(s, lookback=lookback + 1)
        if len(c) > lookback:
            rets[s] = 100 * (c.iloc[-1] / c.iloc[0] - 1)
    if not rets:
        return {}
    universe = float(np.mean(list(rets.values())))
    groups: dict[str, list[str]] = {}
    for s in rets:
        groups.setdefault(sector_of(s, sectors), []).append(s)
    out = {}
    for s, r in rets.items():
        peers = [rets[p] for p in groups[sector_of(s, sectors)] if p != s]
        sector_mean = float(np.mean(peers + [r]))
        out[s] = (r - float(np.mean(peers)) if peers else 0.0, sector_mean - universe)
    return out


def tailwind(betas: dict[str, float], readings: dict[str, dict]) -> float:
    """The 3-month move (%) a stock's macro sensitivities imply from the drivers' last 3 months."""
    return 100 * sum(b * readings[k]["move"] for k, b in betas.items() if k in readings)


def columns(view: PointInTimeView, symbols: list[str], sectors: dict[str, str] | None = None,
            readings: dict | None = None, betas: dict | None = None) -> dict[str, dict[str, float]]:
    """Values for :data:`COLUMNS`, keyed by column then symbol."""
    readings = driver_readings(view) if readings is None else readings
    betas = sensitivities(view, symbols) if betas is None else betas
    sec = sector_numbers(view, symbols, sectors)
    return {
        "sector_rel_6m": {s: v[0] for s, v in sec.items()},
        "sector_6m": {s: v[1] for s, v in sec.items()},
        "macro_tailwind": {s: tailwind({k: b for k, b in betas.get(s, {}).items() if k != "market"}, readings)
                           for s in symbols},
    }


def market_line(view: PointInTimeView, symbols: list[str], readings: dict | None = None) -> str:
    readings = driver_readings(view) if readings is None else readings
    on, total = risk_on(view, symbols, readings)
    parts = [f"{DRIVERS[n].label} {r['pct']:.0f}" for n, r in readings.items()]
    moves = ("Cross-market, each 3-month move as a percentile of its own history "
             "(0 = its biggest fall, 100 = its biggest rise): " + "; ".join(parts) + ".") if parts else \
        "Cross-market: no macro series available."
    return f"{moves} Risk-on signals: {on} of {total}."


def exposure_line(picks: list[str], betas: dict[str, dict[str, float]]) -> str:
    """Average sensitivity of the picks to each driver, for the risk manager."""
    rows = [betas[s] for s in picks if s in betas]
    if not rows:
        return "Exposure: no sensitivities available."
    keys = [k for k in ["market", *BETA_DRIVERS] if any(k in r for r in rows)]
    label = {"market": "UK market", **{k: DRIVERS[k].label for k in DRIVERS}}
    avg = {k: float(np.mean([r.get(k, 0.0) for r in rows])) for k in keys}
    return ("Exposure of these picks together (weekly sensitivity; 1.0 = moves one-for-one): "
            + "; ".join(f"{label[k]} {v:+.2f}" for k, v in avg.items()) + ".")
