"""Performance metrics for a daily equity curve."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
CALENDAR_DAYS = 365


def periods_per_year(index) -> int:
    """252 for weekday bars, 365 for markets that trade every day (crypto)."""
    idx = pd.DatetimeIndex(index)
    if len(idx) < 10:
        return TRADING_DAYS
    return CALENDAR_DAYS if float((idx.dayofweek >= 5).mean()) > 0.1 else TRADING_DAYS


def summarize(equity: pd.Series, fills: pd.DataFrame | None = None) -> dict[str, float]:
    equity = equity.dropna()
    if len(equity) < 2:
        return {}
    rets = equity.pct_change().dropna()
    TRADING_DAYS = periods_per_year(equity.index)  # noqa: N806 - shadows the default per curve
    years = len(rets) / TRADING_DAYS
    total = equity.iloc[-1] / equity.iloc[0] - 1
    vol = rets.std(ddof=1) * np.sqrt(TRADING_DAYS)
    downside = rets[rets < 0].std(ddof=1) * np.sqrt(TRADING_DAYS)
    drawdown = equity / equity.cummax() - 1
    out = {
        "total_return": float(total),
        "cagr": float((1 + total) ** (1 / years) - 1) if years > 0 and total > -1 else float("nan"),
        "volatility": float(vol),
        "sharpe": float(rets.mean() / rets.std(ddof=1) * np.sqrt(TRADING_DAYS)) if rets.std() > 0 else 0.0,
        "sortino": float(rets.mean() * TRADING_DAYS / downside) if downside and downside > 0 else 0.0,
        "max_drawdown": float(drawdown.min()),
        "days": float(len(equity)),
    }
    if fills is not None and not fills.empty:
        traded = (fills["qty"].abs() * fills["price"]).sum()
        out["fills"] = float(len(fills))
        out["turnover_per_year"] = float(traded / equity.mean() / years) if years > 0 else 0.0
        out["fees"] = float(fills["fee"].sum())
        trips, hold = holding_stats(fills, equity.index[-1])
        out["round_trips"] = float(trips)
        out["avg_hold_days"] = hold
    else:
        out["fills"] = 0.0
        out["round_trips"] = 0.0
        out["avg_hold_days"] = float("nan")
    return out


def holding_stats(fills: pd.DataFrame, end: pd.Timestamp) -> tuple[int, float]:
    """Completed round trips and mean calendar days each position was held.

    A holding runs from the fill that opens a position to the fill that
    closes it; positions still open at ``end`` count up to ``end``.
    """
    spells, opened, qty = [], {}, {}
    for row in fills.sort_values("date").itertuples():
        before = qty.get(row.symbol, 0.0)
        after = before + row.qty
        qty[row.symbol] = 0.0 if abs(after) < 1e-9 else after
        if before <= 0 < qty[row.symbol]:
            opened[row.symbol] = row.date
        elif before > 0 and qty[row.symbol] == 0:
            spells.append((row.date - opened.pop(row.symbol)).days)
    trips = len(spells)
    spells += [(pd.Timestamp(end) - d).days for d in opened.values()]
    return trips, float(np.mean(spells)) if spells else float("nan")
