"""Performance metrics for a daily equity curve."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def summarize(equity: pd.Series, fills: pd.DataFrame | None = None) -> dict[str, float]:
    equity = equity.dropna()
    if len(equity) < 2:
        return {}
    rets = equity.pct_change().dropna()
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
    else:
        out["fills"] = 0.0
    return out
