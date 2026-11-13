"""How a strategy did in different market conditions.

A strategy that only works in a rising market is a leveraged bet on the
market, not an edge. This splits the out-of-sample period by the state of
the buy-and-hold benchmark and compares the two in each state.

Each day is labelled from trailing benchmark data only:
* trend: the benchmark's return over the previous 63 trading days
  (rising above +5%, falling below -5%, otherwise sideways);
* volatility: the previous 21 days' realised volatility compared with its
  own trailing one-year median (calm or stormy).

Named stress windows (the 2020 crash, the 2022 rate shock) are reported too
when the data covers them. Labels describe the past; nothing here feeds a
decision.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

STRESS_WINDOWS = {
    "Covid crash (19 Feb to 23 Mar 2020)": ("2020-02-19", "2020-03-23"),
    "Covid rebound (24 Mar to 31 Dec 2020)": ("2020-03-24", "2020-12-31"),
    "Rate shock (2022)": ("2022-01-01", "2022-12-31"),
}


def label_regimes(bench_equity: pd.Series, trend_days: int = 63, band: float = 0.05) -> pd.DataFrame:
    rets = bench_equity.pct_change()
    past = bench_equity.shift(1)
    trend_ret = past / past.shift(trend_days) - 1
    trend = np.where(trend_ret > band, "rising", np.where(trend_ret < -band, "falling", "sideways"))
    trend = pd.Series(trend, index=bench_equity.index).where(trend_ret.notna())
    vol = rets.shift(1).rolling(21).std()
    ref = vol.rolling(252, min_periods=63).median()
    storm = pd.Series(np.where(vol > ref, "stormy", "calm"), index=bench_equity.index).where(ref.notna())
    return pd.DataFrame({"trend": trend, "volatility": storm})


def _compound(r: pd.Series) -> float:
    return float((1 + r.fillna(0)).prod() - 1)


def regime_table(strategy_equity: pd.Series, bench_equity: pd.Series) -> list[dict]:
    """One row per market state: days, strategy and buy-and-hold return."""
    idx = strategy_equity.index.intersection(bench_equity.index)
    s = strategy_equity.loc[idx].pct_change()
    b = bench_equity.loc[idx].pct_change()
    labels = label_regimes(bench_equity).reindex(idx)
    rows = []
    for col, states in (("trend", ["rising", "sideways", "falling"]), ("volatility", ["calm", "stormy"])):
        for state in states:
            mask = labels[col] == state
            if mask.sum() == 0:
                continue
            rows.append({
                "state": f"{state} market" if col == "trend" else f"{state} volatility",
                "days": int(mask.sum()),
                "strategy": _compound(s[mask]),
                "benchmark": _compound(b[mask]),
            })
    for name, (lo, hi) in STRESS_WINDOWS.items():
        mask = (idx >= pd.Timestamp(lo)) & (idx <= pd.Timestamp(hi))
        if mask.sum() >= 5:
            rows.append({"state": name, "days": int(mask.sum()),
                         "strategy": _compound(s[mask]), "benchmark": _compound(b[mask])})
    return rows
