"""Turn what the firewall allows into a short, factual brief for a model.

The brief is a table of numbers computed from prices, volumes and dividends
up to the decision clock, plus one line on the market as a whole. Nothing
else: no news, no company names, no dates.

Amnesia by default: tickers are replaced with neutral labels (``A``, ``B``,
...) in a fixed shuffled order and the date is never shown, so a model can't
recognise "this is Rolls-Royce in 2023" and recall what happened next. The
labels depend only on the strategy's symbol list, never on prices.
"""

from __future__ import annotations

import hashlib
import string
from dataclasses import dataclass

import numpy as np

from silo.firewall import PointInTimeView
from silo.liquidity import Liquidity

COLUMNS = ["ret_1m", "ret_3m", "ret_6m", "ret_12m", "vol", "vs_200d", "drawdown_1y", "yield", "traded_m"]
HEADER = "id | 1-month % | 3-month % | 6-month % | 12-month % | volatility % | vs 200-day avg % | " \
         "1-year drawdown % | dividend yield % | daily value traded (millions)"


def _labels(n: int) -> list[str]:
    letters = string.ascii_uppercase
    out = list(letters) + [a + b for a in letters for b in letters]
    return out[:n]


@dataclass
class Brief:
    rows: dict[str, list[float]]  # symbol -> numbers in COLUMNS order (plus any extra columns)
    alias: dict[str, str]  # symbol -> label shown to the model
    market: str
    header: str = HEADER

    @property
    def unalias(self) -> dict[str, str]:
        return {v: k for k, v in self.alias.items()}

    def table(self, symbols: list[str] | None = None) -> str:
        names = symbols or list(self.rows)
        lines = [self.header]
        for s in names:
            v = self.rows[s]
            lines.append(f"{self.alias[s]} | " + " | ".join(f"{x:+.1f}" if i < 8 else f"{x:.1f}"
                                                          for i, x in enumerate(v)))
        return "\n".join(lines)

    def to_symbols(self, labels) -> list[str]:
        back = self.unalias
        return [back[x] for x in labels if isinstance(x, str) and x in back]


def stock_numbers(view: PointInTimeView, symbol: str) -> list[float] | None:
    bars = view.bars(symbol, lookback=260)
    if len(bars) < 210:
        return None
    c = bars["close"].to_numpy(dtype=float)

    def ret(n):
        return 100 * (c[-1] / c[-1 - n] - 1) if len(c) > n else 0.0

    r = np.diff(np.log(c[-64:]))
    vol = 100 * float(np.std(r, ddof=1) * np.sqrt(252))
    vs200 = 100 * (c[-1] / c[-200:].mean() - 1)
    last = c[-252:]
    dd = 100 * float((last / np.maximum.accumulate(last) - 1).min())
    divs = float(bars["dividend"].iloc[-252:].sum()) if "dividend" in bars else 0.0
    yld = 100 * divs / c[-1]
    traded = float(np.nanmedian((bars["close"] * bars["volume"]).to_numpy()[-60:])) / 1e6
    return [ret(21), ret(63), ret(126), ret(252), vol, vs200, dd, yld, traded]


def build_brief(
    view: PointInTimeView,
    symbols: list[str],
    anonymize: bool = True,
    liquidity: Liquidity | None = Liquidity(),
    limit: int | None = None,
    extra=None,
) -> Brief:
    """Numbers for every symbol that has enough history (and passes liquidity).

    ``extra`` is an optional ``(header_suffix, fn(view, symbol) -> list[float])``
    that appends columns, e.g. the supply-and-demand readings for agents that
    follow the Core Strategy rulebook. Without it the brief is unchanged.
    """
    rows = {}
    for s in symbols:
        if liquidity is not None and not liquidity.check(view, s)[0]:
            continue
        nums = stock_numbers(view, s)
        if nums is not None:
            rows[s] = nums
    if limit and len(rows) > limit:  # keep the prompt small: strongest 6-month trend first
        keep = sorted(rows, key=lambda s: -rows[s][2])[:limit]
        rows = {s: rows[s] for s in keep}
    order = sorted(symbols, key=lambda s: hashlib.sha256(s.encode()).hexdigest())
    alias = dict(zip(order, _labels(len(order)))) if anonymize else {s: s for s in symbols}
    alias = {s: alias[s] for s in rows}
    header = HEADER
    if extra is not None:
        suffix, fn = extra
        header = f"{HEADER} | {suffix}"
        rows = {s: v + list(fn(view, s)) for s, v in rows.items()}
    if rows:
        m3 = np.mean([v[1] for v in rows.values()])
        above = np.mean([v[5] > 0 for v in rows.values()])
        vol = np.median([v[4] for v in rows.values()])
        market = (f"Market: the average stock here moved {m3:+.1f}% over 3 months; "
                  f"{above:.0%} trade above their 200-day average; median volatility {vol:.0f}%.")
    else:
        market = "Market: no stocks have enough history yet."
    return Brief(rows, alias, market, header)
