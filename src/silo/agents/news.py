"""News-event numbers for the brief: results reactions, expectations, warnings, director buying.

What the research supports (docs/news-and-cross-asset-research.md):
* price moves on news days tend to continue and moves without news tend to
  reverse, so momentum is split into news-day and quiet-day parts;
* the share-price reaction around results is a free stand-in for the
  earnings surprise (consensus forecasts aren't available point-in-time);
* UK trading statements say "ahead of / in line with / below market
  expectations", a company-written surprise sign;
* director purchases in UK companies have been followed by abnormal returns.

The deciding model never sees announcement text, headlines or dates, only
these numbers next to the anonymised price columns. When a stock has no
stored announcements, news days fall back to days of unusually heavy
trading (a common proxy), so the columns still mean something.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from silo.data.news import TABLE
from silo.firewall import PointInTimeView
from silo.markets import close_timestamps, market_for

RESULTS_KINDS = {"results", "trading_update", "profit_warning"}
QUIET_KINDS = {"routine"}
WINDOW = 126  # six months of trading days
NO_RESULTS = 260

COLUMNS = [
    ("days_since_results", "trading days since results"),
    ("results_move", "results-day move vs market %"),
    ("expect", "last update vs expectations (+1 ahead, -1 below)"),
    ("warning", "profit warning in 90 days (1 = yes)"),
    ("director_buys", "director buys in 90 days"),
    ("news_mom", "6-month % on news days"),
    ("quiet_mom", "6-month % on quiet days"),
]


def recent_news(view: PointInTimeView, days: int = 400) -> pd.DataFrame:
    df = view.context(TABLE)
    if df.empty:
        return df
    return df[pd.to_datetime(df["available_at"], utc=True) >= view.clock - pd.Timedelta(days=days)]


def _event_positions(dates: pd.DatetimeIndex, symbol: str, times: pd.Series) -> np.ndarray:
    """Index of the first bar whose close is at or after each announcement (the day it could trade)."""
    closes = close_timestamps(dates, market_for(symbol)).as_unit("ns").asi8
    t = pd.DatetimeIndex(pd.to_datetime(times, utc=True)).as_unit("ns").asi8
    return np.searchsorted(closes, t, side="left")


def stock_news(view: PointInTimeView, symbol: str, news: pd.DataFrame, market: pd.Series | None) -> dict:
    bars = view.bars(symbol, lookback=WINDOW + 140)
    out = {"days_since_results": float(NO_RESULTS), "results_move": 0.0, "expect": 0.0, "warning": 0.0,
           "director_buys": 0.0, "news_mom": 0.0, "quiet_mom": 0.0}
    if len(bars) < 30:
        return out
    close = bars["close"].to_numpy(dtype=float)
    dates = pd.DatetimeIndex(bars.index)
    mine = news[news["symbol"] == symbol] if not news.empty else news
    clock = view.clock
    news_day = np.zeros(len(close), dtype=bool)
    if not mine.empty:
        avail = pd.to_datetime(mine["available_at"], utc=True)
        pos = _event_positions(dates, symbol, avail)
        kinds = mine["category"].to_numpy()
        for p, k in zip(pos, kinds):
            if k not in QUIET_KINDS and p < len(close):
                news_day[p] = True
                if p + 1 < len(close):
                    news_day[p + 1] = True
        res = np.isin(kinds, list(RESULTS_KINDS))
        if res.any():
            last = int(np.flatnonzero(res)[-1])
            p = int(pos[last])
            out["days_since_results"] = float(min(NO_RESULTS, max(0, len(close) - 1 - p)))
            if p >= 1 and p + 2 <= len(close) - 1 and len(close) - 1 - p <= WINDOW:
                move = close[p + 2] / close[p - 1] - 1
                if market is not None:
                    m = market.reindex(dates).ffill().to_numpy(dtype=float)
                    if np.isfinite(m[p + 2]) and np.isfinite(m[p - 1]):
                        move -= m[p + 2] / m[p - 1] - 1
                out["results_move"] = 100 * float(move)
            if clock - avail.iloc[last] <= pd.Timedelta(days=180):
                out["expect"] = float(mine["expect"].iloc[last]) if "expect" in mine else 0.0
        recent = (clock - avail) <= pd.Timedelta(days=90)
        exp = mine["expect"].to_numpy() if "expect" in mine else np.zeros(len(mine))
        out["warning"] = float(bool(((kinds == "profit_warning") | (res & (exp < 0)))[recent.to_numpy()].any()))
        if "dealing" in mine:
            out["director_buys"] = float(((mine["dealing"].to_numpy() > 0) & recent.to_numpy()).sum())
    else:  # no announcements stored: heavy-volume days stand in for news days
        vol = bars["volume"].to_numpy(dtype=float) if "volume" in bars else np.zeros(len(close))
        med = pd.Series(vol).rolling(60, min_periods=20).median().to_numpy()
        news_day = np.nan_to_num(vol > 2 * med, nan=False).astype(bool)
    r = np.diff(np.log(close))[-WINDOW:]
    flag = news_day[1:][-WINDOW:]
    out["news_mom"] = 100 * float(np.expm1(r[flag].sum()))
    out["quiet_mom"] = 100 * float(np.expm1(r[~flag].sum()))
    return out


def columns(view: PointInTimeView, symbols: list[str], market: pd.Series | None = None) -> dict[str, dict]:
    if market is None:
        from silo.agents.crossasset import market_index

        market = market_index(view, symbols, lookback=WINDOW + 140)
    news = recent_news(view)
    llm = view.context("ctx:news_llm")
    if not news.empty and not llm.empty and "expect" in news:  # the model's reading fills gaps the rules left
        filled = news["expect"].where(news["expect"] != 0, llm["expect_llm"].reindex(news.index))
        news = news.assign(expect=filled.fillna(0).astype(int))
    rows = {s: stock_news(view, s, news, market) for s in symbols}
    return {key: {s: v[key] for s, v in rows.items()} for key, _ in COLUMNS}


def market_line(view: PointInTimeView, symbols: list[str], cols: dict[str, dict]) -> str:
    n = len(symbols) or 1
    reported = sum(1 for s in symbols if cols["days_since_results"].get(s, NO_RESULTS) <= 21)
    ahead = sum(1 for s in symbols if cols["expect"].get(s, 0) > 0)
    below = sum(1 for s in symbols if cols["expect"].get(s, 0) < 0)
    return (f"News: {reported / n:.0%} of stocks reported results in the last month; "
            f"{ahead} said ahead of expectations and {below} below in their latest update.")
