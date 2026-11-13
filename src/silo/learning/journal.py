"""The trade journal: every position from open to close, why it was opened, and how it went.

Built from a finished run (a backtest, or a paper book's session), so it
needs no extra bookkeeping in the engine. A *trade* runs from the fill that
opens a position to the fill that closes it; adds and trims in between
belong to the same trade.

For each trade the journal keeps:

* what the agent saw when it decided to buy: the brief's numbers, market
  breadth, optionally the Core Strategy zone readings, and for AI desks
  whether the bull backed it or the bear warned about it, plus the trader's
  stated reason (all computed point-in-time at the decision clock);
* the outcome after costs and dividends, and the equal-weight market's
  return over the same days, so a trade that rose 5% while everything rose
  10% counts as a miss.

The journal holds symbols and dates, for people. Anything shown to an AI
reviewer goes through :func:`anonymised` first.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from silo.engine import BacktestResult
from silo.firewall import PointInTimeView, PreparedTables
from silo.learning.lessons import breadth, features_at


@dataclass
class Trade:
    strategy: str
    symbol: str
    opened: pd.Timestamp
    closed: pd.Timestamp
    decided_at: pd.Timestamp
    open: bool
    hold_days: int
    cost: float
    proceeds: float
    fees: float
    dividends: float
    ret: float
    market_ret: float
    features: dict[str, float] = field(default_factory=dict)
    why: str = ""

    @property
    def excess(self) -> float:
        return self.ret - self.market_ret

    @property
    def won(self) -> bool:
        return self.excess > 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(excess=self.excess, opened=str(self.opened.date()), closed=str(self.closed.date()),
                 decided_at=str(self.decided_at))
        return d


def _episodes(fills: pd.DataFrame):
    """Yield (symbol, rows) for each flat-to-flat holding, plus whether it is still open."""
    qty: dict[str, float] = {}
    rows: dict[str, list] = {}
    for r in fills.sort_values("date", kind="stable").itertuples(index=False):
        before = qty.get(r.symbol, 0.0)
        after = before + r.qty
        qty[r.symbol] = 0.0 if abs(after) < 1e-9 else after
        rows.setdefault(r.symbol, []).append(r)
        if before > 0 and qty[r.symbol] == 0:
            yield r.symbol, rows.pop(r.symbol), False
        elif before <= 0 and qty[r.symbol] <= 0:
            rows.pop(r.symbol, None)  # a stray sell with nothing held: not a trade
    for s, rs in rows.items():
        if qty.get(s, 0) > 0:
            yield s, rs, True


def build_journal(result: BacktestResult, tables: dict[str, pd.DataFrame], symbols: list[str] | None = None,
                  core: bool = False, strategy: str | None = None) -> list[Trade]:
    """One :class:`Trade` per position in ``result``, with entry features and outcome."""
    if result.fills is None or result.fills.empty:
        return []
    symbols = list(symbols or tables)
    prepared = PreparedTables({s: tables[s] for s in symbols})
    closes = pd.DataFrame({s: tables[s]["close"] for s in symbols}).sort_index()
    end = result.equity.index[-1]
    notes_at = {str(d["clock"]): d.get("notes") for d in result.decisions}
    divs = pd.DataFrame(result.dividend_events or [], columns=["date", "symbol", "qty", "per_share"])
    market_cache: dict[str, float | None] = {}
    out = []
    for sym, rows, still_open in _episodes(result.fills):
        buys = [r for r in rows if r.qty > 0]
        if not buys:
            continue
        opened, decided = buys[0].date, buys[0].decided_at
        closed = end if still_open else rows[-1].date
        cost = float(sum(r.qty * r.price for r in buys))
        proceeds = float(sum(-r.qty * r.price for r in rows if r.qty < 0))
        if still_open:
            held = sum(r.qty for r in rows)
            last = closes[sym].loc[:end].dropna()
            proceeds += held * float(last.iloc[-1]) if len(last) else 0.0
        fees = float(sum(r.fee for r in rows))
        d = divs[(divs["symbol"] == sym) & (divs["date"] > opened) & (divs["date"] <= closed)]
        dividends = float((d["qty"] * d["per_share"]).sum()) if len(d) else 0.0
        ret = (proceeds + dividends - fees) / cost - 1 if cost else 0.0
        window = closes.loc[opened:closed]
        mret = float(np.nanmean(window.iloc[-1] / window.iloc[0] - 1)) if len(window) > 1 else 0.0
        mret = 0.0 if np.isnan(mret) else mret
        hold = int(tables[sym].loc[opened:closed].shape[0]) - 1
        view = PointInTimeView(prepared, decided)
        key = str(decided)
        if key not in market_cache:
            market_cache[key] = breadth(view, symbols)
        notes = notes_at.get(key) or {}
        feats = features_at(view, sym, core=core, notes=notes, market=market_cache[key])
        out.append(Trade(strategy or result.strategy, sym, opened, closed, decided, still_open, hold, cost,
                         proceeds, fees, dividends, float(ret), mret, feats, str(notes.get("why", ""))))
    return sorted(out, key=lambda t: (t.opened, t.symbol))


def summary(trades: list[Trade]) -> dict:
    closed = [t for t in trades if not t.open]
    if not closed:
        return {"trades": len(trades), "closed": 0}
    ex = np.array([t.excess for t in closed])
    return {
        "trades": len(trades), "closed": len(closed),
        "win_rate_vs_market": float(np.mean(ex > 0)),
        "avg_return": float(np.mean([t.ret for t in closed])),
        "avg_excess": float(ex.mean()),
        "avg_hold_days": float(np.mean([t.hold_days for t in closed])),
        "fees": float(sum(t.fees for t in trades)),
    }


def write_journal(trades: list[Trade], out_dir: str | Path) -> tuple[Path, Path]:
    """trades.csv for people, journal.jsonl for tools. Written fresh for each review run."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = [t.to_dict() for t in trades]
    flat = [{**{k: v for k, v in r.items() if k != "features"}, **r["features"]} for r in rows]
    csv = out / "trades.csv"
    pd.DataFrame(flat).to_csv(csv, index=False)
    jl = out / "journal.jsonl"
    jl.write_text("".join(json.dumps(r, default=str) + "\n" for r in rows))
    return csv, jl


_TICKER = re.compile(r"\b[A-Z0-9]{1,5}\.L\b")
_YEAR = re.compile(r"\b(19|20)\d\d\b")
_MONTH = re.compile(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b", re.IGNORECASE)


def scrub(text: str) -> str:
    """Remove anything that looks like a ticker, a year or a month from free text."""
    return _MONTH.sub("[month]", _YEAR.sub("[year]", _TICKER.sub("[stock]", text)))[:300]


def anonymised(trades: list[Trade]) -> list[dict]:
    """Closed trades with no symbol and no dates, in a fixed shuffled order (not chronological)."""
    closed = [t for t in trades if not t.open]
    rng = np.random.default_rng(0)
    order = rng.permutation(len(closed)) if closed else []
    return [{"hold_days": closed[i].hold_days, "return_pct": round(100 * closed[i].ret, 1),
             "vs_market_pct": round(100 * closed[i].excess, 1), **closed[i].features,
             **({"why": scrub(closed[i].why)} if closed[i].why else {})} for i in order]
