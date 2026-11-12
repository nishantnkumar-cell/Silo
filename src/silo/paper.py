"""Forward paper trading on Silo's own simulator. Never real money.

A paper book runs a strategy forward one trading day at a time as new bars
arrive, using exactly the same :class:`~silo.engine.Session` as a backtest.
It starts after the last bar available when it was created, so it has no
history to fit and every result is genuinely out of sample.

Each book lives in ``paper/<name>/``:

* ``config.json``  what the book trades and under which rules
* ``session.pkl``  the strategy, broker and benchmark state between runs
* ``journal.jsonl`` one line per event, append-only, never rewritten

A buy-and-hold shadow book on the same symbols runs alongside as the
benchmark.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from silo.broker import CostModel
from silo.constraints import Constraints
from silo.data.vault import DataVault
from silo.engine import MarketData, Session
from silo.markets import market_for_symbols
from silo.metrics import summarize
from silo.strategies import REGISTRY
from silo.strategies.classic import BuyAndHold

DISCLAIMER = "Paper trading on a simulator. Research only, not financial advice, never real money."


def _json(obj):
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    return str(obj)


class PaperBook:
    def __init__(self, name: str, root: str | Path = "paper") -> None:
        self.name = name
        self.dir = Path(root) / name

    @property
    def exists(self) -> bool:
        return (self.dir / "config.json").exists()

    def config(self) -> dict:
        return json.loads((self.dir / "config.json").read_text())

    def create(
        self,
        strategy: str,
        symbols: list[str],
        vault: DataVault,
        params: dict | None = None,
        cash: float = 100_000.0,
        constraints: Constraints | None = None,
        costs: CostModel | None = None,
    ) -> dict:
        if self.exists:
            raise FileExistsError(f"paper book {self.name!r} already exists")
        market = market_for_symbols(symbols)
        constraints = constraints or Constraints.position_trading()
        costs = costs or CostModel.for_market(market.code)
        tables = {s: vault.get_bars(s) for s in symbols}
        start_after = max(df.index[-1] for df in tables.values())
        cfg = {
            "name": self.name,
            "strategy": strategy,
            "params": params or {},
            "symbols": symbols,
            "market": market.code,
            "currency": market.currency,
            "cash": cash,
            "constraints": asdict(constraints),
            "costs": asdict(costs),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "start_after": start_after.isoformat(),
        }
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "config.json").write_text(json.dumps(cfg, indent=2))
        state = {
            "book": Session.start(REGISTRY[strategy](symbols, **(params or {})), cash, costs, constraints),
            "benchmark": Session.start(BuyAndHold(symbols), cash, costs, Constraints()),
        }
        self._save(state)
        self._journal({"event": "created", **cfg})
        return cfg

    def _save(self, state: dict) -> None:
        tmp = self.dir / "session.pkl.tmp"
        tmp.write_bytes(pickle.dumps(state))
        tmp.replace(self.dir / "session.pkl")

    def _load(self) -> dict:
        return pickle.loads((self.dir / "session.pkl").read_bytes())  # noqa: S301 - our own file

    def _journal(self, entry: dict) -> None:
        with (self.dir / "journal.jsonl").open("a") as f:
            f.write(json.dumps(entry, default=_json) + "\n")

    def run(self, vault: DataVault) -> list[dict]:
        """Process every new trading day since the last run. Safe to repeat."""
        cfg = self.config()
        state = self._load()
        tables = {s: vault.get_bars(s) for s in cfg["symbols"]}
        data = MarketData(tables)
        start_after = pd.Timestamp(cfg["start_after"])
        done = state["book"].last_date or start_after
        new_days = [d for d in data.dates if d > max(done, start_after)]
        events = []
        for date in new_days:
            ev = state["book"].step(date, data)
            bench = state["benchmark"].step(date, data)
            entry = {
                "event": "day",
                "date": date,
                "equity": ev["equity"],
                "benchmark_equity": bench["equity"],
                "fills": ev["fills"],
                "dividends": ev["dividends"],
            }
            if ev["decision"]:
                entry["decision"] = {
                    "proposed": ev["decision"]["proposed"],
                    "after_rules": ev["decision"]["weights"],
                }
            self._journal(entry)
            events.append(entry)
        self._save(state)
        return events

    def status(self) -> dict:
        cfg = self.config()
        state = self._load()
        book, bench = state["book"], state["benchmark"]
        eq = pd.Series(book.equity, dtype=float)
        beq = pd.Series(bench.equity, dtype=float)
        weights = book.broker.weights(book.last_close) if book.last_close else {}
        return {
            "name": cfg["name"],
            "strategy": cfg["strategy"],
            "currency": cfg["currency"],
            "days": len(eq),
            "last_date": book.last_date,
            "equity": float(eq.iloc[-1]) if len(eq) else cfg["cash"],
            "benchmark_equity": float(beq.iloc[-1]) if len(beq) else cfg["cash"],
            "cash": book.broker.cash,
            "fees_paid": book.broker.fees_paid,
            "positions": {
                s: {"weight": w, "held_days": book.held_days.get(s, 0)} for s, w in sorted(weights.items())
            },
            "metrics": summarize(eq, book.result().fills) if len(eq) > 1 else {},
        }
