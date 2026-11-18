"""One project-wide count of everything that has been tried.

Try enough strategies and one will look good by luck. The deflated Sharpe
test corrects for that, but only if it knows how many attempts there were.
The lab used to count just the parameter settings inside one study. This
ledger records every study and league run (strategy, settings, dates and
the Sharpe it got), and the lab's luck test counts every earlier attempt
whose test period overlaps the one being judged.

Running the same thing again is not a new attempt: entries are keyed by
strategy, settings, symbols and dates, and the newest entry for a key wins.
The file is append-only.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


class TrialLog:
    def __init__(self, path: str | Path = "reports/trials.jsonl") -> None:
        self.path = Path(path)

    @staticmethod
    def key(strategy: str, params: dict, n_symbols: int, start, end) -> str:
        blob = json.dumps([strategy, params, n_symbols, str(pd.Timestamp(start).date()),
                           str(pd.Timestamp(end).date())], sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def record(self, kind: str, strategy: str, params: dict, n_symbols: int, start, end, sharpe: float) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        entry = {"at": datetime.now(timezone.utc).isoformat(), "kind": kind, "strategy": strategy,
                 "params": params, "symbols": n_symbols, "start": str(pd.Timestamp(start).date()),
                 "end": str(pd.Timestamp(end).date()), "sharpe": float(sharpe),
                 "key": self.key(strategy, params, n_symbols, start, end)}
        with self.path.open("a") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def entries(self) -> list[dict]:
        if not self.path.exists():
            return []
        latest: dict[str, dict] = {}
        for line in self.path.read_text().splitlines():
            if line.strip():
                e = json.loads(line)
                latest[e["key"]] = e
        return list(latest.values())

    def overlapping(self, start, end, exclude_key: str | None = None) -> list[float]:
        """Daily Sharpes of earlier attempts whose dates overlap [start, end]."""
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        return [x["sharpe"] for x in self.entries()
                if x["key"] != exclude_key and pd.Timestamp(x["start"]) <= e and pd.Timestamp(x["end"]) >= s]
