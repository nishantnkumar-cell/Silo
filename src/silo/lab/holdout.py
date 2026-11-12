"""The locked holdout: the last slice of data, opened once per strategy version.

Every opening is appended to a log. Opening the same strategy, parameters and
symbols again is allowed but flagged in the report as ``reopened``, because a
holdout looked at twice is no longer out of sample.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


class HoldoutLog:
    def __init__(self, path: str | Path = "reports/holdout_log.jsonl") -> None:
        self.path = Path(path)

    @staticmethod
    def key(strategy: str, params: dict, symbols: list[str], holdout_start: str) -> str:
        blob = json.dumps([strategy, params, sorted(symbols), holdout_start], sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def openings(self, key: str) -> int:
        if not self.path.exists():
            return 0
        with self.path.open() as f:
            return sum(1 for line in f if json.loads(line).get("key") == key)

    def record(self, key: str, **info) -> int:
        """Log an opening; return how many times this key was opened before."""
        before = self.openings(key)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        entry = {"key": key, "opened_at": datetime.now(timezone.utc).isoformat(), **info}
        with self.path.open("a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
        return before
