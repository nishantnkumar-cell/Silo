"""Language-model clients, with every call logged and every reply cached.

Silo runs on free models. The default is a local model served by Ollama
(https://ollama.com) on the same computer, so bulk backtests cost nothing
and no data leaves the machine. Pick a model with ``SILO_LLM``:

* ``ollama:qwen2.5:7b`` (or any model you have pulled with ``ollama pull``)
* ``rulebot`` - an offline, deterministic stand-in that reads the brief's
  numbers and answers in the same JSON shape. It is not a language model;
  it exists so tests and dry runs work with no model installed.

Every call is appended to a call log (prompt, reply, model, the decision
clock, and the model's training cutoff), and replies are cached by a hash of
model and prompt so a rerun replays the same answers instead of asking again.
That keeps backtests reproducible and makes every decision auditable.

Training cutoff: a model trained after a date may remember what happened
next. Results from decisions before ``cutoff`` are contaminated and the lab
report says so. Briefs hide tickers and dates by default to limit this.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

# Training cutoffs for common local models (year-month, from model cards).
KNOWN_CUTOFFS = {
    "qwen2.5": "2024-09",
    "qwen3": "2025-03",
    "llama3.1": "2023-12",
    "llama3.2": "2023-12",
    "gemma2": "2024-06",
    "gemma3": "2024-08",
    "mistral": "2023-12",
    "phi4": "2024-06",
    "deepseek-r1": "2024-07",
}

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


class LLMError(RuntimeError):
    pass


def cutoff_for(model: str) -> str | None:
    base = model.split(":")[0].lower()
    return KNOWN_CUTOFFS.get(base)


@dataclass
class LLMClient:
    """Base client: cache, log, then ask the model."""

    model: str = "base"
    cutoff: str | None = None
    log_dir: Path | None = None
    calls: int = 0
    cache_hits: int = 0
    _cache: dict = field(default_factory=dict, repr=False)

    def _ask(self, system: str, prompt: str) -> str:
        raise NotImplementedError

    def __getstate__(self):  # paper books pickle their strategy; the cache reloads from disk
        state = dict(self.__dict__)
        state["_cache"] = {}
        return state

    def _key(self, system: str, prompt: str) -> str:
        return hashlib.sha256(f"{self.model}\0{system}\0{prompt}".encode()).hexdigest()

    def _load_cache(self) -> None:
        if self.log_dir is None or self._cache:
            return
        path = self.log_dir / "cache.jsonl"
        if path.exists():
            for line in path.read_text().splitlines():
                rec = json.loads(line)
                self._cache[rec["key"]] = rec["reply"]

    def complete(self, system: str, prompt: str, *, role: str = "", clock=None) -> str:
        self._load_cache()
        key = self._key(system, prompt)
        cached = key in self._cache
        if cached:
            reply = self._cache[key]
            self.cache_hits += 1
        else:
            reply = self._ask(system, prompt)
            self._cache[key] = reply
            self.calls += 1
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            if not cached:
                with (self.log_dir / "cache.jsonl").open("a") as f:
                    f.write(json.dumps({"key": key, "reply": reply}) + "\n")
            rec = {
                "at": datetime.now(timezone.utc).isoformat(),
                "clock": None if clock is None else str(clock),
                "model": self.model, "cutoff": self.cutoff, "role": role, "key": key, "cached": cached,
            }
            if not cached:  # the full text once; replays point at it by key
                rec.update(system=system, prompt=prompt, reply=reply)
            with (self.log_dir / "calls.jsonl").open("a") as f:
                f.write(json.dumps(rec) + "\n")
        return reply


@dataclass
class OllamaClient(LLMClient):
    """A model served by Ollama on this computer. Free; nothing leaves the machine."""

    host: str = "http://localhost:11434"
    temperature: float = 0.0
    seed: int = 7
    timeout: float = 600.0

    def __post_init__(self) -> None:
        if urlparse(self.host).hostname not in LOCAL_HOSTS:
            raise LLMError(f"Ollama host must be local (got {self.host}); Silo keeps prompts on this machine")
        if self.cutoff is None:
            self.cutoff = cutoff_for(self.model) or "unknown"

    def _ask(self, system: str, prompt: str) -> str:
        body = json.dumps({
            "model": self.model, "system": system, "prompt": prompt, "stream": False, "format": "json",
            "options": {"temperature": self.temperature, "seed": self.seed},
        }).encode()
        req = urllib.request.Request(f"{self.host}/api/generate", body, {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:  # noqa: S310 - local host only
                return json.loads(r.read())["response"]
        except OSError as exc:
            raise LLMError(f"could not reach Ollama at {self.host}: {exc}. Is `ollama serve` running?") from exc


_NUM = re.compile(r"(?P<id>[A-Z]{1,3}|[A-Z0-9.]+\.L)\s*\|\s*(?P<rest>.*)")


@dataclass
class RuleBot(LLMClient):
    """Offline stand-in for a model. Deterministic; reads numbers in the brief.

    It answers each role's question in the same JSON shape a model would, so
    the whole agent pipeline can run in tests. It is a rule, not an AI, and
    its results say nothing about language models.
    """

    model: str = "rulebot"

    def _ask(self, system: str, prompt: str) -> str:
        rows, header = {}, []
        for line in prompt.splitlines():
            if line.startswith("id | ") and not header:
                header = [c.strip() for c in line.split("|")[1:]]
                continue
            m = _NUM.match(line.strip())
            if not m:
                continue
            nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", m["rest"])]
            if len(nums) >= 4:
                rows[m["id"]] = nums
        # brief columns: ret_1m, ret_3m, ret_6m, ret_12m, vol, vs_200d, drawdown, yield, ...
        score = {k: (v[2] / max(v[4], 1.0)) + 0.5 * v[5] / 10 for k, v in rows.items()}
        if "contrarian" in system.lower():
            score = {k: -v[0] for k, v in rows.items()}
        if "income" in system.lower():
            score = {k: v[7] - 0.05 * v[4] for k, v in rows.items()}
        if "defensive" in system.lower() or "risk" in system.lower():
            score = {k: -v[4] - 0.5 * abs(v[6]) for k, v in rows.items()}
        def col(v, name):  # an extra column by its header, 0 when absent
            i = header.index(name) if name in header else -1
            return v[i] if 0 <= i < len(v) else 0.0

        if "cross-market" in system.lower():
            score = {k: col(v, "sector vs market 6-month %") + col(v, "macro-implied 3-month %")
                     for k, v in rows.items()}
        ranked = sorted(score, key=lambda k: -score[k])
        if '"favours"' in prompt:
            return json.dumps({"note": "rulebot", "favours": ranked[:5], "avoids": ranked[::-1][:5]})
        if '"scores"' in prompt:
            n = max(1, len(ranked))
            return json.dumps({"scores": {k: round(1 - 2 * i / n, 2) for i, k in enumerate(ranked)},
                               "why": "rulebot ranking"})
        if '"argument"' in prompt:
            side = ranked if "bull" in system.lower() else ranked[::-1]
            return json.dumps({"argument": "rulebot", "names": side[:5]})
        if '"veto"' in prompt:
            worst = [k for k, v in rows.items() if v[6] < -30]
            return json.dumps({"veto": worst, "why": "deep drawdown"})
        return json.dumps({"picks": ranked[:8], "why": "rulebot ranking"})


def client_from_spec(spec: str | None = None, log_dir: str | Path | None = None) -> LLMClient:
    """Build a client from ``SILO_LLM`` (or ``spec``): ``ollama:<model>`` or ``rulebot``."""
    spec = spec or os.environ.get("SILO_LLM")
    if not spec:
        raise LLMError(
            "no model chosen: set SILO_LLM to ollama:<model> (free, local; e.g. ollama:qwen2.5:7b) "
            "or rulebot (offline stand-in, not an AI)"
        )
    log = Path(log_dir) if log_dir else Path(os.environ.get("SILO_LLM_LOG", "llm_log"))
    if spec == "rulebot":
        return RuleBot(log_dir=log)
    if spec.startswith("ollama:"):
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        if not host.startswith("http"):
            host = f"http://{host}"
        host = host.replace("0.0.0.0", "localhost")  # a server bind address; connect locally
        return OllamaClient(model=spec.split(":", 1)[1], host=host, log_dir=log)
    raise LLMError(f"unknown SILO_LLM {spec!r}; use ollama:<model> or rulebot")


def parse_json(reply: str) -> dict:
    """Pull the first JSON object out of a reply. Models sometimes add prose."""
    try:
        return json.loads(reply)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", reply, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return {}
