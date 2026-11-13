"""A league of paper books: every contestant on the same UK universe.

``silo league new`` creates one forward paper book per contestant (rules,
AI contestants and Silo's own agent, plus a random control), all starting on
the same day with the same cash and rules. ``silo daily`` then fetches new
prices once, advances every book, prints a leaderboard against buy-and-hold,
and can mirror one chosen book to the Trading 212 **Practice** account.

Everything here is paper. The simulator is the source of truth; the
Practice mirror is a check that orders would have been accepted.
"""

from __future__ import annotations

import json
from pathlib import Path

from silo.data.vault import DataVault
from silo.paper import PaperBook

RULE_CONTESTANTS = ["uk_trend", "momentum", "blend", "random"]
# The luck control rebalances about once a month into a random ~90% of names.
# Momentum was written for short ETF lists (top 3); on ~100 shares it holds 15.
PARAMS = {"random": {"decide_prob": 0.25, "exposure": 0.9}, "momentum": {"top_k": 15}}
AI_CONTESTANTS = ["desk", "panel", "silo_agent"]
# Daily-bar books (silver, crypto): rules that act on daily signals, plus a random control
# that rebalances about every ten days.
DAILY_CONTESTANTS = ["tsmom", "donchian", "rsi2", "bollinger", "random"]
DAILY_PARAMS = {"random": {"decide_prob": 0.1, "exposure": 0.9}}


def book_names(paper_dir: str | Path) -> list[str]:
    root = Path(paper_dir)
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if (p / "config.json").exists())


def create_league(symbols, vault: DataVault, paper_dir="paper", prefix="league", with_ai=False,
                  cash=100_000.0, constraints=None, costs=None, daily=False) -> list[dict]:
    made = []
    contestants = DAILY_CONTESTANTS if daily else RULE_CONTESTANTS + (AI_CONTESTANTS if with_ai else [])
    params = DAILY_PARAMS if daily else PARAMS
    for strat in contestants:
        book = PaperBook(f"{prefix}-{strat}", paper_dir)
        if book.exists:
            continue
        made.append(book.create(strat, symbols, vault, params.get(strat, {}), cash, constraints, costs))
    return made


def fetch_for_books(vault: DataVault, paper_dir, fetcher, log=print) -> list[str]:
    """One new snapshot per symbol used by any book. Returns symbols that failed."""
    symbols = sorted({s for n in book_names(paper_dir) for s in PaperBook(n, paper_dir).config()["symbols"]})
    failed = []
    for s in symbols:
        try:
            vault.put_bars(s, fetcher(s))
        except Exception as exc:  # noqa: BLE001 - keep going; report at the end
            failed.append(s)
            log(f"fetch failed for {s}: {exc}")
    return failed


def run_all(vault: DataVault, paper_dir="paper", log=print) -> list[dict]:
    """Advance every book; return a leaderboard sorted by excess over buy-and-hold."""
    board = []
    for name in book_names(paper_dir):
        book = PaperBook(name, paper_dir)
        try:
            book.run(vault)
        except Exception as exc:  # noqa: BLE001 - one broken book must not stop the rest
            log(f"{name}: run failed: {exc}")
            board.append({"book": name, "error": str(exc)})
            continue
        st = book.status()
        eq, bench = st["equity"], st["benchmark_equity"]
        cash0 = book.config()["cash"]
        board.append({
            "book": name, "strategy": st["strategy"], "days": st["days"], "last_date": st["last_date"],
            "return": eq / cash0 - 1, "benchmark_return": bench / cash0 - 1,
            "excess": (eq - bench) / cash0, "positions": len(st["positions"]), "fees": st["fees_paid"],
        })
    board.sort(key=lambda r: -r.get("excess", float("-inf")))
    return board


def write_leaderboard(board: list[dict], paper_dir="paper") -> Path:
    path = Path(paper_dir) / "leaderboard.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(board, indent=2, default=str))
    return path
