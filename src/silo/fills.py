"""Real Practice fills next to simulated fills: how far is SimBroker from a broker?

Trading 212's API serves no price history, but it does report how each
Practice order actually filled: price, time and the taxes and fees charged.
For a book mirrored to the Practice account, this module

1. collects those fills for every order the mirror sent, appending them to the
   book's ``mirror.jsonl`` (append-only, like the rest of that log), and
2. pairs each one with SimBroker's fill for the same rebalance and symbol, to
   report the gap: price slippage in basis points, fees as a share of traded
   value, and how long after the decision each venue filled.

SimBroker stays the book's record. The gap is a calibration check on its
:class:`~silo.broker.CostModel`, not a correction applied to results.
Practice fills are simulated by Trading 212 too, so they approximate a real
broker's behaviour; they are not real-money executions.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from statistics import median
from typing import Any

import pandas as pd

from silo.broker import CostModel

DISCLAIMER = ("Trading 212 Practice fills against Silo's simulator. Research only, paper money, "
              "not financial advice.")

# Instrument currencies quoted in minor units: price / 100 gives the major unit.
_MINOR = {"GBX": "GBP"}


def _f(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def parse_history_item(item: dict) -> dict | None:
    """One historical order fill from Trading 212, flattened for Silo's log.

    The documented shape is ``{"order": {...}, "fill": {..., "walletImpact":
    {..., "taxes": [...]}}}``. Returns None when there is no trade fill
    (an order cancelled or rejected, or a corporate-action entry).
    """
    order, fill = item.get("order") or {}, item.get("fill") or {}
    if fill.get("type") not in (None, "TRADE"):
        return None
    wallet = fill.get("walletImpact") or {}
    out = {
        "order_id": order.get("id"),
        "fill_id": fill.get("id"),
        "ticker": order.get("ticker") or (order.get("instrument") or {}).get("ticker"),
        "status": order.get("status"),
        "created_at": order.get("createdAt"),
        "filled_at": fill.get("filledAt"),
        "fill_price": _f(fill.get("price")),
        "fill_qty": _f(fill.get("quantity")),
        "instrument_currency": order.get("currency") or (order.get("instrument") or {}).get("currency"),
        "account_currency": wallet.get("currency"),
        "fx_rate": _f(wallet.get("fxRate")),
        "net_value": _f(wallet.get("netValue")),
        "taxes": [{"name": str(t.get("name", "")), "amount": abs(_f(t.get("quantity")) or 0.0)}
                  for t in (wallet.get("taxes") or []) if isinstance(t, dict)],
    }
    out["fees"] = sum(t["amount"] for t in out["taxes"])
    if out["order_id"] is None or not out["fill_price"] or not out["fill_qty"]:
        return None
    return out


def _combine(parts: list[dict]) -> dict:
    """Several fills of one order become one volume-weighted fill."""
    if len(parts) == 1:
        return parts[0]
    qty = sum(abs(p["fill_qty"]) for p in parts)
    out = dict(parts[0])
    out["fill_qty"] = qty * (1 if parts[0]["fill_qty"] > 0 else -1)
    out["fill_price"] = sum(abs(p["fill_qty"]) * p["fill_price"] for p in parts) / qty
    out["filled_at"] = max(str(p["filled_at"]) for p in parts)
    out["fill_id"] = [p["fill_id"] for p in parts]
    out["taxes"] = [t for p in parts for t in p["taxes"]]
    out["fees"] = sum(p["fees"] for p in parts)
    out["net_value"] = sum(p["net_value"] or 0.0 for p in parts)
    return out


FINAL = {"FILLED", "CANCELLED", "REJECTED"}


def _price_in_book_currency(e: dict, book_ccy: str) -> float:
    """Practice fill price in the book's currency (pounds for London shares)."""
    ccy = str(e.get("instrument_currency") or "GBX").upper()
    price = e["fill_price"]
    if ccy in _MINOR:
        price, ccy = price / 100.0, _MINOR[ccy]
    if ccy != book_ccy.upper() and e.get("fx_rate"):
        price *= e["fx_rate"]
    return price


def _sent_orders(log: list[dict]) -> list[dict]:
    return [e for e in log if e.get("kind") == "sent" and e.get("order_id") is not None]


def _filled_ids(log: list[dict]) -> set:
    return {str(e["order_id"]) for e in log if e.get("kind") == "fill"}


def collect_fills(book, client, max_pages: int = 10) -> list[dict]:
    """Ask Trading 212 how each mirrored order filled; log each fill once.

    Read-only on the broker side: it only lists historical orders. An order
    is logged once Trading 212 marks it final (filled, or cancelled after a
    partial fill). Orders sent after the close fill at the next open, so they
    are simply looked for again on the next run.
    """
    from silo.paper import _mirror_append, _mirror_log

    log = _mirror_log(book)
    waiting = {str(e["order_id"]): e for e in _sent_orders(log)}
    for oid in _filled_ids(log):
        waiting.pop(oid, None)
    if not waiting:
        return []
    earliest = min(str(e.get("at") or e["decision_date"]) for e in waiting.values())
    found: dict[str, list[dict]] = {}
    for item in client.order_history(max_pages=max_pages, stop_before=earliest):
        fill = parse_history_item(item)
        if fill is not None and str(fill["order_id"]) in waiting:
            found.setdefault(str(fill["order_id"]), []).append(fill)
    new = []
    for oid, parts in found.items():
        if not any(p["status"] in FINAL for p in parts):
            continue  # partly filled and still working: wait for the rest
        sent = waiting[oid]
        fill = _combine(parts)
        entry = {
            "kind": "fill",
            "decision_date": sent["decision_date"],
            "ticker": fill["ticker"] or sent["ticker"],
            "order_id": fill["order_id"],
            "ordered_qty": sent["quantity"],
            "sent_at": _intent_time(log, sent),
            **{k: v for k, v in fill.items() if k not in ("order_id", "ticker")},
            "instrument_currency": (fill.get("instrument_currency") or "GBX").upper(),
            "logged_at": datetime.now(timezone.utc).isoformat(),
        }
        _mirror_append(book, entry)
        new.append(entry)
    return new


def _intent_time(log: list[dict], sent: dict) -> str | None:
    for e in log:
        if (e.get("kind") == "intent" and e.get("decision_date") == sent["decision_date"]
                and e.get("ticker") == sent["ticker"]):
            return e.get("at")
    return None


def _day(t) -> pd.Timestamp:
    """Trading day of a decision: bar dates are naive, decision times are the UTC close."""
    t = pd.Timestamp(t)
    return (t.tz_convert(None) if t.tzinfo else t).normalize()


def _sim_fill_for(fills: list[dict], decision_date: pd.Timestamp, symbol: str) -> dict | None:
    day = _day(decision_date)
    hits = [f for f in fills if f["symbol"] == symbol and _day(f["decided_at"]) == day]
    if not hits:
        return None
    qty = sum(f["qty"] for f in hits)
    value = sum(abs(f["qty"]) * f["price"] for f in hits)
    return {"date": pd.Timestamp(hits[0]["date"]), "qty": qty, "price": value / abs(qty) if qty else hits[0]["price"],
            "fee": sum(f["fee"] for f in hits)}


def _hours(a, b) -> float | None:
    try:
        ta, tb = pd.Timestamp(a), pd.Timestamp(b)
    except (TypeError, ValueError):
        return None
    if ta is pd.NaT or tb is pd.NaT:
        return None
    ta = ta.tz_localize("UTC") if ta.tzinfo is None else ta
    tb = tb.tz_localize("UTC") if tb.tzinfo is None else tb
    return (tb - ta).total_seconds() / 3600


def fill_gap(book) -> pd.DataFrame:
    """One row per Practice fill, beside SimBroker's fill for the same rebalance.

    Slippage is signed so that a positive number is worse for the book: paying
    more on a buy, receiving less on a sell. ``*_vs_open_bps`` measure each
    venue from the same reference, the day's open that SimBroker started from.
    Fees are a share of traded value so that books of different size compare.
    """
    from silo.paper import _mirror_log, load_mapping

    cfg = book.config()
    costs = CostModel(**cfg["costs"])
    bump_bps = costs.spread_bps / 2 + costs.slippage_bps
    session = book._load()["book"]
    by_ticker = {t: s for s, t in load_mapping(book).items()}
    rows = []
    for e in _mirror_log(book):
        if e.get("kind") != "fill":
            continue
        symbol = by_ticker.get(e["ticker"])
        side = 1 if e["ordered_qty"] > 0 else -1
        price = _price_in_book_currency(e, cfg.get("currency", "GBP"))
        qty = abs(e["fill_qty"])
        value = qty * price
        row = {
            "decision_date": pd.Timestamp(e["decision_date"]).date(),
            "symbol": symbol or "?",
            "ticker": e["ticker"],
            "side": "buy" if side > 0 else "sell",
            "practice_qty": qty * side,
            "practice_price": price,
            "practice_fees": e.get("fees", 0.0),
            "practice_fee_bps": e.get("fees", 0.0) / value * 1e4 if value else float("nan"),
            "practice_filled_at": e.get("filled_at"),
            "practice_delay_h": _hours(e.get("sent_at"), e.get("filled_at")),
        }
        sim = _sim_fill_for(session.broker.fills, pd.Timestamp(e["decision_date"]), symbol) if symbol else None
        if sim is not None:
            sim_value = abs(sim["qty"]) * sim["price"]
            open_price = sim["price"] / (1 + side * bump_bps / 1e4)
            row.update({
                "sim_qty": sim["qty"],
                "sim_price": sim["price"],
                "sim_fee_bps": sim["fee"] / sim_value * 1e4 if sim_value else float("nan"),
                "sim_fill_date": sim["date"].date(),
                "slippage_bps": side * (price - sim["price"]) / sim["price"] * 1e4,
                "practice_vs_open_bps": side * (price - open_price) / open_price * 1e4,
                "sim_vs_open_bps": bump_bps,
            })
            row["fee_gap_bps"] = row["practice_fee_bps"] - row["sim_fee_bps"]
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_gap(gap: pd.DataFrame, costs: CostModel | None = None) -> dict:
    """Medians across fills, and what CostModel would need to match them."""
    if gap.empty:
        return {"fills": 0}
    paired = gap.dropna(subset=["sim_price"]) if "sim_price" in gap else gap.iloc[0:0]
    out = {"fills": int(len(gap)), "paired": int(len(paired))}
    if len(paired):
        out["median_slippage_bps"] = float(median(paired["slippage_bps"]))
        out["median_practice_vs_open_bps"] = float(median(paired["practice_vs_open_bps"]))
        out["sim_vs_open_bps"] = float(paired["sim_vs_open_bps"].iloc[0])
        out["median_fee_gap_bps"] = float(median(paired["fee_gap_bps"].dropna())) if paired["fee_gap_bps"].notna().any() else None
        buys = paired[paired["side"] == "buy"]
        if len(buys):
            out["median_practice_buy_fee_bps"] = float(median(buys["practice_fee_bps"]))
    delays = gap["practice_delay_h"].dropna()
    if len(delays):
        out["median_delay_h"] = float(median(delays))
    if costs is not None and "median_practice_vs_open_bps" in out and out["paired"] >= 10:
        # Keep the spread assumption; move only slippage, never below zero.
        out["suggested_slippage_bps"] = round(max(0.0, out["median_practice_vs_open_bps"] - costs.spread_bps / 2), 1)
    return out


def format_gap(gap: pd.DataFrame, summary: dict, currency: str = "GBP") -> str:
    if gap.empty:
        return "No Practice fills logged yet. Orders sent after the close fill at the next open; run again tomorrow."
    cols = ["decision_date", "symbol", "side", "practice_price", "sim_price", "slippage_bps",
            "practice_fee_bps", "sim_fee_bps", "practice_delay_h", "sim_fill_date"]
    table = gap[[c for c in cols if c in gap]].to_string(index=False, float_format=lambda v: f"{v:,.2f}")
    lines = [table, "", f"{summary['fills']} Practice fill(s), {summary['paired']} paired with a simulator fill"]
    if summary.get("paired"):
        lines.append(f"median price gap: {summary['median_slippage_bps']:+.1f} bp (positive = Practice worse than simulator)")
        lines.append(f"from the open: Practice {summary['median_practice_vs_open_bps']:+.1f} bp, "
                     f"simulator assumes {summary['sim_vs_open_bps']:+.1f} bp")
        if summary.get("median_fee_gap_bps") is not None:
            lines.append(f"median fee gap: {summary['median_fee_gap_bps']:+.1f} bp of traded value")
    if "median_delay_h" in summary:
        lines.append(f"median time from order sent to Practice fill: {summary['median_delay_h']:.1f} h")
    if "suggested_slippage_bps" in summary:
        lines.append(f"CostModel slippage that would match: {summary['suggested_slippage_bps']} bp (not applied)")
    elif summary.get("paired"):
        lines.append("too few paired fills to suggest a CostModel change (need 10)")
    lines.append(f"prices in {currency}. {DISCLAIMER}")
    return "\n".join(lines)


def gap_report(book) -> tuple[pd.DataFrame, dict]:
    cfg = book.config()
    gap = fill_gap(book)
    summary = summarize_gap(gap, CostModel(**cfg["costs"]))
    return gap, summary


def save_gap(book, gap: pd.DataFrame, summary: dict) -> None:
    """Derived report files, rewritten each run (the logs they come from are not)."""
    gap.to_csv(book.dir / "fill_gap.csv", index=False)
    (book.dir / "fill_gap.json").write_text(json.dumps(summary, indent=2))
