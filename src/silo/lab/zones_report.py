"""Supply and demand zones today, per stock, as an HTML page for the dashboard.

For each stock: the latest close, the best demand zone below (where the
Core Strategy would buy), the nearest supply zone above (where it would take
profit, and which decides whether the trade has room for 1:4), and a plain
verdict: in the buy zone, wait for a pull-back, supply too close, or no trade.
A small chart draws the last year of closes with demand zones in green and
supply zones in red. Research only: nothing here places an order.
"""

from __future__ import annotations

import html
import math
from dataclasses import dataclass

import pandas as pd

from silo.strategies.core_zones import Bars, Zone, analyze, find_zones

DISCLAIMER = "Research only. Simulated analysis of past prices; not investment advice."


@dataclass
class ZoneRow:
    symbol: str
    date: str
    close: float
    verdict: str
    detail: str
    score: float | None
    buy_at: float | None
    stop: float | None
    supply: tuple[float, float] | None  # nearest active supply above: (proximal, distal)
    reward_risk: float | None
    demand_zones: list[Zone]
    supply_zones: list[Zone]
    closes: list[float]


def zone_row(symbol: str, daily: pd.DataFrame, min_reward_risk: float = 4.0, min_score: float = 7.0,
             target_frac: float = 0.75, lookback: int = 260) -> ZoneRow | None:
    """Read one stock's zones at its last bar (pass point-in-time bars)."""
    a = analyze(daily)
    if a is None:
        return None
    window = daily.iloc[-lookback:]
    zones = find_zones(Bars.from_frame(window))
    demand = [z for z in zones if z.kind == "demand" and not z.violated]
    supply = [z for z in zones if z.kind == "supply" and not z.used_up]
    close = a.price
    above = sorted((z for z in supply if z.proximal > close), key=lambda z: z.proximal)
    nearest = (above[0].proximal, above[0].distal) if above else None
    st = a.best
    score = buy_at = stop = rr = None
    if st is None:
        verdict, detail = "no trade", "no fresh demand zone below the price"
    else:
        z = st.zone
        score, buy_at, stop = st.score, z.proximal, st.stop
        risk = z.proximal - st.stop
        if st.target is not None and risk > 0:
            rr = (z.proximal + target_frac * (st.target - z.proximal) - z.proximal) / risk
        else:
            rr = math.inf
        if st.decision == "no_action":
            verdict, detail = "no trade", f"trend {st.trend_label} at this height in the range"
        elif st.score < min_score:
            verdict, detail = "no trade", f"zone scores {st.score:.1f}/10, below {min_score:g}"
        elif rr < min_reward_risk:
            verdict, detail = "supply too close", f"room to supply is {rr:.1f}x the risk, needs {min_reward_risk:g}x"
        elif close <= z.proximal or z.in_zone_now:
            verdict, detail = "in buy zone", f"price is back at demand {z.proximal:.2f}, stop {st.stop:.2f}"
        else:
            verdict = "wait for pull-back"
            detail = f"buy zone {z.proximal:.2f} is {100 * (close / z.proximal - 1):.1f}% below the price"
    return ZoneRow(symbol, str(pd.Timestamp(daily.index[-1]).date()), close, verdict, detail, score, buy_at, stop,
                   nearest, rr, demand, supply, [float(x) for x in window["close"]])


def _chart(r: ZoneRow, w: int = 360, h: int = 120) -> str:
    lo = min(r.closes + [z.distal for z in r.demand_zones] + [z.proximal for z in r.supply_zones])
    hi = max(r.closes + [z.proximal for z in r.demand_zones] + [z.distal for z in r.supply_zones])
    if hi <= lo:
        return ""
    n = len(r.closes)

    def y(v):
        return h - 4 - (v - lo) / (hi - lo) * (h - 8)

    def x(i):
        return i / max(n - 1, 1) * w

    rects = []
    for z, cls in [(z, "demand") for z in r.demand_zones] + [(z, "supply") for z in r.supply_zones]:
        top, bot = max(z.proximal, z.distal), min(z.proximal, z.distal)
        x0 = x(z.base_start)
        rects.append(f"<rect class='{cls}' x='{x0:.1f}' y='{y(top):.1f}' width='{w - x0:.1f}' "
                     f"height='{max(y(bot) - y(top), 1):.1f}'/>")
    pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(r.closes))
    return (f"<svg viewBox='0 0 {w} {h}' width='{w}' height='{h}' role='img' "
            f"aria-label='{html.escape(r.symbol)}: closes with supply and demand zones'>"
            f"{''.join(rects)}<polyline class='px' points='{pts}'/></svg>")


def _num(v, fmt="{:.2f}"):
    if v is None:
        return "–"
    if isinstance(v, float) and math.isinf(v):
        return "open"
    return fmt.format(v)


ORDER = {"in buy zone": 0, "wait for pull-back": 1, "supply too close": 2, "no trade": 3}


def write_zones_report(rows: list[ZoneRow], path) -> None:
    rows = sorted(rows, key=lambda r: (ORDER.get(r.verdict, 9), -(r.score or 0), r.symbol))
    counts = {k: sum(r.verdict == k for r in rows) for k in ORDER}
    with_supply = sum(bool([z for z in r.supply_zones if z.proximal > r.close]) for r in rows)
    body = []
    for r in rows:
        sup = f"{r.supply[0]:.2f}–{r.supply[1]:.2f}" if r.supply else "none in the last year"
        body.append(
            f"<tr><td><b>{html.escape(r.symbol)}</b><br><span class='muted'>{r.date}</span></td>"
            f"<td>{r.close:.2f}</td><td class='v {r.verdict.replace(' ', '-')}'>{r.verdict}<br>"
            f"<span class='muted'>{html.escape(r.detail)}</span></td><td>{_num(r.score, '{:.1f}')}</td>"
            f"<td>{_num(r.buy_at)}</td><td>{_num(r.stop)}</td><td>{sup}</td><td>{_num(r.reward_risk, '{:.1f}')}</td>"
            f"<td>{len(r.demand_zones)} / {len(r.supply_zones)}</td><td>{_chart(r)}</td></tr>")
    summary = " · ".join(f"{v} {k}" for k, v in counts.items())
    page = f"""<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'>
<title>Supply and demand zones</title><style>
:root{{--bg:#fff;--fg:#1a1a1a;--muted:#666;--line:#ddd;--px:#333;--dem:rgba(30,140,70,.22);--sup:rgba(200,40,40,.22)}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151515;--fg:#eee;--muted:#999;--line:#333;--px:#ddd;
--dem:rgba(60,190,110,.28);--sup:rgba(240,80,80,.28)}}}}
body{{background:var(--bg);color:var(--fg);font:14px system-ui,sans-serif;margin:16px}}
table{{border-collapse:collapse;width:100%}}td,th{{border-bottom:1px solid var(--line);padding:6px;text-align:left;
vertical-align:top}}.muted{{color:var(--muted);font-size:12px}}.v{{font-weight:600}}
.in-buy-zone{{color:#1e8c46}}.supply-too-close{{color:#c62828}}rect.demand{{fill:var(--dem)}}
rect.supply{{fill:var(--sup)}}polyline.px{{fill:none;stroke:var(--px);stroke-width:1.2}}
.wrap{{overflow-x:auto}}</style></head><body>
<h1>Supply and demand zones</h1><p class='muted'>{DISCLAIMER}</p>
<p>{len(rows)} stocks: {summary}. {with_supply} have an active supply zone above the price.
Green bands are demand (where the rules buy), red bands are supply (where they take profit). A buy needs room
to the nearest supply of at least 4x the risk.</p>
<div class='wrap'><table><thead><tr><th>Stock</th><th>Close</th><th>Verdict</th><th>Score</th><th>Buy at</th>
<th>Stop</th><th>Next supply</th><th>Reward:risk</th><th>Zones (demand / supply)</th><th>Last year</th></tr></thead>
<tbody>{''.join(body)}</tbody></table></div></body></html>"""
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(page, encoding="utf-8")
