"""Backtest every league contestant on the same stocks, side by side.

The paper league takes weeks to say anything. This replays history instead:
every contestant (rules, AI and Silo's own agent) runs through the same
engine, costs, liquidity and position-trading rules over the same period,
and one report compares them with buy-and-hold, overall and by market
conditions.

AI contestants and the training cutoff: a model may remember what happened
after a date it was trained on. Silo hides tickers and dates from the
model, but numbers can still be recognisable, so AI results are labelled
*clean* only for periods after the model's cutoff and *contaminated*
before it. Run AI contestants from the cutoff onwards for a fair test.

The universe is today's candidate list, so every backtest here carries
survivorship bias (firms that failed or left the index are missing). The
engine's liquidity rules are applied point-in-time, so a stock only becomes
tradable once it trades enough.
"""

from __future__ import annotations

import html
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from silo.engine import MarketData, Session
from silo.firewall import is_context
from silo.lab.regimes import regime_table
from silo.metrics import summarize
from silo.strategies import REGISTRY

DISCLAIMER = "Research only. Simulated results, not financial advice. Silo never trades real money."
from silo.league import PARAMS  # noqa: E402 - same settings as the paper league


@dataclass
class Entry:
    name: str
    equity: pd.Series
    fills: pd.DataFrame
    metrics: dict
    regimes: list[dict]
    seconds: float
    llm_calls: int = 0
    cache_hits: int = 0
    cutoff: str | None = None
    status: str = "rules"  # rules | clean | contaminated
    extra: dict = field(default_factory=dict)


def _clients(strategy):
    from silo.agents.contestants import LLMStrategy

    out = [strategy] + list(getattr(strategy, "members", {}).values())
    return [a.client for a in out if isinstance(a, LLMStrategy) and a.client is not None]


def run_one(name, symbols, data: MarketData, start, end, costs, constraints, progress=print) -> Entry:
    strat = REGISTRY[name](symbols, **PARAMS.get(name, {}))
    session = Session.start(strat, 100_000.0, costs, constraints)
    lo = None if start is None else pd.Timestamp(start)
    hi = None if end is None else pd.Timestamp(end)
    days = [d for d in data.dates if (lo is None or d >= lo) and (hi is None or d <= hi)]
    t0, last_report = time.time(), time.time()
    for i, d in enumerate(days):
        session.step(d, data)
        if progress and strat.uses_llm and time.time() - last_report > 60:
            last_report = time.time()
            progress(f"  {name}: {d.date()} ({i + 1}/{len(days)} days, {time.time() - t0:.0f}s)")
    res = session.result()
    clients = _clients(strat)
    cutoff = getattr(strat, "model_cutoff", None) if strat.uses_llm else None
    status = "rules"
    if strat.uses_llm:
        clean = cutoff not in (None, "unknown") and days and days[0] > pd.Timestamp(cutoff) + pd.offsets.MonthEnd(0)
        status = "clean" if clean else "contaminated"
        if cutoff is None:  # the offline rulebot has no cutoff and is not an AI
            status = "rulebot"
    return Entry(
        name, res.equity, res.fills, summarize(res.equity, res.fills), [], time.time() - t0,
        sum(c.calls for c in clients), sum(c.cache_hits for c in clients), cutoff, status,
        {"fees": res.fees_paid, "dividends": res.dividends},
    )


def run_league_backtest(names, symbols, tables, start=None, end=None, costs=None, constraints=None,
                        progress=print) -> tuple[list[Entry], Entry]:
    data = MarketData({s: tables[s] for s in symbols} | {k: v for k, v in tables.items() if is_context(k)})
    bench = run_one("buy_and_hold", symbols, data, start, end, costs, None, progress)
    entries = []
    for n in names:
        if progress:
            progress(f"running {n}...")
        e = run_one(n, symbols, data, start, end, costs, constraints, progress)
        e.regimes = regime_table(e.equity, bench.equity)
        entries.append(e)
        if progress:
            m = e.metrics
            progress(f"  {n}: return {m['total_return']:.1%}, Sharpe {m['sharpe']:.2f}, "
                     f"max drawdown {m['max_drawdown']:.1%} ({e.seconds:.0f}s, {e.llm_calls} model calls)")
    return entries, bench


def to_json(entries, bench) -> dict:
    def row(e):
        return {"name": e.name, "metrics": e.metrics, "regimes": e.regimes, "seconds": e.seconds,
                "llm_calls": e.llm_calls, "cache_hits": e.cache_hits, "cutoff": e.cutoff, "status": e.status,
                **e.extra, "start": str(e.equity.index[0].date()), "end": str(e.equity.index[-1].date())}
    return {"benchmark": row(bench), "contestants": [row(e) for e in entries], "disclaimer": DISCLAIMER}


COLORS = ["#2a78d6", "#d6862a", "#2aa876", "#c2449a", "#7a5cd6", "#c9a227", "#38a3c4", "#d64545"]


def _svg(entries, bench, w=900, h=320) -> str:
    series = [("Buy-and-hold", bench.equity, "var(--ink-2)")] + [
        (e.name, e.equity, COLORS[i % len(COLORS)]) for i, e in enumerate(entries)]
    idx = bench.equity.index
    vals = [100 * s.reindex(idx).ffill() / s.reindex(idx).ffill().iloc[0] for _, s, _ in series]
    lo = min(float(np.nanmin(v)) for v in vals)
    hi = max(float(np.nanmax(v)) for v in vals)
    step = max(1, len(idx) // 400)
    paths = []
    for (name, _, col), v in zip(series, vals):
        pts = " ".join(f"{i / (len(idx) - 1) * w:.1f},{h - (x - lo) / (hi - lo + 1e-9) * h:.1f}"
                       for i, x in list(enumerate(v.to_numpy()))[::step] if not np.isnan(x))
        dash = ' stroke-dasharray="5 4"' if name == "Buy-and-hold" else ""
        paths.append(f'<polyline fill="none" stroke="{col}" stroke-width="1.6"{dash} points="{pts}">'
                     f"<title>{html.escape(name)}</title></polyline>")
    legend = "".join(f'<span><i style="background:{c}"></i>{html.escape(n)}</span>' for n, _, c in series)
    return (f'<div class="legend">{legend}</div><svg viewBox="0 0 {w} {h}" role="img" '
            f'aria-label="Equity of every contestant and buy-and-hold, indexed to 100">{"".join(paths)}</svg>'
            f'<p class="muted">Indexed to 100 at {idx[0].date()}. Range {lo:.0f} to {hi:.0f}.</p>')


def _pct(x):
    return "n/a" if x is None or pd.isna(x) else f"{x:.1%}"


def render(entries, bench, title="League backtest") -> str:
    b = bench.metrics
    head = ("<tr><th scope='col'>Contestant</th><th scope='col'>Total return</th><th scope='col'>CAGR</th>"
            "<th scope='col'>Sharpe</th><th scope='col'>Max drawdown</th><th scope='col'>Round trips</th>"
            "<th scope='col'>Avg hold (days)</th><th scope='col'>Model calls</th><th scope='col'>AI test</th></tr>")
    rows = [f"<tr><th scope='row'>Buy-and-hold</th><td>{_pct(b['total_return'])}</td><td>{_pct(b['cagr'])}</td>"
            f"<td>{b['sharpe']:.2f}</td><td>{_pct(b['max_drawdown'])}</td><td></td><td></td><td></td><td></td></tr>"]
    label = {"rules": "no model", "clean": "after cutoff", "contaminated": "before cutoff: treat with caution",
             "rulebot": "offline stand-in"}
    for e in sorted(entries, key=lambda e: -e.metrics.get("sharpe", 0)):
        m = e.metrics
        cls = " class='warn'" if e.status == "contaminated" else ""
        rows.append(f"<tr{cls}><th scope='row'>{html.escape(e.name)}</th><td>{_pct(m['total_return'])}</td>"
                    f"<td>{_pct(m['cagr'])}</td><td>{m['sharpe']:.2f}</td><td>{_pct(m['max_drawdown'])}</td>"
                    f"<td>{m.get('round_trips', 0):.0f}</td><td>{m.get('avg_hold_days', 0):.0f}</td>"
                    f"<td>{e.llm_calls + e.cache_hits or ''}</td><td>{label[e.status]}"
                    f"{' (' + html.escape(e.cutoff) + ')' if e.cutoff else ''}</td></tr>")
    states = [r["state"] for r in (entries[0].regimes if entries else [])]
    rhead = "<tr><th scope='col'>Market state</th><th scope='col'>Days</th><th scope='col'>Buy-and-hold</th>" + "".join(
        f"<th scope='col'>{html.escape(e.name)}</th>" for e in entries) + "</tr>"
    rrows = []
    for i, st in enumerate(states):
        r0 = entries[0].regimes[i]
        cells = "".join(
            f"<td class='{'pos' if e.regimes[i]['strategy'] >= e.regimes[i]['benchmark'] else 'neg'}'>"
            f"{_pct(e.regimes[i]['strategy'])}</td>" for e in entries)
        rrows.append(f"<tr><th scope='row'>{html.escape(st)}</th><td>{r0['days']}</td>"
                     f"<td>{_pct(r0['benchmark'])}</td>{cells}</tr>")
    span = f"{bench.equity.index[0].date()} to {bench.equity.index[-1].date()}"
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>{html.escape(title)}</title>
<style>
:root {{ --bg:#fbfbfa; --ink:#1d1d1b; --ink-2:#6b6b66; --line:#e3e2de; --pos:#1f7a4d; --neg:#b3261e; --warn:#fff4d6; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#1a1a19; --ink:#ecebe6; --ink-2:#a3a29c; --line:#34332f;
  --pos:#5cc88f; --neg:#f08a80; --warn:#3a3215; }} }}
body {{ background:var(--bg); color:var(--ink); font:15px/1.5 system-ui,sans-serif; margin:0 auto; max-width:1000px; padding:24px 16px; }}
h1 {{ font-size:22px; margin:0 0 4px }} h2 {{ font-size:17px; margin:28px 0 8px }}
.muted {{ color:var(--ink-2) }} .scroll {{ overflow-x:auto }}
table {{ border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums }}
th,td {{ text-align:right; padding:6px 8px; border-bottom:1px solid var(--line); white-space:nowrap }}
th[scope=row], th:first-child {{ text-align:left }} tr.warn {{ background:var(--warn) }}
td.pos {{ color:var(--pos) }} td.neg {{ color:var(--neg) }}
svg {{ width:100%; height:auto; border:1px solid var(--line); border-radius:6px }}
.legend {{ display:flex; flex-wrap:wrap; gap:12px; margin:8px 0; font-size:13px }}
.legend i {{ display:inline-block; width:12px; height:3px; margin-right:6px; vertical-align:middle }}
.note {{ border-left:3px solid var(--ink-2); padding:4px 12px; }}
</style></head><body>
<h1>{html.escape(title)}</h1>
<p class="muted">{span}. {len(entries)} contestants on the same stocks, costs (stamp duty, spread, fees),
liquidity rules and position-trading rules. {DISCLAIMER}</p>
<h2>Equity</h2>{_svg(entries, bench)}
<h2>Results</h2><div class="scroll"><table><thead>{head}</thead><tbody>{''.join(rows)}</tbody></table></div>
<p class="muted">Sorted by Sharpe ratio. Highlighted rows are AI results from before the model's training
cutoff: the model may remember what happened, so they are not a fair test.</p>
<h2>By market conditions</h2>
<p class="muted">Return earned on the days in each state. Green beats buy-and-hold in that state, red trails it.</p>
<div class="scroll"><table><thead>{rhead}</thead><tbody>{''.join(rrows)}</tbody></table></div>
<h2>Caveats</h2><ul class="note">
<li>The stock list is today's, so firms that failed or left the index are missing (survivorship bias). Real results would be somewhat worse.</li>
<li>One history is one sample. A strategy that wins here can still be lucky; use <code>silo lab</code> for walk-forward and luck checks.</li>
<li>Prices from Yahoo's free feed; a few series have gaps or stale stretches.</li></ul>
</body></html>"""


def write(entries, bench, out_dir, title="League backtest") -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    page = out / "league.html"
    page.write_text(render(entries, bench, title), encoding="utf-8")
    data = out / "league.json"
    data.write_text(json.dumps(to_json(entries, bench), indent=2, default=str), encoding="utf-8")
    return page, data


def data_check(tables: dict[str, pd.DataFrame], recent: int = 252) -> pd.DataFrame:
    """Flag stale or broken price series: repeated closes, zero volume, gaps, old last bar."""
    rows = {}
    newest = max(df.index[-1] for df in tables.values())
    for s, df in tables.items():
        last = df.iloc[-recent:]
        c = last["close"]
        same = (c.diff() == 0).astype(int)
        run = int(same.groupby((same == 0).cumsum()).cumsum().max()) if len(same) else 0
        vol = float(np.log(c).diff().std() * np.sqrt(252)) if len(c) > 2 else float("nan")
        rows[s] = {
            "bars": len(df), "last_bar": df.index[-1].date(), "days_behind": int(np.busday_count(df.index[-1].date(), newest.date())),
            "vol_1y": vol, "flat_days_pct": float(same.mean()), "longest_flat_run": run,
            "zero_volume_pct": float((last["volume"] <= 0).mean()) if "volume" in last else float("nan"),
        }
    out = pd.DataFrame(rows).T
    out["suspect"] = (out["longest_flat_run"] >= 5) | (out["flat_days_pct"] > 0.2) | (out["days_behind"] > 5) \
        | (out["zero_volume_pct"] > 0.2) | (out["vol_1y"] < 0.08)
    return out.sort_values(["suspect", "flat_days_pct"], ascending=[False, False])
