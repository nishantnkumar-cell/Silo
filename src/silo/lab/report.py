"""Self-contained HTML report for a lab study. No external assets."""

from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from silo.lab.regimes import regime_table
from silo.lab.study import ANNUAL, LabResult

DISCLAIMER = "Research only. Simulated results, not financial advice. Silo never trades real money."


def _pct(x: float) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{x:.1%}"


def _num(x: float, nd: int = 2) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{x:.{nd}f}"


def _index100(equity: pd.Series) -> pd.Series:
    return 100 * equity / equity.iloc[0]


def _chart_payload(strategy: pd.Series, bench: pd.Series, randoms: list[pd.Series], markers=()) -> dict:
    idx = strategy.index
    s = _index100(strategy)
    b = _index100(bench.reindex(idx).ffill())
    payload = {
        "dates": [d.strftime("%Y-%m-%d") for d in idx],
        "series": [
            {"name": "Strategy", "slot": 1, "values": [round(v, 2) for v in s]},
            {"name": "Buy-and-hold", "slot": 2, "values": [round(v, 2) for v in b]},
        ],
        "markers": [pd.Timestamp(m).strftime("%Y-%m-%d") for m in markers],
    }
    if randoms:
        mat = np.vstack([_index100(r.reindex(idx).ffill()).to_numpy() for r in randoms])
        payload["band"] = {
            "name": "Random strategies, 5th to 95th percentile",
            "lo": [round(v, 2) for v in np.nanpercentile(mat, 5, axis=0)],
            "hi": [round(v, 2) for v in np.nanpercentile(mat, 95, axis=0)],
        }
    return payload


def _metrics_rows(rows: list[tuple[str, dict]]) -> str:
    out = []
    for name, m in rows:
        out.append(
            f"<tr><th scope='row'>{html.escape(name)}</th><td>{_pct(m.get('total_return'))}</td>"
            f"<td>{_pct(m.get('cagr'))}</td><td>{_num(m.get('sharpe'))}</td>"
            f"<td>{_pct(m.get('volatility'))}</td><td>{_pct(m.get('max_drawdown'))}</td>"
            f"<td>{_num(m.get('round_trips'), 0)}</td><td>{_num(m.get('avg_hold_days'), 0)}</td></tr>"
        )
    return "\n".join(out)


def _regime_html(result: LabResult) -> str:
    rows = regime_table(result.oos_equity, result.benchmark.equity)
    if not rows:
        return "<p class='muted'>Not enough out-of-sample history to label market conditions.</p>"
    body = "\n".join(
        f"<tr><th scope='row'>{html.escape(r['state'])}</th><td>{r['days']}</td><td>{_pct(r['strategy'])}</td>"
        f"<td>{_pct(r['benchmark'])}</td><td>{_pct(r['strategy'] - r['benchmark'])}</td></tr>"
        for r in rows
    )
    return ("<div class='scroll'><table><thead><tr><th scope='col'>Market state</th><th scope='col'>Days</th>"
            "<th scope='col'>Strategy</th><th scope='col'>Buy-and-hold</th><th scope='col'>Difference</th></tr></thead>"
            f"<tbody>{body}</tbody></table></div>")


def render(result: LabResult) -> str:
    wf = result.wf
    oos_m = result.oos_metrics()
    bench_m = result.benchmark_metrics()
    rand_srs = [r.metrics().get("sharpe", 0.0) for r in result.randoms]
    rand_m = {
        "total_return": float(np.median([r.metrics()["total_return"] for r in result.randoms])) if result.randoms else None,
        "sharpe": float(np.median(rand_srs)) if rand_srs else None,
    }
    verdict = (
        "Passes every check. Worth a closer look, still research only."
        if result.interesting
        else "Not distinguishable from the benchmark or from luck."
    )
    checks = "\n".join(
        f"<li class='{'pass' if c.passed else 'fail'}'><span class='mark' aria-hidden='true'>"
        f"{'&#10003;' if c.passed else '&#10007;'}</span><span><strong>{html.escape(c.name)}</strong>"
        f"<span class='sr'>{' passed' if c.passed else ' failed'}</span><br>"
        f"<span class='muted'>{html.escape(c.detail)}</span></span></li>"
        for c in result.checks
    )
    folds = "\n".join(
        f"<tr><td>{f.train_start.date()} to {f.train_end.date()}</td><td>{f.test_start.date()} to {f.test_end.date()}</td>"
        f"<td><code>{html.escape(json.dumps(f.params))}</code></td><td>{_num(f.train_sharpe * ANNUAL)}</td>"
        f"<td>{_num(f.test_sharpe * ANNUAL)}</td></tr>"
        for f in wf.folds
    )
    charts = {
        "oos": _chart_payload(
            result.oos_equity, result.benchmark.equity, [r.equity for r in result.randoms],
            markers=[f.test_start for f in wf.folds[1:]],
        )
    }
    holdout_html = "<p class='muted'>The locked holdout was not opened in this run. Open it once, when the strategy is final, with <code>--open-holdout</code>.</p>"
    if result.holdout:
        h = result.holdout
        charts["holdout"] = _chart_payload(h["equity"], h["benchmark_equity"], [])
        warn = (
            f"<p class='warn'>This holdout had already been opened {h['previous_openings']} time(s) for these settings, so it is no longer a clean test.</p>"
            if h["reopened"] else ""
        )
        holdout_html = f"""{warn}
<p>Parameters chosen on the most recent research window: <code>{html.escape(json.dumps(h['params']))}</code></p>
<div class="legend"><span style="--c: var(--series-1)">Strategy</span><span style="--c: var(--series-2)">Buy-and-hold</span></div>
<div class="chart" id="chart-holdout" role="img" aria-label="Holdout equity of the strategy and buy-and-hold, indexed to 100"></div>
<table><thead><tr><th scope="col"></th><th scope="col">Total return</th><th scope="col">CAGR</th><th scope="col">Sharpe</th><th scope="col">Volatility</th><th scope="col">Max drawdown</th><th scope="col">Round trips</th><th scope="col">Avg hold (days)</th></tr></thead>
<tbody>{_metrics_rows([("Strategy", h['metrics']), ("Buy-and-hold", h['benchmark_metrics'])])}</tbody></table>"""

    cutoff = (
        f"Model training cutoff: {html.escape(result.model_cutoff)}. Results before it are contaminated."
        if result.model_cutoff else "No language model involved, so no training-cutoff contamination."
    )
    title = f"{result.strategy} on {', '.join(result.symbols)}"
    data_rows = "\n".join(
        f"<tr><td>{d}</td><td>{s}</td><td>{b}</td></tr>"
        for d, s, b in zip(charts["oos"]["dates"], charts["oos"]["series"][0]["values"], charts["oos"]["series"][1]["values"])
    )
    return TEMPLATE.format(
        title=html.escape(title),
        disclaimer=DISCLAIMER,
        verdict_class="pass" if result.interesting else "fail",
        verdict=verdict,
        checks=checks,
        cutoff=cutoff,
        oos_span=f"{result.oos_equity.index[0].date()} to {result.oos_equity.index[-1].date()}",
        holdout_start=result.holdout_start.date(),
        metrics=_metrics_rows([("Strategy (walk-forward)", oos_m), ("Buy-and-hold", bench_m)])
        + f"\n<tr><th scope='row'>Random strategies (median of {len(result.randoms)})</th><td>{_pct(rand_m['total_return'])}</td><td></td><td>{_num(rand_m['sharpe'])}</td><td></td><td></td><td></td><td></td></tr>",
        ci=f"{_num(result.sharpe_ci[0])} to {_num(result.sharpe_ci[1])}",
        dsr=_num(result.dsr),
        trials=wf.n_trials,
        folds=folds,
        regimes=_regime_html(result),
        holdout=holdout_html,
        data_rows=data_rows,
        charts=json.dumps(charts),
    )


def write_report(result: LabResult, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(result))
    return path


TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Silo report: {title}</title>
<style>
:root {{
  color-scheme: light;
  --surface: #fcfcfb; --surface-2: #f3f2ef; --text: #0b0b0b; --text-2: #52514e; --rule: #dcdbd6; --grid: #ebeae6;
  --series-1: #2a78d6; --series-2: #eb6834; --band: #8a8984;
  --good: #008300; --bad: #c62a2a; --warn-bg: #fff4d6;
}}
@media (prefers-color-scheme: dark) {{
  :root:where(:not([data-theme="light"])) {{
    color-scheme: dark;
    --surface: #1a1a19; --surface-2: #242423; --text: #f0efec; --text-2: #c3c2b7; --rule: #3a3a38; --grid: #2c2c2a;
    --series-1: #3987e5; --series-2: #d95926; --band: #8f8e88;
    --good: #4fb84f; --bad: #e66767; --warn-bg: #3a3014;
  }}
}}
:root[data-theme="dark"] {{
  color-scheme: dark;
  --surface: #1a1a19; --surface-2: #242423; --text: #f0efec; --text-2: #c3c2b7; --rule: #3a3a38; --grid: #2c2c2a;
  --series-1: #3987e5; --series-2: #d95926; --band: #8f8e88;
  --good: #4fb84f; --bad: #e66767; --warn-bg: #3a3014;
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--surface); color: var(--text); font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }}
main {{ max-width: 920px; margin: 0 auto; padding: 24px 16px 48px; }}
h1 {{ font-size: 22px; margin: 0 0 4px; }} h2 {{ font-size: 17px; margin: 32px 0 8px; }}
.muted {{ color: var(--text-2); }} .banner {{ background: var(--surface-2); border: 1px solid var(--rule); border-radius: 8px; padding: 8px 12px; font-size: 13px; }}
.verdict {{ font-size: 17px; font-weight: 600; margin: 16px 0 8px; }} .verdict.pass {{ color: var(--good); }} .verdict.fail {{ color: var(--bad); }}
ul.checks {{ list-style: none; padding: 0; margin: 0; }} ul.checks li {{ display: flex; gap: 10px; padding: 6px 0; border-bottom: 1px solid var(--grid); }}
.mark {{ font-weight: 700; width: 18px; }} li.pass .mark {{ color: var(--good); }} li.fail .mark {{ color: var(--bad); }}
.sr {{ position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); }}
table {{ border-collapse: collapse; width: 100%; font-size: 14px; margin: 8px 0; font-variant-numeric: tabular-nums; }}
th, td {{ text-align: right; padding: 6px 8px; border-bottom: 1px solid var(--grid); }} th:first-child, td:first-child {{ text-align: left; }}
thead th {{ color: var(--text-2); font-weight: 500; }}
.scroll {{ overflow-x: auto; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 16px; font-size: 13px; color: var(--text-2); margin: 4px 0; }}
.legend span::before {{ content: ""; display: inline-block; width: 14px; height: 3px; border-radius: 2px; margin-right: 6px; vertical-align: middle; background: var(--c); }}
.legend .band::before {{ height: 10px; opacity: .25; }}
.chart {{ position: relative; width: 100%; }} .chart svg {{ display: block; width: 100%; height: auto; overflow: visible; }}
.tip {{ position: absolute; pointer-events: none; background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; padding: 6px 8px; font-size: 12px; white-space: nowrap; box-shadow: 0 2px 8px rgba(0,0,0,.12); display: none; }}
.warn {{ background: var(--warn-bg); border-radius: 6px; padding: 8px 12px; }}
code {{ font-size: 12.5px; }}
details {{ margin-top: 8px; }}
</style></head>
<body><main>
<p class="banner">{disclaimer}</p>
<h1>{title}</h1>
<p class="muted">Out-of-sample period {oos_span}. Locked holdout starts {holdout_start}. {cutoff}</p>

<p class="verdict {verdict_class}">{verdict}</p>
<ul class="checks">{checks}</ul>

<h2>Out-of-sample equity</h2>
<p class="muted">Each test window was traded with parameters chosen only on data before it. Dotted lines mark the start of each new window.</p>
<div class="legend"><span style="--c: var(--series-1)">Strategy</span><span style="--c: var(--series-2)">Buy-and-hold</span><span class="band" style="--c: var(--band)">Random strategies, 5th to 95th percentile</span></div>
<div class="chart" id="chart-oos" role="img" aria-label="Out-of-sample equity of the strategy, buy-and-hold and a band of random strategies, indexed to 100"></div>
<details><summary>Show the data as a table</summary><div class="scroll"><table><thead><tr><th scope="col">Date</th><th scope="col">Strategy (index)</th><th scope="col">Buy-and-hold (index)</th></tr></thead><tbody>{data_rows}</tbody></table></div></details>

<h2>Results</h2>
<div class="scroll"><table><thead><tr><th scope="col"></th><th scope="col">Total return</th><th scope="col">CAGR</th><th scope="col">Sharpe</th><th scope="col">Volatility</th><th scope="col">Max drawdown</th><th scope="col">Round trips</th><th scope="col">Avg hold (days)</th></tr></thead>
<tbody>{metrics}</tbody></table></div>
<p class="muted">Strategy Sharpe 95% bootstrap interval: {ci}. Deflated Sharpe ratio: {dsr} after {trials} parameter trials; 0.95 or more means the result is unlikely to be the luckiest of many tries.</p>

<h2>By market conditions</h2>
<p class="muted">Return earned on the days in each state, out of sample. States come from the benchmark's own past: the trend over the previous three months and whether recent volatility is above its one-year median.</p>
{regimes}

<h2>Walk-forward folds</h2>
<div class="scroll"><table><thead><tr><th scope="col">Training window</th><th scope="col">Test window</th><th scope="col">Chosen parameters</th><th scope="col">Train Sharpe</th><th scope="col">Test Sharpe</th></tr></thead>
<tbody>{folds}</tbody></table></div>

<h2>Locked holdout</h2>
{holdout}
</main>
<script>
const CHARTS = {charts};
function draw(id, data) {{
  const el = document.getElementById(id); if (!el) return;
  const css = getComputedStyle(document.documentElement), v = n => css.getPropertyValue(n).trim();
  const W = 880, H = 300, m = {{l: 44, r: 96, t: 10, b: 26}}, n = data.dates.length;
  const all = data.series.flatMap(s => s.values).concat(data.band ? data.band.lo.concat(data.band.hi) : []).filter(Number.isFinite);
  let lo = Math.min(...all), hi = Math.max(...all); const pad = (hi - lo) * 0.06 || 1; lo -= pad; hi += pad;
  const X = i => m.l + (W - m.l - m.r) * (n > 1 ? i / (n - 1) : 0), Y = y => m.t + (H - m.t - m.b) * (1 - (y - lo) / (hi - lo));
  const NS = 'http://www.w3.org/2000/svg', mk = (t, a) => {{ const e = document.createElementNS(NS, t); for (const k in a) e.setAttribute(k, a[k]); return e; }};
  const svg = mk('svg', {{viewBox: `0 0 ${{W}} ${{H}}`, 'font-size': 11}});
  const step = Math.pow(10, Math.floor(Math.log10((hi - lo) / 4))), tick = [1, 2, 5].map(k => k * step).find(s => (hi - lo) / s <= 6) || step * 10;
  for (let t = Math.ceil(lo / tick) * tick; t <= hi; t += tick) {{
    svg.append(mk('line', {{x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t), stroke: v('--grid')}}));
    const lb = mk('text', {{x: m.l - 6, y: Y(t) + 4, 'text-anchor': 'end', fill: v('--text-2')}}); lb.textContent = Math.round(t); svg.append(lb);
  }}
  const years = []; data.dates.forEach((d, i) => {{ if (i === 0 || d.slice(0, 4) !== data.dates[i - 1].slice(0, 4)) years.push(i); }});
  const every = Math.ceil(years.length / 8);
  years.forEach((i, k) => {{ if (k % every) return; const t = mk('text', {{x: X(i), y: H - 6, 'text-anchor': 'middle', fill: v('--text-2')}}); t.textContent = data.dates[i].slice(0, 4); svg.append(t); }});
  (data.markers || []).forEach(d => {{ const i = data.dates.indexOf(d); if (i >= 0) svg.append(mk('line', {{x1: X(i), x2: X(i), y1: m.t, y2: H - m.b, stroke: v('--rule'), 'stroke-dasharray': '2 4'}})); }});
  if (data.band) {{
    const up = data.band.hi.map((y, i) => `${{X(i)}},${{Y(y)}}`), dn = data.band.lo.map((y, i) => `${{X(i)}},${{Y(y)}}`).reverse();
    svg.append(mk('polygon', {{points: up.concat(dn).join(' '), fill: v('--band'), 'fill-opacity': 0.2}}));
  }}
  const ends = [];
  data.series.forEach(s => {{
    const col = v('--series-' + s.slot);
    svg.append(mk('polyline', {{points: s.values.map((y, i) => `${{X(i)}},${{Y(y)}}`).join(' '), fill: 'none', stroke: col, 'stroke-width': 2, 'stroke-linejoin': 'round'}}));
    ends.push({{y: Y(s.values[n - 1]), name: s.name, col}});
  }});
  ends.sort((a, b) => a.y - b.y); for (let k = 1; k < ends.length; k++) if (ends[k].y - ends[k - 1].y < 14) ends[k].y = ends[k - 1].y + 14;
  ends.forEach(e => {{ const t = mk('text', {{x: W - m.r + 8, y: e.y + 4, fill: v('--text')}}); t.textContent = e.name; svg.append(t); }});
  const cross = mk('line', {{y1: m.t, y2: H - m.b, stroke: v('--text-2'), 'stroke-width': 1, visibility: 'hidden'}}); svg.append(cross);
  const dots = data.series.map(s => {{ const c = mk('circle', {{r: 4, fill: v('--series-' + s.slot), stroke: v('--surface'), 'stroke-width': 2, visibility: 'hidden'}}); svg.append(c); return c; }});
  const hit = mk('rect', {{x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: 'transparent'}}); svg.append(hit);
  el.append(svg); const tip = document.createElement('div'); tip.className = 'tip'; el.append(tip);
  const show = evt => {{
    const r = svg.getBoundingClientRect(), x = (evt.clientX - r.left) * W / r.width;
    const i = Math.max(0, Math.min(n - 1, Math.round((x - m.l) / (W - m.l - m.r) * (n - 1))));
    cross.setAttribute('x1', X(i)); cross.setAttribute('x2', X(i)); cross.setAttribute('visibility', 'visible');
    data.series.forEach((s, k) => {{ dots[k].setAttribute('cx', X(i)); dots[k].setAttribute('cy', Y(s.values[i])); dots[k].setAttribute('visibility', 'visible'); }});
    tip.innerHTML = `<strong>${{data.dates[i]}}</strong><br>` + data.series.map(s => `${{s.name}}: ${{s.values[i].toFixed(1)}}`).join('<br>')
      + (data.band ? `<br>Random band: ${{data.band.lo[i].toFixed(1)}} to ${{data.band.hi[i].toFixed(1)}}` : '');
    tip.style.display = 'block'; const px = X(i) * r.width / W; tip.style.left = Math.min(px + 12, r.width - tip.offsetWidth) + 'px'; tip.style.top = '8px';
  }};
  hit.addEventListener('pointermove', show); hit.addEventListener('pointerleave', () => {{ tip.style.display = 'none'; cross.setAttribute('visibility', 'hidden'); dots.forEach(d => d.setAttribute('visibility', 'hidden')); }});
}}
draw('chart-oos', CHARTS.oos); if (CHARTS.holdout) draw('chart-holdout', CHARTS.holdout);
</script>
</body></html>
"""
