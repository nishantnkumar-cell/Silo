"""Local web dashboard: ``silo web``. Research only, never real money.

Shows the paper books (equity against the buy-and-hold copy, positions, the
trade journal, the Trading 212 Practice mirror), the lab reports, and a
settings page for the Trading 212 **Practice** API key.

Standard library only. The server:

* binds to the loopback interface and refuses any other address;
* answers only requests whose ``Host`` is localhost (blocks DNS rebinding);
* requires a per-run token on every form post (blocks cross-site posts);
* reads paper books and journals but never changes them. The only outbound
  calls are read-only requests to ``demo.trading212.com`` (verify a key,
  compare the book with the Practice account). It never places orders.
"""

from __future__ import annotations

import html
import ipaddress
import json
import re
import secrets
import socket
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlsplit

from silo import credentials
from silo.safety import check_environment

DISCLAIMER = "Research only. Paper trading on a simulator, never real money. Not financial advice."
_NAME = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,99}$")
_REPORT = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,150}\.html$")
MAX_BODY = 8192
MAX_ROWS = 300


class NotLocal(ValueError):
    pass


def check_loopback(host: str) -> str:
    """Return ``host`` if it is a loopback address, else refuse."""
    if host == "localhost":
        return host
    try:
        if ipaddress.ip_address(host).is_loopback:
            return host
    except ValueError:
        pass
    raise NotLocal(f"silo web only listens on this computer (127.0.0.1), not {host!r}")


# ---------------------------------------------------------------- formatting

def _e(x) -> str:
    return html.escape(str(x), quote=True)


def _money(x, cur: str = "") -> str:
    try:
        return f"{cur} {float(x):,.0f}".strip()
    except (TypeError, ValueError):
        return "n/a"


def _pct(x) -> str:
    try:
        return f"{float(x):+.1%}"
    except (TypeError, ValueError):
        return "n/a"


def _date(x) -> str:
    return str(x)[:10] if x else "n/a"


def _signed_class(x) -> str:
    try:
        return "good" if float(x) > 0 else ("bad" if float(x) < 0 else "")
    except (TypeError, ValueError):
        return ""


# ---------------------------------------------------------------- data

def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def list_books(paper_dir: Path) -> list[dict]:
    from silo.paper import PaperBook

    rows = []
    if not paper_dir.is_dir():
        return rows
    for d in sorted(p for p in paper_dir.iterdir() if (p / "config.json").exists()):
        if not _NAME.match(d.name):
            continue
        book = PaperBook(d.name, paper_dir)
        try:
            st = book.status()
            cash0 = book.config()["cash"]
            rows.append({**st, "start_cash": cash0, "error": None})
        except Exception as exc:  # noqa: BLE001 - one broken book must not hide the others
            rows.append({"name": d.name, "error": f"{type(exc).__name__}: {exc}"})
    return rows


def list_reports(reports_dir: Path) -> list[dict]:
    if not reports_dir.is_dir():
        return []
    files = [p for p in reports_dir.iterdir() if p.is_file() and _REPORT.match(p.name)]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [{"name": p.name, "modified": datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")}
            for p in files]


def folded_table(head: str, rows: list[str], visible: int = 20, what: str = "rows") -> str:
    """Latest rows visible, the rest behind a disclosure, so long journals stay readable."""
    first = f"<div class='scroll'><table><thead><tr>{head}</tr></thead><tbody>{''.join(rows[:visible])}</tbody></table></div>"
    if len(rows) <= visible:
        return first
    rest = (f"<details><summary>Show {len(rows) - visible} older {what}</summary><div class='scroll'><table>"
            f"<thead><tr>{head}</tr></thead><tbody>{''.join(rows[visible:])}</tbody></table></div></details>")
    return first + rest


def equity_chart(dates: list[str], book: list[float], bench: list[float], cur: str) -> str:
    """Inline SVG line chart, no JavaScript."""
    if len(dates) < 2:
        return "<p class='muted'>The chart appears after the book has processed at least two trading days.</p>"
    w, h, l, r, t, b = 880, 280, 64, 12, 12, 28
    lo, hi = min(book + bench), max(book + bench)
    pad = (hi - lo) * 0.08 or max(abs(hi) * 0.01, 1.0)
    lo, hi = lo - pad, hi + pad
    n = len(dates)

    def x(i):
        return l + (w - l - r) * i / (n - 1)

    def y(v):
        return t + (h - t - b) * (1 - (v - lo) / (hi - lo))

    grid = []
    for k in range(5):
        v = lo + (hi - lo) * k / 4
        grid.append(f"<line x1='{l}' x2='{w - r}' y1='{y(v):.1f}' y2='{y(v):.1f}' class='grid'/>"
                    f"<text x='{l - 6}' y='{y(v) + 4:.1f}' text-anchor='end'>{_e(_money(v))}</text>")
    for i in sorted({0, n // 2, n - 1}):
        anchor = "start" if i == 0 else ("end" if i == n - 1 else "middle")
        grid.append(f"<text x='{x(i):.1f}' y='{h - 8}' text-anchor='{anchor}'>{_e(dates[i])}</text>")

    def line(vals, cls, label):
        pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(vals))
        return f"<polyline class='{cls}' points='{pts}'><title>{_e(label)}</title></polyline>"

    return (f"<svg viewBox='0 0 {w} {h}' role='img' aria-label='Equity of the paper book and the buy-and-hold "
            f"copy in {_e(cur)}'>{''.join(grid)}{line(bench, 's2', 'Buy-and-hold')}{line(book, 's1', 'Paper book')}</svg>")


# ---------------------------------------------------------------- pages

CSS = """
:root { color-scheme: light; --surface:#fcfcfb; --surface-2:#f3f2ef; --text:#0b0b0b; --text-2:#52514e; --rule:#dcdbd6;
  --grid:#ebeae6; --series-1:#2a78d6; --series-2:#eb6834; --good:#008300; --bad:#c62a2a; --warn-bg:#fff4d6; --accent:#2a78d6; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { color-scheme: dark; --surface:#1a1a19; --surface-2:#242423;
  --text:#f0efec; --text-2:#c3c2b7; --rule:#3a3a38; --grid:#2c2c2a; --series-1:#3987e5; --series-2:#d95926; --good:#4fb84f;
  --bad:#e66767; --warn-bg:#3a3014; --accent:#3987e5; } }
:root[data-theme="dark"] { color-scheme: dark; --surface:#1a1a19; --surface-2:#242423; --text:#f0efec; --text-2:#c3c2b7;
  --rule:#3a3a38; --grid:#2c2c2a; --series-1:#3987e5; --series-2:#d95926; --good:#4fb84f; --bad:#e66767; --warn-bg:#3a3014; --accent:#3987e5; }
* { box-sizing: border-box; }
body { margin:0; background:var(--surface); color:var(--text); font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }
header.top { border-bottom:1px solid var(--rule); background:var(--surface-2); }
header.top div { max-width:1000px; margin:0 auto; padding:10px 16px; display:flex; gap:20px; align-items:center; flex-wrap:wrap; }
header.top a { color:var(--text); text-decoration:none; } header.top a.brand { font-weight:700; }
header.top a[aria-current] { text-decoration:underline; text-underline-offset:4px; }
main { max-width:1000px; margin:0 auto; padding:20px 16px 48px; }
h1 { font-size:22px; margin:0 0 4px; } h2 { font-size:17px; margin:28px 0 8px; }
a { color:var(--accent); }
.muted { color:var(--text-2); } .banner { background:var(--surface-2); border:1px solid var(--rule); border-radius:8px; padding:8px 12px; font-size:13px; }
.good { color:var(--good); } .bad { color:var(--bad); }
.msg { border-radius:8px; padding:10px 12px; margin:12px 0; border:1px solid var(--rule); }
.msg.ok { border-color:var(--good); } .msg.err { border-color:var(--bad); background:var(--warn-bg); }
.warn { background:var(--warn-bg); border-radius:6px; padding:8px 12px; }
table { border-collapse:collapse; width:100%; font-size:14px; margin:8px 0; font-variant-numeric:tabular-nums; }
th, td { text-align:right; padding:6px 8px; border-bottom:1px solid var(--grid); }
th:first-child, td:first-child { text-align:left; } thead th { color:var(--text-2); font-weight:500; }
td.l, th.l { text-align:left; }
.scroll { overflow-x:auto; }
.tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; margin:12px 0; }
.tile { border:1px solid var(--rule); border-radius:8px; padding:10px 12px; }
.tile .k { font-size:12px; color:var(--text-2); } .tile .v { font-size:20px; font-weight:600; font-variant-numeric:tabular-nums; }
.legend { display:flex; gap:16px; font-size:13px; color:var(--text-2); }
.legend span::before { content:""; display:inline-block; width:14px; height:3px; border-radius:2px; margin-right:6px; vertical-align:middle; background:var(--c); }
svg { width:100%; height:auto; display:block; } svg text { font-size:11px; fill:var(--text-2); }
svg .grid { stroke:var(--grid); } svg polyline { fill:none; stroke-width:2; stroke-linejoin:round; }
svg .s1 { stroke:var(--series-1); } svg .s2 { stroke:var(--series-2); }
form.inline { display:inline; }
label { display:block; font-weight:600; margin:12px 0 4px; }
input[type=text], input[type=password] { width:100%; max-width:480px; padding:8px; border:1px solid var(--rule); border-radius:6px;
  background:var(--surface); color:var(--text); font:inherit; }
button { font:inherit; padding:7px 14px; border-radius:6px; border:1px solid var(--rule); background:var(--surface-2); color:var(--text); cursor:pointer; margin-top:12px; }
button.primary { background:var(--accent); border-color:var(--accent); color:#fff; }
code { font-size:13px; }
details summary { cursor:pointer; color:var(--text-2); }
"""

_NAV = [("/", "Overview"), ("/reports", "Lab reports"), ("/settings", "Settings")]


def page(title: str, body: str, active: str = "/") -> str:
    current = ' aria-current="page"'
    nav = "".join(f"<a href='{href}'{current if href == active else ''}>{label}</a>" for href, label in _NAV)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Silo: {_e(title)}</title><style>{CSS}</style></head>
<body><header class="top"><div><a class="brand" href="/">Silo</a>{nav}<span class="muted" style="margin-left:auto;font-size:13px">Research only &middot; paper money</span></div></header>
<main><p class="banner">{DISCLAIMER}</p>{body}</main></body></html>"""


def _flash(msg: str | None, err: str | None) -> str:
    out = ""
    if msg:
        out += f"<p class='msg ok' role='status'>{_e(msg)}</p>"
    if err:
        out += f"<p class='msg err' role='alert'>{_e(err)}</p>"
    return out


def overview_page(paper_dir: Path, reports_dir: Path) -> str:
    books = list_books(paper_dir)
    reports = list_reports(reports_dir)
    t212 = credentials.t212_status()
    if books:
        rows = []
        for b in books:
            link = f"<a href='/book/{quote(b['name'])}'>{_e(b['name'])}</a>"
            if b.get("error"):
                rows.append(f"<tr><td>{link}</td><td class='l bad' colspan='7'>Could not read: {_e(b['error'])}</td></tr>")
                continue
            ret = b["equity"] / b["start_cash"] - 1
            bret = b["benchmark_equity"] / b["start_cash"] - 1
            rows.append(
                f"<tr><td>{link}</td><td class='l'>{_e(b['strategy'])}</td><td>{b['days']}</td>"
                f"<td>{_date(b['last_date'])}</td><td>{_money(b['equity'], b['currency'])}</td>"
                f"<td class='{_signed_class(ret)}'>{_pct(ret)}</td><td class='{_signed_class(bret)}'>{_pct(bret)}</td>"
                f"<td>{len(b['positions'])}</td></tr>"
            )
        books_html = (
            "<div class='scroll'><table><thead><tr><th>Book</th><th class='l'>Strategy</th><th>Days</th><th>Last day</th>"
            "<th>Equity</th><th>Book return</th><th>Buy-and-hold return</th><th>Positions</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>"
        )
    else:
        books_html = (
            f"<p class='muted'>No paper books in <code>{_e(paper_dir)}</code> yet. Create one with "
            "<code>silo paper new uk-trend --symbols-file uk.json</code>, then run "
            "<code>silo paper run uk-trend --fetch</code> after each London close.</p>"
        )
    rep = "".join(f"<li><a href='/reports/{quote(r['name'])}'>{_e(r['name'])}</a> <span class='muted'>{r['modified']}</span></li>"
                  for r in reports[:8])
    rep_html = f"<ul>{rep}</ul>" + ("<p><a href='/reports'>All reports</a></p>" if len(reports) > 8 else "") if reports else (
        "<p class='muted'>No lab reports yet. <code>silo lab --demo --strategy bollinger</code> writes one.</p>")
    if t212["source"]:
        t212_html = (f"<p>Trading 212 Practice key: set ({_e(t212['source'])}"
                     + (f", key {_e(t212['key_hint'])}" if t212["key_hint"] and t212["source"] == "file" else "")
                     + "). <a href='/settings'>Settings</a></p>")
    else:
        t212_html = "<p class='muted'>Trading 212 Practice key: not set. <a href='/settings'>Add one</a> to mirror books to a Practice account.</p>"
    return page("Overview", f"""<h1>Overview</h1>
<p class="muted">Paper books run forward on Silo's simulator, each next to a buy-and-hold copy of the same shares.</p>
<h2>Paper books</h2>{books_html}
<h2>Latest lab reports</h2>{rep_html}
<h2>Trading 212 Practice</h2>{t212_html}""")


def book_page(paper_dir: Path, name: str, token: str, check: dict | None = None) -> str | None:
    from silo.paper import PaperBook

    if not _NAME.match(name):
        return None
    book = PaperBook(name, paper_dir)
    if not book.exists:
        return None
    cfg = book.config()
    cur = cfg.get("currency", "")
    try:
        st = book.status()
    except Exception as exc:  # noqa: BLE001
        return page(name, f"<h1>{_e(name)}</h1><p class='msg err'>Could not read this book: {_e(exc)}</p>")
    journal = read_jsonl(book.dir / "journal.jsonl")
    days = [e for e in journal if e.get("event") == "day"]
    dates = [_date(e["date"]) for e in days]
    eq = [float(e["equity"]) for e in days]
    beq = [float(e["benchmark_equity"]) for e in days]
    cash0 = float(cfg["cash"])
    ret, bret = st["equity"] / cash0 - 1, st["benchmark_equity"] / cash0 - 1
    m = st.get("metrics") or {}

    tiles = f"""<div class="tiles">
<div class="tile"><div class="k">Paper book</div><div class="v">{_money(st['equity'], cur)}</div><div class="{_signed_class(ret)}">{_pct(ret)}</div></div>
<div class="tile"><div class="k">Buy-and-hold copy</div><div class="v">{_money(st['benchmark_equity'], cur)}</div><div class="{_signed_class(bret)}">{_pct(bret)}</div></div>
<div class="tile"><div class="k">Ahead of buy-and-hold</div><div class="v {_signed_class(ret - bret)}">{_pct(ret - bret)}</div><div class="muted">percentage points</div></div>
<div class="tile"><div class="k">Cash / fees paid</div><div class="v">{_money(st['cash'], cur)}</div><div class="muted">fees {_money(st['fees_paid'], cur)}</div></div>
</div>"""

    pos_rows = "".join(
        f"<tr><td>{_e(s)}</td><td>{p['weight']:.1%}</td><td>{p['held_days']}</td></tr>" for s, p in st["positions"].items()
    )
    positions = (f"<table><thead><tr><th>Symbol</th><th>Weight</th><th>Held (trading days)</th></tr></thead>"
                 f"<tbody>{pos_rows}</tbody></table>") if pos_rows else "<p class='muted'>No open positions.</p>"

    fills = [(e["date"], f) for e in days for f in e.get("fills") or []]
    fills.reverse()
    fill_rows = [
        f"<tr><td>{_date(d)}</td><td class='l'>{_e(f['symbol'])}</td>"
        f"<td class='l {'good' if f['qty'] > 0 else 'bad'}'>{'Buy' if f['qty'] > 0 else 'Sell'}</td>"
        f"<td>{abs(f['qty']):,.2f}</td><td>{f['price']:,.2f}</td><td>{_money(abs(f['qty']) * f['price'], cur)}</td>"
        f"<td>{f['fee']:,.2f}</td><td>{_date(f.get('decided_at'))}</td></tr>"
        for d, f in fills[:MAX_ROWS]
    ]
    trades = (folded_table("<th>Filled</th><th class='l'>Symbol</th><th class='l'>Side</th><th>Shares</th><th>Price</th>"
                           "<th>Value</th><th>Fee</th><th>Decided</th>", fill_rows, what="fills")
              + (f"<p class='muted'>Showing the latest {MAX_ROWS} of {len(fills)} fills.</p>" if len(fills) > MAX_ROWS else "")
              ) if fills else "<p class='muted'>No trades yet.</p>"

    decisions = [e for e in days if e.get("decision")]
    decisions.reverse()
    dec_rows = []
    for e in decisions[:MAX_ROWS]:
        after = e["decision"].get("after_rules") or {}
        proposed = e["decision"].get("proposed") or {}
        held = ", ".join(f"{_e(s)} {w:.0%}" for s, w in sorted(after.items(), key=lambda kv: -kv[1]) if w > 0) or "all cash"
        changed = sorted(s for s in set(after) | set(proposed) if abs(after.get(s, 0) - proposed.get(s, 0)) > 1e-6)
        dec_rows.append(f"<tr><td>{_date(e['date'])}</td><td class='l'>{held}</td>"
                        f"<td class='l muted'>{_e(', '.join(changed)) or 'none'}</td></tr>")
    rebalances = folded_table("<th>Decided at close</th><th class='l'>Target after Silo's rules</th>"
                              "<th class='l'>Trimmed by rules</th>", dec_rows, visible=10, what="rebalances"
                              ) if dec_rows else "<p class='muted'>No rebalances yet.</p>"

    data_rows = "".join(f"<tr><td>{d}</td><td>{a:,.0f}</td><td>{b:,.0f}</td></tr>" for d, a, b in zip(dates, eq, beq))

    # Trading 212 mirror
    map_path = book.dir / "t212_map.json"
    mlog = read_jsonl(book.dir / "mirror.jsonl")
    sent = [e for e in mlog if e.get("kind") in ("sent", "error")]
    sent.reverse()
    mrows = "".join(
        f"<tr><td>{_date(e.get('decision_date'))}</td><td class='l'>{_e(e.get('ticker'))}</td><td>{e.get('quantity', 0):+,.4g}</td>"
        f"<td class='l {'bad' if e['kind'] == 'error' else ''}'>{_e(e.get('status') or e.get('error') or e['kind'])}</td></tr>"
        for e in sent[:100]
    )
    mirror = ("<p>Ticker map: " + (f"<code>{_e(map_path.name)}</code> present." if map_path.exists() else
              f"not created yet. Run <code>silo paper t212-map {_e(name)}</code> and review the file.") + "</p>")
    mirror += (f"<div class='scroll'><table><thead><tr><th>Decision</th><th class='l'>Trading 212 ticker</th><th>Quantity</th>"
               f"<th class='l'>Result</th></tr></thead><tbody>{mrows}</tbody></table></div>") if mrows else \
        "<p class='muted'>No orders mirrored yet. <code>silo paper run NAME --mirror t212</code> sends the newest rebalance.</p>"
    mirror += (f"<form method='post' action='/book/{quote(name)}/t212-check'><input type='hidden' name='csrf' value='{token}'>"
               "<button type='submit'>Compare with the Practice account</button></form>"
               "<p class='muted' style='font-size:13px'>Read-only: asks demo.trading212.com for positions. It does not place orders.</p>")
    if check is not None:
        if check.get("error"):
            mirror += f"<p class='msg err' role='alert'>{_e(check['error'])}</p>"
        elif not check["rows"]:
            mirror += "<p class='msg ok' role='status'>Neither book holds anything.</p>"
        else:
            crow = "".join(
                f"<tr><td>{_e(s)}</td><td class='l'>{_e(r['ticker'])}</td><td>{r['simulator']:.1%}</td><td>{r['practice']:.1%}</td>"
                f"<td class='{'bad' if abs(r['simulator'] - r['practice']) > 0.02 else ''}'>{r['practice'] - r['simulator']:+.1%}</td>"
                f"<td>{r['practice_qty']:,.4g}</td></tr>"
                for s, r in check["rows"].items()
            )
            mirror += (f"<p class='muted'>Checked {check['at']}. Gaps over 2 points are highlighted.</p><div class='scroll'><table>"
                       "<thead><tr><th>Symbol</th><th class='l'>Ticker</th><th>Simulator</th><th>Practice</th><th>Gap</th>"
                       f"<th>Practice shares</th></tr></thead><tbody>{crow}</tbody></table></div>")

    rules = cfg.get("constraints") or {}
    rules_txt = ", ".join(f"{k.replace('_', ' ')}: {_e(v)}" for k, v in rules.items() if not isinstance(v, dict))
    metrics_line = ""
    if m:
        metrics_line = (f"<p class='muted'>Sharpe {m.get('sharpe', float('nan')):.2f}, max drawdown "
                        f"{m.get('max_drawdown', float('nan')):.1%}, {m.get('round_trips', 0):.0f} round trips.</p>")
    return page(name, f"""<h1>{_e(name)}</h1>
<p class="muted">{_e(cfg['strategy'])} on {len(cfg['symbols'])} {_e(cfg.get('market', ''))} shares &middot; started after
{_date(cfg.get('start_after'))} &middot; {st['days']} trading day(s) processed, last {_date(st['last_date'])}</p>
{tiles}
<h2>Equity against buy-and-hold</h2>
<div class="legend"><span style="--c:var(--series-1)">Paper book</span><span style="--c:var(--series-2)">Buy-and-hold copy</span></div>
{equity_chart(dates, eq, beq, cur)}{metrics_line}
<details><summary>Show the data as a table</summary><div class="scroll"><table><thead><tr><th>Date</th><th>Paper book</th>
<th>Buy-and-hold</th></tr></thead><tbody>{data_rows}</tbody></table></div></details>
<h2>Positions</h2>{positions}
<h2>Trade journal</h2>{trades}
<h2>Rebalance decisions</h2>{rebalances}
<h2>Trading 212 Practice mirror</h2>{mirror}
<h2>Book settings</h2>
<p class="muted">Symbols: {_e(' '.join(cfg['symbols']))}</p>
<p class="muted">Rules: {rules_txt or 'default'}</p>
<p class="muted">Books are changed only from the command line; this page never edits them.</p>""")


def reports_page(reports_dir: Path) -> str:
    reports = list_reports(reports_dir)
    rows = "".join(f"<tr><td><a href='/reports/{quote(r['name'])}'>{_e(r['name'])}</a></td><td>{r['modified']}</td></tr>"
                   for r in reports)
    body = (f"<table><thead><tr><th>Report</th><th>Written</th></tr></thead><tbody>{rows}</tbody></table>" if rows else
            f"<p class='muted'>No reports in <code>{_e(reports_dir)}</code>. Run <code>silo lab --demo --strategy bollinger</code>.</p>")
    return page("Lab reports", f"<h1>Lab reports</h1><p class='muted'>Walk-forward studies written by <code>silo lab</code>, "
                f"each against buy-and-hold and random controls.</p>{body}", "/reports")


def settings_page(token: str, msg: str | None = None, err: str | None = None) -> str:
    st = credentials.t212_status()
    if st["source"] == "environment":
        current = ("<p>A key is set by the environment variables <code>T212_DEMO_API_KEY</code> and "
                   "<code>T212_DEMO_API_SECRET</code>. They take precedence over a key saved here.</p>")
    elif st["stored"]:
        current = (f"<p>Saved key <code>{_e(st['key_hint'])}</code>, verified with the Practice server on "
                   f"{_e(st['verified_at'] or 'n/a')}" + (f" (account in {_e(st['account_currency'])})" if st["account_currency"] else "")
                   + ".</p>"
                   f"<form method='post' action='/settings/delete'><input type='hidden' name='csrf' value='{token}'>"
                   "<button type='submit'>Remove saved key</button></form>")
    else:
        current = "<p class='muted'>No key saved.</p>"
    return page("Settings", f"""<h1>Settings</h1>
{_flash(msg, err)}
<h2>Trading 212 Practice API key</h2>
<p class="warn">Practice (paper money) only. In the Trading 212 app, switch to <strong>Practice</strong> first, then go to
Settings, API (Beta), and create a key there. Silo checks every key with <code>demo.trading212.com</code> before saving it.
A real-money key is rejected by that server and is never stored, and Silo has no way to talk to the live server.</p>
{current}
<form method="post" action="/settings" autocomplete="off">
<input type="hidden" name="csrf" value="{token}">
<label for="key">API key</label><input id="key" name="api_key" type="text" spellcheck="false" autocomplete="off" required>
<label for="secret">API secret</label><input id="secret" name="api_secret" type="password" autocomplete="new-password" required>
<div><button class="primary" type="submit">Verify with Practice server and save</button></div>
</form>
<p class="muted" style="font-size:13px">Saved to <code>{_e(st['path'])}</code>, readable only by your user account, outside
the repository. The secret is never shown again.</p>""", "/settings")


# ---------------------------------------------------------------- server

class Dashboard:
    def __init__(self, paper_dir="paper", reports_dir="reports", host="127.0.0.1", port=8212, verify=None,
                 client_factory=None):
        self.paper_dir = Path(paper_dir)
        self.reports_dir = Path(reports_dir)
        self.host = check_loopback(host)
        self.port = port
        self.token = secrets.token_urlsafe(32)
        self.verify = verify  # for tests: replaces the Practice-server check
        self.client_factory = client_factory  # for tests: replaces Trading212Demo()

    def allowed_hosts(self, port: int) -> set[str]:
        return {f"{h}:{port}" for h in ("127.0.0.1", "localhost", "[::1]")}

    def make_server(self) -> ThreadingHTTPServer:
        check_environment()
        dash = self

        class Handler(_Handler):
            app = dash

        family_host = "::1" if self.host == "::1" else ("127.0.0.1" if self.host == "localhost" else self.host)
        server_cls = _V6Server if ":" in family_host else ThreadingHTTPServer
        srv = server_cls((family_host, self.port), Handler)
        self.port = srv.server_address[1]
        return srv


class _V6Server(ThreadingHTTPServer):
    address_family = socket.AF_INET6


_CSP = ("default-src 'none'; style-src 'unsafe-inline'; img-src data:; form-action 'self'; "
        "frame-ancestors 'none'; base-uri 'none'")
# Lab reports carry their own inline chart script; still no network access.
_REPORT_CSP = ("default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; img-src data:; "
               "frame-ancestors 'none'; base-uri 'none'; form-action 'none'")


class _Handler(BaseHTTPRequestHandler):
    app: Dashboard
    server_version = "silo-web"
    sys_version = ""

    def log_message(self, fmt, *args):  # quiet, and never log request bodies
        pass

    # --- helpers
    def _send(self, status: int, body: str, csp: str = _CSP, extra: dict | None = None) -> None:
        data = body.encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "same-origin")
        self.send_header("Content-Security-Policy", csp)
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def _redirect(self, location: str) -> None:
        self.send_response(HTTPStatus.SEE_OTHER)
        self.send_header("Location", location)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _not_found(self) -> None:
        self._send(404, page("Not found", "<h1>Not found</h1><p><a href='/'>Back to the overview</a></p>"))

    def _host_ok(self) -> bool:
        return (self.headers.get("Host") or "").lower() in self.app.allowed_hosts(self.app.port)

    def _form(self) -> dict | None:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return None
        if length > MAX_BODY or "application/x-www-form-urlencoded" not in (self.headers.get("Content-Type") or ""):
            return None
        fields = parse_qs(self.rfile.read(length).decode(errors="replace"), keep_blank_values=True)
        form = {k: v[0] for k, v in fields.items()}
        origin = self.headers.get("Origin")
        if origin and urlsplit(origin).netloc.lower() not in self.app.allowed_hosts(self.app.port):
            return None
        if not secrets.compare_digest(form.get("csrf", ""), self.app.token):
            return None
        return form

    # --- routes
    def do_HEAD(self):  # noqa: N802
        self.do_GET()

    def do_GET(self):  # noqa: N802
        if not self._host_ok():
            return self._send(403, "Forbidden: open Silo at http://127.0.0.1:%d/" % self.app.port)
        parts = urlsplit(self.path)
        path, query = parts.path, parse_qs(parts.query)
        if path == "/":
            return self._send(200, overview_page(self.app.paper_dir, self.app.reports_dir))
        if path == "/reports":
            return self._send(200, reports_page(self.app.reports_dir))
        if path.startswith("/reports/"):
            return self._report(path.removeprefix("/reports/"))
        if path.startswith("/book/"):
            body = book_page(self.app.paper_dir, _unquote(path.removeprefix("/book/")), self.app.token)
            return self._send(200, body) if body else self._not_found()
        if path == "/settings":
            msg = {"saved": "Key verified with the Practice server and saved.", "removed": "Saved key removed."}
            return self._send(200, settings_page(self.app.token, msg.get((query.get("done") or [""])[0])))
        return self._not_found()

    def do_POST(self):  # noqa: N802
        if not self._host_ok():
            return self._send(403, "Forbidden")
        form = self._form()
        if form is None:
            return self._send(403, page("Refused", "<h1>Request refused</h1><p>The form expired or came from another "
                                                   "site. <a href='/settings'>Reload the page</a> and try again.</p>"))
        path = urlsplit(self.path).path
        if path == "/settings":
            try:
                credentials.save_t212(form.get("api_key", ""), form.get("api_secret", ""), verify=self.app.verify)
            except credentials.CredentialError as exc:
                return self._send(400, settings_page(self.app.token, err=str(exc)))
            return self._redirect("/settings?done=saved")
        if path == "/settings/delete":
            credentials.clear_t212()
            return self._redirect("/settings?done=removed")
        if path.startswith("/book/") and path.endswith("/t212-check"):
            name = _unquote(path.removeprefix("/book/").removesuffix("/t212-check"))
            body = book_page(self.app.paper_dir, name, self.app.token, check=self._check(name))
            return self._send(200, body) if body else self._not_found()
        return self._not_found()

    def _check(self, name: str) -> dict:
        from silo.paper import PaperBook, reconcile

        if not _NAME.match(name):
            return {"error": "bad book name"}
        try:
            client = self.app.client_factory() if self.app.client_factory else _default_client()
            df = reconcile(PaperBook(name, self.app.paper_dir), client)
        except Exception as exc:  # noqa: BLE001 - show the reason on the page
            return {"error": str(exc)}
        return {"rows": df.to_dict(orient="index"), "at": datetime.now().strftime("%Y-%m-%d %H:%M")}

    def _report(self, name: str) -> None:
        name = _unquote(name)
        if not _REPORT.match(name):
            return self._not_found()
        root = self.app.reports_dir.resolve()
        path = (root / name).resolve()
        if path.parent != root or not path.is_file():
            return self._not_found()
        self._send(200, path.read_text(errors="replace"), csp=_REPORT_CSP)


def _unquote(s: str) -> str:
    from urllib.parse import unquote

    return unquote(s)


def _default_client():
    from silo.brokers.trading212 import Trading212Demo

    return Trading212Demo()


def serve(paper_dir="paper", reports_dir="reports", host="127.0.0.1", port=8212, open_browser=True) -> None:
    dash = Dashboard(paper_dir, reports_dir, host, port)
    srv = dash.make_server()
    url = f"http://127.0.0.1:{dash.port}/" if dash.host != "::1" else f"http://[::1]:{dash.port}/"
    print(f"Silo dashboard on {url}  (this computer only; Ctrl+C to stop)")
    print(DISCLAIMER)
    if open_browser:
        import webbrowser

        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        srv.server_close()
