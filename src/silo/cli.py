"""Command line: fetch data, backtest, run lab studies, and paper-trade."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from silo.broker import CostModel
from silo.constraints import Constraints
from silo.data.sources import SOURCES, synthetic_bars
from silo.data.vault import DataVault, normalize_bars
from silo.engine import run_backtest
from silo.markets import market_for_symbols
from silo.strategies import REGISTRY

DISCLAIMER = "Research only. Simulated results, not financial advice. Silo never trades real money."
STUDY_STRATEGIES = [n for n in REGISTRY if n not in ("buy_and_hold", "random")]


def _print_table(rows: dict[str, dict[str, float]]) -> None:
    cols = ["total_return", "cagr", "sharpe", "max_drawdown", "round_trips", "avg_hold_days", "fees"]
    df = pd.DataFrame(rows).T.reindex(columns=cols)
    pct = ["total_return", "cagr", "max_drawdown"]
    print(df.to_string(formatters={**{c: "{:.1%}".format for c in pct}, "sharpe": "{:.2f}".format,
                                   "round_trips": "{:.0f}".format, "avg_hold_days": "{:.0f}".format,
                                   "fees": "{:,.0f}".format}))
    print(f"\n{DISCLAIMER}")


def _rules(args, symbols) -> dict:
    """Costs and constraints from flags; UK books get UK costs automatically."""
    market = market_for_symbols(symbols) if symbols else None
    costs = {"auto": CostModel.for_market(market.code) if market else CostModel(), "us": CostModel(),
             "uk": CostModel.uk(), "none": CostModel(0, 0, 0, 0, 0)}[args.costs]
    if args.style == "position":
        constraints = Constraints.position_trading(
            **{k: v for k, v in {"rebalance": args.rebalance, "min_hold_days": args.min_hold}.items() if v is not None}
        )
    else:
        constraints = Constraints(rebalance=args.rebalance or "daily", min_hold_days=args.min_hold or 0)
    return {"costs": costs, "constraints": constraints}


def _add_rule_flags(p) -> None:
    p.add_argument("--costs", choices=["auto", "us", "uk", "none"], default="auto",
                   help="cost model; auto picks UK costs (with stamp duty) for .L tickers")
    p.add_argument("--style", choices=["position", "any"], default="position",
                   help="position: weekly decisions, 20-day minimum hold, turnover and size caps")
    p.add_argument("--rebalance", choices=["daily", "weekly", "monthly"])
    p.add_argument("--min-hold", type=int, help="minimum trading days to hold a new position")


def cmd_fetch(args) -> int:
    vault = DataVault(args.data_dir)
    symbols = args.symbols
    if args.universe == "uk":
        from silo.universe import UK_CANDIDATES

        symbols = symbols + UK_CANDIDATES
    failed = []
    for symbol in symbols:
        try:
            stored = vault.put_bars(symbol, SOURCES[args.source](symbol))
        except Exception as exc:  # noqa: BLE001 - keep going, report at the end
            failed.append(f"{symbol} ({exc})")
            continue
        print(f"{symbol}: {len(stored)} bars, {stored.index[0].date()} to {stored.index[-1].date()}")
    if failed:
        print("\nskipped: " + ", ".join(failed))
    return 0


def _demo_tables(n=4, n_days=2000):
    symbols = [f"SYN{i}" for i in range(n)]
    return symbols, {s: normalize_bars(synthetic_bars(n_days=n_days, seed=i)) for i, s in enumerate(symbols)}


def _run_all(tables, symbols, names, start=None, end=None, **rules) -> dict[str, dict[str, float]]:
    return {n: run_backtest(REGISTRY[n](symbols), tables, start=start, end=end, **rules).metrics() for n in names}


def cmd_backtest(args) -> int:
    vault = DataVault(args.data_dir)
    tables = {s: vault.get_bars(s) for s in args.symbols}
    names = list(REGISTRY) if args.strategy == "all" else ["buy_and_hold", args.strategy]
    _print_table(_run_all(tables, args.symbols, names, args.start, args.end, **_rules(args, args.symbols)))
    return 0


def cmd_demo(args) -> int:
    symbols, tables = _demo_tables()
    print("Synthetic random-walk data: there is no real signal here, so no strategy")
    print("should reliably beat buy-and-hold. If one does, suspect luck.\n")
    _print_table(_run_all(tables, symbols, list(REGISTRY), costs=CostModel(), constraints=Constraints()))
    return 0


def cmd_lab(args) -> int:
    from silo.lab import HoldoutLog, run_study
    from silo.lab.report import write_report

    if args.demo:
        symbols, tables = _demo_tables()
    else:
        if not args.symbols:
            print("give symbols, or use --demo")
            return 2
        vault = DataVault(args.data_dir)
        symbols = args.symbols
        tables = {s: vault.get_bars(s) for s in symbols}
    names = STUDY_STRATEGIES if args.strategy == "all" else [args.strategy]
    log = HoldoutLog(Path(args.out) / "holdout_log.jsonl")
    for name in names:
        result = run_study(
            REGISTRY[name], symbols, tables,
            holdout_frac=args.holdout, n_folds=args.folds, n_random=args.random,
            open_holdout=args.open_holdout, holdout_log=log, **_rules(args, symbols),
        )
        path = write_report(result, Path(args.out) / f"{name}.html")
        status = "PASS" if result.interesting else "fail"
        print(f"{name:10s} {status}  " + "; ".join(c.detail for c in result.checks) + f"  -> {path}")
    print(f"\n{DISCLAIMER}")
    return 0


def cmd_universe(args) -> int:
    from silo.universe import UK_CANDIDATES, quality_screen, screen

    vault = DataVault(args.data_dir)
    have = [s for s in UK_CANDIDATES if vault.snapshots(s)]
    if not have:
        print("no UK data cached yet; run `silo fetch --universe uk` first")
        return 1
    table = screen({s: vault.get_bars(s) for s in have})
    if args.quality:
        q = quality_screen(list(table.index))
        table = table.join(q[["quality", "reason"]].rename(columns={"reason": "quality_reason"}))
        table["passes"] = table["passes"].astype(bool) & table["quality"].fillna(False).astype(bool)
    with pd.option_context("display.max_rows", None, "display.width", 160):
        print(table.to_string())
    picked = list(table.index[table["passes"].astype(bool)])
    print(f"\n{len(picked)} of {len(have)} pass: {' '.join(picked)}")
    if args.save:
        Path(args.save).write_text(json.dumps(picked))
        print(f"saved to {args.save}")
    return 0


def cmd_paper(args) -> int:
    from silo.paper import DISCLAIMER as PAPER_NOTE
    from silo.paper import PaperBook

    vault = DataVault(args.data_dir)
    book = PaperBook(args.name, args.paper_dir)
    if args.action == "t212-map":
        from silo.brokers.trading212 import Trading212Demo, propose_mapping

        mapping = propose_mapping(book.config()["symbols"], Trading212Demo().instruments())
        path = book.dir / "t212_map.json"
        path.write_text(json.dumps(mapping, indent=2))
        for sym, t in mapping.items():
            print(f"  {sym:8s} -> {t or 'NOT FOUND: edit the file by hand'}")
        print(f"saved to {path}; check every line before mirroring")
        return 0
    if args.action == "t212-check":
        from silo.brokers.trading212 import Trading212Demo
        from silo.paper import reconcile

        print(reconcile(book, Trading212Demo()).to_string())
        return 0
    if args.action == "new":
        symbols = args.symbols or json.loads(Path(args.symbols_file).read_text())
        rules = _rules(args, symbols)
        cfg = book.create(args.strategy, symbols, vault, json.loads(args.params), args.cash,
                          rules["constraints"], rules["costs"])
        print(f"created paper book {cfg['name']}: {cfg['strategy']} on {len(symbols)} {cfg['market']} symbols, "
              f"starts after {cfg['start_after'][:10]}")
    elif args.action == "run":
        if args.fetch:
            for s in book.config()["symbols"]:
                try:
                    vault.put_bars(s, SOURCES[args.source](s))
                except Exception as exc:  # noqa: BLE001
                    print(f"fetch failed for {s}: {exc}")
        events = book.run(vault)
        print(f"processed {len(events)} new trading day(s)")
        if args.mirror == "t212":
            from silo.brokers.trading212 import Trading212Demo
            from silo.paper import mirror_latest_decision

            sent = mirror_latest_decision(book, Trading212Demo(), fractional=args.fractional)
            print(f"Trading 212 Practice: {len(sent)} order(s) sent" if sent else
                  "Trading 212 Practice: no new rebalance to mirror")
            for e in sent:
                print(f"  {e['ticker']:14s} {e['quantity']:+.4g}  {e.get('status') or e.get('error')}")
        for e in events:
            if e.get("decision") or e["fills"]:
                print(f"  {pd.Timestamp(e['date']).date()}: {len(e['fills'])} fill(s)"
                      + (", rebalanced" if e.get("decision") else ""))
    st = book.status()
    print(f"\n{st['name']} ({st['strategy']}) after {st['days']} day(s), last {st['last_date']}")
    print(f"equity {st['currency']} {st['equity']:,.0f} vs buy-and-hold {st['benchmark_equity']:,.0f}; "
          f"cash {st['cash']:,.0f}; fees {st['fees_paid']:,.0f}")
    for s, p in st["positions"].items():
        print(f"  {s:8s} {p['weight']:6.1%}  held {p['held_days']} days")
    print(f"\n{PAPER_NOTE}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="silo", description=DISCLAIMER)
    p.add_argument("--data-dir", default="data")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="download daily bars into the local vault (new snapshot each time)")
    f.add_argument("symbols", nargs="*")
    f.add_argument("--universe", choices=["uk"], help="also fetch the UK candidate list")
    f.add_argument("--source", choices=sorted(SOURCES), default="yfinance")
    f.set_defaults(func=cmd_fetch)

    b = sub.add_parser("backtest", help="backtest strategies against buy-and-hold")
    b.add_argument("symbols", nargs="+")
    b.add_argument("--strategy", choices=["all", *REGISTRY], default="all")
    b.add_argument("--start")
    b.add_argument("--end")
    _add_rule_flags(b)
    b.set_defaults(func=cmd_backtest)

    lab = sub.add_parser("lab", help="walk-forward study with luck checks and an HTML report")
    lab.add_argument("symbols", nargs="*")
    lab.add_argument("--strategy", choices=["all", *STUDY_STRATEGIES], default="all")
    lab.add_argument("--folds", type=int, default=4)
    lab.add_argument("--random", type=int, default=30, help="number of random-entry control runs")
    lab.add_argument("--holdout", type=float, default=0.2, help="fraction of data locked as the final holdout")
    lab.add_argument("--open-holdout", action="store_true", help="evaluate on the locked holdout (logged)")
    lab.add_argument("--out", default="reports")
    lab.add_argument("--demo", action="store_true", help="use synthetic data, offline")
    _add_rule_flags(lab)
    lab.set_defaults(func=cmd_lab)

    u = sub.add_parser("universe", help="screen the UK candidate list on cached data")
    u.add_argument("market", choices=["uk"])
    u.add_argument("--quality", action="store_true", help="also check today's fundamentals (live use only)")
    u.add_argument("--save", help="write passing symbols to this JSON file")
    u.set_defaults(func=cmd_universe)

    pp = sub.add_parser("paper", help="forward paper trading on the simulator")
    pp.add_argument("action", choices=["new", "run", "status", "t212-map", "t212-check"])
    pp.add_argument("name")
    pp.add_argument("--paper-dir", default="paper")
    pp.add_argument("--strategy", choices=STUDY_STRATEGIES, default="uk_trend")
    pp.add_argument("--symbols", nargs="*")
    pp.add_argument("--symbols-file", help="JSON list of symbols, e.g. from `silo universe uk --save`")
    pp.add_argument("--params", default="{}", help="strategy parameters as JSON")
    pp.add_argument("--cash", type=float, default=100_000.0)
    pp.add_argument("--fetch", action="store_true", help="download fresh bars before running")
    pp.add_argument("--mirror", choices=["t212"], help="also send the newest rebalance to a Trading 212 Practice account")
    pp.add_argument("--fractional", action="store_true", help="allow fractional share quantities when mirroring")
    pp.add_argument("--source", choices=sorted(SOURCES), default="yfinance")
    _add_rule_flags(pp)
    pp.set_defaults(func=cmd_paper)

    d = sub.add_parser("demo", help="run every strategy on synthetic data, offline")
    d.set_defaults(func=cmd_demo)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
