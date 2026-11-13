"""Command line: fetch data, run backtests, or run an offline demo."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from silo.data.sources import SOURCES, synthetic_bars
from silo.data.vault import DataVault, normalize_bars
from silo.engine import run_backtest
from silo.strategies import REGISTRY

DISCLAIMER = "Research only. Simulated results, not financial advice. Silo never trades real money."


def _print_table(rows: dict[str, dict[str, float]]) -> None:
    df = pd.DataFrame(rows).T[["total_return", "cagr", "sharpe", "max_drawdown", "fills", "fees"]]
    pct = ["total_return", "cagr", "max_drawdown"]
    print(df.to_string(formatters={**{c: "{:.1%}".format for c in pct}, "sharpe": "{:.2f}".format,
                                   "fills": "{:.0f}".format, "fees": "${:,.0f}".format}))
    print(f"\n{DISCLAIMER}")


def cmd_fetch(args) -> int:
    vault = DataVault(args.data_dir)
    for symbol in args.symbols:
        df = SOURCES[args.source](symbol)
        stored = vault.put_bars(symbol, df)
        print(f"{symbol}: {len(stored)} bars, {stored.index[0].date()} to {stored.index[-1].date()}")
    return 0


def _run_all(tables, symbols, names, start=None, end=None) -> dict[str, dict[str, float]]:
    rows = {}
    for name in names:
        result = run_backtest(REGISTRY[name](symbols), tables, start=start, end=end)
        rows[name] = result.metrics()
    return rows


def cmd_backtest(args) -> int:
    vault = DataVault(args.data_dir)
    tables = {s: vault.get_bars(s) for s in args.symbols}
    names = list(REGISTRY) if args.strategy == "all" else ["buy_and_hold", args.strategy]
    _print_table(_run_all(tables, args.symbols, names, args.start, args.end))
    return 0


def cmd_demo(args) -> int:
    symbols = ["AAA", "BBB", "CCC", "DDD"]
    tables = {s: normalize_bars(synthetic_bars(seed=i)) for i, s in enumerate(symbols)}
    print("Synthetic random-walk data: there is no real signal here, so no strategy")
    print("should reliably beat buy-and-hold. If one does, suspect luck.\n")
    _print_table(_run_all(tables, symbols, list(REGISTRY)))
    return 0


def cmd_lab(args) -> int:
    from silo.lab import HoldoutLog, run_study
    from silo.lab.report import write_report

    if args.demo:
        symbols = ["AAA", "BBB", "CCC", "DDD"]
        tables = {s: normalize_bars(synthetic_bars(n_days=2000, seed=i)) for i, s in enumerate(symbols)}
    else:
        if not args.symbols:
            print("give symbols, or use --demo")
            return 2
        vault = DataVault(args.data_dir)
        symbols = args.symbols
        tables = {s: vault.get_bars(s) for s in symbols}
    names = [n for n in REGISTRY if n not in ("buy_and_hold", "random")] if args.strategy == "all" else [args.strategy]
    log = HoldoutLog(Path(args.out) / "holdout_log.jsonl")
    for name in names:
        result = run_study(
            REGISTRY[name], symbols, tables,
            holdout_frac=args.holdout, n_folds=args.folds, n_random=args.random,
            open_holdout=args.open_holdout, holdout_log=log,
        )
        path = write_report(result, Path(args.out) / f"{name}.html")
        status = "PASS" if result.interesting else "fail"
        print(f"{name:10s} {status}  " + "; ".join(c.detail for c in result.checks) + f"  -> {path}")
    print(f"\n{DISCLAIMER}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="silo", description=DISCLAIMER)
    p.add_argument("--data-dir", default="data")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="download daily bars into the local vault")
    f.add_argument("symbols", nargs="+")
    f.add_argument("--source", choices=sorted(SOURCES), default="yfinance")
    f.set_defaults(func=cmd_fetch)

    b = sub.add_parser("backtest", help="backtest a strategy against buy-and-hold")
    b.add_argument("symbols", nargs="+")
    b.add_argument("--strategy", choices=["all", *REGISTRY], default="all")
    b.add_argument("--start")
    b.add_argument("--end")
    b.set_defaults(func=cmd_backtest)

    lab = sub.add_parser("lab", help="walk-forward study with luck checks and an HTML report")
    lab.add_argument("symbols", nargs="*")
    lab.add_argument("--strategy", choices=["all", *[n for n in REGISTRY if n not in ("buy_and_hold", "random")]], default="all")
    lab.add_argument("--folds", type=int, default=4)
    lab.add_argument("--random", type=int, default=30, help="number of random-entry control runs")
    lab.add_argument("--holdout", type=float, default=0.2, help="fraction of data locked as the final holdout")
    lab.add_argument("--open-holdout", action="store_true", help="evaluate on the locked holdout (logged)")
    lab.add_argument("--out", default="reports")
    lab.add_argument("--demo", action="store_true", help="use synthetic data, offline")
    lab.set_defaults(func=cmd_lab)

    d = sub.add_parser("demo", help="run every strategy on synthetic data, offline")
    d.set_defaults(func=cmd_demo)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
