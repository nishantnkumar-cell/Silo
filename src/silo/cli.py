"""Command line: fetch data, backtest, run lab studies, and paper-trade."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd

from silo.broker import CostModel
from silo.constraints import Constraints
from silo.data.sources import SOURCES, synthetic_bars
from silo.data.vault import DataVault, normalize_bars
from silo.engine import run_backtest
from silo.liquidity import Liquidity
from silo.markets import market_for_symbols
from silo.strategies import REGISTRY

DISCLAIMER = "Research only. Simulated results, not financial advice. Silo never trades real money."
STUDY_STRATEGIES = [n for n in REGISTRY if n not in ("buy_and_hold", "random")]
WRAPPERS = ("learned",)  # needs a base agent and lessons; left out of "all" runs
AI_STRATEGIES = [n for n in REGISTRY if REGISTRY[n].uses_llm]


def _everything() -> list[str]:
    """All strategies; the AI ones only when a model is chosen with SILO_LLM."""
    import os

    if os.environ.get("SILO_LLM"):
        return [n for n in REGISTRY if n not in WRAPPERS]
    return [n for n in REGISTRY if n not in AI_STRATEGIES and n not in WRAPPERS]


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
    auto = CostModel.for_uk_investor(market.code) if market else CostModel()
    if args.style == "daily" and market and market.code == "US":
        auto = CostModel.us_from_uk()  # a UK investor pays FX on dollar ETFs
    costs = {"auto": auto, "us": CostModel(), "us-from-uk": CostModel.us_from_uk(),
             "us-shares": CostModel.us_shares_from_uk(), "uk": CostModel.uk(),
             "crypto": CostModel.crypto(), "none": CostModel(0, 0, 0, 0, 0)}[args.costs]
    if getattr(args, "cost_scale", None):
        costs = costs.scaled(args.cost_scale)
    liq_overrides = {k: v for k, v in {"min_price": args.min_price,
                                       "min_value_traded": args.min_value_traded}.items() if v is not None}
    if args.style == "daily":
        if market is None or market.code == "UK":
            raise SystemExit("--style daily is for the separate silver/crypto books, not UK shares")
        liq = replace(Liquidity.for_market(market.code), **liq_overrides)
        constraints = Constraints.daily_trading(
            liquidity=liq,
            **{k: v for k, v in {"min_hold_days": args.min_hold}.items() if v is not None},
        )
        return {"costs": costs, "constraints": constraints}
    if args.style == "position":
        liq = Liquidity.for_market(market.code if market else "UK")
        liq = replace(liq, **{k: v for k, v in {"min_price": args.min_price,
                                                 "min_value_traded": args.min_value_traded}.items() if v is not None})
        constraints = Constraints.position_trading(
            liquidity=liq,
            **{k: v for k, v in {"rebalance": args.rebalance, "min_hold_days": args.min_hold}.items() if v is not None},
        )
    else:
        constraints = Constraints(rebalance=args.rebalance or "daily", min_hold_days=args.min_hold or 0)
    return {"costs": costs, "constraints": constraints}


def _add_rule_flags(p) -> None:
    p.add_argument("--costs", choices=["auto", "us", "us-from-uk", "us-shares", "uk", "crypto", "none"],
                   default="auto",
                   help="cost model; auto picks UK costs (with stamp duty) for .L tickers, FX fee and 15%% "
                        "dividend tax for US shares, exchange fees "
                        "for crypto, and adds the FX fee for US ETFs in daily books")
    p.add_argument("--cost-scale", type=float,
                   help="multiply every cost, e.g. 2 to check a result survives double the costs")
    p.add_argument("--style", choices=["position", "daily", "any"], default="position",
                   help="position: weekly decisions, 20-day minimum hold, turnover and size caps; "
                        "daily: separate silver/crypto books deciding on each daily close")
    p.add_argument("--rebalance", choices=["daily", "weekly", "monthly"])
    p.add_argument("--min-hold", type=int, help="minimum trading days to hold a new position")
    p.add_argument("--min-price", type=float,
                   help="no penny stocks: lowest share price allowed (default £1 in the UK)")
    p.add_argument("--min-value-traded", type=float,
                   help="lowest median daily value traded over 60 days (default £5m in the UK)")


def cmd_fetch(args) -> int:
    vault = DataVault(args.data_dir)
    symbols = args.symbols
    if args.universe == "uk":
        from silo.universe import UK_CANDIDATES

        symbols = symbols + UK_CANDIDATES
    elif args.universe:
        from silo.universe import DAILY_UNIVERSES

        symbols = symbols + DAILY_UNIVERSES[args.universe]
    if args.symbols_file:
        symbols = symbols + json.loads(Path(args.symbols_file).read_text())
    symbols = list(dict.fromkeys(symbols))
    failed = []
    if args.macro:
        from silo.data.macro import fetch_all

        failed += [f"macro:{n}" for n in fetch_all(vault)]
    for i, symbol in enumerate(symbols):
        if i and args.pause:
            time.sleep(args.pause)  # be gentle with the data source on long lists
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
    from silo.data.macro import synthetic_macro

    symbols = [f"SYN{i}" for i in range(n)]
    tables = {s: normalize_bars(synthetic_bars(n_days=n_days, seed=i)) for i, s in enumerate(symbols)}
    return symbols, tables | synthetic_macro(tables[symbols[0]].index)


def _run_all(tables, symbols, names, start=None, end=None, **rules) -> dict[str, dict[str, float]]:
    return {n: run_backtest(REGISTRY[n](symbols), tables, start=start, end=end, **rules).metrics() for n in names}


def cmd_backtest(args) -> int:
    vault = DataVault(args.data_dir)
    tables = vault.tables(args.symbols)
    names = _everything() if args.strategy == "all" else ["buy_and_hold", args.strategy]
    _print_table(_run_all(tables, args.symbols, names, args.start, args.end, **_rules(args, args.symbols)))
    return 0


def cmd_demo(args) -> int:
    symbols, tables = _demo_tables()
    print("Synthetic random-walk data: there is no real signal here, so no strategy")
    print("should reliably beat buy-and-hold. If one does, suspect luck.\n")
    names = [n for n in REGISTRY if n not in AI_STRATEGIES and n not in WRAPPERS]
    _print_table(_run_all(tables, symbols, names, costs=CostModel(), constraints=Constraints()))
    print("\nAI strategies (desk, panel, silo_agent) need a model: set SILO_LLM=ollama:<model> or rulebot.")
    return 0


def cmd_lab(args) -> int:
    from silo.lab import HoldoutLog, run_study
    from silo.lab.report import write_report

    if args.demo:
        symbols, tables = _demo_tables()
    else:
        if args.universe:
            from silo.universe import DAILY_UNIVERSES

            args.symbols = args.symbols + DAILY_UNIVERSES[args.universe]
        vault = DataVault(args.data_dir)
        if args.all_candidates:
            from silo.universe import UK_CANDIDATES

            symbols = [s for s in UK_CANDIDATES if vault.snapshots(s)]
        elif args.symbols_file:
            symbols = json.loads(Path(args.symbols_file).read_text())
        else:
            symbols = args.symbols
        if not symbols:
            print("give symbols, --symbols-file, --all-candidates, or use --demo")
            return 2
        tables = vault.tables(symbols)  # bars plus cached context tables (macro, news)
        short = {s for s in symbols if len(tables[s]) <= 260}
        symbols = [s for s in symbols if s not in short]
        tables = {k: v for k, v in tables.items() if k not in short}
    if args.strategy != "all":
        names = [args.strategy]
    elif args.style == "daily":
        from silo.league import DAILY_CONTESTANTS

        names = [n for n in DAILY_CONTESTANTS if n != "random"] + ["sma_cross"]
    else:
        names = [n for n in STUDY_STRATEGIES if n in _everything()]
    log = HoldoutLog(Path(args.out) / "holdout_log.jsonl")
    for name in names:
        result = run_study(
            REGISTRY[name], symbols, tables,
            holdout_frac=args.holdout, n_folds=args.folds, n_random=args.random,
            open_holdout=args.open_holdout, holdout_log=log, **_rules(args, symbols),
            model_cutoff=getattr(REGISTRY[name](symbols), "model_cutoff", None) if name in AI_STRATEGIES else None,
        )
        path = write_report(result, Path(args.out) / f"{name}.html")
        status = "PASS" if result.interesting else "fail"
        print(f"{name:10s} {status}  " + "; ".join(c.detail for c in result.checks) + f"  -> {path}")
    print(f"\n{DISCLAIMER}")
    return 0


def cmd_universe(args) -> int:
    from silo.universe import UK_CANDIDATES, quality_screen
    from silo.universe.pool import rank_pool, sectors_today

    vault = DataVault(args.data_dir)
    market = args.market.upper()
    if args.candidates:
        candidates = json.loads(Path(args.candidates).read_text())
    elif market == "UK":
        candidates = UK_CANDIDATES
    else:
        print("pass --candidates pool/us-candidates.json (from `silo pool instruments`)")
        return 1
    have = [s for s in candidates if vault.snapshots(s)]
    if not have:
        print(f"no {market} data cached yet; run `silo fetch --symbols-file ...` first")
        return 1
    tables = {s: vault.get_bars(s) for s in have}
    sectors = None
    if args.sector_cap:
        prelim = rank_pool(tables, liquidity=Liquidity.for_market(market))
        sectors = sectors_today(list(prelim.index[prelim["passes"]]))
    table = rank_pool(tables, liquidity=Liquidity.for_market(market), top=args.top, sectors=sectors,
                      max_volatility=args.max_volatility)
    if args.quality:
        q = quality_screen(list(table.index[table["picked"]]))
        table = table.join(q[["quality", "reason"]].rename(columns={"reason": "quality_reason"}))
        table["picked"] = table["picked"] & table["quality"].fillna(False).astype(bool)
    with pd.option_context("display.max_rows", None, "display.width", 160):
        print(table[table["passes"]].to_string() if args.passing_only else table.to_string())
    picked = list(table.index[table["picked"]])
    print(f"\n{int(table['passes'].sum())} of {len(have)} pass the gate and trend; picked {len(picked)}: "
          f"{' '.join(picked)}")
    print(DISCLAIMER)
    if args.save:
        Path(args.save).write_text(json.dumps(picked))
        print(f"saved to {args.save}")
    return 0


def cmd_pool(args) -> int:
    """Trading 212 Practice instrument list -> Yahoo candidate lists per market."""
    from datetime import datetime, timezone

    from silo.brokers.trading212 import Trading212Demo
    from silo.universe.pool import candidates_from_instruments

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    instruments = Trading212Demo().instruments()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snap = out / f"t212-instruments-{stamp}.json"
    if snap.exists():
        raise SystemExit(f"{snap} already exists; snapshots are never overwritten")
    snap.write_text(json.dumps(instruments))
    by_market = candidates_from_instruments(instruments)
    for market, rows in by_market.items():
        path = out / f"{market.lower()}-candidates.json"
        path.write_text(json.dumps(sorted(rows)))
        print(f"{market}: {len(rows)} shares -> {path}")
    print(f"{len(instruments)} instruments listed; snapshot kept at {snap}")
    print("next: silo fetch --symbols-file pool/us-candidates.json --pause 1, "
          "then silo universe us --candidates pool/us-candidates.json --top 60 --save us.json")
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
    if args.action == "t212-fills":
        from silo.brokers.trading212 import Trading212Demo
        from silo.fills import collect_fills, format_gap, gap_report, save_gap

        new = collect_fills(book, Trading212Demo())
        print(f"{len(new)} new Practice fill(s) logged")
        gap, summary = gap_report(book)
        save_gap(book, gap, summary)
        print(format_gap(gap, summary, book.config()["currency"]))
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
            from silo.fills import collect_fills, gap_report, save_gap
            from silo.paper import mirror_latest_decision

            try:
                got = collect_fills(book, Trading212Demo())
                save_gap(book, *gap_report(book))
                if got:
                    print(f"Trading 212 Practice: {len(got)} new fill(s) logged; see `silo paper t212-fills {book.name}`")
            except Exception as exc:  # noqa: BLE001 - fills are a report; never block mirroring
                print(f"Trading 212 Practice fills not collected: {exc}")
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


def _league_backtest(args, vault) -> int:
    import os

    from silo.lab.league_backtest import run_league_backtest, write
    from silo.league import AI_CONTESTANTS, RULE_CONTESTANTS
    from silo.universe import UK_CANDIDATES

    if args.symbols:
        symbols = args.symbols
    elif args.all_candidates:
        symbols = [s for s in UK_CANDIDATES if vault.snapshots(s)]
    else:
        symbols = json.loads(Path(args.symbols_file).read_text())
    tables = vault.tables(symbols)
    symbols = [s for s in symbols if len(tables[s]) > 260]
    names = args.only or (RULE_CONTESTANTS + (AI_CONTESTANTS if args.with_ai else []))
    if any(REGISTRY[n].uses_llm for n in names):
        if not os.environ.get("SILO_LLM"):
            print("AI contestants need SILO_LLM (e.g. ollama:qwen2.5:7b)")
            return 2
        os.environ.setdefault("SILO_LLM_LOG", str(Path(args.out) / "llm"))  # shared cache: silo_agent reuses desk/panel
    print(f"{len(names)} contestant(s) on {len(symbols)} symbols, {args.start or 'start'} to {args.end or 'end'}")
    entries, bench = run_league_backtest(names, symbols, tables, args.start, args.end, **_rules(args, symbols))
    page, data = write(entries, bench, args.out, args.title)
    print(f"\nreport: {page}\ndata:   {data}\n{DISCLAIMER}")
    return 0


def cmd_check_data(args) -> int:
    from silo.lab.league_backtest import data_check
    from silo.universe import UK_CANDIDATES

    vault = DataVault(args.data_dir)
    symbols = args.symbols or [s for s in UK_CANDIDATES if vault.snapshots(s)]
    table = data_check({s: vault.get_bars(s) for s in symbols})
    with pd.option_context("display.max_rows", None, "display.width", 160):
        print(table.to_string())
    bad = list(table.index[table["suspect"].astype(bool)])
    print(f"\n{len(bad)} suspect series: {' '.join(bad)}")
    return 0


def cmd_league(args) -> int:
    from silo.league import create_league

    vault = DataVault(args.data_dir)
    if args.action == "backtest":
        return _league_backtest(args, vault)
    if args.universe:
        from silo.universe import DAILY_UNIVERSES

        symbols = DAILY_UNIVERSES[args.universe]
    else:
        symbols = args.symbols or json.loads(Path(args.symbols_file).read_text())
    rules = _rules(args, symbols)
    made = create_league(symbols, vault, args.paper_dir, args.prefix, args.with_ai, args.cash,
                         rules["constraints"], rules["costs"], daily=args.style == "daily")
    for cfg in made:
        print(f"created {cfg['name']}: {cfg['strategy']} on {len(symbols)} symbols, starts after {cfg['start_after'][:10]}")
    if not made:
        print("every league book already exists; nothing created")
    return 0


def cmd_daily(args) -> int:
    """Fetch once, advance every paper book, print the leaderboard, optionally mirror one."""
    from silo.league import fetch_for_books, run_all, write_leaderboard

    vault = DataVault(args.data_dir)
    if not args.no_fetch:
        failed = fetch_for_books(vault, args.paper_dir, SOURCES[args.source])
        if args.news:
            from silo.data.news import fetch_news
            from silo.league import book_names
            from silo.paper import PaperBook

            symbols = sorted({s for n in book_names(args.paper_dir) for s in PaperBook(n, args.paper_dir).config()["symbols"]})
            failed += [f"news:{s}" for s in fetch_news(vault, symbols, delay=args.news_delay, log=lambda m: None)]
        if not args.no_macro:
            from silo.data.macro import fetch_all

            failed += [f"macro:{n}" for n in fetch_all(vault, log=lambda m: None)]
        if failed:
            print(f"{len(failed)} symbol(s) not refreshed: {' '.join(failed)}")
    board = run_all(vault, args.paper_dir)
    path = write_leaderboard(board, args.paper_dir)
    print(f"\n{'book':26s} {'days':>5s} {'return':>8s} {'buy&hold':>9s} {'excess':>8s} {'names':>5s}")
    for r in board:
        if "error" in r:
            print(f"{r['book']:26s}  ERROR {r['error']}")
            continue
        print(f"{r['book']:26s} {r['days']:5d} {r['return']:8.2%} {r['benchmark_return']:9.2%} "
              f"{r['excess']:+8.2%} {r['positions']:5d}")
    print(f"saved {path}")
    code = 0
    if args.mirror:
        from silo.brokers.trading212 import Trading212Demo
        from silo.paper import PaperBook, mirror_latest_decision

        mbook = PaperBook(args.mirror, args.paper_dir)
        try:
            from silo.fills import collect_fills, gap_report, save_gap

            got = collect_fills(mbook, Trading212Demo())
            gap, summary = gap_report(mbook)
            save_gap(mbook, gap, summary)
            if got:
                print(f"\nTrading 212 Practice ({args.mirror}): {len(got)} new fill(s) logged; "
                      f"median gap to simulator {summary.get('median_slippage_bps', float('nan')):+.1f} bp")
        except Exception as exc:  # noqa: BLE001 - fills are a report; never block mirroring
            print(f"\nTrading 212 Practice fills not collected: {exc}")
        try:
            sent = mirror_latest_decision(mbook, Trading212Demo(), fractional=args.fractional)
            print(f"\nTrading 212 Practice ({args.mirror}): {len(sent)} order(s) sent" if sent else
                  f"\nTrading 212 Practice ({args.mirror}): no new rebalance to mirror")
            for e in sent:
                print(f"  {e['ticker']:14s} {e['quantity']:+.4g}  {e.get('status') or e.get('error')}")
        except Exception as exc:  # noqa: BLE001 - report, keep the paper results
            print(f"\nTrading 212 Practice mirror failed: {exc}")
            code = 1
    print(f"\n{DISCLAIMER}")
    return code


def _news_symbols(args) -> list[str]:
    if args.symbols:
        return args.symbols
    path = Path(args.symbols_file)
    if path.exists():
        return json.loads(path.read_text())
    from silo.universe import UK_CANDIDATES

    return UK_CANDIDATES


def cmd_news(args) -> int:
    """Company announcements (RNS) for the news agent. Personal use; fetched slowly, stored locally."""
    from silo.data import news

    vault = DataVault(args.data_dir)
    if args.action == "fetch":
        symbols = _news_symbols(args)
        print(f"fetching announcements for {len(symbols)} companies, one request every {args.delay:g}s; "
              "safe to stop and rerun, it resumes where it left off")
        failed = news.fetch_news(vault, symbols, since=args.since, delay=args.delay)
        if failed:
            print(f"failed: {' '.join(failed)}")
        return 0
    try:
        table = vault.get_context(news.TABLE)
    except KeyError:
        print("no announcements stored yet; run `silo news fetch` first")
        return 1
    if args.action == "rescore":
        news.rescore(vault, news.Fetcher(args.delay))
        return 0
    if args.action == "show":
        rows = table[table["symbol"].isin(_news_symbols(args))] if args.symbols else table
        cols = [c for c in ["symbol", "available_at", "category", "expect", "dealing", "headline"] if c in rows]
        with pd.option_context("display.max_rows", None, "display.width", 200, "display.max_colwidth", 70):
            print(rows[cols].tail(args.limit or 40).to_string())
        print("\n" + rows["category"].value_counts().to_string())
        return 0
    from silo.agents.llm import client_from_spec
    from silo.agents.news_extract import probe, run_extraction

    client = client_from_spec(log_dir=Path(args.data_dir) / "news_llm_log")
    if args.action == "extract":
        run_extraction(vault, client, limit=args.limit)
    else:
        res = probe(vault, client, n=args.limit or 50)
        print(f"asked {res['asked']}: model named the company {res['company_hit_rate'] or 0:.0%} of the time "
              f"and the year {res['year_hit_rate'] or 0:.0%}. Well above zero means masking leaks; treat its "
              f"answers before its training cutoff ({client.cutoff}) with suspicion.")
    return 0


def cmd_web(args) -> int:
    from silo.web import serve

    serve(args.paper_dir, args.reports_dir, args.host, args.port, open_browser=not args.no_browser)
    return 0


def _learn_data(args):
    """Symbols and tables for `silo learn`, plus the first day of the locked holdout."""
    if args.demo:
        from silo.markets import UK

        symbols = [f"SYN{i}.L" for i in range(10)]
        tables = {s: normalize_bars(synthetic_bars(n_days=2000, seed=i, start_price=2500), UK, pence=True)
                  for i, s in enumerate(symbols)}
    else:
        vault = DataVault(args.data_dir)
        if args.all_candidates:
            from silo.universe import UK_CANDIDATES

            symbols = [s for s in UK_CANDIDATES if vault.snapshots(s)]
        elif args.symbols:
            symbols = args.symbols
        else:
            symbols = json.loads(Path(args.symbols_file).read_text())
        tables = vault.tables(symbols)  # bars plus cached context tables (macro, news)
        short = {s for s in symbols if len(tables[s]) <= 260}
        symbols = [s for s in symbols if s not in short]
        tables = {k: v for k, v in tables.items() if k not in short}
    dates = sorted(set().union(*(tables[s].index for s in symbols)))
    holdout_start = dates[int(len(dates) * (1 - args.holdout))]
    return symbols, tables, dates, holdout_start


def _before(dates, day):
    earlier = [d for d in dates if d < day]
    return earlier[-1]


def cmd_learn(args) -> int:
    """Learn from past trades: journal, review, gate, walk-forward. Never touches the holdout."""
    import os

    from silo.learning import (
        Ledger,
        ai_review,
        build_journal,
        learning_walk_forward,
        mine,
        run_gate,
    )
    from silo.learning.journal import summary, write_journal

    out = Path(args.out)
    ledger = Ledger(out / "ledger.jsonl")
    if args.action != "list":
        args.strategy = args.strategy or "uk_trend"
    if args.action == "list":
        recs = ledger.lessons(args.strategy)
        if not recs:
            print(f"no lessons in {ledger.path} yet; start with `silo learn review`")
        for i, r in recs.items():
            from silo.learning import Lesson

            last = r["gates"][-1] if r["gates"] else None
            when = f", last gate {last['test_start']} to {last['test_end']}" if last else ""
            print(f"{i}  {r['status']:9s} {r['base']:12s} {Lesson.from_dict(r['lesson']).describe()}"
                  f"  (from trades to {r['learn_end'][:10]}, {r['source']}{when})")
        return 0
    if args.action == "params":
        lessons = [x.to_dict() for x in ledger.accepted(args.strategy)]
        print(json.dumps({"base": args.strategy, "base_params": json.loads(args.params), "lessons": lessons}))
        if not lessons:
            print(f"(no accepted lessons for {args.strategy}: the learned agent would trade exactly like it)",
                  file=sys.stderr)
        return 0

    symbols, tables, dates, holdout_start = _learn_data(args)
    research_end = _before(dates, holdout_start)
    rules = _rules(args, symbols)
    params = json.loads(args.params)
    if REGISTRY[args.strategy].uses_llm and not os.environ.get("SILO_LLM"):
        print("AI agents need SILO_LLM (e.g. ollama:qwen2.5:7b)")
        return 2
    os.environ.setdefault("SILO_LLM_LOG", str(out / "llm"))
    client = None
    if getattr(args, "ai", False):
        from silo.agents.llm import client_from_spec

        client = client_from_spec(log_dir=out / "llm")
    print(f"{args.strategy} on {len(symbols)} symbols; locked holdout from {holdout_start.date()} is never used")

    if args.action == "review":
        start = pd.Timestamp(args.start) if args.start else dates[0]
        end = pd.Timestamp(args.end) if args.end else dates[int(len(dates) * 0.6)]
        if end >= holdout_start:
            print(f"--end must be before the holdout ({holdout_start.date()})")
            return 2
        res = run_backtest(REGISTRY[args.strategy](symbols, **params), tables, start=start, end=end, **rules)
        trades = build_journal(res, tables, symbols, core=args.strategy.startswith("core"))
        csv, _ = write_journal(trades, out / f"journal-{args.strategy}")
        s = summary(trades)
        print(f"journal: {csv} ({s.get('closed', 0)} closed trades, beat the market "
              f"{s.get('win_rate_vs_market', 0):.0%} of the time, {s.get('avg_excess', 0):+.1%} a trade on average)")
        cands, tried = mine(trades)
        lessons = ai_review(trades, client, cands) if client else [c.lesson for c in cands]
        evidence = {c.lesson.id: c.to_dict() for c in cands}
        ids = ledger.propose(lessons, args.strategy, start.date(), end.date(), "ai" if client else "statistics",
                             evidence)
        print(f"tried {tried} splits; proposed {len(ids)} lesson(s):")
        for x in lessons:
            print(f"  {x.id}  {x.describe()}")
        if ids:
            print(f"\nnext: silo learn gate --strategy {args.strategy} {' '.join(ids)}"
                  f"   (tests on data after {end.date()}, before the holdout)")
        return 0

    if args.action == "gate":
        recs = ledger.lessons(args.strategy)
        missing = [i for i in args.ids if i not in recs]
        if missing or not args.ids:
            print(f"unknown lesson id(s) for {args.strategy}: {' '.join(missing) or '(none given)'}; see `silo learn list`")
            return 2
        from silo.learning import Lesson

        lessons = [Lesson.from_dict(recs[i]["lesson"]) for i in args.ids]
        learned_until = max(pd.Timestamp(recs[i]["learn_end"]) for i in args.ids)
        t0 = pd.Timestamp(args.test_start) if args.test_start else next(d for d in dates if d > learned_until)
        t1 = pd.Timestamp(args.test_end) if args.test_end else research_end
        g = run_gate(args.strategy, lessons, symbols, tables, learned_until=learned_until, test_start=t0,
                     test_end=t1, holdout_start=holdout_start, base_params=params, n_random=args.random,
                     ledger=ledger, **rules)
        print(f"test window {t0.date()} to {t1.date()} (lessons drawn from trades up to {learned_until.date()})")
        for c in g.checks:
            mark = {True: "pass", False: "FAIL", None: "info"}[c.passed]
            print(f"  {mark:4s}  {c.name}: {c.detail}")
        print(f"\n{'ACCEPTED' if g.passed else 'REJECTED'}: return {g.variant_return:.1%} with the lesson(s), "
              f"{g.base_return:.1%} without, buy-and-hold {g.bench_return:.1%}. Recorded in {ledger.path}")
        if g.passed:
            print(f"use it: silo paper new <book> --strategy learned --params "
                  f"\"$(silo learn params --strategy {args.strategy})\"")
        print(f"\n{DISCLAIMER}")
        return 0

    # walkforward
    print(f"learning walk-forward, {args.folds} folds, reviewer: {'AI + statistics' if client else 'statistics'}")
    wf = learning_walk_forward(args.strategy, symbols, tables, end=holdout_start, n_folds=args.folds,
                               base_params=params, client=client, progress=print, **rules)
    m = wf.metrics()
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"walkforward-{args.strategy}.json"
    path.write_text(json.dumps({**wf.to_dict(), "disclaimer": DISCLAIMER}, indent=2, default=str))
    print(f"\nout of sample: learning {m['learned_return']:.1%} (Sharpe {m['learned_sharpe']:.2f}), "
          f"never learning {m['base_return']:.1%} ({m['base_sharpe']:.2f}), "
          f"buy-and-hold {m['bench_return']:.1%} ({m['bench_sharpe']:.2f})")
    print(f"learning helped in {m['folds_improved']} of {m['folds']} folds; deflated Sharpe of the "
          f"difference {m['deflated']:.2f} (0.95 or more would be convincing)\nsaved {path}\n\n{DISCLAIMER}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="silo", description=DISCLAIMER)
    p.add_argument("--data-dir", default="data")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="download daily bars into the local vault (new snapshot each time)")
    f.add_argument("symbols", nargs="*")
    f.add_argument("--universe", choices=["uk", "silver", "crypto"],
                   help="also fetch a candidate list: UK shares, silver ETFs (SLV SIVR) or crypto (BTC ETH)")
    f.add_argument("--source", choices=sorted(SOURCES), default="yfinance")
    f.add_argument("--symbols-file", help="JSON list of symbols, e.g. pool/us-candidates.json")
    f.add_argument("--pause", type=float, default=0.0, help="seconds between symbols on long lists")
    f.add_argument("--macro", action="store_true",
                   help="also fetch cross-market series: pound, oil, metals, gilt yield, US shares, VIX")
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
    lab.add_argument("--universe", choices=["silver", "crypto"], help="study a daily-trading universe")
    lab.add_argument("--symbols-file", help="JSON list of symbols, e.g. from `silo universe uk --save`")
    lab.add_argument("--all-candidates", action="store_true",
                     help="every UK candidate with cached data (liquidity still applied point-in-time)")
    _add_rule_flags(lab)
    lab.set_defaults(func=cmd_lab)

    u = sub.add_parser("universe", help="gate and rank a candidate list on cached data (momentum and trend)")
    u.add_argument("market", choices=["uk", "us"])
    u.add_argument("--candidates", help="JSON list of symbols (default: the built-in UK list)")
    u.add_argument("--top", type=int, default=60, help="how many of the best-ranked to pick")
    u.add_argument("--max-volatility", type=float, default=0.60, help="drop names wilder than this a year")
    u.add_argument("--sector-cap", action="store_true",
                   help="at most 25%% of picks from one sector (today's sectors from Yahoo; live use only)")
    u.add_argument("--passing-only", action="store_true", help="print only names that pass")
    u.add_argument("--quality", action="store_true", help="also check today's fundamentals (live use only)")
    u.add_argument("--save", help="write passing symbols to this JSON file")
    u.set_defaults(func=cmd_universe)

    po = sub.add_parser("pool", help="list every UK and US share on Trading 212 Practice as Yahoo candidates")
    po.add_argument("action", choices=["instruments"])
    po.add_argument("--out", default="pool", help="folder for the dated snapshot and candidate lists")
    po.set_defaults(func=cmd_pool)

    pp = sub.add_parser("paper", help="forward paper trading on the simulator")
    pp.add_argument("action", choices=["new", "run", "status", "t212-map", "t212-check", "t212-fills"])
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

    lg = sub.add_parser("league", help="one paper book per contestant on the same universe")
    lg.add_argument("action", choices=["new", "backtest"],
                    help="new: start paper books; backtest: replay history for every contestant")
    lg.add_argument("--symbols", nargs="*")
    lg.add_argument("--symbols-file", default="uk.json", help="JSON list, e.g. from `silo universe uk --save`")
    lg.add_argument("--universe", choices=["silver", "crypto"], help="use a daily-trading universe as the symbols")
    lg.add_argument("--prefix", default="league")
    lg.add_argument("--with-ai", action="store_true",
                    help="add the AI contestants: desk, panel, silo_agent and their cross-market versions (needs SILO_LLM)")
    lg.add_argument("--cash", type=float, default=100_000.0)
    lg.add_argument("--paper-dir", default="paper")
    lg.add_argument("--start", help="backtest: first date, e.g. 2012-01-01")
    lg.add_argument("--end", help="backtest: last date")
    lg.add_argument("--only", nargs="*", choices=list(REGISTRY), help="backtest: just these contestants")
    lg.add_argument("--all-candidates", action="store_true",
                    help="backtest: every UK candidate with data, instead of today's screened list (less look-ahead)")
    lg.add_argument("--out", default="reports/league", help="backtest: folder for league.html and league.json")
    lg.add_argument("--title", default="League backtest")
    _add_rule_flags(lg)
    lg.set_defaults(func=cmd_league)

    cd = sub.add_parser("check-data", help="flag stale or broken price series in the vault")
    cd.add_argument("symbols", nargs="*")
    cd.set_defaults(func=cmd_check_data)

    dy = sub.add_parser("daily", help="fetch prices, advance every paper book, print the leaderboard")
    dy.add_argument("--paper-dir", default="paper")
    dy.add_argument("--source", choices=sorted(SOURCES), default="yfinance")
    dy.add_argument("--no-fetch", action="store_true")
    dy.add_argument("--no-macro", action="store_true", help="skip refreshing the cross-market series")
    dy.add_argument("--news", action="store_true", help="also fetch new company announcements (a few minutes)")
    dy.add_argument("--news-delay", type=float, default=3.0)
    dy.add_argument("--mirror", metavar="BOOK", help="mirror this book's newest rebalance to Trading 212 Practice")
    dy.add_argument("--fractional", action="store_true")
    dy.set_defaults(func=cmd_daily)

    nw = sub.add_parser("news", help="UK company announcements (RNS) for the news agent")
    nw.add_argument("action", choices=["fetch", "show", "rescore", "extract", "probe"],
                    help="fetch: download new announcements (slow, polite, resumable); show: list stored ones; "
                         "rescore: re-run the rules on stored ones after a rule fix; "
                         "extract: a local model reads results text the rules couldn't; probe: test the masking")
    nw.add_argument("--symbols", nargs="*")
    nw.add_argument("--symbols-file", default="uk.json")
    nw.add_argument("--since", default="2012-01-01", help="oldest announcement to fetch")
    nw.add_argument("--delay", type=float, default=3.0, help="seconds between requests")
    nw.add_argument("--limit", type=int, help="rows to show, or announcements to read with the model")
    nw.set_defaults(func=cmd_news)

    w = sub.add_parser("web", help="local dashboard: paper books, lab reports, Trading 212 Practice key")
    w.add_argument("--port", type=int, default=8212)
    w.add_argument("--host", default="127.0.0.1", help="loopback only; anything else is refused")
    w.add_argument("--paper-dir", default="paper")
    w.add_argument("--reports-dir", default="reports")
    w.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    w.set_defaults(func=cmd_web)

    ln = sub.add_parser("learn", help="learn from past trades: journal, lessons, and a gate before anything goes live")
    ln.add_argument("action", choices=["review", "gate", "walkforward", "list", "params"],
                    help="review: journal + proposed lessons; gate: test lessons on later data; "
                         "walkforward: test learn-then-trade itself; list: the ledger; params: for `silo paper new`")
    ln.add_argument("ids", nargs="*", help="gate: lesson ids from `silo learn review`")
    ln.add_argument("--strategy", choices=[n for n in REGISTRY if n not in ("learned", "buy_and_hold")],
                    help="the agent that learns (default uk_trend)")
    ln.add_argument("--params", default="{}", help="the agent's parameters as JSON")
    ln.add_argument("--symbols", nargs="*")
    ln.add_argument("--symbols-file", default="uk.json")
    ln.add_argument("--all-candidates", action="store_true")
    ln.add_argument("--demo", action="store_true", help="synthetic data, offline")
    ln.add_argument("--start", help="review: first day of trades to learn from")
    ln.add_argument("--end", help="review: last day of trades to learn from (default: 60%% of the data)")
    ln.add_argument("--test-start", help="gate: default the day after the lessons' last trade")
    ln.add_argument("--test-end", help="gate: default the day before the locked holdout")
    ln.add_argument("--holdout", type=float, default=0.2, help="fraction locked as holdout (same as `silo lab`)")
    ln.add_argument("--random", type=int, default=20, help="gate: random-filter control runs")
    ln.add_argument("--folds", type=int, default=4)
    ln.add_argument("--ai", action="store_true", help="add the AI reviewer (needs SILO_LLM)")
    ln.add_argument("--out", default="learning", help="folder for the ledger, journals and reports")
    _add_rule_flags(ln)
    ln.set_defaults(func=cmd_learn)

    d = sub.add_parser("demo", help="run every strategy on synthetic data, offline")
    d.set_defaults(func=cmd_demo)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
