# Silo: instructions for Claude

Silo is a research-only trading lab. These rules are absolute:

- Never add code that places orders with a real brokerage or touches real money.
  The only execution venues are `silo.broker.SimBroker` and paper endpoints in
  `silo.safety.PAPER_HOSTS`. Do not widen that allowlist to a live host.
- Every data read by a strategy or agent goes through `silo.firewall.PointInTimeView`.
  New data types must carry an `available_at` UTC column.
- Every new strategy must pass `tests/test_firewall.py::test_truncation_invariance`
  (add it to `silo.strategies.REGISTRY`).
- Results are always reported next to buy-and-hold and labelled research only.
- Never open the locked holdout (`--open-holdout`) to tune a strategy; it is for a final check, and every opening is logged.

- Silo is for UK-first position trading (weeks to months), plus US shares under the same rules
  (separate US books; a book never mixes markets; US costs include the 0.15% FX fee and 15% dividend
  tax via `CostModel.us_shares_from_uk()`). Keep `Constraints.position_trading()` as the default for
  books and CLI runs; no intraday strategies.
- The wider pool comes from Trading 212 Practice's instrument list (`silo pool instruments`), ordinary
  shares only, then the liquidity gate and momentum/trend ranking in `silo.universe.pool.rank_pool`.
- One exception: a separate daily-bar book for liquid silver ETFs and major crypto
  (`silo.universe.daily`, `--style daily`, `Constraints.daily_trading()`). It decides on daily
  closes only (never intraday), keeps the liquidity rules on, and never applies to UK shares.
- Liquidity rules (`silo.liquidity.Liquidity`) stay on for position-trading books: no penny stocks
  (under £1 or $5), no names trading under £5m a day ($20m in the US), positions small enough to
  sell in two days.
- Prices stay split-adjusted, never dividend-adjusted; dividends are paid as cash.
- Trading 212 is Practice only: `demo.trading212.com`. Never add `live.trading212.com` or any
  setting that could point the client at it.
- Never overwrite a vault snapshot or rewrite a paper journal.

Run `pytest` before pushing.
