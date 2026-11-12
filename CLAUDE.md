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

- Silo is for UK position trading (weeks to months). Keep `Constraints.position_trading()`
  as the default for books and CLI runs; no intraday strategies.
- Liquidity rules (`silo.liquidity.Liquidity`) stay on for position-trading books: no penny stocks
  (under £1), no names trading under £5m a day, positions small enough to sell in two days.
- Prices stay split-adjusted, never dividend-adjusted; dividends are paid as cash.
- Trading 212 is Practice only: `demo.trading212.com`. Never add `live.trading212.com` or any
  setting that could point the client at it.
- Never overwrite a vault snapshot or rewrite a paper journal.

Run `pytest` before pushing.
