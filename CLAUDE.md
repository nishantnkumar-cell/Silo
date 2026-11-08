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

Run `pytest` before pushing.
