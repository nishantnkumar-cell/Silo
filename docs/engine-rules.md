# Silo rules engine (v1)

> Research only. Paper and simulated money only. Never connected to a live brokerage.

Silo is a rules-first engine. Written rules decide every entry, size and exit; AI
only summarises and agrees or flags (the reviewer comes later and can never change
a trade). The strategy is `engine` (`silo.strategies.rules_engine.RulesEngine`).
The same data always gives the same decisions, and each decision records the rule
that fired in the journal notes.

Background: the direction doc ("Silo direction: rules-first engine") and the
research doc ("Trading rules that hold up: research and tests"), 2026-11-23.

## Rules

| Rule | What it does | Setting (default) |
| --- | --- | --- |
| R1-R4 | Rank names by 12-1 momentum; only names above their 200-day average; skip a name with a one-day move over 15% in the last 90 days; the top 30 are buyable | `lookback` 252, `skip` 21, `trend` 200, `jump_limit` 0.15, `jump_days` 90, `shortlist` 30 |
| R5 | No new buys while the market closes below its 200-day average (held names stay) | `market` (symbol; none = equal-weight index of the universe), `market_ma` 200, `market_filter` on |
| R6 | Buy in rank order until the book holds `max_names` | `max_names` 15 |
| R7 | Room for 1:4: skip a name whose nearest fresh supply zone is closer than 4R | `room_r` (off; set 4.0 to test) |
| R8 | Skip a name stopped out in the last 20 trading days | `cooloff` 20 |
| R9 | Risk per trade, chosen by the user: weight = risk / stop distance | `risk_per_trade` 0.01, never above 0.05 |
| R10 | At most 10% in one name; no new buys once open risk would pass 10% of the book | `max_weight` 0.10, `max_open_risk` 0.10 |
| R10b | Volatility scaling of the whole book (off by default; to be tested) | `vol_target` (e.g. 0.15) |
| R12-R13 | Stop 3 average daily ranges below entry; a close at or below it sells everything | `stop_atr` 3, `atr_days` 20 |
| R14 | At +2R sell half and move the stop to entry | `partial_at` 2, `partial_frac` 0.5 (0 = let it all run) |
| R15 | Then trail 3 ranges below the highest close | `trail_atr` 3 |
| R16 | Weekly: sell a holding that falls out of the top 60 | `keep_rank` 60 |
| C6 | Profit warning (or "below expectations" results): sell and block for 60 trading days. Needs RNS news in the vault | `warning_exit` on, `warning_block` 60 |

Dropped after the research: blocking buys before FOMC / Bank of England decisions
(C3) and halving risk in those weeks (C7). On 1994-2024 US data, holding cash for
the two days before each FOMC decision cut the market's return from 10.5% to 7.1%
a year. Results-date blocking (C5) waits for scheduled results dates in the vault.

## Running it

```
silo backtest <symbols...> --strategy engine
silo backtest <symbols...> --strategy engine --params '{"risk_per_trade": 0.02, "market": "ISF.L"}'
```

In a league backtest, pass settings per contestant:
`silo league backtest --symbols ... --only engine --params '{"engine": {"stop_atr": 4}}' --twins 20 --tracker ISF.L`.
There the market filter uses an equal-weight index of the universe unless the market symbol is also in `--symbols`.

Buy-and-hold in `silo backtest` runs without position-trading constraints, as in the league: under them the
liquidity gate refused half the names on day one and left 50% in cash for 14 years (+100% instead of +382%,
2012-2026).

Include the market symbol (ISF.L for UK, SPY for US) in the symbol list when you
pass `market`; it is used for the filter and never traded. Books and CLI runs keep
`Constraints.position_trading()` and the liquidity rules.

## Tests to run next (on real data)

1. Market filter on and off, UK and US, 2012-2026, against buy-and-hold and random twins.
2. `vol_target` 0.15 against off.
3. `stop_atr` 3, 4, 5; `partial_frac` 0.5 against 0.
4. `risk_per_trade` 0.01, 0.02, 0.03 with `max_open_risk` 0.10 and 1.0.
5. `room_r` 4 against off; `jump_limit` 0.15 against off.

Every variant counts as a trial in the deflated Sharpe. The holdout stays locked.
