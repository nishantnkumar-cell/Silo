# Silo

A research-only lab for testing whether AI adds anything real to trading
strategies. Silo backtests and paper-trades; **it never trades real money and
has no code path to a live brokerage.**

> Research only. Simulated results are not financial advice.

Plan: [AI Trading Research Harness — Development Plan](https://claude.ai/artifact/EC9GTfZzQMi9KwNZgW4CjL)

## What exists today (Phase 0: foundations)

| Part | File | What it does |
| --- | --- | --- |
| Live-trading guard | `src/silo/safety.py` | Refuses live brokerage hosts and live-looking credentials at import time. Only `paper-api.alpaca.markets` is allowlisted. |
| Data vault | `src/silo/data/vault.py` | Caches daily bars as Parquet; every row carries `available_at`, the UTC moment it became public. |
| Data sources | `src/silo/data/sources.py` | yfinance and Stooq (free), plus synthetic random walks for offline tests. |
| Time firewall | `src/silo/firewall.py` | `PointInTimeView` only returns rows public at the simulated clock; asking for newer data raises `FutureDataError`. |
| Simulated broker | `src/silo/broker.py` | Decide at close, fill at next open; commission, half-spread and slippage on every fill; long-only, no leverage; hook for LLM cost per decision. |
| Backtester | `src/silo/engine.py` | Day-by-day loop that routes every read through the firewall. |
| Baselines | `src/silo/strategies/classic.py` | Buy-and-hold, 50/200 SMA cross, Donchian breakout, RSI(2) dip-buying, Bollinger fade, 12-1 momentum, volatility-target wrapper. |
| Metrics | `src/silo/metrics.py` | Total return, CAGR, Sharpe, Sortino, max drawdown, turnover, fees. |

The key test is `tests/test_firewall.py::test_truncation_invariance`: every
strategy is run on full data and on data cut at day T, and its decisions
before T must be identical. If deleting the future changes the past, something
is peeking.

## Phase 1: backtest lab

`silo lab` runs one honest study per strategy and writes an HTML report:

| Check | File | What it does |
| --- | --- | --- |
| Walk-forward | `src/silo/lab/walkforward.py` | Picks parameters on past data only, then trades the next window untouched. Test windows are stitched into one out-of-sample curve. |
| Locked holdout | `src/silo/lab/holdout.py` | The last 20% of data is never used for research. `--open-holdout` evaluates it once and logs it; a second opening is flagged as no longer clean. |
| Random control | `src/silo/lab/controls.py`, `strategies/random_entry.py` | 30 strategies with no information that trade as often and stay as invested as the real one. |
| Luck statistics | `src/silo/lab/stats.py` | Deflated Sharpe ratio (penalises trying many parameter sets), probabilistic Sharpe, block-bootstrap interval. |
| Report | `src/silo/lab/report.py` | Verdict, pass/fail checks, interactive equity chart against buy-and-hold and the random band, fold table, holdout. Light and dark mode. |

A strategy is only called interesting when it beats buy-and-hold after costs,
survives the deflated Sharpe test, beats 95% of random controls, and (when
opened) beats buy-and-hold on the holdout.

On synthetic random walks, the Bollinger fade beats buy-and-hold and 97% of
random strategies, and even wins on the holdout, yet the deflated Sharpe test
(0.77) still flags it as likely luck. That is the lab doing its job.

## Phase 2: UK stocks and the paper loop

Silo now targets **UK shares held for weeks to months**, and runs strategies
forward on a paper book.

| Part | File | What it does |
| --- | --- | --- |
| Market rules | `src/silo/markets.py` | London close (16:35 local, DST-aware) sets `available_at`; `.L` tickers are UK; a book cannot mix markets. |
| UK prices | `src/silo/data/sources.py`, `data/vault.py` | Yahoo pence converted to pounds, isolated 100x rows repaired. Prices are split-adjusted but not dividend-adjusted; dividends are stored separately and paid as cash on the ex-date. |
| Snapshots | `src/silo/data/vault.py` | Every fetch is a new dated snapshot; nothing is overwritten. `vault.revisions(symbol)` shows rows a vendor changed. |
| UK costs | `src/silo/broker.py` | `CostModel.uk()`: 0.5% stamp duty on buys, £5 dealing fee, 20bp spread, 10bp slippage. Picked automatically for `.L` symbols. |
| Position trading | `src/silo/constraints.py` | Default: decide weekly, hold new positions at least 20 trading days, trade at most 50% of the book per rebalance, cap any name at 20%. Enforced in code. |
| Universe | `src/silo/universe/uk.py` | 70 liquid London candidates. Price screen (point-in-time): £5m a day traded, 20% to 45% volatility, no 50% crash. Optional quality screen (ROE, margins, debt) from today's fundamentals, for picking the live universe only. |
| Strategy | `uk_trend` in `strategies/classic.py` | Names above their 200-day average, ranked by risk-adjusted 6-month momentum, top 8, inverse-volatility sized. |
| Paper loop | `src/silo/paper.py` | Forward-only book on the simulator with a buy-and-hold shadow book, append-only journal, safe to rerun. Uses the same engine as backtests, so paper and backtest agree to the penny. |

Reports now show round trips and average holding period.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[data,dev]'

silo demo                                   # offline, synthetic data

# UK workflow
silo fetch --universe uk                    # ~70 London stocks, prices in pounds
silo universe uk --quality --save uk.json   # liquid, healthy volatility, good fundamentals
silo lab $(python -c "import json;print(' '.join(json.load(open('uk.json'))))") --strategy uk_trend
silo paper new uk-trend --symbols-file uk.json --strategy uk_trend
silo paper run uk-trend --fetch             # once a day after 16:35 London time
silo paper status uk-trend

silo fetch SPY QQQ IWM TLT GLD              # free daily bars from Yahoo
silo backtest SPY QQQ IWM TLT GLD           # every baseline vs buy-and-hold
silo backtest SPY --strategy sma_cross --start 2015-01-01
silo lab --demo --strategy bollinger        # offline lab study -> reports/bollinger.html
silo lab SPY QQQ IWM TLT GLD                 # every strategy, one report each
silo lab SPY QQQ IWM TLT GLD --strategy momentum --open-holdout   # final check, once
pytest
```

`silo fetch --source stooq` is a fallback when Yahoo rate-limits.

## Next phases

3. Contestants: TradingAgents, AI-hedge-fund personas and Kronos adapted to UK stocks, all on the same paper account, every call logged.
4. Own agent: score each part of each contestant, combine the best, Claude proposes and code disposes, run on paper.
Later: amnesia test, synthetic sandbox and strategy evolution experiments.
