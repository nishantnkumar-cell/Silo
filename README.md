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

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[data,dev]'

silo demo                                   # offline, synthetic data
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

2. Agent desk: analyst, debate and trader agents (local models plus Claude), amnesia test, cutoff tagging.
3. Experiments: strategy evolution, regime router, synthetic sandbox, Kronos forecasts.
4. Paper loop: daily paper runs, thesis journal, weekly review.
5. Build our own agent from the best-scoring parts of other frameworks.
