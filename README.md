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
| Position trading | `src/silo/constraints.py` | Default: decide weekly, hold new positions at least 20 trading days, trade at most 50% of the book per rebalance, cap any name at 10%. Enforced in code. |
| Liquidity | `src/silo/liquidity.py` | Hard rules in the engine for every strategy: no penny stocks (under £1), at least £5m traded a day (60-day median), no position bigger than 20% of a day's trading so it can be sold within two days, and each simulated fill limited to 10% of that day's volume. A name that stops passing can be held or trimmed, never added to. |
| Universe | `src/silo/universe/uk.py` | About 140 liquid FTSE 100 and FTSE 250 candidates. Price screen (point-in-time): at least £1 a share, £5m a day traded, 20% to 45% volatility, no 50% crash. Optional quality screen (ROE, margins, debt) from today's fundamentals, for picking the live universe only. |
| Strategy | `uk_trend` in `strategies/classic.py` | Names above their 200-day average, ranked by risk-adjusted 6-month momentum, top 8, inverse-volatility sized. |
| Market conditions | `src/silo/lab/regimes.py` | The lab report splits out-of-sample results into rising, sideways and falling markets, calm and stormy volatility, and the 2020 crash, 2020 rebound and 2022 rate shock, each next to buy-and-hold. |
| Paper loop | `src/silo/paper.py` | Forward-only book on the simulator with a buy-and-hold shadow book, append-only journal, safe to rerun. Uses the same engine as backtests, so paper and backtest agree to the penny. |

Reports now show round trips and average holding period.

**Trading 212 Practice mirror** (`src/silo/brokers/trading212.py`): the
simulator stays the source of truth; the newest rebalance can also be sent as
market orders to a Trading 212 *Practice* account (`demo.trading212.com`).
`live.trading212.com` is on the refused list, `T212_LIVE*` variables stop
Silo from starting, every order is logged as an intent before it is sent so a
rerun never duplicates it, and `t212-check` compares the two books.

## Dashboard

`silo web` opens a local dashboard at <http://127.0.0.1:8212/> (`src/silo/web.py`, standard library only):

* **Overview**: every paper book with its return next to the buy-and-hold copy, the latest lab reports.
* **Book page**: equity chart against buy-and-hold, positions, trade journal, rebalance decisions,
  Trading 212 mirror log, and a read-only "Compare with the Practice account" button.
* **Lab reports**: the HTML reports written by `silo lab`.
* **Settings**: add or remove a Trading 212 **Practice** API key. Silo checks the key with
  `demo.trading212.com` first and only saves keys that server accepts, so a real-money key is never
  stored. Keys go to `~/.config/silo/credentials.json` (owner-only, outside the repo); environment
  variables still take precedence.

The server listens on 127.0.0.1 only (other addresses are refused), answers only `localhost` Host
headers, needs a per-run token on every form, and never changes books or places orders.

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

silo web                                    # dashboard on http://127.0.0.1:8212/

# Optional: mirror the paper book to a Trading 212 Practice account (paper money)
# add the Practice key on the dashboard's Settings page, or:
export T212_DEMO_API_KEY=... T212_DEMO_API_SECRET=...   # key created in Practice mode
silo paper t212-map uk-trend                # propose Yahoo -> Trading 212 tickers; review the file
silo paper run uk-trend --fetch --mirror t212
silo paper t212-check uk-trend              # simulator vs Practice account weights

silo fetch SPY QQQ IWM TLT GLD              # free daily bars from Yahoo
silo backtest SPY QQQ IWM TLT GLD           # every baseline vs buy-and-hold
silo backtest SPY --strategy sma_cross --start 2015-01-01
silo lab --demo --strategy bollinger        # offline lab study -> reports/bollinger.html
silo lab SPY QQQ IWM TLT GLD                 # every strategy, one report each
silo lab SPY QQQ IWM TLT GLD --strategy momentum --open-holdout   # final check, once
pytest
```

`silo fetch --source stooq` is a fallback when Yahoo rate-limits.

## Phase 3 and 4: AI contestants, Silo's own agent, and the league

All AI strategies run on a **free local model** through [Ollama](https://ollama.com), so nothing
leaves the computer and bulk backtests cost nothing. Choose the model with `SILO_LLM`.

| Strategy | File | What it is |
|---|---|---|
| `desk` | `src/silo/agents/contestants.py` | After TradingAgents: a bull and a bear researcher argue, a trader picks up to 12 names, a risk manager can veto. Four model calls a week. |
| `panel` | `src/silo/agents/contestants.py` | After ai-hedge-fund: trend, contrarian, income and defensive investors score every stock; a portfolio manager averages the scores. Four calls a week. |
| `blend` | `src/silo/agents/blend.py` | Silo's own agent, rules only: runs `uk_trend`, `momentum` and `donchian` side by side, scores each on what its last picks earned (recent weeks count most), follows the ones that are working, and halves exposure when nothing is working or most of the market is below its 200-day average. |
| `silo_agent` | `src/silo/agents/blend.py` | The same blend with `desk` and `panel` on board. |

How the AI is kept honest:

* The model sees only a table of numbers from the time firewall (returns, volatility, distance from
  the 200-day average, drawdown, dividend yield, value traded). **Tickers are replaced with letters and
  dates are hidden**, so it can't recognise a company and remember what happened next.
* Every call is logged (`<book>/llm/calls.jsonl` for paper books) with the model and its **training
  cutoff**. The lab adds a check that fails when the test period starts before that cutoff.
* Replies are cached by prompt, so reruns replay the same answers and cost nothing.
* A reply that isn't valid JSON, or names stocks that aren't in the table, means no trade.
* AI strategies go through the same liquidity and position-trading rules as everything else.
* `SILO_LLM=rulebot` is an offline stand-in used by tests. It is a fixed rule, not an AI.

### Running the league on the PN50 (Windows)

```powershell
# once: install Ollama from https://ollama.com, then
ollama pull qwen2.5:7b
$env:SILO_LLM = "ollama:qwen2.5:7b"

silo fetch --universe uk
silo universe uk --save uk.json
silo league new --with-ai                 # one paper book per contestant, same start, same cash
silo daily                                # fetch, advance every book, print the leaderboard
silo daily --mirror league-silo_agent     # also mirror one book to the Trading 212 Practice account
silo web                                  # the dashboard lists every league book

# run it automatically on weekdays at 18:30
powershell -ExecutionPolicy Bypass -File scripts\windows\install-daily-task.ps1 -Model ollama:qwen2.5:7b -Mirror league-silo_agent
```

Without `--with-ai` the league has only the rule-based books (`uk_trend`, `momentum`, `blend` and a
random control), which need no model. Before mirroring, run `silo paper t212-map league-silo_agent`
and check the ticker file.

## Next phases

* Kronos (a price-forecasting model) as a contestant, once it runs on the PN50's CPU.
* News and filings (RNS) through the firewall, so the desk can get its missing analysts.
* Strategy evolution: let the model propose new rules; the lab disposes.
