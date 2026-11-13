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

**Practice fills against the simulator** (`src/silo/fills.py`): Trading 212's
API has no price history, but it does report how each Practice order filled.
`silo paper t212-fills BOOK` reads the account's order history (read-only),
appends each mirrored order's fill to `mirror.jsonl` once, pairs it with
SimBroker's fill for the same rebalance, and reports the gap: price slippage
in basis points (positive = Practice did worse), each venue's distance from
the day's open, fees as a share of traded value (stamp duty and any FX fee),
and hours from order to fill. `paper run --mirror t212` and
`daily --mirror BOOK` collect fills on every run, and the book's dashboard page
shows the table. After 10 paired fills it prints the slippage setting that
would match, but never changes the CostModel itself. The Practice API key
needs the `history:orders` permission. Practice fills are Trading 212's own
simulation, so they are a calibration check, not real executions.

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
silo universe uk --quality --save uk.json   # liquid, in an uptrend, ranked by momentum, good fundamentals
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
silo paper t212-fills uk-trend              # Practice fill prices and fees vs the simulator's

silo fetch SPY QQQ IWM TLT GLD              # free daily bars from Yahoo
silo backtest SPY QQQ IWM TLT GLD           # every baseline vs buy-and-hold
silo backtest SPY --strategy sma_cross --start 2015-01-01
silo lab --demo --strategy bollinger        # offline lab study -> reports/bollinger.html
silo lab SPY QQQ IWM TLT GLD                 # every strategy, one report each
silo lab SPY QQQ IWM TLT GLD --strategy momentum --open-holdout   # final check, once
pytest
```

`silo fetch --source stooq` is a fallback when Yahoo rate-limits.

## The wider pool: every UK and US share on Trading 212

`silo pool instruments` reads Trading 212 Practice's instrument list (names
and codes only, no prices), keeps ordinary shares listed in London or New
York, drops repeats and US ADRs of London companies, saves a dated snapshot
and writes `pool/uk-candidates.json` and `pool/us-candidates.json`. Prices
then come from Yahoo, and `silo universe` gates and ranks them:

```
silo pool instruments
silo fetch --symbols-file pool/us-candidates.json --pause 1   # slow, run once on the PN50
silo universe us --candidates pool/us-candidates.json --top 60 --sector-cap --save us.json
silo league new --symbols-file us.json --prefix us              # a separate US book
```

Gate: ordinary shares, at least £1 / $5, a median £5m / $20m a day traded,
a year of history, volatility under 60% a year. Trend: above the 200-day
average and within 25% of the 52-week high. Rank: 12-month return skipping
the latest month, calmer names first on ties. `--sector-cap` keeps any one
sector to a quarter of the picks (today's sectors, so live use only). US books
pay the 0.15% FX fee each way and have 15% US tax withheld from dividends.
Today's Trading 212 list only holds survivors, so backtests on it are
flattered; forward paper trading is the honest test. Research only.

## Daily book: silver ETFs and crypto

A separate book for **daily-bar** trading of a couple of silver ETFs and the two largest coins. It
decides at each daily close and fills at the next open, so it is daily, never intraday. It never mixes
with the UK position-trading books, which keep their weekly rules.

| Part | File | What it does |
|---|---|---|
| Universe | `src/silo/universe/daily.py` | Silver: `SLV`, `SIVR`. Crypto: `BTC-USD`, `ETH-USD`. Only very liquid names. |
| Crypto market | `src/silo/markets.py` | Trades every day; a UTC day's bar is final at 00:05 UTC the next day. Yahoo's dollar volume is stored as coins. Sharpe and CAGR use 365 days a year for crypto. |
| Partial bars | `data/sources.py`, `paper.py` | A bar from a session still trading (US hours, today's crypto day) is never stored, and a paper book only steps a day once every bar in it is final. |
| Costs | `src/silo/broker.py` | `CostModel.crypto()`: 0.25% exchange fee each way, 10bp spread, 10bp slippage (about 0.8% a round trip). `CostModel.us_from_uk()`: US ETF costs plus a 0.15% FX fee each way. `--cost-scale 2` doubles every cost to check a result survives. |
| Rules | `src/silo/constraints.py` | `Constraints.daily_trading()`: decide daily, hold at least one day, at most 50% in one name, liquidity on. Crypto needs $1bn a day of reported volume and each fill may take only 1% of it; US ETFs need $5 a share and $20m a day. |
| Strategy | `tsmom` in `strategies/classic.py` | Time-series momentum with volatility targeting; trades only when a signal flips or a size drifts 25%. Studied alongside `donchian`, `rsi2`, `bollinger` and `sma_cross`. |

```bash
silo fetch --universe silver
silo fetch --universe crypto
silo lab --universe silver --style daily --out reports/silver          # every daily strategy
silo lab --universe crypto --style daily --out reports/crypto
silo lab --universe crypto --style daily --cost-scale 2 --out reports/crypto-2x   # double costs
silo league new --universe crypto --style daily --prefix crypto        # forward paper books
silo league new --universe silver --style daily --prefix silver
silo daily                                                             # advances every book
```

Notes: UK retail investors generally cannot buy US-domiciled ETFs such as SLV (no UK KID); the
London-listed physical silver ETCs follow the same metal, and SLV's longer history stands in for them.
Trading 212 Invest has no spot crypto, so the crypto book lives on the simulator only.

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

silo fetch --universe uk --macro
silo universe uk --save uk.json
silo league new --with-ai                 # one paper book per contestant, same start, same cash
silo daily                                # fetch, advance every book, print the leaderboard
silo daily --mirror league-silo_agent     # also mirror one book to the Trading 212 Practice account
silo web                                  # the dashboard lists every league book

# run it automatically on weekdays at 18:30
powershell -ExecutionPolicy Bypass -File scripts\windows\install-daily-task.ps1 -Model ollama:qwen2.5:7b -Mirror league-silo_agent
```

### Backtesting the whole league

The paper league needs months to say anything, so replay history too:

```powershell
silo check-data                                         # flag stale or flat price series first
silo league backtest --all-candidates --start 2012-01-01 --out reports\league-rules
silo league backtest --all-candidates --with-ai --start 2024-10-01 --out reports\league-ai
```

Each run writes `league.html` (equity of every contestant, results table, results by market
conditions) and `league.json`. AI contestants are labelled *after cutoff* or *before cutoff*: only
runs that start after the model's training cutoff (qwen2.5: September 2024) are a fair test.
`--all-candidates` uses every UK candidate and lets the engine's liquidity rules pick tradable names
day by day, instead of today's screened list (which would leak today's knowledge into the past).
Model answers are cached in the output folder, so `silo_agent` reuses the desk's and panel's answers
and an interrupted run resumes where it stopped.

Without `--with-ai` the league has only the rule-based books (`uk_trend`, `momentum`, `blend`,
`sector_mom`, `macro_tilt`, `news_mom`, `results_drift`, `director_buys` and a random control), which need no model. Before mirroring, run `silo paper t212-map league-silo_agent`
and check the ticker file.

## Cross-market agent

Research behind it: [docs/news-and-cross-asset-research.md](docs/news-and-cross-asset-research.md).
In short, sector momentum is well evidenced; the pound, oil, metals, gilt yields and US shares mostly
explain risk and regime rather than forecast single stocks. So the agent adds numbers, not opinions:

* **Data**: `silo fetch --macro` stores pound/dollar, Brent, gold, silver, copper, the S&P 500 and VIX
  (Yahoo) and the 10-year gilt yield (Bank of England) as context tables (`ctx:macro:*`) with
  `available_at`. Yahoo series are stamped 23:00 UTC, so a decision at the London close only sees
  them up to the day before. Gilt yields count from noon the next working day. `silo daily`
  refreshes them too (`--no-macro` to skip).
* **Brief columns** (`src/silo/agents/crossasset.py`): each stock's 6-month return against its own
  sector, its sector against the market, and the 3-month move its macro sensitivities imply. Each
  sensitivity comes from a weekly regression on the UK market, pound, oil, copper and gilt yield.
  The market line gives each driver's 3-month move as a percentile of its own history, never a
  level, and counts the risk-on signals. Sector names never reach the model.
* **Contestants** (`src/silo/agents/informed.py`): `panel_macro` is the panel plus a cross-market
  voice. `desk_macro` is the desk with a cross-market analyst briefing both researchers, and its
  risk manager sees the picks' combined sensitivity to each market. The original `panel` and `desk`
  are unchanged, so their caches stay valid and the lab can compare them like for like.
* **Rule baselines** (`src/silo/strategies/crossasset.py`): `sector_mom` (best names in the best
  sectors) and `macro_tilt` (momentum, invested only as far as the risk-on signals allow). If these
  can't beat the random control, the same numbers are unlikely to help the AI.

Sector labels (`src/silo/universe/sectors.py`) are today's, applied to the past: a mild look-ahead,
flagged rather than hidden.

## News agent

Company announcements (RNS) become numbers in the brief. The model never sees announcement text,
headlines or dates.

* **Data**: `silo news fetch` reads each company's announcement list from Investegate, for personal
  research use only. It sends one request every 3 seconds, stops at announcements it already has,
  and stores each company as its own snapshot, so you can stop it and rerun it. The first backfill
  since 2012 for the whole UK list takes many hours, so run it overnight. Text is fetched only for
  results, trading updates, profit warnings and director dealings. It is kept locally in
  `data/news_bodies/` and written once. Publication times are converted from London time to UTC as
  `available_at`, so a 07:00 RNS counts at that day's close and a 16:40 one the next day.
  `silo daily --news` picks up new announcements (a few minutes).
* **Plain rules first** (`src/silo/data/news.py`):
  * the headline gives the category (results, trading update, profit warning, director dealing,
    buyback, M&A, capital raise, board, routine);
  * the "ahead of / in line with / below expectations" wording gives a surprise sign;
  * director dealings are read as buys or sells, ignoring awards, options and share plans.
* **Brief columns** (`src/silo/agents/news.py`):
  * trading days since results;
  * the results-day move against the market (3 days);
  * the last expectations sign;
  * any profit warning in the last 90 days;
  * director buys in the last 90 days;
  * the 6-month return split into news days and quiet days.

  With no announcements stored, heavy-volume days stand in for news days.
* **Optional model reading** (`src/silo/agents/news_extract.py`): `silo news extract` has the local
  model read results text that the rules couldn't place. The text is masked first: company, ticker,
  years, months, dates and amounts are replaced. Answers are stored as numbers in `ctx:news_llm`.
  `silo news probe` asks the model to guess the company and year from masked text; a high hit rate
  means the masking leaks.
* **Contestants**: `panel_news` / `desk_news` (news only) and `panel_full` / `desk_full` (news and
  cross-market). With `panel`, `desk` and the `_macro` versions, the lab can compare each addition.
* **Rule baselines** (`src/silo/strategies/news.py`):
  * `news_mom`: momentum counted on news days only;
  * `results_drift`: holds after strong results or an "ahead" update;
  * `director_buys`: holds names where directors bought, unless they have warned on profits.

Judge any AI result only after the model's training cutoff. For news, also check the probe.

## Next phases

* Kronos (a price-forecasting model) as a contestant, once it runs on the PN50's CPU.
* Strategy evolution: let the model propose new rules; the lab disposes.

## Core Strategy rules (supply and demand)

The project owner studied a supply-and-demand trading course. Its rules are written up in
[`docs/core-strategy-rulebook.md`](docs/core-strategy-rulebook.md) (what each rule says, how it is
coded, and which rules can't be tested) and implemented three ways:

| Contestant | What it is |
|---|---|
| `core_zones` | Rules only. Weekly: score every active daily demand zone out of 10 (strength, time, freshness, weekly trend, monthly curve, profit zone), apply the curve x trend decision table, buy when price is back at a zone that scores 7+ (8.5+ buys at the zone, 7-8 waits for confirmation). Stop 10% of a day's range below the zone, target 75% of the way to the next supply and at least 3x the risk; 2% risk per trade, 6% in total. |
| `core_trail` | The same entries, no fixed target: the stop trails under each new demand zone. |
| `core_desk` | The AI debate desk with the rulebook in its prompts and six zone readings per stock. |

```bash
silo league backtest --all-candidates --start 2012-01-01 --only core_zones core_trail momentum random --out reports/league-core
silo lab --all-candidates --strategy core_zones --out reports/lab-core
```

## Learning from past trades

Silo's agents can learn from their own trades, but nothing they learn goes live until it has
held up on data it wasn't drawn from. Details in [`docs/learning-loop.md`](docs/learning-loop.md).

1. **Journal.** Every position from open to close: the numbers the agent saw when it bought, its
   reasons (for AI desks: which names the bull backed and the bear warned about, the trader's
   "why"), and the result after costs and dividends next to the market over the same weeks.
2. **Review.** A statistical miner looks for entry conditions that kept losing to the market; with
   `--ai` a model reads an anonymised summary (no names, no dates) and proposes at most three
   lessons in a fixed rule format such as "don't buy when the 12-month return is above 60%".
3. **Gate.** A lesson is accepted only if, on a later window and never the locked holdout, the
   agent with it beats the same agent without it, beats 95% of random filters that skip as many
   trades, and passes a deflated Sharpe test that counts every retry. Every proposal and verdict is
   appended to `learning/ledger.jsonl`.
4. **Use.** The `learned` agent wraps any contestant with its accepted lessons: they stop new
   entries (never force a sale, so no extra churn), and AI agents also get them in their prompts.

```bash
silo learn review --all-candidates --strategy core_zones --end 2019-12-31   # journal + proposals
silo learn gate --all-candidates --strategy core_zones <lesson ids>        # test on 2020 onwards
silo learn walkforward --all-candidates --strategy core_zones              # does learning help at all?
silo learn list
silo paper new learned-core --strategy learned --symbols-file uk.json \
    --params "$(silo learn params --strategy core_zones)"
```
