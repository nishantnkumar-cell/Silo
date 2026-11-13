# The learning loop: learning from past trades without fooling ourselves

> Research only. Simulated results, not financial advice. Silo never trades real money.

## Why a loop needs a gate

Silo already has one agent that adapts: `blend` leans on whichever member has been doing well
lately. On 2012-2026 UK data it returned +2% against +379% for equal-weight buy-and-hold, because
it kept switching to whatever had just worked, paid about £48k in fees, and was usually late.
That is what learning without a check looks like: every change is drawn from the same recent
data that made it look good.

The loop below changes an agent only after the change has worked on data it was not drawn from,
and it counts every attempt so that retrying until something passes doesn't work.

## 1. The trade journal (`silo/learning/journal.py`)

Built from any finished run (backtest or paper session). A **trade** runs from the fill that
opens a position to the fill that closes it. For each trade:

| Field | Meaning |
|---|---|
| `opened`, `closed`, `hold_days` | dates and trading days held |
| `cost`, `proceeds`, `fees`, `dividends`, `ret` | the result after stamp duty, spread, commission and dividends |
| `market_ret`, `excess` | the equal-weight universe over the same days, and the trade minus the market |
| `features` | what the agent could see at the decision clock (point-in-time): the brief's numbers, market breadth, Core Strategy zone readings for `core_*` agents, and for AI desks whether the bull backed it or the bear warned about it |
| `why` | the AI trader's stated reason for that rebalance |

Across all trades (open ones marked at the last close), profit adds up exactly to the change in
the book's equity; a test checks this. `silo learn review` writes `trades.csv` for people and
`journal.jsonl` for tools.

## 2. Review (`silo/learning/review.py`)

* **Statistics.** For each feature, the journal's quintiles are tried as cut points. A side whose
  trades lagged the rest of the trades against the market by a Welch t-statistic beyond -2 becomes
  a candidate "avoid" lesson (the strongest cut per feature, three at most). The number of splits
  tried is reported, because trying many and keeping the best is how overfitting happens.
* **AI reviewer** (`--ai`). The model is shown the overall record, results by feature quintile,
  a sample of trades in shuffled order, and the statistical candidates. It never sees a ticker or
  a date (free text is scrubbed of anything that looks like either). It must answer with at most
  three lessons in Silo's format; anything else is dropped:

  ```json
  {"feature": "ret_12m", "op": ">", "value": 60, "action": "avoid", "text": "ran too far already"}
  ```

  `action` is `avoid` (don't open a position when the condition holds), `only` (open new
  positions only when it holds) or `advice` (no condition; text for AI agents' prompts only).
  Features are limited to the list in `silo/learning/lessons.py`, so a lesson can't name a stock
  or a year.

Proposals go to the ledger with the window of trades they came from. Neither reviewer can put a
lesson live.

## 3. The gate (`silo/learning/gate.py`)

`silo learn gate` tests lessons on a window that starts after the last trade they were drawn
from and ends before the locked holdout (the same 20% that `silo lab` locks). It refuses any
other window. To be accepted, all of these must hold:

1. **It acts:** the lesson skips at least 10 new entries in the test window.
2. **It helps:** the agent with the lesson has a higher Sharpe ratio than without, after costs.
3. **It isn't luck:** it beats 95% of random filters that skip the same share of entries.
   Skipping trades alone changes exposure and fees; a coin flip that skips as much is the bar.
4. **It survives retries:** the daily difference in returns passes the deflated Sharpe test
   (0.95), counting every earlier gate run for this agent whose window overlaps.
5. **AI agents:** the test window starts after the model's training cutoff.

Buy-and-hold is reported next to every result. The ledger (`learning/ledger.jsonl`) is
append-only: proposals and gate runs are added, never rewritten, like the holdout log.

## 4. Using accepted lessons

`learned` is a registry agent that wraps any other one:

```bash
silo paper new learned-core --strategy learned --symbols-file uk.json \
    --params "$(silo learn params --strategy core_zones)"
```

* Lessons only stop **new** entries. A name already held is never sold because of a lesson, so
  following lessons doesn't add churn.
* When a lesson can't be evaluated (not enough history), the agent's own decision stands.
* For AI agents the lessons are also added to every prompt. With no lessons the prompts are
  byte-for-byte unchanged, so cached answers from earlier runs stay valid.
* A paper book fixes its lessons when it is created; later ledger changes don't alter it.

## 5. Does learning help at all? (`silo learn walkforward`)

The gate checks one lesson. `silo learn walkforward` checks the loop: for each fold it runs the
agent on the training window, builds the journal, reviews it, then trades the next window with
those lessons, and stitches the test windows together. The result is compared with the same
agent never learning and with buy-and-hold, with a deflated Sharpe test on the difference. If
learning doesn't beat not learning here, the loop is not helping, however good individual
lessons look.

## Core Strategy rulebook

For `core_zones`, `core_trail` and `core_desk`, lessons can use the six zone readings (zone
score, in zone, % above zone, weekly trend, position in the monthly range, reward to risk), so a
lesson can amend the rulebook in its own terms, e.g. "only buy when the weekly trend is up".
Accepted lessons are not written into `docs/core-strategy-rulebook.md` automatically; `silo learn
list` shows them in plain English for a person to add as amendments.

## Limits

* One history is one sample. A lesson that passes is evidence, not proof; paper trading is the
  next check.
* Survivorship bias in the stock list affects lessons too.
* Lessons are single conditions on numbers. Anything subtler stays as `advice`, which only AI
  agents follow and which the gate can test only by rerunning the AI agent (slow on the PN50).
