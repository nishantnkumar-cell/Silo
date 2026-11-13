# Core Strategy rulebook (supply and demand)

> Research only. These are rules from trading course material that the project owner studied,
> restated in our own words so Silo can test them. Silo never trades real money.

Sources: the *Core Strategy* course manual (5 days, 575 slides) and the *Professional Forex
Trader* course book (193 slides) from the same school. The forex book repeats the Core Strategy
and adds forex-specific material. Neither book gives any statistics on how often its rules work.

Code: `src/silo/strategies/core_zones.py` (rules strategies `core_zones`, `core_trail` and `core_managed`),
`src/silo/agents/rulebook.py` (AI contestant `core_desk`), tests in `tests/test_core_zones.py`.

## The idea

Prices turn where buyers and sellers were badly out of balance and orders were left unfilled.
On a chart this shows up as a short pause (a **base** of small, neutral candles) followed by a
strong move away (a **leg** of large candles). The base is a **zone**: **demand** if price left it
upwards, **supply** if downwards. Professionals buy at demand ("wholesale") and sell at supply
("retail"); novices do the opposite, buying after a rally into supply and selling after a fall
into demand.

## Rules, and how Silo codes them

| # | Rule (paraphrased) | In code |
|---|---|---|
| 1 | Candles are **legs** (big body, closes outside the prior candle's range) or **bases** (small, neutral body). A gap counts as part of the leg. | Base = body at most half the candle's range; gap from the prior close counts toward the body. |
| 2 | A zone is leg-in, 1 or more base candles, leg-out. The leg-out must make a new high (demand) or low (supply). Formations: drop-base-rally, rally-base-rally (demand); rally-base-drop, drop-base-drop (supply). | `find_zones` |
| 3 | Lines: **proximal** at the base's highest body (demand) or lowest body (supply), the manual's preferred setting; **distal** at the formation's extreme, including the leg-in for reversals (DBR, RBD) but not for continuations (RBR, DBD). | as stated |
| 4 | **Strength** (0-2): price left by at least 2x the zone's height, and/or broke past the previous opposing level. Both = 2, one = 1, neither = 0. | Move-out = consecutive closes in the leg's direction; "previous opposing level" = the last swing high (demand) or low (supply) before the base. |
| 5 | **Time** (0-1): 1-3 base candles = 1, 4-6 = 0.5, more = 0. | as stated |
| 6 | **Freshness** (0-2): untested = 2, tested up to half way into the zone = 1, deeper = 0. A close beyond the distal line **violates** the zone: drop it. | The current visit (the trade being planned) does not count as a test. |
| 7 | **Trend** (0-2), on a higher time frame: with the trend = 2, sideways = 1, against = 0. Uptrend = higher lows and higher highs; it starts when price breaks the last swing high after a higher low, the latest swing low becomes the *controlling* low on each new break, and a break below the controlling low ends it. Buy demand above the controlling low. | Weekly bars; swings need 2 bars either side; breaks judged on closes. |
| 8 | **Curve** (0-1), on the highest time frame: find the nearest quality demand below and supply above, split the range distal-to-distal into thirds. Demand in the low third = 1, middle = 0.5, top = 0. | Monthly bars, completed months only. With too little history the range is estimated from highs and lows and the setup is flagged. |
| 9 | **Profit zone** (0-2): distance to the next opposing zone over the risk. At least 5:1 = 2, at least 3:1 = 1, less = 0. | Next daily supply above; if none, the monthly supply; if none at all, scored 2 ("let profits run"). A supply zone that price has already wicked through (past its far line) is used up and doesn't count. |
| 10 | Total out of 10. 8.5+ = enter at the proximal line (limit order). 7-8 = wait for confirmation (price back out of the zone). Below 7 = no trade. | `min_score` 7 or 8.5 is chosen by walk-forward. |
| 11 | **Decision matrix** (curve x trend). Long only when: low on the curve (in a downtrend only with 5:1), middle with a sideways or up trend, high only in an uptrend with 5:1. Otherwise no action. | `decision_matrix` |
| 12 | Time frames for trades lasting weeks to months ("monthly income"): monthly for the curve, weekly for the trend, daily for zones. | as stated |
| 13 | **Stop** beyond the distal line by a buffer of 10% of the daily average true range (for weeks-to-months trades). | as stated |
| 14 | **Target** before the next opposing zone, at 75% of the profit zone, and at least 3x the risk. Or manage the trade with a **trailing stop** under each new demand zone and let profits run. | `core_zones` uses the target; `core_trail` trails. |
| 15 | **Size**: risk at most 2% of the account per trade and 6% across open trades; no more than 50% of the account in one position. | Also Silo's 10% cap per name. |
| 16 | Always have a stop; never add to a losing trade; don't overtrade. | Built in: one entry per stock, no averaging down. |

## Trade management (`core_managed`)

The project owner's rules for managing every trade, added after the first backtests:

| Rule | In code |
|---|---|
| Reward-to-risk at least 1:4 | Entry only when 75% of the way to the next supply zone is at least 4x the risk (no supply overhead also passes). |
| Check every open trade every day | `Strategy.manage` runs at every daily close, also between weekly decisions. |
| Stop hit: sell | A daily close at or below the stop sells the whole position at the next open, even inside the 20-day minimum hold. |
| At 1:2, sell half | The first day the high reaches entry + 2x risk, half is sold at the next open. The stop then moves up to the entry price (our addition, `breakeven=True`), so the rest cannot turn into a loss. |
| The rest trails | The stop is raised under each new demand zone that forms (the manual's trailing stop) until it is hit. |

In the engine, these sells are recorded as decisions with an `exits` field, so the paper loop and
the Trading 212 Practice mirror send them like any other order. Silo still has no resting stop
or limit orders: a stop is seen at the close and filled at the next open, so a gap down fills
below the stop. `Constraints(daily_exit_checks=False)` turns the daily check off for comparison.

## What Silo cannot do the way the manual says (and the effect)

* **Order types.** The manual buys with a limit order at the proximal line and exits with a
  stop-market order and a limit target, placed in advance. Silo decides at the close and fills at
  the next open, and by default only decides **weekly** and holds new positions at least **20
  trading days**. So a stop that is hit mid-week is acted on at the next weekly decision (losses
  can be bigger than planned), and an early stop is held until day 20. A second run with daily
  decisions and no minimum hold measures how much this costs.
* **Entry price.** Entries are planned from the last close (price must have touched the zone
  since the last decision and still be within half a zone-height above it), not from the
  proximal line itself, so reward-to-risk is a little worse than on paper.

## Rules that cannot be tested here

* **Short selling** (sell setups at supply zones, the downtrend half of the decision matrix):
  Silo is long-only by design. Supply zones are used only for targets.
* **News filters**: don't open a weeks-to-months trade with a high-impact report or earnings due
  within 5 days; tighten stops before reports. Silo has no point-in-time economic or earnings
  calendar.
* **Novice and professional gaps** beyond what the zone rules already pick up (a gap counts as a
  leg). The manual leaves the detail to an advanced course.
* **Refining time frame**, **pre-trade routine** (check accounts, news, yourself),
  **psychology**, **trade logging** and **reviewing results**: about the trader, not the market.
  Silo's journal and lab reports cover logging and review.
* **Discretion.** The manual relies on a trained eye ("very strong explosive rally", "quality
  zone"). The thresholds above are our reading; a different reading gives different results.

## From the forex book

Applies to UK shares as well:

* The same Core Strategy (rules 1-16).
* An extra odds enhancer: when a related market is at a zone at the same time (for currencies,
  gold or oil), the trade is more likely to work. The stock version would be "the market or the
  sector is also at demand". **Not yet coded**; a candidate for a follow-up test.
* Gap risk: risk less on trades held for weeks, don't enter just before a weekend, avoid
  instruments that gap often. Silo's liquidity rules already skip thinly traded names.
* "Ten laws" and "survival skills": always use a stop, don't let losers run, don't average
  down, don't overtrade, avoid very volatile markets, start small, follow the plan. The strategy
  follows all of these by construction; Silo's universe screen already caps volatility.

Forex only (not used): pips and lot sizes, leverage and margin, rollover interest and the carry
trade, trading sessions and time zones, the US dollar index, central-bank intervention, bid-only
charts and spread rules in pips, currency futures, broker selection, Bitcoin.

## AI contestant

`core_desk` is the existing debate desk (bull, bear, trader, risk manager) with this rulebook in
every prompt and six extra numbers per stock: zone score, whether price is in the zone now, how
far above the zone it is, weekly trend, position in the monthly range, and reward-to-risk. Its
shortlist is the 30 best zone scores, not the 30 strongest recent gainers. The original `desk`
is unchanged, so earlier results and cached answers stay comparable.
