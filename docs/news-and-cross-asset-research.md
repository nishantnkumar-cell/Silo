# News and cross-market agents for Silo

13 Nov 2026 · Research only, not investment advice · Evidence grades: **Strong** (replicated, many markets), **Mixed** (real once, weaker or contested now), **Speculative** (plausible, little or no evidence at our horizon).

## 1. The short answer

- **News helps most as a way to read price moves, not as a mood score.** The best-supported idea for a weeks-to-months book is that price moves *caused by real news* (results, trading updates) keep going, while moves *without news* tend to reverse. Headline sentiment scores mostly pay off within a day or a week, which is too fast for Silo. (Strong for the news/no-news split, Mixed for sentiment at our horizon.)
- **Earnings-expectation signals are real but fading, and the classic data isn't free.** Post-earnings drift was one of the strongest anomalies for decades, but in large and mid caps it has largely disappeared since about 2006 in US data. FTSE 350 is large/mid cap, so expect a small effect. Consensus estimates and revisions have no free point-in-time UK history. The free substitute is the *share price reaction on results day*, plus the "ahead of / in line with / below market expectations" wording that UK trading statements routinely use. (Mixed.)
- **UK director share purchases are the most promising UK-specific news event.** Studies of UK director dealings find abnormal returns after purchases over weeks to months, and every deal is filed as a dated RNS. (Strong historically; needs re-testing on recent FTSE 350 data.)
- **Cross-market links are better for risk and regime than for picking stocks.** Industry momentum and cross-asset trend (time-series momentum) are well evidenced. Oil-price moves and lagged US returns have some predictive power for the overall market, but the effects are small and unstable. Currency, oil and rate sensitivities mainly explain *why* a stock moved; they are a weak forecast of where it goes next. (Strong for momentum-style links, Mixed for macro timing, Speculative for stock selection by macro beta.)
- **The biggest risk is leakage, not lack of signal.** Dated news text breaks Silo's anonymisation, and LLMs measurably "remember" outcomes for events before their training cutoff. Any text must be turned into masked, structured numbers *before* the deciding model sees it, and only post-cutoff results count.

**Recommendation.** Build two agents that produce *numbers*, not prose, for the existing brief: a **News agent** (event features from RNS: results-day reaction, expectations wording, director buying, profit warnings, news vs no-news momentum) and a **Cross-asset agent** (sector-relative momentum, a few macro drivers' trends and each stock's sensitivity to them). Test each signal first as a plain rule against random and buy-and-hold. Only signals that survive go into the panel and desk. Given that only momentum has beaten random so far, the realistic hope is that the news split *improves momentum* (fewer reversals), not a new standalone edge.

## 2. News, sentiment and earnings expectations

| Idea | What the research says | Horizon | Grade for Silo |
|---|---|---|---|
| News vs no-news moves | Stocks that move on public news keep drifting; big moves without news reverse (Chan 2003; Jiang, Li & Wang 2021 find continuation on news days across many event types). | 1–12 months | **Strong.** Fits our horizon and improves momentum, our only working signal. |
| Post-earnings drift (PEAD) | Classic and replicated, including UK (Liu, Strong & Xu 2003). Martineau (2022) finds it gone outside microcaps since ~2006 in the US; Subrahmanyam (2025) finds it survives only with microcaps (t 2.18 all stocks, 1.43 ex-microcaps, 2001–2024). | 1–3 months | **Mixed.** FTSE 350 has no microcaps, so expect little. Still worth one cheap test. |
| Results-day return as surprise proxy | The abnormal return over the 3 days around results captures the surprise without needing analyst forecasts, and has historically predicted drift (Brandt et al. 2008; Chan, Jegadeesh & Lakonishok 1996 for earnings momentum). | 1–6 months | **Mixed.** Free to compute once results dates are known. |
| Analyst revisions | Strong historically, but needs IBES/FactSet-style point-in-time data. No free UK source. | 1–6 months | **Not testable cheaply.** Skip. |
| "Versus market expectations" wording in UK trading updates | UK companies state "ahead of / in line with / below market expectations" in trading statements. This is effectively a free, company-written surprise sign. Profit warnings in UK studies show continued negative drift. | weeks–months | **Speculative/Mixed.** Little direct academic work on the wording itself; easy for a small LLM to extract. |
| UK director purchases | Director buys in UK firms are followed by abnormal returns (Fidrmuc, Goergen & Renneboog 2006, and later UK studies); sells carry much less information. | 1–6 months | **Strong historically.** Re-test needed; may be weaker in FTSE 350 than small caps. |
| Buyback announcements | Positive drift in US (Ikenberry et al. 1995); UK evidence weaker and buybacks are routine for large UK firms. | months | **Mixed.** Low priority. |
| LLM headline sentiment | GPT-4 scores on post-cutoff headlines predict next-day returns and some drift, strongest in small stocks and bad news; returns shrink as LLM use spreads (Lopez-Lira & Tang, v6 2025). Chen, Kelly & Xiu find LLM news embeddings predict returns across 16 markets, mostly at short horizons. | days–1 week | **Mixed, and mostly too fast** for weekly decisions with a 20-day minimum hold. |
| Market-wide media pessimism | Predicts short-term index dips that then reverse (Tetlock 2007). | days | **Too fast.** Possibly a regime input only. |

What this means for Silo: the edge, if any, sits in *events* (results, trading updates, director deals, warnings) and in *classifying* price moves, not in daily tone scores. That is good news for a modest PC because UK events are a few per company per quarter, not thousands of headlines a day.

## 3. Cross-market links

| Link | What the research says | Grade for Silo |
|---|---|---|
| Industry/sector momentum | Industries that did well keep doing well for months (Moskowitz & Grinblatt 1999). A stock lagging its strong sector tends to catch up; within an industry, big firms lead small ones (Hou 2007). | **Strong.** Directly usable: sector-relative and peer-lag columns. |
| Cross-asset trend (time-series momentum) | 12-month trend persists across equities, bonds, currencies and commodities in 58 markets (Moskowitz, Ooi & Pedersen 2012). | **Strong**, but it's a market-timing tool, so it fits the regime line and cash level, not stock picks. |
| Industries leading the market | Returns of metals, petroleum and some other industries lead the broad market by up to two months (Hong, Torous & Valkanov 2007). | **Mixed.** Effect is small and later work is less supportive. |
| Oil predicting equities | Oil price rises predict lower stock returns a month later in many countries including the UK (Driesprong, Jacobsen & Maat 2008). | **Mixed.** Small, has weakened; regime input only. |
| US leading other markets | Lagged US returns predict other countries' monthly returns (Rapach, Strauss & Zhou 2013). | **Mixed.** Small; regime input only. |
| GBP and the FTSE 100 | About three-quarters of FTSE 100 revenue is earned abroad, so a weaker pound lifts it in sterling; the FTSE 250 is more domestic. This is a well-known *sensitivity*, not a forecast. | **Strong as risk, Speculative as signal.** Use to avoid a book that is one big currency bet. |
| Rates and macro ratios | Most market-timing predictors (yield spreads, valuations) fail out of sample (Goyal & Welch 2008). | **Weak.** Don't expect timing alpha. |
| Customer/supplier links | Customer news predicts supplier returns about a month later (Cohen & Frazzini 2008); recent work propagates text signals along supply chains. | **Strong in US, but no free UK supply-chain data.** Skip for now. |
| Commodity → producer (miners, oil majors, silver/crypto book) | Producers move with their commodity; whether the commodity *leads* the share price at weekly-to-monthly frequency is thin. | **Speculative.** Cheap to test with the existing silver/crypto data. |

What this means for Silo: the cross-asset agent should (a) add sector-relative momentum and peer-lag columns, which have real evidence, and (b) act as a risk and regime adviser, flagging when the book is heavily exposed to one driver (pound, oil, rates) and when cross-asset trends are turning. It should not be expected to make money by forecasting macro.

## 4. Leakage: the main danger

**Look-ahead in data.** News must be stamped with when it was *public*, not the date it refers to. RNS items go out at 07:00 UK time before the open, so a 07:00 announcement can be traded at that day's open only if Silo's engine fills at the open after the clock; otherwise it waits a day. Revised data (macro releases, restated results) must use first-release values. Yahoo's "earnings estimates" and current news feeds are *today's* snapshot, not history, and must never be used for backtests.

**LLM memory.** Two recent papers measure this directly. Gao, Jiang & Yan (2025/26) show an LLM's forecasts are more accurate on firm-dates it can recall, and that this "lookahead propensity" collapses to about zero after the training cutoff. Glasserman & Lin (2023) find that removing company names from headlines actually *improved* in-sample results, because the model's general knowledge of famous firms distracts it, especially for large companies like FTSE 100 names. Somoza et al. (2025) find the memory problem is smaller for small models and fine-grained data, which mildly favours our 7B local model. He et al. (2025) train date-consistent models and find look-ahead "modest" but model-specific.

**Dated text breaks anonymisation.** A sentence like "trading in line with expectations despite Red Sea disruption" dates and often identifies the company. Silo currently hides tickers and dates; raw news text would undo both.

**Mitigations (proposed rules):**
1. *Text never reaches the deciding model.* An extraction step turns each announcement into a few fixed fields (event type, direction, vs-expectations, size bucket). The panel and desk only see numbers in the brief.
2. *Mask before extraction.* Replace company, people, places, products, years and absolute amounts with placeholders before the extraction model reads the text. Keep percentages and the expectations phrase.
3. *Rule extraction where possible.* Event type comes from the RNS headline category with plain rules, so most features never touch an LLM at all.
4. *Amnesia probe.* For a sample of masked items, ask the model to guess the company or year. If it can do better than chance, tighten masking. Add a lookahead-propensity check (ask the model the date-only recall question) and report results split by it.
5. *Only post-cutoff results count* for any LLM-fed signal, as now. For qwen2.5 (released Sept 2024) that gives roughly two years, about 1,400 FTSE 350 results announcements plus director deals: enough for an event study, thin for a portfolio verdict.
6. *Macro context is shown as ranks or z-scores, not levels.* "Oil −60% in 3 months" dates itself (2020); "oil 3-month move: bottom 2% of its history" leaks far less.
7. Keep CLAUDE.md rules: every new table carries `available_at` UTC and is read through `PointInTimeView`; no holdout opening while tuning.

## 5. Free or cheap UK data with point-in-time history

| Data | Source | History and caveats |
|---|---|---|
| RNS announcements (headline, category, time) | Investegate archive; LSE company news pages | Covers RNS plus other wires for FTSE 350/AIM. Archive start date not stated. **Terms allow personal reference only and bar reproduction or unauthorised access**, so bulk scraping is a grey area; ask Investegate or keep it to slow, personal, local use. Needs checking before we build on it. |
| Regulated filings archive | FCA National Storage Mechanism (NSM) | Official free archive of regulated announcements with publication timestamps. Coverage depth and bulk-access terms to verify. Best candidate for a clean source. |
| Global news events and tone | GDELT 2.0 (from 2015, 15-minute updates) | Free, timestamped, huge. Matches on company names, so noisy for UK mid caps, and it's tone-based (the weaker signal). Low priority. |
| Results dates | Derived from RNS categories ("Final Results", "Half-year Report", "Trading Statement") | Point-in-time by construction. Fallback with no news at all: flag days with abnormal volume and return as likely news days. |
| Director dealings | RNS "Director/PDMR Shareholding" notices | Dated; needs parsing buy/sell and value. |
| FX, commodities, US indices, VIX | Yahoo (`GBPUSD=X`, `BZ=F`, `GC=F`, `SI=F`, `HG=F`, `^GSPC`, `^VIX`) on the PN50 | Market prices aren't revised, so point-in-time is easy. Yahoo is blocked from the cloud container; fetch on the PN50 as now. |
| UK rates, sterling index | Bank of England database (Bank Rate, gilt yields, sterling ERI) | Free, long history. |
| US rates, oil, macro | FRED / ALFRED (ALFRED keeps first-release vintages) | Free. Use ALFRED vintages for anything revised. |
| Sector labels | ICB sector per FTSE 350 member | Need a point-in-time source; current labels applied to the past are a mild look-ahead. Acceptable for a first pass if flagged. |
| Consensus estimates, revisions | IBES, FactSet, Refinitiv | Not free. Not proposed. |

## 6. Proposed design

Both agents are **feature builders that write new columns into the existing brief**, plus one new voice in each AI contestant. Nothing changes in how the desk or panel decides until a signal passes its rule test.

**News agent (`silo.agents.news`)**
- Ingest: `news` vault table with `symbol, available_at (UTC), category, headline, body_hash` from a permitted source. Snapshots are append-only, per CLAUDE.md.
- Rules first: classify by RNS category into results, trading update, profit warning, director buy/sell, buyback, M&A, capital raise, board change.
- Small LLM step (masked text, local Ollama): for results and trading updates only, extract `vs_expectations ∈ {above, in line, below, none}` and `tone ∈ {−2..+2}`. Cached per announcement, logged with model cutoff.
- Columns added to the brief: days since last results; results-day abnormal return (3-day, vs market); last vs-expectations sign; profit warning in last 90 days; director net buying in last 90 days (count and £ bucket); **news-day vs no-news 6-month return split**.
- Panel: new "event-driven" style that reads those columns. Desk: a "news analyst" note (from columns, not text) before the bull/bear debate, mirroring TradingAgents' structure.

**Cross-asset agent (`silo.agents.crossasset`)**
- Ingest: `macro` vault table of daily series (GBPUSD, sterling ERI, Brent, gold, silver, copper, UK 10y gilt, S&P 500, VIX), each with `available_at`.
- Per stock: rolling 1-year sensitivities to pound, oil, metals, gilt yield and US market; sector-relative 6-month return; lag vs sector peers.
- Market line: each driver's 3- and 12-month trend shown as a percentile of its own history, plus cross-asset trend count ("4 of 7 drivers in uptrends").
- Risk check for the desk's risk manager: book-level exposure to each driver, with a veto when one driver dominates.
- Panel: a "macro" style; desk: a "macro analyst" note.

**Plain-rule baselines (added to `REGISTRY`, must pass `test_truncation_invariance`)**
- `news_momentum`: momentum using only news-day returns.
- `results_drift`: buy after strong results-day reactions or "ahead of expectations", hold 60 trading days.
- `director_buys`: hold names with recent director buying.
- `sector_rel`: sector momentum plus laggard catch-up.
- `macro_tilt`: momentum book scaled by cross-asset trend count.

## 7. Test plan and stop rules

1. **Data first (no LLM).** Build `macro` from Yahoo/BoE/FRED on the PN50 (cheap, clean). For `news`, settle the source terms first (NSM or permission), then backfill.
2. **Event studies.** For each event type, average abnormal return at 5, 20, 60 and 120 days after `available_at`, FTSE 350 only, after 0.5% stamp duty and costs. Compare with random event dates.
3. **Rule baselines in the lab.** Walk-forward, random control, deflated Sharpe, next to buy-and-hold. Holdout stays closed.
4. **Only then the AI.** Ablation on post-cutoff dates only: panel vs panel+news vs panel+macro vs panel+both, same model, same cache settings, plus the amnesia probe.
5. **Stop rules.** Drop a signal if its rule baseline doesn't beat random after costs in walk-forward. Drop the LLM step if it adds nothing over the rule version of the same features. Treat any result that only appears pre-cutoff as memory, not skill.

Effort, roughly: macro table plus cross-asset columns and two rule baselines is one PR; the news pipeline is a second, larger PR that is gated on the data-source decision.

## 8. Sources

- Chan, W. S. (2003), *Stock price reaction to news and no-news: drift and reversal after headlines*, Journal of Financial Economics.
- Jiang, Li & Wang (2021), *Pervasive underreaction: Evidence from high-frequency data*, Journal of Financial Economics.
- Martineau (2022), [Rest in peace post-earnings announcement drift](https://nowpublishers.com/article/Details/CFR-0122), Critical Finance Review.
- Subrahmanyam (2025), summarised in [Is Post-Earnings Announcement Drift a Thing? Again?](https://anderson-review.ucla.edu/is-post-earnings-announcement-drift-a-thing-again), UCLA Anderson Review (secondary source).
- Liu, Strong & Xu (2003), [Post-earnings-announcement drift in the UK](https://econpapers.repec.org/RePEc:bla:eufman:v:9:y:2003:i:1:p:89-116), European Financial Management.
- Brandt, Kishore, Santa-Clara & Venkatachalam (2008), *Earnings announcements are full of surprises*; Chan, Jegadeesh & Lakonishok (1996), *Momentum strategies*, Journal of Finance.
- Fidrmuc, Goergen & Renneboog (2006), *Insider trading, news releases, and ownership concentration*, Journal of Finance (UK director dealings).
- Ikenberry, Lakonishok & Vermaelen (1995), *Market underreaction to open market share repurchases*.
- Lopez-Lira & Tang, [Can ChatGPT forecast stock price movements?](https://arxiv.org/abs/2304.07619v6) (v6, 2025).
- Chen, Kelly & Xiu, *Expected returns and large language models* ([discussion slides](https://jacobslevycenter.wharton.upenn.edu/wp-content/uploads/2024/09/ExpectedReturnAndLLM_Discussion_SophiaZhengziLi_NoPause.pdf)).
- Tetlock (2007), *Giving content to investor sentiment*, Journal of Finance.
- Gao, Jiang & Yan, [Detecting lookahead bias in LLM forecasts](https://arxiv.org/abs/2512.23847) (2025/26).
- Glasserman & Lin (2023), [Assessing look-ahead bias in stock return predictions generated by GPT sentiment analysis](https://arxiv.org/abs/2309.17322).
- Somoza, Didisheim & Fraschini (2025), [AI's predictable memory in financial analysis](https://faculty.essec.edu/research/16148-ais-predictable-memory-in-financial-analysis), Economics Letters.
- He, Lv, Manela & Wu (2025), [Chronologically consistent large language models](https://arxiv.org/abs/2502.21206).
- Moskowitz & Grinblatt (1999), *Do industries explain momentum?*; Hou (2007), *Industry information diffusion and the lead-lag effect*.
- Moskowitz, Ooi & Pedersen (2012), *Time series momentum*, Journal of Financial Economics.
- Hong, Torous & Valkanov (2007), [Do industries lead stock markets?](https://escholarship.org/uc/item/6x49x543), Journal of Financial Economics.
- Driesprong, Jacobsen & Maat (2008), *Striking oil: Another puzzle?*, Journal of Financial Economics.
- Rapach, Strauss & Zhou (2013), *International stock return predictability: What is the role of the United States?*, Journal of Finance.
- Goyal & Welch (2008), *A comprehensive look at the empirical performance of equity premium prediction*.
- Cohen & Frazzini (2008), [Economic links and predictable returns](https://ideas.repec.org/a/bla/jfinan/v63y2008i4p1977-2011.html), Journal of Finance; Yılkı (2026), [Supply chain propagation of textual signals](https://arxiv.org/pdf/2606.29290).
- Investegate [help](https://www.investegate.co.uk/help) and terms of use (read 13 Nov 2026); [GDELT data](https://gdeltproject.org/data.html).

Citations without links are from the published literature as I know it and were not re-checked in this pass. Effect sizes for older studies may not hold in today's FTSE 350.
