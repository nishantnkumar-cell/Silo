# Silo Core Zones: TradingView indicator

> Research only. A chart indicator, not a trading bot: it places no orders and connects to no
> broker. Silo's backtests of these rules (`core_zones`, `core_trail`) have so far lost to
> buy-and-hold and to random entries, so treat the scores as a way to read charts, not as signals.

`core_zones.pine` draws supply and demand zones the way the Core Strategy course defines them,
scores each one out of 10, and (optionally) shows the entry, stop and targets for the best setup.
It is a Pine Script v6 port of Silo's Python rules in `src/silo/strategies/core_zones.py` and
`docs/core-strategy-rulebook.md` (branch `league-backtest`).

## Add it to TradingView

1. Open a chart on tradingview.com (a free account works) and set it to the **daily** time frame.
2. Click **Pine Editor** at the bottom of the screen.
3. Open the menu next to the script name and choose **Create new > Indicator**. Delete the
   sample code.
4. Paste the whole of `core_zones.pine` and click **Save** (name it "Silo Core Zones").
5. Click **Add to chart**.
6. To change settings, click the gear icon next to the indicator's name on the chart.

If TradingView shows a compile error, copy the red message and send it back; the script was
checked against Silo's Python zone finder but not compiled inside TradingView.

## What you see

* **Green boxes**: demand zones. **Red boxes**: supply zones. Darker means a higher score
  (8.5+ darkest, 7-8 medium, below 7 faint). Violated zones (a close beyond the far edge)
  disappear.
* **Box text**: formation (DBR, RBR, RBD, DBD), score out of 10, and "in zone" when price is
  inside it now. The second line is the breakdown: **S**trength 0-2, **T**ime 0-1,
  **F**reshness 0-2, **Tr**end 0-2, **C**urve 0-1, **P**rofit zone 0-2.
* **Dashed boxes**: the nearest monthly supply above price and demand below it (the edges of
  the "curve"). These show the bigger zones even when no daily supply is left.
* **Why there may be no red boxes**: a supply zone is deleted once a daily close goes above its
  top (distal) line, and only the last 260 daily bars are searched. A share that has rallied for
  a year has broken every supply zone above it, so none are drawn; the table's "Zones drawn"
  row says so. Raise "Keep zones from the last N bars" to look further back.
* **Grey step line**: the weekly trend's controlling swing (the low that ends an uptrend if
  price closes below it).
* **Table, top right**: weekly trend, where price sits in the monthly curve, and the best long
  (and short, if turned on).
* **Trade plan** for the best long setup that the decision matrix allows, scores at least the
  minimum (7 by default) and has a target at least **1:4**:
  * blue **Entry** at the proximal line (8.5+: limit order there; 7-8: wait for price to close
    back out of the zone first),
  * red **Stop** beyond the distal line by 10% of the daily ATR,
  * orange dashed **Sell half** at 1:2, then trail the stop under each new demand zone that
    forms,
  * green **Target** at 75% of the way to the next supply zone (or the minimum 1:4 if there is
    no supply overhead).

## Alerts

To be told when price trades into the planned zone: right-click the chart, **Add alert**,
pick "Silo Core Zones" as the condition and **Any alert() function call**. The alert fires once
per daily close while price is in the zone.

## How it maps to Silo's rules

| Rule | Indicator |
|---|---|
| Base candle: body at most half the candle's range, gaps count toward the body | same (`f_kind`) |
| Zone: leg-in, 1-10 base candles, leg-out making a new high (low) | same (`f_detect`) |
| Proximal at the base's highest (lowest) body; distal at the extreme, with the leg-in for DBR/RBD | same |
| Strength, time, freshness | same; the current visit does not count against freshness |
| Trend on weekly bars, controlling swing | same rules, **completed weeks only** (Silo includes the current partial week) |
| Curve on monthly bars, nearest quality zones distal to distal | same, completed months only; zones are picked around the **last monthly close** rather than today's price |
| Profit zone, decision matrix, 8.5 / 7 thresholds | same |
| Stop, 75% target | same, but the plan also requires the user's minimum **1:4** (Silo's backtest used 1:3) |
| Sell half at 1:2, trail the rest | shown as a plan; the indicator does not track an open trade |
| Shorts at supply | optional (off by default); Silo itself is long-only |

A bar-by-bar Python model of the indicator's zone finder was checked against Silo's
`find_zones` on 200 random price series and found the same zones, lines, freshness and strength
every time. TradingView's built-in swing function may treat equal highs slightly differently
from Silo's, which can change a strength or trend score at the margin.
