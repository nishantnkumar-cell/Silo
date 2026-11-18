# Silo Core Zones: TradingView indicator

> Research only. A chart indicator, not a trading bot: it places no orders and connects to no
> broker. Silo's backtests of the earlier zone rules lost to buy-and-hold and to random entries,
> so treat the zones as a way to read charts, not as signals.

`core_zones.pine` draws the **immediate demand zone below price and the immediate supply zone above
it**, the way the Core Strategy manual's zoning process finds them, refines each zone's proximal
line on a lower time frame, and scores each zone out of 10. The rules were re-derived from the
manual itself (OCR text in the project Library: `silo-reports/research/ocr/`); slide numbers are
given in the script's header.

## Add it to TradingView

1. Open a chart on tradingview.com and set it to the **daily** time frame.
2. Click **Pine Editor** at the bottom of the screen.
3. If you added the earlier version, open it; otherwise choose **Create new > Indicator**.
   Select all the code and replace it with the whole of `core_zones.pine`.
4. Click **Save**, then **Add to chart** (or **Update on chart**).
5. Settings: the gear icon next to the indicator's name.

If TradingView shows a compile error, copy the red message and send it back; the script has not
been compiled inside TradingView yet.

## How a zone is found (from the manual)

| Step | Manual | Indicator |
|---|---|---|
| Candles | Leg candles close outside the range of the preceding candle and look bullish or bearish; base candles are neutral and close inside it; a gap counts as part of the candle (slides 79-81, 50) | leg = closes beyond the previous candle's high (low) **and** in the top (bottom) 40% of its range; everything else is a base candle |
| Zone | Leg-in, base, leg-out; a rally out = demand, a drop out = supply (89-91) | same, 1 to 6 base candles |
| Explosive move | Leg candles with consecutive higher (lower) closes that make a **new high (low)**; a big candle that fails to make a new high does not count (72-77) | the move out must take out the last swing high (low) before the base |
| Lines | Base isolation; proximal at the highest (lowest) **body** in the base, the preferred setting; distal at the extreme of the whole DBR/RBD, or of base and move out for RBR/DBD (96-115) | same |
| Which zones | Start at current price and look down (up) and left until you reach the explosive move (140-145) | only the **nearest** live demand below and supply above price are drawn |
| Refining | The Refining Time Frame refines the proximal and distal lines (312). The manual lists it as optional and leaves the method to the advanced course | on the lower time frame (60 min by default), find the turning point inside the zone, take the small base right before the first lower-time-frame leg out of it, and move the proximal line to that base's body. The distal line is the same price on any time frame. The dashed outline shows the unrefined daily zone |
| Violation | Price crossing the distal line disqualifies the zone (127) | any trade beyond the distal line (or only a close, in settings) |
| Score | Strength 2, time 1, freshness 2, trend 2 (weekly), curve 1 (monthly), profit zone 2 | same; the profit zone uses the immediate opposing zone |

Lower-time-frame refining needs intraday data for the days that formed the zone. TradingView
keeps a limited amount of hourly history, so an old zone may say "not refined" and keep its daily
lines.

If there is no daily supply left above price (a share at its highs), the nearest monthly supply is
shown as a dashed box instead, and the same for demand.

## Alerts

Right-click the chart, **Add alert**, choose "Silo Core Zones" and **Any alert() function call**.
It fires once per daily close while price is inside the immediate demand or supply zone.

## Trade plan (off by default)

Turn on "Show stop, entry and targets" to draw, at the immediate demand: entry at the proximal
line, stop 10% of daily ATR beyond the distal line (slide 425), sell half at 1:2 and then trail,
and a target at 75% of the way to the immediate supply. It is only drawn when that target is at
least 1:4 and the decision matrix allows a trade.

## What Silo's core strategy would change to match

Once the chart looks right, `silo.strategies.core_zones` would need these changes:

1. **Candle test:** a leg candle must close beyond the previous candle's high or low. Silo
   currently only checks that the body is more than half the range.
2. **Explosive test:** the move out must break the last swing high (low) before the base. Silo
   only checks the leg-in and base highs.
3. **Breakout:** the move out must cross past the previous opposing *zone*. Silo uses a swing
   high or low.
4. **Violation:** any trade through the distal line, optionally a close. Silo uses a close only,
   and does not check during the move out.
5. **Immediate zone only:** trade only the nearest live demand below price, and use the nearest
   supply above as the profit zone. Silo scores every demand zone below price.
6. **Refining:** Silo has daily bars only. Refining needs hourly bars for at least the past few
   years in the data vault, from a source with point-in-time timestamps, with `available_at` on
   every row. Without them Silo would trade the unrefined daily zone.
7. **Base length:** at most 6 base candles. Silo allows 10.
