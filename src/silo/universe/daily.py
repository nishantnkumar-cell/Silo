"""Daily-trading universe: a couple of silver ETFs and the two largest coins.

These feed a separate daily-bar book (see ``Constraints.daily_trading``),
kept apart from the UK position-trading books. Only very liquid names:

* **Silver:** ``SLV`` (iShares Silver Trust, since 2006, hundreds of millions
  of dollars traded a day) and ``SIVR`` (abrdn Physical Silver, since 2009,
  cheaper to hold, tens of millions a day). Both are US-listed, so they form a
  US-market book with dollar costs plus an FX fee. UK retail investors usually
  cannot buy US-domiciled ETFs (no UK KID); the London-listed physical silver
  ETCs track the same metal, and SLV's long history stands in for them here.
* **Crypto:** ``BTC-USD`` and ``ETH-USD``, Yahoo pairs that trade every day.
  Crypto has its own market rules: 24/7 calendar, a bar is final at 00:05 UTC
  the next day, exchange fees, and a $1bn-a-day liquidity floor.

One book cannot mix markets, so silver and crypto are two books.
"""

from __future__ import annotations

SILVER_ETFS = ["SLV", "SIVR"]
CRYPTO_PAIRS = ["BTC-USD", "ETH-USD"]
DAILY_UNIVERSES = {"silver": SILVER_ETFS, "crypto": CRYPTO_PAIRS}
