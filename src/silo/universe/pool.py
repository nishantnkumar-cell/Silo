"""The wider pool: every share Trading 212 lists in London or New York.

Three steps, each kept separate so it can be checked:

1. :func:`candidates_from_instruments` turns Trading 212's Practice
   instrument list into Yahoo tickers, one list per market. Only ordinary
   shares (``type == "STOCK"``) in the UK and US are kept; ETFs, warrants,
   rights and the rest are dropped, as are European listings for now. A
   company listed in both London and New York (a UK share plus a US ADR)
   keeps only its London line, since a UK investor pays no currency fee
   there. Trading 212's list has no prices or volume, so it says what *can*
   be traded, not what is liquid.
2. Prices for those tickers come from Yahoo (``silo fetch --symbols-file``)
   into the vault, as for every other book.
3. :func:`rank_pool` applies the liquidity gate and ranks the survivors by
   momentum and trend. It is point-in-time: pass ``asof`` and it only reads
   bars public by then, so the same code can pick a pool in a backtest.

Today's Trading 212 list only holds survivors (shares that went bust or were
taken over are gone), so a backtest on a pool picked from it is flattered.
Forward paper trading is the honest test.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from silo.liquidity import Liquidity

# Trading 212 tickers: "AAPL_US_EQ" in New York, "VODl_EQ" in London (the
# lower-case "l" marks the London listing). Other suffixes are other exchanges.
_US = re.compile(r"^(?P<base>.+)_US_EQ$")
_UK = re.compile(r"^(?P<base>.+)l_EQ$")
_NAME_NOISE = re.compile(r"\b(plc|inc|corp|corporation|ltd|limited|group|holdings?|co|company|sa|nv|ag|adr|ads|the)\b|[^a-z0-9 ]")


# Trading 212 codes that differ from Yahoo's, confirmed by hand. `silo pool
# resolve` finds more by looking up the ISIN on Yahoo.
YAHOO_OVERRIDES = {
    "BT.L": "BT-A.L",  # BT Group
    "ASOS.L": "ASC.L",  # ASOS
}


def yahoo_symbol(instrument: dict) -> tuple[str, str] | None:
    """(market, Yahoo ticker) for a Trading 212 share, or None if out of scope."""
    if str(instrument.get("type", "")).upper() != "STOCK":
        return None
    ticker = str(instrument.get("ticker", ""))
    currency = str(instrument.get("currencyCode", "")).upper()
    if (m := _US.match(ticker)) and currency == "USD":
        return "US", m["base"].replace("_", "-").replace(".", "-").upper()
    if (m := _UK.match(ticker)) and currency in {"GBX", "GBP"}:
        sym = m["base"].replace("_", "-").replace(".", "-").upper() + ".L"
        return "UK", YAHOO_OVERRIDES.get(sym, sym)
    return None


def _company(instrument: dict) -> str:
    name = str(instrument.get("name") or instrument.get("shortName") or "").lower()
    return " ".join(_NAME_NOISE.sub(" ", name).split())


def candidates_from_instruments(instruments: list[dict]) -> dict[str, dict[str, dict]]:
    """``{"UK": {yahoo: instrument}, "US": {...}}``, one line per company.

    Repeats of the same ISIN keep the shortest ticker. A US line whose company name
    matches a London share is treated as a duplicate listing (usually an ADR)
    and dropped in favour of London.
    """
    out: dict[str, dict[str, dict]] = {"UK": {}, "US": {}}
    seen_isin: set[str] = set()
    for inst in sorted(instruments, key=lambda i: (len(str(i.get("ticker", ""))), str(i.get("ticker", "")))):
        hit = yahoo_symbol(inst)
        if hit is None:
            continue
        market, symbol = hit
        isin = str(inst.get("isin") or "")
        if isin and isin in seen_isin:
            continue
        seen_isin.add(isin)
        out[market].setdefault(symbol, inst)
    london = {_company(i) for i in out["UK"].values()} - {""}
    out["US"] = {s: i for s, i in out["US"].items() if _company(i) not in london}
    return out


def rank_pool(
    tables: dict[str, pd.DataFrame],
    asof: pd.Timestamp | None = None,
    liquidity: Liquidity | None = None,
    min_history: int = 252,
    max_volatility: float = 0.60,
    near_high: float = 0.25,
    top: int | None = None,
    sectors: dict[str, str] | None = None,
    max_sector_share: float = 0.25,
) -> pd.DataFrame:
    """Gate, then rank by momentum and trend. Best first; one row per symbol.

    Gate (a name failing any is out): at least ``min_history`` days of bars,
    price and median 60-day value traded above the market's liquidity floors,
    volatility no higher than ``max_volatility`` a year.

    Trend (also required): close above its 200-day average and within
    ``near_high`` of its 52-week high.

    Rank: momentum, the return from 12 months ago to 1 month ago (the latest
    month is skipped because it tends to reverse). Ties go to the calmer name.

    With ``top`` set, only the best ``top`` are marked ``picked``; with
    ``sectors`` as well, no sector may take more than ``max_sector_share`` of
    them (sector labels are today's, so use this for live pools only).
    """
    rules = liquidity or Liquidity()
    rows = {}
    for symbol, df in tables.items():
        if asof is not None:
            df = df[df["available_at"] <= pd.Timestamp(asof)]
        if len(df) < min_history:
            rows[symbol] = {"passes": False, "reason": f"only {len(df)} days of history"}
            continue
        last = df.iloc[-252:]
        close = last["close"]
        rets = close.pct_change().dropna()
        vol = float(rets.iloc[-60:].std(ddof=1) * np.sqrt(252))
        value = float(np.nanmedian((close * last["volume"]).iloc[-rules.window:].to_numpy()))
        price = float(close.iloc[-1])
        mom = float(close.iloc[-21] / close.iloc[0] - 1)
        above_200 = bool(price > close.iloc[-200:].mean())
        off_high = float(price / close.max() - 1)
        reasons = []
        if not price >= rules.min_price:
            reasons.append(f"penny stock: {price:.2f} a share")
        if not value >= rules.min_value_traded:
            reasons.append(f"illiquid: {value / 1e6:.1f}m a day")
        if not vol <= max_volatility:
            reasons.append(f"too wild: volatility {vol:.0%}")
        if not above_200:
            reasons.append("below its 200-day average")
        if off_high < -near_high:
            reasons.append(f"{-off_high:.0%} below its 52-week high")
        rows[symbol] = {
            "momentum_12_1": mom,
            "volatility": vol,
            "price": price,
            "value_traded_m": value / 1e6,
            "off_52w_high": off_high,
            "above_200d": above_200,
            "passes": not reasons,
            "reason": "; ".join(reasons),
        }
    table = pd.DataFrame(rows).T
    for col in ("momentum_12_1", "volatility"):
        if col not in table:
            table[col] = np.nan
    table["passes"] = table["passes"].astype(bool)
    table = table.sort_values(["passes", "momentum_12_1", "volatility"], ascending=[False, False, True])
    table["picked"] = False
    if sectors is not None:
        table["sector"] = [sectors.get(s, "unknown") for s in table.index]
    limit = len(table) if top is None else top
    per_sector = max(1, int(limit * max_sector_share)) if sectors is not None else limit
    counts: dict[str, int] = {}
    picked = 0
    for symbol in table.index[table["passes"]]:
        if picked >= limit:
            break
        sector = table.at[symbol, "sector"] if sectors is not None else ""
        if counts.get(sector, 0) >= per_sector and sector != "unknown":
            continue
        counts[sector] = counts.get(sector, 0) + 1
        table.at[symbol, "picked"] = True
        picked += 1
    return table


def resolve_by_isin(missing: dict[str, dict], market: str) -> dict[str, str | None]:  # pragma: no cover
    """For candidates Yahoo had no data for, look the ISIN up on Yahoo.

    Returns ``{trading212_guess: yahoo_symbol or None}``. Only a hit on the
    same market (``.L`` for the UK, no suffix for the US) is accepted.
    """
    import yfinance as yf

    out: dict[str, str | None] = {}
    for guess, inst in missing.items():
        isin = inst.get("isin")
        found = None
        try:
            quotes = yf.Search(isin, max_results=5).quotes if isin else []
        except Exception:  # noqa: BLE001
            quotes = []
        for q in quotes:
            sym = str(q.get("symbol", ""))
            if (market == "UK" and sym.endswith(".L")) or (market == "US" and "." not in sym):
                found = sym
                break
        out[guess] = found
    return out


def sectors_today(symbols: list[str]) -> dict[str, str]:  # pragma: no cover - needs network
    """Today's sector for each symbol from Yahoo. Not point-in-time: live pools only."""
    import yfinance as yf

    out = {}
    for s in symbols:
        try:
            out[s] = yf.Ticker(s).info.get("sector") or "unknown"
        except Exception:  # noqa: BLE001
            out[s] = "unknown"
    return out
