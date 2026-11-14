"""Cross-market series (pound, oil, metals, gilts, US shares, VIX) as context tables.

Each driver is stored as a ``ctx:macro:<name>`` table with a ``value``
column and an ``available_at`` UTC column. Market prices are not revised,
so point-in-time is simply "when was the day's value public":

* Yahoo series (FX, futures, US indices) are stamped 23:00 UTC on their
  date, after every US close. A UK decision at the 16:35 London close
  therefore sees these series only up to the *previous* day. That loses a
  few hours of information but can never peek.
* Bank of England gilt yields are published the next working day, so they
  are stamped 12:00 UTC on the next business day.

Fetching needs the network (Yahoo, Bank of England). Run it on the computer
that runs the league; the cloud sandbox blocks these hosts.
"""

from __future__ import annotations

import io
import urllib.request
from dataclasses import dataclass

import pandas as pd

YAHOO_STAMP = pd.Timedelta(hours=23)


@dataclass(frozen=True)
class Driver:
    name: str
    label: str  # plain words shown to a model
    source: str  # "yahoo" or "boe"
    code: str
    kind: str = "price"  # "price": compare by % change; "level": by change in points (yields)
    risk_on: int = 0  # +1 rising is risk-on, -1 rising is risk-off, 0 ambiguous


DRIVERS = {d.name: d for d in [
    Driver("gbp", "pound vs dollar", "yahoo", "GBPUSD=X"),
    Driver("oil", "Brent oil", "yahoo", "BZ=F"),
    Driver("gold", "gold", "yahoo", "GC=F"),
    Driver("silver", "silver", "yahoo", "SI=F"),
    Driver("copper", "copper", "yahoo", "HG=F", risk_on=1),
    Driver("us_stocks", "US shares (S&P 500)", "yahoo", "^GSPC", risk_on=1),
    Driver("vix", "US volatility index", "yahoo", "^VIX", risk_on=-1),
    Driver("gilt10", "UK 10-year gilt yield", "boe", "IUDMNPY", kind="level"),
]}

# Series kept for bookkeeping, not shown to models or used as signals: Bank Rate pays interest on idle cash.
EXTRA = {d.name: d for d in [
    Driver("bank_rate", "Bank of England base rate", "boe", "IUDBEDR", kind="level"),
]}

# Drivers used for each stock's sensitivities (besides the UK market itself).
BETA_DRIVERS = ["gbp", "oil", "copper", "gilt10"]


def table_name(driver: str) -> str:
    return f"ctx:macro:{driver}"


def to_context(values: pd.Series, available_at: pd.DatetimeIndex) -> pd.DataFrame:
    df = pd.DataFrame({"value": values.astype(float).to_numpy(), "available_at": available_at},
                      index=pd.DatetimeIndex(values.index, name="date"))
    df = df[df["value"].notna()]
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def yahoo_stamp(dates) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(dates)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return (idx.normalize() + YAHOO_STAMP).tz_localize("UTC")


def boe_stamp(dates) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(dates).normalize()
    return (idx + pd.offsets.BDay(1) + pd.Timedelta(hours=12)).tz_localize("UTC")


def from_yahoo_frame(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"] if "Close" in df.columns else df["close"]
    idx = pd.DatetimeIndex(close.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    close = pd.Series(close.to_numpy(dtype=float), index=idx.normalize())
    close = close[close > 0]
    return to_context(close, yahoo_stamp(close.index))


def parse_boe_csv(text: str, code: str) -> pd.DataFrame:
    """Bank of England database CSV (``DATE,<code>``, dates like ``02 Jan 2024``)."""
    raw = pd.read_csv(io.StringIO(text))
    raw.columns = [str(c).strip() for c in raw.columns]
    if "DATE" not in raw.columns or code not in raw.columns:
        raise RuntimeError(f"unexpected Bank of England CSV columns: {list(raw.columns)[:5]}")
    dates = pd.to_datetime(raw["DATE"], format="%d %b %Y")
    values = pd.to_numeric(raw[code], errors="coerce")
    s = pd.Series(values.to_numpy(), index=pd.DatetimeIndex(dates))
    return to_context(s, boe_stamp(s.index))


def fetch_driver(name: str, start: str = "2005-01-01") -> pd.DataFrame:
    d = DRIVERS.get(name) or EXTRA[name]
    if d.source == "yahoo":
        try:
            import yfinance as yf
        except ImportError as exc:  # pragma: no cover - optional extra
            raise RuntimeError("install the data extra: pip install 'silo[data]'") from exc
        df = yf.download(d.code, start=start, auto_adjust=False, progress=False, multi_level_index=False)
        if df.empty:
            raise RuntimeError(f"Yahoo returned no data for {d.code}")
        return from_yahoo_frame(df)
    start_ts = pd.Timestamp(start)
    url = ("https://www.bankofengland.co.uk/boeapps/database/_iadb-fromshowcolumns.asp?csv.x=yes"
           f"&Datefrom={start_ts:%d/%b/%Y}&Dateto=now&SeriesCodes={d.code}&CSVF=TN&UsingCodes=Y&VPD=Y&VFD=N")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research script)"})
    with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310 - fixed https host
        return parse_boe_csv(r.read().decode("utf-8", "replace"), d.code)


def fetch_all(vault, names=None, log=print) -> list[str]:
    """One new snapshot per driver. Returns drivers that failed."""
    failed = []
    for name in names or list(DRIVERS) + list(EXTRA):
        try:
            stored = vault.put_context(table_name(name), fetch_driver(name))
            log(f"{name}: {len(stored)} days, {stored.index[0].date()} to {stored.index[-1].date()}")
        except Exception as exc:  # noqa: BLE001 - keep going; report at the end
            failed.append(name)
            log(f"macro fetch failed for {name}: {exc}")
    return failed


def synthetic_macro(dates, seed: int = 0) -> dict[str, pd.DataFrame]:
    """Random-walk driver tables for offline tests. No real signal."""
    import numpy as np

    rng = np.random.default_rng(seed)
    idx = pd.DatetimeIndex(dates)
    out = {}
    for i, (name, d) in enumerate(DRIVERS.items()):
        steps = rng.normal(0, 0.01, len(idx))
        values = 3 + np.cumsum(steps) * 0.5 if d.kind == "level" else 100 * np.exp(np.cumsum(steps))
        s = pd.Series(values, index=idx)
        out[table_name(name)] = to_context(s, boe_stamp(idx) if d.source == "boe" else yahoo_stamp(idx))
    return out
