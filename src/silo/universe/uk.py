"""UK universe: liquid, good-quality FTSE 350 shares with healthy volatility.

``UK_CANDIDATES`` is a hand-kept starting list of large, liquid London
listings (Yahoo ``.L`` tickers). It is today's list, so backtests on it carry
survivorship bias: companies that fell out of the index are missing. Fetch
failures (delisted or renamed tickers) are skipped and reported.

No penny stocks or thin names: :func:`screen` rejects shares under £1 or
trading less than £5m a day, and the engine enforces the same floors at every
decision (see :mod:`silo.liquidity`), so a strategy cannot buy its way around
the screen.

Two screens:
* :func:`screen` uses prices only and is point-in-time: pass a view's clock
  and it only looks at bars public by then.
* :func:`quality_screen` uses today's fundamentals from Yahoo. Those are
  not point-in-time, so use it to pick the live paper-trading universe, never
  to select names for a historical backtest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

UK_CANDIDATES = [
    "AZN.L", "SHEL.L", "HSBA.L", "ULVR.L", "BP.L", "GSK.L", "RIO.L", "REL.L", "DGE.L", "BATS.L",
    "LSEG.L", "NG.L", "BARC.L", "LLOY.L", "NWG.L", "PRU.L", "AAL.L", "GLEN.L", "VOD.L", "TSCO.L",
    "BA.L", "RR.L", "CPG.L", "EXPN.L", "AHT.L", "IHG.L", "IMB.L", "SSE.L", "LGEN.L", "AV.L",
    "SGE.L", "HLN.L", "SMIN.L", "WTB.L", "NXT.L", "MKS.L", "SBRY.L", "ITRK.L", "HLMA.L", "SPX.L",
    "CRDA.L", "AUTO.L", "RMV.L", "III.L", "BNZL.L", "ANTO.L", "ENT.L", "BDEV.L", "PSN.L", "TW.L",
    "BKG.L", "IAG.L", "EZJ.L", "WPP.L", "INF.L", "PSON.L", "SN.L", "MNDI.L", "STAN.L", "ADM.L",
    "JD.L", "KGF.L", "UU.L", "SVT.L", "HWDN.L", "DPLM.L", "GAW.L", "IMI.L", "WEIR.L", "RTO.L",
    # more FTSE 100 and liquid FTSE 250 names
    "BEZ.L", "HIK.L", "BRBY.L", "CNA.L", "DCC.L", "FRES.L", "MRO.L", "LAND.L", "BLND.L", "SGRO.L",
    "PHNX.L", "SDR.L", "SMDS.L", "JMAT.L", "BME.L", "ICG.L", "HSX.L", "LMP.L", "MNG.L", "BBY.L",
    "BWY.L", "CCC.L", "DLN.L", "DRX.L", "FRAS.L", "GRG.L", "INCH.L", "IGG.L", "TPK.L", "VTY.L",
    "HTWS.L", "ITV.L", "PAGE.L", "QQ.L", "ROR.L", "RS1.L", "SXS.L", "SRP.L", "TATE.L", "TRN.L",
    "VSVS.L", "WOSG.L", "BAB.L", "SPT.L", "FOUR.L", "GNS.L", "JET2.L", "OCDO.L", "PFD.L", "PNN.L",
    "RAT.L", "RWS.L", "SAFE.L", "AJB.L", "PLUS.L", "CRST.L", "BOY.L", "KNOS.L", "HILS.L", "EMG.L",
    "BYG.L", "UTG.L", "DOM.L", "MTO.L", "CKN.L", "HAS.L", "BAG.L", "ASHM.L", "JUP.L", "QLT.L",
]
UK_CANDIDATES = list(dict.fromkeys(UK_CANDIDATES))  # keep order, drop repeats


def screen(
    tables: dict[str, pd.DataFrame],
    asof: pd.Timestamp | None = None,
    min_history: int = 252,
    min_value_traded: float = 5e6,
    min_price: float = 1.0,
    vol_band: tuple[float, float] = (0.20, 0.45),
    max_drawdown: float = -0.5,
) -> pd.DataFrame:
    """Score each symbol on liquidity, volatility and damage, using prices only.

    ``asof`` limits each table to bars public by then (point-in-time).
    Volatility between 20% and 45% a year is "healthy": enough movement to
    reward holding for weeks, not so much that one bad week dominates.
    """
    rows = {}
    for symbol, df in tables.items():
        if asof is not None:
            df = df[df["available_at"] <= pd.Timestamp(asof)]
        if len(df) < min_history:
            rows[symbol] = {"passes": False, "reason": f"only {len(df)} days of history"}
            continue
        last = df.iloc[-252:]
        rets = last["close"].pct_change().dropna()
        vol = float(rets.iloc[-60:].std(ddof=1) * np.sqrt(252))
        value = float((last["close"] * last["volume"]).iloc[-60:].median())
        dd = float((last["close"] / last["close"].cummax() - 1).min())
        trend = bool(last["close"].iloc[-1] > last["close"].iloc[-200:].mean())
        price = float(last["close"].iloc[-1])
        reasons = []
        if price < min_price:
            reasons.append(f"penny stock: £{price:.2f} a share")
        if value < min_value_traded:
            reasons.append(f"illiquid: £{value / 1e6:.1f}m a day")
        if not vol_band[0] <= vol <= vol_band[1]:
            reasons.append(f"volatility {vol:.0%} outside {vol_band[0]:.0%} to {vol_band[1]:.0%}")
        if dd < max_drawdown:
            reasons.append(f"fell {dd:.0%} in a year")
        rows[symbol] = {
            "price": price,
            "volatility": vol,
            "value_traded_m": value / 1e6,
            "max_drawdown_1y": dd,
            "above_200d": trend,
            "passes": not reasons,
            "reason": "; ".join(reasons),
        }
    return pd.DataFrame(rows).T.sort_values(["passes", "volatility"], ascending=[False, True])


QUALITY_RULES = {
    "returnOnEquity": (">=", 0.12),
    "profitMargins": (">=", 0.08),
    "debtToEquity": ("<=", 150.0),  # Yahoo reports this as a percentage
}


def quality_screen(symbols: list[str]) -> pd.DataFrame:  # pragma: no cover - needs network
    """Today's fundamentals from Yahoo, checked against ``QUALITY_RULES``.

    Not point-in-time: never use this to choose names for a backtest.
    """
    import yfinance as yf

    rows = {}
    for symbol in symbols:
        try:
            info = yf.Ticker(symbol).info
        except Exception as exc:  # noqa: BLE001
            rows[symbol] = {"quality": False, "reason": f"no data: {exc}"}
            continue
        row, fails = {}, []
        for key, (op, limit) in QUALITY_RULES.items():
            v = info.get(key)
            row[key] = v
            ok = v is not None and (v >= limit if op == ">=" else v <= limit)
            if not ok:
                fails.append(f"{key} {v} (needs {op} {limit})")
        row["quality"] = not fails
        row["reason"] = "; ".join(fails)
        rows[symbol] = row
    return pd.DataFrame(rows).T
