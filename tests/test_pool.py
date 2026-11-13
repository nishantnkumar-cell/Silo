import numpy as np
import pandas as pd
import pytest

from silo.broker import CostModel, SimBroker
from silo.data.vault import normalize_bars
from silo.liquidity import Liquidity
from silo.markets import US
from silo.universe.pool import candidates_from_instruments, rank_pool, yahoo_symbol


def _bars(daily_drift, vol=0.01, price=50.0, volume=2_000_000, n=400, seed=0):
    rng = np.random.default_rng(seed)
    close = price * np.exp(np.cumsum(daily_drift + vol * rng.standard_normal(n)))
    idx = pd.bdate_range("2024-01-02", periods=n)
    df = pd.DataFrame({"open": close, "high": close * 1.005, "low": close * 0.995, "close": close,
                       "volume": float(volume), "dividend": 0.0}, index=idx)
    return normalize_bars(df, US)


INSTRUMENTS = [
    {"ticker": "AAPL_US_EQ", "type": "STOCK", "currencyCode": "USD", "isin": "US0378331005", "name": "Apple Inc"},
    {"ticker": "BRK_B_US_EQ", "type": "STOCK", "currencyCode": "USD", "isin": "US0846707026",
     "name": "Berkshire Hathaway Inc. Class B"},
    {"ticker": "VODl_EQ", "type": "STOCK", "currencyCode": "GBX", "isin": "GB00BH4HKS39", "name": "Vodafone Group PLC"},
    {"ticker": "BT_Al_EQ", "type": "STOCK", "currencyCode": "GBX", "isin": "GB0030913577", "name": "BT Group plc"},
    {"ticker": "VOD_US_EQ", "type": "STOCK", "currencyCode": "USD", "isin": "US92857W3088", "name": "Vodafone Group plc ADR"},
    {"ticker": "VUSAl_EQ", "type": "ETF", "currencyCode": "GBP", "isin": "IE00B3XXRP09", "name": "Vanguard S&P 500"},
    {"ticker": "SAPd_EQ", "type": "STOCK", "currencyCode": "EUR", "isin": "DE0007164600", "name": "SAP SE"},
    {"ticker": "AAPL2_US_EQ", "type": "STOCK", "currencyCode": "USD", "isin": "US0378331005", "name": "Apple again"},
    {"ticker": "XYZW_US_EQ", "type": "WARRANT", "currencyCode": "USD", "isin": "US000", "name": "XYZ warrant"},
]


def test_trading212_tickers_map_to_yahoo():
    assert yahoo_symbol(INSTRUMENTS[0]) == ("US", "AAPL")
    assert yahoo_symbol(INSTRUMENTS[1]) == ("US", "BRK-B")
    assert yahoo_symbol(INSTRUMENTS[2]) == ("UK", "VOD.L")
    assert yahoo_symbol(INSTRUMENTS[3]) == ("UK", "BT-A.L")
    assert yahoo_symbol(INSTRUMENTS[5]) is None  # ETFs are not in the share pool
    assert yahoo_symbol(INSTRUMENTS[6]) is None  # Europe comes later


def test_candidates_keep_one_line_per_company_and_prefer_london():
    out = candidates_from_instruments(INSTRUMENTS)
    assert sorted(out["UK"]) == ["BT-A.L", "VOD.L"]
    assert sorted(out["US"]) == ["AAPL", "BRK-B"]  # the Vodafone ADR and the repeated ISIN are dropped


def test_rank_pool_gates_then_ranks_by_momentum():
    tables = {
        "FAST": _bars(0.002, seed=1),
        "SLOW": _bars(0.0008, seed=2),
        "DOWN": _bars(-0.002, seed=3),  # below its 200-day average
        "WILD": _bars(0.002, vol=0.06, seed=4),  # over 60% a year
        "PENNY": _bars(0.002, price=1.0, seed=5),  # under $5
        "THIN": _bars(0.002, volume=1_000, seed=6),  # far under $20m a day
        "NEW": _bars(0.002, n=100, seed=7),  # under a year of history
    }
    out = rank_pool(tables, liquidity=Liquidity.for_market("US"))
    assert list(out.index[out["passes"]]) == ["FAST", "SLOW"]
    reasons = out["reason"].to_dict()
    assert "200-day" in reasons["DOWN"] and "wild" in reasons["WILD"]
    assert "penny" in reasons["PENNY"] and "illiquid" in reasons["THIN"] and "history" in reasons["NEW"]


def test_rank_pool_is_point_in_time():
    tables = {"A": _bars(0.002, seed=1), "B": _bars(0.001, seed=2)}
    asof = tables["A"]["available_at"].iloc[320]
    cut = {s: df[df["available_at"] <= asof] for s, df in tables.items()}
    full = rank_pool(tables, asof=asof, liquidity=Liquidity.for_market("US"))
    pd.testing.assert_frame_equal(full, rank_pool(cut, liquidity=Liquidity.for_market("US")))


def test_top_and_sector_cap():
    tables = {f"S{i}": _bars(0.001 + 0.0002 * i, seed=i) for i in range(8)}
    sectors = {s: ("Tech" if i % 2 else "Energy") for i, s in enumerate(tables)}
    sectors["S7"] = sectors["S6"] = sectors["S5"] = "Tech"
    out = rank_pool(tables, liquidity=Liquidity.for_market("US"), top=4, sectors=sectors, max_sector_share=0.5)
    picked = list(out.index[out["picked"]])
    assert len(picked) == 4
    assert sum(sectors[s] == "Tech" for s in picked) == 2


def test_us_shares_pay_fx_and_dividend_tax():
    c = CostModel.us_shares_from_uk()
    assert c.fx_bps == 15.0 and c.dividend_tax == 0.15 and c.commission(100) == 0.0
    assert CostModel.for_uk_investor("US") == c and CostModel.for_uk_investor("UK") == CostModel.uk()
    b = SimBroker(cash=0.0, costs=c)
    b.positions["AAPL"] = 100.0
    b.credit_dividends({"AAPL": 1.0}, pd.Timestamp("2025-01-02"))
    assert b.cash == pytest.approx(85.0)


def test_cli_pool_and_universe(tmp_path, monkeypatch):
    import json

    import silo.brokers.trading212 as t212
    from silo.cli import main
    from silo.data.vault import DataVault

    class FakeDemo:
        def instruments(self):
            return INSTRUMENTS

    monkeypatch.setattr(t212, "Trading212Demo", FakeDemo)
    out = tmp_path / "pool"
    assert main(["pool", "instruments", "--out", str(out)]) == 0
    assert json.loads((out / "us-candidates.json").read_text()) == ["AAPL", "BRK-B"]
    assert len(list(out.glob("t212-instruments-*.json"))) == 1

    vault = DataVault(tmp_path / "data")
    vault.put_bars("AAPL", _bars(0.002, seed=1))
    vault.put_bars("BRK-B", _bars(-0.002, seed=3))
    saved = tmp_path / "us.json"
    assert main(["--data-dir", str(tmp_path / "data"), "universe", "us", "--candidates",
                 str(out / "us-candidates.json"), "--save", str(saved)]) == 0
    assert json.loads(saved.read_text()) == ["AAPL"]


def test_fetch_skips_empty_series_instead_of_crashing(tmp_path, monkeypatch, capsys):
    import silo.cli as cli

    good = _bars(0.001, n=30)
    monkeypatch.setitem(cli.SOURCES, "yfinance", lambda s: good if s == "OK" else good.iloc[:0])
    assert cli.main(["--data-dir", str(tmp_path), "fetch", "EMPTY", "OK"]) == 0
    out = capsys.readouterr().out
    assert "EMPTY: skipped" in out and "OK: 30 bars" in out
