import json

import pandas as pd
import pytest

from silo.brokers.trading212 import DEMO_BASE, Trading212Demo, mirror_orders, propose_mapping
from silo.data.vault import DataVault
from silo.paper import MirrorError, PaperBook, mirror_latest_decision, reconcile
from silo.safety import LiveTradingRefused, check_environment, check_order_endpoint
from tests.test_uk_and_paper import uk_tables


class FakeT212:
    """Records requests and answers like the documented demo API."""

    def __init__(self, cash=10_000.0, positions=None, currency="GBP"):
        self.calls, self.cash, self.pos, self.currency = [], cash, dict(positions or {}), currency

    def __call__(self, method, url, headers, body):
        assert url.startswith("https://demo.trading212.com/api/v0/")
        assert headers["Authorization"].startswith("Basic ")
        path = url.removeprefix("https://demo.trading212.com/api/v0")
        self.calls.append((method, path, json.loads(body) if body else None))
        if path == "/equity/account/summary":
            return 200, {}, {"cash": {"availableToTrade": self.cash, "reservedForOrders": 0, "inPies": 0},
                             "currency": self.currency, "investments": {"currentValue": 0}}
        if path == "/equity/positions":
            return 200, {}, [{"instrument": {"ticker": t}, "quantity": q} for t, q in self.pos.items()]
        if path == "/equity/metadata/instruments":
            return 200, {}, [{"ticker": "T0l_EQ", "shortName": "T0", "currencyCode": "GBX"},
                             {"ticker": "T1l_EQ", "shortName": "T1", "currencyCode": "GBX"},
                             {"ticker": "T1_US_EQ", "shortName": "T1", "currencyCode": "USD"}]
        if path == "/equity/orders/market":
            return 200, {}, {"id": len(self.calls), "status": "NEW"}
        return 404, {}, {"error": "not found"}


def client(fake):
    return Trading212Demo("k", "s", transport=fake, sleep=lambda s: None)


def test_only_demo_host_allowed():
    assert check_order_endpoint(DEMO_BASE)
    with pytest.raises(LiveTradingRefused):
        check_order_endpoint("https://live.trading212.com/api/v0/equity/orders/market")
    with pytest.raises(LiveTradingRefused):
        check_environment({"T212_LIVE_API_KEY": "x"})


def test_mapping_prefers_london_listing():
    m = propose_mapping(["T0.L", "T1.L", "ZZ.L"], client(FakeT212()).instruments())
    assert m == {"T0.L": "T0l_EQ", "T1.L": "T1l_EQ", "ZZ.L": None}


def test_mirror_orders_sells_first_and_whole_shares():
    orders = mirror_orders({"A.L": 0.5, "B.L": 0.0}, {"A.L": 3.0, "B.L": 2.0},
                           {"A.L": "A", "B.L": "B"}, {"B": 10.5}, 1_000.0)
    assert orders == [("B", -10.5), ("A", 166)]


def _book_with_decision(tmp_path):
    syms, full = uk_tables(n=2, n_days=700)
    vault = DataVault(tmp_path / "data")
    for s, df in full.items():
        vault.put_bars(s, df.iloc[:500])
    book = PaperBook("t212", tmp_path / "paper")
    book.create("uk_trend", syms, vault, {"lookback": 126, "top_k": 2})
    for s, df in full.items():
        vault.put_bars(s, df)
    # step one day at a time until the latest day carries a fresh decision
    state = book._load()
    from silo.engine import MarketData

    data = MarketData({s: vault.get_bars(s) for s in syms})
    for date in [d for d in data.dates if d > pd.Timestamp(book.config()["start_after"])]:
        ev = state["book"].step(date, data)
        state["benchmark"].step(date, data)
        if ev["decision"] and ev["decision"]["weights"]:
            break
    book._save(state)
    (book.dir / "t212_map.json").write_text(json.dumps({"T0.L": "T0l_EQ", "T1.L": "T1l_EQ"}))
    return book


def test_mirror_sends_once_and_reconciles(tmp_path):
    book = _book_with_decision(tmp_path)
    fake = FakeT212()
    sent = mirror_latest_decision(book, client(fake))
    assert sent and all(e["kind"] == "sent" for e in sent)
    n_orders = sum(1 for c in fake.calls if c[1] == "/equity/orders/market")
    assert mirror_latest_decision(book, client(fake)) == []  # rerun: nothing sent twice
    assert sum(1 for c in fake.calls if c[1] == "/equity/orders/market") == n_orders
    fake.pos = {"T0l_EQ": 5.0}  # broker filled some of it
    table = reconcile(book, client(fake))
    assert table.loc["T0.L", "practice_qty"] == 5.0


def test_mirror_refuses_currency_mismatch_and_missing_map(tmp_path):
    book = _book_with_decision(tmp_path)
    with pytest.raises(MirrorError):
        mirror_latest_decision(book, client(FakeT212(currency="EUR")))
    (book.dir / "t212_map.json").write_text(json.dumps({"T0.L": "T0l_EQ"}))
    with pytest.raises(MirrorError):
        mirror_latest_decision(book, client(FakeT212()))
