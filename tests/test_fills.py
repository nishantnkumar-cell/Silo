import json

import pandas as pd
import pytest

from silo.fills import _day, collect_fills, fill_gap, gap_report, parse_history_item, summarize_gap
from silo.paper import _mirror_log, mirror_latest_decision
from tests.test_trading212 import FakeT212, _book_with_decision, client


class FakeHistory(FakeT212):
    """Adds the documented /equity/history/orders endpoint, two pages."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.history: list[dict] = []

    def __call__(self, method, url, headers, body):
        path = url.removeprefix("https://demo.trading212.com/api/v0")
        if path.startswith("/equity/history/orders"):
            self.calls.append((method, path, None))
            if "cursor=" in path:
                return 200, {}, {"items": self.history[2:], "nextPagePath": None}
            more = "/api/v0/equity/history/orders?limit=50&cursor=1760346100000" if len(self.history) > 2 else None
            return 200, {}, {"items": self.history[:2], "nextPagePath": more}
        return super().__call__(method, url, headers, body)


def item(order_id, ticker, qty, price_gbx, status="FILLED", fees=(), fill_id=None, filled_at="2026-11-14T08:00:05Z"):
    return {
        "order": {"id": order_id, "ticker": ticker, "status": status, "currency": "GBX",
                  "createdAt": "2026-11-13T18:31:00Z", "quantity": qty},
        "fill": {"id": fill_id or order_id * 10, "filledAt": filled_at, "price": price_gbx, "quantity": qty,
                 "type": "TRADE",
                 "walletImpact": {"currency": "GBP", "fxRate": 1, "netValue": -qty * price_gbx / 100,
                                  "taxes": [{"name": n, "quantity": -v, "currency": "GBP"} for n, v in fees]}},
    }


def test_parse_documented_shape():
    p = parse_history_item(item(7, "T0l_EQ", 10, 250.0, fees=[("STAMP_DUTY_RESERVE_TAX", 1.25)]))
    assert p["order_id"] == 7 and p["fill_price"] == 250.0 and p["fees"] == 1.25
    assert parse_history_item({"order": {"id": 1, "status": "CANCELLED"}, "fill": None}) is None
    split = item(8, "T0l_EQ", 10, 250.0)
    split["fill"]["type"] = "STOCK_SPLIT"
    assert parse_history_item(split) is None


def _sent_book(tmp_path):
    book = _book_with_decision(tmp_path)
    fake = FakeHistory()
    sent = mirror_latest_decision(book, client(fake))
    assert sent
    return book, fake, sent


def test_collect_logs_each_fill_once_and_waits_for_unfilled(tmp_path):
    book, fake, sent = _sent_book(tmp_path)
    first = sent[0]
    assert collect_fills(book, client(fake)) == []  # nothing has filled yet
    fake.history = [item(first["order_id"], first["ticker"], first["quantity"], 1234.0)]
    got = collect_fills(book, client(fake))
    assert len(got) == 1 and got[0]["decision_date"] == first["decision_date"]
    assert collect_fills(book, client(fake)) == []  # rerun logs nothing twice
    assert sum(1 for e in _mirror_log(book) if e["kind"] == "fill") == 1
    assert not any(c[1] == "/equity/orders/market" for c in fake.calls[-3:])  # collecting never orders


def test_partial_fills_combine_after_order_is_final(tmp_path):
    book, fake, sent = _sent_book(tmp_path)
    o = sent[0]
    q = o["quantity"]
    fake.history = [item(o["order_id"], o["ticker"], q / 2, 100.0, status="PARTIALLY_FILLED", fill_id=1)]
    assert collect_fills(book, client(fake)) == []
    fake.history = [item(o["order_id"], o["ticker"], q / 2, 100.0, status="FILLED", fill_id=1),
                    item(o["order_id"], o["ticker"], q / 2, 110.0, status="FILLED", fill_id=2)]
    (got,) = collect_fills(book, client(fake))
    assert got["fill_qty"] == pytest.approx(q)
    assert got["fill_price"] == pytest.approx(105.0)


def test_paging_follows_next_page_path(tmp_path):
    book, fake, sent = _sent_book(tmp_path)
    o = sent[0]
    fake.history = [item(900, "X", 1, 1.0), item(901, "X", 1, 1.0),
                    item(o["order_id"], o["ticker"], o["quantity"], 100.0)]
    assert len(collect_fills(book, client(fake))) == 1
    assert any("cursor=1760346100000" in c[1] for c in fake.calls)


def test_gap_pairs_practice_with_simulator_fill(tmp_path):
    book, fake, sent = _sent_book(tmp_path)
    # let the simulator fill the decision at the next open
    from silo.data.vault import DataVault

    book.run(DataVault(tmp_path / "data"))
    session = book._load()["book"]
    decided = _day(sent[0]["decision_date"])
    sim = [f for f in session.broker.fills if _day(f["decided_at"]) == decided]
    assert sim, "simulator should have filled the rebalance"
    mapping = json.loads((book.dir / "t212_map.json").read_text())
    by_ticker = {t: s for s, t in mapping.items()}
    o = next(e for e in sent if any(f["symbol"] == by_ticker[e["ticker"]] for f in sim))
    s = next(f for f in sim if f["symbol"] == by_ticker[o["ticker"]])
    practice_gbp = s["price"] * 1.001  # 10 bp worse than the simulator
    fake.history = [item(o["order_id"], o["ticker"], o["quantity"], practice_gbp * 100,
                         fees=[("STAMP_DUTY_RESERVE_TAX", 0.005 * abs(o["quantity"]) * practice_gbp)])]
    collect_fills(book, client(fake))
    gap = fill_gap(book)
    row = gap.iloc[0]
    assert row["symbol"] == by_ticker[o["ticker"]]
    sign = 1 if o["quantity"] > 0 else -1
    assert row["slippage_bps"] == pytest.approx(10.0 * sign, abs=0.01)
    assert row["practice_price"] == pytest.approx(practice_gbp)
    assert row["practice_fee_bps"] == pytest.approx(50.0, abs=0.01)
    assert row["practice_delay_h"] > 0
    gap2, summary = gap_report(book)
    assert summary["paired"] == 1 and "suggested_slippage_bps" not in summary  # too few to suggest


def test_summary_suggests_slippage_only_with_enough_fills():
    from silo.broker import CostModel

    gap = pd.DataFrame({"side": ["buy"] * 10, "sim_price": [1.0] * 10, "slippage_bps": [5.0] * 10,
                        "practice_vs_open_bps": [30.0] * 10, "sim_vs_open_bps": [20.0] * 10,
                        "fee_gap_bps": [0.0] * 10, "practice_fee_bps": [50.0] * 10,
                        "practice_delay_h": [13.5] * 10})
    out = summarize_gap(gap, CostModel.uk())
    assert out["suggested_slippage_bps"] == 20.0 and out["median_delay_h"] == 13.5
    assert summarize_gap(gap.iloc[:3], CostModel.uk()).get("suggested_slippage_bps") is None


def test_book_page_and_cli_show_fill_gap(tmp_path, capsys):
    from silo.cli import main
    from silo.web import book_page

    book, fake, sent = _sent_book(tmp_path)
    assert "No Practice fills logged yet" in book_page(tmp_path / "paper", book.name, "tok")
    o = sent[0]
    fake.history = [item(o["order_id"], o["ticker"], o["quantity"], 100.0)]
    collect_fills(book, client(fake))
    html = book_page(tmp_path / "paper", book.name, "tok")
    assert "Practice fills against the simulator" in html and "1 Practice fill(s)" in html
    import silo.brokers.trading212 as t212

    orig = t212.Trading212Demo.__init__
    t212.Trading212Demo.__init__ = lambda self, *a, **k: orig(self, "k", "s", transport=fake, sleep=lambda s: None)
    try:
        assert main(["--data-dir", str(tmp_path / "data"), "paper", "t212-fills", book.name,
                     "--paper-dir", str(tmp_path / "paper")]) == 0
    finally:
        t212.Trading212Demo.__init__ = orig
    out = capsys.readouterr().out
    assert "0 new Practice fill(s) logged" in out and "Research only" in out
    assert (book.dir / "fill_gap.csv").exists() and json.loads((book.dir / "fill_gap.json").read_text())["fills"] == 1
