import http.client
import json
import os
import stat
import threading
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode

import pytest

from silo import credentials
from silo.brokers.trading212 import Trading212Demo
from silo.data.vault import DataVault
from silo.paper import PaperBook
from silo.web import Dashboard, NotLocal, check_loopback
from tests.test_trading212 import FakeT212
from tests.test_uk_and_paper import uk_tables


@pytest.fixture
def book_dirs(tmp_path):
    syms, full = uk_tables(n_days=640)
    vault = DataVault(tmp_path / "data")
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for s, df in full.items():
        vault.put_bars(s, df.iloc[:500], fetched_at=t0)
    book = PaperBook("uk-trend", tmp_path / "paper")
    book.create("uk_trend", syms, vault, {"lookback": 126, "top_k": 2})
    for s, df in full.items():
        vault.put_bars(s, df, fetched_at=t0 + timedelta(days=1))
    book.run(vault)
    (book.dir / "t212_map.json").write_text(json.dumps({s: s.replace(".L", "l_EQ") for s in syms}))
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "bollinger.html").write_text("<html><body>report body</body></html>")
    (tmp_path / "secret.html").write_text("outside")
    return tmp_path


class Running:
    def __init__(self, dash):
        self.dash = dash
        self.srv = dash.make_server()
        self.thread = threading.Thread(target=self.srv.serve_forever, daemon=True)
        self.thread.start()

    def request(self, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.dash.port, timeout=10)
        h = {"Host": f"127.0.0.1:{self.dash.port}", **(headers or {})}
        if body is not None:
            body = urlencode(body)
            h.setdefault("Content-Type", "application/x-www-form-urlencoded")
        conn.request(method, path, body=body, headers=h)
        resp = conn.getresponse()
        return resp.status, dict(resp.getheaders()), resp.read().decode()

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


@pytest.fixture
def app(book_dirs):
    fake = FakeT212(positions={"T0l_EQ": 3})
    calls = []

    def verify(key, secret):
        calls.append((key, secret))
        if key == "livekey123":
            raise credentials.CredentialError("Practice server did not accept this key")
        return "GBP"

    dash = Dashboard(book_dirs / "paper", book_dirs / "reports", port=0, verify=verify,
                     client_factory=lambda: Trading212Demo("k", "s", transport=fake, sleep=lambda s: None))
    run = Running(dash)
    run.verify_calls, run.fake = calls, fake
    yield run
    run.close()


def test_refuses_non_loopback_bind():
    assert check_loopback("127.0.0.1") and check_loopback("::1") and check_loopback("localhost")
    for host in ("0.0.0.0", "192.168.1.5", "example.com", ""):
        with pytest.raises(NotLocal):
            check_loopback(host)


def test_pages_show_book_against_buy_and_hold(app):
    status, headers, body = app.request("GET", "/")
    assert status == 200 and "uk-trend" in body and "Buy-and-hold return" in body and "Research only" in body
    assert "default-src 'none'" in headers["Content-Security-Policy"]
    # no-referrer would make browsers send "Origin: null" on form posts, which the server refuses
    assert headers["Referrer-Policy"] == "same-origin"
    status, _, body = app.request("GET", "/book/uk-trend")
    assert status == 200
    for text in ("Buy-and-hold copy", "<svg", "Trade journal", "Positions", "t212_map.json"):
        assert text in body
    assert app.request("GET", "/book/nope")[0] == 404
    assert app.request("GET", "/book/..%2F..%2Fetc")[0] == 404


def test_reports_served_only_from_reports_dir(app):
    status, headers, body = app.request("GET", "/reports/bollinger.html")
    assert status == 200 and "report body" in body
    assert "connect-src" not in headers["Content-Security-Policy"]  # default-src 'none' blocks network
    assert app.request("GET", "/reports/..%2Fsecret.html")[0] == 404
    assert app.request("GET", "/reports/../secret.html")[0] == 404
    assert "bollinger.html" in app.request("GET", "/reports")[2]


def test_rejects_foreign_host_header(app):
    status, _, _ = app.request("GET", "/", headers={"Host": "evil.example:8212"})
    assert status == 403


def test_settings_require_token_and_never_echo_secret(app):
    status, _, _ = app.request("POST", "/settings", {"api_key": "abcdefgh12", "api_secret": "s3cretvalue99"})
    assert status == 403 and app.verify_calls == []
    form = {"csrf": app.dash.token, "api_key": "abcdefgh12", "api_secret": "s3cretvalue99"}
    status, _, _ = app.request("POST", "/settings", form, headers={"Origin": "http://evil.example"})
    assert status == 403
    status, headers, _ = app.request("POST", "/settings", form)
    assert status == 303 and headers["Location"] == "/settings?done=saved"
    assert credentials.load_t212() == ("abcdefgh12", "s3cretvalue99")
    _, _, body = app.request("GET", "/settings?done=saved")
    assert "s3cretvalue99" not in body and "abcdefgh12" not in body and "…gh12" in body
    path = credentials.config_dir() / "credentials.json"
    if os.name == "posix":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_key_rejected_by_practice_server_is_not_saved(app):
    form = {"csrf": app.dash.token, "api_key": "livekey123", "api_secret": "whatever123"}
    status, _, body = app.request("POST", "/settings", form)
    assert status == 400 and "did not accept" in body
    assert credentials.load_t212() is None


def test_verify_uses_demo_host_and_maps_401():
    seen = []

    def transport(method, url, headers, body):
        seen.append(url)
        return 401, {}, {"code": "Unauthorized"}

    with pytest.raises(credentials.CredentialError, match="Practice"):
        credentials.verify_with_practice_server("abcdefgh12", "s3cretvalue99", transport=transport)
    assert seen and all(u.startswith("https://demo.trading212.com/") for u in seen)


def test_client_falls_back_to_stored_key():
    credentials.save_t212("abcdefgh12", "s3cretvalue99", verify=lambda k, s: "GBP")
    fake = FakeT212()
    Trading212Demo(transport=fake, sleep=lambda s: None).account_summary()
    credentials.clear_t212()
    with pytest.raises(Exception, match="Practice API key"):
        Trading212Demo(transport=fake)


def test_t212_check_is_read_only(app):
    status, _, body = app.request("POST", "/book/uk-trend/t212-check", {"csrf": app.dash.token})
    assert status == 200 and "Practice shares" in body
    assert all(method == "GET" for method, _, _ in app.fake.calls)
