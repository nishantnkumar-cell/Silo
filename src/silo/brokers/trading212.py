"""Trading 212 **Practice** account client. Paper money only.

Trading 212's public API (beta, v0) serves the Practice account from
``demo.trading212.com`` and real money from ``live.trading212.com``. This
client is hard-wired to the demo host, re-checks it against the paper
allowlist on every request, and refuses to start if the live host appears
anywhere in its configuration.

Credentials: create an API key while the Trading 212 app is switched to
**Practice**, then add it on the ``silo web`` settings page (stored in
``~/.config/silo/credentials.json``, owner-only) or set
``T212_DEMO_API_KEY`` and ``T212_DEMO_API_SECRET``.
Never put a real-money key here.

Docs: https://docs.trading212.com/api
"""

from __future__ import annotations

import base64
import json
import math
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

from silo.safety import LiveTradingRefused, check_order_endpoint

DEMO_BASE = "https://demo.trading212.com/api/v0"

# Minimum seconds between calls, from the documented per-account limits.
MIN_INTERVAL = {
    "/equity/account/summary": 5.0,
    "/equity/metadata/instruments": 50.0,
    "/equity/positions": 1.0,
    "/equity/orders": 1.0,
    "/equity/orders/market": 1.3,
}

Transport = Callable[[str, str, dict, bytes | None], tuple[int, dict, Any]]


def _urllib_transport(method: str, url: str, headers: dict, body: bytes | None):  # pragma: no cover - network
    req = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 - host checked by allowlist
            return resp.status, dict(resp.headers), json.loads(resp.read() or b"null")
    except urllib.error.HTTPError as err:
        payload = err.read()
        try:
            data = json.loads(payload or b"null")
        except ValueError:
            data = payload.decode(errors="replace")
        return err.code, dict(err.headers), data


class Trading212Error(RuntimeError):
    pass


class Trading212Demo:
    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        transport: Transport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.base = check_order_endpoint(DEMO_BASE)
        key = api_key or os.environ.get("T212_DEMO_API_KEY")
        secret = api_secret or os.environ.get("T212_DEMO_API_SECRET")
        if not key or not secret:
            from silo.credentials import load_t212

            key, secret = load_t212() or (None, None)
        if not key or not secret:
            raise Trading212Error(
                "no Practice API key: add one on the `silo web` settings page, "
                "or set T212_DEMO_API_KEY and T212_DEMO_API_SECRET"
            )
        self._auth = "Basic " + base64.b64encode(f"{key}:{secret}".encode()).decode()
        self._transport = transport or _urllib_transport
        self._sleep, self._clock = sleep, clock
        self._last: dict[str, float] = {}

    def _request(self, method: str, path: str, body: dict | None = None) -> Any:
        url = self.base + path
        check_order_endpoint(url)
        if "live.trading212.com" in url:
            raise LiveTradingRefused("live Trading 212 host refused")
        key = path.split("?")[0]
        wait = MIN_INTERVAL.get(key, 1.0) - (self._clock() - self._last.get(key, -1e9))
        if wait > 0:
            self._sleep(wait)
        headers = {"Authorization": self._auth, "Content-Type": "application/json"}
        data = json.dumps(body).encode() if body is not None else None
        for attempt in range(2):
            status, resp_headers, payload = self._transport(method, url, headers, data)
            self._last[key] = self._clock()
            if status == 429 and attempt == 0 and method == "GET":
                reset = float(resp_headers.get("x-ratelimit-reset", 0) or 0)
                self._sleep(min(60.0, max(1.0, reset - time.time())))
                continue
            break
        if status >= 400:
            raise Trading212Error(f"{method} {path} -> {status}: {payload}")
        return payload

    # read-only
    def account_summary(self) -> dict:
        return self._request("GET", "/equity/account/summary")

    def total_value(self) -> tuple[float, str]:
        """Cash (free, reserved, in pies) plus investments, in account currency."""
        s = self.account_summary()
        cash = s.get("cash", {})
        total = sum(float(cash.get(k) or 0) for k in ("availableToTrade", "reservedForOrders", "inPies"))
        total += float((s.get("investments") or {}).get("currentValue") or 0)
        return total, s.get("currency", "")

    def positions(self) -> dict[str, float]:
        rows = self._request("GET", "/equity/positions") or []
        return {r["instrument"]["ticker"]: float(r["quantity"]) for r in rows}

    def instruments(self) -> list[dict]:
        return self._request("GET", "/equity/metadata/instruments") or []

    def active_orders(self) -> list[dict]:
        out = self._request("GET", "/equity/orders")
        return out.get("items", out) if isinstance(out, dict) else (out or [])

    # orders (Practice account only)
    def market_order(self, ticker: str, quantity: float) -> dict:
        """Positive quantity buys, negative sells. Not idempotent: callers must dedupe."""
        if quantity == 0 or not math.isfinite(quantity):
            raise ValueError("quantity must be a non-zero number")
        return self._request(
            "POST", "/equity/orders/market", {"ticker": ticker, "quantity": quantity, "extendedHours": False}
        )


def propose_mapping(symbols: list[str], instruments: list[dict]) -> dict[str, str | None]:
    """Match Yahoo ``.L`` symbols to Trading 212 tickers. Review before use.

    Prefers a London-currency (GBX/GBP) stock whose short name equals the
    Yahoo base symbol. Ambiguous or missing matches map to None.
    """
    london = [i for i in instruments if str(i.get("currencyCode", "")).upper() in {"GBX", "GBP"}]
    out: dict[str, str | None] = {}
    for sym in symbols:
        base = sym.upper().removesuffix(".L").replace("-", "").replace(".", "")
        hits = [i["ticker"] for i in london if str(i.get("shortName", "")).upper().replace(".", "") == base]
        if len(hits) != 1:
            prefix = [i["ticker"] for i in london if i["ticker"].upper().startswith(base) and i["ticker"].endswith("_EQ")]
            hits = prefix if len(prefix) == 1 else hits
        out[sym] = hits[0] if len(hits) == 1 else None
    return out


def mirror_orders(
    weights: dict[str, float],
    prices: dict[str, float],
    mapping: dict[str, str],
    held: dict[str, float],
    total_value: float,
    fractional: bool = False,
) -> list[tuple[str, float]]:
    """Orders that move the Practice account toward ``weights``. Sells first."""
    orders = []
    tickers = {mapping[s]: s for s in weights if s in mapping} | {t: s for s, t in mapping.items() if t in held}
    for ticker, symbol in tickers.items():
        target = weights.get(symbol, 0.0) * total_value / prices[symbol] if symbol in prices else 0.0
        target = target if fractional else math.floor(target)
        delta = target - held.get(ticker, 0.0)
        if weights.get(symbol, 0.0) == 0:
            delta = -held.get(ticker, 0.0)  # full exit sells everything, fractions included
        if not fractional and delta > 0:
            delta = math.floor(delta)
        if abs(delta) > 1e-9:
            orders.append((ticker, round(delta, 6)))
    return sorted(orders, key=lambda o: o[1])
