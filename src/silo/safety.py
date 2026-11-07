"""Hard guardrails: Silo must never touch a live brokerage or real money.

These checks run at import time and whenever an endpoint is configured.
They are deliberately strict: when in doubt, refuse.
"""

from __future__ import annotations

import os
from urllib.parse import urlparse


class LiveTradingRefused(RuntimeError):
    """Raised when anything points Silo at a live brokerage."""


# The only remote order endpoints Silo may ever talk to: paper sandboxes.
PAPER_HOSTS = frozenset({"paper-api.alpaca.markets"})

# Known live order-routing hosts. Any match is refused outright.
LIVE_HOSTS = frozenset(
    {
        "api.alpaca.markets",
        "api.tradier.com",
        "api.schwabapi.com",
        "api.tdameritrade.com",
        "api.robinhood.com",
        "api.etrade.com",
        "api.tastyworks.com",
        "api.tastytrade.com",
        "api.binance.com",
        "api.coinbase.com",
        "api.kraken.com",
    }
)

# Environment variables that suggest live credentials are present.
_LIVE_ENV_MARKERS = ("LIVE_API_KEY", "LIVE_SECRET", "LIVE_TRADING")


def check_order_endpoint(url: str) -> str:
    """Return ``url`` if it is an allowed paper endpoint, else refuse."""
    host = (urlparse(url).hostname or "").lower()
    if host in LIVE_HOSTS:
        raise LiveTradingRefused(f"{host} is a live brokerage host; Silo is paper-only.")
    if host not in PAPER_HOSTS:
        raise LiveTradingRefused(
            f"{host or url!r} is not an allowlisted paper endpoint. Allowed: {sorted(PAPER_HOSTS)}"
        )
    return url


def check_environment(environ: dict[str, str] | None = None) -> None:
    """Refuse to run when live-trading credentials or flags are present."""
    env = os.environ if environ is None else environ
    for name, value in env.items():
        upper = name.upper()
        if any(marker in upper for marker in _LIVE_ENV_MARKERS) and value:
            raise LiveTradingRefused(f"Environment variable {name} looks like a live-trading setting.")
        if upper == "APCA_API_BASE_URL" and value:
            check_order_endpoint(value)
