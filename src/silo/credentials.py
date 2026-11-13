"""Local store for Trading 212 **Practice** API keys. Never committed.

Keys live outside the repository, in ``~/.config/silo/credentials.json``
(or ``$SILO_CONFIG_DIR/credentials.json``). The folder is created with
owner-only permissions (0700) and the file with 0600, written atomically.

A key is only stored after Trading 212's Practice server
(``demo.trading212.com``) has accepted it. A real-money key is rejected by
that server, so it is never saved, and Silo has no code path that would send
any key to the live server.

Environment variables ``T212_DEMO_API_KEY`` / ``T212_DEMO_API_SECRET`` still
work and take precedence over the stored key.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

_TOKEN = re.compile(r"^[A-Za-z0-9_\-.+/=]{8,256}$")


class CredentialError(ValueError):
    pass


def config_dir() -> Path:
    env = os.environ.get("SILO_CONFIG_DIR")
    return Path(env).expanduser() if env else Path.home() / ".config" / "silo"


def _path() -> Path:
    return config_dir() / "credentials.json"


def _read() -> dict:
    path = _path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def _write(data: dict) -> None:
    folder = config_dir()
    folder.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(folder, 0o700)
    except OSError:  # pragma: no cover - e.g. Windows
        pass
    tmp = folder / "credentials.json.tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, _path())
    try:
        os.chmod(_path(), 0o600)
    except OSError:  # pragma: no cover
        pass


def validate_format(key: str, secret: str) -> tuple[str, str]:
    key, secret = (key or "").strip(), (secret or "").strip()
    if not key or not secret:
        raise CredentialError("Enter both the API key and the API secret.")
    for label, value in (("key", key), ("secret", secret)):
        if not _TOKEN.match(value):
            raise CredentialError(f"The API {label} does not look like a Trading 212 credential.")
    return key, secret


def load_t212() -> tuple[str, str] | None:
    """Stored Practice key and secret, or None."""
    entry = _read().get("trading212_demo") or {}
    if entry.get("api_key") and entry.get("api_secret"):
        return entry["api_key"], entry["api_secret"]
    return None


def t212_status() -> dict:
    """What the settings page may show: never the secret, only the key's last 4 characters."""
    env = bool(os.environ.get("T212_DEMO_API_KEY") and os.environ.get("T212_DEMO_API_SECRET"))
    entry = _read().get("trading212_demo") or {}
    stored = bool(entry.get("api_key") and entry.get("api_secret"))
    return {
        "source": "environment" if env else ("file" if stored else None),
        "stored": stored,
        "key_hint": ("…" + entry["api_key"][-4:]) if stored else None,
        "verified_at": entry.get("verified_at"),
        "account_currency": entry.get("account_currency"),
        "path": str(_path()),
    }


def save_t212(key: str, secret: str, verify=None) -> dict:
    """Check the key against the Practice server, then store it.

    ``verify(key, secret)`` must return the Practice account currency or raise.
    By default it calls ``demo.trading212.com``; tests pass a fake.
    """
    key, secret = validate_format(key, secret)
    verify = verify or verify_with_practice_server
    currency = verify(key, secret)
    data = _read()
    data["trading212_demo"] = {
        "api_key": key,
        "api_secret": secret,
        "verified_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "account_currency": currency,
        "server": "demo.trading212.com",
    }
    _write(data)
    return t212_status()


def clear_t212() -> None:
    data = _read()
    if data.pop("trading212_demo", None) is not None:
        _write(data)


def verify_with_practice_server(key: str, secret: str, transport=None) -> str:
    """Ask demo.trading212.com for the account summary. A live key fails here."""
    from silo.brokers.trading212 import Trading212Demo, Trading212Error

    client = Trading212Demo(key, secret, transport=transport, sleep=lambda s: None)
    try:
        summary = client.account_summary()
    except Trading212Error as exc:
        msg = str(exc)
        if "-> 401" in msg or "-> 403" in msg:
            raise CredentialError(
                "Trading 212's Practice server did not accept this key. Create the key while the app is "
                "switched to Practice mode. Real-money keys are never accepted or stored."
            ) from None
        raise CredentialError(f"Could not verify the key with the Practice server: {msg}") from None
    except OSError as exc:
        raise CredentialError(f"Could not reach demo.trading212.com: {exc}. Nothing was saved.") from None
    return str((summary or {}).get("currency", ""))
