import pytest

from silo.data.sources import synthetic_bars
from silo.data.vault import normalize_bars


@pytest.fixture
def tables():
    symbols = ["AAA", "BBB", "CCC"]
    return {s: normalize_bars(synthetic_bars(n_days=700, seed=i)) for i, s in enumerate(symbols)}


@pytest.fixture(autouse=True)
def _isolated_config(tmp_path, monkeypatch):
    """Never read or write the real ~/.config/silo during tests."""
    monkeypatch.setenv("SILO_CONFIG_DIR", str(tmp_path / "silo-config"))
    monkeypatch.delenv("T212_DEMO_API_KEY", raising=False)
    monkeypatch.delenv("T212_DEMO_API_SECRET", raising=False)


@pytest.fixture(autouse=True)
def _offline_model(tmp_path, monkeypatch):
    """AI strategies use the offline rulebot in tests and log to a temp folder."""
    monkeypatch.setenv("SILO_LLM", "rulebot")
    monkeypatch.setenv("SILO_LLM_LOG", str(tmp_path / "llm_log"))
