import pytest

from silo.data.sources import synthetic_bars
from silo.data.vault import normalize_bars


@pytest.fixture
def tables():
    symbols = ["AAA", "BBB", "CCC"]
    return {s: normalize_bars(synthetic_bars(n_days=700, seed=i)) for i, s in enumerate(symbols)}
