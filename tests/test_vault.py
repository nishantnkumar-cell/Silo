import pandas as pd

from silo.cli import main
from silo.data.sources import synthetic_bars
from silo.data.vault import DataVault


def test_vault_roundtrip_and_merge(tmp_path):
    vault = DataVault(tmp_path)
    bars = synthetic_bars(n_days=50)
    vault.put_bars("spy", bars.iloc[:30])
    stored = vault.put_bars("SPY", bars.iloc[20:])
    assert len(stored) == 50
    assert vault.symbols() == ["SPY"]
    first = stored.index[0]
    assert stored.loc[first, "available_at"] == pd.Timestamp(first).tz_localize("UTC") + pd.Timedelta(hours=21)


def test_cli_demo_runs(capsys):
    assert main(["demo"]) == 0
    out = capsys.readouterr().out
    assert "buy_and_hold" in out and "not financial advice" in out
