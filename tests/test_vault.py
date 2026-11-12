from datetime import datetime, timezone

import pandas as pd

from silo.cli import main
from silo.data.sources import synthetic_bars
from silo.data.vault import DataVault, fix_pence_glitches, normalize_bars
from silo.markets import UK, US


def test_vault_snapshots_are_kept_and_merged(tmp_path):
    vault = DataVault(tmp_path)
    bars = synthetic_bars(n_days=50)
    vault.put_bars("spy", bars.iloc[:30], fetched_at=datetime(2026, 1, 1, tzinfo=timezone.utc))
    revised = bars.iloc[20:].copy()
    revised.iloc[0, revised.columns.get_loc("close")] += 1.0  # vendor revises one row
    stored = vault.put_bars("SPY", revised, fetched_at=datetime(2026, 1, 2, tzinfo=timezone.utc))
    assert len(stored) == 50
    assert len(vault.snapshots("SPY")) == 2  # nothing overwritten
    assert vault.symbols() == ["SPY"]
    assert stored.loc[revised.index[0], "close"] == revised.iloc[0]["close"]  # newest wins
    assert len(vault.revisions("SPY")) == 1


def test_close_times_follow_each_market_and_dst():
    winter, summer = pd.Timestamp("2025-01-15"), pd.Timestamp("2025-07-15")
    uk = normalize_bars(synthetic_bars(n_days=1, start="2025-01-15"), UK)
    assert uk["available_at"].iloc[0] == pd.Timestamp("2025-01-15 16:35", tz="UTC")
    uk_summer = normalize_bars(synthetic_bars(n_days=1, start="2025-07-15"), UK)
    assert uk_summer["available_at"].iloc[0] == pd.Timestamp("2025-07-15 15:35", tz="UTC")
    us = normalize_bars(synthetic_bars(n_days=1, start="2025-01-15"), US)
    assert us["available_at"].iloc[0] == pd.Timestamp("2025-01-15 21:05", tz="UTC")
    assert winter < summer


def test_pence_converted_to_pounds_and_glitches_fixed():
    raw = synthetic_bars(n_days=30, start_price=2500.0)  # quoted in pence
    raw["dividends"] = 0.0
    raw.iloc[10, raw.columns.get_loc("dividends")] = 12.0  # 12p dividend
    glitch = raw.copy()
    glitch.iloc[15, :4] /= 100  # one row reported in pounds
    fixed, n = fix_pence_glitches(glitch.rename(columns=str.lower))
    assert n == 1
    out = normalize_bars(glitch, UK, pence=True)
    assert 15 < out["close"].median() < 35
    assert out["dividend"].iloc[10] == 0.12
    assert abs(out["close"].iloc[15] / out["close"].iloc[14] - 1) < 0.2


def test_cli_demo_runs(capsys):
    assert main(["demo"]) == 0
    out = capsys.readouterr().out
    assert "buy_and_hold" in out and "not financial advice" in out
