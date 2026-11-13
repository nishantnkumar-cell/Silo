"""Daily-bar books for silver ETFs and crypto."""

import json

import pandas as pd
import pytest

from silo.broker import CostModel, SimBroker
from silo.cli import main
from silo.constraints import Constraints
from silo.data.sources import crypto_volume_to_units, drop_unfinished, synthetic_bars
from silo.data.vault import DataVault, normalize_bars
from silo.engine import run_backtest
from silo.liquidity import Liquidity
from silo.markets import CRYPTO, market_for, market_for_symbols
from silo.metrics import periods_per_year, summarize
from silo.paper import PaperBook
from silo.strategies import REGISTRY
from silo.universe import CRYPTO_PAIRS, SILVER_ETFS


def crypto_tables(n=900):
    out = {}
    for i, s in enumerate(CRYPTO_PAIRS):
        raw = synthetic_bars(n_days=n, seed=10 + i, every_day=True, vol=0.035, start_price=20_000)
        out[s] = normalize_bars(raw, CRYPTO)
    return out


def test_crypto_market_rules():
    assert market_for("BTC-USD").code == "CRYPTO"
    assert market_for("SLV").code == "US"
    assert market_for_symbols(SILVER_ETFS).code == "US"
    with pytest.raises(ValueError):
        market_for_symbols(["SLV", "BTC-USD"])
    df = normalize_bars(synthetic_bars(n_days=5, every_day=True), CRYPTO)
    # a UTC day's bar is final only after midnight: 00:05 UTC the next day
    assert df["available_at"].iloc[0] == pd.Timestamp(df.index[0], tz="UTC") + pd.Timedelta(days=1, minutes=5)


def test_unfinished_bars_are_dropped():
    df = normalize_bars(synthetic_bars(n_days=5, every_day=True), CRYPTO)
    now = df["available_at"].iloc[3] - pd.Timedelta(minutes=1)
    assert len(drop_unfinished(df, now)) == 3


def test_crypto_volume_is_converted_to_coins():
    df = normalize_bars(synthetic_bars(n_days=5, every_day=True, start_price=50_000), CRYPTO)
    units = crypto_volume_to_units(df)
    assert (units["volume"] * units["close"]).round(6).equals(df["volume"].round(6))


def test_crypto_and_fx_costs():
    c = CostModel.crypto()
    assert c.fees(1.0, 10_000) == pytest.approx(25.0)  # 0.25% of 10k, no stamp duty
    assert c.fees(-1.0, 10_000) == pytest.approx(25.0)
    fx = CostModel.us_from_uk()
    assert fx.fees(100, 30.0) == pytest.approx(1.0 + 0.0015 * 3000)
    assert CostModel.for_market("CRYPTO") == CostModel.crypto()
    assert c.scaled(2).spread_bps == 20.0 and c.scaled(2).commission_bps == 50.0


def test_buys_never_overdraw_with_percentage_fees():
    b = SimBroker(cash=10_000, costs=CostModel.crypto(), min_trade_value=1)
    b.target_weights({"BTC-USD": 1.0}, {"BTC-USD": 100.0}, pd.Timestamp("2024-01-01"))
    b.fill_at_open({"BTC-USD": 100.0}, pd.Timestamp("2024-01-02"))
    assert b.cash >= -1e-9


def test_daily_constraints_and_liquidity():
    c = Constraints.daily_trading(liquidity=Liquidity.for_market("CRYPTO"))
    assert c.rebalance == "daily" and c.max_weight == 0.5 and c.liquidity.min_value_traded == 1e9
    assert Constraints.position_trading().rebalance == "weekly"  # the default is unchanged


def test_periods_per_year():
    assert periods_per_year(crypto_tables()["BTC-USD"].index) == 365
    assert periods_per_year(pd.bdate_range("2020-01-01", periods=300)) == 252


def test_crypto_liquidity_blocks_thin_coins():
    tables = crypto_tables()
    thin = tables["ETH-USD"].copy()
    thin["volume"] = 10.0  # about $200k a day: far below the $1bn floor
    tables["ETH-USD"] = thin
    start = tables["BTC-USD"].index[100]  # after the 60-day volume window has filled
    res = run_backtest(REGISTRY["buy_and_hold"](CRYPTO_PAIRS), tables, start=start, costs=CostModel.crypto(),
                       constraints=Constraints.daily_trading(liquidity=Liquidity.for_market("CRYPTO")))
    assert set(res.fills["symbol"]) == {"BTC-USD"}


def test_tsmom_trades_daily_with_costs():
    tables = crypto_tables()
    res = run_backtest(REGISTRY["tsmom"](CRYPTO_PAIRS), tables, costs=CostModel.crypto(),
                       constraints=Constraints.daily_trading(liquidity=Liquidity.for_market("CRYPTO")))
    m = res.metrics()
    assert m["round_trips"] > 5 and m["fees"] > 0
    assert summarize(res.equity)["days"] == len(tables["BTC-USD"])


def test_daily_league_and_paper_skip_partial_days(tmp_path):
    vault = DataVault(tmp_path / "data")
    tables = crypto_tables(400)
    head = {s: df.iloc[:300] for s, df in tables.items()}
    for s, df in head.items():
        vault.put_bars(s, df, fetched_at=pd.Timestamp("2020-01-01", tz="UTC").to_pydatetime())
    rc = main(["--data-dir", str(tmp_path / "data"), "league", "new", "--universe", "crypto",
               "--style", "daily", "--prefix", "crypto", "--paper-dir", str(tmp_path / "paper")])
    assert rc == 0
    cfg = json.loads((tmp_path / "paper" / "crypto-tsmom" / "config.json").read_text())
    assert cfg["market"] == "CRYPTO" and cfg["constraints"]["rebalance"] == "daily"
    assert cfg["costs"]["commission_bps"] == 25.0
    for s, df in tables.items():
        vault.put_bars(s, df.iloc[300:], fetched_at=pd.Timestamp("2020-02-01", tz="UTC").to_pydatetime())
    book = PaperBook("crypto-tsmom", tmp_path / "paper")
    last = tables["BTC-USD"]["available_at"].iloc[-1]
    events = book.run(vault, now=last - pd.Timedelta(minutes=1))
    assert len(events) == 99  # the newest day is not final yet
    assert len(book.run(vault, now=last)) == 1


def test_daily_style_refuses_uk_shares():
    from argparse import Namespace

    from silo.cli import _rules

    args = Namespace(costs="auto", cost_scale=None, style="daily", rebalance=None, min_hold=None,
                     min_price=None, min_value_traded=None)
    with pytest.raises(SystemExit):
        _rules(args, ["AZN.L"])
    silver = _rules(args, SILVER_ETFS)
    assert silver["costs"].fx_bps == 15.0 and silver["constraints"].liquidity.min_value_traded == 2e7


def test_lab_cli_daily_universe(tmp_path, capsys):
    vault = DataVault(tmp_path / "data")
    for s, df in crypto_tables(1200).items():
        vault.put_bars(s, df)
    rc = main(["--data-dir", str(tmp_path / "data"), "lab", "--universe", "crypto", "--style", "daily",
               "--strategy", "tsmom", "--random", "3", "--cost-scale", "2", "--out", str(tmp_path / "r")])
    assert rc == 0 and (tmp_path / "r" / "tsmom.html").exists()
    assert "Research only" in capsys.readouterr().out


def test_locked_positions_that_drifted_never_lever_the_book():
    from silo.constraints import apply_constraints

    current = {"A": 0.52, "B": 0.48}  # both just bought; A drifted up
    out = apply_constraints({"A": 0.5, "B": 0.5}, current, {"A": 0, "B": 0}, Constraints.daily_trading())
    assert sum(out.values()) <= 1.0 + 1e-12
    assert out["A"] >= 0.5 and out["B"] >= 0.48 - 1e-12


def test_buy_and_hold_invests_when_the_run_starts_on_the_first_bar():
    """Liquidity needs 60 days of volume, so a first-day buy used to be refused forever."""
    tables = crypto_tables(400)
    late = tables["ETH-USD"].iloc[100:]  # one coin starts trading later, like ETH after BTC
    tables["ETH-USD"] = late
    res = run_backtest(REGISTRY["buy_and_hold"](CRYPTO_PAIRS), tables, costs=CostModel.crypto(),
                       constraints=Constraints.daily_trading(liquidity=Liquidity.for_market("CRYPTO")))
    # only what trades at the start is bought (a later listing is not added), once it has 60 bars
    assert set(res.fills["symbol"]) == {"BTC-USD"}
    assert res.fills["date"].min() > tables["BTC-USD"].index[59]
    assert res.equity.iloc[-1] != res.equity.iloc[0]
