import numpy as np
import pandas as pd

from silo.agents.brief import HEADER, build_brief
from silo.agents.rulebook import RULEBOOK, CoreDesk
from silo.firewall import PointInTimeView
from silo.strategies.core_zones import (
    Bars,
    CoreZones,
    candle_kinds,
    decision_matrix,
    find_zones,
    resample,
    trend,
)


def bars(rows):
    """rows of (open, high, low, close)."""
    a = np.array(rows, dtype=float)
    return Bars(a[:, 0], a[:, 1], a[:, 2], a[:, 3], pd.RangeIndex(len(a)))


# drop (leg-in), two small base candles, a strong rally (leg-out), then drift above
DBR = [
    (110, 111, 104, 105),  # drop leg
    (105, 105.5, 100, 101),  # drop leg (leg-in)
    (101, 102, 99.5, 101.2),  # base
    (101.2, 102.5, 100.5, 101.5),  # base
    (101.5, 106, 101.4, 105.8),  # rally leg (leg-out)
    (105.8, 110, 105.6, 109.8),  # rally leg
    (109.8, 113, 109.5, 112.8),  # rally leg
    (112.8, 113.5, 111, 112),
    (112, 113, 110.5, 111),
]


def test_candles_are_legs_or_bases():
    k = candle_kinds(bars(DBR))
    assert list(k[:7]) == [-1, -1, 0, 0, 1, 1, 1]


def test_drop_base_rally_zone_lines_and_scores():
    zones = [z for z in find_zones(bars(DBR)) if z.kind == "demand"]
    assert len(zones) == 1
    z = zones[0]
    assert z.formation == "DBR"
    assert z.proximal == 101.5  # highest body in the base
    assert z.distal == 99.5  # lowest low of the whole drop-base-rally
    assert z.n_base == 2 and z.time == 1.0
    assert z.move_out >= 2  # 101.5 -> 113 is far more than twice the 2-point zone
    assert z.freshness == 2.0 and not z.violated


def test_freshness_and_violation():
    tested = DBR + [(111, 111, 100.8, 102), (102, 108, 101.9, 107.5)]  # dips 35% into the zone, then leaves
    z = [z for z in find_zones(bars(tested)) if z.kind == "demand"][0]
    assert 0 < z.penetration <= 0.5 and z.freshness == 1.0
    broken = DBR + [(111, 111, 98, 99)]  # closes below the distal line
    assert [z for z in find_zones(bars(broken)) if z.kind == "demand"][0].violated


def test_current_visit_does_not_spoil_freshness():
    visiting = DBR + [(111, 111, 101, 101.3)]  # price is in the zone right now: this is the entry
    z = [z for z in find_zones(bars(visiting)) if z.kind == "demand"][0]
    assert z.in_zone_now and z.freshness == 2.0


def test_trend_reads_higher_highs_and_lows():
    up, down = [], []
    p = 100.0
    for i in range(60):
        step = 3 if (i // 4) % 2 == 0 else -1.5  # impulse up, shallower correction
        up.append((p, p + 1 + max(step, 0), p - 1 + min(step, 0), p + step))
        p += step
    q = 300.0
    for i in range(60):
        step = -3 if (i // 4) % 2 == 0 else 1.5
        down.append((q, q + 1 + max(step, 0), q - 1 + min(step, 0), q + step))
        q += step
    assert trend(bars(up)).direction == 1
    assert trend(bars(down)).direction == -1


def test_decision_matrix_for_longs():
    assert decision_matrix(0.1, 1) == "long" and decision_matrix(0.1, -1) == "long_if_5to1"
    assert decision_matrix(0.5, 0) == "long" and decision_matrix(0.5, -1) == "no_action"
    assert decision_matrix(0.9, 1) == "long_if_5to1" and decision_matrix(0.9, 0) == "no_action"


def test_resample_weeks_and_months():
    idx = pd.bdate_range("2024-01-01", periods=30)
    df = pd.DataFrame({"open": np.arange(30.0), "high": np.arange(30.0) + 1, "low": np.arange(30.0) - 1,
                       "close": np.arange(30.0) + 0.5}, index=idx)
    w = resample(df, "W")
    assert len(w) == 6 and w["open"].iloc[0] == 0 and w["close"].iloc[0] == 4.5 and w["high"].iloc[0] == 5
    m = resample(df, "M")
    assert len(m) == 2 and m["low"].iloc[1] == 23 - 1


def test_core_zones_risk_rules(tables):
    from silo.engine import run_backtest

    s = CoreZones(list(tables))
    run_backtest(s, tables)
    buys = [x for x in s.log if x["action"] == "buy"]
    for b in buys:
        assert b["stop"] < b["entry"]
        assert (b["target"] - b["entry"]) / (b["entry"] - b["stop"]) >= 3 - 1e-9


def test_core_desk_keeps_original_desk_prompts_unchanged(tables):
    syms = list(tables)
    view = PointInTimeView(tables, tables["AAA"]["available_at"].iloc[-1])
    plain = build_brief(view, syms)
    assert plain.header == HEADER and len(next(iter(plain.rows.values()))) == 9
    core = CoreDesk(syms).brief(view)
    assert core.header.startswith(HEADER + " | zone score")
    assert all(len(v) == 15 for v in core.rows.values())
    assert "demand zone" in RULEBOOK


# -- trade management: daily stops, half off at 1:2, trail the rest


def _scripted(keep, on_day):
    """Buys AAA on the first decision; manage() asks for a cut on a chosen day."""
    from silo.strategies.base import Strategy

    class Scripted(Strategy):
        name = "scripted"

        def reset(self):
            self.n = 0
            self.bought = False

        def decide(self, view):
            if not self.bought:
                self.bought = True
                return {"AAA": 0.1}
            return None

        def manage(self, view):
            self.n += 1
            return {"AAA": keep} if self.n == on_day else None

    return Scripted(["AAA"])


def test_daily_stop_overrides_min_hold_and_weekly_rebalance(tables):
    from silo.constraints import Constraints
    from silo.engine import run_backtest

    c = Constraints.position_trading(liquidity=None)
    full = run_backtest(_scripted(0.0, 8), tables, constraints=c)
    sells = full.fills[full.fills["qty"] < 0]
    assert len(sells) == 1  # sold on day 9, well inside the 20-day minimum hold
    assert full.decisions[-1]["exits"] == {"AAA": 0.0}
    half = run_backtest(_scripted(0.5, 8), tables, constraints=c)
    buy = half.fills[half.fills["qty"] > 0]["qty"].sum()
    sold = -half.fills[half.fills["qty"] < 0]["qty"].sum()
    assert abs(sold / buy - 0.5) < 1e-9
    off = run_backtest(_scripted(0.0, 8), tables, constraints=Constraints.position_trading(
        liquidity=None, daily_exit_checks=False))
    # checked on rebalance days only: the same cut comes weeks later
    assert off.fills[off.fills["qty"] < 0]["date"].iloc[0] > sells["date"].iloc[0] + pd.Timedelta(days=30)


def _tables_from(rows, start="2020-01-01"):
    from silo.data.vault import normalize_bars

    idx = pd.bdate_range(start, periods=len(rows))
    a = np.array(rows, dtype=float)
    df = pd.DataFrame({"open": a[:, 0], "high": a[:, 1], "low": a[:, 2], "close": a[:, 3],
                       "volume": 1e7}, index=idx)
    return {"AAA": normalize_bars(df)}


def _managed_with_trade(tables, entry=100.0, stop=95.0, day=300):
    from silo.strategies.core_zones import CoreManaged, Trade

    s = CoreManaged(["AAA"])
    s.reset()
    opened = tables["AAA"].index[day - 1]
    s.trades["AAA"] = Trade("AAA", entry, stop, None, 0.1 / entry, 0.01, 99.0, opened,
                            entry + 2 * (entry - stop), False, opened)
    s.cash = 0.9
    return s


def _view_at(tables, i):
    return PointInTimeView(tables, tables["AAA"]["available_at"].iloc[i])


def test_managed_stop_sells_everything():
    rows = [(100, 101, 99, 100)] * 300 + [(100, 100.5, 94, 94.5)]
    t = _tables_from(rows)
    s = _managed_with_trade(t)
    assert s.manage(_view_at(t, 300)) == {"AAA": 0.0}
    assert "AAA" not in s.trades and s.log[-1]["why"] == "stop"


def test_managed_half_off_at_1_to_2_then_breakeven_stop():
    rows = [(100, 101, 99, 100)] * 300 + [(100, 110.5, 100, 109), (109, 109.5, 104, 104.5), (104, 104, 99, 99.5)]
    t = _tables_from(rows)
    s = _managed_with_trade(t)
    assert s.manage(_view_at(t, 300)) == {"AAA": 0.5}  # high 110.5 >= 100 + 2 x 5
    tr = s.trades["AAA"]
    assert tr.partial_done and tr.stop == 100.0 and abs(tr.units - 0.05 / 100) < 1e-12
    assert s.manage(_view_at(t, 301)) is None  # 104.5 is above the breakeven stop
    assert s.manage(_view_at(t, 302)) == {"AAA": 0.0}  # the other half stops at entry
    assert s.log[-1]["why"] == "trailing stop"


def test_core_managed_needs_1_to_4(tables):
    from silo.engine import run_backtest
    from silo.strategies.core_zones import CoreManaged

    s = CoreManaged(list(tables))
    run_backtest(s, tables)
    assert s.min_reward_risk == 4.0
    for b in (x for x in s.log if x["action"] == "buy"):
        assert b["stop"] < b["entry"] and b["target"] is None


def test_zones_report_shows_supply_and_a_verdict(tmp_path):
    from silo.cli import main

    out = tmp_path / "zones.html"
    assert main(["zones", "--demo", "--out", str(out)]) == 0
    page = out.read_text()
    assert "rect class='supply'" in page and "Research only" in page
    from silo.lab.zones_report import zone_row

    rows = _tables_from(DBR * 40)
    r = zone_row("AAA", rows["AAA"])
    assert r is not None and r.verdict in ("in buy zone", "wait for pull-back", "supply too close", "no trade")


def test_supply_wicked_through_is_used_up():
    from silo.strategies.core_zones import Zone

    z = Zone("supply", "RBD", 100.0, 102.0, 0, 1, 2, 1, 3.0, True, penetration=1.4)
    assert z.used_up and not z.violated and z.freshness == 0.0
    assert not Zone("supply", "RBD", 100.0, 102.0, 0, 1, 2, 1, 3.0, True, penetration=0.4).used_up
