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
