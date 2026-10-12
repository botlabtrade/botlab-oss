import datetime as dt
import json
import os
import time

import pytest

from honest_backtest.cli import main
from honest_backtest.costs import CostModel, orderbook_slippage
from honest_backtest.pulse import (PulseMonitor, file_fresh, iso_duration_minutes, judge_task, task_interval,
                                   windows_tasks_check)

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE = os.path.join(HERE, "..", "examples", "sample_trades.csv")


# ----------------------------------------------------------------------------- costs
def test_round_trip_cost_and_r():
    cm = CostModel(taker_fee=0.00055, slippage=0.001, funding_8h=0.0001)
    assert cm.round_trip_cost("long") == pytest.approx(0.0031)
    assert cm.round_trip_cost("long", hold_hours=16) == pytest.approx(0.0033)
    assert cm.round_trip_cost("short", hold_hours=16) == pytest.approx(0.0029)
    # long 100 -> 102 with stop at 98: gross +1R, minus 0.31 % / 2 % = 0.155R
    assert cm.r_multiple("long", 100, 102, 98) == pytest.approx(1 - 0.155)
    assert cm.r_multiple("short", 100, 98, 102) == pytest.approx(1 - 0.155)
    assert cm.r_multiple("long", 100, 101, 100) is None
    with pytest.raises(ValueError):
        cm.round_trip_cost("sideways")


def test_tight_stops_are_eaten_by_costs():
    cm = CostModel()
    # same +1R gross move, 0.3 % stop (scalping) vs 3 % stop (swing)
    scalp = cm.r_multiple("long", 100, 100.3, 99.7)
    swing = cm.r_multiple("long", 100, 103, 97)
    assert scalp < 0 < swing


def test_orderbook_slippage():
    asks = [(100.1, 1.0), (100.2, 1.0), (100.5, 10.0)]
    bids = [(99.9, 1.0), (99.8, 1.0), (99.5, 10.0)]
    small = orderbook_slippage(asks, bids, notional=50)
    big = orderbook_slippage(asks, bids, notional=500)
    assert small == pytest.approx(0.001, rel=1e-3)
    assert big > small
    huge = orderbook_slippage(asks, bids, notional=10_000)
    assert huge > big                              # beyond the book -> penalty


# ----------------------------------------------------------------------------- pulse
def test_interval_parsing():
    assert iso_duration_minutes("PT5M") == 5
    assert iso_duration_minutes("PT1H30M") == 90
    assert iso_duration_minutes("P1D") == 1440
    assert iso_duration_minutes("") is None
    assert task_interval(["Time|PT15M||", "Daily|||"]) == 15
    assert task_interval(["Logon|||"]) == "loop"
    assert task_interval(["Event|||"]) is None


def test_judge_task():
    now = dt.datetime(2026, 1, 1, 12, 0, tzinfo=dt.timezone.utc)
    fresh = (now - dt.timedelta(minutes=10)).isoformat()
    stale = (now - dt.timedelta(minutes=90)).isoformat()
    assert judge_task(dict(n="a", s="Ready", r=0, t=fresh, trig=["Time|PT15M||"]), now) is None
    assert "last run" in judge_task(dict(n="a", s="Ready", r=0, t=stale, trig=["Time|PT15M||"]), now)
    assert "result code" in judge_task(dict(n="a", s="Ready", r=1, t=fresh, trig=["Time|PT15M||"]), now)
    assert "not running" in judge_task(dict(n="b", s="Ready", r=0, t=fresh, trig=["Logon|||"]), now)
    assert judge_task(dict(n="b", s="Running", r=267009, t=fresh, trig=["Logon|||"]), now) is None


def test_windows_tasks_check_with_fake_reader():
    now = dt.datetime.now().astimezone()
    fake = lambda: [dict(n="BotA", s="Ready", r=0, t=(now - dt.timedelta(hours=5)).isoformat(), trig=["Time|PT5M||"]),
                    dict(n="Other", s="Ready", r=0, t=now.isoformat(), trig=["Time|PT5M||"])]
    res = windows_tasks_check("^Bot", reader=fake)()
    assert list(res) == ["task:BotA"] and "last run" in res["task:BotA"]


def test_monitor_confirm_and_recovery(tmp_path):
    f = tmp_path / "bot" / "out.log"
    f.parent.mkdir()
    f.write_text("x")
    clock = {"t": time.time()}
    msgs, pubs = [], []
    mon = PulseMonitor([file_fresh(str(f.parent), 30, "bot", now=lambda: clock["t"])],
                       state_path=str(tmp_path / "state.json"), confirm=2, notify=msgs.append, publish=pubs.append)
    assert mon.run_once()["problems"] == 0
    clock["t"] += 3600                             # an hour without output
    s1 = mon.run_once()
    assert s1["problems"] == 1 and msgs == []      # first sighting: no alert yet
    mon.run_once()
    assert len(msgs) == 1 and "silent death" in msgs[0]
    mon.run_once()
    assert len(msgs) == 1                          # no repeated alert
    f.write_text("y")
    clock["t"] = time.time()
    mon.run_once()
    assert len(msgs) == 2 and "OK again" in msgs[1]
    assert len(pubs) == 5
    assert json.load(open(tmp_path / "state.json"))["fresh:bot"]["fails"] == 0


def test_broken_check_does_not_crash(tmp_path):
    def boom():
        raise RuntimeError("x")
    s = PulseMonitor([boom], state_path=str(tmp_path / "s.json")).run_once()
    assert s["problems"] == 1


# ----------------------------------------------------------------------------- CLI
def test_cli_check_text(capsys):
    assert main(["check", SAMPLE, "--trials", "1"]) == 0
    out = capsys.readouterr().out
    assert "STATUS: READY_FOR_MICRO_LIVE" in out
    assert main(["check", SAMPLE, "--trials", "60", "--fail-unless-ready"]) == 1
    assert "STATUS: NOT_READY" in capsys.readouterr().out


def test_cli_json_and_computed_r(tmp_path, capsys):
    p = tmp_path / "raw.csv"
    p.write_text("closed,symbol,side,entry,exit,stop\n"
                 "2025-01-01,AAA,long,100,103,98\n2025-01-02,BBB,short,50,49,51\n2025-01-03,CCC,long,10,9.8,9.8\n")
    assert main(["check", str(p), "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["n"] == 3 and d["status"] == "NOT_READY"


def test_cli_bad_file(capsys):
    assert main(["check", "does_not_exist.csv"]) == 2
