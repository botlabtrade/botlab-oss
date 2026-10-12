import datetime as dt
import random

from honest_backtest.gates import (NOT_READY, READY, WATCH, GateConfig, Trade, evaluate,
                                   format_report)


def make_trades(n=400, p_win=0.5, win=1.5, loss=-1.0, n_syms=20, seed=1, sides=("long", "short")):
    rng = random.Random(seed)
    t = dt.datetime(2025, 1, 1)
    out = []
    for i in range(n):
        t += dt.timedelta(hours=20)
        r = rng.gauss(win, 0.2) if rng.random() < p_win else rng.gauss(loss, 0.1)
        out.append(Trade(r=r, symbol=f"S{rng.randrange(n_syms)}", side=rng.choice(sides),
                         closed=t.isoformat(sep=" ")))
    return out


def test_strong_edge_is_ready():
    rep = evaluate(make_trades(), n_trials=10)
    assert rep.status == READY, format_report(rep)
    assert rep.failed == []


def test_losing_strategy_not_ready():
    rep = evaluate(make_trades(p_win=0.3), n_trials=1)
    assert rep.status == NOT_READY
    assert not rep.gates["G1"].ok and "infinite" in rep.gates["G1"].why
    assert not rep.gates["G2"].ok


def test_g1_floor():
    tr = make_trades(n=150)
    rep = evaluate(tr, n_trials=1)
    assert not rep.gates["G1"].ok and rep.gates["G1"].extra["need"] == 300
    assert rep.status == WATCH                      # G2 + G3 pass, only sample size missing
    assert evaluate(tr, 1, GateConfig(min_n=100)).gates["G1"].ok


def test_many_trials_kill_g2():
    tr = make_trades(n=420, p_win=0.45, seed=31)
    weak_one = evaluate(tr, n_trials=1)
    weak_many = evaluate(tr, n_trials=10_000)
    assert weak_many.dsr < weak_one.dsr
    assert not weak_many.gates["G2"].ok


def test_g3_recent_decay():
    tr = make_trades(n=400)
    for t in tr[-100:]:
        t.r = -0.5                                 # edge disappeared recently
    rep = evaluate(tr, n_trials=1)
    assert not rep.gates["G3"].ok


def test_g4_symbol_concentration():
    tr = make_trades(n=400, n_syms=2)
    rep = evaluate(tr, n_trials=1)
    assert not rep.gates["G4"].ok and "top 3 symbols" in rep.gates["G4"].why


def test_g4_day_concentration():
    tr = make_trades(n=400, p_win=0.34)            # roughly break-even
    tr[10].r, tr[20].r = 400.0, 400.0              # two lucky days carry everything
    rep = evaluate(tr, n_trials=1)
    assert not rep.gates["G4"].ok and "best 2 days" in rep.gates["G4"].why


def test_g4_not_scored_without_columns():
    tr = [Trade(r=t.r, side=t.side) for t in make_trades()]
    rep = evaluate(tr, n_trials=1)
    assert rep.gates["G4"].ok is None
    assert rep.status != READY                     # unscored gate blocks READY


def test_g5_side_leak():
    tr = make_trades(n=400)
    for t in tr:
        if t.side == "short":
            t.r = -abs(t.r) * 0.3
    rep = evaluate(tr, n_trials=1)
    assert not rep.gates["G5"].ok and "LEAK: short" in rep.gates["G5"].why


def test_g6_drawdown():
    tr = make_trades(n=400)
    for t in tr[150:170]:
        t.r = -1.0                                 # 20R losing streak
    rep = evaluate(tr, n_trials=1)
    assert not rep.gates["G6"].ok


def test_strict_mode():
    tr = make_trades(n=420, p_win=0.45, seed=31)
    assert evaluate(tr, 3, strict=True).gates["G2"].ok is (evaluate(tr, 3).dsr >= 0.95)


def test_too_few_trades():
    rep = evaluate([Trade(r=1.0)], n_trials=1)
    assert rep.status == NOT_READY and len(rep.failed) == 6
    assert "STATUS: NOT_READY" in format_report(rep)
