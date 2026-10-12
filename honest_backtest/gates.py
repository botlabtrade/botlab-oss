"""Six readiness gates (G1-G6) for a list of closed trades measured in R.

R = net profit of a trade divided by the amount risked (entry-to-stop distance), after costs.

The thresholds are declared up front in :class:`GateConfig`. Change them BEFORE you look at results,
never after - moving the goalposts is just another form of overfitting.

Status
------
* ``READY_FOR_MICRO_LIVE``: every gate passed. Means "worth considering a small, monitored live test",
  not "this will make money".
* ``WATCH``: G2 (significance) and G3 (recent performance) pass, but something else does not yet
  (usually the sample size).
* ``NOT_READY``: everything else.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence

from . import stats as S

READY, WATCH, NOT_READY = "READY_FOR_MICRO_LIVE", "WATCH", "NOT_READY"
GATES = ("G1", "G2", "G3", "G4", "G5", "G6")


@dataclass(frozen=True)
class GateConfig:
    min_n: int = 300                 # G1 floor: n >= max(MinTRL, min_n)
    psr_min: float = 0.95            # G2: PSR(0) >= psr_min
    dsr_min: float = 0.50            # G2: DSR >= dsr_min ("staging" level)
    dsr_strict: float = 0.95         # G2 strict target, reported only (use --strict to enforce)
    last_long: int = 100             # G3: mean R of the last `last_long` trades > long_min
    long_min: float = 0.0
    last_short: int = 30             # G3: mean R of the last `last_short` trades > short_min
    short_min: float = -0.10
    top_syms: int = 3                # G4: top-3 symbols < 50 % of positive R
    sym_share_max: float = 0.50
    top_days: int = 2                # G4: best 2 days < 50 % of total R
    day_share_max: float = 0.50
    leak_min_n: int = 50             # G5: a side with >= 50 trades and mean R < 0 is a leak
    leak_avg_r: float = 0.0
    max_dd_r: float = 15.0           # G6: max drawdown < 15R

    def describe(self) -> Dict[str, str]:
        return {
            "G1": f"sample size: n >= max(MinTRL for PSR(0)=0.95, {self.min_n})",
            "G2": f"significance: PSR(0) >= {self.psr_min} and DSR >= {self.dsr_min} "
                  f"(strict target DSR >= {self.dsr_strict})",
            "G3": f"recent: mean R of last {self.last_long} > {self.long_min:+g} and last "
                  f"{self.last_short} > {self.short_min:+g}",
            "G4": f"concentration: top {self.top_syms} symbols < {self.sym_share_max:.0%} of positive R; "
                  f"best {self.top_days} days < {self.day_share_max:.0%} of total R",
            "G5": f"side leak: long or short side with n >= {self.leak_min_n} and mean R < {self.leak_avg_r:g} fails",
            "G6": f"drawdown: max drawdown of cumulative R < {self.max_dd_r:g}R",
        }


@dataclass
class Trade:
    r: float
    symbol: Optional[str] = None
    side: Optional[str] = None       # "long" / "short"
    closed: Optional[str] = None     # ISO timestamp or date of exit; used for ordering and G4 days


@dataclass
class GateResult:
    ok: Optional[bool]               # None = could not be scored (missing column)
    why: str
    extra: dict = field(default_factory=dict)


@dataclass
class Report:
    n: int
    n_trials: int
    mean_r: Optional[float]
    sum_r: Optional[float]
    win_rate: Optional[float]
    sr: Optional[float]
    psr0: Optional[float]
    dsr: Optional[float]
    sr0: Optional[float]
    min_trl: Optional[float]
    ci90: tuple
    max_dd_r: Optional[float]
    gates: Dict[str, GateResult]
    status: str
    failed: List[str]
    config: GateConfig

    def to_dict(self) -> dict:
        d = asdict(self)
        d["min_trl"] = None if self.min_trl is None or not math.isfinite(self.min_trl) else self.min_trl
        return d


def _mean(a: Sequence[float]) -> Optional[float]:
    return sum(a) / len(a) if a else None


def evaluate(trades: Sequence[Trade], n_trials: int = 1, config: GateConfig = GateConfig(),
             strict: bool = False) -> Report:
    """Score a list of closed trades against G1-G6.

    ``n_trials`` = how many strategies / parameter sets / variants were tried before this one was
    picked (be honest: count everything, including the ones you threw away). It feeds the DSR.
    """
    cfg = config
    n_trials = max(int(n_trials), 1)
    tr = [t for t in trades if t.r is not None and math.isfinite(t.r)]
    if all(t.closed for t in tr):
        tr = sorted(tr, key=lambda t: str(t.closed))
    rs = [t.r for t in tr]
    n = len(rs)
    G: Dict[str, GateResult] = {}

    if n < 2:
        for g in GATES:
            G[g] = GateResult(False, "need at least 2 trades")
        return Report(n, n_trials, _mean(rs), sum(rs) if rs else None, None, None, None, None, None,
                      None, (None, None), None, G, NOT_READY, list(GATES), cfg)

    sr, sk, ku, _ = S.sharpe_stats(rs)
    p0 = S.psr_from_stats(sr, n, sk, ku, 0.0)
    d, sr0 = S.dsr_from_stats(sr, n, sk, ku, n_trials)
    mtrl = S.min_track_record(sr=sr, skew=sk, kurt=ku) if sr is not None else math.inf
    ci = S.bootstrap_ci(rs, 0.90)
    avg = _mean(rs)
    wr = sum(1 for x in rs if x > 0) / n * 100

    # G1 sample size
    if not math.isfinite(mtrl):
        G["G1"] = GateResult(False, f"n={n}; MinTRL is infinite (mean R <= 0, no edge to prove)")
    else:
        need = max(int(math.ceil(mtrl)), cfg.min_n)
        G["G1"] = GateResult(n >= need, f"n={n} {'>=' if n >= need else '<'} need {need} "
                                        f"(max(MinTRL {int(math.ceil(mtrl))}, floor {cfg.min_n}))",
                             dict(need=need))

    # G2 significance
    dmin = cfg.dsr_strict if strict else cfg.dsr_min
    ok2 = p0 is not None and d is not None and p0 >= cfg.psr_min and d >= dmin
    G["G2"] = GateResult(ok2, f"PSR(0)={_f(p0)} (>= {cfg.psr_min}) · DSR={_f(d)} (>= {dmin}) "
                              f"after {n_trials} trial(s), luck-only SR0={_f(sr0, 3)} vs SR={_f(sr, 3)}",
                         dict(strict_pass=bool(d is not None and d >= cfg.dsr_strict)))

    # G3 recent performance
    lL, lS = rs[-cfg.last_long:], rs[-cfg.last_short:]
    aL, aS = _mean(lL), _mean(lS)
    short_hist = n < cfg.last_short
    ok3 = (not short_hist) and aL > cfg.long_min and aS > cfg.short_min
    G["G3"] = GateResult(ok3, f"last {len(lL)}: {aL:+.3f}R (> {cfg.long_min:+g}) · last {len(lS)}: "
                              f"{aS:+.3f}R (> {cfg.short_min:+g})"
                              + (f" · fewer than {cfg.last_short} trades" if short_hist else ""),
                         dict(last_long=aL, last_short=aS))

    # G4 concentration
    has_sym = all(t.symbol for t in tr)
    has_day = all(t.closed for t in tr)
    if not has_sym and not has_day:
        G["G4"] = GateResult(None, "not scored: needs a symbol and/or closed-time column")
    else:
        parts, ok4 = [], True
        if has_sym:
            by_sym: Dict[str, float] = {}
            for t in tr:
                by_sym[t.symbol] = by_sym.get(t.symbol, 0.0) + t.r
            pos = sorted((v for v in by_sym.values() if v > 0), reverse=True)
            share = sum(pos[:cfg.top_syms]) / sum(pos) if pos else None
            top = [s for s, _ in sorted(by_sym.items(), key=lambda kv: -kv[1])[:cfg.top_syms]]
            ok4 &= share is not None and share < cfg.sym_share_max
            parts.append(f"top {cfg.top_syms} symbols = "
                         + (f"{share:.0%}" if share is not None else "n/a (no profitable symbol)")
                         + f" of positive R ({', '.join(top)}; {len(by_sym)} symbols)")
        if has_day:
            by_day: Dict[str, float] = {}
            for t in tr:
                k = str(t.closed)[:10]
                by_day[k] = by_day.get(k, 0.0) + t.r
            tot = sum(rs)
            best = sorted(by_day.values(), reverse=True)[:cfg.top_days]
            dshare = sum(best) / tot if tot > 0 else None
            ok4 &= dshare is not None and dshare < cfg.day_share_max
            parts.append(f"best {cfg.top_days} days = "
                         + (f"{dshare:.0%}" if dshare is not None else "n/a (total R <= 0)") + " of total R")
        G["G4"] = GateResult(ok4, " · ".join(parts))

    # G5 side leak
    sides = {}
    for sd in ("long", "short"):
        v = [t.r for t in tr if (t.side or "").lower() == sd]
        if v:
            sides[sd] = (len(v), _mean(v))
    if not sides:
        G["G5"] = GateResult(None, "not scored: no side column")
    else:
        leaks = [sd for sd, (k, m) in sides.items()
                 if len(sides) == 2 and k >= cfg.leak_min_n and m < cfg.leak_avg_r]
        G["G5"] = GateResult(not leaks, " · ".join(f"{sd} n={k} mean {m:+.3f}R" for sd, (k, m) in sides.items())
                             + (f" · LEAK: {', '.join(leaks)}" if leaks else ("" if len(sides) == 2 else " · one side only")))

    # G6 drawdown
    dd = S.max_drawdown(rs)
    G["G6"] = GateResult(dd < cfg.max_dd_r, f"max drawdown {dd:.1f}R (< {cfg.max_dd_r:g}R)")

    ok = {g: G[g].ok for g in GATES}
    if all(v is True for v in ok.values()):
        status = READY
    elif ok["G2"] and ok["G3"]:
        status = WATCH
    else:
        status = NOT_READY
    failed = [g for g in GATES if ok[g] is not True]
    return Report(n, n_trials, avg, sum(rs), wr, sr, p0, d, sr0, mtrl, ci, dd, G, status, failed, cfg)


def _f(x, k=2) -> str:
    return "n/a" if x is None else f"{x:.{k}f}"


def format_report(rep: Report) -> str:
    """Human-readable gate report."""
    L = []
    L.append("honest_backtest gate report")
    L.append("=" * 60)
    if rep.mean_r is not None and rep.n >= 2:
        lo, hi = rep.ci90
        L.append(f"trades           {rep.n}")
        L.append(f"mean R / trade   {rep.mean_r:+.4f}   (90% bootstrap CI {_f(lo, 3)} .. {_f(hi, 3)})")
        L.append(f"total R          {rep.sum_r:+.2f}")
        L.append(f"win rate         {rep.win_rate:.1f}%")
        L.append(f"Sharpe / trade   {_f(rep.sr, 3)}")
        L.append(f"PSR(0)           {_f(rep.psr0, 3)}")
        L.append(f"DSR              {_f(rep.dsr, 3)}   (trials N = {rep.n_trials})")
        mt = rep.min_trl
        L.append(f"MinTRL           {'inf' if mt is None or not math.isfinite(mt) else math.ceil(mt)} trades")
        L.append(f"max drawdown     {rep.max_dd_r:.2f}R")
    L.append("-" * 60)
    desc = rep.config.describe()
    for g in GATES:
        r = rep.gates[g]
        mark = "PASS" if r.ok is True else ("----" if r.ok is None else "FAIL")
        L.append(f"[{mark}] {g}  {desc[g]}")
        L.append(f"         {r.why}")
    L.append("-" * 60)
    L.append(f"STATUS: {rep.status}" + (f"   (not passed: {', '.join(rep.failed)})" if rep.failed else ""))
    L.append("A pass is a reason to run a small, monitored forward test - not a promise of profit.")
    return "\n".join(L)
