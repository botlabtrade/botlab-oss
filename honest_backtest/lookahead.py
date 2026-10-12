"""Look-ahead audit: does an indicator / signal use data from the future?

Idea (same as Freqtrade's lookahead-analysis): compute your function
  (a) once on the FULL history (how a vectorised backtest does it), and
  (b) bar by bar on data TRUNCATED at that bar (how a live bot sees it).
If (a)[i] != (b)[-1] for some bar i, the function peeked at bars after i: centred rolling windows,
``shift(-k)``, normalising with full-sample statistics, resampling without shifting, ``bfill`` ...

Also included:
* :func:`warmup_analysis` - how much a value depends on how much history you load (long EMAs on short
  windows give backtest != live).
* :func:`closed_upto` / :func:`check_htf_rows` - multi-timeframe check: when you evaluate a bar that
  closes at time t, every higher-timeframe candle you use must have closed by t (or be a partial candle
  built only from lower-timeframe candles closed by t).

Frames are pandas DataFrames with a ``start`` column (candle open time in ms) and OHLC columns.
"""
from __future__ import annotations

import math
import random
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------- comparison helpers
def _num(x) -> bool:
    return isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, (bool, np.bool_))


def same(a, b, rtol: float = 1e-7, atol: float = 1e-9) -> bool:
    """Tolerant equality for scalars, NaN, dicts and lists."""
    if a is None or b is None:
        return a is None and b is None
    if _num(a) and _num(b):
        if math.isnan(a) and math.isnan(b):
            return True
        return math.isclose(float(a), float(b), rel_tol=rtol, abs_tol=atol)
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(same(a[k], b[k], rtol, atol) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same(x, y, rtol, atol) for x, y in zip(a, b))
    try:
        return bool(a == b)
    except Exception:
        return str(a) == str(b)


def _scalar(x):
    if isinstance(x, (np.floating, float)):
        return float(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def _row(v, i):
    if isinstance(v, pd.DataFrame):
        return {c: _scalar(v[c].iloc[i]) for c in v.columns}
    if isinstance(v, pd.Series):
        return _scalar(v.iloc[i])
    if isinstance(v, dict):
        return {k: _row(x, i) for k, x in v.items()}
    return _scalar(np.asarray(v)[i])


def _diff_keys(a, b, rtol):
    if isinstance(a, dict) and isinstance(b, dict):
        return [k for k in sorted(set(a) | set(b)) if not same(a.get(k), b.get(k), rtol)]
    return [] if same(a, b, rtol) else ["value"]


def sample_indices(lo: int, hi: int, k: int, seed: int = 11) -> List[int]:
    """``k`` sorted random bar indices in [lo, hi) (all of them if the range is small)."""
    if hi - lo <= k:
        return list(range(lo, hi))
    return sorted(random.Random(seed).sample(range(lo, hi), k))


# ----------------------------------------------------------------------------- 1) full vs incremental
def lookahead_analysis(fn: Callable[[pd.DataFrame], Any], df: pd.DataFrame, sample_idx: Iterable[int],
                       rtol: float = 1e-7, max_report: int = 5) -> Dict[str, Any]:
    """Compare ``fn(df)[i]`` (full history) with ``fn(df[:i+1])[-1]`` (truncated) at each sampled bar.

    ``fn`` must return one value per bar: a Series, DataFrame, ndarray or dict of arrays.
    Returns ``{"n": checked, "mismatch": count, "examples": [...]}``. ``mismatch > 0`` = look-ahead.
    """
    full = fn(df)
    bad, n = [], 0
    for i in sample_idx:
        inc = fn(df.iloc[:i + 1].reset_index(drop=True))
        a, b = _row(full, i), _row(inc, -1)
        n += 1
        if not same(a, b, rtol):
            bad.append(dict(i=int(i), keys=_diff_keys(a, b, rtol), full=a, incremental=b))
    return dict(n=n, mismatch=len(bad), examples=bad[:max_report])


def signal_lookahead(full_signals: Dict[Any, Any], recompute: Callable[[Any], Any], keys: Iterable[Any],
                     rtol: float = 1e-7, max_report: int = 5) -> Dict[str, Any]:
    """Pipeline version: ``full_signals[key]`` was produced once over the whole dataset;
    ``recompute(key)`` must rebuild it using ONLY data available at ``key`` (e.g. a timestamp)."""
    bad, n = [], 0
    for k in keys:
        a, b = full_signals.get(k), recompute(k)
        if a is None and b is None:
            continue
        n += 1
        if not same(a, b, rtol):
            bad.append(dict(key=k, full=a, incremental=b))
    return dict(n=n, mismatch=len(bad), examples=bad[:max_report])


# ----------------------------------------------------------------------------- 2) warm-up sensitivity
def warmup_analysis(fn: Callable[[pd.DataFrame], Any], df: pd.DataFrame, eval_idx: Iterable[int],
                    warmups: Sequence[int], rtol_flag: float = 0.001) -> Dict[str, Any]:
    """``fn(frame)`` returns the value at the LAST bar (scalar or dict). For each bar i and each warm-up
    length w, compute it on ``df[i-w+1 : i+1]`` and compare with the longest warm-up.

    Reports relative drift for numbers and flip % for labels / booleans.
    """
    ref_w = max(warmups)
    out: Dict[str, Any] = {}
    eval_idx = list(eval_idx)
    for w in warmups:
        if w == ref_w:
            continue
        drift: Dict[str, list] = {}
        flips: Dict[str, int] = {}
        n_lab: Dict[str, int] = {}
        n = 0
        for i in eval_idx:
            if i - ref_w + 1 < 0:
                continue
            ref = fn(df.iloc[i - ref_w + 1:i + 1].reset_index(drop=True))
            got = fn(df.iloc[i - w + 1:i + 1].reset_index(drop=True))
            n += 1
            rd = ref if isinstance(ref, dict) else {"value": ref}
            gd = got if isinstance(got, dict) else {"value": got}
            for k, rv in rd.items():
                gv = gd.get(k)
                if _num(rv) and _num(gv) and not (math.isnan(rv) or math.isnan(gv)):
                    drift.setdefault(k, []).append(abs(gv - rv) / max(abs(rv), 1e-12))
                else:
                    n_lab[k] = n_lab.get(k, 0) + 1
                    if not same(rv, gv):
                        flips[k] = flips.get(k, 0) + 1
        res: Dict[str, dict] = {}
        for k, xs in drift.items():
            a = np.array(xs)
            res[k] = dict(max_rel=float(a.max()), med_rel=float(np.median(a)),
                          pct_over=float((a > rtol_flag).mean() * 100))
        for k, c in n_lab.items():
            res.setdefault(k, {})["flip_pct"] = flips.get(k, 0) / c * 100
        out[f"w{w}_vs_w{ref_w}"] = dict(n=n, fields=res)
    return out


# ----------------------------------------------------------------------------- 3) multi-timeframe
def closed_upto(df: pd.DataFrame, interval_ms: int, t_ms: int) -> pd.DataFrame:
    """Only candles that have CLOSED at time t (start + interval <= t)."""
    return df[df["start"] + interval_ms <= t_ms].reset_index(drop=True)


def check_htf_rows(htf: pd.DataFrame, htf_iv_ms: int, t_ms: int, ltf: Optional[pd.DataFrame] = None,
                   ltf_iv_ms: Optional[int] = None, tol: float = 1e-9) -> List[dict]:
    """List higher-timeframe rows that could not have been known at t.

    A row whose candle closes after t is only legal if it is a partial candle whose high/low/close
    match the lower-timeframe candles closed by t.
    """
    issues = []
    for _, r in htf[htf["start"] + htf_iv_ms > t_ms].iterrows():
        if r["start"] >= t_ms:
            issues.append(dict(start=int(r["start"]), why="candle starts after t (pure future)"))
            continue
        if ltf is None or ltf_iv_ms is None:
            issues.append(dict(start=int(r["start"]), why="candle not closed at t (no lower timeframe to verify)"))
            continue
        g = ltf[(ltf["start"] >= r["start"]) & (ltf["start"] + ltf_iv_ms <= t_ms)]
        ok = (not g.empty and math.isclose(r["high"], g["high"].max(), rel_tol=tol)
              and math.isclose(r["low"], g["low"].min(), rel_tol=tol)
              and math.isclose(r["close"], g["close"].iloc[-1], rel_tol=tol))
        if not ok:
            issues.append(dict(start=int(r["start"]), why="unclosed candle contains data from after t"))
    return issues
