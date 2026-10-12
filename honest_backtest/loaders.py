"""Load a trade list from CSV.

Accepted columns (case-insensitive, extra columns ignored):

* ``R`` (aliases ``r``, ``r_multiple``) - profit in units of risk, AFTER costs; or
* ``entry``, ``exit``, ``stop`` (stop PRICE) and ``side`` - R is then computed with a :class:`CostModel`
  (optional ``hold_hours`` column for funding);
* optional ``symbol`` (alias ``sym``), ``side``, ``closed`` (aliases ``exit_time``, ``time``, ``date``).
"""
from __future__ import annotations

import csv
from typing import List, Optional

from .costs import CostModel
from .gates import Trade

_ALIASES = {
    "r": ("r", "r_multiple", "r_after_costs"),
    "symbol": ("symbol", "sym", "ticker", "pair"),
    "side": ("side", "direction"),
    "closed": ("closed", "exit_time", "close_time", "time", "date", "timestamp"),
    "entry": ("entry", "entry_price"),
    "exit": ("exit", "exit_price"),
    "stop": ("stop", "stop_price", "sl"),
    "hold_hours": ("hold_hours", "hold_h"),
}


def _pick(row: dict, key: str) -> Optional[str]:
    low = {k.strip().lower(): v for k, v in row.items() if k}
    for a in _ALIASES[key]:
        v = low.get(a)
        if v not in (None, ""):
            return v.strip()
    return None


def _float(x: Optional[str]) -> Optional[float]:
    try:
        return float(x) if x not in (None, "") else None
    except ValueError:
        return None


def load_trades(path: str, cost_model: Optional[CostModel] = None) -> List[Trade]:
    cm = cost_model or CostModel()
    out: List[Trade] = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path}: no rows")
    for i, row in enumerate(rows, start=2):
        r = _float(_pick(row, "r"))
        side = _pick(row, "side")
        if r is None:
            e, x, s = (_float(_pick(row, k)) for k in ("entry", "exit", "stop"))
            if None in (e, x, s) or not side:
                raise ValueError(f"{path}:{i}: need an R column, or entry/exit/stop/side")
            r = cm.r_multiple(side, e, x, s, _float(_pick(row, "hold_hours")))
            if r is None:
                continue
        out.append(Trade(r=r, symbol=_pick(row, "symbol"), side=(side or "").lower() or None,
                         closed=_pick(row, "closed")))
    return out
