"""A simple, explicit trading-cost model for perpetual futures.

Per round trip:   cost = 2 x taker_fee + 2 x slippage + funding_per_8h x (hold_hours / 8) x side_sign
Net return:       net  = side_sign x (exit / entry - 1) - cost
R multiple:       R    = net / stop_frac      (stop_frac = |entry - stop| / entry, the fraction risked)

Funding sign: a positive funding rate is paid by longs and received by shorts, so for a long it is a cost
and for a short a (small) gain. If you don't know the funding rate, use a conservative positive default.

Slippage can be a flat number per side, or estimated by walking a real order book for the notional size
(:func:`orderbook_slippage`). :func:`fetch_bybit_orderbook` is OPTIONAL and the only function in this
package that touches the network (public endpoint, no key).
"""
from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

Level = Tuple[float, float]  # (price, quantity)


@dataclass(frozen=True)
class CostModel:
    taker_fee: float = 0.00055       # per side (0.055 %, a common perp taker fee tier)
    slippage: float = 0.0010         # per side, used when no order book is given
    funding_8h: float = 0.0001       # average funding rate per 8h (0.01 %)

    def round_trip_cost(self, side: str, hold_hours: Optional[float] = None,
                        slippage: Optional[float] = None, funding_8h: Optional[float] = None) -> float:
        sg = _sign(side)
        s = self.slippage if slippage is None else slippage
        f = self.funding_8h if funding_8h is None else funding_8h
        fund = sg * f * (hold_hours / 8.0) if hold_hours else 0.0
        return 2 * self.taker_fee + 2 * s + fund

    def net_return(self, side: str, entry: float, exit_: float, hold_hours: Optional[float] = None,
                   slippage: Optional[float] = None, funding_8h: Optional[float] = None) -> float:
        move = _sign(side) * (exit_ / entry - 1.0)
        return move - self.round_trip_cost(side, hold_hours, slippage, funding_8h)

    def r_multiple(self, side: str, entry: float, exit_: float, stop: float,
                   hold_hours: Optional[float] = None, slippage: Optional[float] = None,
                   funding_8h: Optional[float] = None) -> Optional[float]:
        """R after costs. ``stop`` is the stop PRICE. Returns None if the stop distance is ~0."""
        stop_frac = abs(entry - stop) / entry if entry else 0.0
        if stop_frac <= 1e-5:
            return None
        return self.net_return(side, entry, exit_, hold_hours, slippage, funding_8h) / stop_frac


def _sign(side: str) -> int:
    s = (side or "").strip().lower()
    if s in ("long", "buy", "l", "1", "+1"):
        return 1
    if s in ("short", "sell", "s", "-1"):
        return -1
    raise ValueError(f"unknown side {side!r} (use long/short)")


def orderbook_slippage(asks: Sequence[Level], bids: Sequence[Level], notional: float,
                       beyond_book_penalty: float = 0.01) -> float:
    """Average one-side slippage vs mid for a market order of ``notional`` (quote currency).

    Walks asks (for a buy) and bids (for a sell) and returns the mean of the two. If the book is too
    thin, the remainder is filled at the last level +/- ``beyond_book_penalty``.
    """
    if not asks or not bids or notional <= 0:
        raise ValueError("need non-empty asks/bids and notional > 0")
    asks = sorted(asks, key=lambda x: x[0])
    bids = sorted(bids, key=lambda x: -x[0])
    mid = (asks[0][0] + bids[0][0]) / 2
    qty = notional / mid

    def walk(levels: Sequence[Level], up: bool) -> float:
        got = cost = 0.0
        for p, q in levels:
            take = min(q, qty - got)
            got += take
            cost += take * p
            if got >= qty - 1e-12:
                break
        if got < qty:
            last = levels[-1][0] * ((1 + beyond_book_penalty) if up else (1 - beyond_book_penalty))
            cost += (qty - got) * last
            got = qty
        return abs(cost / got / mid - 1)

    return (walk(asks, True) + walk(bids, False)) / 2


def fetch_bybit_orderbook(symbol: str, depth: int = 200, timeout: float = 15.0
                          ) -> Tuple[List[Level], List[Level]]:
    """OPTIONAL network helper: public Bybit linear-perp order book (no API key). Returns (asks, bids).

    Note: a snapshot of *today's* book is only a rough proxy for historical liquidity.
    """
    url = ("https://api.bybit.com/v5/market/orderbook?category=linear"
           f"&symbol={symbol}&limit={int(depth)}")
    req = urllib.request.Request(url, headers={"User-Agent": "honest-backtest"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        res = json.loads(r.read().decode())["result"]
    asks = [(float(p), float(q)) for p, q in res.get("a", [])]
    bids = [(float(p), float(q)) for p, q in res.get("b", [])]
    return asks, bids
