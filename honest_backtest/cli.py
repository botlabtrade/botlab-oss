"""Command line: ``python -m honest_backtest check trades.csv --trials N``."""
from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from . import __version__
from .costs import CostModel
from .gates import READY, GateConfig, evaluate, format_report
from .loaders import load_trades


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="honest_backtest",
                                description="Overfitting-aware readiness gates for a list of trades.")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="score a trades CSV against gates G1-G6")
    c.add_argument("csv", help="CSV with an R column (or entry/exit/stop/side)")
    c.add_argument("--trials", type=int, default=1,
                   help="how many strategies/variants you tried before picking this one (for DSR). Be honest.")
    c.add_argument("--min-n", type=int, default=300, help="G1 floor on trade count (default 300)")
    c.add_argument("--strict", action="store_true", help="require DSR >= 0.95 instead of 0.5 in G2")
    c.add_argument("--fee", type=float, default=0.00055, help="taker fee per side, if R is computed (default 0.00055)")
    c.add_argument("--slippage", type=float, default=0.0010, help="slippage per side, if R is computed (default 0.001)")
    c.add_argument("--funding-8h", type=float, default=0.0001, help="funding per 8h, if R is computed (default 0.0001)")
    c.add_argument("--json", action="store_true", help="print JSON instead of text")
    c.add_argument("--fail-unless-ready", action="store_true", help="exit code 1 unless every gate passes (for CI)")
    a = p.parse_args(argv)

    if a.cmd == "check":
        cm = CostModel(taker_fee=a.fee, slippage=a.slippage, funding_8h=a.funding_8h)
        try:
            trades = load_trades(a.csv, cm)
        except (OSError, ValueError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        rep = evaluate(trades, n_trials=a.trials, config=GateConfig(min_n=a.min_n), strict=a.strict)
        if a.json:
            print(json.dumps(rep.to_dict(), indent=1, default=str))
        else:
            print(format_report(rep))
        if a.fail_unless_ready and rep.status != READY:
            return 1
        return 0
    return 2
