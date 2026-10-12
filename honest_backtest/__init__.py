"""honest_backtest - overfitting-aware checks for trading strategies.

Statistics (PSR, DSR, MinTRL, bootstrap CI, PBO), six readiness gates, a transparent cost model,
a look-ahead audit and a silent-death monitor. Extracted from the tooling behind https://botlab.org.uk
"""
__version__ = "0.1.0"

from .costs import CostModel, orderbook_slippage  # noqa: E402,F401
from .gates import GateConfig, Report, Trade, evaluate, format_report  # noqa: E402,F401
from .stats import (bootstrap_ci, dsr, expected_max_sr, max_drawdown, min_track_record,  # noqa: E402,F401
                    pbo_cscv, psr, sharpe_stats)
