# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
miniQMT integration for Qlib.

This module bridges Qlib's quantitative research framework with miniQMT (xtquant)
for live trading on China A-share market.

Two main use cases:
1. Data bridging: Convert xtdata historical data to Qlib .bin format for backtesting
2. Live execution: Route Qlib strategy orders to miniQMT for real trading
"""

from .data.data_collector import XtDataCollector
from .trader.trading_loop import TradingLoop

__all__ = ["XtDataCollector", "TradingLoop"]
