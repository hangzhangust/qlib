# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Instrument provider using xtdata stock lists.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import pandas as pd

from qlib.data.data import InstrumentProvider
from qlib.log import get_module_logger
from ..utils import StockCodeConverter

logger = get_module_logger("XtQMTInstrumentProvider")


class XtQMTInstrumentProvider(InstrumentProvider):
    """Instrument provider backed by xtdata sector/stock lists.

    Maps Qlib market names to xtdata sectors:
    - "all" -> "沪深A股"
    - "sse50" -> "上证50"
    - "csi300" -> "沪深300"
    - "csi500" -> "中证500"
    """

    MARKET_SECTOR_MAP = {
        "all": "沪深A股",
        "sse50": "上证50",
        "csi300": "沪深300",
        "csi500": "中证500",
        "sse": "上证A股",
        "szse": "深证A股",
    }

    def __init__(self):
        try:
            from xtquant import xtdata
            self.xtdata = xtdata
        except ImportError:
            raise ImportError("xtquant is required for XtQMTInstrumentProvider.")

    def list_instruments(self, instruments, start_time=None, end_time=None, freq="day", as_list=False):
        """List instruments from xtdata.

        Parameters
        ----------
        instruments : dict or list
            Stockpool config dict with 'market' key, or list of stock codes.
        start_time : str, optional
            Start time (not used for filtering in xtdata, included for API compatibility).
        end_time : str, optional
            End time.
        as_list : bool
            If True, return list of codes. If False, return dict with time spans.

        Returns
        -------
        dict or list
            Instruments.
        """
        inst_type = self.get_inst_type(instruments)

        if inst_type == self.LIST:
            stock_list = list(instruments)
        elif inst_type == self.CONF:
            market = instruments.get("market", "all")
            sector = self.MARKET_SECTOR_MAP.get(market, market)
            xt_codes = self.xtdata.get_stock_list_in_sector(sector)
            stock_list = StockCodeConverter.xt_to_qlib_batch(xt_codes)
        else:
            return instruments

        if as_list:
            return stock_list

        # Return dict format: {stock_id: (start_time, end_time)}
        result = {}
        st = pd.Timestamp(start_time) if start_time else pd.Timestamp("2010-01-01")
        et = pd.Timestamp(end_time) if end_time else pd.Timestamp.now()
        for code in stock_list:
            result[code] = (st, et)
        return result
