# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Real-time feature provider using xtdata for live trading scenarios.

For backtesting, use data_collector.py to convert data to .bin format instead.
This provider is designed for live/online scenarios where real-time data is needed.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from qlib.data.data import FeatureProvider
from qlib.log import get_module_logger
from ..utils import StockCodeConverter

logger = get_module_logger("XtQMTFeatureProvider")


class XtQMTFeatureProvider(FeatureProvider):
    """Feature provider that fetches data from xtdata in real-time.

    Use this for live trading when you need the latest market data that hasn't been
    converted to .bin format yet.

    Note: This is significantly slower than LocalFeatureProvider for large-scale backtesting.
    """

    FIELD_MAP = {
        "$open": "open",
        "$high": "high",
        "$low": "low",
        "$close": "close",
        "$volume": "volume",
        "$amount": "amount",
    }

    def __init__(self, freq: str = "1d"):
        self.freq = freq
        try:
            from xtquant import xtdata
            self.xtdata = xtdata
        except ImportError:
            raise ImportError("xtquant is required for XtQMTFeatureProvider.")

    def feature(self, instrument, field, start_time, end_time, freq):
        """Get feature data from xtdata.

        Parameters
        ----------
        instrument : str
            Qlib instrument code (e.g. "SH600000").
        field : str
            Qlib field name (e.g. "$close").
        start_time : str
            Start time.
        end_time : str
            End time.
        freq : str
            Frequency.

        Returns
        -------
        pd.Series
            Feature data with datetime index.
        """
        xt_field = self.FIELD_MAP.get(field)
        if xt_field is None:
            logger.warning(f"Unsupported field: {field}")
            return pd.Series(dtype=np.float32)

        xt_code = StockCodeConverter.qlib_to_xt(instrument)
        period = self._freq_to_period(freq)

        start_str = pd.Timestamp(start_time).strftime("%Y%m%d") if start_time else ""
        end_str = pd.Timestamp(end_time).strftime("%Y%m%d") if end_time else ""

        data = self.xtdata.get_market_data_ex(
            stock_list=[xt_code],
            period=period,
            start_time=start_str,
            end_time=end_str,
            fields=[xt_field],
        )

        if xt_code not in data or data[xt_code].empty:
            return pd.Series(dtype=np.float32)

        df = data[xt_code]
        df.index = pd.to_datetime(df.index, format="%Y%m%d" if period == "1d" else "%Y%m%d%H%M%S")

        series = df[xt_field].astype(np.float32)
        series.name = field
        return series

    @staticmethod
    def _freq_to_period(freq: str) -> str:
        """Convert Qlib freq string to xtdata period string."""
        freq_map = {
            "day": "1d",
            "1min": "1m",
            "5min": "5m",
            "15min": "15m",
            "30min": "30m",
            "60min": "60m",
        }
        return freq_map.get(freq, freq)
