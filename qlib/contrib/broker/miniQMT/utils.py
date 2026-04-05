# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Stock code conversion utilities and A-share market rules.

Qlib uses codes like "SH600000" or "SZ000001".
xtquant uses codes like "600000.SH" or "000001.SZ".
"""

from __future__ import annotations

import re
from typing import List


class StockCodeConverter:
    """Convert stock codes between Qlib format and xtquant format.

    Qlib format:  SH600000, SZ000001
    xtquant format: 600000.SH, 000001.SZ
    """

    _QLIB_PATTERN = re.compile(r"^(SH|SZ)(\d{6})$")
    _XT_PATTERN = re.compile(r"^(\d{6})\.(SH|SZ)$")

    @classmethod
    def qlib_to_xt(cls, qlib_code: str) -> str:
        """Convert Qlib code to xtquant code. E.g. 'SH600000' -> '600000.SH'"""
        m = cls._QLIB_PATTERN.match(qlib_code)
        if not m:
            raise ValueError(f"Invalid Qlib stock code: {qlib_code}")
        exchange, number = m.group(1), m.group(2)
        return f"{number}.{exchange}"

    @classmethod
    def xt_to_qlib(cls, xt_code: str) -> str:
        """Convert xtquant code to Qlib code. E.g. '600000.SH' -> 'SH600000'"""
        m = cls._XT_PATTERN.match(xt_code)
        if not m:
            raise ValueError(f"Invalid xtquant stock code: {xt_code}")
        number, exchange = m.group(1), m.group(2)
        return f"{exchange}{number}"

    @classmethod
    def qlib_to_xt_batch(cls, qlib_codes: List[str]) -> List[str]:
        return [cls.qlib_to_xt(c) for c in qlib_codes]

    @classmethod
    def xt_to_qlib_batch(cls, xt_codes: List[str]) -> List[str]:
        return [cls.xt_to_qlib(c) for c in xt_codes]


# A-share trading rules
TRADE_UNIT = 100  # 1 lot = 100 shares
PRICE_TICK = 0.01

# Trading sessions (Beijing time)
MORNING_OPEN = "09:30"
MORNING_CLOSE = "11:30"
AFTERNOON_OPEN = "13:00"
AFTERNOON_CLOSE = "15:00"


def round_to_lot(amount: float, trade_unit: int = TRADE_UNIT) -> int:
    """Round down to the nearest lot (100 shares for A-share)."""
    if trade_unit is None or trade_unit <= 0:
        return int(amount)
    return int(amount // trade_unit) * trade_unit


def is_trading_time(hour: int, minute: int) -> bool:
    """Check if the given time (HH:MM) is within A-share trading sessions."""
    t = hour * 60 + minute
    morning = (9 * 60 + 30) <= t <= (11 * 60 + 30)
    afternoon = (13 * 60) <= t <= (15 * 60)
    return morning or afternoon
