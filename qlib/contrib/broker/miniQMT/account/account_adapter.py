# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Adapter wrapping XtQuantTrader account queries into a convenient interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from qlib.log import get_module_logger
from ..utils import StockCodeConverter

logger = get_module_logger("XtAccountAdapter")


@dataclass
class AccountSnapshot:
    """A snapshot of the broker account state."""
    total_asset: float = 0.0
    cash: float = 0.0
    frozen_cash: float = 0.0
    market_value: float = 0.0
    positions: Dict[str, "StockPositionInfo"] = None

    def __post_init__(self):
        if self.positions is None:
            self.positions = {}


@dataclass
class StockPositionInfo:
    """Info for a single stock position from broker."""
    stock_code: str = ""  # Qlib format
    volume: int = 0
    sellable_volume: int = 0
    avg_price: float = 0.0
    market_value: float = 0.0
    profit: float = 0.0


class XtAccountAdapter:
    """High-level account query interface wrapping xtquant trader.

    Parameters
    ----------
    xt_trader : XtQuantTrader
        Connected xtquant trader instance.
    account_id : str
        Trading account ID.
    """

    def __init__(self, xt_trader: Any, account_id: str):
        self.xt_trader = xt_trader
        self.account_id = account_id

    def get_snapshot(self) -> AccountSnapshot:
        """Get a full snapshot of the current account state.

        Returns
        -------
        AccountSnapshot
            Current account state including all positions.
        """
        asset = self.xt_trader.query_stock_asset(self.account_id)
        snapshot = AccountSnapshot(
            total_asset=getattr(asset, "total_asset", 0.0),
            cash=getattr(asset, "cash", 0.0),
            frozen_cash=getattr(asset, "frozen_cash", 0.0),
            market_value=getattr(asset, "market_value", 0.0),
        )

        positions = self.xt_trader.query_stock_positions(self.account_id)
        if positions:
            for pos in positions:
                xt_code = getattr(pos, "stock_code", "")
                volume = getattr(pos, "volume", 0)
                if volume > 0 and xt_code:
                    try:
                        qlib_code = StockCodeConverter.xt_to_qlib(xt_code)
                    except ValueError:
                        continue
                    snapshot.positions[qlib_code] = StockPositionInfo(
                        stock_code=qlib_code,
                        volume=volume,
                        sellable_volume=int(getattr(pos, "can_use_volume", 0)),
                        avg_price=getattr(pos, "open_price", 0.0),
                        market_value=getattr(pos, "market_value", 0.0),
                        profit=getattr(pos, "profit", 0.0),
                    )

        return snapshot

    def get_cash(self) -> float:
        """Get available cash."""
        asset = self.xt_trader.query_stock_asset(self.account_id)
        return getattr(asset, "cash", 0.0) if asset else 0.0

    def get_total_asset(self) -> float:
        """Get total asset value (cash + market value)."""
        asset = self.xt_trader.query_stock_asset(self.account_id)
        return getattr(asset, "total_asset", 0.0) if asset else 0.0

    def get_sellable_amount(self, stock_id: str) -> int:
        """Get sellable amount for a stock (T+1 aware).

        Parameters
        ----------
        stock_id : str
            Qlib stock code.

        Returns
        -------
        int
            Number of shares that can be sold.
        """
        xt_code = StockCodeConverter.qlib_to_xt(stock_id)
        positions = self.xt_trader.query_stock_positions(self.account_id)
        if not positions:
            return 0
        for pos in positions:
            if getattr(pos, "stock_code", "") == xt_code:
                return int(getattr(pos, "can_use_volume", 0))
        return 0

    def get_position_list(self) -> List[str]:
        """Get list of stock codes currently held (Qlib format)."""
        snapshot = self.get_snapshot()
        return list(snapshot.positions.keys())
