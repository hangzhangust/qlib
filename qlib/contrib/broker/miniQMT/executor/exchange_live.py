# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Live exchange that routes orders to miniQMT for real execution.

Replaces the simulated deal_order() with actual broker submission.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from qlib.backtest.decision import Order, OrderDir
from qlib.backtest.exchange import Exchange
from qlib.backtest.position import BasePosition
from qlib.log import get_module_logger
from ..trader.order_manager import XtOrderManager
from ..utils import round_to_lot, TRADE_UNIT

if TYPE_CHECKING:
    from qlib.backtest.account import Account

logger = get_module_logger("XtQMTLiveExchange")


class XtQMTLiveExchange(Exchange):
    """A live exchange that submits orders to miniQMT via XtOrderManager.

    Inherits from Exchange to maintain compatibility with Qlib's executor framework.
    The key override is `deal_order()` which sends orders to the real broker instead of
    simulating fills.

    Parameters
    ----------
    order_manager : XtOrderManager
        The order manager connected to xtquant trader.
    open_cost : float
        Commission rate for buying.
    close_cost : float
        Commission rate for selling.
    min_cost : float
        Minimum commission per trade.
    **kwargs
        Additional arguments passed to Exchange.__init__.
    """

    def __init__(
        self,
        order_manager: XtOrderManager,
        open_cost: float = 0.0005,
        close_cost: float = 0.0015,
        min_cost: float = 5.0,
        trade_unit: int = TRADE_UNIT,
        **kwargs: Any,
    ):
        self.order_manager = order_manager
        self._open_cost = open_cost
        self._close_cost = close_cost
        self._min_cost = min_cost
        self._trade_unit = trade_unit
        self.logger = logger

        # We don't call super().__init__() because Exchange.__init__ loads quote data
        # from Qlib's data system, which we don't need for live trading.
        # Instead, we only initialize the fields that deal_order needs.

    def deal_order(
        self,
        order: Order,
        trade_account: Account | None = None,
        position: BasePosition | None = None,
        dealt_order_amount: Dict[str, float] = defaultdict(float),
    ) -> Tuple[float, float, float]:
        """Execute order via miniQMT broker.

        Parameters
        ----------
        order : Order
            The order to execute.
        trade_account : Account, optional
            Account to update after dealing.
        position : BasePosition, optional
            Position to update after dealing.
        dealt_order_amount : dict
            Previously dealt amounts for volume limit tracking.

        Returns
        -------
        Tuple[float, float, float]
            (trade_val, trade_cost, trade_price)
        """
        if trade_account is not None and position is not None:
            raise ValueError("trade_account and position can only choose one")

        # T+1 check for sell orders
        if order.direction == OrderDir.SELL:
            sellable = self.order_manager.query_sellable_amount(order.stock_id)
            if sellable <= 0:
                logger.warning(f"No sellable shares for {order.stock_id} (T+1 restriction)")
                order.deal_amount = 0.0
                return 0.0, 0.0, np.nan

            # Clamp sell amount to sellable quantity
            order_amount = round_to_lot(min(order.amount, sellable), self._trade_unit)
            if order_amount <= 0:
                order.deal_amount = 0.0
                return 0.0, 0.0, np.nan
        else:
            order_amount = round_to_lot(order.amount, self._trade_unit)

        if order_amount <= 0:
            order.deal_amount = 0.0
            return 0.0, 0.0, np.nan

        # Propagate clamped/rounded amount so order_manager submits the correct quantity
        order.amount = order_amount

        # Submit to broker
        result = self.order_manager.submit_order(order)

        if not result.success or result.deal_amount <= 0:
            logger.warning(f"Order failed for {order.stock_id}: {result.error_msg}")
            order.deal_amount = 0.0
            return 0.0, 0.0, np.nan

        # Update order with actual deal results
        trade_price = result.deal_price
        deal_amount = result.deal_amount
        trade_val = trade_price * deal_amount

        # Calculate commission
        if order.direction == OrderDir.BUY:
            trade_cost = max(trade_val * self._open_cost, self._min_cost)
        else:
            trade_cost = max(trade_val * self._close_cost, self._min_cost)

        # Update the order object
        order.deal_amount = deal_amount
        order.factor = 1.0  # live trading uses real prices, no adjust factor needed

        # Update account/position
        if trade_val > 1e-5:
            if trade_account:
                trade_account.update_order(
                    order=order, trade_val=trade_val, cost=trade_cost, trade_price=trade_price
                )
            elif position:
                position.update_order(
                    order=order, trade_val=trade_val, cost=trade_cost, trade_price=trade_price
                )

        logger.info(
            f"Dealt: {order.stock_id} {'BUY' if order.direction == OrderDir.BUY else 'SELL'} "
            f"price={trade_price:.2f} amount={deal_amount} val={trade_val:.2f} cost={trade_cost:.2f}"
        )

        return trade_val, trade_cost, trade_price

    # Override methods that depend on quote data to be no-ops or raise clear errors
    def check_order(self, order: Order) -> bool:
        """In live mode, we let the broker handle order validation."""
        return True

    def get_close(self, stock_id, start_time, end_time, method=None):
        """Get close price from xtdata for position valuation."""
        try:
            from xtquant import xtdata
            from ..utils import StockCodeConverter

            xt_code = StockCodeConverter.qlib_to_xt(stock_id)
            tick = xtdata.get_full_tick([xt_code])
            if xt_code in tick and tick[xt_code]:
                return tick[xt_code].get("lastPrice", np.nan)
        except Exception as e:
            logger.debug(f"Failed to get close for {stock_id}: {e}")
        return np.nan
