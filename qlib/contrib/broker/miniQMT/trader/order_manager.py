# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Order manager: translates Qlib Order to xtquant order_stock calls.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

from qlib.backtest.decision import Order, OrderDir
from qlib.log import get_module_logger
from ..utils import StockCodeConverter, round_to_lot, TRADE_UNIT
from .callback_handler import DealResult, XtCallbackHandler

logger = get_module_logger("XtOrderManager")


class XtOrderManager:
    """Manages the lifecycle of orders: Qlib Order -> xtquant submission -> deal result.

    Parameters
    ----------
    xt_trader : XtQuantTrader
        The connected xtquant trader instance.
    account_id : str
        The trading account ID (e.g. "1234567890").
    callback : XtCallbackHandler
        The callback handler registered with xt_trader.
    order_timeout : float
        Seconds to wait for order to be dealt.
    """

    def __init__(
        self,
        xt_trader: Any,
        account_id: str,
        callback: XtCallbackHandler,
        order_timeout: float = 60.0,
        price_type: Optional[str] = None,
    ):
        self.xt_trader = xt_trader
        self.account_id = account_id
        self.callback = callback
        self.order_timeout = order_timeout
        self.price_type = price_type

    def submit_order(self, order: Order) -> DealResult:
        """Submit a Qlib Order to xtquant and wait for the result.

        Parameters
        ----------
        order : Order
            The Qlib order to execute.

        Returns
        -------
        DealResult
            The execution result.
        """
        try:
            from xtquant import xtconstant
        except ImportError:
            raise ImportError("xtquant is required.")

        xt_code = StockCodeConverter.qlib_to_xt(order.stock_id)
        amount = round_to_lot(order.amount, TRADE_UNIT)

        if amount <= 0:
            logger.warning(f"Order amount rounded to 0 for {order.stock_id}, skipping")
            return DealResult(success=False, error_msg="Amount rounded to 0")

        if order.direction == OrderDir.BUY:
            xt_direction = xtconstant.STOCK_BUY
        elif order.direction == OrderDir.SELL:
            xt_direction = xtconstant.STOCK_SELL
        else:
            return DealResult(success=False, error_msg=f"Unknown direction: {order.direction}")

        logger.info(f"Submitting order: {xt_code} {'BUY' if order.direction == OrderDir.BUY else 'SELL'} {amount}")

        # Resolve price type: configurable, default to LATEST_PRICE for market-like execution
        if self.price_type is not None:
            xt_price_type = getattr(xtconstant, self.price_type)
        else:
            xt_price_type = xtconstant.LATEST_PRICE

        order_id = self.xt_trader.order_stock(
            self.account_id,
            xt_code,
            xt_direction,
            amount,
            xt_price_type,
            0,
        )

        if order_id is None or order_id < 0:
            error_msg = f"order_stock returned invalid order_id: {order_id}"
            logger.error(error_msg)
            return DealResult(success=False, error_msg=error_msg)

        waiter = self.callback.create_waiter(order_id, target_amount=amount)
        result = waiter.wait()
        return result

    def query_sellable_amount(self, stock_id: str) -> int:
        """Query the sellable amount for a stock from the broker.

        This is critical for T+1 compliance — shares bought today cannot be sold today.

        Parameters
        ----------
        stock_id : str
            Qlib stock code.

        Returns
        -------
        int
            Sellable amount (in shares).
        """
        xt_code = StockCodeConverter.qlib_to_xt(stock_id)

        positions = self.xt_trader.query_stock_positions(self.account_id)
        if not positions:
            return 0

        for pos in positions:
            if pos.stock_code == xt_code:
                return int(getattr(pos, "can_use_volume", 0))
        return 0

    def cancel_order(self, order_id: int) -> bool:
        """Cancel a pending order.

        Parameters
        ----------
        order_id : int
            The xtquant order ID to cancel.

        Returns
        -------
        bool
            True if cancel request was sent successfully.
        """
        try:
            result = self.xt_trader.cancel_order_stock(self.account_id, order_id)
            return result == 0
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
