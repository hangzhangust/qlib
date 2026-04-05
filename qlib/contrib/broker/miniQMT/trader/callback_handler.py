# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Callback handler for xtquant async trading events.

Bridges xtquant's async callback model to synchronous waiting via threading.Event.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from qlib.log import get_module_logger

logger = get_module_logger("XtCallbackHandler")


@dataclass
class DealResult:
    """Result of a single order deal."""
    order_id: int = 0
    stock_code: str = ""
    direction: int = 0  # xtconstant.STOCK_BUY or STOCK_SELL
    deal_price: float = 0.0
    deal_amount: int = 0
    deal_value: float = 0.0
    error_msg: str = ""
    success: bool = False


class OrderWaiter:
    """Synchronous waiter for an async order result.

    Accumulates partial fills with weighted average price.
    Remains active until fully filled or timeout.
    """

    def __init__(self, order_id: int, timeout: float = 60.0, target_amount: int = 0):
        self.order_id = order_id
        self.timeout = timeout
        self.target_amount = target_amount
        self._event = threading.Event()
        self._result = DealResult(order_id=order_id)
        self._total_amount = 0
        self._total_value = 0.0
        self._lock = threading.Lock()

    def add_fill(self, stock_code: str, direction: int, price: float, amount: int) -> None:
        """Accumulate a partial fill. Signals completion when fully filled."""
        with self._lock:
            self._total_amount += amount
            self._total_value += price * amount
            avg_price = self._total_value / self._total_amount if self._total_amount > 0 else 0.0

            self._result.stock_code = stock_code
            self._result.direction = direction
            self._result.deal_price = avg_price
            self._result.deal_amount = self._total_amount
            self._result.deal_value = self._total_value
            self._result.success = True

            if self.target_amount > 0 and self._total_amount >= self.target_amount:
                self._event.set()

    def set_result(self, result: DealResult) -> None:
        """Set final result (used for errors)."""
        self._result = result
        self._event.set()

    def wait(self) -> DealResult:
        """Block until the order result is available or timeout."""
        if not self._event.wait(timeout=self.timeout):
            with self._lock:
                if self._total_amount > 0:
                    # Partial fill before timeout — return what we got
                    logger.warning(
                        f"Order {self.order_id} timed out with partial fill: "
                        f"{self._total_amount}/{self.target_amount}"
                    )
                else:
                    self._result.error_msg = f"Order {self.order_id} timed out after {self.timeout}s"
                    logger.warning(self._result.error_msg)
        return self._result


class XtCallbackHandler:
    """Implements XtQuantTraderCallback to capture trading events.

    Usage:
        handler = XtCallbackHandler()
        xt_trader.register_callback(handler)

        # When placing an order:
        waiter = handler.create_waiter(order_id)
        xt_trader.order_stock(...)
        result = waiter.wait()
    """

    def __init__(self, order_timeout: float = 60.0):
        self.order_timeout = order_timeout
        self._waiters: Dict[int, OrderWaiter] = {}
        self._pending_results: Dict[int, list] = {}
        self._lock = threading.Lock()

    def create_waiter(self, order_id: int, target_amount: int = 0) -> OrderWaiter:
        """Create a waiter for the given order_id.

        If callbacks arrived before waiter creation (race condition),
        they are replayed immediately.
        """
        waiter = OrderWaiter(order_id, timeout=self.order_timeout, target_amount=target_amount)
        with self._lock:
            self._waiters[order_id] = waiter
            # Replay any results that arrived before waiter was created
            pending = self._pending_results.pop(order_id, None)
        if pending:
            for fill in pending:
                waiter.add_fill(**fill)
        return waiter

    def _deliver_result(self, order_id: int, fill: dict) -> None:
        """Deliver a fill to the waiter, or buffer it if waiter doesn't exist yet."""
        with self._lock:
            waiter = self._waiters.get(order_id)
            if waiter is None:
                # Waiter not yet created — buffer the result
                self._pending_results.setdefault(order_id, []).append(fill)
                return
        waiter.add_fill(**fill)

    def _deliver_error(self, order_id: int, error_msg: str) -> None:
        """Deliver an error to the waiter."""
        with self._lock:
            waiter = self._waiters.pop(order_id, None)
        if waiter:
            result = DealResult(order_id=order_id, error_msg=error_msg, success=False)
            waiter.set_result(result)

    # ---- XtQuantTraderCallback interface ----

    def on_disconnected(self) -> None:
        logger.warning("XtQuant trader disconnected")

    def on_stock_order(self, data: Any) -> None:
        """Called when order status changes."""
        logger.debug(f"Order status update: {data}")

    def on_order_error(self, data: Any) -> None:
        """Called when an order error occurs."""
        order_id = getattr(data, "order_id", 0)
        error_msg = getattr(data, "error_msg", str(data))
        logger.error(f"Order error for {order_id}: {error_msg}")
        self._deliver_error(order_id, error_msg)

    def on_order_stock_async_response(self, data: Any) -> None:
        """Called as async response to order_stock_async."""
        logger.debug(f"Async order response: {data}")

    def on_deal_order(self, data: Any) -> None:
        """Called when an order is (partially) dealt.

        This is the primary callback for trade execution confirmation.
        Supports partial fills by accumulating into the waiter.
        """
        order_id = getattr(data, "order_id", 0)
        stock_code = getattr(data, "stock_code", "")
        deal_price = getattr(data, "traded_price", 0.0)
        deal_amount = getattr(data, "traded_volume", 0)
        direction = getattr(data, "order_type", 0)

        logger.info(
            f"Deal: order_id={order_id}, stock={stock_code}, "
            f"price={deal_price}, amount={deal_amount}"
        )

        self._deliver_result(order_id, {
            "stock_code": stock_code,
            "direction": direction,
            "price": deal_price,
            "amount": deal_amount,
        })

    def on_cancel_order_stock_async_response(self, data: Any) -> None:
        logger.debug(f"Cancel order response: {data}")

    def on_account_status(self, data: Any) -> None:
        logger.debug(f"Account status: {data}")
