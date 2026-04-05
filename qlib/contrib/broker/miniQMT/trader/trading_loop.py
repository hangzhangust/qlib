# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Main trading loop for live trading with miniQMT.

Orchestrates: connect -> sync positions -> run strategy -> submit orders -> reconcile.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from qlib.backtest.decision import Order, OrderDir, TradeDecisionWO
from qlib.backtest.position import Position
from qlib.log import get_module_logger
from ..account.account_adapter import XtAccountAdapter
from ..account.position_sync import PositionSynchronizer
from ..executor.exchange_live import XtQMTLiveExchange
from ..trader.callback_handler import XtCallbackHandler
from ..trader.order_manager import XtOrderManager
from ..utils import is_trading_time, round_to_lot, TRADE_UNIT

logger = get_module_logger("TradingLoop")


class TradingLoop:
    """Main loop for live trading with Qlib strategy + miniQMT execution.

    Usage:
        loop = TradingLoop(
            mini_qmt_path="D:/国金证券QMT交易端/userdata_mini",
            account_id="1234567890",
        )
        loop.connect()
        loop.run_once(target_positions={"SH600000": 0.1, "SZ000001": 0.1})

    Parameters
    ----------
    mini_qmt_path : str
        Path to miniQMT userdata directory.
    account_id : str
        Trading account ID.
    order_timeout : float
        Seconds to wait for each order to complete.
    """

    def __init__(
        self,
        mini_qmt_path: str,
        account_id: str,
        order_timeout: float = 60.0,
    ):
        self.mini_qmt_path = mini_qmt_path
        self.account_id = account_id
        self.order_timeout = order_timeout

        self.xt_trader = None
        self.callback = None
        self.order_manager = None
        self.live_exchange = None
        self.position_sync = None
        self.account_adapter = None
        self._connected = False

    def connect(self) -> None:
        """Connect to miniQMT and initialize all components."""
        try:
            from xtquant.xttrader import XtQuantTrader
            from xtquant.xttype import StockAccount
        except ImportError:
            raise ImportError("xtquant is required. Please install it from your miniQMT distribution.")

        session_id = int(time.time())
        self.xt_trader = XtQuantTrader(self.mini_qmt_path, session_id)

        self.callback = XtCallbackHandler(order_timeout=self.order_timeout)
        self.xt_trader.register_callback(self.callback)

        self.xt_trader.start()
        connect_result = self.xt_trader.connect()
        if connect_result != 0:
            raise ConnectionError(f"Failed to connect to miniQMT (code={connect_result})")

        # Subscribe to account
        acc = StockAccount(self.account_id)
        subscribe_result = self.xt_trader.subscribe(acc)
        if subscribe_result != 0:
            logger.warning(f"Account subscribe returned {subscribe_result}")

        self.order_manager = XtOrderManager(
            xt_trader=self.xt_trader,
            account_id=self.account_id,
            callback=self.callback,
            order_timeout=self.order_timeout,
        )

        self.live_exchange = XtQMTLiveExchange(order_manager=self.order_manager)
        self.position_sync = PositionSynchronizer(self.xt_trader, self.account_id)
        self.account_adapter = XtAccountAdapter(self.xt_trader, self.account_id)

        self._connected = True
        logger.info(f"Connected to miniQMT, account={self.account_id}")

    def disconnect(self) -> None:
        """Disconnect from miniQMT."""
        if self.xt_trader:
            self.xt_trader.stop()
            self._connected = False
            logger.info("Disconnected from miniQMT")

    def run_once(
        self,
        target_positions: Dict[str, float],
        current_position: Optional[Position] = None,
        total_value: Optional[float] = None,
    ) -> List[tuple]:
        """Execute a single round of trading to reach target positions.

        Parameters
        ----------
        target_positions : dict
            Target weight for each stock. {qlib_code: weight}.
            Weights should sum to <= 1.0 (remainder is cash).
        current_position : Position, optional
            Current Qlib position. If None, fetches from broker.
        total_value : float, optional
            Total portfolio value for calculating amounts. If None, uses broker total asset.

        Returns
        -------
        List[tuple]
            List of (order, trade_val, trade_cost, trade_price) for each executed order.
        """
        if not self._connected:
            raise RuntimeError("Not connected. Call connect() first.")

        now = datetime.now()
        if not is_trading_time(now.hour, now.minute):
            logger.warning(f"Current time {now:%H:%M} is outside trading hours")

        if current_position is None:
            current_position = self.position_sync.build_qlib_position()

        if total_value is None:
            total_value = self.account_adapter.get_total_asset()

        orders = self._generate_orders(target_positions, current_position, total_value)
        if not orders:
            logger.info("No orders to execute")
            return []

        # Execute sell orders first, then buy orders
        sell_orders = [o for o in orders if o.direction == OrderDir.SELL]
        buy_orders = [o for o in orders if o.direction == OrderDir.BUY]

        results = []
        now_ts = pd.Timestamp.now()

        for order in sell_orders + buy_orders:
            order.start_time = now_ts
            order.end_time = now_ts
            trade_val, trade_cost, trade_price = self.live_exchange.deal_order(
                order, position=current_position
            )
            results.append((order, trade_val, trade_cost, trade_price))

        # Reconcile
        discrepancies = self.position_sync.reconcile(current_position)
        if discrepancies:
            logger.warning(f"Post-trade reconciliation found {len(discrepancies)} discrepancies")

        return results

    def _generate_orders(
        self,
        target_positions: Dict[str, float],
        current_position: Position,
        total_value: float,
    ) -> List[Order]:
        """Generate orders to move from current to target position.

        Parameters
        ----------
        target_positions : dict
            {stock_code: target_weight}
        current_position : Position
            Current position state.
        total_value : float
            Total portfolio value.

        Returns
        -------
        List[Order]
            Orders to execute.
        """
        orders = []
        current_stocks = set(current_position.get_stock_list())
        target_stocks = set(target_positions.keys())

        # Stocks to sell (reduce or exit)
        for stock in current_stocks:
            current_amount = current_position.get_stock_amount(stock)
            current_price = current_position.get_stock_price(stock)

            target_weight = target_positions.get(stock, 0.0)
            if current_price > 0:
                target_amount = round_to_lot(total_value * target_weight / current_price, TRADE_UNIT)
            else:
                target_amount = 0

            diff = current_amount - target_amount
            sell_amount = round_to_lot(diff, TRADE_UNIT)
            if sell_amount >= TRADE_UNIT:
                orders.append(Order(
                    stock_id=stock,
                    amount=sell_amount,
                    direction=OrderDir.SELL,
                    start_time=pd.Timestamp.now(),
                    end_time=pd.Timestamp.now(),
                ))

        # Stocks to buy (increase or enter)
        for stock in target_stocks:
            target_weight = target_positions[stock]
            if target_weight <= 0:
                continue

            current_amount = current_position.get_stock_amount(stock) if stock in current_stocks else 0

            # Estimate price from current position or use a placeholder
            if stock in current_stocks:
                price = current_position.get_stock_price(stock)
            else:
                price = self._get_latest_price(stock)

            if price is None or price <= 0:
                logger.warning(f"Cannot get price for {stock}, skipping buy order")
                continue

            target_amount = round_to_lot(total_value * target_weight / price, TRADE_UNIT)
            diff = target_amount - current_amount
            buy_amount = round_to_lot(diff, TRADE_UNIT)
            if buy_amount >= TRADE_UNIT:
                orders.append(Order(
                    stock_id=stock,
                    amount=buy_amount,
                    direction=OrderDir.BUY,
                    start_time=pd.Timestamp.now(),
                    end_time=pd.Timestamp.now(),
                ))

        return orders

    def _get_latest_price(self, stock_id: str) -> Optional[float]:
        """Get latest price for a stock from xtdata."""
        try:
            from xtquant import xtdata
            from ..utils import StockCodeConverter

            xt_code = StockCodeConverter.qlib_to_xt(stock_id)
            tick = xtdata.get_full_tick([xt_code])
            if xt_code in tick and tick[xt_code]:
                return tick[xt_code].get("lastPrice", None)
        except Exception as e:
            logger.debug(f"Failed to get price for {stock_id}: {e}")
        return None
