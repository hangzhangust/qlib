# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Live executor that uses XtQMTLiveExchange for real trading.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, List, Tuple, Union

import numpy as np
import pandas as pd

from qlib.backtest.decision import BaseTradeDecision, Order
from qlib.backtest.executor import BaseExecutor, _retrieve_orders_from_decision
from qlib.backtest.utils import CommonInfrastructure
from qlib.log import get_module_logger

from .exchange_live import XtQMTLiveExchange

logger = get_module_logger("XtQMTLiveExecutor")


class XtQMTLiveExecutor(BaseExecutor):
    """Executor that sends orders to miniQMT for live execution.

    This is a drop-in replacement for SimulatorExecutor in live trading mode.
    It uses XtQMTLiveExchange.deal_order() which routes to the real broker.

    Parameters
    ----------
    time_per_step : str
        Trade time per step (e.g. "day").
    live_exchange : XtQMTLiveExchange
        The live exchange connected to miniQMT.
    trade_type : str
        "serial" to execute orders one by one (recommended for live).
    **kwargs
        Additional arguments for BaseExecutor.
    """

    TT_SERIAL = "serial"

    def __init__(
        self,
        time_per_step: str,
        live_exchange: XtQMTLiveExchange | None = None,
        start_time: Union[str, pd.Timestamp] = None,
        end_time: Union[str, pd.Timestamp] = None,
        indicator_config: dict = {},
        generate_portfolio_metrics: bool = False,
        verbose: bool = True,
        track_data: bool = False,
        common_infra: CommonInfrastructure | None = None,
        trade_type: str = TT_SERIAL,
        **kwargs: Any,
    ):
        self._live_exchange = live_exchange
        self.trade_type = trade_type

        super().__init__(
            time_per_step=time_per_step,
            start_time=start_time,
            end_time=end_time,
            indicator_config=indicator_config,
            generate_portfolio_metrics=generate_portfolio_metrics,
            verbose=verbose,
            track_data=track_data,
            trade_exchange=live_exchange,
            common_infra=common_infra,
            **kwargs,
        )

    @property
    def trade_exchange(self) -> XtQMTLiveExchange:
        if self._live_exchange is not None:
            return self._live_exchange
        return super().trade_exchange

    def _collect_data(self, trade_decision: BaseTradeDecision, level: int = 0) -> Tuple[List[object], dict]:
        """Execute all orders from the trade decision via live exchange.

        Parameters
        ----------
        trade_decision : BaseTradeDecision
            The trade decision containing orders.
        level : int
            Executor level.

        Returns
        -------
        Tuple[List[object], dict]
            (execute_result, {"trade_info": execute_result})
        """
        trade_start_time, _ = self.trade_calendar.get_step_time()
        execute_result: list = []

        orders = _retrieve_orders_from_decision(trade_decision)
        logger.info(f"Executing {len(orders)} orders at {trade_start_time}")

        for order in orders:
            now_deal_day = self.trade_calendar.get_step_time()[0].floor(freq="D")
            if self.deal_day is None or now_deal_day > self.deal_day:
                self.dealt_order_amount = defaultdict(float)
                self.deal_day = now_deal_day

            trade_val, trade_cost, trade_price = self.trade_exchange.deal_order(
                order,
                trade_account=self.trade_account,
                dealt_order_amount=self.dealt_order_amount,
            )
            execute_result.append((order, trade_val, trade_cost, trade_price))
            self.dealt_order_amount[order.stock_id] += order.deal_amount

            if self.verbose:
                direction_str = "SELL" if order.direction == Order.SELL else "BUY"
                logger.info(
                    f"[{trade_start_time:%Y-%m-%d %H:%M:%S}]: {direction_str} {order.stock_id}, "
                    f"price {trade_price:.2f}, amount {order.amount}, deal_amount {order.deal_amount}, "
                    f"value {trade_val:.2f}"
                )

        return execute_result, {"trade_info": execute_result}
