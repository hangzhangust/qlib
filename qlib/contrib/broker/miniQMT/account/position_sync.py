# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Synchronize broker positions with Qlib Position objects.

Critical for:
- Process restart recovery (rebuild state from broker)
- End-of-day reconciliation (detect discrepancies)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from qlib.backtest.position import Position
from qlib.log import get_module_logger
from ..utils import StockCodeConverter

logger = get_module_logger("PositionSync")


class PositionSynchronizer:
    """Synchronize positions between xtquant broker and Qlib.

    Parameters
    ----------
    xt_trader : XtQuantTrader
        Connected xtquant trader.
    account_id : str
        Trading account ID.
    """

    def __init__(self, xt_trader: Any, account_id: str):
        self.xt_trader = xt_trader
        self.account_id = account_id

    def fetch_broker_positions(self) -> Tuple[float, Dict[str, Dict[str, float]]]:
        """Fetch current positions from the broker.

        Returns
        -------
        Tuple[float, dict]
            (cash, {qlib_code: {"amount": float, "price": float}})
        """
        # Query account asset for cash
        asset = self.xt_trader.query_stock_asset(self.account_id)
        cash = getattr(asset, "cash", 0.0) if asset else 0.0

        # Query stock positions
        positions = self.xt_trader.query_stock_positions(self.account_id)
        position_dict = {}

        if positions:
            for pos in positions:
                xt_code = getattr(pos, "stock_code", "")
                volume = getattr(pos, "volume", 0)
                if volume > 0 and xt_code:
                    try:
                        qlib_code = StockCodeConverter.xt_to_qlib(xt_code)
                    except ValueError:
                        logger.warning(f"Skipping unrecognized code: {xt_code}")
                        continue
                    price = getattr(pos, "market_value", 0.0) / volume if volume > 0 else 0.0
                    position_dict[qlib_code] = {
                        "amount": float(volume),
                        "price": price,
                    }

        return cash, position_dict

    def build_qlib_position(self) -> Position:
        """Build a Qlib Position object from current broker state.

        Useful for process restart recovery.

        Returns
        -------
        Position
            A new Position object matching the broker's current state.
        """
        cash, position_dict = self.fetch_broker_positions()
        position = Position(cash=cash, position_dict=position_dict)
        logger.info(
            f"Built position from broker: cash={cash:.2f}, "
            f"stocks={len(position_dict)}"
        )
        return position

    def reconcile(self, qlib_position: Position) -> List[str]:
        """Compare Qlib position with broker position and report discrepancies.

        Parameters
        ----------
        qlib_position : Position
            The Qlib-side position to check against broker.

        Returns
        -------
        List[str]
            List of discrepancy descriptions. Empty if positions match.
        """
        broker_cash, broker_positions = self.fetch_broker_positions()
        discrepancies = []

        # Check cash
        qlib_cash = qlib_position.get_cash()
        if abs(broker_cash - qlib_cash) > 1.0:  # 1 yuan tolerance
            discrepancies.append(
                f"Cash mismatch: broker={broker_cash:.2f}, qlib={qlib_cash:.2f}"
            )

        # Check stock positions
        qlib_stocks = set(qlib_position.get_stock_list())
        broker_stocks = set(broker_positions.keys())

        for stock in qlib_stocks - broker_stocks:
            qlib_amount = qlib_position.get_stock_amount(stock)
            discrepancies.append(f"{stock}: in Qlib (amount={qlib_amount}) but not in broker")

        for stock in broker_stocks - qlib_stocks:
            broker_amount = broker_positions[stock]["amount"]
            discrepancies.append(f"{stock}: in broker (amount={broker_amount}) but not in Qlib")

        for stock in qlib_stocks & broker_stocks:
            qlib_amount = qlib_position.get_stock_amount(stock)
            broker_amount = broker_positions[stock]["amount"]
            if abs(qlib_amount - broker_amount) > 0.5:
                discrepancies.append(
                    f"{stock}: amount mismatch broker={broker_amount}, qlib={qlib_amount}"
                )

        if discrepancies:
            for d in discrepancies:
                logger.warning(f"Reconciliation: {d}")
        else:
            logger.info("Reconciliation passed: positions match")

        return discrepancies

    def force_sync(self, qlib_position: Position) -> None:
        """Force Qlib position to match broker state.

        WARNING: This overwrites the Qlib position. Use after confirming discrepancies.

        Parameters
        ----------
        qlib_position : Position
            The position to overwrite.
        """
        cash, broker_positions = self.fetch_broker_positions()

        # Clear and rebuild using Position internal methods (_del_stock, _init_stock)
        # since there is no public API for bulk position replacement.
        stock_list = qlib_position.get_stock_list()
        for stock in stock_list:
            qlib_position._del_stock(stock)

        qlib_position.position["cash"] = cash

        for code, info in broker_positions.items():
            qlib_position._init_stock(code, info["amount"], info.get("price"))

        qlib_position.position["now_account_value"] = qlib_position.calculate_value()
        logger.info(f"Force synced position: cash={cash:.2f}, stocks={len(broker_positions)}")
