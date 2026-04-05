# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Tests for qlib.contrib.broker.miniQMT module.

These tests run without xtquant installed by mocking the xtquant dependencies.
They verify: stock code conversion, trading utilities, callback handler threading,
data collector file I/O, order manager logic, position synchronization,
live exchange T+1 compliance, and order generation.
"""

from __future__ import annotations

import sys
import tempfile
import threading
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from qlib.contrib.broker.miniQMT.utils import (
    StockCodeConverter,
    round_to_lot,
    is_trading_time,
    TRADE_UNIT,
)


# ============================================================
# StockCodeConverter
# ============================================================

class TestStockCodeConverter:
    def test_qlib_to_xt(self):
        assert StockCodeConverter.qlib_to_xt("SH600000") == "600000.SH"
        assert StockCodeConverter.qlib_to_xt("SZ000001") == "000001.SZ"

    def test_xt_to_qlib(self):
        assert StockCodeConverter.xt_to_qlib("600000.SH") == "SH600000"
        assert StockCodeConverter.xt_to_qlib("000001.SZ") == "SZ000001"

    def test_batch_conversion(self):
        qlib_codes = ["SH600000", "SZ000001"]
        xt_codes = ["600000.SH", "000001.SZ"]
        assert StockCodeConverter.qlib_to_xt_batch(qlib_codes) == xt_codes
        assert StockCodeConverter.xt_to_qlib_batch(xt_codes) == qlib_codes

    def test_invalid_qlib_code(self):
        with pytest.raises(ValueError):
            StockCodeConverter.qlib_to_xt("INVALID")

    def test_invalid_xt_code(self):
        with pytest.raises(ValueError):
            StockCodeConverter.xt_to_qlib("INVALID")


# ============================================================
# Trading utilities
# ============================================================

class TestTradingUtils:
    def test_round_to_lot(self):
        assert round_to_lot(150, 100) == 100
        assert round_to_lot(250, 100) == 200
        assert round_to_lot(99, 100) == 0
        assert round_to_lot(1000, 100) == 1000
        assert round_to_lot(0, 100) == 0

    def test_is_trading_time_morning(self):
        assert is_trading_time(9, 30) is True
        assert is_trading_time(10, 0) is True
        assert is_trading_time(11, 30) is True

    def test_is_trading_time_afternoon(self):
        assert is_trading_time(13, 0) is True
        assert is_trading_time(14, 0) is True
        assert is_trading_time(15, 0) is True

    def test_is_trading_time_outside(self):
        assert is_trading_time(8, 0) is False
        assert is_trading_time(9, 29) is False
        assert is_trading_time(11, 31) is False
        assert is_trading_time(12, 0) is False
        assert is_trading_time(12, 59) is False
        assert is_trading_time(15, 1) is False
        assert is_trading_time(16, 0) is False

    def test_trade_unit_constant(self):
        assert TRADE_UNIT == 100


# ============================================================
# Callback handler
# ============================================================

class TestCallbackHandler:
    def test_deal_result_defaults(self):
        from qlib.contrib.broker.miniQMT.trader.callback_handler import DealResult

        r = DealResult()
        assert r.success is False
        assert r.deal_amount == 0
        assert r.deal_price == 0.0

    def test_order_waiter_full_fill(self):
        from qlib.contrib.broker.miniQMT.trader.callback_handler import OrderWaiter

        waiter = OrderWaiter(order_id=1, timeout=2.0, target_amount=200)
        waiter.add_fill("600000.SH", 1, 10.5, 100)
        waiter.add_fill("600000.SH", 1, 10.6, 100)
        result = waiter.wait()
        assert result.success is True
        assert result.deal_amount == 200
        assert abs(result.deal_price - 10.55) < 0.01

    def test_order_waiter_timeout(self):
        from qlib.contrib.broker.miniQMT.trader.callback_handler import OrderWaiter

        waiter = OrderWaiter(order_id=2, timeout=0.1, target_amount=100)
        result = waiter.wait()
        assert result.success is False
        assert "timed out" in result.error_msg

    def test_order_waiter_partial_fill_timeout(self):
        from qlib.contrib.broker.miniQMT.trader.callback_handler import OrderWaiter

        waiter = OrderWaiter(order_id=3, timeout=0.1, target_amount=200)
        waiter.add_fill("600000.SH", 1, 10.0, 50)
        result = waiter.wait()
        assert result.success is True
        assert result.deal_amount == 50

    def test_handler_async_callback(self):
        from qlib.contrib.broker.miniQMT.trader.callback_handler import XtCallbackHandler

        handler = XtCallbackHandler(order_timeout=1.0)
        waiter = handler.create_waiter(order_id=100, target_amount=100)

        def simulate():
            import time
            time.sleep(0.05)
            handler.on_deal_order(
                type("D", (), {"order_id": 100, "stock_code": "600000.SH", "traded_price": 11.0, "traded_volume": 100, "order_type": 1})()
            )

        t = threading.Thread(target=simulate)
        t.start()
        result = waiter.wait()
        t.join()
        assert result.success is True
        assert result.deal_amount == 100

    def test_handler_pending_results(self):
        from qlib.contrib.broker.miniQMT.trader.callback_handler import XtCallbackHandler

        handler = XtCallbackHandler(order_timeout=1.0)
        # Callback arrives BEFORE waiter creation
        handler.on_deal_order(
            type("D", (), {"order_id": 200, "stock_code": "000001.SZ", "traded_price": 15.0, "traded_volume": 100, "order_type": 1})()
        )
        waiter = handler.create_waiter(order_id=200, target_amount=100)
        result = waiter.wait()
        assert result.success is True

    def test_handler_error_delivery(self):
        from qlib.contrib.broker.miniQMT.trader.callback_handler import XtCallbackHandler

        handler = XtCallbackHandler(order_timeout=1.0)
        waiter = handler.create_waiter(order_id=300, target_amount=100)
        handler.on_order_error(type("D", (), {"order_id": 300, "error_msg": "Insufficient funds"})())
        result = waiter.wait()
        assert result.success is False
        assert "Insufficient funds" in result.error_msg


# ============================================================
# Data collector
# ============================================================

class TestXtDataCollector:
    def test_init(self):
        from qlib.contrib.broker.miniQMT.data.data_collector import XtDataCollector

        collector = XtDataCollector(target_dir=tempfile.mkdtemp(), freq="1d")
        assert collector.freq == "1d"

    def test_write_bin_file(self):
        from qlib.contrib.broker.miniQMT.data.data_collector import XtDataCollector

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.bin"
            timestamps = pd.DatetimeIndex(["2023-01-03", "2023-01-04", "2023-01-05"])
            values = np.array([10.0, 11.0, 12.0], dtype=np.float32)
            cal_index = {pd.Timestamp("2023-01-03"): 0, pd.Timestamp("2023-01-04"): 1, pd.Timestamp("2023-01-05"): 2}
            XtDataCollector._write_bin_file(filepath, timestamps, values, cal_index, 3)

            data = np.fromfile(filepath, dtype="<f")
            assert data[0] == 0.0
            assert abs(data[1] - 10.0) < 0.01
            assert abs(data[2] - 11.0) < 0.01
            assert abs(data[3] - 12.0) < 0.01

    def test_write_bin_file_with_gaps(self):
        from qlib.contrib.broker.miniQMT.data.data_collector import XtDataCollector

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_gap.bin"
            timestamps = pd.DatetimeIndex(["2023-01-03", "2023-01-05"])
            values = np.array([10.0, 12.0], dtype=np.float32)
            cal_index = {pd.Timestamp("2023-01-03"): 0, pd.Timestamp("2023-01-04"): 1, pd.Timestamp("2023-01-05"): 2}
            XtDataCollector._write_bin_file(filepath, timestamps, values, cal_index, 3)

            data = np.fromfile(filepath, dtype="<f")
            assert abs(data[1] - 10.0) < 0.01
            assert np.isnan(data[2])  # gap
            assert abs(data[3] - 12.0) < 0.01

    def test_save_and_read_calendar(self):
        from qlib.contrib.broker.miniQMT.data.data_collector import XtDataCollector

        with tempfile.TemporaryDirectory() as tmpdir:
            collector = XtDataCollector(target_dir=tmpdir)
            calendar = [pd.Timestamp("2023-01-03"), pd.Timestamp("2023-01-04")]
            collector._save_calendar(calendar)

            cal = collector._read_existing_calendar()
            assert len(cal) == 2
            assert cal[0] == pd.Timestamp("2023-01-03")

    def test_save_and_read_instruments(self):
        from qlib.contrib.broker.miniQMT.data.data_collector import XtDataCollector

        with tempfile.TemporaryDirectory() as tmpdir:
            collector = XtDataCollector(target_dir=tmpdir)
            instruments = {"SH600000": (pd.Timestamp("2023-01-03"), pd.Timestamp("2023-01-04"))}
            collector._save_instruments(instruments)

            stocks = collector._read_existing_stock_list()
            assert stocks == ["600000.SH"]

    def test_end_to_end_collect(self):
        from qlib.contrib.broker.miniQMT.data.data_collector import XtDataCollector

        # Setup mock xtquant
        mock_xtquant = types.ModuleType("xtquant")
        mock_xtdata = MagicMock()
        mock_xtquant.xtdata = mock_xtdata
        sys.modules["xtquant"] = mock_xtquant
        sys.modules["xtquant.xtdata"] = mock_xtdata

        try:
            stock_data = {
                "600000.SH": pd.DataFrame(
                    {
                        "open": [10.0, 10.1],
                        "high": [10.5, 10.6],
                        "low": [9.8, 9.9],
                        "close": [10.2, 10.4],
                        "volume": [1e6, 1.1e6],
                        "amount": [1e7, 1.1e7],
                        "preClose": [10.0, 10.2],
                    },
                    index=["20230103", "20230104"],
                ),
            }
            mock_xtdata.get_market_data_ex.side_effect = lambda stock_list, period, start_time, end_time, fields: {
                code: stock_data[code] for code in stock_list if code in stock_data
            }
            mock_xtdata.download_history_data2 = MagicMock()

            with tempfile.TemporaryDirectory() as tmpdir:
                collector = XtDataCollector(target_dir=tmpdir)
                collector.collect(stock_list=["600000.SH"], start_date="20230103", end_date="20230104")

                assert (Path(tmpdir) / "calendars" / "day.txt").exists()
                assert (Path(tmpdir) / "instruments" / "all.txt").exists()
                assert (Path(tmpdir) / "features" / "SH600000" / "$close.bin").exists()

                close_data = np.fromfile(Path(tmpdir) / "features" / "SH600000" / "$close.bin", dtype="<f")
                assert abs(close_data[1] - 10.2) < 0.01
                assert abs(close_data[2] - 10.4) < 0.01
        finally:
            sys.modules.pop("xtquant", None)
            sys.modules.pop("xtquant.xtdata", None)


# ============================================================
# Account adapter and position sync
# ============================================================

class TestAccountAdapter:
    def _make_mock_trader(self):
        mock_trader = MagicMock()
        mock_asset = MagicMock()
        mock_asset.total_asset = 1000000.0
        mock_asset.cash = 500000.0
        mock_asset.frozen_cash = 10000.0
        mock_asset.market_value = 490000.0
        mock_trader.query_stock_asset.return_value = mock_asset

        mock_pos = MagicMock()
        mock_pos.stock_code = "600000.SH"
        mock_pos.volume = 1000
        mock_pos.can_use_volume = 800
        mock_pos.open_price = 10.5
        mock_pos.market_value = 10500.0
        mock_pos.profit = 500.0
        mock_trader.query_stock_positions.return_value = [mock_pos]
        return mock_trader

    def test_get_total_asset(self):
        from qlib.contrib.broker.miniQMT.account.account_adapter import XtAccountAdapter

        adapter = XtAccountAdapter(self._make_mock_trader(), "123")
        assert adapter.get_total_asset() == 1000000.0

    def test_get_cash(self):
        from qlib.contrib.broker.miniQMT.account.account_adapter import XtAccountAdapter

        adapter = XtAccountAdapter(self._make_mock_trader(), "123")
        assert adapter.get_cash() == 500000.0

    def test_get_snapshot(self):
        from qlib.contrib.broker.miniQMT.account.account_adapter import XtAccountAdapter

        adapter = XtAccountAdapter(self._make_mock_trader(), "123")
        snapshot = adapter.get_snapshot()
        assert "SH600000" in snapshot.positions
        assert snapshot.positions["SH600000"].volume == 1000
        assert snapshot.positions["SH600000"].sellable_volume == 800

    def test_get_sellable_amount(self):
        from qlib.contrib.broker.miniQMT.account.account_adapter import XtAccountAdapter

        adapter = XtAccountAdapter(self._make_mock_trader(), "123")
        assert adapter.get_sellable_amount("SH600000") == 800


class TestPositionSynchronizer:
    def _make_mock_trader(self):
        mock_trader = MagicMock()
        mock_asset = MagicMock()
        mock_asset.cash = 500000.0
        mock_trader.query_stock_asset.return_value = mock_asset

        mock_pos = MagicMock()
        mock_pos.stock_code = "600000.SH"
        mock_pos.volume = 1000
        mock_pos.market_value = 10500.0
        mock_trader.query_stock_positions.return_value = [mock_pos]
        return mock_trader

    def test_fetch_broker_positions(self):
        from qlib.contrib.broker.miniQMT.account.position_sync import PositionSynchronizer

        sync = PositionSynchronizer(self._make_mock_trader(), "123")
        cash, positions = sync.fetch_broker_positions()
        assert cash == 500000.0
        assert "SH600000" in positions
        assert positions["SH600000"]["amount"] == 1000

    def test_reconcile_match(self):
        from qlib.contrib.broker.miniQMT.account.position_sync import PositionSynchronizer
        from qlib.backtest.position import Position

        sync = PositionSynchronizer(self._make_mock_trader(), "123")
        pos = Position(cash=500000.0, position_dict={"SH600000": {"amount": 1000, "price": 10.5}})
        assert len(sync.reconcile(pos)) == 0

    def test_reconcile_mismatch(self):
        from qlib.contrib.broker.miniQMT.account.position_sync import PositionSynchronizer
        from qlib.backtest.position import Position

        sync = PositionSynchronizer(self._make_mock_trader(), "123")
        pos = Position(cash=500000.0, position_dict={"SH600000": {"amount": 500, "price": 10.5}})
        discrepancies = sync.reconcile(pos)
        assert len(discrepancies) > 0


# ============================================================
# Live exchange
# ============================================================

class TestXtQMTLiveExchange:
    def test_deal_order_buy(self):
        from qlib.contrib.broker.miniQMT.executor.exchange_live import XtQMTLiveExchange
        from qlib.contrib.broker.miniQMT.trader.callback_handler import DealResult
        from qlib.contrib.broker.miniQMT.trader.order_manager import XtOrderManager
        from qlib.backtest.decision import Order, OrderDir

        mock_mgr = MagicMock(spec=XtOrderManager)
        mock_mgr.submit_order.return_value = DealResult(success=True, deal_price=10.5, deal_amount=100, deal_value=1050.0)

        exchange = XtQMTLiveExchange(order_manager=mock_mgr)
        order = Order(stock_id="SH600000", amount=100, direction=OrderDir.BUY, start_time=pd.Timestamp.now(), end_time=pd.Timestamp.now())
        trade_val, trade_cost, trade_price = exchange.deal_order(order)

        assert trade_val == 1050.0
        assert trade_price == 10.5
        assert trade_cost >= 5.0

    def test_deal_order_sell_t_plus_1_block(self):
        from qlib.contrib.broker.miniQMT.executor.exchange_live import XtQMTLiveExchange
        from qlib.contrib.broker.miniQMT.trader.order_manager import XtOrderManager
        from qlib.backtest.decision import Order, OrderDir

        mock_mgr = MagicMock(spec=XtOrderManager)
        mock_mgr.query_sellable_amount.return_value = 0

        exchange = XtQMTLiveExchange(order_manager=mock_mgr)
        order = Order(stock_id="SH600000", amount=100, direction=OrderDir.SELL, start_time=pd.Timestamp.now(), end_time=pd.Timestamp.now())
        trade_val, trade_cost, trade_price = exchange.deal_order(order)

        assert trade_val == 0.0
        assert np.isnan(trade_price)

    def test_check_order_always_true(self):
        from qlib.contrib.broker.miniQMT.executor.exchange_live import XtQMTLiveExchange
        from qlib.contrib.broker.miniQMT.trader.order_manager import XtOrderManager
        from qlib.backtest.decision import Order, OrderDir

        exchange = XtQMTLiveExchange(order_manager=MagicMock(spec=XtOrderManager))
        order = Order(stock_id="SH600000", amount=100, direction=OrderDir.BUY, start_time=pd.Timestamp.now(), end_time=pd.Timestamp.now())
        assert exchange.check_order(order) is True


# ============================================================
# Trading loop order generation
# ============================================================

class TestTradingLoopOrderGeneration:
    def test_generate_buy_orders(self):
        from qlib.contrib.broker.miniQMT.trader.trading_loop import TradingLoop
        from qlib.backtest.decision import OrderDir
        from qlib.backtest.position import Position

        loop = TradingLoop(mini_qmt_path="D:/test", account_id="123")
        loop._connected = True
        loop._get_latest_price = lambda s: 15.0

        pos = Position(cash=1000000.0, position_dict={})
        orders = loop._generate_orders({"SZ000001": 0.05}, pos, 1000000.0)

        buy_orders = [o for o in orders if o.direction == OrderDir.BUY]
        assert len(buy_orders) == 1
        assert buy_orders[0].stock_id == "SZ000001"
        # target = round_to_lot(1000000 * 0.05 / 15.0) = round_to_lot(3333) = 3300
        assert buy_orders[0].amount == 3300

    def test_generate_sell_orders(self):
        from qlib.contrib.broker.miniQMT.trader.trading_loop import TradingLoop
        from qlib.backtest.decision import OrderDir
        from qlib.backtest.position import Position

        loop = TradingLoop(mini_qmt_path="D:/test", account_id="123")
        loop._connected = True

        pos = Position(cash=900000.0, position_dict={"SH600000": {"amount": 1000, "price": 10.0}})
        # Target has no SH600000 -> sell all
        orders = loop._generate_orders({"SZ000001": 0.1}, pos, 1000000.0)

        sell_orders = [o for o in orders if o.direction == OrderDir.SELL]
        assert len(sell_orders) == 1
        assert sell_orders[0].stock_id == "SH600000"
        assert sell_orders[0].amount == 1000

    def test_not_connected_raises(self):
        from qlib.contrib.broker.miniQMT.trader.trading_loop import TradingLoop

        loop = TradingLoop(mini_qmt_path="D:/test", account_id="123")
        with pytest.raises(RuntimeError, match="Not connected"):
            loop.run_once(target_positions={})


# ============================================================
# Order manager (requires mock xtquant)
# ============================================================

class TestXtOrderManager:
    def setup_method(self):
        mock_xtconstant = types.ModuleType("xtquant.xtconstant")
        mock_xtconstant.STOCK_BUY = 23
        mock_xtconstant.STOCK_SELL = 24
        mock_xtconstant.LATEST_PRICE = 5
        sys.modules["xtquant"] = types.ModuleType("xtquant")
        sys.modules["xtquant.xtconstant"] = mock_xtconstant

    def teardown_method(self):
        sys.modules.pop("xtquant", None)
        sys.modules.pop("xtquant.xtconstant", None)

    def test_submit_order_success(self):
        from qlib.contrib.broker.miniQMT.trader.order_manager import XtOrderManager
        from qlib.contrib.broker.miniQMT.trader.callback_handler import XtCallbackHandler
        from qlib.backtest.decision import Order, OrderDir

        mock_trader = MagicMock()
        callback = XtCallbackHandler(order_timeout=1.0)
        mgr = XtOrderManager(xt_trader=mock_trader, account_id="123", callback=callback, order_timeout=1.0)

        mock_trader.order_stock.return_value = 1001

        def simulate():
            import time
            time.sleep(0.05)
            callback.on_deal_order(
                type("D", (), {"order_id": 1001, "stock_code": "600000.SH", "traded_price": 10.5, "traded_volume": 100, "order_type": 23})()
            )

        t = threading.Thread(target=simulate)
        t.start()
        order = Order(stock_id="SH600000", amount=100, direction=OrderDir.BUY, start_time=pd.Timestamp.now(), end_time=pd.Timestamp.now())
        result = mgr.submit_order(order)
        t.join()

        assert result.success is True
        assert result.deal_amount == 100

    def test_submit_order_amount_zero(self):
        from qlib.contrib.broker.miniQMT.trader.order_manager import XtOrderManager
        from qlib.contrib.broker.miniQMT.trader.callback_handler import XtCallbackHandler
        from qlib.backtest.decision import Order, OrderDir

        mgr = XtOrderManager(xt_trader=MagicMock(), account_id="123", callback=XtCallbackHandler(), order_timeout=1.0)
        order = Order(stock_id="SH600000", amount=50, direction=OrderDir.BUY, start_time=pd.Timestamp.now(), end_time=pd.Timestamp.now())
        result = mgr.submit_order(order)
        assert result.success is False

    def test_cancel_order(self):
        from qlib.contrib.broker.miniQMT.trader.order_manager import XtOrderManager
        from qlib.contrib.broker.miniQMT.trader.callback_handler import XtCallbackHandler

        mock_trader = MagicMock()
        mgr = XtOrderManager(xt_trader=mock_trader, account_id="123", callback=XtCallbackHandler())
        mock_trader.cancel_order_stock.return_value = 0
        assert mgr.cancel_order(1001) is True
        mock_trader.cancel_order_stock.return_value = -1
        assert mgr.cancel_order(1001) is False
