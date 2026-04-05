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


# ============================================================
# XtQMTLiveExchange extended tests
# ============================================================

class TestXtQMTLiveExchangeExtended:
    def _make_exchange(self, **mgr_overrides):
        from qlib.contrib.broker.miniQMT.executor.exchange_live import XtQMTLiveExchange
        from qlib.contrib.broker.miniQMT.trader.order_manager import XtOrderManager

        mock_mgr = MagicMock(spec=XtOrderManager)
        for k, v in mgr_overrides.items():
            setattr(mock_mgr, k, v) if not callable(v) else setattr(type(mock_mgr), k, property(lambda s, val=v: val))
        # Default: configure return values on mock_mgr methods
        return XtQMTLiveExchange(order_manager=mock_mgr), mock_mgr

    def test_deal_order_sell_partial_sellable(self):
        from qlib.contrib.broker.miniQMT.executor.exchange_live import XtQMTLiveExchange
        from qlib.contrib.broker.miniQMT.trader.callback_handler import DealResult
        from qlib.contrib.broker.miniQMT.trader.order_manager import XtOrderManager
        from qlib.backtest.decision import Order, OrderDir

        mock_mgr = MagicMock(spec=XtOrderManager)
        mock_mgr.query_sellable_amount.return_value = 150
        mock_mgr.submit_order.return_value = DealResult(success=True, deal_price=10.0, deal_amount=100, deal_value=1000.0)

        exchange = XtQMTLiveExchange(order_manager=mock_mgr)
        order = Order(stock_id="SH600000", amount=300, direction=OrderDir.SELL, start_time=pd.Timestamp.now(), end_time=pd.Timestamp.now())
        trade_val, trade_cost, trade_price = exchange.deal_order(order)

        # sellable=150 -> round_to_lot(min(300,150))=100 -> submitted as 100
        assert mock_mgr.submit_order.called
        submitted_order = mock_mgr.submit_order.call_args[0][0]
        assert submitted_order.amount == 100
        assert trade_val == 1000.0

    def test_deal_order_buy_rounds_to_zero(self):
        from qlib.contrib.broker.miniQMT.executor.exchange_live import XtQMTLiveExchange
        from qlib.contrib.broker.miniQMT.trader.order_manager import XtOrderManager
        from qlib.backtest.decision import Order, OrderDir

        mock_mgr = MagicMock(spec=XtOrderManager)
        exchange = XtQMTLiveExchange(order_manager=mock_mgr)
        order = Order(stock_id="SH600000", amount=50, direction=OrderDir.BUY, start_time=pd.Timestamp.now(), end_time=pd.Timestamp.now())
        trade_val, trade_cost, trade_price = exchange.deal_order(order)

        assert trade_val == 0.0
        assert trade_cost == 0.0
        assert np.isnan(trade_price)
        mock_mgr.submit_order.assert_not_called()

    def test_deal_order_failed_submission(self):
        from qlib.contrib.broker.miniQMT.executor.exchange_live import XtQMTLiveExchange
        from qlib.contrib.broker.miniQMT.trader.callback_handler import DealResult
        from qlib.contrib.broker.miniQMT.trader.order_manager import XtOrderManager
        from qlib.backtest.decision import Order, OrderDir

        mock_mgr = MagicMock(spec=XtOrderManager)
        mock_mgr.submit_order.return_value = DealResult(success=False, error_msg="Rejected")

        exchange = XtQMTLiveExchange(order_manager=mock_mgr)
        order = Order(stock_id="SH600000", amount=100, direction=OrderDir.BUY, start_time=pd.Timestamp.now(), end_time=pd.Timestamp.now())
        trade_val, trade_cost, trade_price = exchange.deal_order(order)

        assert trade_val == 0.0
        assert np.isnan(trade_price)

    def test_deal_order_updates_position(self):
        from qlib.contrib.broker.miniQMT.executor.exchange_live import XtQMTLiveExchange
        from qlib.contrib.broker.miniQMT.trader.callback_handler import DealResult
        from qlib.contrib.broker.miniQMT.trader.order_manager import XtOrderManager
        from qlib.backtest.decision import Order, OrderDir

        mock_mgr = MagicMock(spec=XtOrderManager)
        mock_mgr.submit_order.return_value = DealResult(success=True, deal_price=10.0, deal_amount=100, deal_value=1000.0)

        exchange = XtQMTLiveExchange(order_manager=mock_mgr)
        mock_position = MagicMock()
        order = Order(stock_id="SH600000", amount=100, direction=OrderDir.BUY, start_time=pd.Timestamp.now(), end_time=pd.Timestamp.now())
        exchange.deal_order(order, position=mock_position)

        mock_position.update_order.assert_called_once()
        call_kwargs = mock_position.update_order.call_args[1]
        assert call_kwargs["trade_val"] == 1000.0
        assert call_kwargs["trade_price"] == 10.0

    def test_deal_order_updates_trade_account(self):
        from qlib.contrib.broker.miniQMT.executor.exchange_live import XtQMTLiveExchange
        from qlib.contrib.broker.miniQMT.trader.callback_handler import DealResult
        from qlib.contrib.broker.miniQMT.trader.order_manager import XtOrderManager
        from qlib.backtest.decision import Order, OrderDir

        mock_mgr = MagicMock(spec=XtOrderManager)
        mock_mgr.submit_order.return_value = DealResult(success=True, deal_price=10.0, deal_amount=100, deal_value=1000.0)

        exchange = XtQMTLiveExchange(order_manager=mock_mgr)
        mock_account = MagicMock()
        order = Order(stock_id="SH600000", amount=100, direction=OrderDir.BUY, start_time=pd.Timestamp.now(), end_time=pd.Timestamp.now())
        exchange.deal_order(order, trade_account=mock_account)

        mock_account.update_order.assert_called_once()
        call_kwargs = mock_account.update_order.call_args[1]
        assert call_kwargs["trade_val"] == 1000.0

    def test_deal_order_both_account_and_position_raises(self):
        from qlib.contrib.broker.miniQMT.executor.exchange_live import XtQMTLiveExchange
        from qlib.contrib.broker.miniQMT.trader.order_manager import XtOrderManager
        from qlib.backtest.decision import Order, OrderDir

        exchange = XtQMTLiveExchange(order_manager=MagicMock(spec=XtOrderManager))
        order = Order(stock_id="SH600000", amount=100, direction=OrderDir.BUY, start_time=pd.Timestamp.now(), end_time=pd.Timestamp.now())

        with pytest.raises(ValueError):
            exchange.deal_order(order, trade_account=MagicMock(), position=MagicMock())

    def test_commission_min_cost_for_buy(self):
        from qlib.contrib.broker.miniQMT.executor.exchange_live import XtQMTLiveExchange
        from qlib.contrib.broker.miniQMT.trader.callback_handler import DealResult
        from qlib.contrib.broker.miniQMT.trader.order_manager import XtOrderManager
        from qlib.backtest.decision import Order, OrderDir

        mock_mgr = MagicMock(spec=XtOrderManager)
        # trade_val = 10000 -> raw_cost = 10000 * 0.00025 = 2.5 < min_cost 5.0 -> cost = 5.0
        mock_mgr.submit_order.return_value = DealResult(success=True, deal_price=100.0, deal_amount=100, deal_value=10000.0)

        exchange = XtQMTLiveExchange(order_manager=mock_mgr, open_cost=0.00025, min_cost=5.0)
        order = Order(stock_id="SH600000", amount=100, direction=OrderDir.BUY, start_time=pd.Timestamp.now(), end_time=pd.Timestamp.now())
        trade_val, trade_cost, trade_price = exchange.deal_order(order)

        assert trade_cost == 5.0

    def test_commission_close_cost_for_sell(self):
        from qlib.contrib.broker.miniQMT.executor.exchange_live import XtQMTLiveExchange
        from qlib.contrib.broker.miniQMT.trader.callback_handler import DealResult
        from qlib.contrib.broker.miniQMT.trader.order_manager import XtOrderManager
        from qlib.backtest.decision import Order, OrderDir

        mock_mgr = MagicMock(spec=XtOrderManager)
        mock_mgr.query_sellable_amount.return_value = 200
        # trade_val = 10000 -> cost = 10000 * 0.00125 = 12.5
        mock_mgr.submit_order.return_value = DealResult(success=True, deal_price=100.0, deal_amount=100, deal_value=10000.0)

        exchange = XtQMTLiveExchange(order_manager=mock_mgr, close_cost=0.00125, min_cost=5.0)
        order = Order(stock_id="SH600000", amount=100, direction=OrderDir.SELL, start_time=pd.Timestamp.now(), end_time=pd.Timestamp.now())
        trade_val, trade_cost, trade_price = exchange.deal_order(order)

        assert abs(trade_cost - 12.5) < 0.01

    def test_get_close_mocked_xtdata(self):
        mock_xtquant = types.ModuleType("xtquant")
        mock_xtdata = MagicMock()
        mock_xtquant.xtdata = mock_xtdata
        sys.modules["xtquant"] = mock_xtquant
        sys.modules["xtquant.xtdata"] = mock_xtdata
        mock_xtdata.get_full_tick.return_value = {"600000.SH": {"lastPrice": 10.5}}

        try:
            from qlib.contrib.broker.miniQMT.executor.exchange_live import XtQMTLiveExchange
            from qlib.contrib.broker.miniQMT.trader.order_manager import XtOrderManager

            exchange = XtQMTLiveExchange(order_manager=MagicMock(spec=XtOrderManager))
            price = exchange.get_close("SH600000", None, None)
            assert price == 10.5
        finally:
            sys.modules.pop("xtquant", None)
            sys.modules.pop("xtquant.xtdata", None)


# ============================================================
# XtQMTLiveExecutor tests
# ============================================================

class TestXtQMTLiveExecutor:
    def _make_executor(self):
        from unittest.mock import patch, PropertyMock
        from qlib.contrib.broker.miniQMT.executor.live_executor import XtQMTLiveExecutor
        from qlib.backtest.executor import BaseExecutor
        from collections import defaultdict

        mock_exchange = MagicMock()
        mock_calendar = MagicMock()
        mock_calendar.get_step_time.return_value = (pd.Timestamp("2026-04-05 10:00:00"), pd.Timestamp("2026-04-05 15:00:00"))

        # Build a mock level_infra that returns our mock_calendar
        mock_level_infra = MagicMock()
        mock_level_infra.get.side_effect = lambda key: mock_calendar if key == "trade_calendar" else MagicMock()

        executor = XtQMTLiveExecutor.__new__(XtQMTLiveExecutor)
        executor._live_exchange = mock_exchange
        executor.level_infra = mock_level_infra
        executor.trade_account = MagicMock()
        executor.dealt_order_amount = defaultdict(float)
        executor.deal_day = None
        executor.verbose = False
        executor.trade_type = "serial"

        return executor, mock_exchange, mock_calendar

    def test_collect_data_empty_orders(self):
        executor, mock_exchange, _ = self._make_executor()
        mock_decision = MagicMock()

        from qlib.backtest.executor import _retrieve_orders_from_decision
        from unittest.mock import patch
        with patch("qlib.contrib.broker.miniQMT.executor.live_executor._retrieve_orders_from_decision", return_value=[]):
            result, info = executor._collect_data(mock_decision)

        assert result == []
        assert info == {"trade_info": []}

    def test_collect_data_multiple_orders(self):
        from qlib.backtest.decision import Order, OrderDir
        from unittest.mock import patch

        executor, mock_exchange, _ = self._make_executor()
        mock_exchange.deal_order.return_value = (1000.0, 5.0, 10.0)

        orders = [
            Order(stock_id="SH600000", amount=100, direction=OrderDir.BUY, start_time=pd.Timestamp.now(), end_time=pd.Timestamp.now()),
            Order(stock_id="SZ000001", amount=200, direction=OrderDir.SELL, start_time=pd.Timestamp.now(), end_time=pd.Timestamp.now()),
            Order(stock_id="SH600036", amount=300, direction=OrderDir.BUY, start_time=pd.Timestamp.now(), end_time=pd.Timestamp.now()),
        ]
        for o in orders:
            o.deal_amount = 0

        mock_decision = MagicMock()
        with patch("qlib.contrib.broker.miniQMT.executor.live_executor._retrieve_orders_from_decision", return_value=orders):
            result, info = executor._collect_data(mock_decision)

        assert len(result) == 3
        assert mock_exchange.deal_order.call_count == 3

    def test_collect_data_tracks_dealt_amount(self):
        from qlib.backtest.decision import Order, OrderDir
        from unittest.mock import patch

        executor, mock_exchange, _ = self._make_executor()
        mock_exchange.deal_order.return_value = (1000.0, 5.0, 10.0)

        orders = [
            Order(stock_id="SH600000", amount=100, direction=OrderDir.BUY, start_time=pd.Timestamp.now(), end_time=pd.Timestamp.now()),
            Order(stock_id="SH600000", amount=200, direction=OrderDir.BUY, start_time=pd.Timestamp.now(), end_time=pd.Timestamp.now()),
        ]
        orders[0].deal_amount = 100
        orders[1].deal_amount = 200

        mock_decision = MagicMock()
        with patch("qlib.contrib.broker.miniQMT.executor.live_executor._retrieve_orders_from_decision", return_value=orders):
            executor._collect_data(mock_decision)

        assert executor.dealt_order_amount["SH600000"] == 300

    def test_trade_exchange_property(self):
        executor, mock_exchange, _ = self._make_executor()
        assert executor.trade_exchange is mock_exchange

    def test_collect_data_resets_on_new_day(self):
        from qlib.backtest.decision import Order, OrderDir
        from unittest.mock import patch
        from collections import defaultdict

        executor, mock_exchange, mock_calendar = self._make_executor()
        mock_exchange.deal_order.return_value = (1000.0, 5.0, 10.0)

        # Set deal_day to yesterday
        executor.deal_day = pd.Timestamp("2026-04-04")
        executor.dealt_order_amount = defaultdict(float, {"SH600000": 500})

        orders = [
            Order(stock_id="SH600000", amount=100, direction=OrderDir.BUY, start_time=pd.Timestamp.now(), end_time=pd.Timestamp.now()),
        ]
        orders[0].deal_amount = 100

        mock_decision = MagicMock()
        with patch("qlib.contrib.broker.miniQMT.executor.live_executor._retrieve_orders_from_decision", return_value=orders):
            executor._collect_data(mock_decision)

        # deal_day updated to new day, dealt_order_amount was reset then accumulated
        assert executor.deal_day == pd.Timestamp("2026-04-05")
        assert executor.dealt_order_amount["SH600000"] == 100


# ============================================================
# TradingLoop extended tests
# ============================================================

class TestTradingLoopExtended:
    def _make_loop(self):
        from qlib.contrib.broker.miniQMT.trader.trading_loop import TradingLoop

        loop = TradingLoop(mini_qmt_path="D:/test", account_id="123")
        loop._connected = True

        loop.live_exchange = MagicMock()
        loop.live_exchange.deal_order.return_value = (1000.0, 5.0, 10.0)
        loop.position_sync = MagicMock()
        loop.position_sync.reconcile.return_value = []
        loop.account_adapter = MagicMock()
        loop.account_adapter.get_total_asset.return_value = 1000000.0
        loop.xt_trader = MagicMock()
        loop._get_latest_price = lambda s: 10.0

        return loop

    def test_run_once_sell_before_buy(self):
        from qlib.backtest.decision import OrderDir
        from qlib.backtest.position import Position

        loop = self._make_loop()
        pos = Position(cash=500000.0, position_dict={"SH600000": {"amount": 1000, "price": 10.0}})

        call_order = []
        original_deal = loop.live_exchange.deal_order

        def track_deal(order, **kwargs):
            call_order.append(order.direction)
            return (1000.0, 5.0, 10.0)

        loop.live_exchange.deal_order = track_deal

        # Target: sell SH600000, buy SZ000001
        loop.run_once(target_positions={"SZ000001": 0.1}, current_position=pos, total_value=1000000.0)

        assert len(call_order) >= 2
        sell_indices = [i for i, d in enumerate(call_order) if d == OrderDir.SELL]
        buy_indices = [i for i, d in enumerate(call_order) if d == OrderDir.BUY]
        assert sell_indices and buy_indices
        assert max(sell_indices) < min(buy_indices)

    def test_run_once_fetches_position_when_none(self):
        from qlib.backtest.position import Position

        loop = self._make_loop()
        mock_pos = Position(cash=1000000.0, position_dict={})
        loop.position_sync.build_qlib_position.return_value = mock_pos

        loop.run_once(target_positions={}, current_position=None, total_value=1000000.0)
        loop.position_sync.build_qlib_position.assert_called_once()

    def test_run_once_fetches_total_value_when_none(self):
        from qlib.backtest.position import Position

        loop = self._make_loop()
        pos = Position(cash=1000000.0, position_dict={})

        loop.run_once(target_positions={}, current_position=pos, total_value=None)
        loop.account_adapter.get_total_asset.assert_called_once()

    def test_run_once_empty_returns_empty(self):
        from qlib.backtest.position import Position

        loop = self._make_loop()
        pos = Position(cash=1000000.0, position_dict={})

        results = loop.run_once(target_positions={}, current_position=pos, total_value=1000000.0)
        assert results == []

    def test_run_once_reconciliation_after_trades(self):
        from qlib.backtest.position import Position

        loop = self._make_loop()
        pos = Position(cash=900000.0, position_dict={"SH600000": {"amount": 1000, "price": 10.0}})

        loop.run_once(target_positions={}, current_position=pos, total_value=1000000.0)
        loop.position_sync.reconcile.assert_called_once()

    def test_generate_orders_mixed(self):
        from qlib.backtest.decision import OrderDir
        from qlib.backtest.position import Position

        loop = self._make_loop()
        pos = Position(cash=800000.0, position_dict={"SH600000": {"amount": 1000, "price": 10.0}})

        orders = loop._generate_orders({"SZ000001": 0.1}, pos, 1000000.0)

        sell_orders = [o for o in orders if o.direction == OrderDir.SELL]
        buy_orders = [o for o in orders if o.direction == OrderDir.BUY]
        assert len(sell_orders) == 1
        assert sell_orders[0].stock_id == "SH600000"
        assert len(buy_orders) == 1
        assert buy_orders[0].stock_id == "SZ000001"

    def test_generate_orders_price_zero_skips_buy(self):
        from qlib.backtest.decision import OrderDir
        from qlib.backtest.position import Position

        loop = self._make_loop()
        loop._get_latest_price = lambda s: 0
        pos = Position(cash=1000000.0, position_dict={})

        orders = loop._generate_orders({"SZ000001": 0.1}, pos, 1000000.0)
        buy_orders = [o for o in orders if o.direction == OrderDir.BUY]
        assert len(buy_orders) == 0

    def test_generate_orders_price_none_skips_buy(self):
        from qlib.backtest.decision import OrderDir
        from qlib.backtest.position import Position

        loop = self._make_loop()
        loop._get_latest_price = lambda s: None
        pos = Position(cash=1000000.0, position_dict={})

        orders = loop._generate_orders({"SZ000001": 0.1}, pos, 1000000.0)
        buy_orders = [o for o in orders if o.direction == OrderDir.BUY]
        assert len(buy_orders) == 0

    def test_generate_orders_increase_position(self):
        from qlib.backtest.decision import OrderDir
        from qlib.backtest.position import Position

        loop = self._make_loop()
        # Current: 1000 shares at 10.0; target weight 0.5 -> target_amount = round_to_lot(1000000*0.5/10) = 50000
        pos = Position(cash=900000.0, position_dict={"SH600000": {"amount": 1000, "price": 10.0}})
        orders = loop._generate_orders({"SH600000": 0.5}, pos, 1000000.0)

        buy_orders = [o for o in orders if o.direction == OrderDir.BUY]
        assert len(buy_orders) == 1
        assert buy_orders[0].stock_id == "SH600000"
        # buy_amount = round_to_lot(50000 - 1000) = 49000
        assert buy_orders[0].amount == 49000

    def test_generate_orders_decrease_position(self):
        from qlib.backtest.decision import OrderDir
        from qlib.backtest.position import Position

        loop = self._make_loop()
        # Current: 5000 shares at 10.0; target weight 0.01 -> target_amount = round_to_lot(1000000*0.01/10) = 1000
        pos = Position(cash=950000.0, position_dict={"SH600000": {"amount": 5000, "price": 10.0}})
        orders = loop._generate_orders({"SH600000": 0.01}, pos, 1000000.0)

        sell_orders = [o for o in orders if o.direction == OrderDir.SELL]
        assert len(sell_orders) == 1
        assert sell_orders[0].stock_id == "SH600000"
        # sell_amount = round_to_lot(5000 - 1000) = 4000
        assert sell_orders[0].amount == 4000

    def test_disconnect(self):
        loop = self._make_loop()
        loop.disconnect()

        loop.xt_trader.stop.assert_called_once()
        assert loop._connected is False

    def test_connect_failure_raises(self):
        from qlib.contrib.broker.miniQMT.trader.trading_loop import TradingLoop
        from unittest.mock import patch

        loop = TradingLoop(mini_qmt_path="D:/test", account_id="123")

        mock_xttrader_cls = MagicMock()
        mock_trader_instance = MagicMock()
        mock_xttrader_cls.return_value = mock_trader_instance
        mock_trader_instance.connect.return_value = -1

        mock_stock_account_cls = MagicMock()

        with patch.dict(sys.modules, {
            "xtquant": types.ModuleType("xtquant"),
            "xtquant.xttrader": MagicMock(XtQuantTrader=mock_xttrader_cls),
            "xtquant.xttype": MagicMock(StockAccount=mock_stock_account_cls),
        }):
            with pytest.raises(ConnectionError):
                loop.connect()


# ============================================================
# PositionSynchronizer extended tests
# ============================================================

class TestPositionSynchronizerExtended:
    def _make_mock_trader(self, positions=None, cash=500000.0):
        mock_trader = MagicMock()
        mock_asset = MagicMock()
        mock_asset.cash = cash
        mock_trader.query_stock_asset.return_value = mock_asset

        if positions is None:
            mock_pos = MagicMock()
            mock_pos.stock_code = "600000.SH"
            mock_pos.volume = 1000
            mock_pos.market_value = 10000.0
            positions = [mock_pos]
        mock_trader.query_stock_positions.return_value = positions
        return mock_trader

    def test_build_qlib_position(self):
        from qlib.contrib.broker.miniQMT.account.position_sync import PositionSynchronizer

        sync = PositionSynchronizer(self._make_mock_trader(), "123")
        pos = sync.build_qlib_position()

        assert pos.get_cash() == 500000.0
        assert "SH600000" in pos.get_stock_list()
        assert pos.get_stock_amount("SH600000") == 1000

    def test_force_sync_overwrites(self):
        from qlib.contrib.broker.miniQMT.account.position_sync import PositionSynchronizer
        from qlib.backtest.position import Position

        mock_pos_new = MagicMock()
        mock_pos_new.stock_code = "000001.SZ"
        mock_pos_new.volume = 2000
        mock_pos_new.market_value = 30000.0

        sync = PositionSynchronizer(self._make_mock_trader(positions=[mock_pos_new], cash=600000.0), "123")

        # Old position has SH600000
        qlib_pos = Position(cash=500000.0, position_dict={"SH600000": {"amount": 1000, "price": 10.0}})
        sync.force_sync(qlib_pos)

        # After force_sync: old stock removed, new stock added
        stock_list = qlib_pos.get_stock_list()
        assert "SH600000" not in stock_list
        assert "SZ000001" in stock_list
        assert qlib_pos.get_cash() == 600000.0

    def test_reconcile_extra_in_broker(self):
        from qlib.contrib.broker.miniQMT.account.position_sync import PositionSynchronizer
        from qlib.backtest.position import Position

        sync = PositionSynchronizer(self._make_mock_trader(), "123")
        # Qlib has no positions but broker has SH600000
        qlib_pos = Position(cash=500000.0, position_dict={})
        discrepancies = sync.reconcile(qlib_pos)

        assert len(discrepancies) > 0
        assert any("SH600000" in d and "broker" in d for d in discrepancies)

    def test_fetch_skips_unrecognized_codes(self):
        from qlib.contrib.broker.miniQMT.account.position_sync import PositionSynchronizer

        mock_pos_bad = MagicMock()
        mock_pos_bad.stock_code = "INVALID_CODE"
        mock_pos_bad.volume = 100
        mock_pos_bad.market_value = 1000.0

        mock_pos_good = MagicMock()
        mock_pos_good.stock_code = "600000.SH"
        mock_pos_good.volume = 500
        mock_pos_good.market_value = 5000.0

        sync = PositionSynchronizer(self._make_mock_trader(positions=[mock_pos_bad, mock_pos_good]), "123")
        cash, positions = sync.fetch_broker_positions()

        # Bad code skipped, good code present
        assert "SH600000" in positions
        assert len(positions) == 1


# ============================================================
# End-to-end integration tests
# ============================================================

class TestIntegrationE2E:
    def _make_full_loop(self):
        """Create a fully mocked TradingLoop with all components wired up."""
        from qlib.contrib.broker.miniQMT.trader.trading_loop import TradingLoop

        loop = TradingLoop(mini_qmt_path="D:/test", account_id="123")
        loop._connected = True

        # Mock all components
        mock_mgr = MagicMock()
        mock_mgr.query_sellable_amount.return_value = 10000

        from qlib.contrib.broker.miniQMT.executor.exchange_live import XtQMTLiveExchange
        from qlib.contrib.broker.miniQMT.trader.callback_handler import DealResult

        exchange = XtQMTLiveExchange(order_manager=mock_mgr)
        mock_mgr.submit_order.return_value = DealResult(success=True, deal_price=10.0, deal_amount=100, deal_value=1000.0)

        loop.live_exchange = exchange
        loop.position_sync = MagicMock()
        loop.position_sync.reconcile.return_value = []
        loop.account_adapter = MagicMock()
        loop.account_adapter.get_total_asset.return_value = 1000000.0
        loop.xt_trader = MagicMock()
        loop._get_latest_price = lambda s: 10.0

        return loop, mock_mgr

    def test_full_pipeline(self):
        from qlib.backtest.position import Position

        loop, mock_mgr = self._make_full_loop()

        pos = Position(cash=900000.0, position_dict={"SH600000": {"amount": 1000, "price": 10.0}})

        # connect is already done via _connected=True
        results = loop.run_once(
            target_positions={"SH600000": 0.05, "SZ000001": 0.05},
            current_position=pos,
            total_value=1000000.0,
        )

        assert isinstance(results, list)
        # disconnect
        loop.disconnect()
        assert loop._connected is False

    def test_sell_before_buy_integration(self):
        from qlib.backtest.decision import OrderDir
        from qlib.backtest.position import Position

        loop, mock_mgr = self._make_full_loop()

        pos = Position(cash=500000.0, position_dict={"SH600000": {"amount": 5000, "price": 10.0}})

        deal_directions = []
        original_submit = mock_mgr.submit_order

        def tracking_submit(order):
            deal_directions.append(order.direction)
            from qlib.contrib.broker.miniQMT.trader.callback_handler import DealResult
            return DealResult(success=True, deal_price=10.0, deal_amount=order.amount, deal_value=order.amount * 10.0)

        mock_mgr.submit_order = tracking_submit

        results = loop.run_once(
            target_positions={"SZ000001": 0.1},
            current_position=pos,
            total_value=1000000.0,
        )

        # Sell SH600000 should come before buy SZ000001
        sell_idx = [i for i, d in enumerate(deal_directions) if d == OrderDir.SELL]
        buy_idx = [i for i, d in enumerate(deal_directions) if d == OrderDir.BUY]
        if sell_idx and buy_idx:
            assert max(sell_idx) < min(buy_idx)

    def test_reconciliation_after_trades(self):
        from qlib.backtest.position import Position

        loop, mock_mgr = self._make_full_loop()

        pos = Position(cash=900000.0, position_dict={"SH600000": {"amount": 1000, "price": 10.0}})

        loop.run_once(
            target_positions={"SH600000": 0.05},
            current_position=pos,
            total_value=1000000.0,
        )

        loop.position_sync.reconcile.assert_called_once()


# ============================================================
# Commission default alignment
# ============================================================

class TestCommissionAlignment:
    def test_default_commission_matches_config(self):
        import inspect
        from qlib.contrib.broker.miniQMT.executor.exchange_live import XtQMTLiveExchange

        sig = inspect.signature(XtQMTLiveExchange.__init__)
        assert sig.parameters["open_cost"].default == 0.0005
        assert sig.parameters["close_cost"].default == 0.0015

    def test_trading_loop_commission_pass_through(self):
        from qlib.contrib.broker.miniQMT.trader.trading_loop import TradingLoop

        loop = TradingLoop(
            mini_qmt_path="D:/test",
            account_id="123",
            open_cost=0.001,
            close_cost=0.002,
            min_cost=10.0,
        )
        assert loop._open_cost == 0.001
        assert loop._close_cost == 0.002
        assert loop._min_cost == 10.0

    def test_trading_loop_default_commission(self):
        import inspect
        from qlib.contrib.broker.miniQMT.trader.trading_loop import TradingLoop

        sig = inspect.signature(TradingLoop.__init__)
        assert sig.parameters["open_cost"].default == 0.0005
        assert sig.parameters["close_cost"].default == 0.0015
        assert sig.parameters["min_cost"].default == 5.0


# ============================================================
# Report generator
# ============================================================

class TestReportGenerator:
    def _make_synthetic_data(self):
        dates = pd.date_range("2023-01-01", periods=60, freq="B")
        np.random.seed(42)
        report = pd.DataFrame({
            "return": np.random.randn(60) * 0.01,
            "bench": np.random.randn(60) * 0.008,
            "cost": np.abs(np.random.randn(60)) * 0.0001,
        }, index=dates)
        analysis = {
            "risk": {
                "mean": 0.001,
                "std": 0.02,
                "annualized_return": 0.25,
                "information_ratio": 1.2,
                "max_drawdown": -0.15,
            }
        }
        return report, analysis

    def test_generate_html_report(self):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[0] / ".."))
        from scripts.miniQMT.report_generator import generate_html_report

        report, analysis = self._make_synthetic_data()
        with tempfile.TemporaryDirectory() as tmpdir:
            html_path = generate_html_report(report, analysis, output_dir=tmpdir)
            assert Path(html_path).exists()
            content = Path(html_path).read_text(encoding="utf-8")
            assert "data:image/png;base64," in content
            assert "0.001" in content
            assert "annualized_return" in content

    def test_csv_output(self):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[0] / ".."))
        from scripts.miniQMT.report_generator import generate_html_report

        report, analysis = self._make_synthetic_data()
        with tempfile.TemporaryDirectory() as tmpdir:
            generate_html_report(report, analysis, output_dir=tmpdir)
            csv_files = list(Path(tmpdir).glob("*.csv"))
            assert len(csv_files) == 2
            names = {f.name for f in csv_files}
            assert any("report_" in n for n in names)
            assert any("analysis_" in n for n in names)


# ============================================================
# TopkDropout live strategy
# ============================================================

class TestTopkDropoutPositions:
    def _import_func(self):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[0] / ".."))
        from scripts.miniQMT.daily_trading import build_topk_dropout_positions
        return build_topk_dropout_positions

    def test_rotation_when_new_beats_old(self):
        build = self._import_func()
        pred = pd.Series(
            [10, 9, 8, 7, 6, 5, 4, 3],
            index=["A", "B", "C", "F", "G", "D", "E", "H"],
        )
        # Currently hold: A B C D E (D scores 5, E scores 4)
        current = ["A", "B", "C", "D", "E"]
        result = build(pred, current, topk=5, n_drop=2)

        # Combined: A(10) B(9) C(8) F(7) G(6) D(5) E(4)
        # bottom 2: D, E -> both in current -> sell D, E
        # buy: F, G
        assert "D" not in result
        assert "E" not in result
        assert "F" in result
        assert "G" in result
        assert len(result) == 5

    def test_empty_holdings(self):
        build = self._import_func()
        pred = pd.Series(
            [10, 9, 8, 7, 6],
            index=["A", "B", "C", "D", "E"],
        )
        result = build(pred, [], topk=3, n_drop=2)

        assert len(result) == 3
        assert "A" in result
        assert "B" in result
        assert "C" in result

    def test_fewer_candidates(self):
        build = self._import_func()
        pred = pd.Series(
            [10, 9, 8, 7],
            index=["A", "B", "C", "D"],
        )
        current = ["A", "B", "C"]
        result = build(pred, current, topk=3, n_drop=3)

        # Only 1 new candidate (D), combined bottom 3: B C D
        # sell from current: B, C; buy: D (only 1 available)
        # final: A + D = 2 stocks (fewer than topk)
        assert len(result) <= 3
        assert "A" in result

    def test_all_retained(self):
        build = self._import_func()
        pred = pd.Series(
            [10, 9, 8, 1, 0.5],
            index=["A", "B", "C", "D", "E"],
        )
        current = ["A", "B", "C"]
        result = build(pred, current, topk=3, n_drop=2)

        # New candidates D, E rank low. Combined bottom 2: D, E (not in current)
        # No sells, no buys -> portfolio unchanged
        assert "A" in result
        assert "B" in result
        assert "C" in result
        assert len(result) == 3

    def test_equal_weights(self):
        build = self._import_func()
        pred = pd.Series([10, 9, 8], index=["A", "B", "C"])
        result = build(pred, [], topk=3, n_drop=1)

        for w in result.values():
            assert abs(w - 1.0 / 3) < 1e-10
