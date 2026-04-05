# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Convert xtdata historical data to Qlib .bin format for backtesting.

Usage:
    from qlib.contrib.broker.miniQMT.data.data_collector import XtDataCollector

    collector = XtDataCollector(target_dir="~/.qlib/qlib_data/xt_cn_data")
    collector.collect(stock_list=["600000.SH", "000001.SZ"], start_date="20200101", end_date="20231231")

After collection, use standard Qlib init:
    qlib.init(provider_uri="~/.qlib/qlib_data/xt_cn_data")
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from qlib.log import get_module_logger
from ..utils import StockCodeConverter

logger = get_module_logger("XtDataCollector")

# Qlib .bin file uses C float (4 bytes) with calendar index
BIN_DTYPE = "<f"  # little-endian float32

# Standard Qlib fields mapping from xtdata columns
XTDATA_TO_QLIB_FIELDS = {
    "open": "$open",
    "high": "$high",
    "low": "$low",
    "close": "$close",
    "volume": "$volume",
    "amount": "$amount",
}

# xtdata additional fields
XTDATA_FACTOR_FIELD = "preClose"  # used to derive adjust factor


class XtDataCollector:
    """Download data from xtdata and convert to Qlib .bin format.

    This is the recommended approach for backtesting — it produces standard .bin files
    that work with Qlib's LocalProvider at full speed.
    """

    def __init__(
        self,
        target_dir: str = "~/.qlib/qlib_data/xt_cn_data",
        freq: str = "1d",
        field_map: Optional[Dict[str, str]] = None,
    ):
        """
        Parameters
        ----------
        target_dir : str
            Directory to store converted .bin files. Structure will be:
            target_dir/
                calendars/day.txt
                instruments/all.txt
                features/<stock_id>/
                    $open.bin, $close.bin, ...
        freq : str
            xtdata frequency. "1d" for daily, "1m" for 1-minute, etc.
        field_map : dict, optional
            Custom mapping from xtdata column names to Qlib field names.
        """
        self.target_dir = Path(os.path.expanduser(target_dir))
        self.freq = freq
        self.field_map = field_map or XTDATA_TO_QLIB_FIELDS

    def collect(
        self,
        stock_list: Optional[List[str]] = None,
        market: str = "SH",
        start_date: str = "20100101",
        end_date: Optional[str] = None,
        include_factor: bool = True,
    ) -> None:
        """Download and convert data.

        Parameters
        ----------
        stock_list : list of str, optional
            List of xtquant stock codes (e.g. ["600000.SH"]). If None, fetches all stocks.
        market : str
            Market identifier for xtdata, used if stock_list is None.
        start_date : str
            Start date in "YYYYMMDD" format.
        end_date : str, optional
            End date in "YYYYMMDD" format. Defaults to today.
        include_factor : bool
            Whether to compute and save $factor field for adjust.
        """
        try:
            from xtquant import xtdata
        except ImportError:
            raise ImportError("xtquant is required. Please install it from your miniQMT distribution.")

        if end_date is None:
            end_date = pd.Timestamp.now().strftime("%Y%m%d")

        if stock_list is None:
            stock_list = xtdata.get_stock_list_in_sector(f"沪深A股")
            logger.info(f"Fetched {len(stock_list)} stocks from sector")

        logger.info(f"Downloading data for {len(stock_list)} stocks, period {start_date}-{end_date}")
        xtdata.download_history_data2(stock_list=stock_list, period=self.freq, start_time=start_date, end_time=end_date)

        all_calendars = set()
        instruments_info = {}
        stock_data = {}

        # Pass 1: Fetch all stock data into memory, build unified calendar
        for xt_code in stock_list:
            qlib_code = StockCodeConverter.xt_to_qlib(xt_code)
            try:
                df = self._fetch_single_stock(xtdata, xt_code, start_date, end_date)
            except Exception as e:
                logger.warning(f"Failed to fetch {xt_code}: {e}")
                continue

            if df is None or df.empty:
                logger.debug(f"No data for {xt_code}, skipping")
                continue

            all_calendars.update(df.index.tolist())
            instruments_info[qlib_code] = (df.index[0], df.index[-1])
            stock_data[qlib_code] = df

        if not all_calendars:
            logger.warning("No data collected. Check xtdata connection and stock list.")
            return

        # Write calendar first so .bin files can use it
        sorted_calendar = sorted(all_calendars)
        self._save_calendar(sorted_calendar)
        cal_index = {t: i for i, t in enumerate(sorted_calendar)}
        cal_len = len(sorted_calendar)

        # Pass 2: Write .bin files using the unified calendar index
        for qlib_code, df in stock_data.items():
            self._save_stock_features_with_calendar(qlib_code, df, include_factor, cal_index, cal_len)

        self._save_instruments(instruments_info)

        logger.info(
            f"Collection complete. {len(instruments_info)} stocks, "
            f"{len(all_calendars)} trading days saved to {self.target_dir}"
        )

    def collect_incremental(
        self,
        stock_list: Optional[List[str]] = None,
        days_back: int = 5,
    ) -> None:
        """Incremental update: only download recent data and append to existing .bin files.

        Parameters
        ----------
        stock_list : list of str, optional
            Stock codes. If None, reads from existing instruments/all.txt.
        days_back : int
            Number of calendar days to re-download for overlap safety.
        """
        try:
            from xtquant import xtdata
        except ImportError:
            raise ImportError("xtquant is required.")

        if stock_list is None:
            stock_list = self._read_existing_stock_list()

        calendar = self._read_existing_calendar()
        if calendar:
            start_date = (calendar[-1] - pd.Timedelta(days=days_back)).strftime("%Y%m%d")
        else:
            start_date = "20100101"

        end_date = pd.Timestamp.now().strftime("%Y%m%d")
        self.collect(stock_list=stock_list, start_date=start_date, end_date=end_date)

    def _fetch_single_stock(self, xtdata, xt_code: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch data for a single stock from xtdata."""
        data = xtdata.get_market_data_ex(
            stock_list=[xt_code],
            period=self.freq,
            start_time=start_date,
            end_time=end_date,
            fields=list(self.field_map.keys()) + [XTDATA_FACTOR_FIELD],
        )
        if xt_code not in data or data[xt_code].empty:
            return None

        df = data[xt_code].copy()
        df.index = pd.to_datetime(df.index, format="%Y%m%d" if self.freq == "1d" else "%Y%m%d%H%M%S")

        # Filter out rows where close is 0 or NaN (suspended days)
        if "close" in df.columns:
            df = df[df["close"] > 0]

        return df

    def _save_stock_features(self, qlib_code: str, df: pd.DataFrame, include_factor: bool) -> None:
        """Save one stock's features to .bin files.

        Note: For full collect(), use _save_stock_features_with_calendar() instead
        to ensure correct calendar alignment across all stocks.
        """
        calendar = self._read_existing_calendar()
        if not calendar:
            calendar = sorted(df.index.tolist())
        cal_index = {t: i for i, t in enumerate(calendar)}
        self._save_stock_features_with_calendar(qlib_code, df, include_factor, cal_index, len(calendar))

    def _save_stock_features_with_calendar(
        self, qlib_code: str, df: pd.DataFrame, include_factor: bool, cal_index: dict, cal_len: int
    ) -> None:
        """Save one stock's features to .bin files using a pre-computed calendar index."""
        feature_dir = self.target_dir / "features" / qlib_code
        feature_dir.mkdir(parents=True, exist_ok=True)

        for xt_field, qlib_field in self.field_map.items():
            if xt_field not in df.columns:
                continue
            self._write_bin_file(
                feature_dir / f"{qlib_field}.bin",
                df.index,
                df[xt_field].values,
                cal_index,
                cal_len,
            )

        if include_factor and XTDATA_FACTOR_FIELD in df.columns and "close" in df.columns:
            # Compute cumulative adjust factor from preClose/close ratio
            factor = self._compute_adjust_factor(df)
            if factor is not None:
                self._write_bin_file(
                    feature_dir / "$factor.bin",
                    df.index,
                    factor,
                    cal_index,
                    cal_len,
                )

    def _compute_adjust_factor(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """Compute forward adjust factor from preClose data.

        The factor represents: adjusted_price = raw_price * factor
        """
        close = df["close"].values
        pre_close = df[XTDATA_FACTOR_FIELD].values

        if len(close) < 2:
            return np.ones(len(close), dtype=np.float32)

        # Ratio of preClose to previous close indicates adjust events
        factor = np.ones(len(close), dtype=np.float64)
        for i in range(1, len(close)):
            if close[i - 1] > 0 and pre_close[i] > 0:
                ratio = pre_close[i] / close[i - 1]
                if abs(ratio - 1.0) > 1e-6:
                    factor[i] = ratio
                else:
                    factor[i] = 1.0
            else:
                factor[i] = 1.0

        # Cumulative product gives the adjust factor
        cum_factor = np.cumprod(factor).astype(np.float32)
        return cum_factor

    @staticmethod
    def _write_bin_file(
        filepath: Path,
        timestamps: pd.DatetimeIndex,
        values: np.ndarray,
        cal_index: dict,
        cal_len: int,
    ) -> None:
        """Write a Qlib .bin feature file.

        Qlib .bin format (matches qlib/data/storage/file_storage.py):
        - First 4 bytes: start_index (float32) — offset into calendar
        - Then N float32 values, one per calendar day from start_index
        - Missing days filled with NaN
        """
        indices = []
        for t in timestamps:
            if t in cal_index:
                indices.append(cal_index[t])

        if not indices:
            return

        start_idx = min(indices)
        end_idx = max(indices)
        length = end_idx - start_idx + 1

        data = np.full(length, np.nan, dtype=np.float32)
        for t, v in zip(timestamps, values):
            if t in cal_index:
                pos = cal_index[t] - start_idx
                data[pos] = v

        with open(filepath, "wb") as f:
            # Qlib .bin: 1 float32 start_index, then float32 data
            # Must match qlib/data/storage/file_storage.py format
            np.hstack([np.float32(start_idx), data]).astype("<f").tofile(f)

    def _save_calendar(self, calendar: list) -> None:
        """Save calendar to day.txt."""
        cal_dir = self.target_dir / "calendars"
        cal_dir.mkdir(parents=True, exist_ok=True)

        freq_name = "day" if self.freq == "1d" else self.freq.replace("m", "min")
        with open(cal_dir / f"{freq_name}.txt", "w") as f:
            for dt in calendar:
                f.write(pd.Timestamp(dt).strftime("%Y-%m-%d") + "\n")

    def _save_instruments(self, instruments_info: Dict[str, tuple]) -> None:
        """Save instruments list to all.txt."""
        inst_dir = self.target_dir / "instruments"
        inst_dir.mkdir(parents=True, exist_ok=True)

        with open(inst_dir / "all.txt", "w") as f:
            for code, (start, end) in sorted(instruments_info.items()):
                start_str = pd.Timestamp(start).strftime("%Y-%m-%d")
                end_str = pd.Timestamp(end).strftime("%Y-%m-%d")
                f.write(f"{code}\t{start_str}\t{end_str}\n")

    def _read_existing_calendar(self) -> list:
        """Read existing calendar file if present."""
        freq_name = "day" if self.freq == "1d" else self.freq.replace("m", "min")
        cal_file = self.target_dir / "calendars" / f"{freq_name}.txt"
        if not cal_file.exists():
            return []
        dates = []
        with open(cal_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    dates.append(pd.Timestamp(line))
        return sorted(dates)

    def _read_existing_stock_list(self) -> List[str]:
        """Read existing instrument list and convert back to xt format."""
        inst_file = self.target_dir / "instruments" / "all.txt"
        if not inst_file.exists():
            raise FileNotFoundError(f"No existing instruments found at {inst_file}")
        codes = []
        with open(inst_file, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if parts:
                    codes.append(StockCodeConverter.qlib_to_xt(parts[0]))
        return codes
