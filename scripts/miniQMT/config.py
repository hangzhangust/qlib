"""Shared configuration for miniQMT scripts."""

import logging
import sys
from pathlib import Path

# --- Data & Broker Paths ---
DEFAULT_PROVIDER_URI = "~/.qlib/qlib_data/xt_cn_data"
DEFAULT_MINI_QMT_PATH = "D:/国金证券QMT交易端/userdata_mini"
DEFAULT_ACCOUNT_ID = "8885183811"

# --- Trading Parameters ---
DEFAULT_FREQ = "1d"
DEFAULT_MARKET = "csi300"
DEFAULT_BENCHMARK = "SH000300"
DEFAULT_TOPK = 30
DEFAULT_N_DROP = 5
DEFAULT_MAX_WEIGHT = 0.10
DEFAULT_CASH_RESERVE = 0.05
DEFAULT_ORDER_TIMEOUT = 60.0
DEFAULT_ACCOUNT_AMOUNT = 1_0000_0000  # 1e8

# --- LightGBM Hyperparameters ---
DEFAULT_LGB_KWARGS = {
    "loss": "mse",
    "colsample_bytree": 0.8879,
    "learning_rate": 0.0421,
    "subsample": 0.8789,
    "lambda_l1": 205.6999,
    "lambda_l2": 580.9768,
    "max_depth": 8,
    "num_leaves": 210,
    "num_threads": 20,
}

# --- Exchange / Commission ---
DEFAULT_EXCHANGE_KWARGS = {
    "freq": DEFAULT_FREQ,
    "limit_threshold": 0.095,
    "deal_price": "close",
    "open_cost": 0.0005,
    "close_cost": 0.0015,
    "min_cost": 5,
}


def setup_logging(name: str, log_file: str = None) -> logging.Logger:
    """Configure logger with console + optional file output."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    # Suppress noisy xtquant logs
    logging.getLogger("xtquant").setLevel(logging.WARNING)
    return logger
