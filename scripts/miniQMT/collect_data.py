"""Download market data from xtdata and convert to Qlib .bin format.

Usage:
    python scripts/miniQMT/collect_data.py --start-date 20100101
    python scripts/miniQMT/collect_data.py --incremental --days-back 5
    python scripts/miniQMT/collect_data.py --stock-list 600519.SH,000858.SZ
"""

import argparse
import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.miniQMT.config import DEFAULT_FREQ, DEFAULT_PROVIDER_URI, setup_logging

logger = setup_logging("collect_data")


def parse_args():
    p = argparse.ArgumentParser(description="Collect xtdata market data → Qlib .bin")
    p.add_argument("--stock-list", type=str, default=None,
                    help="Comma-separated xt-format codes (e.g. 600519.SH,000858.SZ). Default: all A-shares")
    p.add_argument("--start-date", type=str, default="20100101", help="Start date YYYYMMDD (default: 20100101)")
    p.add_argument("--end-date", type=str, default=None, help="End date YYYYMMDD (default: today)")
    p.add_argument("--target-dir", type=str, default=DEFAULT_PROVIDER_URI, help="Output directory for .bin files")
    p.add_argument("--freq", type=str, default=DEFAULT_FREQ, help="Data frequency (default: 1d)")
    p.add_argument("--incremental", action="store_true", help="Incremental update mode")
    p.add_argument("--days-back", type=int, default=5, help="Overlap days for incremental mode (default: 5)")
    return p.parse_args()


def main():
    args = parse_args()
    from qlib.contrib.broker.miniQMT.data.data_collector import XtDataCollector

    stock_list = args.stock_list.split(",") if args.stock_list else None
    collector = XtDataCollector(target_dir=args.target_dir, freq=args.freq)

    if args.incremental:
        logger.info("Incremental update (days_back=%d) → %s", args.days_back, args.target_dir)
        collector.collect_incremental(stock_list=stock_list, days_back=args.days_back)
    else:
        logger.info("Full collection [%s, %s] → %s", args.start_date, args.end_date or "today", args.target_dir)
        collector.collect(stock_list=stock_list, start_date=args.start_date, end_date=args.end_date)

    logger.info("Data collection finished.")


if __name__ == "__main__":
    main()
