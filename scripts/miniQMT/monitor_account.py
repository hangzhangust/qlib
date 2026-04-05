"""Connect to miniQMT and monitor account status in real-time.

Usage:
    python scripts/miniQMT/monitor_account.py --once
    python scripts/miniQMT/monitor_account.py --interval 30
    python scripts/miniQMT/monitor_account.py --reconcile
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.miniQMT.config import DEFAULT_ACCOUNT_ID, DEFAULT_MINI_QMT_PATH, setup_logging

logger = setup_logging("monitor_account")


def parse_args():
    p = argparse.ArgumentParser(description="miniQMT account monitor")
    p.add_argument("--mini-qmt-path", default=DEFAULT_MINI_QMT_PATH, help="miniQMT userdata path")
    p.add_argument("--account-id", default=DEFAULT_ACCOUNT_ID, help="Trading account ID")
    p.add_argument("--interval", type=int, default=30, help="Refresh interval in seconds (default: 30)")
    p.add_argument("--once", action="store_true", help="Query once and exit")
    p.add_argument("--reconcile", action="store_true", help="Run position reconciliation")
    return p.parse_args()


def print_snapshot(snapshot):
    """Pretty-print an AccountSnapshot."""
    print(f"\n{'=' * 60}")
    print(f"  Total Asset : {snapshot.total_asset:>15,.2f}")
    print(f"  Cash        : {snapshot.cash:>15,.2f}")
    print(f"  Frozen Cash : {snapshot.frozen_cash:>15,.2f}")
    print(f"  Market Value: {snapshot.market_value:>15,.2f}")
    print(f"{'=' * 60}")

    if snapshot.positions:
        print(f"  {'Stock':<12} {'Volume':>8} {'Sellable':>8} {'AvgPrice':>10} {'MktValue':>12} {'Profit':>10}")
        print(f"  {'-' * 62}")
        for code, pos in sorted(snapshot.positions.items()):
            print(f"  {code:<12} {pos.volume:>8d} {pos.sellable_volume:>8d} "
                  f"{pos.avg_price:>10.3f} {pos.market_value:>12,.2f} {pos.profit:>10,.2f}")
    else:
        print("  (no positions)")
    print()


def main():
    args = parse_args()
    from qlib.contrib.broker.miniQMT.trader.trading_loop import TradingLoop

    loop = TradingLoop(
        mini_qmt_path=args.mini_qmt_path,
        account_id=args.account_id,
    )

    logger.info("Connecting to miniQMT at %s …", args.mini_qmt_path)
    loop.connect()
    logger.info("Connected.")

    try:
        while True:
            snapshot = loop.account_adapter.get_snapshot()
            print_snapshot(snapshot)

            if args.reconcile:
                qlib_pos = loop.position_sync.build_qlib_position()
                diffs = loop.position_sync.reconcile(qlib_pos)
                if diffs:
                    logger.warning("Reconciliation discrepancies:")
                    for d in diffs:
                        logger.warning("  %s", d)
                else:
                    logger.info("Positions reconciled — no discrepancies.")

            if args.once:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        loop.disconnect()
        logger.info("Disconnected.")


if __name__ == "__main__":
    main()
