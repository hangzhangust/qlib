"""End-to-end daily trading: data update -> predict -> generate targets -> execute orders -> reconcile.

Usage:
    python scripts/miniQMT/daily_trading.py --model-path model.pkl --dry-run
    python scripts/miniQMT/daily_trading.py --model-path model.pkl --topk 30
"""

import argparse
import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.miniQMT.config import (
    DEFAULT_ACCOUNT_ID,
    DEFAULT_CASH_RESERVE,
    DEFAULT_EXCHANGE_KWARGS,
    DEFAULT_MAX_WEIGHT,
    DEFAULT_MARKET,
    DEFAULT_MINI_QMT_PATH,
    DEFAULT_N_DROP,
    DEFAULT_ORDER_TIMEOUT,
    DEFAULT_PROVIDER_URI,
    DEFAULT_TOPK,
    setup_logging,
)

today_str = datetime.date.today().strftime("%Y%m%d")
logger = setup_logging("daily_trading", log_file=f"trading_{today_str}.log")


def parse_args():
    p = argparse.ArgumentParser(description="miniQMT daily auto-trading pipeline")
    p.add_argument("--provider-uri", default=DEFAULT_PROVIDER_URI)
    p.add_argument("--mini-qmt-path", default=DEFAULT_MINI_QMT_PATH)
    p.add_argument("--account-id", default=DEFAULT_ACCOUNT_ID)
    p.add_argument("--model-path", required=True, help="Path to trained model .pkl")
    p.add_argument("--market", default=DEFAULT_MARKET)
    p.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    p.add_argument("--n-drop", type=int, default=DEFAULT_N_DROP)
    p.add_argument("--max-weight", type=float, default=DEFAULT_MAX_WEIGHT)
    p.add_argument("--cash-reserve", type=float, default=DEFAULT_CASH_RESERVE)
    p.add_argument("--order-timeout", type=float, default=DEFAULT_ORDER_TIMEOUT)
    p.add_argument("--days-back", type=int, default=5, help="Incremental data overlap days")
    p.add_argument("--fit-start", default="2020-01-01", help="Handler fit start date")
    p.add_argument("--fit-end", default="2023-12-31", help="Handler fit end date")
    p.add_argument("--dry-run", action="store_true", help="Print target positions without executing")
    return p.parse_args()


def build_topk_dropout_positions(pred_today, current_holdings, topk, n_drop):
    """Build target positions using TopkDropout rotation logic.

    Mirrors TopkDropoutStrategy.generate_trade_decision() from
    qlib/contrib/strategy/signal_strategy.py.

    Parameters
    ----------
    pred_today : pd.Series
        Today's prediction scores indexed by instrument code.
    current_holdings : list[str]
        List of currently held stock codes.
    topk : int
        Number of stocks to hold.
    n_drop : int
        Number of stocks to rotate out per rebalance.

    Returns
    -------
    dict
        {stock_code: weight} for the target portfolio.
    """
    import pandas as pd

    # 1. Sort current holdings by today's prediction score
    last = pred_today.reindex(current_holdings).sort_values(ascending=False).index

    # 2. Candidate new stocks (not in current holdings), sorted by score
    today = pred_today[~pred_today.index.isin(last)].sort_values(ascending=False).index
    today = today[: n_drop + topk - len(last)]

    # 3. Combine and sort to find lowest-scored stocks to drop
    comb = pred_today.reindex(last.union(pd.Index(today))).sort_values(ascending=False).index

    # 4. Select stocks to sell (bottom n_drop of combined list, only from current holdings)
    sell = last[last.isin(comb[-n_drop:])] if len(comb) > 0 else last[:0]

    # 5. Select stocks to buy (enough to fill topk after sells)
    buy = today[: len(sell) + topk - len(last)]

    # 6. Final portfolio = current holdings minus sells plus buys
    final = [s for s in last if s not in sell] + list(buy)

    # 7. Equal weight
    if not final:
        return {}
    weight = 1.0 / len(final)
    return {s: weight for s in final}


def main():
    args = parse_args()

    # --- Step 1: Incremental data update ---
    logger.info("Step 1/7: Incremental data update (days_back=%d)", args.days_back)
    from qlib.contrib.broker.miniQMT.data.data_collector import XtDataCollector

    collector = XtDataCollector(target_dir=args.provider_uri)
    collector.collect_incremental(days_back=args.days_back)

    # --- Step 2: Init qlib ---
    logger.info("Step 2/7: Initializing Qlib")
    import qlib
    qlib.init(provider_uri=args.provider_uri, region="cn")

    # --- Step 3: Load model & predict ---
    logger.info("Step 3/7: Loading model and generating predictions")
    import pandas as pd
    from qlib.contrib.data.handler import Alpha158
    from qlib.contrib.model.gbdt import LGBModel
    from qlib.data.dataset import DatasetH

    today = pd.Timestamp.now().strftime("%Y-%m-%d")
    handler = Alpha158(
        instruments=args.market,
        start_time=args.fit_start,
        end_time=today,
        fit_start_time=args.fit_start,
        fit_end_time=args.fit_end,
    )
    dataset = DatasetH(
        handler=handler,
        segments={"test": (today, today)},
    )

    model = LGBModel.load(args.model_path)
    pred = model.predict(dataset, segment="test")
    logger.info("Predictions: %d rows", len(pred))

    if pred.empty:
        logger.error("No predictions generated for today. Exiting.")
        return

    # Get the latest date in predictions
    latest_date = pred.index.get_level_values("datetime").max()
    pred_today = pred.loc[latest_date]
    # Flatten to instrument-indexed Series
    if isinstance(pred_today.index, pd.MultiIndex):
        pred_today = pred_today.droplevel("datetime") if "datetime" in pred_today.index.names else pred_today
    pred_today = pred_today.squeeze()

    # --- Step 4: Connect to broker, get current holdings ---
    logger.info("Step 4/7: Connecting to broker and fetching current holdings")
    from qlib.contrib.broker.miniQMT.trader.trading_loop import TradingLoop

    loop = TradingLoop(
        mini_qmt_path=args.mini_qmt_path,
        account_id=args.account_id,
        order_timeout=args.order_timeout,
        open_cost=DEFAULT_EXCHANGE_KWARGS["open_cost"],
        close_cost=DEFAULT_EXCHANGE_KWARGS["close_cost"],
        min_cost=DEFAULT_EXCHANGE_KWARGS["min_cost"],
    )

    if not args.dry_run:
        loop.connect()
        current_holdings = loop.position_sync.build_qlib_position().get_stock_list()
        logger.info("Current holdings: %d stocks", len(current_holdings))
    else:
        current_holdings = []
        logger.info("DRY RUN - assuming empty holdings for TopkDropout calculation")

    # --- Step 5: Build target positions using TopkDropout logic ---
    logger.info("Step 5/7: Building target positions (topk=%d, n_drop=%d)", args.topk, args.n_drop)
    target_positions = build_topk_dropout_positions(
        pred_today, current_holdings, args.topk, args.n_drop
    )

    logger.info("Target positions (%d stocks):", len(target_positions))
    for code, w in sorted(target_positions.items(), key=lambda x: -x[1]):
        logger.info("  %-12s  %.4f", code, w)

    # --- Step 6: Dry-run check or execute ---
    if args.dry_run:
        logger.info("DRY RUN - no orders will be placed.")
        return

    logger.info("Step 6/7: Executing orders")
    try:
        results = loop.run_once(target_positions)
        logger.info("Executed %d orders.", len(results))
        for order, trade_val, trade_cost, trade_price in results:
            logger.info("  %s  val=%.2f  cost=%.2f  price=%.3f", order, trade_val, trade_cost, trade_price)

        # --- Step 7: Reconcile & summary ---
        logger.info("Step 7/7: Reconciliation and summary")
        qlib_pos = loop.position_sync.build_qlib_position()
        diffs = loop.position_sync.reconcile(qlib_pos)
        if diffs:
            logger.warning("Discrepancies found:")
            for d in diffs:
                logger.warning("  %s", d)
        else:
            logger.info("No discrepancies.")

        snapshot = loop.account_adapter.get_snapshot()
        logger.info("Account summary: total=%.2f  cash=%.2f  market_value=%.2f",
                     snapshot.total_asset, snapshot.cash, snapshot.market_value)
    finally:
        loop.disconnect()
        logger.info("Disconnected. Trading session complete.")


if __name__ == "__main__":
    main()
