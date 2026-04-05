"""Train LGBModel on Alpha158 features, then run backtest and risk analysis.

Usage:
    python scripts/miniQMT/train_and_backtest.py
    python scripts/miniQMT/train_and_backtest.py --market csi300 --topk 30 --save-model model.pkl
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.miniQMT.config import (
    DEFAULT_ACCOUNT_AMOUNT,
    DEFAULT_BENCHMARK,
    DEFAULT_EXCHANGE_KWARGS,
    DEFAULT_LGB_KWARGS,
    DEFAULT_MARKET,
    DEFAULT_N_DROP,
    DEFAULT_PROVIDER_URI,
    DEFAULT_TOPK,
    setup_logging,
)

logger = setup_logging("train_backtest")


def parse_args():
    p = argparse.ArgumentParser(description="Alpha158 + LGBModel train → backtest → risk analysis")
    p.add_argument("--provider-uri", default=DEFAULT_PROVIDER_URI)
    p.add_argument("--market", default=DEFAULT_MARKET)
    p.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    p.add_argument("--train-start", default="2020-01-01")
    p.add_argument("--train-end", default="2022-06-30")
    p.add_argument("--valid-start", default="2022-07-01")
    p.add_argument("--valid-end", default="2022-12-31")
    p.add_argument("--test-start", default="2023-01-01")
    p.add_argument("--test-end", default="2023-12-31")
    p.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    p.add_argument("--n-drop", type=int, default=DEFAULT_N_DROP)
    p.add_argument("--account", type=float, default=DEFAULT_ACCOUNT_AMOUNT)
    p.add_argument("--save-model", type=str, default=None, help="Path to save trained model (.pkl)")
    return p.parse_args()


def main():
    args = parse_args()

    import qlib
    from qlib.contrib.data.handler import Alpha158
    from qlib.contrib.evaluate import backtest_daily, risk_analysis
    from qlib.contrib.model.gbdt import LGBModel
    from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
    from qlib.data.dataset import DatasetH

    # 1. Init qlib
    qlib.init(provider_uri=args.provider_uri, region="cn")
    logger.info("Qlib initialized with provider_uri=%s", args.provider_uri)

    # 2. Data handler
    handler = Alpha158(
        instruments=args.market,
        start_time=args.train_start,
        end_time=args.test_end,
        fit_start_time=args.train_start,
        fit_end_time=args.train_end,
    )
    dataset = DatasetH(
        handler=handler,
        segments={
            "train": (args.train_start, args.train_end),
            "valid": (args.valid_start, args.valid_end),
            "test": (args.test_start, args.test_end),
        },
    )

    # 3. Train
    model = LGBModel(**DEFAULT_LGB_KWARGS)
    logger.info("Training LGBModel …")
    model.fit(dataset)
    logger.info("Training complete.")

    if args.save_model:
        model.to_pickle(args.save_model)
        logger.info("Model saved to %s", args.save_model)

    # 4. Predict
    pred = model.predict(dataset, segment="test")
    logger.info("Prediction shape: %s", pred.shape)

    # 5. Backtest
    strategy_config = {
        "class": "TopkDropoutStrategy",
        "module_path": "qlib.contrib.strategy.signal_strategy",
        "kwargs": {
            "signal": pred,
            "topk": args.topk,
            "n_drop": args.n_drop,
        },
    }
    report, positions = backtest_daily(
        start_time=args.test_start,
        end_time=args.test_end,
        strategy=strategy_config,
        account=args.account,
        benchmark=args.benchmark,
        exchange_kwargs=DEFAULT_EXCHANGE_KWARGS,
    )

    # 6. Risk analysis
    analysis = risk_analysis(report["return"] - report["bench"] - report["cost"])
    logger.info("=== Risk Analysis ===")
    for metric, value in analysis["risk"].items():
        logger.info("  %-25s: %.6f", metric, value)

    return report, analysis


if __name__ == "__main__":
    main()
