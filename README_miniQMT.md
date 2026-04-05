# miniQMT 集成使用指南

## 概述

`qlib/contrib/broker/miniQMT/` 模块将 Qlib 量化研究框架与 miniQMT (xtquant) 对接，支持中国 A 股市场的量化投资全流程：

1. **数据采集回测** — 通过 `XtDataCollector` 从 xtdata 下载历史行情，转换为 Qlib `.bin` 格式，使用 Qlib 标准流程进行策略回测
2. **实盘交易** — 通过 `TradingLoop` 将 Qlib 策略信号路由到 miniQMT 客户端，完成真实下单

```
架构示意：

┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  xtdata API │────▶│  XtDataCollector  │────▶│  Qlib .bin 数据  │
│  (历史行情)  │     │  (数据采集转换)    │     │  (回测用)        │
└─────────────┘     └──────────────────┘     └────────┬────────┘
                                                       │
                                                       ▼
                                              ┌──────────────────┐
                                              │  Qlib 策略回测    │
                                              │  (模型训练/预测)   │
                                              └────────┬─────────┘
                                                       │ 目标持仓
                                                       ▼
┌─────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  miniQMT    │◀────│   TradingLoop    │◀────│  策略信号/权重     │
│  (券商交易端)│     │  (实盘交易循环)    │     │                  │
└─────────────┘     └──────────────────┘     └──────────────────┘
```

---

## 环境准备

### 基础依赖

- Python 3.8+
- Qlib 安装（参考项目根目录 README）

```bash
pip install -e .  # 或 make install
```

### miniQMT 客户端

1. 从券商处获取 miniQMT 安装包并安装（如国金证券 QMT 交易端）
2. 启动 miniQMT 客户端，登录交易账户
3. 确保客户端处于**已连接**状态（程序需要通过 miniQMT 客户端中转下单）

### xtquant 安装

xtquant 包随 miniQMT 发行包提供，**不在 PyPI 上发布**。安装方式：

```bash
# 方式一：将 miniQMT 安装目录下的 xtquant 包复制到 Python 环境
# 通常位于：<miniQMT安装目录>/bin.x64/xtquant/
cp -r "D:/国金证券QMT交易端/bin.x64/xtquant" $(python -c "import site; print(site.getsitepackages()[0])")

# 方式二：添加到 PYTHONPATH
export PYTHONPATH="D:/国金证券QMT交易端/bin.x64:$PYTHONPATH"
```

验证安装：

```bash
python -c "import xtquant; print('OK')"
```

---

## 数据采集、模型训练与回测

### 第一步：采集数据

使用 `XtDataCollector` 从 xtdata 下载历史行情并转换为 Qlib `.bin` 格式：

```python
from qlib.contrib.broker.miniQMT.data.data_collector import XtDataCollector

collector = XtDataCollector(target_dir="~/.qlib/qlib_data/xt_cn_data")

# 全量采集：指定股票列表和时间范围
collector.collect(
    stock_list=["600000.SH", "000001.SZ"],  # xtquant 格式股票代码
    start_date="20200101",
    end_date="20231231",
)

# 或不指定 stock_list，自动获取全部沪深 A 股
collector.collect(start_date="20200101", end_date="20231231")
```

采集完成后，数据目录结构如下：

```
~/.qlib/qlib_data/xt_cn_data/
├── calendars/
│   └── day.txt              # 交易日历
├── instruments/
│   └── all.txt              # 股票列表及上市区间
└── features/
    ├── SH600000/
    │   ├── $open.bin
    │   ├── $high.bin
    │   ├── $low.bin
    │   ├── $close.bin
    │   ├── $volume.bin
    │   ├── $amount.bin
    │   └── $factor.bin       # 复权因子
    ├── SZ000001/
    │   └── ...
    └── ...
```

#### 增量更新

已有数据后，可使用增量更新只下载最近几天的数据：

```python
collector.collect_incremental(days_back=5)
```

`days_back` 参数控制重叠天数以确保数据连续性。

### 第二步：初始化 Qlib

```python
import qlib
from qlib.data import D

# 初始化 Qlib，指向采集的数据目录
qlib.init(provider_uri="~/.qlib/qlib_data/xt_cn_data", region="cn")

# 验证数据可访问
df = D.features(["SH600000"], ["$close", "$volume"], start_time="2023-01-01", end_time="2023-12-31")
print(df.head())
```

### 第三步：模型训练（Python API）

使用 Alpha158 数据处理器 + LightGBM 模型进行训练：

```python
import qlib
from qlib.data.dataset import DatasetH
from qlib.contrib.data.handler import Alpha158
from qlib.contrib.model.gbdt import LGBModel

qlib.init(provider_uri="~/.qlib/qlib_data/xt_cn_data", region="cn")

# 1. 创建数据处理器（Alpha158 包含 158 个量价因子）
handler = Alpha158(
    instruments="csi300",
    start_time="2020-01-01",
    end_time="2023-12-31",
    fit_start_time="2020-01-01",
    fit_end_time="2022-12-31",
)

# 2. 创建数据集，划分训练/验证/测试集
dataset = DatasetH(
    handler=handler,
    segments={
        "train": ("2020-01-01", "2022-06-30"),
        "valid": ("2022-07-01", "2022-12-31"),
        "test":  ("2023-01-01", "2023-12-31"),
    },
)

# 3. 创建并训练 LightGBM 模型
model = LGBModel(
    loss="mse",
    colsample_bytree=0.8879,
    learning_rate=0.2,
    subsample=0.8789,
    lambda_l1=205.6999,
    lambda_l2=580.9768,
    max_depth=8,
    num_leaves=210,
    num_threads=20,
)
model.fit(dataset)

# 4. 生成预测
pred = model.predict(dataset)
print(pred.head(20))
```

### 第四步：回测评估（Python API）

使用 `backtest_daily` 进行策略回测并分析结果：

```python
from qlib.contrib.evaluate import backtest_daily, risk_analysis
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy

# 1. 创建策略：持有预测分数最高的 50 只股票，每天换 5 只
strategy_config = {
    "class": "TopkDropoutStrategy",
    "module_path": "qlib.contrib.strategy.signal_strategy",
    "kwargs": {
        "signal": pred,       # 模型预测结果
        "topk": 50,           # 持仓股票数
        "n_drop": 5,          # 每日最大换仓数
    },
}

# 2. 执行回测
report_normal, positions_normal = backtest_daily(
    start_time="2023-01-01",
    end_time="2023-12-31",
    strategy=strategy_config,
    account=100_000_000,          # 初始资金 1 亿
    benchmark="SH000300",         # 基准：沪深 300
    exchange_kwargs={
        "limit_threshold": 0.095, # 涨跌停阈值
        "deal_price": "close",    # 收盘价成交
        "open_cost": 0.0005,      # 买入佣金
        "close_cost": 0.0015,     # 卖出佣金（含印花税）
        "min_cost": 5,            # 最低佣金
    },
)

# 3. 分析回测指标
analysis = risk_analysis(report_normal["return"] - report_normal["bench"])
print("=== 回测结果 ===")
print(f"年化收益率: {analysis['annualized_return']:.4f}")
print(f"信息比率:   {analysis['information_ratio']:.4f}")
print(f"最大回撤:   {analysis['max_drawdown']:.4f}")
```

#### 回测结果可视化

```python
from qlib.contrib.report.analysis_position.report import report_graph
from qlib.contrib.report.analysis_model.analysis_model_performance import model_performance_graph

# 组合分析图表（净值曲线、回撤等）
report_graph(report_normal, show_notebook=True)

# 模型分析图表（IC、分组收益等）
# pred_label 需要包含预测值和标签列
pred_label = dataset.prepare("test", col_set=["feature", "label"], data_key="infer")
pred_label["score"] = pred
model_performance_graph(pred_label, show_notebook=True)
```

### 使用 qrun 命令行运行完整流程

上面的训练和回测也可以通过一条命令完成。参考示例配置文件：

**`qlib/contrib/broker/miniQMT/config/examples/backtest_with_xtdata.yaml`**

以下是一个完整的 qrun 配置（含模型训练 + 回测），仅需修改 `provider_uri` 指向你的 xtdata 数据目录：

```yaml
qlib_init:
    provider_uri: "~/.qlib/qlib_data/xt_cn_data"
    region: cn

market: &market csi300
benchmark: &benchmark SH000300

data_handler_config: &data_handler_config
    start_time: "2020-01-01"
    end_time: "2023-12-31"
    fit_start_time: "2020-01-01"
    fit_end_time: "2022-12-31"
    instruments: *market

port_analysis_config: &port_analysis_config
    strategy:
        class: TopkDropoutStrategy
        module_path: qlib.contrib.strategy
        kwargs:
            signal: <PRED>
            topk: 50
            n_drop: 5
    backtest:
        start_time: "2023-01-01"
        end_time: "2023-12-31"
        account: 100000000
        benchmark: *benchmark
        exchange_kwargs:
            limit_threshold: 0.095
            deal_price: close
            open_cost: 0.0005
            close_cost: 0.0015
            min_cost: 5

task:
    model:
        class: LGBModel
        module_path: qlib.contrib.model.gbdt
        kwargs:
            loss: mse
            colsample_bytree: 0.8879
            learning_rate: 0.2
            subsample: 0.8789
            lambda_l1: 205.6999
            lambda_l2: 580.9768
            max_depth: 8
            num_leaves: 210
            num_threads: 20
    dataset:
        class: DatasetH
        module_path: qlib.data.dataset
        kwargs:
            handler:
                class: Alpha158
                module_path: qlib.contrib.data.handler
                kwargs: *data_handler_config
            segments:
                train: [2020-01-01, 2022-06-30]
                valid: [2022-07-01, 2022-12-31]
                test:  [2023-01-01, 2023-12-31]
    record:
        - class: SignalRecord
          module_path: qlib.workflow.record_temp
          kwargs:
            model: <MODEL>
            dataset: <DATASET>
        - class: SigAnaRecord
          module_path: qlib.workflow.record_temp
          kwargs:
            ana_long_short: False
            ann_scaler: 252
        - class: PortAnaRecord
          module_path: qlib.workflow.record_temp
          kwargs:
            config: *port_analysis_config
```

运行方式：

```bash
qrun qlib/contrib/broker/miniQMT/config/examples/backtest_with_xtdata.yaml
```

运行后可通过 MLflow 查看实验结果：

```bash
mlflow ui  # 默认端口 5000，浏览器打开 http://localhost:5000
```

---

## 实盘监控与交易

### 连接 miniQMT

```python
from qlib.contrib.broker.miniQMT.trader.trading_loop import TradingLoop

loop = TradingLoop(
    mini_qmt_path="D:/国金证券QMT交易端/userdata_mini",  # miniQMT userdata 目录
    account_id="1234567890",                              # 交易账号
    order_timeout=60.0,                                   # 单笔订单超时（秒）
)

loop.connect()
```

### 生成目标持仓

通过 Qlib 模型预测生成目标持仓权重（权重之和 <= 1.0，剩余为现金）：

```python
# 示例：使用模型预测结果生成目标权重
target_positions = {
    "SH600000": 0.05,   # 浦发银行 5%
    "SZ000001": 0.05,   # 平安银行 5%
    "SH601318": 0.08,   # 中国平安 8%
}
```

### 执行交易

```python
results = loop.run_once(target_positions=target_positions)

# results 是 [(order, trade_val, trade_cost, trade_price), ...] 列表
for order, trade_val, trade_cost, trade_price in results:
    print(f"{order.stock_id}: 成交金额={trade_val:.2f}, 手续费={trade_cost:.2f}, 成交价={trade_price:.2f}")
```

`run_once()` 的执行流程：

1. 从券商查询当前持仓（若未传入 `current_position`）
2. 从券商查询总资产（若未传入 `total_value`）
3. 对比目标持仓与当前持仓，生成买卖订单
4. **先执行卖单，再执行买单**（释放资金后再买入）
5. 下单后与券商端对账（reconcile），检测差异

#### T+1 合规

系统自动处理 A 股 T+1 规则：
- 卖出前查询 `can_use_volume`（可卖数量），当日买入的股票不可卖出
- 卖出数量自动截断到可卖数量
- 买卖数量按 100 股（1 手）取整

### 实盘账户监控

使用 `XtAccountAdapter` 定时查询账户状态：

```python
from qlib.contrib.broker.miniQMT.account.account_adapter import XtAccountAdapter

adapter = XtAccountAdapter(xt_trader=loop.xt_trader, account_id="1234567890")

# 获取完整账户快照
snapshot = adapter.get_snapshot()
print(f"总资产:   {snapshot.total_asset:,.2f}")
print(f"可用现金: {snapshot.cash:,.2f}")
print(f"冻结资金: {snapshot.frozen_cash:,.2f}")
print(f"持仓市值: {snapshot.market_value:,.2f}")
print(f"持仓数:   {len(snapshot.positions)}")

# 遍历持仓明细
for code, pos in snapshot.positions.items():
    print(f"  {code}: 数量={pos.volume}, 可卖={pos.sellable_volume}, "
          f"成本价={pos.avg_price:.2f}, 市值={pos.market_value:,.2f}, 盈亏={pos.profit:,.2f}")

# 单独查询接口
total = adapter.get_total_asset()
cash = adapter.get_cash()
sellable = adapter.get_sellable_amount("SH600000")
held_stocks = adapter.get_position_list()
```

### 仓位巡检与对账

使用 `PositionSynchronizer` 进行定时对账：

```python
from qlib.contrib.broker.miniQMT.account.position_sync import PositionSynchronizer

sync = PositionSynchronizer(xt_trader=loop.xt_trader, account_id="1234567890")

# 从券商构建 Qlib Position 对象（进程重启恢复时使用）
position = sync.build_qlib_position()

# 与本地仓位对账，返回差异列表
discrepancies = sync.reconcile(local_position)
if discrepancies:
    print("仓位差异：")
    for d in discrepancies:
        print(f"  - {d}")
else:
    print("仓位一致，无差异")

# 强制同步：以券商为准覆盖本地仓位（慎用）
sync.force_sync(local_position)
```

对账检查内容：
- 现金差异（容差 1 元）
- 本地有但券商没有的持仓
- 券商有但本地没有的持仓
- 同一股票持仓数量不一致（容差 0.5 股）

### 异常处理

#### 连接断开重连

```python
import time
import logging

logger = logging.getLogger(__name__)

def run_with_retry(loop, target_positions, max_retries=3, retry_delay=10):
    """带重连的交易执行"""
    for attempt in range(max_retries):
        try:
            if not loop._connected:
                logger.info("尝试重新连接 miniQMT...")
                loop.connect()
            results = loop.run_once(target_positions=target_positions)
            return results
        except ConnectionError as e:
            logger.warning(f"连接异常 (第 {attempt+1} 次): {e}")
            loop._connected = False
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
    raise ConnectionError(f"重连 {max_retries} 次后仍失败")
```

#### 订单超时处理

`TradingLoop` 的 `order_timeout` 参数控制单笔订单等待时间。超时后，`XtCallbackHandler` 返回已部分成交的结果（使用成交量加权均价）。如需调整：

```python
loop = TradingLoop(
    mini_qmt_path="D:/国金证券QMT交易端/userdata_mini",
    account_id="1234567890",
    order_timeout=120.0,  # 大单可设更长超时
)
```

### 日志配置

推荐在交易脚本开头添加日志配置，便于排查问题：

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.FileHandler("trading.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

# 降低第三方库日志级别
logging.getLogger("xtquant").setLevel(logging.WARNING)
```

关键日志来源：
- `qlib.contrib.broker.miniQMT.trader.trading_loop` — 交易循环和订单生成
- `qlib.contrib.broker.miniQMT.account.position_sync` — 对账同步日志
- `qlib.contrib.broker.miniQMT.trader.callback_handler` — 成交回调

### 定时自动交易脚本

以下是一个完整的每日自动交易脚本，串联模型预测 → 生成信号 → 执行交易 → 对账：

```python
#!/usr/bin/env python
"""daily_trading.py — 每日自动交易脚本"""

import logging
import sys
from datetime import datetime

import qlib
from qlib.data.dataset import DatasetH
from qlib.contrib.data.handler import Alpha158
from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.broker.miniQMT.trader.trading_loop import TradingLoop
from qlib.contrib.broker.miniQMT.data.data_collector import XtDataCollector

# ==================== 配置 ====================
PROVIDER_URI = "~/.qlib/qlib_data/xt_cn_data"
MINI_QMT_PATH = "D:/国金证券QMT交易端/userdata_mini"
ACCOUNT_ID = "1234567890"
MODEL_PATH = "lgb_model.pkl"       # 已训练好的模型路径（可选）
TOPK = 30                          # 持仓股票数
MAX_WEIGHT = 0.10                  # 单只股票最大权重
CASH_RESERVE = 0.05                # 现金保留比例

# ==================== 日志 ====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.FileHandler(f"trading_{datetime.now():%Y%m%d}.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("daily_trading")

def main():
    today = datetime.now().strftime("%Y-%m-%d")
    logger.info(f"===== 开始每日交易流程 {today} =====")

    # 1. 增量更新数据
    logger.info("Step 1: 增量更新行情数据")
    collector = XtDataCollector(target_dir=PROVIDER_URI)
    collector.collect_incremental(days_back=5)

    # 2. 初始化 Qlib
    logger.info("Step 2: 初始化 Qlib")
    qlib.init(provider_uri=PROVIDER_URI, region="cn")

    # 3. 创建数据集并生成预测
    logger.info("Step 3: 生成模型预测")
    handler = Alpha158(
        instruments="csi300",
        start_time="2020-01-01",
        end_time=today,
        fit_start_time="2020-01-01",
        fit_end_time="2023-12-31",
    )
    dataset = DatasetH(
        handler=handler,
        segments={"test": ("2024-01-01", today)},
    )

    model = LGBModel()
    model.load(MODEL_PATH)        # 加载已训练的模型
    pred = model.predict(dataset)

    # 4. 从预测结果生成目标持仓权重
    logger.info("Step 4: 生成目标持仓")
    latest_pred = pred.xs(pred.index.get_level_values(0)[-1], level=0)
    top_stocks = latest_pred.nlargest(TOPK)

    # 等权分配，限制单只最大权重
    weight = min(1.0 / TOPK, MAX_WEIGHT) * (1 - CASH_RESERVE)
    target_positions = {code: weight for code in top_stocks.index}
    logger.info(f"目标持仓 {len(target_positions)} 只: {list(target_positions.keys())[:5]}...")

    # 5. 连接 miniQMT 并执行交易
    logger.info("Step 5: 连接 miniQMT 并执行交易")
    loop = TradingLoop(
        mini_qmt_path=MINI_QMT_PATH,
        account_id=ACCOUNT_ID,
        order_timeout=60.0,
    )
    loop.connect()

    try:
        results = loop.run_once(target_positions=target_positions)
        logger.info(f"交易完成，共执行 {len(results)} 笔订单")
        for order, trade_val, trade_cost, trade_price in results:
            logger.info(f"  {order.stock_id}: 金额={trade_val:.2f}, 费用={trade_cost:.2f}, 价格={trade_price:.2f}")

        # 6. 对账
        logger.info("Step 6: 仓位对账")
        position = loop.position_sync.build_qlib_position()
        discrepancies = loop.position_sync.reconcile(position)
        if discrepancies:
            logger.warning(f"发现 {len(discrepancies)} 处仓位差异:")
            for d in discrepancies:
                logger.warning(f"  - {d}")
        else:
            logger.info("对账通过，仓位一致")

        # 7. 打印账户摘要
        snapshot = loop.account_adapter.get_snapshot()
        logger.info(f"账户摘要: 总资产={snapshot.total_asset:,.2f}, "
                    f"现金={snapshot.cash:,.2f}, 持仓数={len(snapshot.positions)}")
    finally:
        loop.disconnect()

    logger.info(f"===== 每日交易流程结束 {today} =====")

if __name__ == "__main__":
    main()
```

### Windows 任务计划程序部署

#### 方式一：使用 `schtasks` 命令行

```bash
# 创建每个交易日 09:35 执行的定时任务
schtasks /Create /TN "QlibDailyTrading" /TR "python D:\scripts\daily_trading.py" /SC DAILY /ST 09:35 /F

# 查看任务状态
schtasks /Query /TN "QlibDailyTrading"

# 删除任务
schtasks /Delete /TN "QlibDailyTrading" /F
```

#### 方式二：使用 GUI

1. 打开「任务计划程序」（Win+R → `taskschd.msc`）
2. 点击「创建基本任务」
3. 名称：`QlibDailyTrading`
4. 触发器：每天，开始时间 09:35
5. 操作：启动程序
   - 程序/脚本：`python`（或 conda 环境的完整路径，如 `C:\Users\xxx\anaconda3\envs\qlib\python.exe`）
   - 添加参数：`D:\scripts\daily_trading.py`
   - 起始于：`D:\scripts`
6. 在「条件」页取消勾选「只有在计算机使用交流电源时才启动此任务」
7. 在「设置」页勾选「如果任务失败，按以下频率重新启动」，设为 5 分钟 / 3 次

> **注意**：A 股交易时段为 9:30-11:30 和 13:00-15:00。建议将交易脚本安排在 9:35（开盘后）或 14:30（尾盘）执行，避开集合竞价时段。

### 断开连接

```python
loop.disconnect()
```

---

## 端到端工作流

以下是一个完整的端到端示例，串联数据采集 → Qlib 初始化 → 模型训练 → 回测评估 → 实盘交易：

```python
#!/usr/bin/env python
"""end_to_end_workflow.py — 从数据采集到实盘交易的完整流程"""

import qlib
from qlib.data.dataset import DatasetH
from qlib.contrib.data.handler import Alpha158
from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.evaluate import backtest_daily, risk_analysis
from qlib.contrib.broker.miniQMT.data.data_collector import XtDataCollector
from qlib.contrib.broker.miniQMT.trader.trading_loop import TradingLoop

# ========== 阶段 1: 数据采集 ==========
print("=== 阶段 1: 数据采集 ===")
collector = XtDataCollector(target_dir="~/.qlib/qlib_data/xt_cn_data")
collector.collect(start_date="20200101", end_date="20231231")

# ========== 阶段 2: 初始化 Qlib ==========
print("=== 阶段 2: 初始化 Qlib ===")
qlib.init(provider_uri="~/.qlib/qlib_data/xt_cn_data", region="cn")

# ========== 阶段 3: 模型训练 ==========
print("=== 阶段 3: 模型训练 ===")
handler = Alpha158(
    instruments="csi300",
    start_time="2020-01-01",
    end_time="2023-12-31",
    fit_start_time="2020-01-01",
    fit_end_time="2022-12-31",
)
dataset = DatasetH(
    handler=handler,
    segments={
        "train": ("2020-01-01", "2022-06-30"),
        "valid": ("2022-07-01", "2022-12-31"),
        "test":  ("2023-01-01", "2023-12-31"),
    },
)
model = LGBModel(
    loss="mse", colsample_bytree=0.8879, learning_rate=0.2,
    subsample=0.8789, lambda_l1=205.6999, lambda_l2=580.9768,
    max_depth=8, num_leaves=210, num_threads=20,
)
model.fit(dataset)
pred = model.predict(dataset)

# ========== 阶段 4: 回测评估 ==========
print("=== 阶段 4: 回测评估 ===")
report_normal, positions_normal = backtest_daily(
    start_time="2023-01-01",
    end_time="2023-12-31",
    strategy={
        "class": "TopkDropoutStrategy",
        "module_path": "qlib.contrib.strategy.signal_strategy",
        "kwargs": {"signal": pred, "topk": 50, "n_drop": 5},
    },
    account=100_000_000,
    benchmark="SH000300",
    exchange_kwargs={
        "limit_threshold": 0.095, "deal_price": "close",
        "open_cost": 0.0005, "close_cost": 0.0015, "min_cost": 5,
    },
)
analysis = risk_analysis(report_normal["return"] - report_normal["bench"])
print(f"年化收益: {analysis['annualized_return']:.4f}")
print(f"信息比率: {analysis['information_ratio']:.4f}")
print(f"最大回撤: {analysis['max_drawdown']:.4f}")

# ========== 阶段 5: 实盘交易（可选） ==========
print("=== 阶段 5: 实盘交易 ===")
# 获取最新一天的预测，生成目标持仓
latest = pred.xs(pred.index.get_level_values(0)[-1], level=0)
top30 = latest.nlargest(30)
target = {code: 0.03 for code in top30.index}  # 等权 3%

loop = TradingLoop(
    mini_qmt_path="D:/国金证券QMT交易端/userdata_mini",
    account_id="1234567890",
)
loop.connect()
try:
    results = loop.run_once(target_positions=target)
    print(f"执行 {len(results)} 笔交易")
finally:
    loop.disconnect()
```

---

## 进阶用法

### 自定义 Provider

模块提供三个自定义 Provider，适用于实盘交易中需要实时数据的场景：

#### XtQMTCalendarProvider

从 xtdata 获取交易日历，保证日历信息始终最新：

```python
from qlib.contrib.broker.miniQMT.data.calendar_provider import XtQMTCalendarProvider

provider = XtQMTCalendarProvider()
calendar = provider.load_calendar(freq="day", future=False)
```

#### XtQMTInstrumentProvider

从 xtdata 获取股票列表，支持按板块筛选：

```python
from qlib.contrib.broker.miniQMT.data.instrument_provider import XtQMTInstrumentProvider

provider = XtQMTInstrumentProvider()
# 支持的 market 名称: all, sse50, csi300, csi500, sse, szse
instruments = provider.list_instruments({"market": "csi300"}, as_list=True)
```

市场名称与 xtdata 板块的对应关系：

| Qlib market | xtdata 板块 |
|-------------|------------|
| `all`       | 沪深A股     |
| `sse50`     | 上证50     |
| `csi300`    | 沪深300    |
| `csi500`    | 中证500    |
| `sse`       | 上证A股    |
| `szse`      | 深证A股    |

#### XtQMTFeatureProvider

实时获取行情特征数据（适用于实盘，回测建议使用 `.bin` 文件）：

```python
from qlib.contrib.broker.miniQMT.data.feature_provider import XtQMTFeatureProvider

provider = XtQMTFeatureProvider(freq="1d")
close_series = provider.feature("SH600000", "$close", "2024-01-01", "2024-03-01", "day")
```

支持的字段：`$open`, `$high`, `$low`, `$close`, `$volume`, `$amount`

#### 在回测中使用自定义 Provider

可以在 `qlib.init()` 时注册自定义 Provider，使回测直接读取 xtdata 实时数据（适用于少量股票的快速验证）：

```python
import qlib

qlib.init(
    provider_uri="~/.qlib/qlib_data/xt_cn_data",
    region="cn",
    custom_ops=[],  # 可注册自定义算子
)

# 或通过 C 直接替换 Provider
from qlib.config import C
from qlib.contrib.broker.miniQMT.data.calendar_provider import XtQMTCalendarProvider
from qlib.contrib.broker.miniQMT.data.feature_provider import XtQMTFeatureProvider

# 注意：替换 Provider 需要了解 Qlib Provider 注册机制，建议仅在实盘场景使用
```

### 佣金配置

`XtQMTLiveExchange` 支持自定义佣金参数：

```python
from qlib.contrib.broker.miniQMT.executor.exchange_live import XtQMTLiveExchange

exchange = XtQMTLiveExchange(
    order_manager=order_manager,
    open_cost=0.00025,     # 买入佣金率 (万2.5)
    close_cost=0.00125,    # 卖出佣金率 (含印花税，千1.25)
    min_cost=5.0,          # 每笔最低佣金 5 元
    trade_unit=100,        # 每手 100 股
)
```

### 股票代码格式转换

`StockCodeConverter` 在 Qlib 格式和 xtquant 格式之间转换：

```python
from qlib.contrib.broker.miniQMT.utils import StockCodeConverter

# Qlib -> xtquant
StockCodeConverter.qlib_to_xt("SH600000")   # "600000.SH"

# xtquant -> Qlib
StockCodeConverter.xt_to_qlib("600000.SH")  # "SH600000"

# 批量转换
StockCodeConverter.qlib_to_xt_batch(["SH600000", "SZ000001"])  # ["600000.SH", "000001.SZ"]
```

### XtQMTLiveExecutor

`XtQMTLiveExecutor` 是 Qlib `BaseExecutor` 的实盘替代，可集成到 Qlib 的 executor 框架中：

```python
from qlib.contrib.broker.miniQMT.executor.live_executor import XtQMTLiveExecutor

executor = XtQMTLiveExecutor(
    time_per_step="day",
    live_exchange=live_exchange,
    trade_type="serial",  # 逐笔执行（实盘推荐）
)
```

### 账户查询

`XtAccountAdapter` 提供便捷的账户查询接口：

```python
from qlib.contrib.broker.miniQMT.account.account_adapter import XtAccountAdapter

adapter = XtAccountAdapter(xt_trader=loop.xt_trader, account_id="1234567890")

# 查询总资产
total = adapter.get_total_asset()

# 查询可用现金
cash = adapter.get_cash()

# 查询某只股票可卖数量（T+1 感知）
sellable = adapter.get_sellable_amount("SH600000")

# 获取完整账户快照
snapshot = adapter.get_snapshot()
print(f"总资产: {snapshot.total_asset}, 现金: {snapshot.cash}, 持仓数: {len(snapshot.positions)}")
```

---

## 模块结构

```
qlib/contrib/broker/miniQMT/
├── __init__.py                        # 模块入口，导出 XtDataCollector, TradingLoop
├── utils.py                           # 代码转换 (StockCodeConverter)、交易常量、工具函数
├── data/
│   ├── data_collector.py              # XtDataCollector — xtdata 数据采集转 .bin 格式
│   ├── calendar_provider.py           # XtQMTCalendarProvider — 实时交易日历
│   ├── instrument_provider.py         # XtQMTInstrumentProvider — 实时股票列表
│   └── feature_provider.py           # XtQMTFeatureProvider — 实时行情特征
├── trader/
│   ├── trading_loop.py                # TradingLoop — 实盘交易主循环
│   ├── order_manager.py               # XtOrderManager — 订单提交与生命周期管理
│   └── callback_handler.py            # XtCallbackHandler — 异步成交回调处理
├── executor/
│   ├── exchange_live.py               # XtQMTLiveExchange — 实盘交易所（替代模拟撮合）
│   └── live_executor.py               # XtQMTLiveExecutor — 实盘执行器（替代 SimulatorExecutor）
├── account/
│   ├── account_adapter.py             # XtAccountAdapter — 账户查询适配器
│   └── position_sync.py              # PositionSynchronizer — 仓位同步与对账
└── config/
    └── examples/
        ├── backtest_with_xtdata.yaml  # 回测示例配置
        └── live_trading.yaml          # 实盘交易示例配置
```

---

## 常见问题

### Q: `import xtquant` 报错 ModuleNotFoundError

xtquant 不在 PyPI 上发布，需要从 miniQMT 安装目录手动安装或添加到 `PYTHONPATH`。参考「环境准备」章节。

### Q: `connect()` 返回连接失败

- 确认 miniQMT 客户端已启动并登录
- 确认 `mini_qmt_path` 指向正确的 userdata_mini 目录
- 检查客户端是否处于已连接状态

### Q: 卖出订单未执行

A 股 T+1 规则：当日买入的股票当日不可卖出。系统会自动查询 `can_use_volume`，若可卖数量为 0 则跳过卖单。

### Q: 回测数据和实盘数据有差异

- 回测使用 `XtDataCollector` 采集的 `.bin` 文件，是一次性快照
- 建议定期执行 `collect_incremental()` 更新数据
- 实盘环境可搭配 `XtQMTFeatureProvider` 获取最新实时行情

### Q: 部分成交（Partial Fill）如何处理

`XtCallbackHandler` 内部通过 `OrderWaiter` 累积部分成交，使用成交量加权平均计算成交价。当累计成交量达到目标数量或超时后返回结果。

### Q: 如何每日自动执行交易

参考「定时自动交易脚本」章节的 `daily_trading.py` 示例和「Windows 任务计划程序部署」章节。

### Q: 如何切换不同模型

Qlib 的模型接口统一为 `fit()` / `predict()`，只需替换模型类即可：

```python
# LightGBM（默认推荐）
from qlib.contrib.model.gbdt import LGBModel
model = LGBModel(loss="mse", num_leaves=210, max_depth=8)

# XGBoost
from qlib.contrib.model.xgboost import XGBModel
model = XGBModel()

# CatBoost
from qlib.contrib.model.catboost_model import CatBoostModel
model = CatBoostModel()

# 线性模型
from qlib.contrib.model.linear import LinearModel
model = LinearModel()

# 深度学习模型（需要 PyTorch）
from qlib.contrib.model.pytorch_lstm import LSTM
model = LSTM(d_feat=158, hidden_size=64, num_layers=2)

# 训练和预测接口完全一致
model.fit(dataset)
pred = model.predict(dataset)
```

在 YAML 配置中切换模型也很简单，只需修改 `task.model` 部分：

```yaml
task:
    model:
        class: XGBModel                     # 修改这里
        module_path: qlib.contrib.model.xgboost  # 和这里
        kwargs:
            # 模型特有参数
```

### Q: 数据频率支持（日频 vs 分钟频）

`XtDataCollector` 的 `freq` 参数控制数据频率：

```python
# 日频数据（默认）
collector = XtDataCollector(target_dir="~/.qlib/qlib_data/xt_cn_1d", freq="1d")

# 1 分钟数据
collector = XtDataCollector(target_dir="~/.qlib/qlib_data/xt_cn_1m", freq="1m")

# 5 分钟数据
collector = XtDataCollector(target_dir="~/.qlib/qlib_data/xt_cn_5m", freq="5m")
```

注意分钟频数据量较大，采集时间会更长。对应的 Qlib 初始化和回测也需要调整频率参数。
