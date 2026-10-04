#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.
"""
Qlib provides two kinds of interfaces.
(1) Users could define the Quant research workflow by a simple configuration.
(2) Qlib is designed in a modularized way and supports creating research workflow by code just like building blocks.

The interface of (1) is `qrun XXX.yaml`.  The interface of (2) is script like this, which nearly does the same thing as `qrun XXX.yaml`
"""

from copy import deepcopy

import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord
from qlib.tests.data import GetData
from qlib.tests.config import CSI300_BENCH, CSI300_GBDT_TASK

if __name__ == "__main__":
    # use default data
    provider_uri = "~/.qlib/qlib_data/cn_data"  # target_dir
    # NOTE: 原版此处调用 GetData().qlib_data(..., exists_skip=True)。
    # 本机自建数据含指数特征（sh000300 等用于回测基准）且不在 instruments/all.txt 中，
    # exists_qlib_data 会误判"数据不存在"，进而走下载+删除(delete_old=True)分支，
    # 曾在交互确认处试图删除整个数据目录。数据已在位，这里直接初始化 qlib。
    qlib.init(provider_uri=provider_uri, region=REG_CN)

    # 数据窗口适配：本机数据日线特征覆盖 2020-01-02 ~ 2026-09-30，
    # 且 instruments/csi300.txt 成分股更新至 2026-07-02，
    # 官方默认 segments（train 2008-2014 / valid 2015-2016 / test 2017-2020.08）
    # 超出数据覆盖会取到空数据，因此按本机数据窗口重设：
    # train 4 年 / valid 1 年 / test 约 18 个月，回测区间与 test 对齐。
    task = deepcopy(CSI300_GBDT_TASK)
    task["dataset"]["kwargs"]["segments"] = {
        "train": ("2020-01-02", "2023-12-31"),
        "valid": ("2024-01-01", "2024-12-31"),
        "test": ("2025-01-01", "2026-07-01"),
    }
    task["dataset"]["kwargs"]["handler"]["kwargs"]["start_time"] = "2020-01-02"
    task["dataset"]["kwargs"]["handler"]["kwargs"]["end_time"] = "2026-07-01"

    model = init_instance_by_config(task["model"])
    dataset = init_instance_by_config(task["dataset"])

    port_analysis_config = {
        "executor": {
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {
                "time_per_step": "day",
                "generate_portfolio_metrics": True,
            },
        },
        "strategy": {
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy.signal_strategy",
            "kwargs": {
                "signal": (model, dataset),
                "topk": 50,
                "n_drop": 5,
            },
        },
        "backtest": {
            # 回测区间与 test 段对齐（受 csi300 成分股数据截至 2026-07-02 限制）
            "start_time": "2025-01-01",
            "end_time": "2026-07-01",
            "account": 100000000,
            "benchmark": CSI300_BENCH,
            "exchange_kwargs": {
                "freq": "day",
                "limit_threshold": 0.095,
                "deal_price": "close",
                "open_cost": 0.0005,
                "close_cost": 0.0015,
                "min_cost": 5,
            },
        },
    }

    # NOTE: This line is optional
    # It demonstrates that the dataset can be used standalone.
    example_df = dataset.prepare("train")
    print(example_df.head())

    # start exp
    with R.start(experiment_name="workflow"):
        R.log_params(**flatten_dict(task))
        model.fit(dataset)
        R.save_objects(**{"params.pkl": model})

        # prediction
        recorder = R.get_recorder()
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()

        # Signal Analysis
        sar = SigAnaRecord(recorder)
        sar.generate()

        # backtest. If users want to use backtest based on their own prediction,
        # please refer to https://qlib.readthedocs.io/en/latest/component/recorder.html#record-template.
        par = PortAnaRecord(recorder, port_analysis_config, "day")
        par.generate()
