# 第 1 课动手实践代码（基于 examples/workflow_by_code.py 改造）
# 用法：在仓库根目录执行  python docs/learning/code/01_task.py
# 任务点见下方 【任务N】注释

import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord
from qlib.tests.data import GetData
from qlib.tests.config import CSI300_BENCH, CSI300_GBDT_TASK

if __name__ == "__main__":
    provider_uri = "~/.qlib/qlib_data/cn_data"
    GetData().qlib_data(target_dir=provider_uri, region=REG_CN, exists_skip=True)
    qlib.init(provider_uri=provider_uri, region=REG_CN)

    model = init_instance_by_config(CSI300_GBDT_TASK["model"])
    dataset = init_instance_by_config(CSI300_GBDT_TASK["dataset"])

    # 【任务3】观察 Alpha158 因子数量与数据形状
    train_df = dataset.prepare("train")
    print("train shape:", train_df.shape)
    print("因子数(列数):", train_df.shape[1])
    print(train_df.head())

    # 【任务2】把 topk=50, n_drop=5 改成 topk=30, n_drop=3，重跑对比指标
    port_analysis_config = {
        "executor": {
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True},
        },
        "strategy": {
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy.signal_strategy",
            "kwargs": {
                "signal": (model, dataset),
                "topk": 30,      # 原值 50
                "n_drop": 3,     # 原值 5
            },
        },
        "backtest": {
            "start_time": "2017-01-01",
            "end_time": "2020-08-01",
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

    with R.start(experiment_name="workflow_lesson01"):
        R.log_params(**flatten_dict(CSI300_GBDT_TASK))
        model.fit(dataset)
        R.save_objects(**{"params.pkl": model})

        recorder = R.get_recorder()
        SignalRecord(model, dataset, recorder).generate()
        SigAnaRecord(recorder).generate()
        PortAnaRecord(recorder, port_analysis_config, "day").generate()

        # 【任务4】查看预测结果格式（datetime + instrument 两级索引的 Series）
        pred = recorder.load_object("pred.pkl")
        print("pred type:", type(pred))
        print(pred.head())
        print("IC:", recorder.list_metrics().get("IC"))
