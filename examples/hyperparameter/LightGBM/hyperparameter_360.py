import qlib
import optuna
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config
from qlib.tests.config import get_dataset_config, CSI300_MARKET, DATASET_ALPHA360_CLASS

# NOTE: 新版 get_dataset_config 的签名为 (dataset_class, train/valid/test, handler_kwargs)，
# 无 market 参数，标的全集改经 handler_kwargs 传入
DATASET_CONFIG = get_dataset_config(dataset_class=DATASET_ALPHA360_CLASS, handler_kwargs={"instruments": CSI300_MARKET})


def objective(trial):
    task = {
        "model": {
            "class": "LGBModel",
            "module_path": "qlib.contrib.model.gbdt",
            "kwargs": {
                "loss": "mse",
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1),
                "learning_rate": trial.suggest_float("learning_rate", 0, 1),
                "subsample": trial.suggest_float("subsample", 0, 1),
                "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 1e4, log=True),
                "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 1e4, log=True),
                "max_depth": 10,
                "num_leaves": trial.suggest_int("num_leaves", 1, 1024),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
                "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 50),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            },
        },
    }

    evals_result = dict()
    model = init_instance_by_config(task["model"])
    model.fit(dataset, evals_result=evals_result)
    # NOTE: 新版 LGBModel.fit 的 evals_result 为 {"valid": {指标名: [逐轮损失]}}，
    # min(字典) 取到的是键（字符串），须取第一个指标序列的最小值（详见 158 版注释）
    return float(min(next(iter(evals_result["valid"].values()))))


if __name__ == "__main__":
    provider_uri = "~/.qlib/qlib_data/cn_data"
    # NOTE: 数据已在位，跳过 GetData（对自建数据会误判，见 workflow_by_code.py 注释）
    qlib.init(provider_uri=provider_uri, region=REG_CN)

    # 数据窗口适配：按本机数据覆盖（2020-01-02 起）重设切分与 handler 日期
    from copy import deepcopy
    dataset = deepcopy(DATASET_CONFIG)
    dataset["kwargs"]["segments"] = {
        "train": ("2020-01-02", "2023-12-31"),
        "valid": ("2024-01-01", "2024-12-31"),
        "test": ("2025-01-01", "2026-07-01"),
    }
    dataset["kwargs"]["handler"]["kwargs"]["start_time"] = "2020-01-02"
    dataset["kwargs"]["handler"]["kwargs"]["end_time"] = "2026-07-01"
    # NOTE: 原配置的 fit_start/fit_end 是 <dataset.kwargs.segments...> 模板占位符，
    # Alpha360 在 handler 初始化即拟合处理器、模板尚未被解析会导致按字符串切片报错，
    # 故显式写死与训练段一致的日期
    dataset["kwargs"]["handler"]["kwargs"]["fit_start_time"] = "2020-01-02"
    dataset["kwargs"]["handler"]["kwargs"]["fit_end_time"] = "2023-12-31"
    # NOTE: 直传 init_instance_by_config 不经 qrun/task_train 的占位符解析，
    # Alpha360 的处理器链会消费 fit_*（模板串 "<dataset.kwargs...>" 未被替换会导致
    # 按字符串切片报错），这里一并设为具体日期（拟合只用训练窗，语义也更正确）
    dataset["kwargs"]["handler"]["kwargs"]["fit_start_time"] = "2020-01-02"
    dataset["kwargs"]["handler"]["kwargs"]["fit_end_time"] = "2023-12-31"
    dataset = init_instance_by_config(dataset)

    # NOTE: 存储改用独立文件 db_360.sqlite3（原版与 158 版同名会互相覆盖）
    study = optuna.create_study(study_name="LGBM_360", storage="sqlite:///db_360.sqlite3", direction="minimize", load_if_exists=True)
    # NOTE: 原版未设 n_trials 会无限搜索；多线程共享 dataset 不稳，改串行。
    # optuna _run_trial 的 finally 断言会偶发把 tell 阶段异常掩盖成
    # AssertionError: Should not reach（库健壮性问题），捕获后重启搜索即可
    # 续接 sqlite 中已完成的 trial
    import time as _time

    _done = 0
    while _done < 20:
        try:
            study.optimize(objective, n_trials=20 - _done, n_jobs=1)
            _done = 20
        except AssertionError:
            _done = len([t for t in study.trials if t.state.name == "COMPLETE"])
            print(f"optuna 偶发断言，已完成 {_done}/20，重启搜索续接")
            _time.sleep(2)
