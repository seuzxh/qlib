import qlib
import optuna
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config
from qlib.tests.config import CSI300_DATASET_CONFIG


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
    # NOTE: 新版 LGBModel.fit 将 evals_result 委托给 lightgbm 的 record_evaluation，
    # 结构为 {"valid": {指标名: [逐轮损失]}}；原版 min(evals_result["valid"]) 取的是
    # 字典键（字符串"l2"）而非损失序列，会向 optuna 返回字符串导致其内部异常，
    # 故改为取第一个指标序列的最小值
    return min(next(iter(evals_result["valid"].values())))


if __name__ == "__main__":
    provider_uri = "~/.qlib/qlib_data/cn_data"
    # NOTE: 数据已在位，跳过 GetData（对自建数据会误判，见 workflow_by_code.py 注释）
    qlib.init(provider_uri=provider_uri, region="cn")

    # 数据窗口适配：按本机数据覆盖（2020-01-02 起）重设切分与 handler 日期
    from copy import deepcopy
    dataset_config = deepcopy(CSI300_DATASET_CONFIG)
    dataset_config["kwargs"]["segments"] = {
        "train": ("2020-01-02", "2023-12-31"),
        "valid": ("2024-01-01", "2024-12-31"),
        "test": ("2025-01-01", "2026-07-01"),
    }
    dataset_config["kwargs"]["handler"]["kwargs"]["start_time"] = "2020-01-02"
    dataset_config["kwargs"]["handler"]["kwargs"]["end_time"] = "2026-07-01"
    dataset = init_instance_by_config(dataset_config)

    study = optuna.create_study(study_name="LGBM_158", storage="sqlite:///db.sqlite3", direction="minimize", load_if_exists=True)
    study.optimize(objective, n_trials=20, n_jobs=1)  # NOTE: 原版未设 n_trials 会无限搜索，此处限定 20 次；
    # n_jobs=6 时多线程共享同一 dataset 对象存在线程安全问题（optuna trial 异常结束），改串行
