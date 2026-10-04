# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
The motivation of this demo
- To show the data modules of Qlib is Serializable, users can dump processed data to disk to avoid duplicated data preprocessing
"""

from copy import deepcopy
from pathlib import Path
import pickle
from pprint import pprint
from ruamel.yaml import YAML
import subprocess

from qlib import init
from qlib.data.dataset.handler import DataHandlerLP
from qlib.log import TimeInspector
from qlib.model.trainer import task_train
from qlib.utils import init_instance_by_config

# For general purpose, we use relative path
DIRNAME = Path(__file__).absolute().resolve().parent

if __name__ == "__main__":
    init()

    repeat = 2
    exp_name = "data_mem_reuse_demo"

    config_path = DIRNAME.parent / "benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml"
    yaml = YAML(typ="safe", pure=True)
    task_config = yaml.load(config_path.open())

    # 1) without using processed data in memory
    with TimeInspector.logt("The original time without reusing processed data in memory:"):
        for i in range(repeat):
            task_train(task_config["task"], experiment_name=exp_name)

    # 2) prepare processed data in memory.
    hd_conf = task_config["task"]["dataset"]["kwargs"]["handler"]
    pprint(hd_conf)
    hd: DataHandlerLP = init_instance_by_config(hd_conf)

    # 3) with reusing processed data in memory
    # NOTE: 第 1) 步的 task_train 会原地填充 task 里的 <MODEL>/<DATASET> 占位符，
    # 且结束时 dump_all=False 会剥离 handler 的内存数据（_infer/_learn），
    # 直接 deepcopy(task_config["task"]) 会复用这些被处置过的旧对象而报错，
    # 故重新从 YAML 加载干净的 task 配置
    new_task = deepcopy(yaml.load(config_path.open())["task"])
    new_task["dataset"]["kwargs"]["handler"] = hd
    print(new_task)

    with TimeInspector.logt("The time with reusing processed data in memory:"):
        # this will save the time to reload and process data from disk(in `DataHandlerLP`)
        # It still takes a lot of time in the backtest phase
        # NOTE: 原版循环 repeat 次不可行——第一次 task_train 收尾的
        # dataset.config(dump_all=False, recursive=True) 会剥离共享 handler 的内存数据，
        # 之后再复用同一 handler 会得到空数据，故仅执行一次
        task_train(new_task, experiment_name=exp_name)

    # 4) User can change other parts exclude processed data in memory(handler)
    # NOTE: 与第 3 步同理，重新加载干净的 task 配置；训练段改成本机数据窗口内的子区间
    new_task = deepcopy(yaml.load(config_path.open())["task"])
    new_task["dataset"]["kwargs"]["handler"] = hd
    new_task["dataset"]["kwargs"]["segments"]["train"] = ("2020-01-02", "2021-12-31")
    with TimeInspector.logt("The time with reusing processed data in memory:"):
        task_train(new_task, experiment_name=exp_name)
