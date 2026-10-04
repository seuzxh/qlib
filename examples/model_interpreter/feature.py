#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.


import qlib
from qlib.constant import REG_CN

from copy import deepcopy
from qlib.utils import init_instance_by_config
from qlib.tests.config import CSI300_GBDT_TASK

if __name__ == "__main__":
    # use default data
    provider_uri = "~/.qlib/qlib_data/cn_data"  # target_dir
    # NOTE: 数据已在位，跳过 GetData 下载（exists_qlib_data 对含指数特征的自建数据会
    # 误判而进入"下载+删除旧数据"分支，详见 examples/workflow_by_code.py 同款注释）
    qlib.init(provider_uri=provider_uri, region=REG_CN)

    # 数据窗口适配：与 workflow_by_code.py 相同，按本机数据覆盖重设 segments/日期
    task = deepcopy(CSI300_GBDT_TASK)
    task["dataset"]["kwargs"]["segments"] = {
        "train": ("2020-01-02", "2023-12-31"),
        "valid": ("2024-01-01", "2024-12-31"),
        "test": ("2025-01-01", "2026-07-01"),
    }
    task["dataset"]["kwargs"]["handler"]["kwargs"]["start_time"] = "2020-01-02"
    task["dataset"]["kwargs"]["handler"]["kwargs"]["end_time"] = "2026-07-01"

    ###################################
    # train model
    ###################################
    # model initialization
    model = init_instance_by_config(task["model"])
    dataset = init_instance_by_config(task["dataset"])
    model.fit(dataset)

    # get model feature importance
    feature_importance = model.get_feature_importance()
    print("feature importance:")
    print(feature_importance)
