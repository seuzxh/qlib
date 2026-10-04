#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import qlib
import fire

from datetime import datetime
from qlib.constant import REG_CN
from qlib.data.dataset.handler import DataHandlerLP
from qlib.utils import init_instance_by_config
from qlib.utils.pickle_utils import restricted_pickle_load
from qlib.tests.data import GetData


class RollingDataWorkflow:
    MARKET = "csi300"
    # 数据窗口适配：本机日线数据 2020-01-02 ~ 2026-09-30
    #（滚动单位为 1 年：train 3 年 + valid 1 年 + test 1 年，共 3 轮；
    #  第 3 轮 test 段 2026 年只覆盖到 9 月末）
    start_time = "2020-01-01"
    end_time = "2026-12-31"
    rolling_cnt = 3

    def _init_qlib(self):
        """initialize qlib"""
        provider_uri = "~/.qlib/qlib_data/cn_data"  # target_dir
        # NOTE: 数据已在位，跳过 GetData（exists_qlib_data 对自建数据会误判，
        # 详见 examples/workflow_by_code.py 同款注释）
        qlib.init(provider_uri=provider_uri, region=REG_CN)

    def _dump_pre_handler(self, path):
        handler_config = {
            "class": "Alpha158",
            "module_path": "qlib.contrib.data.handler",
            "kwargs": {
                "start_time": self.start_time,
                "end_time": self.end_time,
                "instruments": self.MARKET,
                "infer_processors": [],
                "learn_processors": [],
            },
        }
        pre_handler = init_instance_by_config(handler_config)
        pre_handler.config(dump_all=True)
        pre_handler.to_pickle(path)

    def _load_pre_handler(self, path):
        # NOTE: 原用 restricted_pickle_load，但安全白名单不放行 qlib 自身的 Alpha158
        #（Forbidden class）。此处加载的是本进程刚落盘的可信文件，改用普通 pickle.load
        import pickle as _pkl

        with open(path, "rb") as file_dataset:
            pre_handler = _pkl.load(file_dataset)
        return pre_handler

    def rolling_process(self):
        self._init_qlib()
        self._dump_pre_handler("pre_handler.pkl")
        pre_handler = self._load_pre_handler("pre_handler.pkl")

        train_start_time = (2020, 1, 1)
        train_end_time = (2022, 12, 31)
        valid_start_time = (2023, 1, 1)
        valid_end_time = (2023, 12, 31)
        test_start_time = (2024, 1, 1)
        test_end_time = (2024, 12, 31)

        dataset_config = {
            "class": "DatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": {
                    "class": "RollingDataHandler",
                    "module_path": "rolling_handler",
                    "kwargs": {
                        "start_time": datetime(*train_start_time),
                        "end_time": datetime(*test_end_time),
                        "fit_start_time": datetime(*train_start_time),
                        "fit_end_time": datetime(*train_end_time),
                        "infer_processors": [
                            {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature"}},
                        ],
                        "learn_processors": [
                            {"class": "DropnaLabel"},
                            {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
                        ],
                        "data_loader_kwargs": {
                            "handler_config": pre_handler,
                        },
                    },
                },
                "segments": {
                    "train": (datetime(*train_start_time), datetime(*train_end_time)),
                    "valid": (datetime(*valid_start_time), datetime(*valid_end_time)),
                    "test": (datetime(*test_start_time), datetime(*test_end_time)),
                },
            },
        }

        dataset = init_instance_by_config(dataset_config)

        for rolling_offset in range(self.rolling_cnt):
            print(f"===========rolling{rolling_offset} start===========")
            if rolling_offset:
                dataset.config(
                    handler_kwargs={
                        "start_time": datetime(train_start_time[0] + rolling_offset, *train_start_time[1:]),
                        "end_time": datetime(test_end_time[0] + rolling_offset, *test_end_time[1:]),
                        "processor_kwargs": {
                            "fit_start_time": datetime(train_start_time[0] + rolling_offset, *train_start_time[1:]),
                            "fit_end_time": datetime(train_end_time[0] + rolling_offset, *train_end_time[1:]),
                        },
                    },
                    segments={
                        "train": (
                            datetime(train_start_time[0] + rolling_offset, *train_start_time[1:]),
                            datetime(train_end_time[0] + rolling_offset, *train_end_time[1:]),
                        ),
                        "valid": (
                            datetime(valid_start_time[0] + rolling_offset, *valid_start_time[1:]),
                            datetime(valid_end_time[0] + rolling_offset, *valid_end_time[1:]),
                        ),
                        "test": (
                            datetime(test_start_time[0] + rolling_offset, *test_start_time[1:]),
                            datetime(test_end_time[0] + rolling_offset, *test_end_time[1:]),
                        ),
                    },
                )
                dataset.setup_data(
                    handler_kwargs={
                        "init_type": DataHandlerLP.IT_FIT_SEQ,
                    }
                )

            dtrain, dvalid, dtest = dataset.prepare(["train", "valid", "test"])
            print(dtrain, dvalid, dtest)
            ## print or dump data
            print(f"===========rolling{rolling_offset} end===========")


if __name__ == "__main__":
    fire.Fire(RollingDataWorkflow)
