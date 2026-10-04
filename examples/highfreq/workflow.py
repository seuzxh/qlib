#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import fire

import qlib
from qlib.constant import REG_CN
from qlib.config import HIGH_FREQ_CONFIG

from qlib.utils import init_instance_by_config
import pickle

# NOTE: 本示例原使用 restricted_pickle_load 重载数据集，但其安全白名单只放行
# pandas/numpy 与本地白名单类，DatasetH 及本目录自定义 Handler 均会被拒绝
#（Forbidden class）。此处加载的是本进程刚落盘的可信文件，故改用普通 pickle.load；
# 加载来源不明的 pickle 时仍应使用 restricted_pickle_load。
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.ops import Operators
from qlib.data.data import Cal
from qlib.tests.data import GetData

from highfreq_ops import get_calendar_day, DayLast, FFillNan, BFillNan, Date, Select, IsNull, Cut


class HighfreqWorkflow:
    SPEC_CONF = {"custom_ops": [DayLast, FFillNan, BFillNan, Date, Select, IsNull, Cut], "expression_cache": None}

    # 数据窗口适配：本机分钟数据（cn_data_1min）特征自 2024-01-02 起，
    # 官方默认窗口 2020-09 ~ 2021-01 超出覆盖，平移至 2024 年 3~7 月（csi300 标的池）
    MARKET = "csi300"

    start_time = "2024-03-01 00:00:00"
    end_time = "2024-07-31 16:00:00"
    train_end_time = "2024-06-30 16:00:00"
    test_start_time = "2024-07-01 00:00:00"

    DATA_HANDLER_CONFIG0 = {
        "start_time": start_time,
        "end_time": end_time,
        "fit_start_time": start_time,
        "fit_end_time": train_end_time,
        "instruments": MARKET,
        "infer_processors": [{"class": "HighFreqNorm", "module_path": "highfreq_processor"}],
    }
    DATA_HANDLER_CONFIG1 = {
        "start_time": start_time,
        "end_time": end_time,
        "instruments": MARKET,
    }

    task = {
        "dataset": {
            "class": "DatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": {
                    "class": "HighFreqHandler",
                    "module_path": "highfreq_handler",
                    "kwargs": DATA_HANDLER_CONFIG0,
                },
                "segments": {
                    "train": (start_time, train_end_time),
                    "test": (
                        test_start_time,
                        end_time,
                    ),
                },
            },
        },
        "dataset_backtest": {
            "class": "DatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": {
                    "class": "HighFreqBacktestHandler",
                    "module_path": "highfreq_handler",
                    "kwargs": DATA_HANDLER_CONFIG1,
                },
                "segments": {
                    "train": (start_time, train_end_time),
                    "test": (
                        test_start_time,
                        end_time,
                    ),
                },
            },
        },
    }

    def _init_qlib(self):
        """initialize qlib"""
        # use cn_data_1min data
        QLIB_INIT_CONFIG = {**HIGH_FREQ_CONFIG, **self.SPEC_CONF}
        provider_uri = QLIB_INIT_CONFIG.get("provider_uri")
        # NOTE: 数据已在位，跳过 GetData 下载（exists_qlib_data 对含指数特征的自建
        # 数据会误判而进入"下载+删除旧数据"分支；且官方 1min 包会覆盖本地分钟数据）
        qlib.init(**QLIB_INIT_CONFIG)

    def _prepare_calender_cache(self):
        """preload the calendar for cache"""

        # This code used the copy-on-write feature of Linux to avoid calculating the calendar multiple times in the subprocess
        # This code may accelerate, but may be not useful on Windows and Mac Os
        Cal.calendar(freq="1min")
        get_calendar_day(freq="1min")

    def get_data(self):
        """use dataset to get highreq data"""
        self._init_qlib()
        self._prepare_calender_cache()

        dataset = init_instance_by_config(self.task["dataset"])
        xtrain, xtest = dataset.prepare(["train", "test"])
        print(xtrain, xtest)

        dataset_backtest = init_instance_by_config(self.task["dataset_backtest"])
        backtest_train, backtest_test = dataset_backtest.prepare(["train", "test"])
        print(backtest_train, backtest_test)

        return

    def dump_and_load_dataset(self):
        """dump and load dataset state on disk"""
        self._init_qlib()
        self._prepare_calender_cache()
        dataset = init_instance_by_config(self.task["dataset"])
        dataset_backtest = init_instance_by_config(self.task["dataset_backtest"])

        ##=============dump dataset=============
        dataset.to_pickle(path="dataset.pkl")
        dataset_backtest.to_pickle(path="dataset_backtest.pkl")

        del dataset, dataset_backtest
        ##=============reload dataset=============
        with open("dataset.pkl", "rb") as file_dataset:
            dataset = pickle.load(file_dataset)

        with open("dataset_backtest.pkl", "rb") as file_dataset_backtest:
            dataset_backtest = pickle.load(file_dataset_backtest)

        self._prepare_calender_cache()
        ##=============reinit dataset=============
        dataset.config(
            handler_kwargs={
                "start_time": "2024-07-22 00:00:00",
                "end_time": "2024-07-26 16:00:00",
            },
            segments={
                "test": (
                    "2024-07-22 00:00:00",
                    "2024-07-26 16:00:00",
                ),
            },
        )
        dataset.setup_data(
            handler_kwargs={
                "init_type": DataHandlerLP.IT_LS,
            },
        )
        dataset_backtest.config(
            handler_kwargs={
                "start_time": "2024-07-22 00:00:00",
                "end_time": "2024-07-26 16:00:00",
            },
            segments={
                "test": (
                    "2024-07-22 00:00:00",
                    "2024-07-26 16:00:00",
                ),
            },
        )
        dataset_backtest.setup_data(handler_kwargs={})

        ##=============get data=============
        xtest = dataset.prepare("test")
        backtest_test = dataset_backtest.prepare("test")

        print(xtest, backtest_test)
        return


if __name__ == "__main__":
    fire.Fire(HighfreqWorkflow)
