# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
This example is about how can simulate the OnlineManager based on rolling tasks.
"""

from pprint import pprint
import fire
import qlib
from qlib.model.trainer import DelayTrainerR, DelayTrainerRM, TrainerR, TrainerRM
from qlib.workflow import R
from qlib.workflow.online.manager import OnlineManager
from qlib.workflow.online.strategy import RollingStrategy
from qlib.workflow.task.gen import RollingGen
from qlib.workflow.task.manage import TaskManager
from qlib.tests.config import CSI100_RECORD_LGB_TASK_CONFIG_ONLINE, CSI100_RECORD_XGBOOST_TASK_CONFIG_ONLINE
import pandas as pd
from qlib.contrib.evaluate import backtest_daily
from qlib.contrib.evaluate import risk_analysis
from qlib.contrib.strategy import TopkDropoutStrategy


class OnlineSimulationExample:
    def __init__(
        self,
        provider_uri="~/.qlib/qlib_data/cn_data",
        region="cn",
        exp_name="rolling_exp",
        task_url="mongodb://127.0.0.1:27017",  # NOTE: 原版 10.0.0.4 为官方内网地址，改本机
        task_db_name="rolling_db",  # not necessary when using TrainerR or DelayTrainerR
        task_pool="rolling_task",
        rolling_step=80,
        start_time="2024-07-01",
        end_time="2024-09-30",
        tasks=None,
        trainer="TrainerR",
    ):
        """
        Init OnlineManagerExample.

        Args:
            provider_uri (str, optional): the provider uri. Defaults to "~/.qlib/qlib_data/cn_data".
            region (str, optional): the stock region. Defaults to "cn".
            exp_name (str, optional): the experiment name. Defaults to "rolling_exp".
            task_url (str, optional): your MongoDB url. Defaults to "mongodb://10.0.0.4:27017/".
            task_db_name (str, optional): database name. Defaults to "rolling_db".
            task_pool (str, optional): the task pool name (a task pool is a collection in MongoDB). Defaults to "rolling_task".
            rolling_step (int, optional): the step for rolling. Defaults to 80.
            start_time (str, optional): the start time of simulating. Defaults to "2018-09-10".
            end_time (str, optional): the end time of simulating. Defaults to "2018-10-31".
            tasks (dict or list[dict]): a set of the task config waiting for rolling and training
        """
        if tasks is None:
            # 数据窗口适配：本机日线 2020-01-02 起、csi100 成分过期改 csi300
            #（官方默认 2018 年窗口 + CSI100）
            from copy import deepcopy

            tasks = [deepcopy(CSI100_RECORD_XGBOOST_TASK_CONFIG_ONLINE), deepcopy(CSI100_RECORD_LGB_TASK_CONFIG_ONLINE)]
            for _task in tasks:
                _task["dataset"]["kwargs"]["segments"] = {
                    "train": ("2024-01-01", "2024-03-31"),
                    "valid": ("2024-04-01", "2024-05-31"),
                    "test": ("2024-06-01", "2024-06-30"),
                }
                _task["dataset"]["kwargs"]["handler"]["kwargs"].update({
                    "start_time": "2024-01-01",
                    "end_time": "2024-09-30",
                    "fit_start_time": "2024-01-01",
                    "fit_end_time": "2024-03-31",
                    "instruments": "csi300",
                })
        self.exp_name = exp_name
        self.task_pool = task_pool
        self.start_time = start_time
        self.end_time = end_time
        mongo_conf = {
            "task_url": task_url,
            "task_db_name": task_db_name,
        }
        qlib.init(provider_uri=provider_uri, region=region, mongo=mongo_conf)
        self.rolling_gen = RollingGen(
            step=rolling_step, rtype=RollingGen.ROLL_SD, ds_extra_mod_func=None
        )  # The rolling tasks generator, ds_extra_mod_func is None because we just need to simulate to 2018-10-31 and needn't change the handler end time.
        if trainer == "TrainerRM":
            self.trainer = TrainerRM(self.exp_name, self.task_pool)
        elif trainer == "TrainerR":
            self.trainer = TrainerR(self.exp_name)
        else:
            # TODO: support all the trainers: TrainerR, TrainerRM, DelayTrainerR
            raise NotImplementedError(f"This type of input is not supported")
        self.rolling_online_manager = OnlineManager(
            RollingStrategy(exp_name, task_template=tasks, rolling_gen=self.rolling_gen),
            trainer=self.trainer,
            begin_time=self.start_time,
        )
        self.tasks = tasks

    # Reset all things to the first status, be careful to save important data
    def reset(self):
        if isinstance(self.trainer, TrainerRM):
            TaskManager(self.task_pool).remove()
        exp = R.get_exp(experiment_name=self.exp_name)
        for rid in exp.list_recorders():
            exp.delete_recorder(rid)

    # Run this to run all workflow automatically
    def main(self):
        print("========== reset ==========")
        self.reset()
        print("========== simulate ==========")
        self.rolling_online_manager.simulate(end_time=self.end_time)
        print("========== collect results ==========")
        print(self.rolling_online_manager.get_collector()())
        print("========== signals ==========")
        signals = self.rolling_online_manager.get_signals()
        print(signals)
        # Backtesting
        # - the code is based on this example https://qlib.readthedocs.io/en/latest/component/strategy.html
        CSI300_BENCH = "SH000300"  # NOTE: 本机无中证 100 指数(000903)数据，改用沪深 300
        STRATEGY_CONFIG = {
            "topk": 30,
            "n_drop": 3,
            "signal": signals.to_frame("score"),
        }
        strategy_obj = TopkDropoutStrategy(**STRATEGY_CONFIG)
        report_normal, positions_normal = backtest_daily(
            start_time=signals.index.get_level_values("datetime").min(),
            end_time=signals.index.get_level_values("datetime").max(),
            strategy=strategy_obj,
        )
        analysis = dict()
        analysis["excess_return_without_cost"] = risk_analysis(report_normal["return"] - report_normal["bench"])
        analysis["excess_return_with_cost"] = risk_analysis(
            report_normal["return"] - report_normal["bench"] - report_normal["cost"]
        )

        analysis_df = pd.concat(analysis)  # type: pd.DataFrame
        pprint(analysis_df)

    def worker(self):
        # train tasks by other progress or machines for multiprocessing
        # FIXME: only can call after finishing simulation when using DelayTrainerRM, or there will be some exception.
        print("========== worker ==========")
        if isinstance(self.trainer, TrainerRM):
            self.trainer.worker()
        else:
            print(f"{type(self.trainer)} is not supported for worker.")


if __name__ == "__main__":
    ## to run all workflow automatically with your own parameters, use the command below
    # python online_management_simulate.py main --experiment_name="your_exp_name" --rolling_step=60
    fire.Fire(OnlineSimulationExample)
