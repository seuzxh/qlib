# 第 1 课：认识 qlib —— 框架全景 + 环境搭建 + 跑通 QuickStart

## 学习目标

1. 说清楚 qlib 是什么、解决什么问题、整体架构分哪几层
2. 搭好可运行的开发环境（本地 conda 环境 / 源码安装）
3. 下载示例数据并跑通官方 `workflow_by_code.py`，理解每一段代码在做什么

## 一、qlib 是什么

qlib 是微软开源的 **AI 量化投资平台**，覆盖量化机器学习的完整链路：

```text
数据(Data) → 特征(Handler) → 模型(Model) → 信号(Signal) → 策略(Strategy) → 回测(Backtest) → 分析(Report)
                ↑____________________ 实验管理 Recorder 全程跟踪 ____________________↑
```

与传统回测框架（如 backtrader、聚宽）最大的区别：**qlib 以「AI 模型」为中心**，把「预测 → 组合 → 交易 → 评估」做成标准化流水线，强调实验可复现、可对比。

## 二、框架全景：源码目录速览

结合本仓库（`/workspace` 即 qlib 源码），核心模块与目录的映射：

| 层 | 目录 | 职责 |
|----|------|------|
| 数据层 | [qlib/data/](../../qlib/data/) | 数据存储（二进制 bin）、表达式引擎、数据 Provider（`D` 接口） |
| 数据集/特征 | [qlib/data/dataset/](../../qlib/data/dataset/) + [qlib/contrib/data/](../../qlib/contrib/data/) | DataHandler、Processor、内置 Alpha158/Alpha360 因子库 |
| 模型层 | [qlib/model/](../../qlib/model/) + [qlib/contrib/model/](../../qlib/contrib/model/) | Model 基类 + 20+ 内置模型（LightGBM、LSTM、Transformer…） |
| 策略层 | [qlib/strategy/](../../qlib/strategy/) + [qlib/contrib/strategy/](../../qlib/contrib/strategy/) | 策略基类、TopkDropout 等信号策略 |
| 回测层 | [qlib/backtest/](../../qlib/backtest/) | Exchange（撮合）、Executor（执行器）、账户与持仓 |
| 实验管理 | [qlib/workflow/](../../qlib/workflow/) | Recorder / Experiment（基于 MLflow）、Record 模板 |
| 结果分析 | [qlib/contrib/report/](../../qlib/contrib/report/) | IC 分析、收益分析、风险分析图表 |
| 高频/强化学习 | [qlib/rl/](../../qlib/rl/) | RL 订单执行（进阶，后期再学） |
| 示例 | [examples/](../../examples/) | benchmarks 下每个模型一个可运行 YAML 配置 |

**两种使用方式**（第一课只认脸熟，第三课深入）：

1. **配置模式**：`qrun examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml`
2. **代码模式**：像搭积木一样写 Python，见 [examples/workflow_by_code.py](../../examples/workflow_by_code.py)

## 三、环境搭建

### 3.1 推荐方式：conda + 源码安装（本仓库已是源码）

```bash
# qlib 官方文档建议 Python 3.8；社区实测 3.9~3.11 均可。不要用 3.14（过新，Cython 依赖可能编译失败）
conda create -n qlib python=3.10 -y
conda activate qlib

cd /workspace
pip install numpy cython
pip install -e .          # 源码可编辑安装
pip install lightgbm      # 本课示例模型依赖；torch 后续深度学习模型再装
```

验证：

```bash
python -c "import qlib; print(qlib.__version__)"
```

> 沙箱备忘：当前沙箱未安装 qlib 且 Python 为 3.14，建议在你自己的开发机上按上面操作；如需我在沙箱里尝试安装，随时说。

### 3.2 下载示例数据（A 股日频，约几百 MB）

```bash
python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn
```

或在 Python 里（workflow_by_code.py 用的就是这种）：

```python
from qlib.tests.data import GetData
from qlib.constant import REG_CN
GetData().qlib_data(target_dir="~/.qlib/qlib_data/cn_data", region=REG_CN, exists_skip=True)
```

数据落盘结构（第 2 课详解）：`~/.qlib/qlib_data/cn_data/{calendars,instruments,features}/`

## 四、跑通 QuickStart：逐段精读 workflow_by_code.py

运行：

```bash
python examples/workflow_by_code.py
```

对照 [examples/workflow_by_code.py](../../examples/workflow_by_code.py) 逐段理解（这是全课最重要的一段代码）：

**① 初始化**（第 21-23 行）

```python
provider_uri = "~/.qlib/qlib_data/cn_data"
GetData().qlib_data(target_dir=provider_uri, region=REG_CN, exists_skip=True)
qlib.init(provider_uri=provider_uri, region=REG_CN)
```

- `qlib.init` 是所有 qlib 程序的入口，告诉 qlib 数据在哪、市场区域（cn 会做涨跌停处理等本地化配置），见 [qlib/config.py](../../qlib/config.py)

**② 按配置实例化模型与数据集**（第 25-26 行）

```python
model = init_instance_by_config(CSI300_GBDT_TASK["model"])
dataset = init_instance_by_config(CSI300_GBDT_TASK["dataset"])
```

- `init_instance_by_config` 是 qlib 的灵魂函数：**一个 dict（class + module_path + kwargs）就能实例化任意组件**，见 [qlib/utils/mod.py](../../qlib/utils/mod.py)
- `CSI300_GBDT_TASK` 定义在 [qlib/tests/config.py](../../qlib/tests/config.py)：模型是 `LGBModel`，数据集是 `DatasetH + Alpha158 handler`，沪深 300 股票池，train/valid/test = 2008-2014 / 2015-2016 / 2017-2020.08

**③ 数据集可独立使用**（第 64 行）

```python
example_df = dataset.prepare("train")
```

- `dataset.prepare("train")` 返回训练集 DataFrame（多级索引 instrument+datetime），先跑一下看看因子长什么样

**④ 实验管理 + 训练 + 记录**（第 68-85 行）

```python
with R.start(experiment_name="workflow"):
    R.log_params(...)          # 记录参数
    model.fit(dataset)         # 训练
    R.save_objects(...)        # 保存模型
    SignalRecord(model, dataset, recorder).generate()   # 生成预测信号 pred.pkl
    SigAnaRecord(recorder).generate()                   # 信号分析：IC/RankIC
    PortAnaRecord(recorder, port_analysis_config, "day").generate()  # 回测+组合分析
```

- `R` 是 qlib 的实验管理器（底层 MLflow），见 [qlib/workflow/](../../qlib/workflow/)
- 三个 Record 模板对应「预测 → 信号评估 → 组合回测」三段，定义在 [qlib/workflow/record_temp.py](../../qlib/workflow/record_temp.py)
- 回测配置里的策略是 `TopkDropoutStrategy`（持有打分前 50 名、每天换仓 5 只），回测区间 2017-01~2020-08

**运行产出**：当前目录生成 `mlruns/`，里面有参数、指标（IC、年化收益、最大回撤…）、`pred.pkl` 等文件。

## 五、动手实践（必做）

把代码保存到 `docs/learning/code/01_task.py` 或直接改 example：

1. 跑通 `python examples/workflow_by_code.py`，确认看到训练日志和 IC 输出
2. 修改 `TopkDropoutStrategy` 的 `topk=30, n_drop=3`，重跑，对比指标变化
3. 在 `model.fit` 之前打印 `dataset.prepare("train").shape`，回答：Alpha158 有多少个因子？
4. 用 `R.get_recorder().load_object("pred.pkl")`（或直接在 mlruns 里找）查看预测结果格式

## 六、自检清单

- [ ] 能画出 qlib 的七层架构并说出每层职责
- [ ] 能解释 `init_instance_by_config` 的 dict 配置三要素（class/module_path/kwargs）
- [ ] 能解释 `workflow_by_code.py` 每一段的含义
- [ ] 能在 `mlruns/` 里找到 IC 指标和 pred.pkl
- [ ] 完成动手实践 1-4

## 学习记录

> 学完后在此记录心得与疑问，并更新 [README.md](README.md) 进度表。

（待填写）

## 下节预告

第 2 课「数据层」：qlib 的二进制数据格式、表达式引擎（`$close`、`Ref`、`Mean`…）、`D.features` 的用法与缓存机制 —— 这是后续自定义因子的地基。
