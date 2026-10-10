# Qlib 学习路线（由浅入深）

> 个人学习档案：目标是在 8 次课程内系统掌握微软 qlib 量化投资框架，达到能独立做因子研究、模型训练、策略回测全流程的水平。
> 每课一个文件，学完后在下方进度表打勾并记录心得。

**在线阅读（GitHub Pages）**：https://seuzxh.github.io/qlib/

## 学习目标

1. 理解 qlib 的整体架构与核心设计思想（数据层 / 工作流 / 模型 / 回测 / 分析）
2. 熟练两种使用方式：`qrun xxx.yaml` 配置模式 与 代码积木模式
3. 掌握表达式引擎，能自定义因子
4. 掌握 DataHandler → Dataset → Model → Strategy → Backtest 全链路
5. 能解读实验结果（IC、回测报告），并用 Recorder 管理实验
6. 最终产出：一个自定义因子 + 自定义模型 + 回测分析的完整实战项目

## 学习方法（每课四步）

1. **读**：读课程笔记 + 对应源码（笔记中均有链接）
2. **跑**：运行课程中的示例代码
3. **改**：修改参数/代码，观察输出变化
4. **记**：在课程文件末尾的「学习记录」区写下心得与疑问

## 课程路线图

| 课次 | 主题 | 核心内容 | 状态 |
|------|------|----------|------|
| [01](01-quickstart.md) | 认识 qlib | 框架全景、环境搭建、跑通 QuickStart | ✅ 已就绪 |
| [02](04-data-handler.md#附录第-23-课大纲待补全) | 数据层 | 数据准备、二进制数据格式、表达式引擎、`D` 接口 | ⬜ 大纲 |
| [03](04-data-handler.md#附录第-23-课大纲待补全) | 工作流系统 | `qrun` 与 YAML 配置、`init_instance_by_config` 机制、代码模式 | ⬜ 大纲 |
| [04](04-data-handler.md) | DataHandler | Alpha158/Alpha360 因子库、自定义标签、Processor | ✅ 已就绪 |
| [05](05-model.md) | 模型层 | Model 接口、内置模型（GBDT/MLP/LSTM…）、自定义模型 | ⬜ 未开始 |
| [06](06-strategy-backtest.md) | 策略与回测 | TopkDropoutStrategy、Executor、Exchange、成本模型 | ⬜ 未开始 |
| [07](07-recorder-analysis.md) | 实验管理与分析 | Recorder/Experiment、SigAnaRecord、PortAnaRecord、报告解读 | ⬜ 未开始 |
| [08](08-advanced-project.md) | 高级主题与实战 | 滚动训练、在线服务、高频/RL 概览；自定义全流程实战 | ⬜ 未开始 |

> 课程文件会在学到对应阶段前逐一补充详细内容，保持「学到哪、写到哪」，避免一次性信息过载。

## 进度记录

| 日期 | 课次 | 完成情况 | 心得 / 疑问 |
|------|------|----------|-------------|
| 2026-09-14 | — | 建立学习路线 | 初始化 |
| | | | |

## 目录约定

```text
docs/learning/
├── README.md                  # 本文件：路线图 + 进度追踪
├── 01-quickstart.md           # 第 1 课
├── 02-data-layer.md           # 第 2 课
├── ...
└── code/                      # 每课动手实践的代码（按课次命名，如 01_xxx.py）
```

## 环境备忘

- 本仓库即 qlib 源码（`/workspace`），推荐源码安装：`pip install -e .`
- 沙箱当前状态：qlib **未安装**，系统 Python 为 3.14（偏新）；建议本地用 conda 建 Python 3.8~3.11 环境，详见第 1 课
- 示例数据下载目录：`~/.qlib/qlib_data/cn_data`
