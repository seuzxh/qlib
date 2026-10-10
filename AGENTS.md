# AGENTS.md

本文件是 AI 助手在本仓库工作时的规则。先读此文件，再动手。

## 1. 项目背景与目标

- 本仓库是微软 **qlib** 量化投资框架源码（远端 `origin = github.com/seuzxh/qlib`）。
- 当前核心目标不是二次开发 qlib，而是**由浅入深学习 qlib**，产出可追溯的
  学习方案与笔记，最终能独立完成因子研究、模型训练、滚动训练、策略回测全流程。
- 使用者背景：国新证券金融科技部，财富管理研发方向。

## 2. 目录边界（务必区分，不要放错位置）

| 路径 | 性质 | 处理规则 |
|------|------|----------|
| `qlib/`、`examples/`、`scripts/`、`tests/` | 上游源码与官方示例 | 只读引用；学习实验代码不得直接改动这些目录 |
| `docs/`（除 learning 外） | 上游 Sphinx 文档 | 不修改 |
| `docs/learning/` | 个人学习笔记 | 所有学习产出放这里 |
| `docs/learning/code/` | 动手实践脚本 | 命名 `NN_主题.py`，NN 与课次一致 |
| `.github/workflows/pages.yml` | Pages 发布 | 非必要不改 |

## 3. 学习内容组织规则

1. 课程总览与进度表只看 `docs/learning/README.md`；已规划 8 课：
   QuickStart → 数据层 → 工作流 → DataHandler → 模型 → 策略回测 →
   实验管理与分析 → 高级主题与实战。
2. 每课一个文件 `docs/learning/NN-主题.md`，统一六段结构：
   学习目标 / 正文 / 动手实践 / 自检清单 / 学习记录 / 下节预告。
3. 课程按进度逐课补充详细内容，不提前灌输后续课程。
4. 每完成一课，更新 README 进度表（日期、课次、完成情况、心得）。
5. 内容必须基于本仓库真实源码，引用代码用相对路径链接并标注关键行号，
   不杜撰 API、参数或运行结果。
6. 实践脚本可直接运行；涉及数据下载的，注明
   `~/.qlib/qlib_data/cn_data` 与 `GetData().qlib_data(...)`。

## 4. qlib 领域速记（避免讲错）

- 两种入口：`qrun xxx.yaml` 配置模式；代码积木模式（见
  `examples/workflow_by_code.py`）。
- 组件实例化统一走 `init_instance_by_config`，配置三要素：
  `class` / `module_path` / `kwargs`。
- 主链路：数据 `D` → DataHandler（Alpha158/Alpha360）→ DatasetH →
  Model → SignalRecord/SigAnaRecord/PortAnaRecord → 策略回测；
  实验用 `R`（MLflow）记录，产物在 `mlruns/`。
- 滚动训练用 `RollingGen`：`ROLL_EX`=扩张窗口，`ROLL_SD`=固定长度滑动；
  `step` 单位是交易日，valid 段随窗口同步滚动；`trunc_days` 防标签泄露。

## 5. 环境约束

- 沙箱未安装 qlib，系统 Python 3.14 过新；建议本地 conda 建
  Python 3.8~3.11 环境后 `pip install -e .`。
- 不在沙箱假定网络与凭据可用；涉及安装、下载、推送，先说明前置条件。

## 6. 提交与发布

- 推送目标：`main` 分支（`git push origin HEAD:main`）。
- 推送 main 会触发 GitHub Pages：Jekyll 构建 `docs/learning/`，
  站点 `https://seuzxh.github.io/qlib/`。
- 沙箱无持久凭据；需要推送时用设备授权流程，token 只用于当次操作，不落盘。
- 不主动创建提交，除非用户明确要求。

## 7. 输出形态与行文（所有回复与产出遵守）

1. 行文用受控中文：同一术语全文一个叫法，不换同义词；一句话只说
   一件事，单句不超过 30 字；主动句式，先说谁做再说做什么；
   不复述用户问题，不加客套收尾。
2. 结论先行：先给结论（不超过 3 行），再给依据，细节在后；
   用户喊停即停。
3. 按内容选形态，选最低够用级：结构/流程/调用关系画 Mermaid 图
   （节点 ≤10，箭头标关系名）；精确数值用表格；纯枚举用列表；
   一段话能说清的不用图，一张图能说清的不做网页。
4. 解释复杂机制（参数敏感性、阈值影响等）时，主动建议生成单文件、
   自包含的 HTML 交互页：首屏只放一张关键图和一句结论，
   参数可实时调节。
5. 用完即弃的产物（临时网页/图/视频）放临时目录，不进项目仓库；
   用户确认要复用或归档的，才整理进项目。
