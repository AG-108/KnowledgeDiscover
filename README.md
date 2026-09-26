# KnowledgeDiscover (KD)

用于可复现 symbolic regression、ODE discovery 和 PDE discovery 的 Python 框架。
统一数据加载、方法适配、独立子进程运行与评估，记录方法适用范围、实验协议、
失败原因和统计分母。

[文档索引](docs/README.md) · [运行说明](docs/benchmark.md) ·
[路线图](docs/benchmark_roadmap.md)

## 快速开始

使用 Python 3.10，从仓库根目录执行：

~~~bash
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows PowerShell 使用：.venv/Scripts/Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e ".[dev]"
python run_benchmark.py --list
python run_benchmark.py --config configs/benchmark/core_v2.json --dry-run
~~~

`requirements.txt` 提供核心依赖，不会安装所有外部 baseline。GPU PyTorch、Julia/PySR、
PyOperon、官方 E2E checkpoint 和 LLM-SR 服务需要各自配置，参见
[服务器环境](docs/server_setup.md)。Conda 用户可从根目录执行
`conda env create -f environment.yml`。

默认配置位于 `configs/benchmark/benchmark.json`，方法参数位于
`configs/benchmark/baselines/`，独立诊断配置位于 `configs/benchmark/tuning/`。
`--dry-run` 生成实验清单，不代表方法已经训练或验证。

## 数据与实验状态

仓库包含小型示例数据、SR 定义和可生成的 ODE/PDE 系统。大体积原始数据、
上游参考快照、checkpoint 和实验结果保留在本地，获取与放置方式见
[数据说明](docs/data.md)。新克隆中缺少这些数据的集成测试会明确跳过。

Core ODE 的正式 440-case 实验尚未完成；候选函数协议和 Robertson 噪声导数协议
仍待确认。当前 PIC 排序校准未通过验收，物理可信度评估是离线原型。
运行成功、兼容性预检和方程恢复质量分别报告。具体边界见
[方法卡片](docs/benchmark_v2_method_cards.md) 与 [指标协议](docs/benchmark_metrics.md)。

## 目录

| 路径 | 内容 |
| --- | --- |
| `kd/` | 数据、模型适配器、指标与可视化 |
| `configs/` | 可复现实验配置 |
| `examples/`、`quick_start.ipynb` | 使用示例 |
| `tests/` | 单元测试和集成测试 |
| `scripts/` | 运行、结果合并、审计与 profiling 工具 |
| `hpc/`、`docker/` | 服务器运行与环境构建配方 |
| `docs/` | 当前指南与协议说明 |
| `results/`、`.local/` | 本地结果、私有配置与备份；不提交 |

## 开发与验证

~~~bash
python -m pytest -q -rs
# 在完整数据工作站上要求所有外部数据：
python -m pytest -q --require-external-data
~~~

格式与检查规则见 [代码风格](docs/code_style.md)。本地服务器设置使用
`hpc/server.env.example` 对应的 `.local/server.env`；不要将连接凭据、原始数据、
模型权重或运行日志强制加入 Git。

## 致谢与许可证

项目代码许可证见 [LICENSE](LICENSE)。第三方实现和数据保留各自来源与条款；
方法变体和移植边界见方法卡片。项目参考了 DISCOVER、Deep Symbolic Optimization、
SymPy、DeepXDE、PySR、LLM-SR 与 PyOperon 等开源工作。
