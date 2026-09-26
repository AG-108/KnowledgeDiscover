# Baseline 与 Dataset 兼容性报告

检查日期：2026-09-15；`weakform` 的 23 个 PDE 组合于 2026-09-22 在替换为
WSINDy-PDE 后重新检查（13 兼容、10 明确不兼容、0 错误）。本报告仍采用原 18-baseline
快照范围，不包含后来加入的 E2E Transformer；不要把总数当作当前全部入口的实时清单。

本次检查使用 `python run_benchmark.py --check-compatibility`，覆盖 benchmark 生成的全部
2,455 个 baseline/dataset 组合。检查内容包括：加载一个最小 dataset 实例、生成任务适配层
输入、导入 baseline 运行时，以及执行 benchmark 已声明的维度、网格和时间切分约束。
该检查不会训练模型，因此“兼容”表示组合能够进入训练阶段，不表示模型一定收敛或恢复正确方程。

## 汇总

| 任务 | Dataset 工作负载 | Baseline | 组合数 | 兼容 | 明确不兼容 | 错误 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Symbolic regression | 274 | 8 | 2,192 | 1,918 | 0 | 274 |
| ODE discovery | 1 | 10 | 10 | 9 | 0 | 1 |
| PDE discovery | 23 | 11 | 253 | 121 | 132 | 0 |
| **总计** | **298** | **18** | **2,455** | **2,048** | **132** | **275** |

错误的 275 个组合全部来自当前 Windows/Python 3.9 环境无法安装 `pyoperon`。官方
PyOperon 0.6.1 要求 Python 3.10 或更高版本；Linux/Python 3.9 可安装兼容版 0.5.0。在服务器
安装对应版本后，按更新后的 WSINDy 兼容范围预计汇总为 2,323 个兼容、132 个明确不兼容、
0 个导入错误。`llmsr` 的
275 个组合已通过数据和适配器预检；实际搜索仍需配置 completion 服务，预检不会发送 LLM
请求。

2026-09-17 的 Linux/Python 3.9 全量复核发生在 WSINDy 替换之前；当时使用 PyOperon 0.5.0
和 PySR 1.5.10 得到 2,311 个兼容、144 个明确不兼容、0 个导入错误。2026-09-22 只重跑了
受本次改动影响的 23 个 `weakform` 组合，结果保存在
`results/wsindy_compatibility_20260922/`，并据此更新本页的 PDE 数字。

Catalog 中共有 299 个 dataset 入口。`rubber_test` 是 `rubber_train` 的官方固定测试集，
会在加载 `rubber_train` 时一起检查，因此不会单独生成训练工作负载。另有 10 个已明确排除的
源数据入口：1 个底层文件结构尚未解析的 `advection_diffusion`，以及 9 个没有时间轴的 TLC
稳态文件。

## CYT、VGS 与固体本构数据

这三组领域数据已经包含在上述总数和逐组合结果中：

| 数据集组 | 已接入入口 | 最小实例检查 |
| --- | ---: | --- |
| CYT | `cyt_flatplate`、`cyt_naca0012` | 每个入口 7 个兼容；PyOperon 等待服务器依赖 |
| 固体本构 | `solid_dif`、`solid_strain_stress`、`solid_hardening` | 每个入口 7 个兼容；PyOperon 等待服务器依赖 |
| VGS | 2 个物理 case、每个 case 2 个时间窗 | 每个入口兼容 9/11 个 PDE baseline |

VGS 可以进入 DSCV、SPR、DLGA、DeepMoD、PDE-FIND、WSINDy-PDE (`weakform`)、EqGPT、
SINDy 和 PySINDy。SGA 缺少对应的方程预设；PDE-Net 当前适配器只接受二维空间网格。

## Baseline 覆盖

| Baseline | 兼容/检查组合 | 说明 |
| --- | ---: | --- |
| `dscv` | 288/298 | SR、ODE 全部兼容；PDE 需要一维标量规则网格。 |
| `spr` | 13/23 | 需要一维标量规则网格。 |
| `sga` | 3/23 | 当前仅有 Burgers、KdV、Chafee–Infante 三个 SolverConfig 预设。 |
| `dlga` | 13/23 | 需要一维标量规则网格。 |
| `deepmod` | 13/23 | 需要一维标量规则网格。 |
| `pdefind` | 13/23 | 需要一维标量规则网格。 |
| `pdenet` | 1/23 | 当前 example adapter 只接受二维规则网格。 |
| `weakform` | 13/23 | 标量一维 WSINDy-PDE；要求规则、等距的空间与时间网格。 |
| `eqgpt` | 13/23 | 需要一维标量规则网格。 |
| `dso` | 275/275 | 274 个 SR 工作负载和 1 个 ODE 工作负载均兼容。 |
| `symbolicgpt` | 275/275 | 274 个 SR 工作负载和 1 个 ODE 工作负载均兼容。 |
| `gplearn` | 275/275 | 274 个 SR 工作负载和 1 个 ODE 工作负载均兼容。 |
| `physo` | 275/275 | 274 个 SR 工作负载和 1 个 ODE 工作负载均兼容。 |
| `pysr` | 275/275 | 274 个 SR 工作负载和 1 个 ODE 工作负载均兼容；需要 Julia 运行时。 |
| `sindy` | 14/24 | ODE 兼容；PDE 需要一维标量规则网格。 |
| `pysindy` | 14/24 | ODE 兼容；PDE 需要一维标量规则网格。 |
| `llmsr` | 275/275 | SR 与 ODE 数据适配兼容；训练时需要本地或 API completion 服务。 |
| `pyoperon` | 0/275 | 当前 Windows/Python 3.9 无可用官方 wheel；Linux/Python 3.9 可用 0.5.0，Python 3.10+ 可用 0.6.1。 |

## PDE 兼容矩阵

`Y` 表示能够进入训练阶段，`N` 表示被明确的适配器约束拒绝。

| Dataset | dscv | spr | sga | dlga | deepmod | pdefind | pdenet | weakform | eqgpt | sindy | pysindy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PDE_compound | Y | Y | N | Y | Y | Y | N | Y | Y | Y | Y |
| PDE_divide | Y | Y | N | Y | Y | Y | N | Y | Y | Y | Y |
| burgers | Y | Y | Y | Y | Y | Y | N | Y | Y | Y | Y |
| chafee-infante | Y | Y | Y | Y | Y | Y | N | Y | Y | Y | Y |
| fisher | Y | Y | N | Y | Y | Y | N | Y | Y | Y | Y |
| fisher_linear | Y | Y | N | Y | Y | Y | N | Y | Y | Y | Y |
| kdv | Y | Y | Y | Y | Y | Y | N | Y | Y | Y | Y |
| pdeformer_sinus | Y | Y | N | Y | Y | Y | N | Y | Y | Y | Y |
| tlc_burgers1d | N | N | N | N | N | N | N | N | N | N | N |
| tlc_burgers2d | N | N | N | N | N | N | N | N | N | N | N |
| tlc_heat_complex | N | N | N | N | N | N | N | N | N | N | N |
| tlc_heat_darcy | N | N | N | N | N | N | N | N | N | N | N |
| tlc_heat_longtime | N | N | N | N | N | N | N | N | N | N | N |
| tlc_heat_multiscale | N | N | N | N | N | N | N | N | N | N | N |
| tlc_heat_multiscale_lesspoints | N | N | N | N | N | N | N | N | N | N | N |
| tlc_ns_long | N | N | N | N | N | N | N | N | N | N | N |
| tlc_wave_darcy | N | N | N | N | N | N | N | N | N | N | N |
| vgs_II_0-1000 | Y | Y | N | Y | Y | Y | N | Y | Y | Y | Y |
| vgs_II_1000-2000 | Y | Y | N | Y | Y | Y | N | Y | Y | Y | Y |
| vgs_I_0-100 | Y | Y | N | Y | Y | Y | N | Y | Y | Y | Y |
| vgs_I_100-200 | Y | Y | N | Y | Y | Y | N | Y | Y | Y | Y |
| wave_breaking | Y | Y | N | Y | Y | Y | N | Y | Y | Y | Y |
| wdwake | N | N | N | N | N | N | Y | N | N | N | N |

132 个不兼容组合均有预期原因：99 个来自 TLC 散点数据尚无规则网格适配器；13 个来自
PDE-Net 与一维数据之间的维度不匹配；10 个来自 SGA 缺少对应方程预设；10 个来自
`wdwake` 二维双通道场与一维标量 baseline（现在包括 WSINDy）之间的输入不匹配。没有发现
未分类的运行错误。

逐组合结果见 [`compatibility_pairs.csv`](compatibility_pairs.csv)。
