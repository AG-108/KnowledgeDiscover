# FDE 原始数据整理

2026-10-08 从用户提供的 `kd/dataset/FDE.zip` 提取。压缩包中的 `dataset/`
外层目录已去除，文件名和文件内容保持原样：

```text
kd/dataset/
  FDE.zip
  FDE/
    data/                       # 7 个 MAT、3 个 DAT
    reference/fade_plot_3d.py    # 原始参考脚本
    manifest.json               # 来源、逐文件校验、数组结构和已发现的问题
```

原压缩包保留。`FDE/` 按项目外部数据约定排除在 Git 外；本说明保留在项目文档中。
压缩包 SHA256 为
`cce3cd979a009b50bcfcf075af261c634104a96cd0026f14ad4e312475420bc5`。
11 个提取文件均与压缩包成员逐字节一致；10 个数据文件的数值均可读取且全部有限。

## 文件与数组结构

下表的文件路径相对于 `kd/dataset/FDE/data/`。MAT 字段保留原始轴顺序；
DAT 是无表头、空白分隔的数值矩阵。

| 文件 | 数据字段 / 形状 | 坐标与解释 |
| --- | --- | --- |
| `burgers.mat` | `usol (256, 201)`；轴为 `(x, t)` | 内置 `x (1, 256)`、`t (1, 201)`；与已有 Burgers 数据完全相同 |
| `Convection_diffusion.dat` | 矩阵 `(150, 120)` | 未内置坐标，当前无法确认轴顺序、步长或分数阶阶数 |
| `creep.mat` | `ex (56, 1)`、`t (56, 1)` | 56 个观测；时间严格递增且不等间隔，`ex` 的物理含义与单位未附说明 |
| `Ground_water_flow.dat` | 矩阵 `(100, 101)` | 未内置坐标，当前无法确认轴顺序、步长或分数阶阶数 |
| `KdV.mat` | `uu (512, 201)`；轴为 `(x, t)` | 内置 `x (1, 512)`、`tt (1, 201)`；与已有 KdV 数据完全相同 |
| `pt_fade.mat` | `un (15, 100)`；轴为 `(t, x)` | `t_det = 1, …, 15`；`x_lin = 0.15, 0.45, …, 29.85` |
| `pt_fde.mat` | `un (15, 100)`；轴为 `(t, x)` | `t_det = 1, …, 15`；`x_lin = 0.075, 0.225, …, 14.925` |
| `silt.mat` | `ex (43, 1)`、`t (43, 1)` | 43 个观测；不等间隔，前两行时间相同、观测不同，保留两行 |
| `tsfade_fft.dat` | 矩阵 `(150, 120)`；轴为 `(t, x)` | 与同名 MAT 的 `Exact` 对应，仅文本精度不同 |
| `tsfade_fft.mat` | `Exact (150, 120)`；轴为 `(t, x)` | 内置 `t (1, 150)`、`x (1, 120)`；坐标包含右端点 |

## 同源数据与坐标差异

`burgers.mat` 与 `kd/dataset/data/burgers.mat`、`Burgers_equation.mat`，以及
`kd/dataset/EqGPT/data/Burgers_equation/burgers_sine.mat` 逐字节相同。
`KdV.mat` 与 `kd/dataset/data/Kdv.mat`、`KdV_equation.mat`，以及
`kd/dataset/EqGPT/data/KdV_equation/KdV-PINN.mat` 逐字节相同。
这些文件保留在来源包的整理目录中，后续库存统计沿用既有实例身份。

`tsfade_fft.dat` 与 `tsfade_fft.mat` 的 `Exact` 最大绝对差为
`4.9996997208978655e-08`，对应同一份场观测的两种存储格式。
MAT 内置坐标与参考脚本初次读取原始数据时采用的坐标存在差异：

| 坐标 | MAT 内置值 | 参考脚本读取 DAT 时的设置 |
| --- | --- | --- |
| `x`，120 点 | 0 至 30，步长 `30/119 ≈ 0.25210084` | 0 至 29.75，步长 0.25 |
| `t`，150 点 | 0 至 15，步长 `15/149 ≈ 0.10067114` | 0 至 14.9，步长 0.1 |

读取 MAT 时应保留文件内置坐标；读取 DAT 时可以依据两格式的对应关系显式采用
MAT 坐标，并披露上述冲突。两者都保留，当前没有重采样、修正坐标或生成导数。
其他 DAT 文件的坐标仍待确认，不能直接套用这一网格。

`silt.mat` 的前两行均为 `t = 0.02447`，对应 `ex = 0.91021` 和 `1.15183`。
原始行顺序和两条观测保留，后续时间导数或分数阶算子计算需要先明确处理规则。

## 参考代码与接入状态

`reference/fade_plot_3d.py` 展示了读取 `tsfade_fft.dat`、加入相对均匀噪声、
构建神经网络和计算 Caputo 型分数阶导数所用积分的代码。
脚本设置 `noise_level = 25`、`alpha = 0.99`、`beta = 1.5`、积分点数 `n = 5`；
这些是脚本配置，压缩包没有证据确认它们就是各数据文件的生成参数。
噪声在脚本读取数据后添加，本次提取未加噪声。

脚本依赖未随包提供的 `STRidge` 模块和
`model_save/draft-2000-25/draft-20000.pkl` 权重，当前仅保留作参考，未执行。
包中未提供来源论文、仓库版本、数据许可、单位及完整参考方程。

## Benchmark 响应拟合接口

已通过 `kd.dataset.load_dataset()` 接入 7 个 `fde_*` 条目，数值调度为 `sr`，
科学任务为 `fractional_equation_response`，协议为 `fde-response-v1` / 指标 2.3。
用户选择当前评估观测响应拟合；分数阶算子、阶数及方程发现参考仍待补充来源。

| 注册名 | 预测目标 | 当前可用性 |
| --- | --- | --- |
| `fde_creep` | `ex=f(t)` | 可直接运行 |
| `fde_silt` | `ex=f(t)` | 可直接运行，重复时间对应的两行保留 |
| `fde_pt_fade` | `u=f(x,t)` | 可直接运行，保留 MAT 内置坐标 |
| `fde_pt_fde` | `u=f(x,t)` | 可直接运行，保留 MAT 内置坐标 |
| `fde_tsfade_fft` | `u=f(x,t)` | 可直接运行，采用 MAT 而非绘图脚本网格 |
| `fde_convection_diffusion` | `u=f(x,t)` | 需提供坐标及其来源，缺省记录 `source_blocked` |
| `fde_ground_water_flow` | `u=f(x,t)` | 需提供坐标及其来源，缺省记录 `source_blocked` |

加载示例：

```python
from kd.dataset import load_dataset

dataset = load_dataset("fde_tsfade_fft")
print(dataset.X.shape, dataset.y.shape)  # (18000, 2), (18000,)
print(dataset.variable_names)           # ["x", "t"]
train, test, metadata = dataset.temporal_holdout(test_size=0.2)
```

两个无坐标 DAT 的加载选项为 `coordinates={"x": ..., "t": ..., "layout": "tx",
"evidence": "坐标来源说明"}`；`layout` 可显式设为 `xt`。
向量必须与原矩阵匹配、有限且严格递增；通过配置时放在
`overrides.datasets.<注册名>.coordinates`。已有 MAT 的内置坐标不可覆盖。

### 划分与评分

按排序后的唯一时间值划分，训练使用最早的 `floor(n_times * (1-test_size))`
个时间，测试使用余下的较晚时间；两侧各保留至少一个时间值。同一时刻的全部空间点
和重复观测放在同一侧。划分仅依赖坐标，不依赖响应值或随机种子。
`max_train_samples` 只在训练块内按 seed 无放回采样；完整测试块保留。
这是同一记录内的时间外推评估，不提供跨记录独立泛化结论。

输入、目标均保留原始数值，不添加噪声、不估计导数、不进行拟合式归一化。
复用 SR 的 MSE、RMSE、R²、有限预测覆盖率；`test_nrmse` 的分母是测试响应标准差。
没有参考公式，`ground_truth` 为 null，所有结构恢复指标不适用，
`recovery_eligible=false`。结果包含 `metric_target=observed_response` 与
`fde_response_protocol_version`，并按科学任务及指标版本分组。

缺失原始文件记录 `error / missing_asset`，缺少 DAT 坐标记录
`skipped / source_blocked`；两者都保留已声明的 `u` 或 `ex` 目标及分母。
无效数值、缺失 MAT 字段及坐标/形状错误直接报错，不自动修补。

### 配置与运行

[独立配置](../configs/benchmark/fde_response.json) 选择 5 个可直接运行的记录和全部
10 个 SR 适配器，共 50 个单 seed 标量拟合。配置使用 smoke 参数，每目标拟合上限
90 秒；模型依赖与外部权重的可用性沿用各适配器的检查。

```bash
# 生成清单，不拟合
python run_benchmark.py --config configs/benchmark/fde_response.json --dry-run

# 使用 gplearn 检查数据与接口
python run_benchmark.py --config configs/benchmark/fde_response.json --models gplearn --check-compatibility --output-dir results/fde_response_check

# 对五个记录做有界 smoke 拟合
python run_benchmark.py --config configs/benchmark/fde_response.json --models gplearn --output-dir results/fde_response_smoke
```

FDE 条目为 opt-in，默认 `datasets=["*"]` 的既有矩阵不扩展。
显式选择 `--tasks sr --datasets "fde_*"` 可审计全部 7 个响应注册名，包括两个坐标待补齐的条目。
Burgers、KdV 和 tsfade 文本副本只进入重复资产清单，不产生独立拟合。
本次接入位于开发树，服务器当前冻结实验使用原源码、原配置和原分母。

相关回归 281 项通过；gplearn 对五个可用记录的真实 smoke 全部正常返回。
测试范围、误差解释与证据路径见 [验收记录](validation.md#fde-响应拟合接入2026-10-08)。

## 全部 PDE 方法覆盖

按用户补充要求，为 5 个时空场增加后缀 `_pde` 的网格接口。其中
`fde_pt_fade_pde`、`fde_pt_fde_pde`、`fde_tsfade_fft_pde` 可直接运行；
`fde_convection_diffusion_pde`、`fde_ground_water_flow_pde` 仍需显式坐标及其来源。
这些接口使用响应拟合中的同一份观测和物理家族身份，原始 `(t,x)` 数据只转置为
PDE 接口的 `(x,t)`，不插值、不增加空间轴。`creep`、`silt` 只有时间曲线，继续用于响应拟合。

[PDE 配置](../configs/benchmark/fde_pde.json) 选择 3 个可用场与全部 13 个 PDE 方法，
单 seed 共 39 cases。方法包括 `pdefind_nd`、`implicit_pde`、`dscv`、`spr`、`sga`、
`dlga`、`deepmod`、`pdefind`、`pdenet`、`weakform`、`eqgpt`、`sindy`、`pysindy`。
实际兼容性检查为 36 compatible、3 incompatible：现有 PDE-Net 适配器要求二维空间，
这 3 个场只有一维空间，其目标保留为 `skipped / method_unsupported`。

当前科学任务标记为 `fractional_field_integer_pde_diagnostic`，协议
`fde-pde-diagnostic-v1`。现有方法拟合整数阶演化 RHS；`implicit_pde` 在同一组
整数阶导数上搜索隐式关系，保留单位范数归一化和歧义诊断，隐式关系不保证能解为演化 RHS。
评分目标为 `observed_equation_residual`，`fractional_operator_evaluated=false`。
分数阶算子、阶数和参考方程尚未认证，结构恢复均不适用。

所有 PDE 方法使用相同划分：最后 `max(3, ceil(n_t * test_size))` 帧为测试块，
前面紧邻的两帧隔离，其余较早帧训练，训练至少需要 5 帧。
15 帧的两份记录在默认 20% 留出时采用 10 帧训练、2 帧隔离、3 帧测试。
先划分观测，再在各块内部估计导数，不共享初始帧，也不从测试数据选择候选方程。
无验证边界条件，场预测与 rollout 为 `abstain`。

FDE 的传统 PDE 残差评分覆盖完整测试块，包括首帧；支持观测坐标 `x` / `x1`、`t`
出现在 RHS 中。SGA 在拟合预算内导出原生 STRidge 系数、默认候选项及生成树的空间导数，
原始树形展示文本另外保留，避免把无系数的展示文本当作最终方程。
共同评分采用严格符号运算；含除零等非有限候选保留 `nonfinite_candidate` 和零覆盖率，
不授予误差分数。无法导出或解析的候选保留相应 `equation_residual_status`。

评分沿用各 PDE 适配器的既有协议：传统方法为 2.1，`pdefind_nd` / `implicit_pde`
为 2.2，分别保留其残差归一化定义与有限覆盖率。报告按科学任务、指标版本分组，
不能把残差分数与响应拟合 RMSE 混合排名。两份接口也不作为新增独立数据记录计权。

[合并配置](../configs/benchmark/fde_all_methods.json) 包含 50 个响应拟合 cases
和 39 个 PDE cases，共 89 cases，使用 smoke 参数、每目标拟合上限 90 秒。
使用方式：

```bash
# 审计响应与全部 PDE 方法的组合
python run_benchmark.py --config configs/benchmark/fde_all_methods.json --dry-run

# 检查全部 PDE 方法的原生入口与数据适用性
python run_benchmark.py --config configs/benchmark/fde_pde.json --check-compatibility

# 有界执行 PDE 诊断
python run_benchmark.py --config configs/benchmark/fde_pde.json
```

两个新增配置与原响应配置均为 opt-in，默认矩阵和服务器冻结实验沿用原范围。

最终相关回归 362 项通过。39 组合的原生入口短预算 smoke 为 33 正常返回、
3 PDE-Net 不适用、3 EqGPT 超时；评分修订后的 DSCV / SGA / PDE-FIND 共 9 cases
全部返回，其中 8 个残差有限、1 个 SGA 候选非有限。EqGPT 单独使用默认 90 秒拟合预算
仍在本地 CPU 超时，不能将其称为已完成有效拟合。证据与边界见
[PDE 验收记录](validation.md#fde-全部-pde-方法覆盖2026-10-08)。
