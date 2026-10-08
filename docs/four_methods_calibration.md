# 四方法主实验前校准

用户在 2026-10-05 要求先用无噪声简单数据调整参数，再运行 PhySO、SISSO、SymbolicGPT、DLGA 的全量实验，并确认范围为 279 个数据集、5 个 seed、每 target 300 秒。

## 校准边界

此次明确授权修订 D3 的开发集用途：允许在独立合成开发数据上选择一套方法参数；正式测试响应与参考方程不得用于优化或参数选择。每方法、每任务仍只允许一套冻结参数，不按主实验数据集覆盖。所有试验及失败记录单独保留，校准结果不进入主实验表。

校准使用固定数据生成 seed 314159，算法 seed 0、1、2，全过程 fit 上限 300 秒，四核资源槽。SR 方法各验证 64 个训练点、128 个独立评分点上的仿射与二次函数；DLGA 验证 512 个随机时空观测点上的多项式行波输运。模型接收坐标与观测值，参考公式仅用于拟合后的评分；DLGA 神经网络优化不接收 PDE 残差、真实导数或参考系数。

门槛采用项目现有指标：`exact_recovery=1` 是忽略数值系数的加法项结构恢复；`coefficient_error` 是相对系数 L2 误差，须严格小于 `1e-5`。这与符号表达式完全相等的 `algebraic_exact_recovery` 不同；没有放宽恢复指标或修改阈值。

复现实验入口为 `scripts/calibrate_four_methods.py --suite <新的独立目录/suite.json>`。suite 是包含 method、fixture、seed、budget、parameters 的 JSON 数组，fixture 为 affine、quadratic 或 advection；可用 `cpu_affinity` 指定本机四个核。脚本针对 Linux 隔离 worker，拒绝覆盖已有试验目录。原始试验记录保存在服务器 `/opt/kd/evidence/four-methods-20261005`；公开结果使用相对路径。

## 实现与参数变化

- PhySO 使用 dimensionless config0，三个可训练常数、batch 128，更严格的常数 LBFGS 容差。CPU 表达式计算仍占用整个 GPU 资源槽与该槽四个核，GPU 保留成本照常记录。官方[性能说明](https://physo.readthedocs.io/en/latest/r_performances.html)建议对表达式与常数优化使用 CPU。导出时只在副本中将自由常数移至 CPU；对实数输入安全展开 log/exp 恒等式，`force=False`，不强制分支化简。
- SISSO 保留二维、两轮描述符搜索，最多 20 个 SIS 候选；仅用训练误差在原生返回的不同维度模型中选择达到容差的最低维模型。原生特征池不足时把候选数限制到实际容量，原请求、实际容量与选中维度均保留；同一 fit 预算内完成，终态结果不重跑。
- SymbolicGPT 保留逐 fit 合成预训练的小 Transformer 变体，64 维嵌入、2 层、2 个 head，1500 条合成语料、最多 40 epochs、2000 个候选。合成语料与预训练占当前 fit 剩余时间的最多 55%，其余供采样与常数拟合；这些成本全部计入 300 秒，不复用跨数据集权重。
- DLGA 保留观测神经网络、自动微分、遗传项搜索与原有系数求解。新增可选 float64、网络宽深、学习率衰减、非零偏置初始化及仅针对观测值的神经最小二乘精修；精修使用精确参数 Jacobian，并预留时间给遗传搜索。导出保留完整浮点系数；一阶时间导数是任务统一声明，未限制到参考 RHS 的项。

这些是项目 adapter 的显式实现变体，不声称与论文原始设置完全一致，也不保证在复杂、含噪或刚性主实验数据上恢复正确。

## 当前状态

独立校准 21/21 通过；所有方法均验证 seed 0、1、2。下表是校准结果，不是主实验结果。

| 方法 | 通过 | 最大相对系数误差 |
| --- | --- | --- |
| PhySO | 6/6 | 9.9963e-8 |
| SISSO | 6/6 | 6.3810e-16 |
| SymbolicGPT | 6/6 | 4.8269e-8 |
| DLGA | 3/3 | 8.3540e-6 |

冻结配置为 `configs/benchmark/four_methods_full.json`，证据为
`results/calibration/four_methods_20261005/gate.json`，包含全部试验 JSON 的 SHA256，
CPU 诊断与曾发生资源重叠的试验明确排除。PhySO 原始失败的 log/exp 导出保留，
修复实数恒等式导出后在相同参数、相同 seed 下另建试验验证，没有覆盖历史记录。

DLGA 冻结为 Sin 网络 1 层 × 16、float64、1000 Adam steps、学习率衰减 0.3、
偏置初始化尺度 1.0、最多 10000 次观测最小二乘评估、优化容差 1e-14；
精修占剩余 fit 时间最多 70%，遗传搜索 population 400 / generations 100、epi 1e-6。
主实验沿用 full 的 10% 训练观测采样规则，每 target 的总 fit 仍限制为 300 秒。

主矩阵准入审计通过：268 SR、8 Core ODE、3 PDE，共 4155 cases / 4290 declared fits。
用户确认容器仍有 48 小时上限，后续提供新容器，因此队列必须分块并保留续跑检查点。
可移植资源绑定使用 `run_slots.py --portable-resources`：实际 CPU 编号和 GPU UUID
在每次 worker 的资源记录中保留；身份比较要求四核分配、内存上限、CPU 型号、
GPU 型号/显存/驱动和代码/环境/资产保持一致。旧 slots-v1/v2 身份规则不变。
主队列使用独立源码快照，后续本地开发不会改变已冻结实验的实现。
最终完整 Linux 回归为 620 passed、0 failed、2 Windows-only skipped，385.11 秒；
两项 Windows 专属检查在本地补跑 2 passed。真实 target 分母核验并封存为
4155 cases / 4290 targets，273 个分块的主实验队列已在服务器启动。
该队列尚未全部完成，校准表不能作为主实验性能结论。

首批主实验出现真实恢复失败，精简记录见
`results/validation/four_methods_main_initial.json`：Nguyen-1/seed0 的 PhySO 超时，
SISSO 结构恢复为 0；SymbolicGPT 预测误差很小，但额外常数项使严格结构恢复为 0。
DLGA 的 burgers/seed0 系数误差约 0.266、支持恢复为 0。
仿射、二次与行波开发数据的准入通过不能推断所有无噪声主实验都恢复正确。
这些结果已保留；正式队列继续使用冻结配置，不从主实验结果反向选择参数。

2026-10-07 已在新容器恢复 1658 个首次终态并继续原队列；资源、环境和实现身份严格核验一致。
用户确认保留 PhySO/SymbolicGPT 原槽位和串行安排。新环境镜像未包含实验目录，恢复边界
为最后有效本地检查点；之后未保存的旧进度不可取回，用户同意按该边界续跑，迁移审计单独保留。
