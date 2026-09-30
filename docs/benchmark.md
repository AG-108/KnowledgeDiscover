# 统一 benchmark

当前工程验收见 [验收摘要](validation.md)，来源条件见 [待解决问题](deferred_source_issues.md)。数据 domain-v4；新增轨指标 2.3。

`run_benchmark.py` 统一组织 symbolic regression（`sr`）、ODE discovery（`ode`）和 PDE
discovery（`pde`）任务。数据清单来自 `kd.dataset.DATASET_REGISTRY`、
`kd/dataset/data/benchmarks.csv` 和 WaveBreaking 适配器；模型按需导入，每个
model/dataset/seed 组合（case）先枚举 instance/target，再为每个 target 启动独立子进程与工作目录。

指标定义见 [指标协议](benchmark_metrics.md)，方法变体与适用范围见
[方法与数据卡片](methods.md)，预算公平性规则与主实验计划见 [计划](benchmark_plan.md)。

## 配置结构

```text
configs/benchmark/
├── benchmark.json                # 总配置：任务、profile、设备、输出和通用运行参数
├── core_ode_track.json           # Core ODE：8 系统 × 11 方法 × 5 seeds = 440 cases
├── core_v2*.json                 # 集成测试套件
├── llmsr_llama31_8b_1k.json      # LLM-SR 严格 1k-call 对照批次
├── baselines/<name>.json         # 23 个方法键 的模型参数及 smoke/full 差异
└── tuning/                       # 探索性配置（含 per-dataset override），不进主表
```

总配置 [benchmark.json](../configs/benchmark/benchmark.json) 的主要字段：

| 字段 | 作用 |
| --- | --- |
| `profile` | 选择 `smoke` 或 `full` 预算 |
| `tasks` | 选择 `sr`、`ode`、`pde`、`steady_pde`、`green` |
| `models` / `datasets` | 名称或通配符列表；`["*"]` 选择默认轨；A5/A6 新增名称须显式指定名称或模式 |
| `seeds` | 随机种子列表 |
| `devices` | 每个 baseline 的 `cpu`、`cuda` 或 `cuda:N` |
| `output_dir` / `timeout` / `resume` | 输出目录、每个 target 的 fit 上限 T（秒）和断点恢复 |
| `timeout_grace` | fit 硬终止前的宽限，默认 5 秒；硬上限 T + grace |
| `setup_timeout` / `score_timeout` | 准备/启动与拟合后评分各自的上限，默认各 300 秒；不占用 T |
| `profiles.*.run` | benchmark 适配层的通用预算 |
| `overrides` | 对 `models`、`datasets`、`model_datasets` 或 `run` 做最终覆盖 |

每个 baseline 文件包含基础 `model` 参数、默认 `device`，以及 `profiles.smoke/full` 下的
增量参数。自定义 `--config` 覆盖仓库总配置，因此批次配置通常只需少量字段：

```json
{
  "profile": "full",
  "tasks": ["sr", "ode"],
  "models": ["dso", "physo"],
  "datasets": ["Koza-*", "ode_core_*"],
  "seeds": [0, 1, 2],
  "output_dir": "results/server_sr",
  "timeout": 21600,
  "resume": true,
  "devices": {"dso": "cuda:0", "physo": "cuda:0"},
  "overrides": {"models": {}, "datasets": {}, "run": {}}
}
```

命令行选择和 `--device` 高于总配置；总配置的 `overrides` 高于 baseline profile。
解析后的 `device` 和参数写入 manifest、`case.json`；计时策略另存 `execution.json` 及结果行。
执行协议为 `target-v1`，进入 case 身份哈希。新增计时选项也接受同名连字符 CLI 参数。
T 计入适配器的 `fit` / `train` / 搜索调用；启动、数据准备与模型构造记为 setup，
调用返回后的预测、表达式提取、指标及辅助评估记为 score。各阶段单独限时，超时终止进程树。
当前只有硬上限；到 T 返回当前最优的软截止钩子仍是计划 C3。

`--resume` 保留所有已落盘终态（`ok` / `error` / `timeout` / `skipped`），只补没有终态的 target。
中断尝试的日志和工作目录保留，新尝试写入递增的 `attempts/0002` 等目录。
worker 已写出结果而父进程尚未合并时，续跑会恢复该结果。对同一 case 的并发调度由文件锁拒绝。
已有尝试需要 `--resume`；换预算或重做失败项必须使用新的输出目录。旧执行协议的结果和
manifest 不可用于新 runner 的续跑，历史批次须继续使用原工作树。

并行脚本 `--reuse-dir` 可复制相同配置及计时策略的完整/中断历史，包含失败和超时；
已有目标目录不会被覆盖。`--retries` 只接受 `0`，默认也是 `0`。

## 领域数据集

物理任务与数值 dispatch 分列；工程交付见 [验收](validation.md)，来源条件见 [问题清单](deferred_source_issues.md)，
当前与待接入实例通过 scripts/export_dataset_inventory.py 导出。下表明确区分可运行能力与来源阻塞。

| 数据来源 / 名称 | 物理任务 | 当前接口 | 接入状态与目标 |
| --- | --- | --- | --- |
| CYT：`cyt_flatplate`、`cyt_naca0012` | 湍流闭合关系发现 | `sr` | 15 个局部特征 → `Mut`；按坐标递归划分 16 个空间块、整块留出；跨工况 OOD 和单位仍待验证 |
| 固体：`solid_dif` | 应变率相关本构关系发现 | `sr` | 63 个源曲线实例，各自以应变率 → DIF；按观测应变率留出，重复应变率不跨集合 |
| 固体：`solid_strain_stress` | 应力–应变本构关系发现 | `sr` | 12 个源材料/研究实例，各自以应变、应变率 → 应力；整应变率曲线留出 |
| 固体：`solid_hardening` | 硬化与应变率本构发现 | `sr` | 10 个源材料/研究实例；应变、应变率 → 应力，默认排除预匹配 DIF；整应变率留出 |
| 橡胶：`rubber_train` + `rubber_test` | 条件化橡胶本构发现 | `sr` | stretch、组分、温度 → 应力；已使用官方划分，test 不单独计任务 |
| 落球：`ball_drop` | 实验 ODE 动力学发现 | `ode` | 位置、速度、质量、直径 → 位置/速度导数；整轨迹留出 |
| VGS：`vgs_I_0-100`、`vgs_I_100-200`、`vgs_II_0-1000`、`vgs_II_1000-2000` | 自由界面时变 PDE 发现 | `pde` | 各 500 × 500 的 `h(x,t)`；相邻窗口需按同一物理家族管理 |
| 波破碎：`wave_breaking` | 实验波面 PDE 发现 | `pde` | runner 专用加载器；23 个原始实例，需验证插值投影与边界 |
| TLC：9 个 `tlc_*` 名称 | 一维/二维时变 PDE，含多场系统 | `pde` | 3 项已无损恢复规则网格；`tlc_burgers1d` 可进入一维适配器；二维两项接入 pdefind_nd；散点六项保留来源阻塞 |
| TLC：另 9 个稳态文件 | 二维/三维稳态 PDE 与稳态流动约束 | steady_pde | 独立稳态接口已接入；有效域/界面来源仍阻塞 |
| WDwake：`wdwake` | 二维双速度场 PDE 发现 | `pde` | 坐标来源、不同方向步长和边界未解决；现在预检查明确阻止不正确的 PDE-Net 拟合 |
| Green function：6 基础家族及参数/噪声变体 | Green 核函数 / 算子反演 | green，opt-in | 23 算子实例，联合 forcing 积分响应损失；两份身份未明副本不注册 |
| Green 目录中的 2 个 pyqg NetCDF | 亚网格闭合关系发现 | sr，opt-in | run0→run1，q_subgrid_forcing 两层独立目标；离线评分 |

另新增 `heat`、`allen_cahn`、`convection_diffusion`、`burgers_sine` 四个一维 PDE 名称。
Heat 缺发布的系数真值；不把从数据拟合的扩散系数登记为真值。EqGPT 的 PDE_compound
仍因说明符号冲突及较大参考残差留在待核验清单。

当前数据协议为 `domain-v4`，物理任务、来源、实例、目标和划分写入 catalog、case 与 manifest；
结果按数据协议和物理任务分组，配对比较也隔离协议。该字段参与 case 身份哈希，新旧协议不能续跑混用。
固体原始 `load_dataset` / `load_solid_*` API 保留合并表用于数据检查；benchmark 始终通过
`load_constitutive_instances` 展开，材料身份依据文件名而非跨研究的人工身份核验。
空间块划分只使用坐标，不使用标签；这仍是同工况空间泛化，不声称跨几何 OOD。

默认 `full` 单 seed 枚举 **330 个独立名称、3024 cases、4470 个 target fits**（含不适用组合）。
TLC 当前 89/99 组合不适用；矩阵枚举不等于正式主轨已经冻结。
已接入、来源阻塞、opt-in 名称及重复资产可用以下命令导出；无需训练模型：

```bash
python scripts/export_dataset_inventory.py --expand-instances --output-dir results/dataset_inventory
```

生成 `dataset_inventory.json` / `.csv`，包括来源可用性、物理任务、阻塞项、实例/target 数；
加 `--sha256` 可记录源资产哈希。正式矩阵还需完成 [计划中的接入与冻结项](benchmark_plan.md#4-容量与预算冻结)。

## GPU 支持

| Baseline | 设备 | GPU 上运行的主要部分 |
| --- | --- | --- |
| `dso` | 可指定 | PyTorch 策略网络；表达式奖励仍含 CPU 计算 |
| `symbolicgpt` | 可指定 | Transformer 训练和生成 |
| `e2e` | 可指定 | 预训练 Transformer 推理 |
| `eqgpt` | 可指定 | GPT、代理网络和自动微分 |
| `pdenet` | 可指定 | PDE-Net 训练和预测 |
| `deepmod` | 可指定 | 神经网络、自动微分和稀疏约束 |
| `dlga` | 可指定 | 神经网络和导数；遗传搜索在 CPU |
| `physo` | 可指定 | `physo.SR(..., device=...)` 的张量计算 |
| `spr` | 部分 | PINN 训练；DISCOVER 搜索在 CPU |
| `llmsr` | CPU + 外部服务 | 候选评分和 BFGS 在 CPU；vLLM backbone 占用 GPU |
| 其余（`dscv`、`sga`、`pdefind`、`weakform`、`gplearn`、`pysr`、`pyoperon`、`sindy`、`pysindy`） | CPU | — |

总配置默认全部使用 CPU。启动前会检查 CUDA 可用性和编号；也可只做检查：

```bash
python -u run_benchmark.py --models dso --device dso=cuda:0 --check-devices
```

`run_benchmark.py` 按 case 串行调度；`scripts/run_parallel_benchmark.py` 可按 manifest
分 CPU/GPU 池并发。计划中的资源槽规则（计划 D2、C5）尚未实现。

## 常用命令

```bash
python -u run_benchmark.py --list
python -u run_benchmark.py --dry-run
python -u run_benchmark.py --tasks sr --models gplearn dso --datasets Koza-2 Keijzer-2
python -u run_benchmark.py --tasks ode --models sindy pysindy --datasets ode_core_duffing
python -u run_benchmark.py --tasks pde --models pdefind weakform --datasets kdv burgers
python -u run_benchmark.py --config configs/benchmark/core_ode_track.json --dry-run
python -u run_benchmark.py --check-compatibility --output-dir results/compatibility_audit
```

服务器上先按 [服务器设置](server_setup.md) 激活环境，再用 `nohup bash
scripts/run_benchmark.sh <config> [参数] > logs/run.log 2>&1 &` 后台运行。多个进程并行时
必须使用不同的 `--output-dir`。LLM-SR 需要先启动 backbone 服务，见
[方法卡片中的 LLM-SR 协议](methods.md#llm-sr-protocol-llama-31-8b-instruct-1k-calls)。
其他 OpenAI 兼容服务可在 override 中设置 `endpoint` 与 `api_model`（或
`LLMSR_ENDPOINT` / `LLMSR_MODEL`），密钥只从 `LLMSR_API_KEY` 等环境变量读取；
换模型或预算即为不同协议，需单独记录。

## 输出

```text
results/<output_dir>/
├── benchmark_manifest.json
├── benchmark_summary.{csv,json}       # 每个 target 一行
├── benchmark_aggregate.{csv,json}     # method/task 聚合
├── benchmark_seed_family.{csv,json}
├── benchmark_paired.{csv,json}
└── cases/<case-name>/
    ├── case.json / execution.json / targets.json
    ├── result.json / partial.json                 # 合并视图
    ├── preparation/0001/                         # 枚举阶段日志
    └── targets/i0000_t0000/
        ├── result.json                           # 首次终态，不覆盖
        └── attempts/0001/{target.json,phase.json,worker.json,run.log,result.json,work/}
```

状态为 `ok`、`skipped`、`error`、`timeout`。单个 case 失败不中止后续实验；存在 `error` 或
`timeout` 时主进程返回非零退出码。`--check-compatibility` 写出
`compatibility_report.json` 与 `compatibility_pairs.csv`，只证明组合能进入训练阶段。

整理实验产物时保留 manifest、配置、case/target/attempt 元数据、首次终态和汇总，以及
必要的失败诊断与环境锁定文件。仅在终态已落盘后清理对应 fit 的临时模型、缓存与重复日志；
无终态的工作目录继续保留。重复结果归档须核对 JSON/CSV 已有相同副本后才删除，
历史代码快照、原始数据、源权重和精简验收证据另行保留。

每行另有 `stop_reason`、`stop_reason_source`、`timeout_phase` 和 setup/fit/score 秒数；
聚合输出停止原因数量及比例，语义见[指标协议](benchmark_metrics.md#resource-and-status-fields)。

验证：`python -m pytest tests/test_run_benchmark.py -q`。

A1/A2 来源、资格、同源观测与协议迁移见 [验收记录](validation.md)；
旧 domain-v1/v2/v3 与新 domain-v4 结果不混用，来源不足尚未正式就绪。

## A3/A4 方程子轨

pdefind_nd（pde）与 implicit_pde（pde/steady_pde）逐方程独立 fit/T；
NS 仅声明 u/v 动量，p 为辅助场。稳态没有 t，Re 属已知参数。
PDE-Net 为 system_diagnostic，一次联合 fit 只计一份预算；仅放行有证据的均匀周期端点排除网格。
configs/benchmark/a3_a4_smoke.json 是有时间上限的功能验证配置，不是正式实验。

## A5/A6 有界运行

~~~bash
python run_benchmark.py --config configs/benchmark/a5_a6_smoke.json
python run_benchmark.py --tasks green --models green_dso green_basis --datasets green_laplace --timeout 90
python run_benchmark.py --tasks sr --models gplearn --datasets pyqg_subgrid dso_file_Custom-2
~~~

新增 49 名称 opt-in；Green 仅开放 green_dso／green_basis，普通 SR 方法不声明支持积分损失。
pyqg 每层独立 T，Green 每算子一次联合 fit。缺资产保留目标失败分母，resume 不重试终态失败。
