# Benchmark 当前计划

更新：2026-10-02。代码与结果文件优先；本地运行交接见根目录 `STATUS.md`。
规范：[运行](benchmark.md) · [指标](benchmark_metrics.md) · [方法](methods.md) ·
[数据](data.md) · [验收](validation.md) · [来源问题](deferred_source_issues.md) · [服务器](server_setup.md)。

## 1. 当前状态

A1–A6 可验证工程交付与验收完成，完整回归 496 passed，0 failed，0 skipped。
C1/C2/C4 已完成：每 instance/target 独立 worker、fit/setup/score 分阶段限时、
首次终态与中断恢复、失败/超时不自动重跑、停止原因及来源记录。
E1–E3 已实现并通过针对性验收：ODE 系统轨迹评分、独立参考行、冻结结果统计与作图。
C5/C6 已实现并通过本地进程回归：两槽绑核、CUDA/线程隔离、RSS 超限终止、
CPU/GPU 成本记录与汇总；完整代码回归 531 passed、0 failed、0 skipped。
C7 full 方法并行参数已与 D2 一致，17 项专项与 548 项完整代码回归通过，0 failed、0 skipped；
证据 `results/validation/c7_targeted.xml` 与 `results/validation/c7.xml`。配置与槽绑定均记录有效并行数，
并限制并行任务内部的数值线程。C3 七类方法软截止已实现，22 项专项与
570 项最终完整代码回归通过（0 failed、0 skipped），证据 `results/validation/c3.xml`。
Linux/真实 CUDA/vLLM 部署 smoke 待验收，
不能把当前实现称为已满足完整公平资源协议。本轮未运行全量实验。

数据 domain-v4、执行 target-v2；指标 2.1/2.2/2.3 分区聚合。
23 方法键、111 注册名、380 runner 名（rubber_test 为配对集，独立名称 379）。
默认单 seed full 为 3024 cases / 4470 fits；49 新名称显式 opt-in 增加
280 cases / 289 fits，全能力容量 3304 cases / 4759 fits。均包含不适用组合。
注册、加载、短训练通过不等于来源或方程恢复已认证。

历史全量、服务器补跑、tuning 与 smoke 来自不同代码、预算和协议，只作诊断与测速，不能拼成主表。
正式主实验尚未启动。ODE 同时报告导数和通过参考校验的系统轨迹评分；PDE rollout 仅已验证的一维均匀周期标量场。
PIC 排序校准未通过，物理可信度缺专家验证，保留为辅助原型。
LLM-SR 客户端已实现，但 backbone 未下载、真实生成未验证；vLLM 环境独立建立。

## 2. 已定决策（2026-09-29，用户确认）

### D1. 公平性 = 统一资源配置 + 统一 wall-clock 上限 T（按 target 计）

- 每个 target 的 fit 时间上限为 T；case 的 fit 硬上限总额 = n_targets × (T + grace)，
  准备与评分另计，不用一个覆盖所有阶段的 case 计时器。
- 能随时返回当前最优解的方法（GP、PySR、Operon、DSO、DSCV、LLM-SR、PhySO）接收软截止时间，
  到时返回当前最优表达式，记 `stop_reason=time_limit`；超过 T + grace 强制终止，记 `status=timeout`。
- 方法内部预算使用冻结的推荐配置（优先上游默认/官方推荐值），T 是所有方法共同的外部上限。
- 结果评分（预测与指标计算）不计入 T，单独设上限。

### D2. 硬件与资源槽

可用容器：8 核 CPU、60 GB 内存、1 × RTX 4090 24 GB、30 GB 共享内存，**单个容器最长 48 小时**。

定义两个并发资源槽（slot），同一时刻各运行一个 target fit：

| Slot | CPU | 内存上限 | GPU | 运行方法 |
| --- | --- | --- | --- | --- |
| `cpu` | 核 0–3（4 核） | 28 GB | 无 | dscv、gplearn、pysr、pyoperon、sindy、pysindy、pdefind、weakform、sga、pdefind_nd、implicit_pde、green_basis |
| `gpu` | 核 4–7（4 核） | 28 GB | 整张 4090 独占 | dso、symbolicgpt、physo、e2e、eqgpt、deepmod、dlga、pdenet、spr、llmsr、green_dso |

- 方法可用满本槽 4 核（PySR `procs=4`、Operon `n_threads=4`、gplearn `n_jobs=4`、
  torch 4 线程、LLM-SR 4 个 evaluator），不再统一限为单线程。
- LLM-SR 的 vLLM 服务占用 `gpu` 槽的 GPU 与核，属于 LLM-SR 的资源；LLM-SR 阶段 GPU 不与其他方法共享。
- 表格中标注 GPU 方法；CPU 组与 GPU 组的成本分开比较。
- 每行记录 wall 时间、CPU 秒（进程树）、GPU 占用 wall 秒、峰值 RSS、CUDA 分配峰值。

### D3. 调参规则 (a)：不针对数据集调参

- 每个方法、每类任务（SR / ODE / PDE）只有一套配置；主表内禁止 per-dataset override。
- 配置来源：上游默认或官方推荐；与上游不同的值必须在 [方法卡片](methods.md) 中说明理由。
- 2026-10-05 用户明确授权四方法在不相交的独立合成开发数据上调整参数，
  须先达到结构恢复与系数误差 `<1e-5`，随后冻结每方法、每任务的一套配置；
  试验与失败记录全部保留，不以正式测试响应或参考方程选择参数。
  本轮细节见 [四方法校准](four_methods_calibration.md)；其余方法沿用先前开发集用途。
- 函数集/算子库属于任务说明（语法），不算调参：SR 使用数据集发布的 `function_set`
  （`search_space_policy=declared`）；Core ODE 需预先声明**一套对 8 个系统共用**的语法
  （按系统声明会泄露真值）。`declared_plus_reference` 为 oracle 诊断，永不进主表。
- 规则 (b)（每方法最多 K 套配置、训练集 CV 选择）留作后续。

### D4. 预训练与 LLM 方法进主表，另附对照组

- E2E、SymbolicGPT、LLM-SR 与其他方法同表，加列标注"预训练先验 / LLM"，附污染风险说明。
  预训练成本不计入单题预算；SymbolicGPT 逐 fit 训练计入。
- LLM-SR 主表条目：1000 调用为内部预算，T 生效，跑不完记 `time_limit`。
- 对照组：严格 1000 调用、不设 T（仅 43200 s 安全网），与 LLM-SRBench 设置对照，
  配置为 `configs/benchmark/llmsr_llama31_8b_1k.json`。

### 待冻结的实验选择

1. 文章核心论点与投稿目标（决定数据公开工作量）。
2. T 的最终取值与 SR 主队列规模、seed 数（由第 4 节测速后决定）。
3. Core ODE 共用语法的具体算子集合。

## 3. 剩余工程工作

| # | 改动 | 位置 | 验收 |
| --- | --- | --- | --- |
| C1（完成） | 每个 instance/target 独立子进程 + 按 target 的 fit 硬超时；setup/score 单独限时；case 结果合并 | `run_benchmark.py`、`kd/benchmark_execution.py` | 真实双 RHS 检查：首个超时，第二个仍成功；终止进程树；保留全部行 |
| C2（完成） | 已落盘终态（ok / error / timeout / skipped）视为完成，只补无结果中断项；保留首次结果及全部尝试 | runner、并行脚本 | 中断前后/父进程漏合并恢复通过；失败和超时不自动重试；配置/预算不符拒绝复用 |
| C3（完成） | 统一累计 fit 软截止；可取消 LLM 请求与评估、原生时间参数、代际/训练/批次钩子；最后步骤观察截止 | `kd/benchmark_deadline.py`、runner、LLM-SR、DSO/DSCV；PhySO 公开 logger | 当前最优与 time_limit、无候选失败、进程清理、随机演化一致性、原生有界检查及终态续跑通过；target-v2 隔离旧协议 |
| C4（完成） | 停止原因、来源字段、适配器信号传递、失败/超时归类（C5 增加第七类 resource_limit） | adapter + 结果行 + 聚合 | 每行有值；聚合数量/比例/来源；无信号的原生返回标为 native_return |
| C5（完成） | 两槽绑核、CUDA/线程隔离；28 GB RSS 采样超限终止进程树，记 resource_limit；本机 vLLM 服务树纳入槽；资源策略与首次结果冻结 | `scripts/run_slots.py`、`kd/benchmark_resources.py`、`kd/benchmark_execution.py`、`hpc/activate_kd_server.sh`；旧并行入口 CPU worker 默认改为 1 | 本地两槽并发/不重叠核、环境、RSS 子进程树终止、服务计费与续跑通过；Linux/CUDA/vLLM 部署 smoke 待验证 |
| C6（完成） | 采样进程树 CPU 秒、独占 GPU 槽 wall 秒；准备/耦合评分共享成本单列并按 case 去重 | 父进程资源监视器、`run_benchmark.py` 汇总、冻结与配对报告 | summary/aggregate JSON/CSV 有字段与测量覆盖；失败和超限保留成本，不混资源配置 |
| C7（完成） | full PySR multiprocessing/procs=4、Operon n_threads=4、gplearn n_jobs=4、LLM-SR num_evaluators=4；torch 4 线程；并行任务数值线程为 1；绑定后参数/设备写入新身份 | `configs/benchmark/baselines/*.json`、`kd/benchmark_resources.py`、`scripts/run_slots.py` | SR/ODE full 配置与 D2 一致；真实 gplearn/PySR 四 worker 绑核、Operon 四线程与 torch 线程检查通过；预算和原 manifest 保留 |
| C8（可选） | anytime 轨迹：记录（时间、训练误差、表达式），在 60 s / 300 s / T 处回放测试指标 | 支持 C3 的 adapter | 一次运行得到预算敏感性曲线 |

评估补充：

| # | 改动 | 验收 |
| --- | --- | --- |
| E1（完成） | `ode_rollout.py`：完整 RHS 系统在整条留出轨迹初值上积分；独立评分进程与缓存，不覆盖 target 首次终态 | 解析解及 Robertson/Radau 参考校验；缺 RHS、发散、无效 RHS、评分超时明确记状态 |
| E2（完成） | `--include-reference`：真实方程走同一评分流水线；缺真值显式 abstain，未认证来源仅作诊断 | 单独的 `model=reference` 行；聚合、配对和主表排除参考行；不拟合模型 |
| E3（完成） | `scripts/report_frozen_benchmark.py`：冻结目录生成主表、Pareto、status 图及 bootstrap CI | 仅目录与 manifest SHA256；校验目标计划和预算；缺失保留分母；整 seed bootstrap，单 seed 不给 CI |

原生返回无停止信号时暂记 `budget_exhausted`，来源为 `native_return`；不证明耗尽内部预算。
Windows 原子替换竞争有回归保护；服务器/Linux 进程组仍需部署后 smoke。

### 实施约束

- 训练侧完成归一化、导数、候选选择及早停；测试响应和参考真值不得影响训练。
- 保留所有已知 target 分母；能力不适用、来源阻塞、error/timeout 分开报告。
- 无认证真值不授予 recovery；低残差不证明唯一恢复；观测残差与场/轨迹预测分列。
- 每 target fit 使用 T；硬截止 T+grace，默认 grace=5 s，setup/score 默认各 300 s。
  Green 联合 forcing 用一份 T，pyqg 两层独立 T，联合系统与逐 RHS 预算预先声明。
- case 身份、聚合及配对隔离协议、配置和预算；首次终态不可覆盖，重跑使用新目录。
- 外部资产和许可证未知时保持明确状态，公开证据使用相对路径，不发布观测或私有连接。
- 接口变化后运行相关回归；完整交付后做一次完整回归，真实数据检查加 `--require-external-data`。
  最终证据集中保存 `results/validation/`，避免堆积重复中间版本。

## 4. 容量与预算冻结

默认单 seed 4470 fits：代数公式 2412、闭合 18、本构 774、ODE 308、时变 PDE 942、稳态 16。
新增 opt-in 容量 289 fits。正式容量需排除不适用组合，按真实实例、独立目标和种子展开。
导出：`python scripts/export_dataset_inventory.py --expand-instances --output-dir results/inventory`。

1. 开发集与主队列分开：ODE 补充系统、SR Livermore2 子集、PDE fisher/fisher_linear。
2. 冻结方法配置后宽上限测速，记录原生预算耗时与可获得的 anytime 轨迹。
3. 结合多数方法完成预算的时间和总机时，选定 T、SR 主队列规模与 seed 数；不看测试指标选 T。
4. 按最坏预计 ≤40 h 分块，适配容器 48 h 上限；LLM-SR 独立占 GPU 槽。

## 5. 协议冻结清单（主实验前必须完成）

- [x] C1、C2、C4 实现并通过针对性测试。
- [x] E1–E3 实现、针对性回归、4-case CLI 与续跑不变性验收；详见 validation.md。
- [x] 本地数据目录、注册表、目标类型、代表实例加载和 PDE 形状预检查审计（见 validation.md；不代表所有数据已科学认证）。
- [x] A0 台账与数据协议实现；A1 的空间/材料划分、A2 可验证的一维接入和规则网格恢复通过测试。
- [ ] 冻结全家族覆盖台账、来源哈希、方法适用性、实例/变体/副本与 target 分母。
- [x] A1–A6 可验证工程实现与回归验收。
- [ ] 拟纳入正式运行子轨的剩余来源与参考条件；阻塞见 deferred_source_issues.md。
- [ ] 从修正后的领域矩阵重新估算容量；TLC、Green、WDwake、CYT、本构均有明确子轨与接入状态。
- [x] C5/C6 实现与本地进程回归；证据见 validation.md（未运行全量实验）。
- [ ] C5/C6 Linux 绑核、真实 CUDA 与 vLLM 服务部署 smoke。
- [x] C7 实现与专项回归；旧 C5 策略与新 d2-parallel-v1 分区，详见 validation.md。
- [x] C3 实现并通过测试；22 项专项与 570 项完整回归，未运行全量实验。
- [ ] 移除主轨配置中的 per-dataset override：`core_ode_track.json`、`core_v2*.json` 的
      SINDy/PySINDy `model_datasets` 多项式阶数（按真值选择阶数，违反 D3），改为一套共用阶数；
      `configs/benchmark/tuning/` 全部为探索性配置，不进主表。
- [ ] 审计每个方法 full profile 与上游默认的差异（physo 20 epochs、deepmod 20 epochs、dscv 10 iterations
      等是临时缩减值），在方法卡片中列出。
- [ ] 声明 Core ODE 共用语法并写入数据集元数据。
- [ ] 开发集测速，确定 T、SR 主队列和 seed 数。
- [ ] Robertson：干净数据进主表；噪声/稀疏轨在观测时间表与导数估计器验证前不进主表。
- [ ] LLM-SR：backbone 下载后执行 `check_llmsr_backbone.py --sha256` 与 `--server --calls 4`，再测速。
- [ ] 冻结代码提交、环境锁定文件、manifest SHA256；主表只从这一次运行产生。

## 6. 执行顺序

1. 完成资源槽/方法并行度及软截止的部署 smoke；E1–E3 已验收，来源问题按记录后续核验。
2. 完成冻结清单，按实际独立实例和目标重新估算容量。
3. 按已就绪的 ODE、PDE、SR、闭合/本构、多维/稳态与 Green 子轨运行，未就绪项显式阻塞。
4. LLM-SR 主协议和严格 1k 调用对照分开；预算敏感性、污染变体与鲁棒性按预声明范围开展。
5. 从冻结结果生成统计、图表和论文；原过时草稿已移除。

## 7. 其他未决技术问题

- SGA SVD 不收敛诊断；DSCV 在 SR 上全部 timeout 的原因。
- DSO（TF1→PyTorch 移植）数值保真度对照；E2E 语料重叠与完整论文设置保真度。
- EqGPT 排除映射只覆盖 `kdv`；SymbolicGPT 合成语料重叠记录。
- 真正的 train/test OOD 划分；噪声与稀疏的独立受控研究。
- PIC 导数质量与 Burgers/KdV 重新校准；物理可信度的专家标注与判定器比较。
- AI Feynman、SINDy-PI 未集成。

## 8. 交接约定

不覆盖其他协议或服务器工作树；缺失、失败、超时和无真值样本不从分母隐藏。
AIC/BIC 需要训练似然来源；原始数据、权重和私有配置不公开。
历史诊断保留汇总与终态，已完成 fit 的日志和中间模型不作为必须长期保留的资产。
