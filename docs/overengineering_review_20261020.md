# 过度工程复查：2026-10-20

结论：存在局部过度工程，主要表现为无关配置进入实验身份、模块拆分后仍反向依赖入口、身份检查覆盖范围过大。没有证据支持整体重写 runner、统一所有任务协议或大幅删除测试。当前更紧急的是基础接口与历史验收记录不一致，必须先确认可用源码版本。

本次检查的是本地工作树，包含审查前已有的大量暂存、未暂存和未跟踪内容；不是仅审查 HEAD。未修改生产代码、配置、测试、实验结果或 Git 暂存区。新增本报告及 `.local/overengineering_review_20261020/` 下的审查探针和输出。

## 判断依据与范围

检查 runner、benchmark 模块、数据公共接口、评估模块、配置合并、两个批次入口、CI 和既有工程验收记录。没有逐行审查全部上游算法。

当前 `run_benchmark.py` 为 4,278 行、87 个顶层函数，另有 11 个 `kd/benchmark*.py` 模块；24 个 baseline、64 份 benchmark JSON 配置、185 个结果 CSV 字段。这些数字只用于定位复杂度，不能单独证明过度工程。

依据是：某项机制是否解决实际需求；同一规则是否多处维护；无关改动是否传播到其他方法；已有成本或有效性证据是否支持继续建设。已阅读 [10 月 13 日审查](overengineering_review_20261013.md)、[精简验收](engineering_simplification_20261013.md)及[10 月 14 日工程计划修订](engineering_plan_20261011.md)，不把历史问题自动当作当前问题。

## 0. 优先阻断项：源码、公共接口与验收记录不一致

这属于基础一致性问题，不是过度工程本身，但应排在重构之前。

- [kd/dataset/__init__.py:3](../kd/dataset/__init__.py#L3) 当前没有绑定 `GridPDEDataset`、`ODEDataset`、`TabularRegressionDataset`、`DATASET_REGISTRY`、`load_dataset`。`run_benchmark.dataset_catalog()` 在 [runner:424](../run_benchmark.py#L424) 导入注册表失败；选取的四个回归测试在模块收集阶段因 `GridPDEDataset` 导入失败而中止。当前 `_base.py` 也没有 `GridPDEDataset` 类，不能简单补一行 export 就认定修复。
- [pyproject.toml:6](../pyproject.toml#L6) 声明 Python `>=3.8`，第 12 行的依赖列表为空；而 [CI:30](../.github/workflows/core.yml#L30) 依赖 `pip install . pytest... ruff...` 建立环境。默认 Python 3.12 环境实测导入 runner 因缺 pandas 失败。未新建干净环境或执行远程 CI，因此不声称已复现完整安装流程失败。
- [PDEFindModel.__init__:30](../kd/model/kd_pdefind.py#L30) 当前不接受 `terms`，`fit` 在第 85 行把 `alpha/max_iter` 传给 PySINDy STLSQ；[现有测试:653](../tests/test_run_benchmark.py#L653) 却传入 `terms`，并期待旧参数被忽略。这与历史“最小二乘后阈值截断”的验收描述不一致。此项通过源码核对确认，测试尚未运行到模型构造阶段。

最小处理：先从可信提交或可核验的备份确认预期实现，再恢复公共 API、打包声明与模型契约的一致性。不能根据时间戳猜版本，也不能通过补兼容壳或修改断言掩盖算法差异。本次没有确定这些差异的发生时间或原因，不能归因于某个同步服务。

科研影响：PDEFind 的求解器差异可能改变方法定义及结果，不能当作纯接口整理。历史验收记录仅证明历史快照；不能证明当前工作树可重复执行。

## 1. 高优先级：方法无关参数污染 case 与配置身份

**证据。** [build_experiments:516](../run_benchmark.py#L516) 将全局 `run_defaults` 复制到每个方法，随后在第 604 行完整存入 `case["run_params"]`。全局配置同时包含 `dscv_epochs`、`spr_pinn_epochs`、`dlga_population` 等方法专用预算。[method_configuration:74](../kd/benchmark_protocol.py#L74) 将整个 `run_params` 计入方法配置 hash；[runner:633](../run_benchmark.py#L633) 将整个 case 计入名称。

**已复现。** 用真实 planner 和最小合成 SR 元数据构造 gplearn case，仅把 `spr_pinn_epochs` 从 20 改成 21：

- `model_params` 和 `dataset_params` 完全相同；
- case 名称和 `method_config_hash` 都改变；
- gplearn case 携带全部 11 个 run 参数，包括 SPR 和 DLGA 参数。

[run_regression:1752](../run_benchmark.py#L1752) 的 gplearn 拟合路径不读取 SPR 预算；该预算的执行读取位于 SPR 分支。这里确认的是配置和身份传播，没有运行 gplearn 搜索，不声称两次原生训练逐位相同。

**为何属于过度工程。** 为不同算法统一一份大参数字典，使无关调参也生成不同实验身份、重复目录和分组。复杂度直接泄漏给科研使用者。

**最小建议。** 通用 run 只保留真正共用的切分和采样参数；方法预算归入已有 baseline 配置，在构造新 case 时仅保留适用项。用少量显式规则实现即可，不建立配置继承框架。保留输入配置的来源记录。

**验收边界。** 新协议中，无关方法参数不改变该方法的有效配置；相关采样、切分、预算改变必须仍改变身份。旧 manifest 和 hash 不改写，新旧批次不自动合并。

## 2. 中优先级：文件已拆开，依赖边界仍未形成

**证据。** 当前可解析的 `kd` 源码中有 7 个文件、10 处 `from run_benchmark import ...`：

- [ode_rollout.numeric_expression:20](../kd/evaluation/ode_rollout.py#L20) 为解析表达式导入入口脚本；
- [equation.run:214](../kd/evaluation/equation.py#L214) 也从入口导入 `expression_sympy`；
- [closure.finish:97](../kd/evaluation/closure.py#L97) 从入口取回归指标；
- [benchmark_batch.save_checkpoint:64](../kd/benchmark_batch.py#L64) 从入口取报告和 JSON 写入；
- 其余在 `green.py`、`ode_worker.py`、`reference.py`。

此外，[check_compatibility:643](../run_benchmark.py#L643) 把整个 runner 模块传给预检，后者访问 `_load_instances`、`_case_problems`、`_pde_target_bindings` 等私有函数。这样的接口不能清晰声明预检真正需要什么。

**实际代价。** 修改入口脚本会影响底层评估；单独复用表达式解析和评分仍要求入口在导入路径上。测试替换依赖时也必须理解 runner 内部。没有复现循环导入崩溃，不把这种风险写成已发生故障。

**最小建议。** 下次修改表达式处理时，将已有解析函数及其必要辅助函数整体移到一个普通评估模块，让 runner 和 rollout 共用。随后按实际修改需要处理纯指标或报告函数。不要为此引入插件发现、通用 adapter 基类或一次性重写 4,278 行入口。

**验收边界。** 复用已有表达式/指标样例作差分，保护算子语义、保护除法、变量映射和失败状态。不同 SR/ODE/PDE 数学任务仍保持明确的评估分支。

## 3. 中优先级、有取舍：身份检查把无关开发变化变成续跑约束

**证据。** [source_fingerprint:75](../kd/benchmark_identity.py#L75) 不接收所选方法或任务，递归纳入 `kd/`、`scripts/`、`configs/benchmark/` 的指定文件类型。当前覆盖 514 个文件，其中包括全部 64 份配置和纯结果绘图脚本 `scripts/build_visual_report_20261017.py`。[verify:329](../kd/benchmark_identity.py#L329) 严格比较完整 code/environment 对象。

[environment_fingerprint:90](../kd/benchmark_identity.py#L90) 对固定 21 项依赖中已安装包的实现文件做哈希。这里确认了机制，没有重测环境哈希耗时。[main:4235](../run_benchmark.py#L4235) 在 `--dry-run` 返回前也进行完整身份绑定，因此该命令实际承担了正式 manifest 准备工作。

**判断。** 内容指纹、原生资产校验和冻结环境有必要；问题是检查范围偏保守，绘图脚本或未使用配置变化也能使旧身份校验失败。这是维护成本与保守复现策略之间的取舍，不是理由充分的“删掉哈希”。

**最小建议。** 旧冻结运行继续使用原快照。日常只检查配置时直接使用现有 case 构造入口，不触发正式身份捕获。若以后确有大量无关变更阻断运行，再明确版本化指纹范围，优先核对是否可以排除独立报告脚本；不要先建设自动依赖图或多层缓存系统。

**验收边界。** 任何会影响拟合、切分、评分、随机性和实际输入的变化仍必须被识别。不能用包版本号替代当前内容证据，也不能静默改变既有 `--dry-run` 的 manifest 契约。

## 4. 低优先级：checkpoint 与完整派生报告绑定

**证据。** [run_slots:273](../scripts/run_slots.py#L273) 每个 case 事件调用 checkpoint；[save_checkpoint:77](../kd/benchmark_batch.py#L77) 再调用完整 [save_summary:3493](../run_benchmark.py#L3493)。后者写 9 个报告文件，对 5 个指标遍历所有方法对；每次 `paired_comparison` 又扫描全部结果并重新计算有 execution policy 的行的比较 hash。

**本轮探针。** 12 个合成方法、120 行结果，一次 `save_summary` 触发 40,200 次 `_comparison_row` 调用，即：

`120 × (5 × C(12,2) + 5)`。

JSON/CSV 写入全部替换为内存 sink，未产生实验结果文件。此计数不含 `aggregate_results` 内自己的工作量，也不是实际磁盘或服务器性能测量。槽位线程可以继续消费任务，不能推断主线程生成报告必然阻塞 worker。

**已有反证。** [10 月 14 日测量](engineering_plan_20261011.md#L43) 已用三个真实输入重放，记录单次中位数约 0.13–0.28 秒，独立复测约 0.21–0.36 秒；当时决定暂不节流。这里引用历史记录，本轮未重跑该测量。旧样本不能直接代表当前大批次。

**最小建议。** 保持低优先级。有实际汇总延迟、事件积压或较高累计占比后，才将派生报告改为阶段结束或定期刷新。首次终态继续即时保存，最终报告保持一致。无需数据库、事件溯源或增量统计框架。

## 5. 研究范围控制：辅助评估已有原型，先验证价值再扩建

[辅助评估文档](auxiliary_evaluators.md) 记录 PIC 排名验收未通过，物理可信度原型尚缺专家标签和 judge 对比。当前已有 ANN/PINN 缓存、候选重拟合、多个质量门槛，以及冻结证据卡和离线 judge payload。

这些机制有明确用途；PIC 是 opt-in，physical credibility 也没有成为主排行榜的必经路径。本次不能据此认定现有原型应删除。

建议把下一步限制为原有的决定性验证：PIC 能否稳定区分真结构与预先固定的反例；证据辅助判断是否在专家标注上产生可测收益。验证前不继续扩展通用 judge 管理或综合分数。这是防止未来过度建设的建议，不是已证明的性能缺陷。

## 应保留的复杂度

- 任务能力检查、明确方程目标、时空切分先于导数构造：保护数学任务及泄漏边界。
- 首次终态不可变、失败目标计入分母、显式缺失状态：保护比较公平性。
- 独立 worker、硬超时和方法特定 soft-stop：处理不同原生运行时的真实差异；不能统一成一个伪通用回调。
- 资产内容身份、冻结快照、原子写入、协议隔离：保护复现与长批次恢复。
- 配对比较、按 seed/family 汇总、冻结报告的不同统计目标：不能为减少函数数目而合并其统计含义。
- 正式资源槽与诊断并行入口：两者约束不同，当前没有足够证据支持合并。

## 实际执行与证据

审查产物：[探针源码](../.local/overengineering_review_20261020/probe.py)、[探针输出](../.local/overengineering_review_20261020/probe.json)。输出含关键源文件 SHA-256，便于确定本次结论对应的源码。探针使用真实 planner、summary 和 source fingerprint 函数；SR catalog 和 summary 行是合成输入，summary 的文件写入被替换。

执行环境为已有 `kd-all-baselines-py310-20260929`，Python 3.10.21；未安装依赖、启动服务或模型训练。PowerShell 命令：

```powershell
$auditPython = 'C:/Users/22412/miniconda3/envs/kd-all-baselines-py310-20260929/python.exe'
& $auditPython -B .local/overengineering_review_20261020/probe.py
& $auditPython -B -m pytest -q -p no:cacheprovider `
  tests/test_run_benchmark.py::test_pdefind_uses_configured_terms_for_fit_and_rollout `
  tests/test_run_benchmark.py::test_repository_has_one_config_per_baseline `
  tests/test_run_benchmark.py::test_execution_protocol_versions_are_not_pooled `
  tests/test_run_benchmark.py::test_aggregate_does_not_pool_protocol_versions
```

结果：

- runner 在 Python 3.10 环境可导入，但真实 `dataset_catalog()` 导入注册表失败。
- pytest 在收集阶段出现 **1 个 collection error**，四个指定测试均未执行；不是“4 failed”，更不是通过。
- 内存探针执行完成，确认配置身份传播、反向导入、报告重复扫描和指纹范围。
- AST 扫描发现上游资产目录 `kd/dataset/EqGPT/code/surrogate_model.py:307` 有语法错误；探针最初因此退出，随后记录并跳过该文件，其余统计完成。不声称全仓库语法检查通过；未判断该资产文件是否进入活动算法路径。
- 默认 Python 3.12.3 导入 runner 因缺 pandas 失败；这是另一个环境的观察，不与 Python 3.10 的收集错误混为一谈。

## 建议执行顺序与科研边界

1. 核实并修复当前数据 API、打包与 PDEFind 契约的一致性，恢复少量关键回归。
2. 收窄方法有效配置，解决已经复现的无关参数身份传播。
3. 随表达式或评分需求收回反向依赖，每次只处理明确共享的函数。
4. 指纹范围和报告刷新等待实际成本依据；辅助评估先做研究验证。

本轮没有改变预处理、评估、算法或实验协议，因此没有通过本轮修改直接改变已有科学结果。已发现的问题意味着不能用历史通过记录替代当前可复现性检查。后续配置、求解器或身份范围修改必须记录新协议/实验身份，保留历史结果；是否保持科学结论需要相应差分与实验验证，不能由“代码运行成功”推断。
