# A1–A6 工程验收

更新：2026-10-02。已完成可验证工程实现；来源不足项见 [待解决问题](deferred_source_issues.md)。
正式全量实验尚未启动，功能验证结果不作为方法性能排名。

## 已验收能力

| 工作 | 交付与验证 | 保留的边界 |
| --- | --- | --- |
| A1 | CYT 16 坐标块留出；85 固体源实例保留全部观测、整应变率留出；目标泄漏拒绝、资格判定与失败分母；rubber 官方训练/测试组合隔离；WaveBreaking 按帧空间投影 | 单位、物理材料身份、跨几何 OOD 未认证 |
| A2 | TLC 三网格无损重排；MAT 嵌入坐标与轴顺序；分侧计算训练导数；Allen–Cahn、对流扩散、Burgers sine 参考残差；Heat 不臆造系数 | compound 符号冲突、Heat 系数及未解码字段仍阻塞 |
| A3 | 多维分轴导数、多场/辅助场、守护散点局部多项式、二阶时间与混合导数、固定参数基；pdefind_nd 独立目标预算 | WDwake 与部分散点/参数数据缺坐标或几何证据；高阶 rollout 未完成 |
| A4 | 12 份稳态文件显式状态；独立 steady_pde；空间支持隔离；implicit_pde 非平凡性、秩/歧义与训练尺度固定的残差评分 | 缺几何/掩码数据保持 source_blocked；低残差不证明唯一恢复，无 BVP 时预测 abstain |
| A5 | 23 Green 实例、6 家族；整 forcing 60/20/20 留出；实际积分权重、固定 U_hom；DSO 积分响应损失及有限核基；训练/测试/参考隔离 | periodic/jump 缺 ExactGreen；两个来源未知 MAT 不注册；噪声尺度和物理 forcing 身份未认证 |
| A6 | pyqg 两层离线 PV closure、run0→run1 留出；25 文件 SR；443 资产哈希与预训练暴露标签；18 导出修复 | 在线稳定性、生成器版本、许可及部分语料重叠未认证 |

固体 137 个 XLSX/PKL 曾与固定上游 revision
`b4af0b5b49a1c1d0392a6504ec12a2c763136957` 的 Git blob 核对一致。
hardening 的 56 条曲线与 strain_stress 对应观测完全相同，不能当作独立观测重复计权。
固定源：[pack_data 与 DIF 匹配](https://github.com/woshixuhao/Discovery_of_soild_consititutive/blob/b4af0b5b49a1c1d0392a6504ec12a2c763136957/codes/Discovery_of_hardening_strain_rate.py)。

## 验证证据

| 检查 | 验收结果 |
| --- | --- |
| A1/A2 完整回归 | 368 passed，0 failed，0 skipped |
| A3/A4 完整回归 | 421 passed，0 failed，0 skipped |
| A1–A6 最终完整回归 | 496 passed，0 failed，0 skipped；230.89 秒 |
| A5/A6 专项 | 233 passed，0 failed，0 skipped |
| 最终真实 CLI | 20 cases / 21 targets 全部 ok；有限覆盖 1；续跑 attempt 保持 1 |
| 兼容性 | 20 compatible，0 incompatible，0 error |
| 资产 | 443 资产与 2 关联训练资源重新哈希一致；空 checkout 重建 443 local_missing |

精简证据保存在本地 `results/validation/`：`a1_a6.xml`、`acceptance_checks.json`、
`smoke_summary.json`。三份逐文件来源清单与原验收记录合并压缩为
`source_evidence.json.gz`，保留诊断数值、来源字段和阻塞证据；原始观测没有复制。
历史实验保留 case/target 终态及汇总；已清理已完成 fit 的文本日志与一次性校验文件。
服务器归档中的 JSON/CSV 已提取保留。后续按用户要求完成临时产物清理，删除 8448 份
文件、约 735.55 MiB，包括已完成 fit 的中间模型、Python/测试缓存、旧公开 checkout
及已有一致结果副本的归档。保留历史服务器代码快照、Git bundle、统一环境源码归档、
环境验证归档，以及无终态的工作目录。清理报告位于本地
`.local/experiment_cleanup_20260930.json`。

清理后完整回归：**496 passed、0 failed、0 skipped**（408.61 秒），JUnit 为
`results/validation/cleanup.xml`。Ruff 与 diff 检查通过；数据哈希及文档链接核对见本地清理报告。

本次追加清理核对 8636 份结果/验收元数据及 445 项数据/训练资产哈希；未运行全量实验，
未重跑完整回归。提交涉及的 Python 文件通过 Ruff，公开 API 导入回归 1 passed，
JUnit 为 `results/validation/cleanup_commit.xml`。

## 当前契约与容量

数据 `domain-v4`，方程 `equation-v1`，Green `green-v1`，闭合 `closure-v1`，执行 `target-v2`。
原指标 2.1、方程 2.2、新 Green/闭合/文件 SR 2.3 分区聚合，旧协议不得续跑复用。
23 方法键、111 注册名、380 runner 名（379 独立名称）。
默认单 seed full 仍为 3024 cases / 4470 fits；新增 49 opt-in 名称提供
280 cases / 289 fits，总容量 3304 cases / 4759 fits，包含能力不适用组合。
Green 每个算子联合 forcing 用一份 T；pyqg 两层分别用 T。

## 复现

统一 Python 3.10 环境中运行：

```bash
python -m pytest --require-external-data -q
python run_benchmark.py --config configs/benchmark/a3_a4_smoke.json
python run_benchmark.py --config configs/benchmark/a5_a6_smoke.json
python scripts/export_dataset_inventory.py --expand-instances --sha256 --output-dir results/inventory
python scripts/export_asset_manifest.py --expected-manifest docs/asset_manifest.json --output-dir results/assets
```

`export_asset_manifest.py` 不下载数据，只生成哈希、来源、映射和缺失状态。
已完成任务的验收测试归入 `test_domain_protocols.py`；数值正确性、泄漏与超时恢复回归继续保留。
后续实施及正式实验冻结条件见 [计划](benchmark_plan.md)。

## E1–E3 验收

ODE 耦合 rollout、独立参考行与冻结报告已实现。2026-09-30 针对性回归
241 passed、0 failed，证据 `results/validation/e1_e3.xml`；未运行全量实验或完整回归。
最终 E1–E3 专项 22 passed、0 failed，证据 `results/validation/e1_e3_additional.xml`。
覆盖解析振子、Robertson/Radau、参考不一致、RHS 缺失/非有限/发散、评分超时缓存、
参考无训练、PDE/Green 共享评分，以及缺失分母、重复目标、预算/hash 冲突、单 seed CI。

真实小规模 CLI：4 cases / 8 targets 全 ok，轨迹评分全部 ok；参考轨迹 NRMSE 约 1e-8。
续跑核对 8 份 target 首次终态与 4 份评分缓存哈希不变、attempt 仍为 1。
`results/e1_e3_smoke/` 保留终态与报告；`analysis/5075816d6536/` 包含表格和 PNG/PDF。
改动文件 Ruff 与 diff 检查通过。使用入口与能力边界见 benchmark.md、benchmark_metrics.md。

## C5/C6 验收（2026-10-01）

完整代码回归 **531 passed、0 failed、0 skipped**，232.86 秒，包含
`--require-external-data`；JUnit 为 `results/validation/c5_c6.xml`。
本轮未运行全量 benchmark 实验。资源边界最终复核专项 **135 passed、0 failed、0 skipped**，
91.83 秒，证据 `results/validation/c5_c6_targeted.xml`；改动 Python 文件 Ruff、
Bash 激活脚本语法检查通过。

本地 Windows 上用真实进程验证两槽并发、四核且核集合不重叠、CUDA/线程环境隔离，
以及子进程 CPU 计费、RSS 子进程超限后整树终止、`resource_limit` 失败分母与成本保留。
模拟长期 GPU 服务验证先前 CPU 不重复计费、RSS 纳入本槽，普通完成保留服务，超限终止服务。
实际 SINDy 双 RHS worker 验证资源字段写入 summary/aggregate、资源变更拒绝续跑，
首次终态与正常续跑字节不变；监督完成但父进程未合并的候选可恢复。
调度测试验证每槽串行、两槽并发；dry-run 不加载数据或拟合。冻结报告拒绝资源标签改写，
准备/耦合评分成本按 case 计一次，评分资源失败不覆盖 fit 首次结果。

新增资源协议为 `slots-v1`，原执行协议仍为 `target-v1`，资源策略进入新 case 身份及
执行策略，不复用无资源约束的旧结果。28 GB 上限按十进制字节转换为历史 MiB 字段。
CPU/RSS 为采样估计，可能遗漏采样间隔内完整结束的短进程或瞬时峰值；GPU 秒为独占槽
wall 时间，不是 GPU kernel/utilization 测量。Linux taskset、真实 CUDA 与 vLLM 仍待服务器
部署 smoke；该轮 C3 软截止、C7 方法内部并行参数未完成，C7 后续验收见下节。

## C7 验收（2026-10-01）

最终完整代码回归 **548 passed、0 failed、0 skipped**（281.90 秒），包含
`--require-external-data`；JUnit 为 `results/validation/c7.xml`。改动 Python 文件 Ruff、
Bash 激活脚本语法与 diff 检查通过；没有运行全量 benchmark 实验。

full 配置与槽绑定统一到 D2：gplearn 4 worker、Operon 4 线程、PySR multiprocessing
4 worker、LLM-SR 4 evaluator、torch 4 线程。并行任务的数值库线程为 1，
JuliaCall 线程与 Julia 环境同时显式设置；应用环境进入 summary 的 `thread_environment`。
新策略标记 `parallelism_protocol=d2-parallel-v1` 并改变资源哈希，旧 C5 策略与终态保持可读。

专项 **17 passed、0 failed、0 skipped**（57.28 秒），JUnit 为
`results/validation/c7_targeted.xml`。检查 SR/ODE full 参数、旧/过量并行参数校正、
smoke 的小预算与串行设置、源 manifest 不变、新 case 身份与旧 C5 策略隔离。
完整回归另外验证真实 target worker 路径设置 torch 的 4 线程，而无需真实 GPU 训练。
真实 gplearn 小规模 fit 后检查四个 joblib worker 的绑核与单线程数值库；
Operon 32 个体/2 代四线程搜索返回有限预测；PySR 24 点/1 迭代/4 worker 搜索返回有限
预测，父进程采样确认四个 Julia worker 继承槽 affinity，Julia 主运行时为 1 线程。
这些是有界原生接口验证，没有使用 full 训练预算或运行全量实验。

C7 同时校正槽绑定后的模型构造 `device`，防止源 manifest 的 CPU 构造参数与
GPU 槽设备不一致。LLM-SR 客户端本身使用 CPU，vLLM 服务仍按 GPU 槽记录；
本轮没有启动或验证真实 vLLM 服务。Linux、真实 CUDA/vLLM 的部署 smoke 与 C3 软截止
仍待完成；本地验收不解除其他来源、预算与语法冻结条件。

## C3 验收（2026-10-02）

最终完整代码回归 **570 passed、0 failed、0 skipped**（440.71 秒），包含
`--require-external-data`；JUnit 为 `results/validation/c3.xml`。其中 C3 专项 **22 passed**。
改动 Python 文件 Ruff、diff 检查通过；没有运行全量 benchmark 实验。

七类方法共用从 fit 入口开始的软截止，累计 fit 不重置预算，setup/score 排除：
LLM-SR 可终止请求进程与 evaluator；PySR/Operon 注入原生剩余时间；
gplearn 按代 warm_start 并保持原有随机演化；DSO/DSCV 保留已完成候选并在训练边界退出；
PhySO 通过公开 logger 钩子，在完成批次进入 Pareto 后返回最优候选。
最后一个训练步骤跨过 T 时也观察截止，避免与内部预算耗尽混淆。
无有效候选保持 error，超 T+grace 强制终止保持 timeout/hard_timeout。

专项覆盖整数/RandomState/全局 RNG 三种 gplearn 演化一致性、最优候选保留、
慢 HTTP 取消与无存活子进程、evaluator 取消、无候选失败、真实 DSO/DSCV 钩子、
单 target worker 的 time_limit 与首次终态续跑不变、严格 1000 调用硬安全网及终末步骤。
原生 PySR/Operon/PhySO 检查仅用 24 点玩具数据，由独立进程 watchdog 限时；
PySR 使用 4 worker、4 核槽及 60 秒 fit 预算，覆盖冷初始化和搜索，验证原有数值/函数早停条件保留；
Operon 为 0.01 秒软预算（原生整数秒向上取整），PhySO 为 0.3 秒/32 个体批次。
较高迭代计数只用于确保触发截止，不能绕过硬上限，未用于 full 科学实验。

新协议为 `target-v2`，保留 `domain-v4` 与 `d2-parallel-v1`。
默认 `soft_deadline=true`；严格 1000 调用对照为 false，并保持 43200 秒硬安全网。
旧 target-v1 首次终态及历史文件未改动，可读取；新 runner 不向旧目录追加实验。
PySR/Operon 缺乏可靠原生停止码时，来源 `fit_deadline` 表示已观察到 fit 截止。
代际、批次、长原生调用与初始化仍可能触发 T+grace 硬超时。
Linux/真实 CUDA/vLLM 部署和服务器端 HTTP 断连取消生成仍待验证；
本地工程验收不解除来源、正式预算、语法与矩阵冻结条件。
