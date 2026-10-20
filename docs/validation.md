# A1–A6 工程验收

> 2026-10-20 清理说明：`results/validation/`、`results/calibration/` 等证据目录已于 2026-10-20 删除，下文 `results/...` 与 `.local/...` 路径只作历史记录，本地已不存在；范围见[工程记录](#工程记录)。

更新：2026-10-12。已完成可验证工程实现；来源不足项见 [待解决问题](deferred_source_issues.md)。
四方法主队列已获用户授权；代码与独立精度校准结果不作为主实验性能排名。

## 替换容器接续（2026-10-12）

用户授权继续未完成实验；新容器使用原冻结源码与持久化结果，未同步本地开发树的数据协议修订。

- 挂载与结果链接核验通过。启动前四方法保留 1936/4290 个首次终态：SISSO 1425、DLGA 15、
  PhySO 410、SymbolicGPT 86；LLM-SR 保留 230/1425，均为 `ok`。
  完成数包含错误和超时，不等同于方程恢复成功。
- 两个矩阵的实现、统一环境、输入资产身份及 target-plan seal 一致；125 个 baseline 与
  15 个 LLM-SR 完整分块封存逐一通过核验，记录启动前 2166 个首次终态 SHA256。
  模型四个权重分片、tokenizer/config 与离线依赖锁均通过 SHA256 核验。
- 迁移仅调整独立调度器的实际 CPU 编号和 GPU UUID，冻结方法代码、参数、种子与预算不变。
  保持 slots-v3 相同硬件、每槽四核和 28 GB RSS、GPU 独占，以及原首次终态不可覆盖规则。
  vLLM 服务保留数值设备编号，实验资源绑定实际 UUID；服务启动后的真实请求返回 HTTP 200。
  新服务树 RSS 为 2,964,721,664 bytes，两个下一分块的资源绑定 dry-run 通过；停止服务后显存 1 MiB。
- 09:55（台北）交替队列启动，保留旧队列的 next_chunk 顺序：`gpu-0033`，随后
  `gpu_llmsr-0015`；剩余 148 个 baseline 分块与 75 个 LLM-SR 分块。
  以新容器 PID 1 创建时间重新计算 44 小时截止，完整下一分块预算不足时保存状态并停止派发。
- 10:01（台北）复核首个新增终态：PhySO / Livermore-5 / seed4，`timeout`、attempt=1，
  已直接写入挂载，执行器继续下一目标。新结果记录实际四核绑定及 GPU UUID。
  启动前 2166 个首次终态文件及两个父矩阵声明的 SHA256 全部保持不变。

旧队列计划、状态、启动信息与驱动保留副本。接续证据位于持久化交替队列的
`resume_audits/20261012_42006/`；连接与操作信息仅记录在私有部署说明中。

## 持久化恢复与 LLM-SR 入队（2026-10-09）

用户授权恢复并续跑原实验，将 LLM-SR 加入队列；实验结果直接写入持久化挂载，
模型权重与运行环境保留在用户将保存的镜像内。

- 恢复 10 月 8 日 00:56 检查点：1718 个首次终态，SISSO 1425、DLGA 15、PhySO 192、
  SymbolicGPT 86。压缩包 SHA256、62,589 份文件、111 个完整分块封存，以及冻结代码、
  统一环境与输入资产身份全部通过。启动后复核原 1718 个首次终态哈希不变。
  原 4155 cases / 4290 targets 与矩阵 seal 保留，错误和超时不重跑。
- 原始 BF16 `meta-llama/Llama-3.1-8B-Instruct` 的四个权重分片 SHA256 与 HF 元数据一致，
  tokenizer/config 同时绑定；独立 vLLM 0.10.2 / torch 2.8.0+cu128 环境离线安装并通过依赖检查。
  原统一环境保留 torch 2.5.1+cu121，未将隔离服务依赖安装到其中。
- 四次真实 LLM-SR 调用全部完成、无需重试，四个程序均通过解析和求值；
  独立合成任务的训练 MSE 为 0.06236，仅证明部署链路，不作为精确恢复验收。
  HTTP 断开前观测到 1 个运行任务，断开后约 0.29 秒清零，下一请求返回 HTTP 200。
  服务与验收客户端合计 RSS 峰值 3,130,527,744 bytes，均绑定 GPU 槽的四个 CPU 编号。
- 关闭 vLLM 后 GPU 显存回到 1 MiB；重启并更换服务 PID 后，同一冻结 LLM-SR 分块的
  D2 dry-run 通过，确认科学身份与运行时进程绑定分离。
- LLM-SR 独立 main 矩阵：1380 cases / 1425 targets、5 seeds、T=300 秒、软截止、
  1000 调用内部预算、4 evaluators、temperature=0.8、max_tokens=1024。
  两项独立 SR/ODE 数据准备与原比较矩阵的目标 metadata 完全一致，其余成功声明按相同数据输入复用。
  90 个 LLM-SR 分块与剩余 162 个 baseline GPU 分块交替，21:04（台北）已启动。
  首个 LLM-SR target 已保存 `ok`，`stop_reason=time_limit`、`stop_reason_source=fit_deadline`，
  fit=300.0135 秒、score=1.1219 秒；含服务 RSS 峰值 4196.46 MiB，资源超限与监控错误均为空。
  队列已继续下一 target；所有阶段记录直接保存在挂载内。

恢复、服务验收、重启检查、冻结矩阵与调度记录均保留在服务器持久化实验目录。
私有路径和操作入口见 `.local/server_eitech_20261005/README-full-main.md`；
严格 1000 调用无 T 对照及 FDE 诊断不混入此次主矩阵。

## FDE 全部 PDE 方法覆盖（2026-10-08）

新增 5 个 FDE 时空场的 `_pde` 网格接口，其中 3 个内置坐标场可直接运行。
`configs/benchmark/fde_pde.json` 包含全部 13 个 PDE 方法，共 39 cases；
`fde_all_methods.json` 合并原 50 个响应拟合 cases，共 89 cases。
科学任务为整数阶 PDE 残差诊断，分数阶算子和恢复仍待来源认证。
详细协议、导出和运行方式见 [FDE 说明](fde_dataset.md#全部-pde-方法覆盖)。

- 最终相关回归 362 passed、0 failed、0 skipped，149.46 秒；启用
  `--require-external-data`，覆盖响应/PDE 接口、真实资产、划分隔离、原生稀疏/弱形式/SGA 拟合、
  SGA 系数与微分导出、原 runner、加载器、domain、Green/closure、冻结、方程协议及规划。
  证据 `results/validation/fde_pde_final_regression_20261008.xml`。
- 全部 39 组合的实际兼容性检查：36 compatible、3 incompatible、0 error。
  不适用项全部为 PDE-Net 的二维空间要求，目标保留 `method_unsupported`；
  证据 `results/validation/fde_pde_compatibility_20261008/compatibility_report.json`。
- 全部 39 组合的原生入口 smoke：拟合预算 8 秒、宽限 1 秒，33 `ok`、3 `skipped`、
  3 `timeout`、0 `error`；证据 `results/validation/fde_pde_smoke_20261008/benchmark_summary.json`。
  此次记录用于验证方法入口及终态分母；其评分先于首帧与 SGA 导出修订，不作为最终评分证据。
- 评分修订后的 DSCV / SGA / PDE-FIND 真实 CLI：9 cases / 9 targets 全部返回，
  8 个残差有限、1 个 SGA 候选为 `nonfinite_candidate`，覆盖率为 0，不授予误差分数。
  SGA 的另外两个场均成功导出带系数方程并评分。证据
  `results/validation/fde_pde_scoring_smoke_20261008/benchmark_summary.json`。
- EqGPT 单独采用默认 90 秒拟合预算仍触发 `hard_timeout`，phase 为 fit，
  保留缺少候选的终态与分母；证据 `results/validation/fde_pde_eqgpt_smoke_20261008/`。
  数据加载及模型导入兼容，尚未验证此 CPU 预算内的有效 EqGPT 拟合。
- 展开库存见 `results/validation/fde_pde_inventory_20261008/`。响应/PDE 接口沿用同一物理家族与文件，
  原有 455 项资产哈希无变化。默认矩阵仍为 3311 cases，四方法主配置仍为 4155 cases，
  服务器冻结队列未修改。Ruff 与差异空白检查通过。

## FDE 响应拟合接入（2026-10-08）

按用户选择接入观测响应拟合，科学任务为 `fractional_equation_response`，数值调度
为 `sr`，使用 `fde-response-v1` / 指标 2.3。5 个记录可直接运行；另 2 个 DAT 条目
需要显式坐标及其来源，缺省保留 `source_blocked` 目标。重复资产不新增独立拟合。
接口、划分与运行方式见 [FDE 说明](fde_dataset.md)。

- 相关回归 281 passed、0 failed，124.18 秒；开启 `--require-external-data`。
  覆盖 FDE、统一加载、既有 benchmark、domain 协议、Green/closure 与冻结报告；
  证据 `results/validation/fde_response_regression_20261008.xml`。Ruff 与差异空白检查通过。
- gplearn 真实 smoke：5 cases / 5 targets 全部 `ok`，每目标拟合上限 10 秒，
  使用 smoke 参数。全部预测有限，时间块无交叉，恢复资格均为 false；
  证据 `results/validation/fde_response_smoke_20261008/benchmark_summary.json`。
  此结果验证执行链路；预测误差保留在结果中，不作为精度校准或方程恢复结论。
- 7 个注册名的实际兼容性检查：5 compatible、2 source_blocked；
  证据 `results/validation/fde_response_compatibility_20261008/compatibility_report.json`。
- 库存新增 12 项 FDE 资产；此前资产无缺失、SHA256 无变化。
  原 ZIP 与全部 11 个提取文件逐字节复核通过；
  资产变化及展开库存见 `results/validation/fde_response_inventory_20261008/`。
- 独立配置 `configs/benchmark/fde_response.json` 默认选择 5 个记录、10 个 SR 适配器，
  单 seed 共 50 个标量拟合，使用 smoke 参数。完整多方法实验尚未执行。
  FDE 为 opt-in；原四方法配置仍生成 4155 cases，本次未操作服务器冻结队列。

## 本地临时资料清理（2026-10-08）

按用户要求直接删除无用文件，不创建备份。共删除 17,876 份文件、约 1,235.04 MiB，
包括旧隔离测试环境、pip 缓存、wheel、10 月 7 日代码清理备份、旧挂载服务器部署包、
已完成任务的一次性脚本及重复解包的校准证据。校准证据的 213 个解包文件与保留归档逐一一致。

最新检查点的 62,589 份文件通过 SHA256 校验；39 份旧检查点中的全部结果文件都已被
最新检查点以相同哈希覆盖，因而删除旧归档，保留最近两份。
当前冻结源码、校准归档、环境锁、监控与续跑工具、本地统一环境配置和回归证据保留。
未修改服务器、实验源码或参数；本轮未重跑代码回归。

清理前检查已未发现本地备份助手进程；最近已保存检查点时间为 10 月 8 日 00:56（台北）。
本轮没有启动助手或创建新检查点，后续实验迁移以现有最新检查点为准。

## 本地代码整理回归（2026-10-07）

- 移除 56 个无现用引用的旧脚本、上游独立测试和 TensorFlow 遗留文件；
  DLGA 测试另整理为 `tests/test_kd_dlga.py`，将初始化断言并入已有元数据测试。
  Python 文件数 306 → 269，净减少 5,887 行；范围见[工程记录](#工程记录)。
- 本地 Python 3.10 完整回归：615 passed、0 failed、6 skipped，566.63 秒；
  `results/validation/code_cleanup_20261007.xml`。开启 `--require-external-data`，
  未因测试数据缺失跳过；6 项跳过均为本地没有 SISSO++ 原生绑定的集成测试。
  此前 Linux 6 项 SISSO 原生验收仍以原冻结源码的证据为准。
- 合并差分函数及精简常数优化器的 20 组输入与原实现完全一致；
  公共调度与 21 个报告文件一致（进度更新时间除外）。修改后的入口、公共模块和
  DLGA 测试通过 Ruff 与差异空白检查。
- 新 wheel 不含已移除文件，运行时模块、模型配置、词表、手册和 aTEAM 许可证均在包内；
  从独立解包目录导入 PDE-Net、DISCOVER、DLGA、调度辅助模块成功，
  DISCOVER 构造、DSO 默认配置和 EqGPT 手册读取通过。
- 服务器主实验、冻结源码、首次终态和周期备份不受本地整理影响；
  未来主实验续跑仍使用原冻结快照。

## 四方法校准、回归与全量队列（2026-10-05）

- Linux 完整回归：620 passed、0 failed、2 Windows-only skipped，385.11 秒；
  `results/validation/four_methods_linux_full.xml`。两项 Windows 专属检查在本地补跑 2 passed。
- 独立无噪声开发数据：PhySO/SISSO/SymbolicGPT 各 6/6，DLGA 3/3；
  每个方法覆盖 seed 0、1、2，所有通过项 exact structure=1、relative coefficient L2 <1e-5。
  `results/calibration/four_methods_20261005/gate.json` 保留试验来源、统一参数和全部试验 JSON 哈希；
  失败历史、CPU 诊断、资源重叠记录独立保存，没有覆盖为成功结果。
- 新配置准入审计无错误：268 SR、8 Core ODE、3 PDE，5 seeds、T=300、grace=5；
  4155 cases / 4290 declared fits。源码和参数冻结后再开始主拟合。
- 新增 slots-v3 允许仅物理 CPU 编号/GPU UUID 的迁移，仍严格核验四核/28GB、
  CPU 型号、GPU 型号/显存/驱动、代码、环境与资产；旧 slots-v1/v2 身份保留。
  本地实际资源与续跑专项 54 passed，新增三个可移植身份/硬件/首结果测试全部通过，
  Linux 真实四核 CPU/GPU 硬件核验及上述完整回归通过。
- 服务器流水线与独立源码快照位于 `/opt/kd` 内部，完整矩阵先准备并封存真实分母，
  主队列按不超过 6 小时的保守阶段预算分片，容器安全余量内只暂停排队；容器由用户保存。
  所有 fit 均独立 worker 重新加载数据；共享分母准备成本记在矩阵级，不充作方法训练成本。
  固定开始顺序为 Nguyen-1/seed0 的三个 SR 方法及 burgers/seed0 的 DLGA，随后继续完整矩阵，
  不基于主实验结果修改参数。完成后的分片按同一父矩阵合并，未运行项始终保留分母。

参数、适用范围与保真度限制见 [校准说明](methods.md#four-method-calibration-variants-2026-10-05)。
流水线仍在执行，当前没有完整主实验性能结论。

首批真实四方法分片合并与冻结报告入口核验通过：4 首次终态、完整 4290 target 分母、
4286 missing，原首次终态 SHA256 不变。精简证据为
`results/validation/four_methods_main_initial.json`，不能作为四方法性能排名。
首批 Nguyen-1/seed0：PhySO fit 硬超时；SISSO 结构恢复为 0；SymbolicGPT 测试 NRMSE
1.5716e-7，但额外微小常数项令严格结构恢复为 0。DLGA 的 burgers/seed0 支持恢复为 0、
相对系数误差 0.2658。开发数据校准通过并不保证主实验无噪声数据都能恢复；
原表达式与失败记录保留，没有删除小系数、放宽指标或根据这些结果修改冻结参数。

10 月 6 日 00:01（台北）的实际运行核验：365/4290 首次终态，SISSO 357、PhySO 5、
DLGA 2、SymbolicGPT 1；源码指纹与完整 target-plan seal 均未改变，单个调度器、
CPU/GPU 各一个 worker 且分别绑定四核，本地隐藏备份助手仍运行。
`results/validation/four_methods_main_health.json` 保存本次核验，不是最终性能报告。

10 月 7 日迁移至 42003 容器：环境镜像不含实验目录，从 10 月 6 日 15:19（台北）
的最后有效本地检查点恢复 1658 个首次终态（SISSO 1425、DLGA 15、PhySO 132、
SymbolicGPT 86）。61788 份文件逐一通过 SHA256，107 个分片封存索引通过核验；
冻结源码、Python/原生依赖、输入资产和实际 CPU/GPU 硬件身份全部一致。
slots-v3 只重新绑定本容器实际 CPU 编号与 GPU UUID，参数、时间预算、槽位与调度器字节不变。
旧端口主机身份已变化，备份之后未保存的进度无法取回；用户明确同意从现有检查点继续。
恢复审计记录为 `results/validation/four_methods_resume_42003.json`，
续跑与旧结果哈希保留核验为 `four_methods_resume_check_42003.json`。
正式矩阵仍保留 4290 target 分母，尚未完成；本轮没有把旧结果失败改为成功或根据结果调参。

## SISSO 与 Linux 代码回归（2026-10-05）

当前本地代码（包括未提交进度）已同步到配置好的服务器容器。原代码先备份，
项目文件及本轮所需的 320 个外部测试资产逐文件核对 SHA256；工作与证据均在容器内部。

| 检查 | 结果 |
| --- | --- |
| Linux 完整 pytest，强制要求 SISSO 和外部测试数据 | 607 passed，0 failed，2 skipped；385.82 秒 |
| 两项 Linux 跳过的 Windows 专属检查 | 本地补跑 2 passed |
| 本地核心 CI 测试集合 | 56 passed |
| SISSO adapter 契约与原生集成（包含在完整回归内） | 9 项契约、6 项原生检查全部通过 |
| 独立 SISSO CLI（Koza-2、damped oscillator） | 2 cases / 3 targets 全部 ok |
| 24 点线性 SISSO adapter 拟合 | RMSE `1.7554167342883506e-16` |
| 新增 SISSO 代码及核心 CI 静态检查 | Ruff 0.12.12 通过 |

证据：`results/validation/sisso_linux_full.xml`、`sisso_acceptance.json`，
私有详细日志保留在本地与服务器证据目录。Windows 专属检查是原子文件替换共享锁重试和
官方 E2E checkpoint 路径映射；没有因缺少 SISSO 或外部测试数据而跳过。

修复了 SISSO++ v1.2.6 的固定零截距预测偏移，输出公式代入实际系数并与原生特征预测一致。
原生检查覆盖线性、平方、双描述符 full 配置、SR/ODE worker 和 fit 硬超时。
完整回归也修复了依赖 Windows 进程启动耗时的 LLM-SR 超时夹具、Linux 僵尸回收竞态及
过期的公共 SINDy 配置/矩阵数量断言。新增方法的范围、预算及限制见 [SISSO](sisso.md)。
本轮未启动正式全量实验，也没有验收真实 vLLM 服务或完整公平资源部署协议。

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
环境验证归档，以及无终态的工作目录。该次临时清理清单
`.local/experiment_cleanup_20260930.json` 已于 2026-10-08 按用户要求删除。

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

## 工程记录

`results/validation/`、`results/calibration/` 下的证据目录已于 2026-10-20 删除。本文及其他文档提到的 commit `718588b` 是本地归档标签 `archive/pre-publish-20261020` 中的中间提交，未推送到 GitHub；已删除的原始文档只在本地可取回。

- **四方法校准（2026-10-05）**：独立无噪声合成开发数据，PhySO/SISSO/SymbolicGPT 各 6/6、DLGA 3/3 通过，冻结 `configs/benchmark/four_methods_full.json`。仍生效的参数与边界见[方法卡片](methods.md#four-method-calibration-variants-2026-10-05)；校准结果不作为主实验性能结论。
- **代码整理（2026-10-07）**：移除 56 个无现用引用的旧文件（PDE-Net 独立训练/调试脚本及 aTEAM 上游测试、DISCOVER 的 TensorFlow 组件与无调用方 dataset、绑定旧结果路径的 hpc 队列脚本）。公共调度、进度与分方法报告提取到 `kd/benchmark_batch.py`；`load_config_new` 为 `load_config` 的兼容别名；保留 PDE-Net 运行时导入的 `aTEAM/pdetools/example/` 及其许可证。Python 文件 306 → 269；服务器主实验继续使用冻结快照 `/opt/kd/releases/four_methods_20261005_v1`。临时恢复副本已于 10-08 删除。
- **工程计划（2026-10-11，10-14 修订）**：P1-1～P1-4（任务能力、实际输入身份、SGA 方程导出、全实例预检与资产角色）完成，见[数据接入修复](data.md#domain-v5-接入修复2026-10-11)。封存与 summary snapshot 改为原子发布，复用已有 snapshot 前核对 hash 与名称，只覆盖进程中断边界，不等于断电持久化；显式恢复工具仅在出现实际恢复需求时追加。核心 CI 补入 4 个现有测试入口和 9 个模块的 Ruff，本地隔离环境 153 passed，未取得远程 CI 结果。summary 成本三组真实输入重放中位数 0.13–0.28 秒，证据不足以支持节流，暂不优化。来源认证与原生短拟合为独立条件分支，未启动。
- **工程精简（2026-10-13）**：删除 `_failure` 中被覆盖的能力状态赋值和冻结报告的重复 hash；PDEFind 活动配置与示例移除 `alpha/max_iter`（显式传入仍接受并警告，求解算法不变）；配置 CLI 测试替换环境指纹扫描，共用 helper 移入 `tests/benchmark_helpers.py`；PDE 准备与切片移入 `kd/benchmark_preparation.py`，预检和执行共用 `prepare_views` 与导数 stencil 规则，统一指标版本推导与执行策略 hash。相关回归 512 passed。源码与 PDEFind 配置变化，新运行使用新输出身份。
- **results 清理（2026-10-20）**：按用户要求，先将 smoke、dry-run、preflight 产物（1,812 个文件，23.12 MiB）移入 `.local/results_cleanup_20261020/quarantine/`；随后同日删除 `results/` 下其余内容，只保留 `results/analysis/{combined_results_20261017,cloud_results_20261016,pde_extension_20261017}/` 和从根目录移入的 `results/knowledge_discover_results_20261016_summary.tar.gz`。隔离目录已不存在。未修改函数、实验配置或评价协议，未改写历史 smoke 的成功或失败记录。
- **同步冲突清理（2026-10-20）**：73 个未跟踪的 `-Solaris` 文件均有对应无后缀文件（71 个内容不同）；原字节归档在 `.local/sync_cleanup_20261020/`。该次清理实际把较新的 `*-Solaris` 副本隔离掉了；同日恢复 69 个文件。OneDrive 同时把 `.git` 中的本地 main 退回到 1 月的 `bc404ed`，真实历史为 GitHub 上的 `741b837`；清理结果最终作为一次提交接在 `741b837` 之后。`kd/benchmark_identity.py:source_fingerprint` 递归纳入 `kd/` 等目录，文件变化会改变新运行的源码身份；继续旧批次须使用与旧身份一致的冻结源码，不能绕过身份检查。
