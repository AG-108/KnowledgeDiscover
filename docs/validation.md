# A1–A6 工程验收

更新：2026-09-30。已完成可验证工程实现；来源不足项见 [待解决问题](deferred_source_issues.md)。
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

数据 `domain-v4`，方程 `equation-v1`，Green `green-v1`，闭合 `closure-v1`，执行 `target-v1`。
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
