# 代码整理（2026-10-07）

本轮整理针对当前本地开发树。服务器四方法主实验继续使用
`/opt/kd/releases/four_methods_20261005_v1` 的冻结源码，后续迁移仍从原快照恢复。

## 移除的文件

共移除 56 个旧文件；DLGA 测试另由 `kd_dlga_test.py` 整理为
`tests/test_kd_dlga.py`。删除依据是仓库内导入、配置及入口引用核对。

| 范围 | 清理内容 | 现用入口 |
| --- | --- | --- |
| `kd/model/PDENet2` | 独立训练、测试、系数比较、调试、方程专用 SLURM 脚本及 aTEAM 上游测试 | `kd/model/kd_pdenet.py`、统一 benchmark runner |
| `kd/model/discover` | TensorFlow controller、state manager、attention、trainer、统计日志与方差实验 | PyTorch `controller.py`、`state_manager.py`、`searcher.py` |
| `kd/model/discover` | 无调用方的旧 dataset 与未注册的 `wpde_pinn.py` 副本 | `kd/data.py`、`task/pde/weak_form.py` |
| `hpc` | 三个绑定 2026-09-25 结果路径与旧预算的临时队列脚本 | `scripts/run_slots.py`、`scripts/run_parallel_benchmark.py` |

PDE-Net 的模型、初始化、损失、优化器和数据生成依赖均保留，
`aTEAM/pdetools/example/` 实际由运行时导入。上游来源、引用与 aTEAM 许可证保留，
README 已改为当前适配器的使用说明。

## 合并的实现

- 公共调度、进度与分方法报告提取到 `kd/benchmark_batch.py`，两个 runner 共享该模块。
- DISCOVER 的 `utils_v1.py` 重导出 `utils_fd.py` 中 10 个相同函数，
  保留不同的三维边界实现和原导入路径。
- `load_config_new` 改为 `load_config` 的兼容别名；移除重复函数体。
- DISCOVER 常数优化器移除未注册且有未定义变量的 `BfgsMinimize` 及注释掉的旧实现，
  保留当前 BFGS 行为。
- DLGA 测试使用实际的小网络，移除模拟网络、手工目录恢复和重复的初始化测试；
  初始化断言并入元数据测试，基因生成与变异参数化，系数精度和双精度检查保留。

本轮 Python 文件数从 306 降至 269，净减少 5,887 行。
DSO 与 DISCOVER 仍分别保留自身程序状态、奖励和约束实现；局部代码相似不代表方法行为相同。
实验配置、原始数据、预训练资产与历史结果保持现有内容。

安装包检查同时修复了 DISCOVER 的 DSO 辅助目录缺少包初始化文件的问题，
并补充模型所需配置、词表、DSO 表格、EqGPT 手册及 aTEAM 许可证的打包声明。
旧生成目录 `build/` 移入本地备份后重新构建，防止已删除的旧文件进入 wheel。

## 验证与后续清理

20 组差分和常数优化器输入与清理前逐项完全一致；公共调度结果及 21 个报告文件
与原实现一致（进度更新时间除外）。完整代码回归结果见 [验收记录](validation.md)。

清理时曾在 `.local/code_cleanup_20261007/` 保存开发树快照、删除文件备份和临时验证产物。
2026-10-08 按用户要求直接删除了整个临时目录，没有再次备份；这些恢复副本已不再保留。
回归及精简验收证据仍位于 `results/validation/code_cleanup_20261007.xml` 和同名 JSON。
