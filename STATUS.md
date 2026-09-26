# Project handoff — 2026-09-26

## Project goal

建立可复现的 SR / ODE / PDE discovery benchmark，统一运行与评估，保留方法变体、
适用范围、失败原因和统计分母。当前代码与结果文件优先于历史交接文字。

## Current task

本轮公开仓库整理已完成：合并冗余文档、隔离本地数据/私有配置，并验证可发布的代码快照。
同步目标为 origin/main；提交与远端状态使用 git log、git status 和远端引用核对。
整理前的文件与 diff 已保存在忽略目录 `.local/repository_cleanup_20260926/`。
服务器地址、SSH key 路径和旧环境导出留在本地备份；公开脚本使用环境变量。
## Verification — 2026-09-26

- 完整数据工作区：302 passed，0 failed / 0 skipped；使用 --require-external-data。
- 从暂存区导出的公开快照：275 passed / 27 skipped，0 failed；跳过项明确依赖未发布的外部数据。
- 公开快照 CLI dry-run：core_v2 为 70 cases，core_ode_track 为 440 cases；不代表实际训练完成。
- Sphinx 严格构建（-W --keep-going）通过，0 warnings；文档相对链接检查通过。
- Shell 语法、服务器/CPU 激活脚本隔离检查、新增测试文件 Ruff 和暂存区 diff --check 通过。
- 当前公开文件与待推送历史未发现所检查的私钥/API token 模式；外部数据、结果及私有配置被忽略。
- 修复 CYT 非法列名先读取外部文件的问题；参数校验先于数据读取，并有回归保护。
- 依赖清单移除本机构建路径，补齐已验证环境中的运行依赖；pywin32 仅在 Windows 安装。
- 验证日志与 JUnit 位于 results/repository_cleanup_20260926/，保留初次失败及最终通过记录。

## Canonical documents

- [文档索引](docs/README.md)：指南与报告入口。
- [路线图](docs/benchmark_roadmap.md)：已实现能力、科学边界与待解决事项。
- [方法卡片](docs/benchmark_v2_method_cards.md)：WSINDy、Core ODE 与协议变体。
- [指标定义](docs/benchmark_metrics.md)：协议 2.1 与统计分母。
- [服务器设置](docs/server_setup.md)、[数据说明](docs/data.md)：复现所需本地资产。
- [历史验证](docs/reports/benchmark_validation_history.md)：旧版本测试和实验记录。

## Last observed server state

以下为 **2026-09-25 22:25 UTC+08:00** 的历史观测，不代表当前实时状态：

- CPU PDE：23/23 cases 已于 19:21:55 结束；99 target rows，19 error/timeout rows。
- PhySO：正在处理 213/287；范围为 274 SR + 13 ODE，seed 0、20 epochs、CUDA。
- GPU 缺项队列：等待 PhySO 后依次运行 SymbolicGPT 150 和 DSO 269 cases。
- 原始目录、预算和 manifest 路径见
  [续跑报告](docs/reports/server_missing_pair_runs_20260925.md)。重查日志后才能更新完成状态。

## Confirmed implementation and open issues

- SymbolicGPT 默认逐 fit 从头训练；显式 `reuse_pretraining_within_case=true` 才启用
  调用方持有的单 case/instance 缓存。缓存键包含配置、输入/conditioning 规模、设备和 dtype；
  每个 RHS 复制权重并恢复随机状态，候选采样/常数拟合仍独立进行。
- 默认 full 保持 3000 corpus / 50 epochs / 100 candidates、零 DataLoader workers。
  标准 CLI 的同设备/同 seed 300 秒对照中，reuse 完成 2/2 RHS（147.57 秒）；逐 fit
  完成 1/2，另一 RHS timeout。已评估结构恢复均为 0。详见
  [协议比较](docs/reports/symbolicgpt_protocol_comparison_20260925.md)。
- Core ODE 固定八个系统、11 方法、5 seeds，440-case 正式声明语法主榜尚未执行。
  当前 `configured_fallback` 不得混为 `declared`；协议冻结后再考虑正式运行。
- Robertson 非均匀梯度实现已验证，但 1% 噪声在极小时间间隔上显著放大导数误差。
  时间截断尚未获正式采用；噪声/稀疏主榜暂停，见
  [观测审计](docs/reports/robertson_observation_audit_20260925.md)。
- SGA SVD 不收敛、LLM-SR endpoint/model、E2E/预训练语料重叠审计仍待处理。
  PIC 排序校准未通过，物理可信度原型未完成专家验证。

## Next steps

1. 重查服务器三批任务，核对 manifest/case/target/status 分母，SHA256 校验下载归档。
   保留首次失败结果，定向补跑必须使用独立目录。
2. 按路线图处理候选函数协议、Robertson 观测/导数协议和 SGA 数值失败。
3. LLM-SR 配置真实 endpoint/model 后先做生成与常数拟合检查。

## Invariants

- 不覆盖旧实验、旧服务器工作树、用户现有改动或其他协议的结果。
- per-case timeout 覆盖该 model/dataset/seed 的全部 RHS；保留已完成 target。
- 不把缺失、解析/评估失败、timeout 或无真值样本从分母中隐藏。
- SR/ODE `exact_recovery` 比较忽略数值系数的结构；严格代数相等和系数误差单列。
- 不以测试误差充当训练似然；AIC/BIC 需要明确 training-MLE provenance。
- 不公开原始数据、模型权重、运行日志和私有连接信息；小型现有 fixtures 保持可用。
