# 已记录的来源问题

更新：2026-09-30。用户指示先记录、后续解决。下列问题不自动解除。

| 范围 | 未决问题 | 当前处理 |
| --- | --- | --- |
| A1 CYT | 单位、缩放、特征导出及几何身份 | 拒绝未验证泄漏特征；不声明跨几何 OOD |
| A1 固体／橡胶 | 应力／温度单位、跨研究材料身份、compound 编码 | 使用源研究／曲线身份，不声明跨材料泛化 |
| A1 VGS／WaveBreaking | 仿真/实验尺度、几何/边界、记录独立性与投影误差基准 | 保留局部窗口/帧划分，不认证跨系统 OOD |
| A2 Heat／compound | Heat 系数真值；compound 源符号冲突 | 不计认证恢复，不调阈值放行 |
| A3 WDwake／TLC | 物理坐标、边界、六份散点几何与界面 | source_blocked |
| A3 Beam／裸 NPY | Beam 阶数／系数；两个参数 NPY 坐标 | source_blocked |
| A4 TLC／Smile／EITech | 九份稳态几何；两个掩码／非数值域及来源约束 | source_blocked |
| A5 periodic／jump | 原 MAT 无 ExactGreen；plot_Green.py 发现结果不是参考真值 | 留出响应拟合可运行；参考核评分 abstain |
| A5 两份 data2 同名 MAT | negative_helmholtz、periodic_helmholtz 字节不同，来源未知 | 不注册、不覆盖、不增加家族分母 |
| A5 Green 来源 | U_hom 边界生成记录、单位、噪声尺度、forcing 物理身份 | 固定供应 U_hom；家族列序共享划分，存 F 哈希；不认证物理等价 |
| A6 pyqg | 生成器版本、精确 filter cutoff、在线稳定性 | 按 NetCDF stored forcing 符号做离线闭合；在线未评估 |
| A6 外部资产 | 数据许可证／可再分发依据 | manifest 记录 unknown；不用代码许可替代数据许可 |
| A6 预训练 | E2E 原语料；SymbolicGPT 精确结构交集；EqGPT 系数／观测重叠 | 分开记录结构暴露与观测重叠；不下载权重 |

证据：[资产清单](asset_manifest.json) 与 [验收摘要](validation.md)；逐文件历史诊断压缩保存在本地 results/validation/source_evidence.json.gz。
正式实验仍须 C3/C5–C7、选定来源条件、预算与矩阵冻结。
