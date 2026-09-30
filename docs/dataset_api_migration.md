# Dataset public API migration — A6

2026-09-30。以下 18 个名字此前仅列在 __all__，实际没有绑定，star import 会失败。
已从公开导出移除，没有用未经证明的相似数据替代。请显式选择 load_dataset(name) 并核对来源。
已绑定 loader 与统一入口保留；新增 GreenOperatorDataset、ClosureDataset、FileRegressionDataset。

| 移除名 | 迁移检查 |
| --- | --- |
| Burgers_equation_shock | burgers / burgers_sine：核验初边值和 shock 语义 |
| KdV_equation | kdv：核验历史文件身份 |
| KdV_equation_sine | kdv：sine 初值身份未证明 |
| Chaffee_Infante_equation | chafee-infante：旧名称文件身份未证明 |
| KG_equation | klein_gordon：二阶时间目标，核验旧源 |
| Allen_Cahn_equation | allen_cahn：核验来源版本 |
| Wave_equation | wave：二阶时间目标，核验旧源 |
| KS_equation | KS：确认原数据身份后选择 |
| Beam_equation | beam：来源阶数阻塞 |
| Heat_equation | heat：扩散系数真值未知 |
| Heat_equation_sin | heat：sin 初值身份未证明 |
| Convection_diffusion_equation_solution | convection_diffusion：solution 版本身份未证明 |
| Convection_diffusion_equation_simulation | convection_diffusion：simulation 版本身份未证明 |
| Diffusion_equation | 无证明等价的 canonical，不能替换成 Heat |
| Parametric_convection_diffusion | parametric_convection_diffusion：裸 NPY 坐标阻塞 |
| Parametric_Burgers_equation | parametric_burgers：裸 NPY 坐标阻塞 |
| Parametric_wave_equation | parametric_wave：核验系数基版本 |
| Burgers_2D | burgers2d：核验旧文件身份和轴顺序 |

包／star import 无外部观测 I/O、模型加载或 xarray 导入，见 [测试](../tests/test_green_closure_protocols.py)。
