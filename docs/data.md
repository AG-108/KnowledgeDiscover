# Data and external assets

当前工程验收见 [验收摘要](validation.md)，来源条件见 [待解决问题](deferred_source_issues.md)。本地开发树数据协议 `domain-v5`；服务器冻结实验仍为 `domain-v4`，两者不混用。新增轨指标 2.3；本地修复范围见[domain-v5 接入修复](#domain-v5-接入修复2026-10-11)。

The public checkout includes the existing small fixtures in `kd/dataset/data/`,
the SR benchmark definitions, generated Core ODE systems and the PDEformer sinus
generator. These support bounded examples without the full experiment collection.
Catalog registration does not mean an external file is present.

The [asset manifest](asset_manifest.json) records local assets and byte aliases.
Current physical task types and implementation boundaries are summarized in
[validation](validation.md); TLC Cartesian fields are restored without interpolation,
and Green operators use integral-response discovery with whole-forcing holdout.

Data protocol `domain-v5` retains the `domain-v4` separation of physical task labels from numerical dispatch.
CYT uses spatial-block holdouts. Benchmark solid tasks expand source curves or
material/study identifiers into separate instances and hold out strain rates;
the low-level solid loaders retain pooled raw tables for inspection. These raw
tables must not be fitted as a single unidentified material law. Hardening's
pre-matched DIF stays in raw data but is excluded from benchmark predictors.

Export current and pending tasks, source availability and expanded fit counts:

~~~bash
python scripts/export_dataset_inventory.py --expand-instances --output-dir results/dataset_inventory
~~~

The JSON/CSV ledger includes opt-in Green/file SR/pyqg, blocked source identities and PDE geometry, duplicates and training assets. Optional `--sha256` hashes the
referenced files. Counts include unsupported combinations and are not a frozen
formal experiment plan.

## Optional data layout

Paths below are relative to `kd/dataset/`. Obtain the corresponding original
research data or a verified experiment archive and restore this exact layout.
Keep source citations, original licenses and checksums with downloaded assets.
Loaders do not automatically download or fabricate replacements.

| Local path | Used by |
| --- | --- |
| `discovery-of-physics-from-data/data/Ball_drops_data.xls` and `balls.txt` in the same directory | ball_drop |
| `Discovery_of_soild_consititutive/data/data_rubber/{train,test}/*.xlsx` | rubber_train / rubber_test |
| `Discovery_of_soild_consititutive/data/data_DIF/*.xlsx` | solid_dif |
| `Discovery_of_soild_consititutive/data/data_strain_stress/*.xlsx` | solid_strain_stress |
| `Discovery_of_soild_consititutive/data/saved_data_hardening_strain_rate/*.pkl` | solid_hardening |
| `CYT/{FlatPlate_lk0.215andPplus,NACA0012_Re4e5_MLen_BEST}/Output/FlowFeature.dat` | cyt_flatplate / cyt_naca0012 |
| `ViscousGravityCurrent/data/vgs_{I,II}_<window>.dat` | Four VGS windows; exact names in `kd.dataset._base._VGS_CASE_CONFIG` |
| `TLC/<family>/*.csv` | Time-series cases declared in `kd.dataset._catalog` |
| `WDwake/TI8_U.npy` and `TI8_V.npy` in the same directory | wdwake |
| `WaveBreaking.pkl` | wave_breaking instances |
| `Discover_Green_function/dso/task/pde/{data,data2}/*.mat` | 23 opt-in Green operator names in 6 families via green_dso / green_basis; two data2 byte variants remain unregistered pending source identity |
| `Discover_Green_function/dso/task/pde/data/*.nc` | Opt-in pyqg_subgrid offline two-layer closure regression; run0 train / run1 test; online stability unverified |
| `EqGPT/data/` | Mixed evolution/steady PDE assets; some duplicate registered fixtures, others unregistered |
| `FDE/data/*.{mat,dat}` and `FDE/reference/fade_plot_3d.py` | Opt-in FDE: five response records; the same three space-time fields also cover all 13 PDE adapters. Two DAT fields require explicit coordinates. See [FDE notes](fde_dataset.md) |

Ball-drop data accompanies *Discovery of Physics from Data: Universal Laws and
Discrepancies*. The solid-data snapshot accompanies *Beyond empirical models:
Discovering constitutive laws in solids with graph-based equation discovery*;
its local snapshot retains the original license. CYT, TLC, VGS, WDwake and
WaveBreaking source URLs/revisions are not yet recorded consistently. Obtain a
verified source/archive before claiming a fully reproducible public data release.
This checkout does not provide download URLs for those unverified snapshots.

`DeepSymbolicOptimization/`, `Discover_Green_function/`, `EqGPT/` and
`SymbolicGPT/` under `kd/dataset/` are optional upstream snapshots. Maintained
model ports live in `kd/model/` and do not require these source snapshots at runtime.
However, the snapshots also contain scientific data: Green has 26 MAT and two
NetCDF files; EqGPT includes unregistered evolution and steady PDE fields; DSO
includes additional file-based regression candidates. These must not be dismissed
as code-only assets. Duplicates, parameter/noise variants and training corpora
must be distinguished from independent test tasks; see the audit. Packaged SR
benchmark definitions remain included. Optional snapshots are not a guarantee of
public data availability or verified redistribution rights.

## Availability checks and tests

~~~bash
python run_benchmark.py --list
python run_benchmark.py --dry-run
python -m pytest -q -rs
~~~

External-data integration tests have explicit `external_data` markers. Missing
assets yield a skip with the required path; present but invalid data still fails
its real assertions. A full-data workstation can require every asset:

~~~bash
python -m pytest -q --require-external-data
~~~

A fresh-clone run with external-data skips is not full dataset validation.
No-fit compatibility and fitting results must be reported separately. Repository
cleanup preserves external data, final experiment results and unfinished worker data.

## Checkpoints and private settings

E2E needs a separately obtained source checkout and trusted checkpoint with a
verified SHA256; see [the E2E card](methods.md#e2e-official-pretrained-adapter). Keep assets under
`external/` and `checkpoints/` or outside the repository. Store credentials in
local environment variables or ignored `.env` files. Raw results, model binaries
and archives are excluded from commits.

A1/A2 来源、资格、同源观测与协议迁移见 [验收记录](validation.md)；
旧 domain-v1/v2/v3/v4 与本地 domain-v5 结果不混用，来源不足尚未正式就绪。

## domain-v5 接入修复（2026-10-11）

本地 `domain-v5` / `benchmark-identity-v2` 的修复不替换服务器冻结实验；runner 修复后的运行使用新输出身份，`domain-v4` 旧结果只作历史记录读取。

- `kd/benchmark_capabilities.py` 集中声明 PDE 适配器支持的目标形式、时间阶数、空间维数、通道、网格与边界要求；预检、目标准备和实际执行共用同一检查。
- Wave / Klein–Gordon 使用 `pdefind_nd` 的 `u_tt` 目标，implicit_mixed 使用 `implicit_pde`；旧入口返回 `method_unsupported`。这是 runner 适配限制，不能据此判断 DLGA、EqGPT 原生方法没有二阶能力。
- 失败目标保留声明、元数据、真值和恢复资格。普通旧 PDE 入口以一个 `PDE` 作为预算单位，方程声明另存为 `equation_targets`；高级入口独立执行各方程目标，预检与执行使用相同目标身份。
- 普通 SGA PDE 与 FDE 导出带拟合系数、默认项和生成项的 RHS，原生树展示文本单独保留；导出失败标为错误并停止数值评分。
- PDE-Net 保留原始网格与已验证边界元数据，使用实际各轴间距，拒绝将辅助压力场自动当作演化目标。
- 高级 PDE 短时序切分至少保留 5 帧训练、3 帧测试和 2 帧间隔；不为短记录降低支持量要求。
- 完整预检模式（`--check-compatibility --compatibility-mode full`，PDE/稳态范围见 `configs/benchmark/dataset_preflight.json`）逐 seed、实例、目标记录加载、准备、方法能力、实例资格与运行依赖。兼容只说明这些检查通过，`fit_evidence=not_run`，不是原生拟合成功的证据。
- 固体 seed 0 的 5 个实例记为 `instance_ineligible` 并保留在目标分母中：DIF `Lin(HRB335)_2016_u`（训练目标恒定）；strain_stress 与 hardening 的 Cadoni core/peripheral 各 2 个（应变率组不足）。

输入身份：合成 SR 的 `function_sets.csv`、rubber_train 的官方测试 Excel、WaveBreaking 使用的 `examples/_common.py`、EqGPT 词表和手册均进入指纹；语法表与 DSO 固定提交 `8348d5b08d1eef6170fdbfd222a492ded990ea12` 的对应文件逐字节一致。外部数据根目录和自定义公式表按实际读取文件绑定；同内容路径迁移可恢复，不同根目录的同名文件不会相互覆盖。自定义公式内容与认证定义不同时，取消继承的来源认证和参考真值，主轨准入拒绝该输入。

资产角色：Diffusion 原始文本登记为待接入；20 对 DSO 固定 CSV 登记为固定样本变体，Nguyen-1 与 Test-1 的别名关系保留，均不增加独立拟合权重。资产分类另区分语法、定义、模型、配置、图片和上游结果。清单含 455 份可用资产，没有未归属的观测资产；“已归属”包含有阻塞原因的待接入条目，不表示可执行或来源已认证。

未解除的来源条件见 [deferred_source_issues.md](deferred_source_issues.md)；分数阶数据仍只提供响应回归与整数阶 PDE 诊断。后续先基于预检选择有资格的组合做原生短预算拟合，再安排新身份的正式实验。

## A3/A4 参数方程与二维对流扩散（2026-10-16）

本地 `domain-v5` 新增三个可执行 PDE 注册名；服务器已冻结矩阵不扩充。

| 注册名 | 原始布局与坐标 | 参考方程与来源边界 |
| --- | --- | --- |
| `parametric_burgers` | `u[t,x]` 为 200×251；`x=0.004*arange(251)`，`t=0.005*arange(200)` | `u_t=-u*u_x+0.02*exp(t)*u_xx`；初值 `(1.5-x)*sin(2*pi*x)`，两端零 Dirichlet。形状、初边值及数值残差与文献一致，尚未认证上游文件字节身份 |
| `parametric_convection_diffusion` | `u[t,x]` 为 250×201；`x=0.04*arange(201)`，`t=0.02*arange(250)` | `u_t=-(1+0.25*sin(x))*u_x+u_xx`；初值 `(8-x)*sin(x)`，两端零 Dirichlet。与 PIC 官方 `variable-GWFE.npy` 逐字节一致 |
| `advection_diffusion` | `Expression1` 按 C 顺序还原为 51×51×61×4，通道为 `x,y,t,u`；`x,y∈[-5,5]`，步长 0.2；`t∈[3,9]`，步长 0.1 | 与 DeePyMoD 官方数据逐字节一致；最高二阶空间导数为方法配置。生成系数与边界尚未认证，不将拟合系数写为真值 |

两个 NPY 的时间轴不含右端点。依据：[Adaptive DLGA-PDE，III.A.1–2](https://arxiv.org/pdf/2005.07916)；[PIC 数据，固定 revision](https://github.com/woshixuhao/PIC_code/blob/0230007c3457022b32b7f38b421ec0f9c817cd5d/data/Parametric_convection_diffusion/variable-GWFE.npy)；[DeePyMoD notebook，固定 revision](https://github.com/PhIMaL/DeePyMoD/blob/543c2e5fc7a6a5637be660775873724d884790ba/examples/PDE_2D_Advection-Diffusion.ipynb)。
SHA256、证据链接与未决项保存在 `_advanced.py`、`_core_sources.py`、`_provenance.py` 和资产清单中；加载时检查已核对文件的 SHA256，避免将既定坐标赋给其他观测。

```python
from kd.dataset import load_dataset

data = load_dataset("parametric_burgers")  # usol[channel,x,t]
data2d = load_dataset("advection_diffusion")  # usol[channel,x,y,t]
```

`configs/benchmark/parametric_advection.json` 对三项统一使用 `pdefind_nd`，候选系数基固定为 `x,t,sin(x),cos(x),exp(t)`，没有逐数据集方法覆盖。这是显式声明候选函数族的有限库诊断，不代表无需先验的系数函数发现。默认 baseline 的空坐标基保持原样；两个参数数据在未配置对应坐标基时返回 `method_unsupported`，旧 PDE 适配器也不实现其变系数发现。二维观测通过 `pdefind_nd`；旧一维适配器拒绝其空间维度，PDE-Net 所需周期边界没有来源依据。

训练与测试按时间块留出，先切分再求导，中间留两帧支撑间隔；时间系数使用原始绝对 `t`，测试段不重置时间。观测导数命名空间只含实际坐标、场和导数，参考方程不参与候选库构造或拟合，只用于独立数值诊断。三项均不声明认证结构恢复或前向 rollout，物理单位仍未知。2026-10-16 的短拟合中三项均报告候选库近零空间歧义 `ambiguous`，未按这些结果调整候选基或阈值。回归测试为 `tests/test_parametric_advection.py`。

同一协议下另注册 burgers2d、wave、klein_gordon、implicit_mixed；KG 副本不重复计权。
12 份稳态资产注册 steady_pde，经 SteadyPDEDataset 提供空间观测，不生成 t。
TLC 原始字段标签/单位与方程符号分列。非数值/外部填充值不自动删点，零不自动判域外。
逐文件哈希见 asset_manifest.json；坐标、几何与 source_blocked 条件见 deferred_source_issues.md。原 `results/validation/source_evidence.json.gz` 历史诊断已于 2026-10-20 随 `results/validation/` 删除。

## A5/A6 assets and API

See the [asset manifest](asset_manifest.json) and [source conditions](deferred_source_issues.md).
Green: 23 operator instances / 6 families; CSV: 25 candidates; pyqg: one two-layer system, run0 train / run1 test.
Optional xarray imports only in the pyqg loader. Unknown source, unknown license and missing asset are separate states.

The 18 names below were listed in `__all__` without bindings, so star import failed. They were removed from the public exports (2026-09-30) and not replaced with unproven look-alike data; select `load_dataset(name)` explicitly and check the source. Bound loaders and the unified entry point remain; `GreenOperatorDataset`, `ClosureDataset` and `FileRegressionDataset` were added.

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
| Parametric_convection_diffusion | `load_dataset('parametric_convection_diffusion')`；已接入，需显式变系数候选基，见[上节](#a3a4-参数方程与二维对流扩散2026-10-16) |
| Parametric_Burgers_equation | `load_dataset('parametric_burgers')`；已接入，需时间系数候选基；上游字节身份仍未认证 |
| Parametric_wave_equation | parametric_wave：核验系数基版本 |
| Burgers_2D | burgers2d：核验旧文件身份和轴顺序 |

Package / star import performs no external observation I/O, model loading or xarray import; see [the test](../tests/test_green_closure_protocols.py).
