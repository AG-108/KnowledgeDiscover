# Data and external assets

当前工程验收见 [验收摘要](validation.md)，来源条件见 [待解决问题](deferred_source_issues.md)。数据 domain-v4；新增轨指标 2.3。

The public checkout includes the existing small fixtures in `kd/dataset/data/`,
the SR benchmark definitions, generated Core ODE systems and the PDEformer sinus
generator. These support bounded examples without the full experiment collection.
Catalog registration does not mean an external file is present.

The [asset manifest](asset_manifest.json) records local assets and byte aliases.
Current physical task types and implementation boundaries are summarized in
[validation](validation.md); TLC Cartesian fields are restored without interpolation,
and Green operators use integral-response discovery with whole-forcing holdout.

Data protocol `domain-v4` separates physical task labels from numerical dispatch.
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
| `Discover_Green_function/dso/task/pde/{data,data2}/*.mat` | Green-kernel inverse problems; not integrated |
| `Discover_Green_function/dso/task/pde/data/*.nc` | pyqg subgrid closure fields; not integrated |
| `EqGPT/data/` | Mixed evolution/steady PDE assets; some duplicate registered fixtures, others unregistered |

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
旧 domain-v1/v2/v3 与新 domain-v4 结果不混用，来源不足尚未正式就绪。

## A3/A4 新接口

注册 burgers2d、wave、klein_gordon、implicit_mixed；KG 副本不重复计权。
12 份稳态资产注册 steady_pde，经 SteadyPDEDataset 提供空间观测，不生成 t。
TLC 原始字段标签/单位与方程符号分列。非数值/外部填充值不自动删点，零不自动判域外。
逐文件哈希见 asset_manifest.json；坐标、几何与 source_blocked 条件见 deferred_source_issues.md，详细历史诊断保存在 results/validation/source_evidence.json.gz。

## A5/A6 assets and API

See the [asset manifest](asset_manifest.json), [source conditions](deferred_source_issues.md) and [18-name migration](dataset_api_migration.md).
Green: 23 operator instances / 6 families; CSV: 25 candidates; pyqg: one two-layer system, run0 train / run1 test.
Optional xarray imports only in the pyqg loader. Unknown source, unknown license and missing asset are separate states.
