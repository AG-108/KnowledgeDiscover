# Data and external assets

The public checkout includes the existing small fixtures in `kd/dataset/data/`,
the SR benchmark definitions, generated Core ODE systems and the PDEformer sinus
generator. These support bounded examples without the full experiment collection.
Catalog registration does not mean an external file is present.

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

Ball-drop data accompanies *Discovery of Physics from Data: Universal Laws and
Discrepancies*. The solid-data snapshot accompanies *Beyond empirical models:
Discovering constitutive laws in solids with graph-based equation discovery*;
its local snapshot retains the original license. CYT, TLC, VGS, WDwake and
WaveBreaking source URLs/revisions are not yet recorded consistently. Obtain a
verified source/archive before claiming a fully reproducible public data release.
This checkout does not provide download URLs for those unverified snapshots.

`DeepSymbolicOptimization/`, `Discover_Green_function/`, `EqGPT/` and
`SymbolicGPT/` under `kd/dataset/` are optional reference snapshots. Maintained
ports live in `kd/model/` and do not require those snapshots at runtime.
Packaged SR benchmark CSVs remain included.

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
cleanup preserves existing external data and experiment outputs on disk.

## Checkpoints and private settings

E2E needs a separately obtained source checkout and trusted checkpoint with a
verified SHA256; see [the adapter guide](benchmark_v2_e2e.md). Keep assets under
`external/` and `checkpoints/` or outside the repository. Store credentials in
local environment variables or ignored `.env` files. Raw results, model binaries
and archives are excluded from commits.
