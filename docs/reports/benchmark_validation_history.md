# Benchmark validation history through 2026-09-25

Counts apply to the specific revisions and environments described below, not to the current checkout. Raw artifacts are local. Current priorities are in [the roadmap](../benchmark_roadmap.md).



- Final focused tests: **101 passed in 10.96 seconds**, covering runner, metrics,
  PIC, weak form, ODE, credibility, E2E, integration and native SINDy.
- WSINDy replacement focused tests: **15 passed**, including a guard that makes
  `np.gradient` fail if the fit path attempts pointwise field differentiation.
  Default runner settings recovered the declared Burgers and KdV equations to
  floating-point accuracy and all three Chafee--Infante coefficients within
  approximately `1.8e-4` on the packaged clean datasets.
- After installing the project-pinned optional dependencies into the existing
  `kd-env` Conda environment, the complete repository suite finished with
  **236 passed / 0 failed in 62.23 seconds**. `pip check` reported no broken
  requirements. The temporary benchmark venv was then removed.
- Actual CLI smoke `weakform x burgers` completed `ok`, recovered
  `-1.00000000016*u*u_x + 0.100000000014*u_xx`, and serialized the selected
  threshold, MSTLS loss, weak residual, support, test powers, sample count, and
  provenance. Artifacts are under `results/wsindy_runner_smoke_20260922/`.
- PIC v3 follow-up focused tests: **79 passed in 12.21 seconds** in `kd-env`,
  covering PIC numerical/backend/cache paths, metrics, runner, and v2 integration.
- Final Python compilation and `git diff --check` passed.
- Before the expansion, four ODE families ran through the actual subprocess CLI: **4 cases / 8 target rows,
  all execution status `ok`**, saved to `results/v2_validation_ode_core/`.
  `ok` is execution success, not exact equation recovery.
- That directory contains raw JSON/CSV, aggregate JSON/CSV, seed-family JSON/CSV,
  paired JSON/CSV and per-case logs/manifests. Single-seed runs provide no measured
  seed uncertainty; a single method provides no between-method pairs.
- Real CPU PIC smoke: `results/v2_validation/pic_smoke.json`, status `ok`,
  r_loss approximately 0.01994, p_loss approximately 0.00733, PIC approximately
  0.00014616. ANN RMSE approximately 0.188 and refitted heat coefficient approximately
  2.025 versus submitted 1.0 show that this tiny-budget example is **execution
  validation only**, not identification accuracy or converged-paper validation.
- The v3 smoke is stored separately as `results/v2_validation/pic_smoke_v3.json`;
  it exercises the new derivative graph, explicit PINN loss diagnostics, and v3
  protocol identity. Historical v2 artifacts were not relabeled.
- PIC v3 ranking calibration artifacts preserve successive 30/300-epoch, multimode,
  sine-activation and higher-budget trials. The latest configured Burgers/KdV run
  ranks the true structure first in 3/6 seed-problem trials (Burgers 2/3, KdV 1/3),
  below the fixed 6/6 acceptance threshold. An earlier exploratory setting reached
  4/6 but is not selected or reported as passing. This negative result demonstrates
  that state-fit NRMSE alone does not ensure stable high-order ANN derivatives.
- Real runner smoke: `SINDy x burgers x seed 17` completed with both baseline and
  PIC status `ok`; the discovered equation was coefficient-correct, PIC was about
  `2.24e-5`, and all component/coefficient/cost fields were present in JSON and CSV.
  A separate-process rerun hit the reference cache with `reference_seconds=0` and
  reproduced the same PIC value. Artifacts are under
  `results/v2_validation/pic_runner_{real,cache_hit}_smoke/`.
- Synthetic weak-form integration test validates temporal holdout, coefficients,
  boundary-metadata preservation and reference-checked shared rollout.
- The pre-expansion `core_v2.json` dry-run selected **50 cases** and the later
  all-twelve-system integration draft selected **90 cases**. After freezing the
  eight-system Core ODE Track, the current `core_v2.json` selects **70 cases**
  and its noisy/sparse ODE companion selects **40 cases**, each with five seeds.
  The formal all-ODE-method `core_ode_track.json` selects **440 cases** (eight
  systems, eleven methods, five seeds). These are manifests, not completed runs.
- The expanded ODE bridge test covers all twelve systems. A three-case SINDy CLI
  smoke (damped oscillator, pendulum, Robertson) completed **3 cases / 7 target
  rows**, all status `ok`, under `results/ode_expansion_smoke_20260923/`.
  This checks execution only; it is not a full recovery or noisy-data study.
- The updated focused suite finished with **121 passed** in `kd-env`; the complete
  repository suite then finished with **255 passed / 0 failed**. Existing DSO,
  pkg_resources and SciPy deprecation/runtime warnings remain.
- After freezing the eight-system Core ODE Track, a one-seed SINDy smoke run
  completed **8 cases / 17 derivative targets**, all execution status `ok`,
  under `results/core_ode_track_sindy_smoke_20260923/`. Structural recovery was
  7/17 in this smoke run; this is not a multi-method or multi-seed benchmark.
  The updated complete repository suite finished with **257 passed / 0 failed**.
- E2E contract tests use mocked official API objects. Separately, the official
  source and checkpoint were cached outside the worktree and a real-checkpoint
  CLI smoke completed **1 case / 2 derivative targets**, both status `ok`, under
  `results/e2e_dependency_smoke_20260924/`. This is execution validation only.
- PyOperon 0.6.1 is installed in a separate Python 3.10 `kd-operon` environment.
  A real runner smoke completed **1 case / 2 derivative targets**, both status
  `ok`, under `results/operon_dependency_smoke_20260924/`. `pip check` passed in
  both `kd-env` and `kd-operon` after dependency installation.
- The one-seed, no-fit Core ODE compatibility preflight found **72/88** pairs
  compatible in the main `kd-env`; its 16 expected errors were the eight E2E
  pairs without tracked checkpoint paths and eight PyOperon pairs without that
  package in `kd-env`. Separate preflights passed **8/8** PyOperon pairs in
  `kd-operon` and **8/8** E2E pairs with the local checkpoint override. Reports
  are under `results/core_ode_track_*_compat_20260924/`. The check only loads
  data and adapters; it does not prove a fit, especially for LLM-SR, whose
  endpoint/model are still unset.
- After the Windows E2E checkpoint-path fix, the focused E2E/runner suite passed
  **67 tests** and the complete repository suite passed **258 tests / 0 failed**.
  Only existing DSO `pkg_resources`/NumPy and SciPy runtime warnings remain.
- Optional PySINDy and other uninstalled external baselines were not claimed as
  tested; the full optional-dependency repository suite was not run.

Generated `results/` artifacts are gitignored; source/config/docs are separate.

### SymbolicGPT validity follow-up (2026-09-25)

- Arithmetic grammar, complete sampling boundaries, finite fitted constants/loss,
  and predictions on every training row are now checked before accepting a
  candidate. Failed refits clear the previous solution. The implementation is
  still the per-fit-from-scratch variant, not the official pretrained protocol.
- The adapter/runner/integration regression set passed **111 tests** in `kd-env`;
  this includes 44 SymbolicGPT tests. That local run was a focused subset; the
  subsequent server full-suite result is recorded below.
- The original Nguyen-8 tiny-budget CLI now reports an explicit fit failure
  (0/2 valid candidates) without exporting malformed expressions. A larger-budget
  Nguyen-1 CLI smoke completed `ok`, with 20/20 finite held-out predictions and
  a parsed expression; structural recovery was 0. These are execution checks.
  Artifacts and JUnit: `results/tuning/symbolicgpt_validity_20260925/`.
  The new smoke config is `configs/benchmark/tuning/symbolicgpt_validity_smoke.json`;
  it was run on CPU with `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1`.

- Subsequent server validation passed **288 tests / 1 Windows-only skip** in
  150.54 seconds using the unified Python 3.10 environment. Source changes were
  backed up and hash-checked during synchronization; artifacts were verified and
  copied back to the local worktree.
- With the original full configuration on the RTX 4090, one damped-oscillator
  case produced one finite-prediction `ok` RHS in 194.83 seconds; the second RHS
  timed out when the shared 300-second case budget expired. A separate 250-corpus,
  8-epoch, 24-candidate pilot covered all eight systems but returned **17 fit
  errors / 0 timeouts**, accepting none of 408 samples. That pilot is not a
  usable replacement configuration, and its recovery metrics were not evaluated.
- Clean Core ODE audit: 17/17 references parsed and 17/17 training derivative
  targets finite. Robertson's five-seed noise/sparsity audit found maximum
  derivative relative RMS errors of about 7.63e4 at 1% observation noise and
  3.81e5 at 5% noise, measured against paired clean-state analytic RHS values.
  Finiteness alone is insufficient for the noisy-data protocol.
- These server results are under `results/server_symbolicgpt_validation_20260925_134231/`.
  Both fitting experiments use configured method grammars, not a declared-function
  main ranking; the complete 440-case experiment has not been run.
