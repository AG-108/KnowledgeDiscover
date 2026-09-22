# Benchmark v2 implementation status

Updated 2026-09-22 after PIC v3 follow-up implementation and validation. This records
observed local artifacts/tests, not an assertion of continuous background work.
The earlier pending-restart entries are superseded by the evidence below.

## Delivered locally

| Package | Verified implementation | Important boundary |
| --- | --- | --- |
| Evaluation protocol v2 | Explicit support/coefficient recovery, predeclared denominators including worker timeouts, finite coverage, version-separated aggregates, method/family/seed and paired files | SR/ODE exact equality is still strict algebraic equality; missing truth/evaluator support remains explicit |
| Shared PDE rollout | Runner-integrated reference-validated FFT method of lines, common candidate path and separate native predictor metrics | Scalar uniform 1D periodic endpoint-excluded grids only; missing boundary/reference/solver metadata abstains |
| Information criteria | Opt-in AIC/BIC with parameter/likelihood provenance and estimated variance counted | Requires explicitly supplied training-MLE statistics; never substitutes held-out MSE; not every baseline is an MLE estimator |
| PIC | Runner-integrated opt-in PyTorch evaluator, strict equation-to-term parser including `u_xxx`, cloned candidate PINNs, epoch-wise TLS refits, real cross-process cache, train-only reference domain, explicit quality/failure statuses and complete coefficient/cost records | Bounded scalar 1D grammar; Burgers/KdV ranking calibration is recorded but did not pass, so PIC is not a validated ranking metric |
| Weak-form baseline | `integral_weak_pde` registered, quadrature/integration by parts, STLSQ, parseable expanded equations | Restricted native method, not complete WSINDy-PDE |
| ODE core | Oscillatory/population/chaotic/rational families registered; fixed unknown parameters hidden; deterministic generation and trajectory holdout | Noise/sparsity are applied; true train/test OOD split remains pending |
| Official E2E | Optional official-wrapper adapter, explicit source/checkpoint/trust/hash gates, feature-mapped tree export and inference timing | Mock API tests only; no real checkpoint available or loaded |
| Physical credibility | Frozen cards/checks, unit/domain/sign evidence, five statuses, offline blinded judge payload, AST arithmetic parser, blank annotation workflow | No actual LLM calls, expert validation or composite credibility score |

Sol agents `sol_eval_resume`, `sol_pic_resume` and `sol_credibility` delivered
bounded packages. Parent reviewed the artifacts, requested follow-up fixes,
integrated coverage/E2E, and reran tests. No production deployment or remote job
modification was performed.

## Verification and saved artifacts

Interpreter: `C:\Users\22412\.cache\kd-benchmark-v2-venv\Scripts\python.exe`
(isolated venv with existing CPU Torch).

- Final focused tests: **101 passed in 10.96 seconds**, covering runner, metrics,
  PIC, weak form, ODE, credibility, E2E, integration and native SINDy.
- PIC v3 follow-up focused tests: **79 passed in 12.21 seconds** in `kd-env`,
  covering PIC numerical/backend/cache paths, metrics, runner, and v2 integration.
- Final Python compilation and `git diff --check` passed.
- Four ODE families through the actual subprocess CLI: **4 cases / 8 target rows,
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
- `core_v2.json` dry-run creates **50 cases** with five seeds. No full core run
  has been launched. Noisy/sparse ODE config is validated with actual data generation.
- E2E tests use mocked official API objects. They are not checkpoint validation.
- Optional PySINDy and other uninstalled external baselines were not claimed as
  tested; the full optional-dependency repository suite was not run.

Generated `results/` artifacts are gitignored; source/config/docs are separate.

## Remaining work before a benchmark-v2 publication/full run

1. Acquire/review/pin the official E2E checkpoint and isolated source environment,
   then run real inference/fidelity tests and audit pretrained-corpus overlap.
2. Add genuine train/test OOD splits and broader independently controlled noise
   and sparsity sweeps. Current combined noisy+sparse config is only a smoke setting.
3. Investigate derivative-aware ANN validation/regularization and repeat the frozen
   Burgers/KdV ranking suite before using PIC to rank candidates. The current suite,
   grammar, runner wiring and artifacts exist, but the scientific acceptance gate fails.
4. Review physical dataset cards; collect real expert annotations and compare
   numeric-only/direct-LLM/evidence-assisted judges. No agreement figures exist yet.
5. Complete corpus-exclusion audits and decide additional baselines (AI Feynman,
   SINDy-PI are still not integrated). See `benchmark_v2_method_cards.md`.
6. Review production matrix, budgets, method variants and evaluation priors before
   deploying or rerunning experiments. Existing remote runs/results were untouched.
