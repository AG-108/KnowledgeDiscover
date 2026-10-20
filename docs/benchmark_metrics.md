# Benchmark metric protocol

当前工程验收见 [验收摘要](validation.md)，来源条件见 [待解决问题](deferred_source_issues.md)。本地开发树数据协议 `domain-v5`；服务器冻结实验仍为 `domain-v4`，两者不混用。新增轨指标 2.3；本地修复范围见[数据接入修复](data.md#domain-v5-接入修复2026-10-11)。

Raw per-target records are written to `benchmark_summary.json/csv`; method/task aggregates
are written to `benchmark_aggregate.json/csv`. Original adapter records declare
`metric_protocol_version = "2.1"`. Older records remain readable, but failed legacy rows
cannot be retroactively classified as recovery-eligible when that fact was not stored.
Aggregates are partitioned by metric, execution (`target-v2` for current runs;
legacy `target-v1` remains readable) and dataset (`domain-v5` locally,
`domain-v4` for the frozen server experiment) protocol versions, and by `scientific_task` within each method.
Paired and seed-family reports also isolate dataset versions. CYT closure and
material constitutive targets are separate physical tasks even though both use
the numerical `sr` adapter. `n_cases`, `n_target_rows`, and `denominator_unit = target_row` make clear that a
multi-target ODE contributes several target rows but only one configured case.
The same applies to source material/curve instances expanded from one solid
dataset name. Target-weighted means are not family-balanced statistics; the
planned E3 analysis must account for related materials and repeated variants.

## Recovery and denominators

The protocol separately reports run success (`status` and `success_rate`), evaluator/parse
coverage (`*_parse_coverage`), and recovery. `*_conditional_rate` is conditional on a
parsed valid output. `*_lower_bound` treats all other eligible rows as zero. The principal
`*_recovery_rate` uses every eligible row but is emitted only when every valid algorithm
output was evaluated, so evaluator bugs do not silently become algorithm failures.
`recovery_eligible` is predeclared for cases with trusted ground truth. The unconditional
denominator contains every eligible row, including `error` and `timeout`; skipped tasks
and rows without trusted truth are excluded. Evaluator coverage remains visible so an
evaluator failure is not silently described as an algorithm failure.

For SR/ODE, protocol 2.1 `exact_recovery` means structural recovery: after variable
normalization and additive expansion, discovered and reference expressions must have the
same additive terms, but numeric coefficients are ignored. Numeric exponents remain part
of the structure, so `x**2` and `x**3` do not match. Numeric scale or offset inside a
function argument is treated as a coefficient rather than a structural operator.
`structural_recovery` is an explicit alias, and
`exact_recovery_semantics = additive_term_structure_ignoring_numeric_coefficients` is
stored with every row. `exact_recovery_rate` uses the eligible denominator.

`coefficient_error` is reported separately, and only after structural recovery. It is the
relative L2 error between coefficients aligned by the recovered structural terms, with an
absolute L2 fallback when the reference coefficient norm is zero. Protocol 2.1 retains the
old strict SymPy equality as `algebraic_exact_recovery`; protocol 2.0 `exact_recovery`
records should therefore be interpreted as the legacy strict algebraic diagnostic, not
pooled with protocol 2.1 structural recovery.

For PDE, `pde_support_recovery` means equality of nonzero additive term-support sets; it
deliberately ignores coefficient error. `term_support_accuracy`, precision, and recall give
Jaccard and directional scores. `coefficient_error` separately gives aligned relative L2
coefficient error when numeric ground-truth coefficients are available.
`coefficient_recovery` additionally requires aligned coefficients to satisfy
`rtol=0.05, atol=1e-8`; the tolerances are stored in each result.

The old `exact_symbolic_recovery` field remains as an explicitly labelled legacy alias of
`pde_support_recovery`; it is not exact coefficient-aware symbolic recovery. New records
also contain `exact_symbolic_recovery_semantics` and
`pde_support_metric_version = "2.0"`. Aggregates expose new `pde_support_*` names while
retaining the old aggregate rate as the same compatibility alias.

## Prediction errors

SR/ODE `test_nrmse` is RMSE divided by target standard deviation; `test_r2` is
`1 - MSE / variance`. ODE splits hold out complete trajectories. Prediction records expose
total points, finite points, and finite coverage; errors are withheld unless coverage is one.

PDE residual metrics compare finite-difference `u_t` with the discovered RHS on a fixed
held-out domain:

- `equation_residual_total_points`: all declared residual points;
- `equation_residual_points`: points where target and prediction are finite;
- `equation_residual_finite_coverage`: their ratio.

MSE/NRMSE are reported only when coverage is 1. A candidate with NaN/Inf predictions
cannot obtain a deceptively good score by dropping invalid-domain points. Aggregates report
mean finite coverage separately. Rollout remains a separate metric family.

Native model prediction and equation rollout are distinct estimands and retain separate
fields. Validated equation rollout currently supports only scalar, uniform 1-D periodic
grids whose periodic endpoint is excluded. Metadata must declare `periodic` boundaries,
`periodic_endpoint_excluded`, `spectral_fft`, the SciPy solver and tolerances, and a
reference NRMSE tolerance. The ground-truth equation is rolled out and checked against the
reference trajectory before any candidate is scored. Missing metadata, unsupported grids,
or failed reference validation produce an explicit abstention status. Older raw rollout
fields are legacy diagnostics and are not aggregated as validated equation rollout.

## Information criteria and statistical reporting

AIC/BIC are opt-in through dataset evaluation metadata and require a known free-parameter
count plus a supplied log-likelihood or an explicitly declared Gaussian RSS likelihood.
The runner additionally requires `likelihood_data_role=training`,
`parameter_estimation=maximum_likelihood`, the actual `n_observations` and either
`log_likelihood` or `residual_sum_squares` in the opt-in evaluator configuration.
It never substitutes held-out MSE for maximized training likelihood. Without these
inputs it abstains. These declarations are a provenance contract, not a certificate
that an arbitrary baseline really computes a maximum-likelihood estimate.
For `gaussian_mle_variance`, the estimated variance adds one free parameter and is recorded
separately. Syntax tokens are never treated as free parameters, and raw criteria are not
averaged across unrelated datasets.

Paired comparisons include dataset, instance, target, seed, metric/execution protocol, profile, and regime;
duplicates are rejected rather than overwritten. Reports expose paired and per-model
evaluable counts. Family uncertainty is grouped by method/family/metric-and-execution-protocol/profile/regime,
first averages available targets within a seed, then reports variability across seeds.
These are explicitly labelled available-case means, not unbiased estimates under missingness.
Each instance/target runs in its own worker. A timeout produces only that target's
terminal row; subsequent targets still run. Resume includes terminal failures and
recovers worker commits missed by an interrupted parent. Preparation failures preserve
known ODE target and generated-instance counts; an unknowable remainder is explicitly
marked `target_manifest_complete=false` and counted as `n_incomplete_target_manifests`.
Such cases require resolution before interpreting a formal experiment's denominator.

The shared rollout metadata's solver block must explicitly contain `method`,
`rtol`, `atol` and `max_step` (`null` explicitly permits adaptive steps without an
additional maximum). Settings, endpoint convention and reference tolerance are
saved with each scored rollout. PDE residual errors still use estimated derivatives;
rollout and residual are different diagnostics and should not be conflated.

## Resource and status fields

Each row records wall-clock runtime, peak CPU RSS, optional CUDA allocator peak, and one of
`ok`, `timeout`, `error`, or `skipped`. Success and timeout rates use all rows. Missing
values serialize as JSON `null`, never non-standard NaN.

Fields inherited from `target-v1` include `target_id`, attempt number, execution policy,
`setup_seconds`, `fit_seconds`, `score_seconds`, and `worker_elapsed_seconds`.
The legacy `runtime` measures the target adapter including scoring; killed workers use
elapsed worker time. Use `fit_seconds` when comparing the fit budget. Preparation is
separately supervised, and fit is hard-limited to T + grace. Scoring has its own cap.
`timeout_phase` distinguishes setup, fit and score failures; a score timeout is never a
successful recovery. C5/C6 resource accounting is described below; C7 method parallelism
is implemented. `target-v2` also implements cooperative soft stopping at the fit deadline;
native blocking work remains protected by the T + grace hard limit.

Every new row has one of seven `stop_reason` values: `budget_exhausted`, `time_limit`,
`converged`, `early_stop`, `hard_timeout`, `error`, `resource_limit`. Explicit adapter `stop_reason_`
signals take priority: SINDy records support convergence, DSO and LLM-SR record their
early-stopping conditions, and gplearn's shortened generation history records early stop.
Where the backend exposes no reason, a normal return uses `budget_exhausted` with
`stop_reason_source=native_return`; this is a fallback classification, not evidence that
every native budget unit was consumed. Other sources are `adapter`, `generation_count`
and `runner`. Failed/skipped rows use `error`, RSS-limit failures use `resource_limit`,
and hard timeouts use `hard_timeout`;
the separate status retains the failure/skip distinction. `target-v2` records soft
`time_limit` signals from the cooperative fit deadline. Aggregates include reason counts/rates and source counts. Legacy rows
without a reason remain `unknown` in these distributions; their files are not rewritten.

## C5/C6 resource protocol slots-v1

The parent supervisor samples each worker process tree, including interpreter startup,
setup, fit and target scoring, and retains costs when a worker errors or is killed.
`cpu_seconds` sums user + system CPU seconds per observed PID/create-time; it is a
sampled estimate, not wall time. CPU totals retain exited children's last measurement
and exclude parent-reported children CPU to avoid double counting. Short-lived children
and RSS spikes entirely between polls can be missed (slot default: 50 ms). A final
worker self CPU measurement supplements the sampler on normal completion. Missing
CPU measurements remain null; `resource_monitor_source` and measurement coverage are
reported rather than replacing missing values with zero.

`gpu_occupied_seconds` measures exclusive GPU-slot wall time through worker cleanup,
including setup and scoring. It measures reserved GPU capacity, not kernel execution
time or GPU utilization. CPU slots record zero. Unconstrained diagnostic CUDA runs use
device-assignment wall time and label `gpu_occupied_source=device_assignment_wall`;
exclusive slots label `exclusive_slot_wall`. These measurements are kept in distinct
resource cohorts. `peak_memory_mb` is sampled process-tree RSS (shared pages can be
counted per process); `peak_gpu_memory_mb` remains the optional worker CUDA allocator
peak, not whole-device VRAM or external vLLM allocator memory.

`resource_policy`, `resource_policy_hash`, `resource_slot` and the execution policy
record affinity, visible CUDA device, thread count, RSS ceiling, sampling interval and
declared service PIDs. `resource_limit=rss` maps to `status=error`,
`stop_reason=resource_limit`; it remains in the denominator and `n_resource_limit`.
Preparation failures that prevent enumerating actual targets keep the existing
incomplete-manifest marker; they cannot certify a frozen denominator.

C7 adds `parallelism_protocol=d2-parallel-v1` to new resource policies and records
the applied `thread_environment` in summary JSON/CSV. This changes the resource hash
and isolates the new per-task numerical thread limits from older C5 histories, whose
original policy and terminal rows remain readable. Slot manifests record the effective
method worker counts before execution and case hashing; search budgets are not changed
by this binding. See the [method card](methods.md#d2-method-parallelism-c7).

Summary JSON/CSV retain these fields. Aggregate JSON/CSV partition by resource policy
hash and contain `cpu_seconds_total/mean/measured_rows`,
`gpu_occupied_seconds_total/mean/measured_rows`, and `worker_elapsed_seconds_total`.
Shared case preparation and coupled ODE scoring are separate row fields
`preparation_cpu_seconds`, `preparation_gpu_occupied_seconds`,
`system_score_cpu_seconds`, `system_score_gpu_occupied_seconds`; their aggregate
`*_total` counts each case once, rather than once per RHS. Complete case cost is target
worker totals plus these shared totals. Repeated interrupted attempts remain in their
attempt histories and are not silently added to first-terminal-row costs.

Declared vLLM service trees contribute incremental CPU and simultaneous RSS alongside
the worker. GPU slot wall time already covers both, so it is not doubled for the
service. Frozen reports validate case resource policies and isolate resource cohorts;
seed-family and paired reports also isolate resource policy hashes.

## LLM evaluation scope

Any optional LLM judge is scoped only to physical credibility under declared physical
conditions and evidence provenance. It must not score scientific explanatory value,
novelty, familiarity, or conformity to a conventional equation form. Such a judge is not
part of the numeric recovery fields above and requires separate validation.

A1/A2 来源、资格、同源观测与协议迁移见 [验收记录](validation.md)；
旧 domain-v1 与新 domain-v3 结果不混用，来源不足尚未正式就绪。

## 方程残差协议 2.2

新方法使用 metric_protocol_version=2.2、equation_protocol_version=equation-v1。
规范化残差使用训练列 RMS 与系数单位范数；整体缩放/反号不改变评分。
固定训练尺度不依赖测试残差；保留全部观测分母及导数有限覆盖。
与旧 2.1 的 u_t NRMSE 分别聚合；steady_pde 独立汇总。
库秩、近零空间与 ambiguity 写入 details；未认证来源或多关系歧义不计唯一恢复。
没有 BVP/rollout 时 prediction_status=abstain，不报告虚假场预测或全零目标 R²。

## A5/A6 metric protocol 2.3

Green primary score: held-out whole-forcing response MSE and RMSE divided by training (U-U_hom) RMS. Per-forcing error, inverse rank/conditioning and coverage are recorded. Reference AST is scoring-only; reference-forward floor uses all-response RMS as a separate diagnostic. Fixed offdiagonal mask abs(x-xi)>1e-12 includes exclusion fraction; forward integration retains the diagonal. No certified kernel recovery.

FDE responses use metric protocol 2.3 and `fde-response-v1`, numerical task `sr`,
scientific task `fractional_equation_response`. Targets are observed `u(x,t)` or
`ex(t)`; no integer or fractional derivatives are estimated. The latest unique
times form the test block; tied times and all spatial points in each frame stay
together. Training caps sample only the earlier block, with raw units retained.
Scores reuse SR MSE, RMSE, test-standard-deviation NRMSE, R² and finite prediction
coverage. Reference formulas and fractional orders are unknown, so recovery is
ineligible. Missing coordinates retain `source_blocked` targets; byte duplicates
and the rounded tsfade DAT alias add no independent fits. See [FDE](fde_dataset.md).

FDE PDE views use `fde-pde-diagnostic-v1` and the separate scientific task
`fractional_field_integer_pde_diagnostic`. All 13 PDE adapters are enumerated;
PDE-Net retains an unsupported status for scalar 1D fields. Integer derivative
residuals retain the adapters' metric protocols 2.1 or 2.2 and their respective
normalization definitions. All methods use disjoint temporal blocks with two
unused frames and at least three test frames; derivatives follow the split.
Implicit search uses the same observations with an implicit target representation.
No fractional operator is evaluated, recovery is ineligible, and rollout abstains
because boundary evidence is absent. These are views of existing observations;
response scores and residual scores belong to separate comparison groups.

pyqg scores physical s^-2 MSE/RMSE, held-out-run layer/frame NRMSE and finite coverage of declared sampled points. Training-only normalization, indices/weights are recorded; expressions use standardized inputs/output. Offline closure does not establish online stability. File SR keeps official pairs or fixed input-range interpolation holdout, with unknown truth. New tracks are recovery-ineligible until source certification.

## E1–E3 evaluation extensions

Derivative metric protocol 2.1 is retained. ODE trajectory fields additionally identify
`ode-rollout-v1`; reference controls identify `reference-v1`. The scoring configuration
is part of case identity and the frozen report's protocol grouping.

ODE RHS are assembled in state-variable order only after all component fits finish.
The scoring view uses the same whole-trajectory split as derivative regression and
supplied static covariates. An initial observation starts each integration and is
excluded from error. NRMSE uses each held-out trajectory/state's standard deviation,
then averages trajectories equally. Constant-state normalization is an explicit
abstention. Candidate expression evaluation uses strict symbolic arithmetic, rather
than a baseline's protected numerical operators. Unknown symbols and unsupported
syntax are reported, without inventing parameters or substituting another RHS.

The declared reference must reproduce the observed trajectories within the frozen
tolerance before a candidate rollout is scored. Reference disagreement, incomplete
systems, unparsed/nonfinite RHS, divergence, solver failure and scoring limits retain
their statuses and target denominators. A failed trajectory gives no finite overall
trajectory score. This validation does not certify noisy Robertson derivatives or
online stability. Reference/test observations are never used for model fitting or
candidate selection. System scoring time is recorded separately from target fit time.

Reference rows use the shared observation scoring code without fitting coefficients.
Declared formula/generator/coefficient references can provide a numerical error floor;
unverified source references remain diagnostics. Missing or unsupported truth is an
explicit abstention. Reference recovery and ranking are ineligible.

`frozen-report-v2` validates manifest SHA256, row identity/timing and declared targets.
It isolates metric/data/execution protocols, task, profile, scoring settings and CPU/GPU.
Error means average targets within instance, instances within dataset, datasets within
family, families within seed, then seeds equally. Available-success means expose
evaluable target coverage. Recovery lower bounds retain eligible failed/unparsed rows
as zero. Bootstrap resamples whole seeds (2000 samples, 95% percentile interval,
deterministic manifest-derived RNG); fewer than two evaluable seeds have no CI.
Status counts and success denominators include missing terminal outputs. All source
metadata remain unchanged; input hashes accompany standalone PNG/PDF figures.
