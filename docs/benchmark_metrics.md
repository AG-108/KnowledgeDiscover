# Benchmark metric protocol

当前工程验收见 [验收摘要](validation.md)，来源条件见 [待解决问题](deferred_source_issues.md)。数据 domain-v4；新增轨指标 2.3。

Raw per-target records are written to `benchmark_summary.json/csv`; method/task aggregates
are written to `benchmark_aggregate.json/csv`. Original adapter records declare
`metric_protocol_version = "2.1"`. Older records remain readable, but failed legacy rows
cannot be retroactively classified as recovery-eligible when that fact was not stored.
Aggregates are partitioned by metric, execution (`target-v1`) and dataset
(`domain-v4`) protocol versions, and by `scientific_task` within each method.
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

`target-v1` additionally records `target_id`, attempt number, execution policy,
`setup_seconds`, `fit_seconds`, `score_seconds`, and `worker_elapsed_seconds`.
The legacy `runtime` measures the target adapter including scoring; killed workers use
elapsed worker time. Use `fit_seconds` when comparing the fit budget. Preparation is
separately supervised, and fit is hard-limited to T + grace. Scoring has its own cap.
`timeout_phase` distinguishes setup, fit and score failures; a score timeout is never a
successful recovery. Resource-slot enforcement and CPU/GPU cost accounting remain pending.

Every new row has one of six `stop_reason` values: `budget_exhausted`, `time_limit`,
`converged`, `early_stop`, `hard_timeout`, `error`. Explicit adapter `stop_reason_`
signals take priority: SINDy records support convergence, DSO and LLM-SR record their
early-stopping conditions, and gplearn's shortened generation history records early stop.
Where the backend exposes no reason, a normal return uses `budget_exhausted` with
`stop_reason_source=native_return`; this is a fallback classification, not evidence that
every native budget unit was consumed. Other sources are `adapter`, `generation_count`
and `runner`. Failed/skipped rows use `error`, and hard timeouts use `hard_timeout`;
the separate status retains the failure/skip distinction. Soft `time_limit` signals will
be wired in C3. Aggregates include reason counts/rates and source counts. Legacy rows
without a reason remain `unknown` in these distributions; their files are not rewritten.

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

pyqg scores physical s^-2 MSE/RMSE, held-out-run layer/frame NRMSE and finite coverage of declared sampled points. Training-only normalization, indices/weights are recorded; expressions use standardized inputs/output. Offline closure does not establish online stability. File SR keeps official pairs or fixed input-range interpolation holdout, with unknown truth. New tracks are recovery-ineligible until source certification.
