# Benchmark metric protocol

Raw per-target records are written to `benchmark_summary.json/csv`; method/task aggregates
are written to `benchmark_aggregate.json/csv`. New records declare
`metric_protocol_version = "2.1"`. Older records remain readable, but failed legacy rows
cannot be retroactively classified as recovery-eligible when that fact was not stored.
Aggregates are partitioned by protocol version, so legacy and v2 rows are never silently
pooled. `n_cases`, `n_target_rows`, and `denominator_unit = target_row` make clear that a
multi-target ODE contributes several target rows but only one configured case.

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

Paired comparisons include dataset, instance, target, seed, protocol, profile, and regime;
duplicates are rejected rather than overwritten. Reports expose paired and per-model
evaluable counts. Family uncertainty is grouped by method/family/protocol/profile/regime,
first averages available targets within a seed, then reports variability across seeds.
These are explicitly labelled available-case means, not unbiased estimates under missingness.
Whole-worker timeouts preserve every predeclared ODE target and retain completed
partial targets without duplication. When a timeout affects multiple remaining
targets, leftover worker time is assigned once, not multiplied by target count.

The shared rollout metadata's solver block must explicitly contain `method`,
`rtol`, `atol` and `max_step` (`null` explicitly permits adaptive steps without an
additional maximum). Settings, endpoint convention and reference tolerance are
saved with each scored rollout. PDE residual errors still use estimated derivatives;
rollout and residual are different diagnostics and should not be conflated.

## Resource and status fields

Each row records wall-clock runtime, peak CPU RSS, optional CUDA allocator peak, and one of
`ok`, `timeout`, `error`, or `skipped`. Success and timeout rates use all rows. Missing
values serialize as JSON `null`, never non-standard NaN.

## LLM evaluation scope

Any optional LLM judge is scoped only to physical credibility under declared physical
conditions and evidence provenance. It must not score scientific explanatory value,
novelty, familiarity, or conformity to a conventional equation form. Such a judge is not
part of the numeric recovery fields above and requires separate validation.
