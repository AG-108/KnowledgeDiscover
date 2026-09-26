# Benchmark roadmap

Updated 2026-09-26. This is the maintained implementation and follow-up plan.
It consolidates the previous work order, owner review tasks, implementation status
and next-run requirements. Historical evidence is in
[the validation history](reports/benchmark_validation_history.md).
Current code and actual run artifacts take precedence over older notes.

## Implemented capabilities and boundaries


| Package | Verified implementation | Important boundary |
| --- | --- | --- |
| Evaluation protocol v2.1 | Explicit support/coefficient recovery, predeclared denominators including worker timeouts, finite coverage, version-separated aggregates, method/family/seed and paired files | SR/ODE `exact_recovery` matches additive-term structure ignoring numeric coefficients; strict equality is retained as `algebraic_exact_recovery`, with coefficient error separate; missing truth/evaluator support remains explicit |
| Shared PDE rollout | Runner-integrated reference-validated FFT method of lines, common candidate path and separate native predictor metrics | Scalar uniform 1D periodic endpoint-excluded grids only; missing boundary/reference/solver metadata abstains |
| Information criteria | Opt-in AIC/BIC with parameter/likelihood provenance and estimated variance counted | Requires explicitly supplied training-MLE statistics; never substitutes held-out MSE; not every baseline is an MLE estimator |
| PIC | Runner-integrated opt-in PyTorch evaluator, strict equation-to-term parser including `u_xxx`, cloned candidate PINNs, epoch-wise TLS refits, real cross-process cache, train-only reference domain, explicit quality/failure statuses and complete coefficient/cost records | Bounded scalar 1D grammar; Burgers/KdV ranking calibration is recorded but did not pass, so PIC is not a validated ranking metric |
| Weak-form baseline | `weakform` now runs an independent scalar-1D WSINDy-PDE adaptation with compact polynomial tests, valid FFT weak convolution, scale-normalized columns, MSTLS threshold-path selection and parseable expanded equations | Full upstream multidimensional/multifield, trigonometric, automatic spectral-support and RGLS options are not exposed |
| Core ODE Track | Eight fixed fully observed systems, including Michaelis--Menten (`ode_core_rational`) and named Robertson kinetics; four other generated systems remain supplementary | Polynomial SINDy cannot exactly represent the pendulum sine or rational term; the five-seed noisy Robertson audit found severe derivative instability; Core ODE currently uses configured method grammars without declared function-set metadata; true train/test OOD split remains pending |
| Official E2E | Optional official-wrapper adapter, explicit source/checkpoint/trust/hash gates, feature-mapped tree export and inference timing | Local real-checkpoint CLI smoke passed; corpus overlap and full-paper fidelity remain unknown |
| Physical credibility | Frozen cards/checks, unit/domain/sign evidence, five statuses, offline blinded judge payload, AST arithmetic parser, blank annotation workflow | No actual LLM calls, expert validation or composite credibility score |

## Priorities

1. Recheck the PhySO, CPU PDE and GPU continuation outputs in the
   [run report](reports/server_missing_pair_runs_20260925.md). Verify manifest,
   case, target and status denominators and archive checksums before reporting
   completion. Preserve first attempts and use separate outputs for retries.
2. Freeze Core ODE function-set metadata and method applicability. The eight-system,
   eleven-method, five-seed manifest contains 440 cases, but the formal declared-grammar
   experiment has not run. Current fallback-grammar diagnostics do not establish a ranking.
3. Validate Robertson observation schedules and derivative estimators for noisy/sparse
   data. The nonuniform gradient implementation passed its audit, but short intervals
   amplify noise severely. No time cutoff is adopted. Analytic RHS values stay confined
   to the audit/evaluator, away from discovery methods.
4. Diagnose SGA SVD nonconvergence on newly admitted scalar 1D grids.
   Adapter compatibility does not establish numerical fit success.
5. Configure LLM-SR endpoint/model and validate real generation plus constant fitting
   before its batch. Import/no-fit preflight is insufficient.
6. Extend E2E checkpoint smoke validation to fidelity and corpus-overlap audits.
   Review EqGPT exclusion, SymbolicGPT synthetic overlap and DSO port fidelity.
   AI Feynman and SINDy-PI remain unintegrated.
7. Add genuine train/test OOD splits and independently controlled noise and sparsity
   studies. Scaling the entire cohort does not create an OOD holdout.
8. Resolve PIC derivative quality and repeat frozen Burgers/KdV calibration before
   ranking with PIC. Final 3/6 true-structure top-1 fails the 6/6 gate; the exploratory
   4/6 result is not a passing replacement.
9. Review physical cards, collect real expert annotations and compare numeric,
   direct-LLM and evidence-assisted judges. No agreement figures or validated
   composite credibility score exist.
10. Extend the SymbolicGPT protocol comparison across systems/seeds if required.
    Keep per-fit and within-case reuse separate; one case does not establish general
    speed or recovery advantages.

## Evaluation follow-up


1. The structural/coefficient split is implemented. If a binary coefficient-recovery metric
   is later added, its tolerances must be preregistered rather than inferred from this batch.
   Continue preserving full-eligible, conditional and lower-bound denominators.
2. Introduce a shared candidate evaluation representation carrying the equation, independent fitted parameters, constraints, training observations and likelihood provenance. Parameter count is not syntax token count and should account for identifiability/redundancy.
3. For SR, evaluate the discovered equation on a common training split; where appropriate, refit its free constants with a declared likelihood. For ODE, choose and record either derivative-fit or trajectory-fit likelihood. For PDE, declare a common residual observation model and derivative estimator; account for time/space dependence or label an iid residual approximation explicitly. Do not mix different responses, splits or likelihoods in the same AIC/BIC comparison.
4. Compute AIC=2k-2logL and BIC=k*log(n)-2logL from the fitted training likelihood. Include estimated noise parameters in k. Keep refitted candidate scores distinct from the original method's predictions. Record optimization failure and zero/invalid RSS handling explicitly; do not use test MSE as training likelihood.
5. Wire the common evaluator into PDE as well as SR/ODE; export likelihood, n, k, parameter/refit provenance, status and failure reason. Retain SGA's native score in a separately named field.
6. Expand PIC first within supported PDE systems, sharing a training-only ANN reference and equal candidate budgets. Report r_loss, p_loss, combined PIC and calibration status. Any ODE adaptation requires a separately named and validated protocol. Ordinary algebraic SR has no PDE dynamics residual; use not_applicable unless an explicit physical constraint and corresponding validated extension exist.
7. Compare AIC/BIC within the same dataset/target/split/likelihood using differences and paired rankings. Do not average raw values across unrelated datasets. Preserve held-out predictive error and recovery metrics alongside these information criteria.

## Historical audit findings

These figures refer to the archived audit, not to new runs:


- In protocol 2.0 and the archived runs, SR/ODE `exact_recovery` was strict SymPy algebraic
  equality after variable normalization. Protocol 2.1 replaces that primary statistic with
  structural recovery while retaining strict equality as a diagnostic. Missing values still
  include parser failures, so parse coverage and denominator semantics must be retained.
- PySR: 61/274 SR result rows have test NRMSE < 1e-4, including 49 with exact_recovery=0. GrammarVAE-1 predicts x1 + sin(x1*x1) + 0.33333334 against 1/3 + x1 + sin(x1*x1), with test NRMSE 1.1454e-9 but exact=0. This is numerical agreement, not proof of algebraic identity.
- The archived PhySO display expressions caused low evaluator coverage. After recovery,
  13/251 parsed successful SR expressions match the ground-truth additive-term structure;
  the 0/34 strict algebraic figure remains limited to historically evaluated rows.
- PDENet uses a two-dimensional regular uniform-grid adapter. In the 23-dataset compatibility report, only wdwake passes; 13 one-dimensional grids and 9 scatter datasets do not. The wrapper also documents a single dx for both spatial axes and periodic padding; broader grid support needs explicit validation.
- The archived SGA run recognized only burgers, kdv and chafee-infante. Generic external
  grid configuration has since been added, but Burgers and KdV previously failed with SVD
  nonconvergence while Chafee-Infante completed. Numerical robustness therefore remains a
  separate open issue.
- Standard AIC/BIC evaluation currently accepts explicitly declared training likelihood, sample size and parameter count in the SR/ODE path; it does not extract them automatically. The PDE path does not call the shared evaluator. SGA writes its internal search score into aic using 2*aic_ratio*k + 2*log(MSE); retain that separately from standard AIC.
- PIC is opt-in, integrated only for a bounded scalar 1D PDE grammar. Local ranking calibration is unvalidated: final 3/6 true-structure top-1, best observed configuration 4/6, against a declared 6/6 threshold. Keep auxiliary status and do not select a favorable calibration result as evidence of general validity.


## Acceptance and reporting rules

- Preserve original results and explicit metric/protocol versions. Report case,
  instance, target and seed counts separately; execution success is not recovery.
- Report eligible denominators, parser/evaluator and finite prediction coverage,
  method errors, timeouts and unsupported tasks separately.
- Shared PDE rollout requires scalar uniform 1D periodic endpoint-excluded grids,
  boundary/solver metadata and a validated reference. Native predictor scores stay
  separate; missing metadata requires abstention.
- PIC uses training-only references and equal candidate budgets. Retain coefficient
  provenance, component scores, costs and failed quality gates.
- Physical checks use frozen, candidate-independent cards with units, assumptions,
  domains, tolerances and budgets. Unknown evidence is not a proven violation.
  Judge input excludes method identity/truth; generated mutations are not labels.
- Preserve old worktrees/results when syncing server code. Back up changed files and
  compare normalized source hashes before and after transfer.

Canonical details: [metrics](benchmark_metrics.md),
[method cards](benchmark_v2_method_cards.md), [PIC](benchmark_v2_pic.md),
[physical credibility](benchmark_v2_physical_credibility.md).
Session handoff: [STATUS.md](../STATUS.md).
