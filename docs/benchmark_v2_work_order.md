# Benchmark v2 implementation work order

Requested by the user after review of the current benchmark, 2026-09-21.
Implementation lead: GPT-5.6 Sol. Research direction and review: parent agent.

## Scope and operating constraints

- Implement and validate locally; preserve the current remote experiment and its results.
- The worktree already contains extensive user changes and untracked implementation files.
  Do not reset, discard, or replace unrelated work; inspect before editing.
- No production benchmark launches, server deployment, paid LLM calls, or external
  communications as part of this work order. Prepare reproducible launch configurations.
- Preserve old result readability. Introduce explicit metric/protocol versions and
  distinguish old and new evaluation semantics. Do not silently relabel old results.
- Use official sources for method definitions. Label ports, adaptations, and incomplete
  implementations honestly. Missing checkpoints or dependencies are not successful tests.
- Human annotations and scientific validation cannot be fabricated. An LLM evaluator
  prototype must not be presented as an already validated scientific metric.

## Delivery A: evaluation protocol v2

1. Audit the runner and implement unambiguous structure recovery versus coefficient-aware
   recovery. Keep compatibility aliases only with explicit semantics/version metadata.
2. Report unconditional recovery on the predeclared eligible, ground-truth-bearing cases;
   distinguish output validity, evaluator coverage, method failures, and unsupported tasks.
   Evaluator bugs must not silently become algorithm failures.
3. Report finite prediction coverage. Do not reward a candidate by quietly dropping its
   nonfinite predictions from the scoring domain.
4. Provide a shared equation rollout path with explicit boundary conditions, spatial
   discretization, integration settings, failure statuses, and ground-truth solver checks.
   Keep native predictor scores separate. Abstain when required problem metadata are absent.
5. Wire consistent information-criterion evaluation where assumptions and free-parameter
   counts are known. Do not substitute syntax node count for parameter count, manufacture
   likelihood assumptions, or average raw AIC/BIC across unrelated datasets.
6. Support task/family-level reporting, paired comparisons and seed uncertainty with
   documented denominators. Prepare a small core suite and multi-seed configurations.
7. Document the migration and add meaningful tests for the above failure modes.

## Delivery B: faithful PDE PIC evaluator

Sources:
- https://arxiv.org/abs/2208.03322
- https://arxiv.org/pdf/2208.03322 (Methods equations 12-17 and supplementary settings)
- https://github.com/woshixuhao/PIC_code

The current ParsimonyInformationCriterion is a custom additive penalized score,
not the paper's Physics-informed Information Criterion. Rename/deprecate explicitly.

Implement PIC = r_loss * p_loss:
- r_loss: mean coefficient CV across overlapping temporal windows; author code uses
  abs(std(coefficient) / mean(coefficient)). Reproduce the coefficient-fitting procedure,
  not just the displayed scalar formula.
- p_loss: RMSE between consistently normalized outputs of the common reference ANN and
  a PINN initialized from that ANN and trained under the candidate PDE constraint.
- Audit the author's coefficient refitting, normalization, windowing and training
  protocol; document any necessary deviations or numerical safeguards.
- Isolate common data/reference ANN preparation from per-candidate evaluation. Use
  identical budgets/initialization for candidates, no test-data leakage, and caching
  keyed by data split, configuration, seed and implementation version.
- Return components, status, costs, and original versus refitted coefficients separately.
- Include edge-case tests and a bounded synthetic PDE smoke validation. Do not enable
  a costly PIC pass on every existing experiment by default.

## Delivery C: coverage, method provenance and core suite

- Audit current baseline implementation fidelity. Known issues to investigate include
  weakform being finite-difference/lstsq rather than WSINDy; per-case SymbolicGPT
  pretraining; DSO's PyTorch port; EqGPT's omitted final refinement and target-corpus
  exclusion mapping; PhySO's dimensionless configuration.
- Add a true weak-form PDE baseline first, with correct integration-by-parts semantics
  and provenance. Do not rename the current finite-difference implementation WSINDy.
- Add an official-pretrained SR adapter, prioritizing E2E Transformer. Keep optional
  dependencies/checkpoints explicit and record training/inference costs separately.
  Official source: https://github.com/facebookresearch/symbolicregression
- Expand ODE coverage with reproducibly generated, well-specified systems spanning
  oscillatory, chaotic, population and rational dynamics; multiple initial conditions,
  trajectory-level splits, known equations and suitable solver tolerances are required.
- Prepare core-suite noise/sparsity/OOD and multi-seed configurations at practical scale.
- Add method and dataset cards for units, available prior information, task families,
  applicability, provenance and training-data overlap checks.
- Consider AI Feynman and SINDy-PI after the high-priority additions; report their actual
  integration status rather than claiming all proposed baselines have been completed.

## Delivery D: evidence-backed LLM evaluator prototype

- User decision: evaluate physical credibility, not scientific explanatory value,
  novelty, familiarity, or conformity to a conventional equation. Constraints must
  have explicit provenance and applicability conditions. Unsupported physical claims
  remain unknown; absence of evidence is not a failure. Report credibility conditional
  on the declared physical setting rather than claiming an equation is universally true.
- Define a dataset-card and frozen, candidate-independent check-list schema.
- Implement evidence records for dimensional, symbolic and numerical checks, with
  pass/fail/unknown, assumptions, provenance and coverage.
- Let an optional judge consume equation and evidence, with model identity and ground
  truth withheld. Version prompts/configuration and support offline mock testing.
- Prepare a human annotation workflow and candidate-pair challenge-set generator:
  equivalent rewrites, missing/redundant terms, wrong signs, invalid domains and OOD errors.
- Distinguish generated challenge cases from expert-validated labels.
- Include order-swap/renaming checks and a plan to compare numeric-only, direct-LLM,
  and evidence-assisted judges. Do not invoke paid services or claim measured human
  agreement without data.

## Coordination and acceptance

The Sol lead may delegate two bounded independent work packages to Sol agents while
doing integration work locally. Reserve shared files (especially run_benchmark.py and
registry/export files) to one owner at a time; agree module interfaces before integration.

For each delivery, report modified files, definitions and assumptions, tests actually run,
remaining limitations, and any work that needs human annotations or unavailable assets.
Keep a progress log at docs/benchmark_v2_implementation_status.md. All code and configs
must be reviewable before any decision to deploy or rerun production experiments.

## Execution verification update

On the next user turn, the previous Sol lead was absent from the live-agent list
and had not created its implementation status file. Its acknowledgement is not
evidence of completed work or continuing background execution. The parent will
verify file artifacts and actual checks before reporting implementation progress.
