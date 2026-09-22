# Follow-up review tasks (local only)

## Evaluation owner

Root has finished registering `integral_weak_pde`, its config/example and the ODE
catalog/truth hook. Preserve these changes. Runner ownership is returned to you.
Fix the remaining integration gaps in runner/evaluation modules and tests:

- `symbolic_pde_rollout_metrics` still executes explicit Euler with `np.gradient`
  and no boundary declaration. Connect a genuine shared method-of-lines path for
  every parseable equation, including native-predictor baselines. Default must
  abstain with explicit status when boundary/reference metadata is absent. Native
  predictor remains separate. Restrict to uniform 1D periodic grids initially if
  necessary; require endpoint convention, finite difference/spectral settings,
  reference validation and explicit tolerances. Do not claim unused BC dictionaries
  enforce boundary conditions. Add analytic advection/diffusion checks and tests
  for missing metadata and failed reference validation. Record raw legacy fields
  only as legacy; do not aggregate them into validated equation rollout scores.
- `information_criteria` currently standalone. Wire opt-in evaluator metadata
  from dataset/evaluation configuration, refuse undeclared parameter counts;
  estimated Gaussian variance is an additional free parameter, explicitly recorded.
- `seed_family_summary` currently pools all models together. Group by method,
  family and protocol; never pair duplicate dataset/target/seed keys silently or
  mix protocol/profile/regime in pairing. Wire saved family/seed reports. Expose
  available-case coverage rather than calling partial-case means unbiased.
- `_score_regression` currently treats arbitrary evaluator exceptions as algorithm
  failures. Separate invalid model predictions from evaluator infrastructure errors.
- Root fixed ODE beta symbol parsing; family metadata now propagates from catalog.
- Run actual focused tests and accurately document remaining limits. Do not edit
  root-owned ODE/weak-form modules, E2E or credibility/PIC files.

## Physical-credibility owner

Review fixes required in your exclusive credibility files:

- Current `MappingProxyType` is shallow; recursively freeze nested parameters,
  checks, domains, assumptions, observations so fingerprints/check definitions
  cannot change through caller-owned lists/dicts. Test mutation after construction.
- Known dimensional contradictions (adding incompatible units, sin of a dimensional
  argument) must fail, not be indistinguishable from missing-unit unknown.
- Bare fitted numeric constants may have unknown units. Require an explicit
  candidate-independent numeric-literal policy or nondimensional declaration,
  instead of silently treating all fitted coefficients as dimensionless. Document.
- Domain check currently checks lhs-rhs and demands the output variable domain.
  Offer explicit fixed target `rhs`/`lhs`/`residual` and use RHS for explicit-law
  validity. Sign checks must support `rhs` so they can test candidate-dependent
  dissipation/derivative signs, not only a fixed unrelated expression like time.
- Check expressions may name a declared variable absent from candidate: avoid
  `symbols[k]` KeyError; construct symbols from card/check names consistently.
- `_probe_points` constructs all 3^N combinations BEFORE slicing. Bound generation
  with islice and use candidate-independent variable order from the frozen card
  so renaming cannot change selected probes under small budgets. Report discrete
  probe coverage honestly. No continuous-proof claims.
- Parse failure should yield evaluation_error/unknown evidence, not crash report.
- SymPy simplification can hide holes such as x/x at zero. Preserve the original
  expression domain for validity checks, with regression tests.
- Tests must actually compare equivalent rewrites and candidate order reversal,
  not just reversal of checklist order. Keep unlabeled expert workflow.

## E2E adapter owner

Implement optional official-pretrained adapter in NEW `kd/model/kd_e2e.py`, tests,
docs and example only. Root will register runner/config after handoff. Read-only
sources: facebookresearch/symbolicregression main Example.ipynb and
symbolicregression/model/sklearn_wrapper.py. Notebook directly torch.loads a full
model object from model1.pt and passes it to SymbolicTransformerRegressor(model,
max_input_points=200,n_trees_to_refine=100,rescale=True). No checkpoint is present.

Explicit checkpoint path and source checkout required; never auto-download or
fallback to training. Full pickle load is executable: require explicit trusted
checkpoint opt-in + expected SHA256, record loaded package source/commit/device.
Optional import isolated. Fit(X,y,variable_names=None), best_expression_, predict.
Use official `retrieve_tree(with_infos=True)` relabed_predicted_tree, preserving
feature selection mapping. Official predict currently passes nonexistent tree_idx
to retrieve_tree: prefer selected relabeled tree via model.env.simplifier on full
X, document the narrow compatibility workaround. Audit exact infix operations.
Record checkpoint load time and fit total; use a forwarding timer around official
model inference to split inference from residual refinement/preprocessing time,
or report unavailable timing rather than pretending split is measured. Record
pretraining cost unavailable/amortized, no per-case pretraining. Seed explicit.
Mock API contract tests plus missing-dependency/checkpoint/hash failures. Clearly
mark real-checkpoint validation unavailable, not a completed method reproduction.

Shared test interpreter:
`C:\Users\22412\.cache\kd-benchmark-v2-venv\Scripts\python.exe`.
Use apply_patch, preserve dirty worktree, no remote changes or paid APIs.
