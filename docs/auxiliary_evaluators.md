# Auxiliary evaluators: PIC and physical credibility

Both evaluators are opt-in and not validated for ranking. The plan recommends
reporting them in an appendix or as future work.

## PDE PIC (PyTorch reference implementation, v3)

Implements the Physics-informed Information Criterion of Xu et al. (2022) as
`PIC = r_loss * p_loss` (not the legacy additive parsimony score). Enabled only
through `evaluation.physics_informed_pic`; see `configs/benchmark/pic_runner_example.json`.

Protocol: `prepare_torch_pic_reference` trains one CPU ANN on the temporal training
block and evaluates it on a rectangular `(x,t)` grid derived only from training
coordinates. Each candidate starts from an exact clone with the same optimizer,
collocation grid, seed and epoch budget. Every PINN epoch computes `u_t`, `u_x`,
`u_xx` by autodiff, refits coefficients by homogeneous TLS, then takes one Adam step
on `MSE_observation + physics_weight * MSE_PDE`. The reference cache key is SHA-256
over version, config, training indices and data bytes; cached state is a safe NumPy
archive. Optional gates (`reference_max_normalized_rmse`,
`pinn_min_relative_loss_improvement`, `pinn_max_final_loss`) turn inadequate fits
into explicit failure statuses. Results record `original_coefficients` (submitted),
`reference_fit_coefficients`, `window_coefficients` and `refitted_coefficients`,
plus costs and ANN diagnostics. PIC failures never change the baseline's status.

Deviations from the official `PIC_PDE_discovery.py`
(<https://github.com/woshixuhao/PIC_code>, accessed 2026-09-21): only scalar 1D
`u_t = Theta c` with terms `1, u, u_x, u_xx, u_xxx, u*u_x, u*u_xx, u*u_xxx, u^2, u^3`;
`tanh` or sine activations (no rational activation or equation-specific convergence
procedure); explicit TLS sign convention; windows follow the author's ten
half-range windows shifted by range/20 but sample one fixed grid; underdetermined
TLS systems and near-zero mean window coefficients are rejected.

Calibration status: `scripts/run_pic_ranking_validation.py` evaluates the true
structure and missing/extra-term candidates on multimode Burgers and KdV for three
seeds against a 6/6 true-structure top-1 threshold. Recorded runs ranged 1/6 to
4/6; the latest 5000-epoch sine run gave 3/6. **PIC does not pass and must not
determine rankings.** Smoke: `python scripts/run_pic_torch_example.py --config
configs/benchmark/pic_torch_smoke.json` (execution check only).

## Physical credibility (offline prototype)

Scope (user decision): assess consistency with observed evidence and declared
physical assumptions on a specified domain. Do not score explanatory value,
novelty, familiarity or narrative appeal. Passing checks does not prove an equation
is the governing law.

Each check has a stable identifier, property, source and version; whether it is
supplied, derived or hypothesized (hypotheses are diagnostic only); applicable
variables, units, regime, domain and boundary conditions; a candidate-independent
definition with tolerances and budget; and one of `pass`, `fail`, `unknown`,
`not_applicable`, `evaluation_error`, with reproducible evidence. Cards are frozen
before viewing methods or candidates.

Implemented: dimensional homogeneity, fixed-domain real/finite probes, and declared
sign constraints on `lhs`, `rhs` or `residual`. Missing units (including unknown
coefficient units) give `unknown`; known contradictions give `fail`; evaluator
exceptions give `evaluation_error`, never a physical failure. Numeric literals other
than -1/0/1 have unknown units unless a reviewed card sets
`numeric_literal_policy="dimensionless"`. Probes are bounded diagnostics, not proofs;
unsimplified parse trees keep removable singularities such as `x/x` detectable.
Candidate text is parsed with an AST allowlist.

The optional LLM judge is dependency-injected and offline. It receives a hashed
payload with the normalized equation, evidence, physical setting, units, assumptions
and check definitions, but no method identity, ground truth or dataset name;
responses containing protected fields are rejected. It may not override measured
evidence.

Not done: expert labels, judge comparison, any composite score.
`python scripts/generate_physical_credibility_challenges.py --output-dir
artifacts/credibility` generates blank candidate-pair annotation sheets (equivalent
rewrites, wrong signs, missing/redundant terms, invalid domains); generator intent
is not a label. Validation should compare numeric-only, direct-LLM and
evidence-assisted judges on expert agreement, violation detection, unsupported
claims, abstention and sensitivity to order, renaming and equivalent forms.
