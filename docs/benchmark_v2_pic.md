# PDE PIC v3 (opt-in PyTorch reference implementation)

This package implements the Physics-informed Information Criterion from Xu et al.,
2022 as `PIC = r_loss * p_loss`. It is not the legacy additive parsimony score.
The implementation is opt-in and is not enabled in benchmark-wide runs unless
`evaluation.physics_informed_pic` is supplied explicitly.

## Protocol

`prepare_torch_pic_reference` trains one CPU PyTorch ANN on an explicitly supplied
training split, evaluates it on a rectangular `(x,t)` grid, and stores its state.
Every candidate starts from an exact clone of that state and receives the same
optimizer, collocation grid, seed, and epoch budget. At every PINN epoch, automatic
differentiation produces `u_t`, `u_x`, and `u_xx`; homogeneous TLS refits the
candidate coefficients; then the network takes one Adam step on
`MSE_observation + physics_weight * MSE_PDE`.

The reference cache key is SHA-256 over the implementation version, full config,
training-index hash, and selected `(x,t,u)` bytes. When `cache_dir` is supplied, the
ANN state and prepared derivative/reference arrays are stored in a safe NumPy archive;
subsequent processes load that archive without retraining. Observation min/max is computed
only from that training split and shared by the ANN/PINN normalization. Component
costs report reference and candidate epochs, seconds, observations, and collocation
points. ANN training RMSE and finite PINN completion are diagnostics, not claims of
convergence or correctness. Optional `reference_max_normalized_rmse`,
`pinn_min_relative_loss_improvement`, and `pinn_max_final_loss` gates turn inadequate
reference/PINN fits into explicit failure statuses. A null PINN threshold means the
paper-style fixed finite epoch budget is used; it is not described as mathematical convergence.

`original_coefficients` means coefficients submitted with the discovered equation.
It is `None` when none were submitted. `reference_fit_coefficients` is the full-grid
TLS initialization, `window_coefficients` are the r-loss fits, and
`refitted_coefficients` is the final PINN-epoch fit.

## Source audit and deviations

The implementation was checked against the official `PIC_PDE_discovery.py` at
<https://github.com/woshixuhao/PIC_code> (accessed 2026-09-21), especially
`calculate_cv` and the PINN loop. The author uses ten windows of half the time range,
shifted by one twentieth of the range, refits by SVD in every epoch, reloads one ANN
checkpoint per candidate, and uses a shared observed-data min/max.

This is a bounded adaptation, not a line-for-line reproduction:

- It currently accepts only scalar 1D equations `u_t = Theta c` and terms `1`, `u`,
  `u_x`, `u_xx`, `u_xxx`, `u*u_x`, `u*u_xx`, `u*u_xxx`, `u^2`, and `u^3`. Other derivatives, systems,
  arbitrary expression trees, and non-rectangular domains are rejected.
- It supports `tanh` and the official source's sine activation/initialization, with
  configurable CPU networks and budgets. The rational activation and the paper/code's
  larger equation-specific convergence procedure are not reproduced.
- TLS uses the explicit sign convention `u_t = Theta c`. The official file has
  inconsistent sign handling between `calculate_cv` and other regression paths.
- Window endpoints exactly follow the author's half-range windows with range/20
  shifts for ten windows. The author evaluates a fresh regular 100-by-100 grid in
  each window; this bounded port selects points from one fixed candidate-independent
  reference grid instead, so sampling density can differ near endpoints.
- The reference grid domain is derived only from training coordinates. Excluded or
  held-out coordinates and values cannot expand the domain or change its cache key.
- A TLS system with fewer rows than augmented columns is rejected. With
  `full_matrices=False`, its last returned singular vector is not a nullspace vector.
- Near-zero mean window coefficients are rejected because coefficient CV is undefined;
  no epsilon silently turns them into a score.
- Timing is wall-clock diagnostic metadata and varies by machine.

## Benchmark runner integration

`run_benchmark.py` converts a discovered additive scalar equation into the bounded
term grammar and records `pic`, `pic_r_loss`, `pic_p_loss`, status, cache identity,
all coefficient stages, window fits, costs, ANN diagnostics, and protocol version.
Unsupported equations and PIC evaluator failures do not change the baseline's own
success status. See `configs/benchmark/pic_runner_example.json`. PIC remains off in
the ordinary smoke/full configurations.

The runner always prepares the reference from the temporal training block. An optional
`max_observations` bound samples training observations deterministically while retaining
all four spatial/time domain corners. Held-out values are never included.

## Ranking calibration status

`scripts/run_pic_ranking_validation.py` generates independently integrated multimode
Burgers and KdV fields, evaluates the true structure plus a missing-term and an
extra-term candidate, and reports every candidate for three seeds. The configured
acceptance threshold is deliberately 100% true-structure top-1.

The current bounded calibration **does not pass that threshold**. Across the recorded
runs, true-structure top-1 ranged from 1/6 to 4/6 and remained sensitive to ANN seed,
reference budget, activation, and the identifiability of the trajectory. The latest
5000-epoch sine run produced 3/6. These are negative validation results, not a basis
for selecting a favorable configuration. PIC therefore remains an auxiliary,
uncalibrated metric and must not yet determine benchmark rankings.

Run the bounded example with the isolated environment or another environment containing
NumPy and CPU PyTorch:

```powershell
python scripts/run_pic_torch_example.py --config configs/benchmark/pic_torch_smoke.json
```

The example fits synthetic `u(x,t)=exp(-t) sin(x)` and evaluates the heat candidate
`u_t = u_xx`. A finite result establishes that the end-to-end implementation executes;
it is not a theorem of neural-network convergence or PDE identification consistency.
