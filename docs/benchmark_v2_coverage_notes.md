# Benchmark v2 coverage notes

## Integral weak-form PDE baseline

`kd.model.kd_wsindy.IntegralWeakPDEModel` is a native, restricted 1D scalar
PDE baseline.  It forms every target and feature by quadrature and transfers
observed time and space derivatives to compact polynomial test functions via
integration by parts.  Thus it never differentiates the measured field.

This implementation is **not** a reproduction of the complete WSINDy-PDE
algorithm.  The official MATLAB repository (`dm973/WSINDy_PDE`, also linked
from the MathBioCU projects) includes adaptive test-function/support choices,
library tagging and selection, scaling, and robust generalized least-squares
routines.  The native baseline currently supports only regular 1D grids and
conservative library columns `d_x^q(u^p)`.  Register it under a distinct key
`integral_weak_pde`; retain the existing `weakform` name and its honest
finite-difference description for old-result readability. Runner integration now
calls `fit(dataset.usol, dataset.x, dataset.t)` on scalar 1D `GridPDEDataset`
objects. Exported expressions expand conservative derivatives into ordinary
jet variables (`d_x(u^2)` becomes `2*u*u_x`); original weak expressions remain
available. Derivatives are limited to orders 0 through 3 for the chosen test
function smoothness. Validation rejects nonfinite grids and invalid budgets.

## ODE core

`kd.dataset._ode_core.generate_ode_core` provides harmonic oscillator,
Lotka--Volterra population, Lorenz chaotic, and rational saturation-decay
systems.  It returns the existing `ODEDataset`; equations, variables,
parameters, every initial condition, seed, sample count, DOP853 method, rtol,
atol, and the whole-trajectory split requirement are recorded in
`generation_metadata`. Registered catalog keys are `ode_core_oscillator`,
`ode_core_population`, `ode_core_chaotic`, and `ode_core_rational`, each
implemented as a small wrapper around the generator.  The current benchmark
runner already splits `ODEDataset.n_traj` before flattening, so registering
these objects preserves trajectory-level holdout. By default fixed governing
parameters are **unknown constants**, not features supplied to the method; the
ground-truth equations substitute their numerical values. An explicitly separate
`expose_parameters=true` setting supplies known fixed covariates, and must not be
described as parameter-generalization evaluation.

`observation_noise` injects Gaussian noise relative to each trajectory/state's
standard deviation before derivatives are estimated. `time_stride` subsamples
the generated time series. Independent noise RNG streams preserve identical
initial conditions across clean/noisy/sparse variants. Derivative targets for
noisy data are finite-difference estimates, not clean oracle derivatives.

`configs/benchmark/core_v2.json` is a modest 50-case, five-seed integration suite
using native SINDy and integral weak PDE only; it is not the final representative
baseline lineup. `core_v2_ode_noisy_sparse.json` applies 1% noise and stride 2.
Neither config has been launched as a production benchmark. Five runner seeds
vary split/algorithm randomness on a fixed generator seed (default 0); varying
the generator requires an explicit dataset seed override and separate reporting.
`initial_condition_scale` shifts the entire cohort: it is not a dedicated OOD
train/test split. A true fixed-ID OOD holdout remains pending.

## Official pretrained E2E Transformer

The official archived `facebookresearch/symbolicregression` repository needs
its own source checkout/environment, an explicit checkpoint directory passed
through `evaluate.py --reload_checkpoint`, and legacy dependencies including
PyTorch (the README reports 1.3) plus the `pakamienny/sympytorch` fork.  Its
example notebook refers to a pretrained model, but no checkpoint is present in
this worktree and none was downloaded.  A future adapter should run in an
isolated optional environment, validate the checkpoint before claiming method
availability, translate tabular arrays through the official encoder/decoder,
and report checkpoint inference separately from any tuning/refinement cost.
An optional `E2ETransformerModel` adapter is now registered as `e2e`, with
checkpoint/trust/SHA256 gates and offline API contract tests. See
`benchmark_v2_e2e.md`. Until actual assets and compatibility checks exist, it is
reported as unavailable rather than replaced by per-case training. Real-checkpoint
performance and full paper-protocol fidelity remain unverified.
