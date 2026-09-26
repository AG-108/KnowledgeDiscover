# Implementation provenance and applicability cards

These cards describe inspected local implementations, not reproduction guarantees.
Existing result names remain unchanged. For publication, include implementation
variant and protocol version in tables rather than only the paper's method name.

| Runner key | Actual local variant | Applicability / supplied information | Important caveat |
| --- | --- | --- | --- |
| weakform | Independent scalar-1D WSINDy-PDE adaptation: compact test functions, valid FFT weak convolution, column scaling and MSTLS threshold-path selection | Uniform scalar 1D grids; conservative library `D_x^q(u^p)`; orders and supports preregistered in config | Faithful to the core weak-discretization/MSTLS method, but not the full upstream multidimensional/multifield, trigonometric, automatic spectral-support or RGLS feature set |
| sindy | Native polynomial-library STLSQ | ODE derivatives or scalar 1D PDE derivative features | Rational ODEs are intentionally outside the default polynomial library; report this expressivity limit |
| pysindy | PySINDy polynomial library + SR3 | Same supplied derivative bridge as SINDy | Optional dependency; not SINDy-PI |
| symbolicgpt | Small model with either per-fit synthetic pretraining or explicit within-case, within-instance pretraining reuse | SR or each ODE RHS; synthetic corpus matched to feature count and conditioning point count | Neither variant uses the original large shared checkpoint; keep the two protocols in separate result groups and include training cost |
| e2e | Optional adapter to official pretrained Transformer | SR or each ODE RHS; explicit trusted upstream checkpoint | Local source/checkpoint and one ODE CLI smoke verified; corpus overlap and full-paper fidelity unknown |
| dso | TensorFlow 1.x to PyTorch port | SR / each ODE RHS | Port fidelity still needs numerical comparison against official implementation |
| eqgpt | Handbook pretraining, surrogate, generation/optimization | Scalar 1D PDE | Omits original final PINN refinement; exact exclusion mapping currently only `kdv` |
| physo | Dimensionless current default | SR / each ODE RHS | Does not exercise physical-unit priors without explicit unit cards |
| pdefind | Thresholded least squares wrapper | Scalar 1D PDE derivative library | Current wrapper is not STRidge |
| pdenet | Example-wrapper configuration | 2D PDE fields | Periodic padding and one x-derived spacing need dataset-specific applicability review |

## Training-data overlap policy

EqGPT's `_KNOWN_EQUATION_NAMES` maps only `kdv` to `KdV equation`; unknown keys
train on the full handbook. Therefore non-KdV rows have **unknown/uncontrolled
target overlap**, not verified exclusion. Even row exclusion does not exclude
algebraically equivalent or close-family examples. Audit corpus equations after
canonicalization and report exact/family overlap separately before claiming a
held-out-pretraining result. This audit does not alter ongoing experiments.

SymbolicGPT's per-case random synthetic corpus also needs overlap logging; it
must not inherit an overlap claim from the original paper. For official pretrained
models, record source revision, checkpoint SHA256, upstream training-corpus
description, and whether benchmark overlap is known, excluded, or unknown.

SymbolicGPT's default `fit()` trains from scratch for every target. The
`synthetic_per_fit` protocol therefore generates and trains a new small model
for each ODE RHS. The opt-in `synthetic_case_instance_reuse` protocol generates
and trains once for each dataset instance in a model/dataset/seed case, then
copies the model and restores its post-training random state for each RHS;
candidate sampling and constant fitting still run separately per RHS. It is
not an official shared checkpoint or a reusable model across processes,
datasets, seeds, cases or instances. The result JSON records
`method_provenance.protocol` and `cache_hit`; on a target that never finishes,
`cache_hit: null` means the cache state at interruption is unknown.

The paired diagnostic configurations are
`configs/benchmark/tuning/symbolicgpt_case_reuse.json` and
`configs/benchmark/tuning/symbolicgpt_per_fit.json`. They share the full
3000-corpus/50-epoch/100-candidate budget, zero DataLoader workers, seed,
datasets and 300-second case timeout. Only the reuse switch and output
directory differ. Both currently use `configured_fallback` operators because
the Core ODE datasets do not declare a frozen benchmark function set. These
diagnostics must be reported separately from any declared-grammar main track.
The matched seed-0 damped-oscillator result and its denominators are recorded
in `docs/reports/symbolicgpt_protocol_comparison_20260925.md`.

## ODE dataset cards

The eight Core ODE Track systems and four supplementary generated systems are
dimensionless unless a separate
physical unit mapping is supplied. Do not invent SI units from variable names.
Known equation metadata is available to numeric recovery evaluation only, not to
the discovery method or LLM judge. Default unknown fixed coefficients are absent
from input features. Splits use whole trajectories, not interleaved time samples.

| Dataset | Family | States | Regime and caveat |
| --- | --- | --- | --- |
| ode_core_oscillator | Oscillatory | x, v | Harmonic oscillator, omega=1.3, varied initial phase/amplitude |
| ode_core_population | Population | prey, predator | Lotka-Volterra; positive initial populations |
| ode_core_chaotic | Chaotic | x, y, z | Lorenz, sigma=10/rho=28/beta=8/3; long-time pointwise errors need chaos-aware interpretation |
| ode_core_rational | Michaelis--Menten (core) | s | Positive saturation decay, vmax=1.5/km=0.7; rational expressivity required for exact recovery |
| ode_core_damped_oscillator | Damped oscillatory | x, v | Linear damping with zeta=0.1; distinct from the undamped oscillator |
| ode_core_pendulum | Nonlinear oscillatory | theta, v | Damped pendulum with sin(theta); a polynomial-only library cannot exactly recover it |
| ode_core_duffing | Nonlinear oscillatory | x, v | Autonomous damped Duffing oscillator with cubic restoring force |
| ode_core_van_der_pol | Nonlinear oscillatory | x, v | Self-excited oscillator with state-dependent damping, mu=2 |
| ode_core_sir | Epidemic | susceptible, infected, recovered | SIR compartments; clean trajectories conserve total population |
| ode_core_fitzhugh_nagumo | Excitable | v, w | Two-state excitable dynamics with cubic nonlinearity and slow recovery |
| ode_core_brusselator | Chemical | x, y | Two-state autocatalytic reaction model with cubic interaction |
| ode_core_robertson | Stiff chemical | x, y, z | Stiff reaction kinetics; Radau solver and logarithmic observation times; clean trajectories conserve total concentration |

Core ODE Track v1 comprises `ode_core_damped_oscillator`, `ode_core_pendulum`,
`ode_core_duffing`, `ode_core_van_der_pol`, `ode_core_sir`,
`ode_core_fitzhugh_nagumo`, `ode_core_rational`, and `ode_core_robertson`.
The harmonic oscillator, population, chaotic, and Brusselator rows above are
supplementary; `ode_core_rational` is the single Michaelis--Menten case, not a
separate ninth system. All methods declaring ODE support should be assessed on
the same eight-case set, with execution failures and grammar-ineligible targets
reported explicitly.

Actual initial states, solver/tolerances, time sampling, seed and observation settings are recorded
by each generated dataset. The integration smoke tests validate finite generation,
determinism and trajectory separation, not broad scientific representativeness.

## Remaining coverage work

AI Feynman and SINDy-PI are not integrated by this change. True OOD splits, broader
noise/sparsity grids and controlled pretrained-corpus overlap need follow-up.
Runtime comparisons across hardware must remain separate from equation-quality
comparisons, with same budgets, precision, seeds, thread count and stopping policy.

## Weak-form implementation scope


The public `weakform` runner key now uses
`kd.model.kd_wsindy.WSINDyPDEModel`. The previous two-dimensional
finite-difference/least-squares adapter was removed, as was the temporary
`integral_weak_pde` benchmark key that would otherwise duplicate this method.
`IntegralWeakPDEModel` remains only as a Python import alias for compatibility;
it is not a separately ranked baseline.

For a scalar field on a uniform one-dimensional spatial grid, every candidate
column and the time-derivative target are formed by valid sliding convolution
against compact polynomial test functions. Integration by parts moves all
derivatives onto those test functions, so the fit path never differentiates the
observed field. Candidate columns are normalized before regression, an MSTLS
threshold path is scored by projection change plus support size, and the selected
model is refit and exported in ordinary jet-variable syntax. Conservative terms
remain available in `weak_expression`; for example, `D_x(u^2)` is expanded to
`2*u*u_x` in the common evaluator expression.

The implementation was independently written against the algorithm in the
[official WSINDy-PDE repository](https://github.com/MathBioCU/WSINDy_PDE) and
paper; reference commit `d9296be4c17c5e0b4df14472f4cd8276a8ae4eed` was audited.
It implements the core convolutional weak discretization and MSTLS selection,
not every upstream option. In particular, the benchmark adapter does not expose
multidimensional/multifield libraries, trigonometric and mixed derivatives,
automatic Fourier-corner support selection, or upstream RGLS variants. Supports,
library degree/order, threshold path, scaling diagnostics, selected threshold,
weak residuals, and sample counts are recorded so this scope is testable.


## ODE generation and observation protocol

`kd.dataset._ode_core.generate_ode_core` returns the existing `ODEDataset`;
equations, variables, parameters, every initial condition,
seed, sample count, solver/tolerances, time sampling, and the whole-trajectory
split requirement are recorded in `generation_metadata`. Robertson uses Radau
and logarithmic observation times; the other systems use DOP853 and uniform
observation times. All twelve are registered as `ode_core_<system>`. The current benchmark
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
Robertson's very short initial time steps amplify observation noise during
finite-difference derivative estimation; the 1% noisy/sparse setting is a
numerical stress case and must be assessed separately from clean recovery.
The five-seed nonuniform-grid audit in
`docs/reports/robertson_observation_audit_20260925.md` confirms low clean derivative
error but severe 1% noise amplification; no time cutoff or alternative
derivative estimator has been adopted for a formal noisy track.

`configs/benchmark/core_ode_track.json` lists all currently integrated methods
that declare ODE support and the same eight datasets. Its five-seed full-profile
manifest has 440 method/dataset/seed cases; this is a coverage target, not a
completed run. Checkpoints/endpoints and optional packages are still required
for some methods. `core_v2.json` selects a smaller 70-case, five-seed integration
suite using native SINDy and WSINDy-PDE; `core_v2_ode_noisy_sparse.json` selects
40 ODE cases and applies 1% noise and stride 2. These use a degree-3 polynomial
library for the three cubic core systems and degree 1 for the damped oscillator;
the pendulum's sine and Michaelis--Menten's rational term remain
outside the polynomial SINDy library. This is an explicit expressivity limit.
None of these configs has been launched as a production benchmark. Five runner seeds
vary split/algorithm randomness on a fixed generator seed (default 0); varying
the generator requires an explicit dataset seed override and separate reporting.
`initial_condition_scale` shifts the entire cohort: it is not a dedicated OOD
train/test split. A true fixed-ID OOD holdout remains pending.
An eight-case, one-seed SINDy execution smoke under
`results/core_ode_track_sindy_smoke_20260923/` produced 17 derivative-target
rows, all with execution status `ok`; its 7/17 structural recovery is a smoke
result, not a cross-method Core ODE Track score.


E2E assets are documented in [the E2E guide](benchmark_v2_e2e.md).
