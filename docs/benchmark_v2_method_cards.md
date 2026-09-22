# Implementation provenance and applicability cards

These cards describe inspected local implementations, not reproduction guarantees.
Existing result names remain unchanged. For publication, include implementation
variant and protocol version in tables rather than only the paper's method name.

| Runner key | Actual local variant | Applicability / supplied information | Important caveat |
| --- | --- | --- | --- |
| weakform | Finite-difference library + thresholded ordinary least squares | Regular 2D spatial grids; independent response channels | Not weak form or WSINDy; no cross-channel terms |
| integral_weak_pde | Native integral weak form + normalized STLSQ | Scalar 1D grid; preregistered library `d_x^q(u^p)`; orders 0..3 | Not complete WSINDy-PDE: no adaptive supports or robust GLS |
| sindy | Native polynomial-library STLSQ | ODE derivatives or scalar 1D PDE derivative features | Rational ODEs are intentionally outside the default polynomial library; report this expressivity limit |
| pysindy | PySINDy polynomial library + SR3 | Same supplied derivative bridge as SINDy | Optional dependency; not SINDy-PI |
| symbolicgpt | Small model pretrained from scratch for each fit | SR or each ODE RHS; synthetic corpus matched to feature count | Not original large shared-pretrained checkpoint protocol; include per-case training cost |
| e2e | Optional adapter to official pretrained Transformer | SR or each ODE RHS; explicit trusted upstream checkpoint | Registered and mock-contract-tested; actual checkpoint unavailable, overlap unknown |
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

## ODE dataset cards

All four core families are dimensionless synthetic systems unless a separate
physical unit mapping is supplied. Do not invent SI units from variable names.
Known equation metadata is available to numeric recovery evaluation only, not to
the discovery method or LLM judge. Default unknown fixed coefficients are absent
from input features. Splits use whole trajectories, not interleaved time samples.

| Dataset | Family | States | Regime and caveat |
| --- | --- | --- | --- |
| ode_core_oscillator | Oscillatory | x, v | Harmonic oscillator, omega=1.3, varied initial phase/amplitude |
| ode_core_population | Population | prey, predator | Lotka-Volterra; positive initial populations |
| ode_core_chaotic | Chaotic | x, y, z | Lorenz, sigma=10/rho=28/beta=8/3; long-time pointwise errors need chaos-aware interpretation |
| ode_core_rational | Rational | s | Positive saturation decay, vmax=1.5/km=0.7; rational expressivity required for exact recovery |

Actual initial states, solver tolerances, seed and observation settings are recorded
by each generated dataset. The integration smoke tests validate finite generation,
determinism and trajectory separation, not broad scientific representativeness.

## Remaining coverage work

AI Feynman and SINDy-PI are not integrated by this change. True OOD splits, broader
noise/sparsity grids and controlled pretrained-corpus overlap need follow-up.
Runtime comparisons across hardware must remain separate from equation-quality
comparisons, with same budgets, precision, seeds, thread count and stopping policy.
