# Dataset-aware candidate tuning (2026-09-23)

These are staged exploratory runs, not paper reproductions. Archived benchmark outputs remain unchanged. Counts below are single-seed pilots; use the recorded result files and manifests for each row.

## Reporting protocol

For the 268 synthetic SR cases with a trusted reference expression, 15 dataset declarations omit a literal unary function required by their target: Keijzer (8), Korns (4), Nguyen-8, Constant-5, and Vladislavleva-6. Six other SR cases have no trusted reference expression for this audit. Nguyen-8, for example, is `sqrt(x1)` although its declared Koza candidate set omits `sqrt`.

The **main SR/ODE track** uses `search_space_policy=declared` and the dataset's published `function_set`. The **reference diagnostic** uses `declared_plus_reference`, which adds only missing literal unary functions (`sin`, `cos`, `exp`, `log`, `sqrt`, `tan`, `tanh`, `abs`) from known truth. It is oracle-assisted and must never be pooled into the main leaderboard. Keep data, seed, search budget, constants, and all other model parameters identical. Every newly tuned SR configuration defaults to the main policy; use the CLI flag for a matched diagnostic:

```powershell
python run_benchmark.py --config configs/benchmark/tuning/gplearn_dataset_repaired.json --search-space-policy declared --output-dir results/tuning/gplearn_main
python run_benchmark.py --config configs/benchmark/tuning/gplearn_dataset_repaired.json --search-space-policy declared_plus_reference --output-dir results/tuning/gplearn_reference_diagnostic
```

Results record the resolved candidate operators in `search_space`; each case identity and manifest include the effective model parameters. An absent operator makes a target unrepresentable in that candidate grammar, so a zero recovery score must not automatically be read as a search failure. The repair is literal-function-only: aliases, variable exponents, numeric constants, protected operator semantics, and the `poly` terminal can still limit representability.

PDE datasets do not publish the same `function_set` metadata. The PDE main configuration uses each method's generic library. Its paired diagnostic narrows candidates from the *known reference equations* only for Burgers, KdV, and Chafee–Infante; all other PDE cases retain the generic library. This diagnostic is also oracle-assisted. Both PDE configurations use identical non-library budgets. New PDE results record `search_space.policy` and the effective library parameters; PDE-FIND also records its resolved terms.

## Matched pilots

All structural counts use metric protocol 2.1. SR/ODE `exact_recovery` is additive-term structure ignoring numeric coefficients; PDE `pde_support_recovery` is the PDE term-support metric. Runtime is the sum of case runtimes, not a suite estimate.

| Method / sample | Main completed / recovered | Reference diagnostic completed / recovered | Main median test NRMSE | Diagnostic median test NRMSE |
| --- | ---: | ---: | ---: | ---: |
| gplearn, 6 SR cases | 6/6, 1/6 | 6/6, 2/6 | 0.104007 | 0.092526 |
| PySR, 2 SR cases | 2/2, 1/2 | 2/2, 2/2 | 2.06e-6 | 5.73e-10 |
| PDE-FIND, 3 named PDE cases | 3/3, 2/3 | 3/3, 2/3 | not an SR metric | not an SR metric |
| Weak-form, same 3 PDE cases | 3/3, 3/3 | 3/3, 1/3 | not an SR metric | not an SR metric |

For gplearn the main and diagnostic sum to 45.8 s and 39.8 s respectively. For PySR they sum to 37.0 s and 41.4 s. The SR improvement occurs on Nguyen-8, the known missing-`sqrt` case. The six gplearn cases are GrammarVAE-1, Koza-2, Nguyen-7, Nguyen-8, Korns-1, and Korns-7; the PySR cases are GrammarVAE-1 and Nguyen-8. This tiny sample cannot establish a general recovery rate.

The PDE pilot includes Burgers, Chafee–Infante, and KdV. Narrowing the weak-form dictionary made it omit a true term on Burgers and Chafee–Infante, despite both cases succeeding with the generic dictionary. KdV PDE-FIND still omitted the small `u_xxx` term even when the dictionary contained only the two true terms; its equation residual NRMSE was about 3.39 on both tracks. Candidate pruning alone did not fix coefficient selection. The main PDE-FIND library now includes `u^3`, and the wrapper now honors `derivative_order` and explicit `terms` during both fitting and rollout; that code change means these new results cannot be directly compared to its older fixed-library archive.

On four local ODE core systems, SINDy and PySINDy each produced eight derivative-target rows. Lower sparse thresholds reduced recovery from 3/8 to 2/8 and from 4/8 to 3/8, so those thresholds were reverted. With the original thresholds, degree 1 for the oscillator and degree 3 for the rational system retained 3/8 and 4/8. The rational approximation's test NRMSE fell from about 0.156 to 0.057, but a polynomial library cannot exactly recover its rational structure.

A deliberately tiny SymbolicGPT interface smoke run (24 generated equations, one training epoch, two sampled candidates) produced an invalid sampled expression and zero finite predictions. This is a failed interface-quality pilot, not a score for the 250-equation/eight-epoch tuning configuration. SymbolicGPT's generator also implements `sqrt` as `sqrt(abs(x))`, which can differ structurally from a declared `sqrt(x)` target even on a positive test domain.

## Configuration coverage and limits

| Applicable method(s) | Main configuration | Validated locally |
| --- | --- | --- |
| gplearn | `gplearn_dataset_repaired.json` | 279-case manifest; six SR fits |
| PyOperon | `pyoperon_dataset_repaired.json` | 279-case historical manifest; 0.6.1 installed in isolated Python 3.10 `kd-operon` environment; one ODE smoke case completed |
| PySR | `pysr_dataset_repaired.json` | 279-case manifest; two paired SR fits |
| PhySO | `physo_dataset_repaired.json` | 279-case manifest; tuned config2 fit pending |
| DSO | `dso_dataset_repaired.json` | 279-case manifest; tuned fit pending |
| DISCOVER (DSCV) | `discover_dataset_adaptive.json` | 279-case SR/ODE manifest; fit pending |
| SymbolicGPT | `symbolicgpt_dataset_adaptive.json` | 279-case manifest; tiny smoke failed as above |
| SINDy / PySINDy | `sindy_polynomial_adaptive.json` | 56-case manifest; four ODE systems fitted |
| 11 PDE methods (DSCV, SPR, SGA, DLGA, DeepMoD, PDE-FIND, PDE-Net, weak-form, EqGPT, SINDy, PySINDy) | `pde_library_declared.json` and `pde_library_reference_diagnostic.json` | 253-case manifests each; PDE-FIND and weak-form paired on three systems |

The per-dataset overrides target all applicable cases, not every incompatible method/dataset pair. For SR/ODE, cases without a declared function set use the method's recorded fallback grammar. In the PDE configurations, SGA, EqGPT, and PDE-Net expose only their existing internal or fixed library through this benchmark interface, so their changes are coarse budgets, not per-dataset operator vocabularies. PDE-Net accepts only 2-D grids in this adapter and will skip the 1-D pilot PDEs. SPR is PDE-only. E2E's full candidate generation rules are fixed by its pretrained decoder; its official source/checkpoint and checksum are configured only in a local smoke override, not in the tracked benchmark config. LLM-SR has no configured completion endpoint/model, so no batch should be launched until a real minimal generation-and-fitting call succeeds. Neither method has a validated candidate-set tuning run here.

The unrun 279-case and 253-case manifests prove configuration selection only. They do not establish runtime success or better accuracy. Full local execution was intentionally deferred because of cost; schedule per-method batches on suitable hardware, keep both tracks paired, and report completed, error, skipped, and timeout denominators separately.

After the earlier expansion from four to twelve available synthetic ODE systems,
the 279-case and 56-case counts in the
table are historical pilot manifests. Regenerating the current gplearn and
SINDy/PySINDy manifests selects 287 and 72 cases, respectively, with one seed.
These added cases have not been included in the matched tuning results above.
These broad tuning manifests are exploratory and include the four systems now
classified as supplementary; they are not the fixed eight-system Core ODE Track.
The current SINDy/PySINDy configuration sets degree 3 for Duffing, Van der Pol,
FitzHugh--Nagumo, and Brusselator, whose declared equations contain cubic terms,
and degree 1 for the damped oscillator. The pendulum sine remains unavailable to
the polynomial library; its exact-recovery denominator needs that expressivity
caveat. These library overrides have not been tuned against full-run outcomes.
