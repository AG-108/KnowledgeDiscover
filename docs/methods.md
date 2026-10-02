# Method and dataset cards

当前工程验收见 [验收摘要](validation.md)，来源条件见 [待解决问题](deferred_source_issues.md)。数据 domain-v4；新增轨指标 2.3。

These cards describe the inspected local implementations, not reproduction
guarantees. Report the implementation variant and protocol version in tables, not
only the paper's method name. Budget and fairness rules are in
[the plan](benchmark_plan.md); metric definitions are in [metrics](benchmark_metrics.md).

## Implementation variants

| Runner key | Tasks | Local variant | Important caveat |
| --- | --- | --- | --- |
| `dscv` | SR, ODE, PDE | DISCOVER adapter | CPU only; all six historical SR full runs timed out |
| `spr` | PDE | DISCOVER/SPR with PINN (CUDA) | Search part runs on CPU |
| `sga` | PDE | SGA with SolverConfig presets | SVD nonconvergence on newly admitted grids |
| `dlga`, `deepmod`, `eqgpt`, `pdenet` | PDE | Neural derivative / surrogate methods (CUDA) | EqGPT omits the original final PINN refinement; PDE-Net accepts only 2D grids |
| `pdefind` | PDE | Thresholded least squares wrapper | Not STRidge |
| `weakform` | PDE | Independent scalar-1D WSINDy-PDE (see below) | Not the full upstream feature set |
| `sindy` | ODE, PDE | Native polynomial-library STLSQ | Cannot represent sine or rational terms |
| `pysindy` | ODE, PDE | PySINDy polynomial library + SR3 | Not SINDy-PI |
| `dso` | SR, ODE | TensorFlow 1.x to PyTorch port | Port fidelity needs a numerical comparison with the official code |
| `symbolicgpt` | SR, ODE | Small model, synthetic pretraining per fit (default) or reused within one case instance | Neither variant is the original shared checkpoint; report the two protocols separately and include training cost |
| `e2e` | SR, ODE | Official pretrained Transformer (see below) | Corpus overlap and full-paper fidelity unknown |
| `gplearn`, `pysr`, `pyoperon` | SR, ODE | Library defaults through their Python APIs | PySR needs Julia |
| `physo` | SR, ODE | Dimensionless default | Physical-unit priors unused without unit cards |
| `llmsr` | SR, ODE | LLM-SRBench LLM-SR searcher (see below) | Backbone must be served separately |
| pdefind_nd | PDE | 多维内部差分 + 同库 STLS；u_tt 与配置坐标基 | 观测残差子轨，缺来源几何时阻塞 |
| green_dso | green | Uniform grammar DSO integral-response search | Joint forcing fit; no reference/test in optimizer or early stopping |
| green_basis | green | Fixed finite kernel basis response regression | Diagnostic; small residual does not prove arbitrary-kernel uniqueness |
| implicit_pde | PDE, steady_pde | 规范化稀疏零空间 / 已知源 STLS | 拒绝零、自复制、共同趋零因子；歧义不计唯一恢复 |

Current full-profile budgets in `configs/benchmark/baselines/` include ad hoc
reductions (for example PhySO and DeepMoD 20 epochs, DSCV 10 iterations). Under
decision D3 of the plan they must be replaced by upstream defaults or justified here
before the main run.

## D2 method parallelism (C7)

The `full` profile allocates the four cores of each D2 slot as follows. These are
hardware-protocol choices, shared across datasets and SR/ODE tasks, rather than
dataset-specific tuning. Search/evaluation budgets retain their existing values.

| Method | Full setting | Numerical threads per search task |
| --- | --- | --- |
| gplearn | `n_jobs=4` | 1 BLAS/OpenMP thread per joblib worker |
| PySR | `parallelism=multiprocessing`, `procs=4`, `deterministic=false` | 1 Julia/JuliaCall and BLAS thread per process |
| Operon | `n_threads=4` | 1 BLAS/OpenMP thread per native search task |
| LLM-SR | `num_evaluators=4` | 1 numerical thread per evaluator; local vLLM shares the GPU slot |
| torch methods | `torch.set_num_threads(4)` in each target worker | 4 threads inside the allocated slot |

PySR's `procs` is ignored in multithreading mode in the installed API; multiprocessing
makes the process count explicit. JuliaCall also defaults to `auto` independently of
`JULIA_NUM_THREADS`, so slot workers set both `JULIA_NUM_THREADS` and
`PYTHON_JULIACALL_THREADS`. Parallel PySR searches are not deterministic; smoke keeps
serial/deterministic mode. gplearn and Operon smoke profiles retain one search worker.
LLM-SR already defaulted to four evaluators; the full profile now records that choice
explicitly.

`run_slots.py` resolves method parallelism from the actual slot before hashing the new
case and writing its manifest. Full settings use four cores, including for source
manifests with stale or excessive worker settings; smoke settings are capped at the
allocated cores. Source manifests are preserved. GPU model constructor devices follow
the slot's exposed `cuda:0` device. Results record the applied `thread_environment`.
The resource policy additionally identifies `parallelism_protocol=d2-parallel-v1`, so
old C5 resource histories remain readable and cannot be pooled with the new policy.

Direct diagnostic runners do not enforce D2 affinity or these per-task thread limits.
Native Linux/CUDA/vLLM deployment verification and the remaining budget/grammar audits
are still required before the main experiment.

## Search spaces

SR datasets publish a `function_set`. The main track uses
`search_space_policy=declared`. The diagnostic `declared_plus_reference` adds only
missing literal unary functions (`sin`, `cos`, `exp`, `log`, `sqrt`, `tan`, `tanh`,
`abs`) taken from the known truth; it is oracle-assisted and never enters the main
table. Fifteen of the 268 reference-known synthetic SR declarations omit a unary
function their target needs (Keijzer 8, Korns 4, Nguyen-8, Constant-5,
Vladislavleva-6), so zero recovery there is partly unrepresentability. Results
record the resolved operators in `search_space`.

Core ODE and PDE datasets declare no function set. Cases without one currently use
each method's `configured_fallback` grammar, which must not be pooled with
`declared` results. The plan requires one shared Core ODE grammar declared before
the main run. Files in `configs/benchmark/tuning/` are exploratory and contain
per-dataset overrides; they are not main-track configurations.

## Applicability

`python run_benchmark.py --check-compatibility` loads a minimal instance of every
selected pair and runs adapter preflight without fitting. It verifies entry into
training only. Current engineering coverage and source boundaries are recorded in
[validation](validation.md); compatibility checks alone do not certify numerical correctness:

| Method group | Compatible PDE inputs |
| --- | --- |
| DSCV, SPR, SGA, DLGA, DeepMoD, PDE-FIND, WSINDy, EqGPT, SINDy, PySINDy | 18 scalar 1D regular-grid names: the previous 13 plus `tlc_burgers1d`, `heat`, `allen_cahn`, `convection_diffusion`, `burgers_sine`; SGA has known fitting failures |
| PDE-Net | Rectangular/anisotropic grids supported with verified periodic endpoint-excluded boundaries; retained WDwake/TLC lack that evidence |
| pdefind_nd | 两份恢复的 TLC 2D 网格及 Burgers2D/Wave/KG；有已验证几何时支持守护散点导数 |
| Source blocked | 六份保留的 TLC 散点缺几何/界面证据 |

SR and ODE input preparation admits their registered method groups; this does not
establish scientific validity or successful fitting. Under `domain-v4`, CYT uses
coordinate-only spatial blocks. Solid laws use 63 source DIF curves, 12 source
material/study stress instances and 10 hardening instances, with whole-rate
holdouts; hardening excludes the pre-matched DIF predictor. Units and cross-study
material identity still need verification. Steady PDE has an independent implicit adapter; Green inverse problems still require A5.

## Training-data overlap

EqGPT's `_KNOWN_EQUATION_NAMES` maps only `kdv`; unknown keys train on the full
handbook, so non-KdV rows have unknown overlap. Even row exclusion does not exclude
equivalent or close-family equations. SymbolicGPT's per-case synthetic corpus needs
overlap logging. For pretrained models record source revision, checkpoint SHA256,
upstream corpus description and whether benchmark overlap is known, excluded or
unknown. The Core ODE systems are textbook equations likely known to LLM backbones.

## SymbolicGPT protocols

The default `synthetic_per_fit` protocol trains a new small model for each target.
The opt-in `reuse_pretraining_within_case=true` (`synthetic_case_instance_reuse`)
trains once per dataset instance in a case, then copies the model and restores its
random state per RHS; sampling and constant fitting stay per RHS. Results record
`method_provenance.protocol` and `cache_hit` (`null` when a fit was interrupted).
Under `target-v2`, successful target calls persist that instance cache atomically so the
next isolated target worker can restore it; each attempt has its own working directory.
A matched historical seed-0 damped-oscillator diagnostic (300 s per case) completed 2/2 targets
with reuse and 1/2 per fit; neither recovered the structure.

## E2E (official pretrained adapter)

The adapter wraps the official `SymbolicTransformerRegressor` from
`facebookresearch/symbolicregression` (commit
`c4144c99d078dd611795338e24fb3da49a32b9d8`), checkpoint `model1.pt` (SHA256
`169569a0648ae1a4f1ac2bbb376fa5ebb70ebce4bfcf23d6666323596990ff91`), and its pinned
sympytorch fork (`cb4cd0f516c2f1495ebbddd78fa93279e5f24c56`, installed with
`--no-deps`). Configure absolute `source_dir`, `checkpoint_path`, a verified
`checkpoint_sha256` and `trust_checkpoint=true` in a local override; the tracked
config has no asset paths, so E2E is unavailable until supplied. The checkpoint is
an executable Torch pickle: a checksum detects substitution but does not make an
untrusted pickle safe. Nothing is downloaded or unpickled at import or dry-run.

The wrapper uses `retrieve_tree(with_infos=True)['relabed_predicted_tree']` and the
official simplifier, avoiding an upstream `predict` keyword mismatch without
changing the tree. Metadata separates checkpoint loading, model-forward inference
and remaining fit time. Smoke refines 10 trees, full refines 100 as in the official
example. On Windows a scoped unpickler maps the pickled `PosixPath`.

## LLM-SR protocol: Llama-3.1-8B-Instruct, 1k calls

Follows the LLM-SR searcher of
[LLM-SRBench](https://github.com/deep-symbolic-mathematics/llm-srbench)
(`configs/llmsr_llama31_8b.yaml`, `methods/llmsr/`). Adapter:
`kd/model/kd_llmsr.py`; settings: `configs/benchmark/baselines/llmsr.json`
(smoke 8 calls, full 1000); strict 1k control batch:
`configs/benchmark/llmsr_llama31_8b_1k.json`.

| Setting | Value | Option |
| --- | --- | --- |
| Backbone | `meta-llama/Llama-3.1-8B-Instruct` via vLLM OpenAI chat API, `127.0.0.1:10005` | `api_model`, `endpoint` |
| Sampling | temperature 0.8; `top_p` server default; 1024 max new tokens | `temperature`, `top_p`, `max_tokens` |
| Programs per prompt | 4 chat requests (`n=1` each) | `samples_per_prompt` |
| Budget | 1000 completed calls per target (250 prompts) | `max_llm_calls` |
| Evaluators | 4 worker processes, 30 s per program (parse + BFGS + scoring) | `num_evaluators`, `evaluation_timeout` |
| Islands / examples | 10 islands, 2 in-context programs ordered worst to best | `num_islands`, `functions_per_prompt` |
| Parameters | `MAX_NPARAMS = 10`; SciPy BFGS from `[1.0]*10`, SciPy defaults | `max_parameters` |
| Score | negative training MSE; NaN/inf rejects the program | fixed |
| Buffer | score clusters, cluster temperature 0.1 over a 30000-program period, shorter programs preferred, weaker half of islands reset every 4 h | `cluster_sampling_*`, `reset_period` |
| Stop | budget exhausted or zero training MSE | fixed |
| Prompt data | none; target and input names only | `prompt_samples=0` |

A call counts when a request completes. Failed requests are retried at most
`max_request_retries` times (4xx other than 429 are not retried) and do not use
budget; exhausted retries fail the fit. `method_provenance` records calls, prompts,
served model, tokens, rejection reasons, evaluation timeouts, island resets and the
best program's call index; `best_program` stores its source.

Differences from LLM-SRBench: generated bodies are not `exec`ed but restricted to
assignments, augmented assignments, tuple unpacking (including `params[a:b]`),
docstrings, `import numpy as np` and one `return`, then inlined into a validated
NumPy expression (loops, conditionals and unlisted calls are rejected and counted);
variables are described by name only; each request sends a derived `seed`, but vLLM
batching is not bit-reproducible; retries are finite.

Serving: weights are never downloaded by this repository. Keep vLLM in its own
environment (it pins its own PyTorch).

```bash
python scripts/check_llmsr_backbone.py --model-path "$LLMSR_BACKBONE_PATH" --sha256
CUDA_VISIBLE_DEVICES=0 LLMSR_BACKBONE_PATH=... LLMSR_VLLM_ENV=... \
  bash scripts/serve_llmsr_backbone.sh          # HF_HUB_OFFLINE=1, port 10005
python scripts/check_llmsr_backbone.py --server --calls 4   # real generation + BFGS
python -u run_benchmark.py --models llmsr --datasets ode_core_duffing --tasks ode \
  --output-dir results/llmsr_smoke --timeout 3600
```

If the server sets `VLLM_API_KEY`, export the same value for the benchmark; the
adapter sends `LLMSR_API_KEY`, then `VLLM_API_KEY`. Only fake-endpoint tests have
run so far.

## Weak-form (WSINDy-PDE) scope

`weakform` uses `kd.model.kd_wsindy.WSINDyPDEModel`, written against the
[official WSINDy-PDE](https://github.com/MathBioCU/WSINDy_PDE) algorithm (audited
commit `d9296be4c17c5e0b4df14472f4cd8276a8ae4eed`). For a scalar field on a uniform
1D grid, candidate columns and `u_t` are formed by valid convolution with compact
polynomial test functions, so the fit never differentiates the data. Columns are
scaled, an MSTLS threshold path is scored by projection change plus support size,
and the selected model is refit and exported in jet-variable syntax (`D_x(u^2)`
becomes `2*u*u_x`; the conservative form stays in `weak_expression`). Not exposed:
multidimensional/multifield libraries, trigonometric and mixed derivatives,
automatic Fourier-corner support selection, RGLS. `IntegralWeakPDEModel` is only an
import alias.

## ODE dataset cards

Systems are dimensionless unless a unit mapping is supplied. Ground truth is used
only by the evaluator. Fixed coefficients are unknown constants, not input features
(`expose_parameters=true` is a separate setting, not parameter generalization).
Splits hold out whole trajectories.

| Dataset | Role | States | Notes |
| --- | --- | --- | --- |
| `ode_core_damped_oscillator` | Core | x, v | Linear damping, zeta=0.1 |
| `ode_core_pendulum` | Core | theta, v | Damped, sin(theta); polynomial libraries cannot recover it |
| `ode_core_duffing` | Core | x, v | Damped, cubic restoring force |
| `ode_core_van_der_pol` | Core | x, v | mu=2 |
| `ode_core_sir` | Core | susceptible, infected, recovered | Conserves total population |
| `ode_core_fitzhugh_nagumo` | Core | v, w | Cubic, slow recovery |
| `ode_core_rational` | Core | s | Michaelis–Menten, vmax=1.5, km=0.7; needs rational expressivity |
| `ode_core_robertson` | Core | x, y, z | Stiff; Radau and log-spaced observation times |
| `ode_core_oscillator` | Supplementary (dev) | x, v | Harmonic, omega=1.3 |
| `ode_core_population` | Supplementary (dev) | prey, predator | Lotka–Volterra |
| `ode_core_chaotic` | Supplementary (dev) | x, y, z | Lorenz; pointwise long-time errors need chaos-aware reading |
| `ode_core_brusselator` | Supplementary (dev) | x, y | Cubic autocatalysis |

`kd.dataset._ode_core.generate_ode_core` records equations, parameters, initial
conditions, seed, solver and tolerances, and time sampling in `generation_metadata`
(DOP853 uniform times; Robertson Radau). `observation_noise` adds Gaussian noise
relative to each trajectory/state standard deviation before derivative estimation;
`time_stride` subsamples. Five runner seeds vary split and algorithm randomness on a
fixed generator seed; `initial_condition_scale` shifts the whole cohort and is not an
OOD split.

Robertson audit (2026-09-25, five generator seeds): `np.gradient` on the nonuniform
grid is correct (clean relative derivative error below 1e-3), but 1% observation
noise divided by spacings as small as 8e-8 gives relative errors around 5e4; even
scoring only `t >= 1` leaves about 20%. Clean Robertson stays in the main track; a
noisy/sparse track needs a predeclared observation schedule and derivative
estimator, validated per component and time region. Analytic RHS values are for
auditing only.

`configs/benchmark/core_ode_track.json` selects the eight core systems, eleven ODE
methods and five seeds (440 cases). `core_v2.json` (70 cases, SINDy + WSINDy) and
`core_v2_ode_noisy_sparse.json` (40 cases, 1% noise, stride 2) are integration
suites. None has run as a production benchmark.

A1/A2 来源、资格、同源观测与协议迁移见 [验收记录](validation.md)；
旧 domain-v1 与新 domain-v3 结果不混用，来源不足尚未正式就绪。

## A6 prior exposure

Records label [training overlap](../kd/dataset/_overlap.py). EqGPT handbook structure exposure is confirmed for listed families; field/coefficients remain unaudited. E2E corpus is unknown; SymbolicGPT configuration-only synthetic corpus has possible structure overlap. These labels do not establish clean checkpoint lineage.

## C3 fit deadlines (target-v2)

The runner enables a shared cooperative deadline at fit entry. LLM-SR confines blocking HTTP
threads to a disposable request process, caps retries and evaluators by the remaining fit
time, and retains previously evaluated programs. Submitted calls and completed responses
are reported separately as `llm_calls` and `llm_calls_completed`.

gplearn advances one generation per warm-start call, replaying the original RNG state.
Without a time stop its populations and final selection match a normal fit; on a time stop
it returns the best completed generation by the configured training fitness.
PySR receives `timeout_in_seconds` plus an absolute Julia `time_ns()` stop predicate,
anchored before native search initialization and combined with its existing early stop.
Operon receives integer `max_time` rounded up from
remaining seconds. Existing shorter native limits are retained and constructor parameters
are restored after fitting. Their APIs lack reliable termination codes, so `fit_deadline`
identifies an observed deadline rather than claiming an unexposed native stop signal.

DSO and DSCV retain completed candidate rewards before checking the deadline, then stop
before another training step. PhySO uses its public logger hook after a completed batch
has entered its Pareto archive. Atomic native calls, batch operations and initialization
can still exceed the soft budget; the independent T + grace watchdog remains authoritative.
A deadline without a valid candidate is an error, and a forced termination is a hard timeout.

The strict 1000-call LLM-SR control disables soft deadlines, retaining its existing 43200 s
hard safety limit. Real Linux/CUDA/vLLM deployment, including server-side cancellation after
HTTP disconnect, still needs smoke validation. These engineering checks do not freeze
scientific budgets or run the full benchmark.
