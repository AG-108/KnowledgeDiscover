# Server setup

Use a separate experiment worktree and retain the validated runtime environment
in the saved container image. Keep source/config changes locally, back up the remote files before synchronization and compare
normalized SHA256 hashes. Preserve previous worktrees and raw result directories.

## Experiment storage

As agreed on 2026-10-09, server experiments write declarations, target results,
checkpoints, logs and reports directly to a persistent mounted filesystem.
Before starting or resuming a run, check the mount and the resolved output path;
an unmounted directory or the container's overlay filesystem is not a valid
destination. Keep each experiment's existing identity and first terminal records
when changing its storage location.

Configure `KD_STORAGE_ROOT` as the mounted storage root in private server settings.
Use explicit output paths beneath it, or verified symlinks for existing relative
`results/` paths and `/opt/kd/experiments`. An environment variable alone does not
redirect a benchmark configuration's `output_dir`. Keep actual mount paths private.
On replacement containers, verify the mount and links before dispatching any fits.

Model weights, dependency bundles and runtime environments may remain in the
container image, which the user will save. They do not need to be copied into the
result mount. Local checkpoint downloads are optional additional backups; current
experiment progress must remain available without a local backup process.

## Python environments

- The 19 baselines registered in `run_benchmark.py` can share the Python 3.10
  stack below. Keep their task/data compatibility rules and external runtime
  requirements separate from Python package compatibility.
- `requirements/core.txt`: portable core stack and native/example development.
- `requirements/server-operon-py310.txt`: pinned Python 3.10 numeric stack and PyOperon.
- `requirements/server-unified-py310.txt`: extra packages for the validated unified
  server stack; install alongside the Operon list, not as a standalone core recipe.
- `requirements/docs.txt`: Sphinx builder for `docs/`.

The historical unified server used Python 3.10, PyTorch 2.5.1 with CUDA 12.1,
PyOperon 0.6.1 and the pins in those files. Its E2E sympytorch fork and Julia
installation are separate assets.

~~~bash
conda create -n kd-server python=3.10 pip
conda activate kd-server
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements/server-operon-py310.txt
python -m pip install -r requirements/server-unified-py310.txt
python -m pip install -e .
python -m pip check
~~~

Choose the PyTorch build for the actual device. Do not copy a Windows Conda export
or local build-cache paths into a Linux installation.

The two server requirement files are a separate installation recipe. Do not
combine them with `requirements/core.txt` in one resolver invocation: the recipes
pin different setuptools, PyYAML and graphviz versions.

The shared numeric stack uses NumPy 2.0.2 and scikit-learn 1.6.1 for PyOperon
0.6.1 and PySINDy 2.0.0. NumPy 2.0.2 also satisfies Numba 0.60.0's `<2.1`
limit. PyTorch 2.5.1 requires SymPy 1.13.1 on Python 3.10. Keep setuptools
80.9.0 for the DSO `pkg_resources` import. Python 3.10 is the validated
interpreter; this recipe does not establish support for every newer Python.

Current DSCV, SPR and DSO adapters use PyTorch. Historical TensorFlow 1 source
files retained under `kd/model/discover/` are outside these registered execution
paths and are not required by this environment.

## External runtimes and services

| Baseline | Requirement beyond the shared Python packages |
| --- | --- |
| PySR | A compatible Julia executable and its Julia project; the first import can compile packages. The activation script can select a persistent Julia installation. |
| E2E | The reviewed upstream source, its pinned sympytorch fork, and a trusted checkpoint with a verified SHA256; configure them explicitly as described in [E2E card](methods.md#e2e-official-pretrained-adapter). |
| LLM-SR | A vLLM server for the local Llama-3.1-8B-Instruct weights in a separate environment; see [LLM-SR protocol](methods.md#llm-sr-protocol-llama-31-8b-instruct-1k-calls). Other OpenAI-compatible services need a model name and their credentials in environment variables. Installing Python dependencies does not provide a generation service. |

An import or `--check-compatibility` result verifies only the reported checks.
Validate actual fitting/inference separately; a numerical failure, unsupported
dataset shape or exhausted smoke budget does not by itself imply incompatible
Python dependencies.

## Private settings and activation

Copy `server/env/server.env.example` to ignored `.local/server.env` and set the actual
storage root, environment prefix, Conda activation path and optional asset paths.
The activation script resolves the project from its own location unless
`KD_PROJECT` is provided. An already selected environment can be supplied via
`KD_ENV`. No SSH host, username, key path or private storage path is published.

~~~bash
mkdir -p .local
cp server/env/server.env.example .local/server.env
# Edit .local/server.env for the host before activation.
source server/env/activate_kd_server.sh
python -m pip check
python run_benchmark.py --config configs/benchmark/core_ode_track.json --dry-run
~~~

E2E paths in environment variables are conveniences for local setup; supply
explicit source/checkpoint paths, trust and verified hash in the benchmark's
local override config. See [E2E card](methods.md#e2e-official-pretrained-adapter). Keep LLM-SR endpoint
credentials local and validate real generation before a full batch.

## D2 slot execution

`server/env/activate_kd_server.sh` defaults `KD_SLOT_THREADS` to 4 and uses it for unset
OpenMP/BLAS/NumExpr thread variables. The slot executor overrides inherited thread
variables per worker. Prepare a selected manifest, then validate resources without
fitting:

~~~bash
python scripts/run_slots.py --manifest results/<selection>/benchmark_manifest.json \
  --output-dir results/<new_slot_run> --timeout 300 --dry-run
~~~

After explicit authorization to run experiments, remove `--dry-run`. Defaults allocate
CPU cores 0–3 and GPU cores 4–7, four cores and 28 decimal GB RSS each, with one worker per slot.
Use `--cpu-cpus` / `--gpu-cpus` for container CPU numbering and `--gpu-device` for the
physical CUDA index or UUID. Linux requires `taskset` and `psutil`; unavailable affinity
or monitoring fails before launching. Validate binding, CUDA visibility and sampled
resource-limit behavior with a small deployment smoke before a formal batch.
LLM-SR's already bound local vLLM service requires `--gpu-service-pid` and a separate
LLM-SR GPU queue; its CPU/RSS belongs to that slot. The 2026-10-09 deployment passed
four real LLM-SR calls, HTTP disconnect cancellation, CPU binding, aggregate RSS sampling
and service restart checks. Full-run resource evidence is recorded per target.
Alternate LLM-SR and other GPU work only after stopping vLLM and verifying GPU memory
release. See [execution](benchmark.md#d2-资源槽执行c5c6) and
[resource metrics](benchmark_metrics.md#c5c6-resource-protocol-slots-v1).

C7 full settings use four gplearn workers, four native Operon threads, four PySR
processes in multiprocessing mode, and four LLM-SR evaluators. Slot binding records
these effective parameters before case hashing. Their numerical-library thread count
is one per task; other torch workers use four threads. Both `JULIA_NUM_THREADS` and
`PYTHON_JULIACALL_THREADS` are set before Julia initializes. New resource policies
identify `parallelism_protocol=d2-parallel-v1`; older C5 policies and first terminal
rows are preserved. Local native-worker checks do not replace Linux/CUDA/vLLM deployment
verification. See the [method card](methods.md#d2-method-parallelism-c7).

## Server files

Server-only files live under `server/`: `env/` (activation scripts and the private
settings template), `docker/sissopp/` (SISSO++ image recipe) and `launch/`
(`serve_llmsr_backbone.sh`, `tune_pde_20261017.py`). The 2026-09 launch and
persistence scripts were removed on 2026-10-20 together with their result
directories; they remain in commit `718588b`.

Recheck processes, logs and exit files before resuming an old output directory.
An exit code of 1 can mean a completed batch contains fit errors/timeouts.
Validate case/target/status denominators and download archives with SHA256 checks.
