# Server setup

Use a separate experiment worktree and persistent environment. Keep source/config
changes locally, back up the remote files before synchronization and compare
normalized SHA256 hashes. Preserve previous worktrees and raw result directories.

## Python environments

- The 19 baselines registered in `run_benchmark.py` can share the Python 3.10
  stack below. Keep their task/data compatibility rules and external runtime
  requirements separate from Python package compatibility.
- `requirements.txt`: portable core stack and native/example development.
- `requirements-operon-py310.txt`: pinned Python 3.10 numeric stack and PyOperon.
- `requirements-server-unified-py310.txt`: extra packages for the validated unified
  server stack; install alongside the Operon list, not as a standalone core recipe.
- `requirements-server-py310.txt`: small optional PyOperon add-on for an existing
  environment. It is not the unified-environment specification.

The historical unified server used Python 3.10, PyTorch 2.5.1 with CUDA 12.1,
PyOperon 0.6.1 and the pins in those files. Its E2E sympytorch fork and Julia
installation are separate assets.

~~~bash
conda create -n kd-server python=3.10 pip
conda activate kd-server
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-operon-py310.txt
python -m pip install -r requirements-server-unified-py310.txt
python -m pip install -e .
python -m pip check
~~~

Choose the PyTorch build for the actual device. Do not copy a Windows Conda export
or local build-cache paths into a Linux installation.

The two server requirement files are a separate installation recipe. Do not
combine them with `requirements.txt` in one resolver invocation: the recipes
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

Copy `hpc/server.env.example` to ignored `.local/server.env` and set the actual
storage root, environment prefix, Conda activation path and optional asset paths.
The activation script resolves the project from its own location unless
`KD_PROJECT` is provided. An already selected environment can be supplied via
`KD_ENV`. No SSH host, username, key path or private storage path is published.

~~~bash
mkdir -p .local
cp hpc/server.env.example .local/server.env
# Edit .local/server.env for the host before activation.
source hpc/activate_kd_server.sh
python -m pip check
python run_benchmark.py --config configs/benchmark/core_ode_track.json --dry-run
~~~

E2E paths in environment variables are conveniences for local setup; supply
explicit source/checkpoint paths, trust and verified hash in the benchmark's
local override config. See [E2E card](methods.md#e2e-official-pretrained-adapter). Keep LLM-SR endpoint
credentials local and validate real generation before a full batch.

## Historical launch recipes

The dated `hpc/run_*_20260925.sh` recipes retain their original selections and
budgets. CPU/GPU missing-pair recipes require the manifests and compatibility
reports from the original run; those artifacts are local and must exist before
launching. Older dated scripts in `scripts/` likewise require their original
manifests. They do not start when the repository is cloned or imported.

For historical persistence scripts, KD_PROJECT identifies the source checkout.
Export KD_PERSIST_DIR to select a separate destination; its default is the
September 17 run directory under KD_STORAGE_ROOT.

Recheck processes, logs and exit files before resuming an old output directory.
An exit code of 1 can mean a completed batch contains fit errors/timeouts.
Validate case/target/status denominators and download archives with SHA256 checks.
