# Server setup

Use a separate experiment worktree and persistent environment. Keep source/config
changes locally, back up the remote files before synchronization and compare
normalized SHA256 hashes. Preserve previous worktrees and raw result directories.

## Python environments

- `requirements.txt`: portable core stack and native/example development.
- `requirements-operon-py310.txt`: pinned Python 3.10 numeric stack and PyOperon.
- `requirements-server-unified-py310.txt`: extra packages for the validated unified
  server stack; install alongside the Operon list, not as a standalone core recipe.
- `requirements-server-py310.txt`: small optional PyOperon add-on for an existing
  environment. It is not the unified-environment specification.

The historical unified server used Python 3.10, PyTorch 2.5.1 with CUDA 12.1,
PyOperon 0.6.1 and the pins in those files. Its E2E sympytorch fork and Julia
installation are separate assets. Historical validation counts and failures are in
[the dated server report](reports/server_validation_20260925.md).

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
local override config. See [E2E setup](benchmark_v2_e2e.md). Keep LLM-SR endpoint
credentials local and validate real generation before a full batch.

## Historical launch recipes

The dated `hpc/run_*_20260925.sh` recipes retain their original selections and
budgets. CPU/GPU missing-pair recipes require the manifests and compatibility
reports listed in [the run report](reports/server_missing_pair_runs_20260925.md);
those artifacts are local and must exist before launching. Older dated scripts
in `scripts/` likewise require their original manifests. They do not start
when the repository is cloned or imported.

For historical persistence scripts, KD_PROJECT identifies the source checkout.
Export KD_PERSIST_DIR to select a separate destination; its default is the
September 17 run directory under KD_STORAGE_ROOT.

Recheck processes, logs and exit files before resuming an old output directory.
An exit code of 1 can mean a completed batch contains fit errors/timeouts.
Validate case/target/status denominators and download archives with SHA256 checks.
