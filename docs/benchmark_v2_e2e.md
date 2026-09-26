# Optional official-pretrained E2E adapter

`e2e` has an optional runner adapter. The official source checkout and pretrained
checkpoint are cached outside the worktree on this machine. A one-case Core ODE
Track CLI smoke on 2026-09-24 completed both derivative targets with status `ok`
under `results/e2e_dependency_smoke_20260924/`; this verifies execution, not
recovery quality or paper-protocol fidelity. Missing assets produce an explicit
skipped/unavailable result, never replacement training.
With the same local asset override, the no-fit compatibility check accepted
all eight Core ODE systems under `results/core_ode_track_e2e_compat_20260924/`.

Sources inspected on 2026-09-21:

- https://github.com/facebookresearch/symbolicregression/blob/main/Example.ipynb
- https://github.com/facebookresearch/symbolicregression/blob/main/symbolicregression/model/sklearn_wrapper.py
- https://github.com/facebookresearch/symbolicregression/blob/main/symbolicregression/envs/generators.py

The local checkout is `external/e2e-source` at commit
`c4144c99d078dd611795338e24fb3da49a32b9d8`; the official checkpoint is
`model1.pt` (SHA256
`169569a0648ae1a4f1ac2bbb376fa5ebb70ebce4bfcf23d6666323596990ff91`).
The requested sympytorch fork is cached at `external/sympytorch-source`
at commit `cb4cd0f516c2f1495ebbddd78fa93279e5f24c56` and installed in
`kd-env` with
`python -m pip install --no-deps --no-build-isolation external/sympytorch-source`.
This keeps the environment's already installed Torch and SymPy versions. The
runner smoke uses the local, gitignored config
`results/e2e_dependency_smoke_20260924_config.json`:

```powershell
conda run -n kd-env python run_benchmark.py --config results/e2e_dependency_smoke_20260924_config.json
```

Configure absolute `source_dir`, `checkpoint_path`, a verified
`checkpoint_sha256`, and `trust_checkpoint=true` only after reviewing the source
and checkpoint provenance.
The official notebook uses an executable full-model Torch pickle; a checksum
detects substitution but does not make an untrusted pickle safe. The adapter never
downloads or unpickles assets during import or dry-run.

The wrapper instantiates the official `SymbolicTransformerRegressor`, invokes fit,
and uses `retrieve_tree(with_infos=True)['relabed_predicted_tree']` to preserve
upstream feature selection. Predictions use the official simplifier on this tree
and full input X. This narrowly avoids an upstream `predict` keyword mismatch
(`tree_idx` versus `dataset_idx`) without changing the discovered tree. Supported
tree nodes are translated structurally for the common equation evaluator; unsupported
operators fail explicitly and need additional fidelity checks.

No per-case pretraining occurs. Metadata separates checkpoint loading, synchronized
model-forward inference, total fit, and remaining fit time (preprocessing **plus**
refinement, not a claimed pure refinement cost). Original pretraining time and
benchmark training-data overlap remain unknown. Device, seed, checkpoint hash,
source path and source wrapper hash are recorded. Pin the entire source revision
and environment externally before production use.

The tracked default config has no machine-specific asset paths, so it remains
unavailable until an explicit override is supplied. On Windows, the adapter maps
the Linux `PosixPath` in this official full-model pickle through a scoped
unpickler; it does not modify global `pathlib` behavior. The
smoke profile refines 10 trees; full refines 100 as in the official example. These
are bounded configurations, not a verified reproduction of all paper settings.
The local dependency smoke further limits refinement to one tree.
