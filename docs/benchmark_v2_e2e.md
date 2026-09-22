# Optional official-pretrained E2E adapter

`e2e` now has an optional runner adapter. No official checkpoint/source installation
is present here; only offline API contract tests have run. Missing assets produce
an explicit skipped/unavailable result, never replacement training.

Sources inspected on 2026-09-21:

- https://github.com/facebookresearch/symbolicregression/blob/main/Example.ipynb
- https://github.com/facebookresearch/symbolicregression/blob/main/symbolicregression/model/sklearn_wrapper.py
- https://github.com/facebookresearch/symbolicregression/blob/main/symbolicregression/envs/generators.py

Use an isolated environment compatible with the upstream repository. Configure
absolute `source_dir`, `checkpoint_path`, a verified `checkpoint_sha256`, and
`trust_checkpoint=true` only after reviewing the source and checkpoint provenance.
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

The default config is deliberately unavailable until assets are supplied. The
smoke profile refines 10 trees; full refines 100 as in the official example. These
are bounded configurations, not a verified reproduction of all paper settings.
