# Documentation

| Document | Scope |
| --- | --- |
| [Plan](benchmark_plan.md) | Agreed decisions, current capabilities and remaining work |
| [Benchmark CLI](benchmark.md) | Configurations, devices, commands and outputs |
| [Metrics](benchmark_metrics.md) | Recovery, coverage, denominators, rollout and resources |
| [Methods](methods.md) | Method variants, search spaces, applicability and ODE cards |
| [Data](data.md) | Data protocols, external assets and inventory export |
| [Validation](validation.md) | Consolidated A1–A6 acceptance and reproduction commands |
| [Deferred issues](deferred_source_issues.md) | Unresolved source, geometry and licensing conditions |
| [Asset manifest](asset_manifest.json) | Relative paths, hashes and availability for 443 assets |
| [API migration](dataset_api_migration.md) | Public dataset exports |
| [Server setup](server_setup.md) | Environments, external runtimes and private configuration |
| [Auxiliary evaluators](auxiliary_evaluators.md) | PIC and physical credibility prototypes |

Sphinx API sources are in `source/`; build with `make html`.
Runtime code is in `kd/`, configurations in `configs/`, examples in `examples/`,
regression tests in `tests/`, reusable commands in `scripts/` and server recipes in `hpc/`.
External data, weights, results and private settings are ignored by Git.
