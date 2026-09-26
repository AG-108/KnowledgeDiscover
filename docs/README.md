# Documentation

## Start here

- [Project overview and quick start](../README.md)
- [Benchmark CLI and configuration](benchmark.md)
- [Data availability and external assets](data.md)
- [Server setup and portable launch scripts](server_setup.md)
- [Implementation status and remaining work](benchmark_roadmap.md)

## Protocols and implementation references

| Document | Maintained scope |
| --- | --- |
| [Metrics](benchmark_metrics.md) | Recovery, coverage, denominators, resources and information criteria |
| [Method and dataset cards](benchmark_v2_method_cards.md) | Implementation variants, WSINDy scope, ODE generation, applicability and overlap |
| [Candidate tuning](benchmark_candidate_tuning.md) | Configured versus declared grammars and matched pilots |
| [E2E adapter](benchmark_v2_e2e.md) | Source/checkpoint setup, trust, hashes and adapter limits |
| [PIC](benchmark_v2_pic.md) | Opt-in evaluator, numerical protocol and failed calibration gate |
| [Physical credibility](benchmark_v2_physical_credibility.md) | Physical contract, offline implementation and annotation workflow |
| [Compatibility report](compatibility_report.md) | Dated no-fit matrix; not fitting or recovery evidence |
| [Code style](code_style.md) | Maintained-code formatting and upstream exclusions |

The [paper outline](paper/knowledge_discover_benchmark_outline.tex) is a working
draft. Sphinx API sources remain in `source/`.

## Repository organization and publication

Source lives in `kd/`, configurations in `configs/`, examples in `examples/`,
tests in `tests/`, reusable commands in `scripts/`, and server recipes in `hpc/`.
The root `requirements.txt` is the portable core recipe; separate server requirement
files capture the optional unified stack. Keep current instructions in canonical
guides and store dated run evidence with local experiment results.

External datasets, checkpoints, raw results, generated logs, archives, caches,
credentials and local overrides are excluded by `.gitignore`. Existing small
scientific fixtures remain tracked. Store private settings under `.local/` and
copy `hpc/server.env.example` to `.local/server.env` when needed.
The cleanup backup is local under `.local/repository_cleanup_20260926/`.
