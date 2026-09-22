# LLM-SR upstream provenance

This integration is based on the official
[`deep-symbolic-mathematics/LLM-SR`](https://github.com/deep-symbolic-mathematics/LLM-SR)
implementation at commit `41c212312df6c16d936c9cb395356a62774c47e3`
(downloaded 2026-09-15).

The upstream pipeline is tied to its four bundled tasks. `KD_LLMSR` adapts its
program-skeleton, numerical constant-optimization, and island-search design to
the benchmark's generic `fit(X, y)` contract. It also replaces the upstream
hard-coded service URL and unbounded request retry loops with explicit runtime
configuration and finite timeouts. No upstream datasets, generated logs, model
weights, or Python caches are vendored here.

The upstream repository declares the MIT license; a copy is stored beside this
file in `LICENSE`.
