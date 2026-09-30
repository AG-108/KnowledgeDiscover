# LLM-SR upstream provenance

This integration is based on the official
[`deep-symbolic-mathematics/LLM-SR`](https://github.com/deep-symbolic-mathematics/LLM-SR)
implementation at commit `41c212312df6c16d936c9cb395356a62774c47e3`
(downloaded 2026-09-15). Its search protocol follows the LLM-SR searcher in
[`deep-symbolic-mathematics/llm-srbench`](https://github.com/deep-symbolic-mathematics/llm-srbench)
(`methods/llmsr/` and `configs/llmsr_llama31_8b.yaml`, `main` branch, read
2026-09-29): prompt/specification format, BFGS objective, experience buffer, call
counting and evaluator settings.

The upstream pipeline is tied to its bundled tasks. `KD_LLMSR` adapts its
program-skeleton, numerical constant-optimization, and island-search design to
the benchmark's generic `fit(X, y)` contract. It also replaces the upstream
hard-coded service URL, unrestricted `exec` of generated code and unbounded
request retry loops with explicit runtime configuration, a restricted expression
evaluator and finite retries. The protocol and remaining differences are
documented in the LLM-SR section of `docs/methods.md`. No upstream datasets,
generated logs, model weights, or Python caches are vendored here.

The upstream repository declares the MIT license; a copy is stored beside this
file in `LICENSE`.
