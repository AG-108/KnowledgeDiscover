# SISSO++ baseline

Runner key `sisso` supports SR and individual ODE right-hand sides through
`kd.model.SISSOModel`. It uses the official SISSO++ v1.2.6 Python bindings in
the same Python 3.10 environment as the other baselines. The bindings require a
separate CMake installation; they are optional for core imports and CI.
See the [official installation guide](https://sissopp_developers.gitlab.io/sissopp/quick_start/Installation.html)
and [scikit-learn interface](https://sissopp_developers.gitlab.io/sissopp/tutorial/2b_sklearn.html).

The current server installation uses the CPU backend. All inputs and responses
are dimensionless, nonlinear parameter operators are disabled, and selection
uses training data only. The adapter retains the highest requested descriptor
dimension and exports its actual coefficients, rather than the `a0/c0`
placeholders in the upstream model string. This is a regression method; it
does not provide a PDE-specific feature library or physical-unit priors.

Native descriptors are evaluated through `ModelNode.eval_many`; fitted linear
coefficients and the intercept are applied explicitly. This also handles the
v1.2.6 zero-intercept case, where coefficients omit the intercept but the
upstream model evaluator adds the last coefficient as an offset.

`configs/benchmark/baselines/sisso.json` defines bounded smoke settings
(`n_dim=1`, `max_rung=1`, `n_sis_select=1`) and an engineering full profile
(`n_dim=2`, `max_rung=2`, `n_sis_select=10`, `n_residual=1`). These budgets are
not a paper reproduction. The fixed operator set is shared across datasets;
there is no reference-informed grammar or held-out model selection.

The baseline is included in wildcard matrices, the Core ODE configuration and
the main candidate configuration. Each target stores native files under its
own `sisso/` work directory. SISSO uses the CPU slot and inherits its thread
environment. The existing supervisor enforces fit, setup and scoring hard
limits. SISSO has no native soft deadline or recovery of partial candidates;
an interrupted search is recorded as `timeout/hard_timeout`.

Run the native integration checks and the complete code regression in the
configured environment:

```bash
python -m pytest -q tests/test_kd_sisso.py tests/test_sisso_integration.py --require-sisso
python -m pytest -q --require-sisso --require-external-data
python run_benchmark.py --models sisso --datasets Koza-2 ode_core_damped_oscillator --timeout 30 --output-dir results/sisso_smoke
```

Ordinary pytest discovery includes the native checks. Without installed
bindings they skip with an explicit reason; `--require-sisso` makes absence a
failure. The lightweight CI runs adapter contracts without native libraries.
Integration checks cover linear and square fits, held-out predictions, both
intercept modes, the two-descriptor full configuration, real SR/ODE workers,
and the supervisor's hard timeout. Runtime identity includes hashes of the
native extension and Python wrappers, even for CMake installs without package
metadata.

Validation on 2026-10-05: all six native checks passed within the Linux full
regression (607 passed, two Windows-only skips). The two platform-specific
checks passed on Windows. Independent CLI smoke completed all three SR/ODE
targets, and the adapter's 24-point linear fit had RMSE `1.7554167342883506e-16`.
See [validation records](validation.md) for evidence and scope.
