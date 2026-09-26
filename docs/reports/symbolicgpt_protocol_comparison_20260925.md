# SymbolicGPT within-case reuse versus per-fit pretraining (2026-09-25)

This is a **single-case, seed-0 diagnostic**, not a Core ODE main-track score.
The standard `run_benchmark.py` CLI ran both protocols on the server's RTX
4090, Python 3.10.21 and PyTorch 2.5.1+cu121. Each used the full SymbolicGPT
budget (3,000 synthetic equations, 50 epochs, 100 sampled candidates), zero
DataLoader workers, the same generated damped-oscillator trajectories and a
300-second timeout shared by both RHS targets. The case manifests are identical
except for `model_params.reuse_pretraining_within_case` and the derived case
name. Both retain the `configured_fallback` operator policy; Core ODE has no
frozen declared function set.

| Protocol | `d(x)/dt` | `d(v)/dt` | Completed / declared targets | Structural recovery |
| --- | --- | --- | ---: | ---: |
| Synthetic pretraining per fit | `ok`, 152.43 s, test NRMSE 0.000162 | `timeout`, 147.86 s of remaining case time | 1 / 2 | 0 / 1 evaluable; one timeout |
| Synthetic pretraining reused within case instance | `ok`, 143.98 s, test NRMSE 0.000162 | `ok`, 3.59 s, test NRMSE 1.000427 | 2 / 2 | 0 / 2 evaluable |

The first RHS has the **same expression**, train/test trajectory split,
candidate-validation counts (100/100 accepted), and test metrics in both runs.
Its small test error does not imply structural recovery: the expression has an
extra quadratic term and a constant. In the reuse run, the second RHS also had
100/100 accepted candidates and finite prediction coverage 1.0, but the fitted
expression did not recover the reference structure and its test NRMSE was about
1.0. The timeout row remains in the per-fit denominator and records
`method_provenance.protocol=synthetic_per_fit` with `cache_hit=null` because
the interrupted fit's internal state is unknown.

The case-level time budget makes the completion comparison meaningful. It does
not establish a broad speedup or equation-quality advantage across systems and
seeds. The reuse protocol trains a synthetic model once per dataset instance;
it is **not** the original large shared SymbolicGPT checkpoint. Keep its results
separate from `synthetic_per_fit` in any table or aggregate. Do not launch the
440-case Core ODE main track on these diagnostics: the candidate function-set
protocol is still unfrozen, and this case has no structural recovery under
either observed protocol.

## Reproduction and artifacts

- Reuse config: `configs/benchmark/tuning/symbolicgpt_case_reuse.json`.
  Immutable local result copy: `results/server_symbolicgpt_cli_reuse_20260925_161500/`.
- Per-fit config: `configs/benchmark/tuning/symbolicgpt_per_fit.json`.
  Result copy: `results/server_symbolicgpt_per_fit_20260925_matched/per_fit/`.
  Archive SHA256: `883e3feb939593eeba6a75b8cdf969f557bc182390922b5cde8b4462619111d7`.
- Both commands selected only `ode_core_damped_oscillator` from their two-dataset
  diagnostic configs, with `--device symbolicgpt=cuda:0` and a separate
  `--output-dir`. The full resolved manifests and per-target JSON rows are in
  their respective directories.
- The server baseline file had an extra default full-profile reuse flag before
  this run. Its exact prior copy is retained under
  `results/server_symbolicgpt_per_fit_20260925_matched/source_before/`; the
  server baseline was restored to the matching local default, while both
  diagnostic configs set their reuse switch explicitly.
