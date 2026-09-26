# Server validation on 2026-09-25

Historical observations from a separately synchronized worktree. These counts
apply to the recorded revisions, not the current checkout. Private host, key
and storage settings are maintained locally. See [server setup](../server_setup.md)
for the portable environment recipe.

Validation on the server: Core ODE Track dry-run selected 440 cases; the
unified environment completed 257 tests with one Windows-only test skipped,
and `pip check` reported no broken requirements. The one-seed, no-fit
compatibility preflight passed all 88 method/system pairs in the same Python
3.10 interpreter, using E2E's local asset override. E2E on CUDA, PyOperon, and
PySR each completed one ODE case with two `ok` target rows. A seven-method
smoke run on the damped oscillator returned 11 `ok`, one `error`, and two
`timeout` target rows with a 120-second per-case (model/dataset/seed) limit. DSCV, gplearn,
PhySO, SINDy, and PySINDy completed both targets. DSO completed the first
target and timed out on the second. SymbolicGPT produced a nonfinite prediction
on the first and timed out on the second. These are execution checks, not a
full recovery benchmark. The SR no-fit preflight passed 14/14 method/dataset
pairs. The PDE no-fit preflight returned 20 compatible pairs and the same two
expected PDE-Net/1D incompatibilities; PDE-Net on the 2D `wdwake` dataset was
compatible. PDE-Net's `Iterator` import was updated for Python 3.10 before
these final PDE checks. The `llmsr` preflight checks only its adapter import
and data shape; LLM-SR still lacks a configured completion endpoint and model,
so a real generation call is pending.


## Server-first follow-up and local synchronization

The September 25 follow-up compared 305 relevant files, backed up the seven
changed source/config/document files, applied them with pre/post SHA256 checks,
and ran the complete suite: **288 passed / 1 Windows-only skipped** in 150.54 s.
The machine had one available RTX 4090 with 24 GiB memory.

All follow-up artifacts are in
`results/server_symbolicgpt_validation_20260925_134231/` on both machines:

- `sync_manifest.json` and `sync_backup/`: verified source update and rollback copies.
- `server_tests.xml` / `server_tests.log`: complete server regression results.
- `formal_budget/`: original SymbolicGPT full configuration, one seed-0 damped
  oscillator case; one RHS `ok` at 194.83 s, the other `timeout` after the shared
  300-second case budget expired. The successful RHS had full finite prediction
  coverage but did not recover the reference structure.
- `budget_pilot_core8/`: separate 250-corpus/8-epoch/24-candidate trial, with
  conditioning capped at 256 points and DataLoader workers set to zero; eight
  cases, 17 fit errors, no timeouts, and no accepted candidates out of 408.
  This rejected pilot is reproducible with
  `configs/benchmark/tuning/symbolicgpt_core_ode_budget_pilot.json`; it is not
  the original full configuration or a validated replacement.
- `core_ode_numerics.json` / `audit_core_ode.py`: clean eight-system parsing and
  derivative audit, plus five-seed Robertson noise/sparsity diagnostics. All 17
  clean training targets were finite and references parseable. No declared
  function-set metadata was present. Robertson's log-grid finite differences
  amplified 1% state-standard-deviation observation noise to a maximum relative
  RMS derivative error of about 7.63e4 (3.81e5 at 5% noise).
- `server_validation_summary.json`: consolidated counts, rejection causes and
  numerical audit summaries, with explicit failure/evaluation denominators.

The two fitting experiments use `configured_fallback` grammars and the
per-fit-from-scratch variant. They are budget diagnostics, not the 440-case main
benchmark. The 74 experiment artifacts were downloaded with archive SHA256
verification; source changes remain available locally.
