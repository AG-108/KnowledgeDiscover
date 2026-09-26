# Server benchmark continuation (2026-09-25)

These runs use a separately synchronized worktree and unified Python 3.10
environment. Connection settings are local; see [server setup](../server_setup.md).
All runs write new result directories.
Historical outputs have not been overwritten. The local and server copies of
the runner and relevant model/config files were checked by SHA256 after newline
normalization.

## Launch configuration (2026-09-25)

| Batch | Selected cases | Budget and output | State at launch |
| --- | ---: | --- | --- |
| PhySO current full SR/ODE | 287 (274 SR, 13 ODE; seed 0) | 20 epochs, CUDA:0, 1,800 s per case; `results/server_physo_full_20260925/` | Started first; sequential CLI writes `run.log`, case results and final `run.exit` |
| New current-compatible PDE pairs | 23 (13 WSINDy `weakform`, 10 SGA) | CPU, 2 workers, 3,600 s per case; `results/server_cpu_missing_20260925/` | Running with a checkpoint after each case |
| Historical full-matrix GPU gaps | 419 (150 SymbolicGPT, 269 DSO) | CUDA:0, one worker; SymbolicGPT 3,600 s/case then DSO 7,200 s/case | Persistent queue waits for PhySO `run.exit`, then writes `results/server_{symbolicgpt,dso}_missing_20260925/` |

`hpc/run_physo_full_20260925.sh`, `hpc/run_cpu_missing_20260925.sh`, and
`hpc/run_gpu_missing_after_physo_20260925.sh` contain the exact launch commands.
The GPU queue status is in
`results/server_gpu_missing_queue_20260925/queue.log`. The queue can take many
days at the existing full budgets; a launched or compatible pair is not a
completed or successful fit. Each raw row, including errors and timeouts,
remains in its method's result denominator. An exit code of 1 can indicate
recorded case failures even when the batch finished its entire selection.

## How the missing pairs were selected

The 2026-09-17/18 non-LLM historical full manifest declared **2,036**
model/dataset/seed cases. Archived formal summaries contain at least one
attempted row for **1,617** unique declared cases. The difference is exactly
**419** cases: 150 SymbolicGPT and 269 DSO. This is based on case identity,
not on `status=ok`: an error or timeout is an attempted case, not a missing
case. The filtered current-code manifests are
`results/server_gpu_missing_preflight_20260925/missing_{symbolicgpt,dso}_manifest.json`.
The current preflight returned **419/419 compatible**, with no incompatibility
or adapter error. Current code and budgets are recorded in each case manifest;
these continuations must not be silently pooled with the older implementation
or metric protocol.

The current PDE preflight for `weakform` and `sga` found **26 compatible**
combinations (13 each), 20 explicit incompatibilities and no errors. Historical
SGA full runs attempted Burgers, KdV and Chafee–Infante, leaving 10 newly
supported SGA pairs. The only historical `weakform` full result was `wdwake`
under the superseded two-dimensional adapter; it is not a result for today's
scalar one-dimensional WSINDy implementation. All 13 current `weakform`
compatible pairs are therefore new full runs. The exact 23-case selection is
`results/server_cpu_missing_preflight_20260925/missing_manifest.json`.
Preflight only establishes that the adapter can enter training. Early SGA
fitting errors of `LinAlgError: SVD did not converge in Linear Least Squares`
are preserved as errors; they are not compatibility failures.

The new PhySO full selection contains the old 274 SR workloads plus 13 ODE
datasets, including `ball_drop` and all twelve generated ODE systems. Its
single-seed ODE rows and configured fallback grammars are diagnostic; they do
not constitute the five-seed, declared-grammar 440-case Core ODE main track.
LLM-SR remains excluded because it has no validated completion endpoint/model.

## Follow-up

After each batch finishes, download its entire result directory with an
archive SHA256 check, verify `benchmark_status.json` or the CLI summary against
the manifest's case count, and report `ok`/`error`/`timeout` target denominators
separately. Review failed cases before any targeted retry; do not overwrite the
first attempt. The PhySO queue must finish before the two GPU gap batches start.

## Last observed progress

At 2026-09-25 22:25 UTC+08:00, CPU PDE had finished 23/23 cases
(99 target rows, 19 error/timeout rows) at 19:21:55. PhySO was processing
case 213/287, and GPU continuation was waiting. This is a historical observation;
recheck logs and exit files before reporting the September 26 state.
