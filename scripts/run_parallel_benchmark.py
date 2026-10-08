"""Run compatible benchmark cases concurrently and organize the results.

This utility consumes one or more manifests produced by ``run_benchmark.py``.
It is intended for long, resumable benchmark runs where CPU cases may execute
in parallel while accelerator cases are kept on a smaller, separate pool.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kd.benchmark_batch import organize_by_baseline, round_robin, save_checkpoint  # noqa: E402
from kd.benchmark_identity import attach as attach_identity  # noqa: E402
from kd.benchmark_identity import restore_paths  # noqa: E402
from kd.benchmark_identity import verify as verify_identity  # noqa: E402
from kd.benchmark_protocol import (  # noqa: E402
    inherited_policy,
    read_manifests,
    validate_shared_manifests,
)
from kd.benchmark_store import freeze_run, verify_freeze  # noqa: E402
from kd.dataset._metadata import DATASET_PROTOCOL_VERSION  # noqa: E402
from run_benchmark import (  # noqa: E402
    EXECUTION_PROTOCOL_VERSION,
    case_lock,
    execute_case,
    execution_policy,
    save_manifest,
    save_summary,
    write_json,
)


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _reuse_results(cases: list[dict], output: Path, sources: list[Path], policy: dict) -> int:
    """Copy a compatible first attempt, including terminal failures or interruptions."""
    reused = 0
    for case in cases:
        destination = output / "cases" / case["name"]
        if destination.exists():
            continue
        for source in sources:
            candidate = source / "cases" / case["name"]
            if not (candidate / "execution.json").exists():
                continue
            with case_lock(candidate):
                if (_load_json(candidate / "execution.json") != policy
                        or _load_json(candidate / "case.json") != case):
                    continue
                with case_lock(destination):
                    if any(path.name != ".lock" for path in destination.iterdir()):
                        break
                    shutil.copytree(candidate, destination, dirs_exist_ok=True,
                                    ignore=shutil.ignore_patterns(".lock"))
            reused += 1
            break
    return reused


def _run_one(case: dict, output: Path, timeout: float | None, **timing) -> list[dict]:
    # Configuration/I/O/locking errors belong to the orchestrator, not a fake
    # case-level row that would overwrite completed targets or hide denominators.
    return execute_case(case, output, timeout=timeout, resume=True, **timing)


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", type=Path, required=True)
    parser.add_argument("--compatibility", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reuse-dir", action="append", type=Path, default=[])
    parser.add_argument("--exclude-model", action="append", default=[])
    parser.add_argument("--cpu-workers", type=int, default=1,
                        help="Diagnostic pool only; use scripts/run_slots.py for the D2 resource protocol")
    parser.add_argument("--gpu-workers", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=None, help="Fit seconds per target (inherits source)")
    parser.add_argument("--soft-deadline", action=argparse.BooleanOptionalAction, default=None,
                        help="Return the best completed candidate at T (default enabled)")
    parser.add_argument("--timeout-grace", type=float, default=None)
    parser.add_argument("--setup-timeout", type=float, default=None)
    parser.add_argument("--score-timeout", type=float, default=None)
    parser.add_argument("--retries", type=int, choices=[0], default=0,
                        help="Terminal failures are immutable; automatic retries are disabled")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    manifests = read_manifests(args.manifest)
    validate_shared_manifests(manifests)
    if any(manifest.get("benchmark_mode") == "main" for manifest in manifests):
        raise SystemExit("Formal main matrices require scripts/run_slots.py and D2 resources")
    inherited = inherited_policy(manifests, {key: getattr(args, key) for key in
                                 ("timeout", "soft_deadline", "timeout_grace", "setup_timeout", "score_timeout")})
    args.timeout = inherited.pop("timeout")
    if args.cpu_workers < 1 or args.gpu_workers < 1:
        raise SystemExit("worker counts must be positive")
    if (args.timeout is not None and args.timeout <= 0) or args.retries < 0 or args.checkpoint_every < 1:
        raise SystemExit("timeout/checkpoint must be positive and retries non-negative")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    compatibility = _load_json(args.compatibility.resolve())
    compatible = {
        (row["task"], row["baseline"], row["dataset"])
        for row in compatibility["pairs"]
        if row["status"] == "compatible"
    }
    excluded = set(args.exclude_model)
    unique = {}
    timing = inherited
    policy = execution_policy(args.timeout, **timing)
    source_manifests = []
    for manifest_path in args.manifest:
        resolved = manifest_path.resolve()
        manifest = _load_json(resolved)
        bindings_path = resolved.parent / "runtime_asset_paths.json"
        source_cases = restore_paths(manifest["cases"], _load_json(bindings_path) if bindings_path.is_file() else {})
        if manifest.get("identity"):
            verify_identity(manifest["identity"], ROOT, source_cases)
        if manifest.get("execution_protocol_version") != EXECUTION_PROTOCOL_VERSION:
            raise SystemExit("Legacy manifest; generate a new target-v1 manifest")
        if any(case.get("dataset_protocol_version") != DATASET_PROTOCOL_VERSION
               for case in manifest["cases"]):
            raise SystemExit(f"Legacy dataset protocol; generate a new {DATASET_PROTOCOL_VERSION} manifest")
        source_manifests.append(str(resolved))
        for case in source_cases:
            key = (case["task"], case["model"], case["dataset"])
            if case["model"] not in excluded and key in compatible:
                unique[case["name"]] = case
    cases = list(unique.values())
    if not cases:
        raise SystemExit("no compatible cases selected")

    manifest = {
            "benchmark_mode": manifests[0].get("benchmark_mode", "diagnostic"),
            "configuration_sources": manifests[0].get("configuration_sources", {}),
            "execution_protocol_version": EXECUTION_PROTOCOL_VERSION,
            "dataset_protocol_version": DATASET_PROTOCOL_VERSION,
            "execution_policy": policy,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_manifests": source_manifests,
            "compatibility_report": str(args.compatibility.resolve()),
            "excluded_models": sorted(excluded),
            "cpu_workers": args.cpu_workers,
            "gpu_workers": args.gpu_workers,
            "timeout": args.timeout,
            "retries": args.retries,
            "cases": cases,
        }
    bindings = attach_identity(manifest, ROOT)
    cases = manifest["cases"]
    save_manifest(output, manifest)
    write_json(output / "runtime_asset_paths.json", bindings)
    if (output / "benchmark_freeze.json").exists():
        verify_freeze(output)
    shutil.copy2(args.compatibility.resolve(), output / "compatibility_report.json")
    reused = _reuse_results(cases, output, [path.resolve() for path in args.reuse_dir], policy)
    print(
        f"Selected {len(cases)} compatible cases; reused {reused} case histories; "
        f"CPU workers={args.cpu_workers}, GPU workers={args.gpu_workers}",
        flush=True,
    )

    started_at = datetime.now(timezone.utc).isoformat()
    results: dict[str, list[dict]] = {}
    pending = cases
    orchestration_errors = {}
    attempt = 1
    cpu_cases = round_robin(
        [case for case in pending if not str(case["device"]).startswith("cuda")]
    )
    gpu_cases = round_robin(
        [case for case in pending if str(case["device"]).startswith("cuda")]
    )
    print(
        f"Attempt {attempt}/{args.retries + 1}: "
        f"CPU cases={len(cpu_cases)}, GPU cases={len(gpu_cases)}",
        flush=True,
    )
    future_cases: dict[Future, dict] = {}
    with ThreadPoolExecutor(max_workers=args.cpu_workers) as cpu_pool, ThreadPoolExecutor(
        max_workers=args.gpu_workers
    ) as gpu_pool:
        for case in cpu_cases:
            future_cases[cpu_pool.submit(_run_one, case, output, args.timeout, **timing)] = case
        for case in gpu_cases:
            future_cases[gpu_pool.submit(_run_one, case, output, args.timeout, **timing)] = case
        completed_this_attempt = 0
        checkpoint_time = time.monotonic()
        for future in as_completed(future_cases):
            case = future_cases[future]
            try:
                case_rows = future.result()
            except Exception as exc:
                orchestration_errors[case["name"]] = f"{type(exc).__name__}: {exc}"
                write_json(output / "orchestration_errors.json", orchestration_errors)
                print(f"Orchestration error: {case['name']}: {exc}", flush=True)
                continue
            results[case["name"]] = case_rows
            completed_this_attempt += 1
            statuses = ",".join(row.get("status", "unknown") for row in case_rows)
            print(
                f"[{len(results)}/{len(cases)}] {case['name']} -> {statuses}",
                flush=True,
            )
            if (
                completed_this_attempt % args.checkpoint_every == 0
                or time.monotonic() - checkpoint_time >= 60
            ):
                save_checkpoint(
                    output,
                    cases,
                    results,
                    started_at=started_at,
                    attempt=attempt,
                    retries=args.retries,
                )
                checkpoint_time = time.monotonic()

    save_checkpoint(
        output,
        cases,
        results,
        started_at=started_at,
        attempt=attempt,
        retries=args.retries,
    )

    rows = [row for case_rows in results.values() for row in case_rows]
    rows.sort(
        key=lambda row: (
            row.get("task", ""),
            row.get("model", ""),
            row.get("dataset", ""),
            row.get("instance", ""),
            row.get("target", ""),
            row.get("seed", -1),
        )
    )
    save_summary(output, rows)
    organize_by_baseline(output, rows)
    freeze_run(output)
    failures = sum(row.get("status") in {"error", "timeout"} for row in rows)
    print(
        f"Finished {len(cases)} cases / {len(rows)} rows; failures={failures}; "
        f"results={output}",
        flush=True,
    )
    return 1 if failures or orchestration_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
