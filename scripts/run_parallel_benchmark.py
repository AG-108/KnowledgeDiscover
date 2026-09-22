"""Run compatible benchmark cases concurrently and organize the results.

This utility consumes one or more manifests produced by ``run_benchmark.py``.
It is intended for long, resumable benchmark runs where CPU cases may execute
in parallel while accelerator cases are kept on a smaller, separate pool.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_benchmark import _base_row, execute_case, save_summary, write_json


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _round_robin(cases: list[dict]) -> list[dict]:
    """Interleave models so early progress covers more than one baseline."""
    grouped: dict[str, deque] = defaultdict(deque)
    for case in cases:
        grouped[case["model"]].append(case)
    ordered = []
    names = sorted(grouped)
    while names:
        remaining = []
        for name in names:
            if grouped[name]:
                ordered.append(grouped[name].popleft())
            if grouped[name]:
                remaining.append(name)
        names = remaining
    return ordered


def _successful_result(path: Path) -> bool:
    try:
        rows = _load_json(path)
    except (OSError, ValueError, TypeError):
        return False
    return bool(rows) and all(row.get("status") == "ok" for row in rows)


def _reuse_successes(cases: list[dict], output: Path, sources: list[Path]) -> int:
    reused = 0
    for case in cases:
        destination = output / "cases" / case["name"]
        if _successful_result(destination / "result.json"):
            continue
        for source in sources:
            candidate = source / "cases" / case["name"]
            if _successful_result(candidate / "result.json"):
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(candidate, destination, dirs_exist_ok=True)
                reused += 1
                break
    return reused


def _status_payload(
    cases: list[dict],
    results: dict[str, list[dict]],
    *,
    started_at: str,
    attempt: int,
    retries: int,
) -> dict:
    expected_by_model = Counter(case["model"] for case in cases)
    finished_by_model = Counter(
        case["model"] for case in cases if case["name"] in results
    )
    rows = [row for case_rows in results.values() for row in case_rows]
    return {
        "started_at": started_at,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "attempt": attempt,
        "maximum_attempts": retries + 1,
        "expected_cases": len(cases),
        "finished_cases": len(results),
        "remaining_cases": len(cases) - len(results),
        "expected_by_model": dict(sorted(expected_by_model.items())),
        "finished_by_model": dict(sorted(finished_by_model.items())),
        "row_statuses": dict(Counter(row.get("status") for row in rows)),
    }


def _failed(case_rows: list[dict]) -> bool:
    return any(row.get("status") in {"error", "timeout"} for row in case_rows)


def _run_one(case: dict, output: Path, timeout: float | None) -> list[dict]:
    try:
        return execute_case(case, output, timeout=timeout, resume=True)
    except Exception as exc:  # Preserve orchestration failures as ordinary rows.
        row = _base_row(case)
        row.update(status="error", error=f"orchestrator: {type(exc).__name__}: {exc}")
        directory = output / "cases" / case["name"]
        directory.mkdir(parents=True, exist_ok=True)
        write_json(directory / "result.json", [row])
        return [row]


def _save_checkpoint(
    output: Path,
    cases: list[dict],
    results: dict[str, list[dict]],
    *,
    started_at: str,
    attempt: int,
    retries: int,
) -> None:
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
    write_json(
        output / "benchmark_status.json",
        _status_payload(
            cases,
            results,
            started_at=started_at,
            attempt=attempt,
            retries=retries,
        ),
    )


def _organize_by_baseline(output: Path, rows: list[dict]) -> None:
    by_model: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_model[row.get("model", "unknown")].append(row)
    overview = []
    for model, model_rows in sorted(by_model.items()):
        directory = output / "by_baseline" / model
        save_summary(directory, model_rows)
        statuses = Counter(row.get("status") for row in model_rows)
        overview.append(
            {
                "baseline": model,
                "rows": len(model_rows),
                "ok": statuses["ok"],
                "skipped": statuses["skipped"],
                "error": statuses["error"],
                "timeout": statuses["timeout"],
            }
        )
    write_json(output / "benchmark_overview.json", overview)
    with (output / "benchmark_overview.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("baseline", "rows", "ok", "skipped", "error", "timeout"),
        )
        writer.writeheader()
        writer.writerows(overview)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", type=Path, required=True)
    parser.add_argument("--compatibility", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reuse-dir", action="append", type=Path, default=[])
    parser.add_argument("--exclude-model", action="append", default=[])
    parser.add_argument("--cpu-workers", type=int, default=6)
    parser.add_argument("--gpu-workers", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=21600.0)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.cpu_workers < 1 or args.gpu_workers < 1:
        raise SystemExit("worker counts must be positive")
    if args.timeout <= 0 or args.retries < 0 or args.checkpoint_every < 1:
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
    source_manifests = []
    for manifest_path in args.manifest:
        resolved = manifest_path.resolve()
        manifest = _load_json(resolved)
        source_manifests.append(str(resolved))
        for case in manifest["cases"]:
            key = (case["task"], case["model"], case["dataset"])
            if case["model"] not in excluded and key in compatible:
                unique[case["name"]] = case
    cases = list(unique.values())
    if not cases:
        raise SystemExit("no compatible cases selected")

    write_json(
        output / "benchmark_manifest.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_manifests": source_manifests,
            "compatibility_report": str(args.compatibility.resolve()),
            "excluded_models": sorted(excluded),
            "cpu_workers": args.cpu_workers,
            "gpu_workers": args.gpu_workers,
            "timeout": args.timeout,
            "retries": args.retries,
            "cases": cases,
        },
    )
    shutil.copy2(args.compatibility.resolve(), output / "compatibility_report.json")
    reused = _reuse_successes(cases, output, [path.resolve() for path in args.reuse_dir])
    print(
        f"Selected {len(cases)} compatible cases; reused {reused} successful cases; "
        f"CPU workers={args.cpu_workers}, GPU workers={args.gpu_workers}",
        flush=True,
    )

    started_at = datetime.now(timezone.utc).isoformat()
    results: dict[str, list[dict]] = {}
    pending = cases
    case_by_name = {case["name"]: case for case in cases}
    for attempt in range(1, args.retries + 2):
        cpu_cases = _round_robin(
            [case for case in pending if not str(case["device"]).startswith("cuda")]
        )
        gpu_cases = _round_robin(
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
                future_cases[cpu_pool.submit(_run_one, case, output, args.timeout)] = case
            for case in gpu_cases:
                future_cases[gpu_pool.submit(_run_one, case, output, args.timeout)] = case
            completed_this_attempt = 0
            checkpoint_time = time.monotonic()
            for future in as_completed(future_cases):
                case = future_cases[future]
                case_rows = future.result()
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
                    _save_checkpoint(
                        output,
                        cases,
                        results,
                        started_at=started_at,
                        attempt=attempt,
                        retries=args.retries,
                    )
                    checkpoint_time = time.monotonic()

        _save_checkpoint(
            output,
            cases,
            results,
            started_at=started_at,
            attempt=attempt,
            retries=args.retries,
        )
        failed_names = [name for name, rows in results.items() if _failed(rows)]
        if not failed_names:
            break
        pending = [case_by_name[name] for name in failed_names]
        print(f"Retrying {len(pending)} failed/timeout cases", flush=True)

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
    _organize_by_baseline(output, rows)
    failures = sum(row.get("status") in {"error", "timeout"} for row in rows)
    print(
        f"Finished {len(cases)} cases / {len(rows)} rows; failures={failures}; "
        f"results={output}",
        flush=True,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
