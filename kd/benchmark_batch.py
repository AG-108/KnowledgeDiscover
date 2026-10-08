"""Shared ordering, progress and reports for benchmark batch entry points."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path


def round_robin(cases: list[dict]) -> list[dict]:
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


def status_payload(
    cases: list[dict],
    results: dict[str, list[dict]],
    *,
    started_at: str,
    attempt: int,
    retries: int,
) -> dict:
    expected_by_model = Counter(case["model"] for case in cases)
    finished_by_model = Counter(case["model"] for case in cases if case["name"] in results)
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
        "stop_reasons": dict(Counter(row.get("stop_reason") for row in rows)),
    }


def save_checkpoint(
    output: Path,
    cases: list[dict],
    results: dict[str, list[dict]],
    *,
    started_at: str,
    attempt: int,
    retries: int,
) -> None:
    from run_benchmark import save_summary, write_json

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
        status_payload(
            cases,
            results,
            started_at=started_at,
            attempt=attempt,
            retries=retries,
        ),
    )


def organize_by_baseline(output: Path, rows: list[dict]) -> None:
    from run_benchmark import save_summary, write_json

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
    with (output / "benchmark_overview.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("baseline", "rows", "ok", "skipped", "error", "timeout"),
        )
        writer.writeheader()
        writer.writerows(overview)
