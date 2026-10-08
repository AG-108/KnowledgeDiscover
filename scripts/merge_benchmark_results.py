"""Merge disjoint sealed shards against an explicitly frozen parent matrix."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kd.benchmark_identity import canonical_case, file_digest, stable_execution  # noqa: E402
from kd.benchmark_protocol import validate_shared_manifests  # noqa: E402
from kd.benchmark_store import (  # noqa: E402
    contained,
    freeze_run,
    read_json,
    verify_freeze,
    verify_matrix_plan,
)
from kd.evaluation.frozen_report import load_frozen  # noqa: E402
from run_benchmark import save_manifest, save_summary, write_json  # noqa: E402


def merge_results(output_dir: Path, shard_dirs: list[Path], *, parent_manifest: Path) -> list[dict]:
    parent_manifest = Path(parent_manifest).resolve()
    parent_dir = parent_manifest.parent
    verify_matrix_plan(parent_dir)
    parent = read_json(parent_manifest)
    parents = {case["name"]: case for case in parent["cases"]}
    if len(parents) != len(parent["cases"]):
        raise ValueError("Duplicate case in parent matrix")
    if not parent.get("identity") or not parent.get("parent_matrix_hash"):
        raise ValueError("Merge requires a fingerprinted frozen parent matrix")
    rows, shards, covered, copies = [], [], set(), {}
    for shard_dir in map(Path, shard_dirs):
        seal = verify_freeze(shard_dir)
        manifest_path = shard_dir / "benchmark_manifest.json"
        shard, _, _ = load_frozen(shard_dir, file_digest(manifest_path))
        validate_shared_manifests([parent, shard])
        if stable_execution(shard.get("execution_policy")) != stable_execution(parent.get("execution_policy")):
            raise ValueError("Shard timing budget differs from frozen parent matrix")
        for case in shard["cases"]:
            name = case["name"]
            if name in covered:
                raise ValueError(f"Duplicate/overlapping shard case: {name}")
            if name not in parents or canonical_case(case) != canonical_case(parents[name]):
                raise ValueError(f"Shard configuration differs from parent matrix: {name}")
            relative = f"cases/{name}/targets.json"
            if file_digest(contained(shard_dir, relative)) != file_digest(contained(parent_dir, relative)):
                raise ValueError(f"Shard target denominator differs from parent: {name}")
            covered.add(name)
        for relative in seal["input_sha256"]:
            if relative.startswith("cases/"):
                copies[relative] = contained(shard_dir, relative)
        shard_rows = read_json(shard_dir / "benchmark_summary.json")
        rows.extend(shard_rows)
        shards.append({"directory": shard_dir.name, "freeze_sha256": seal["sha256"],
                       "rows": len(shard_rows), "cases": len(shard["cases"]),
                       "statuses": dict(Counter(row.get("status") for row in shard_rows))})
    for name in parents:
        relative = f"cases/{name}/targets.json"
        copies.setdefault(relative, contained(parent_dir, relative))
    output_dir = Path(output_dir).resolve()
    for relative, source in copies.items():
        destination = contained(output_dir, relative)
        if destination.exists() and file_digest(destination) != file_digest(source):
            raise ValueError(f"Existing first declaration/result differs: {relative}")
    if (output_dir / "benchmark_freeze.json").exists():
        verify_freeze(output_dir)
    save_manifest(output_dir, parent)
    for relative, source in copies.items():
        destination = contained(output_dir, relative)
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.read_bytes())
    rows.sort(key=lambda row: (row["name"], row["target_id"]))
    save_summary(output_dir, rows)
    write_json(output_dir / "benchmark_shards.json", {
        "parent_manifest_sha256": file_digest(parent_manifest), "shards": shards,
        "n_covered_cases": len(covered), "n_missing_cases": len(parents) - len(covered),
        "denominator": "all targets in frozen parent; missing shards remain in report denominator",
    })
    freeze_run(output_dir)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("shard_dirs", type=Path, nargs="+")
    parser.add_argument("--parent-manifest", type=Path, required=True)
    args = parser.parse_args()
    rows = merge_results(args.output_dir, args.shard_dirs, parent_manifest=args.parent_manifest)
    print(
        f"Merged {len(rows)} rows into {args.output_dir}: "
        f"{dict(Counter(row.get('status') for row in rows))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
