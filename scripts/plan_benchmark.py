"""Prepare target denominators without fitting, then emit resumable <=40h shards."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kd.benchmark_identity import restore_paths  # noqa: E402
from kd.benchmark_identity import verify as verify_identity  # noqa: E402
from kd.benchmark_planning import plan_chunks  # noqa: E402
from kd.benchmark_store import read_json, seal_matrix_plan, verify_matrix_plan  # noqa: E402
from run_benchmark import _ensure_target_plan, case_lock, save_manifest, write_json  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true", help="Load/partition data in bounded workers; never fit")
    parser.add_argument("--max-hours", type=float, default=40)
    parser.add_argument("--overhead-hours", type=float, default=0.5)
    args = parser.parse_args(argv)
    path = args.manifest.resolve()
    directory = path.parent
    manifest = read_json(path)
    bindings_path = directory / "runtime_asset_paths.json"
    bindings = read_json(bindings_path) if bindings_path.is_file() else {}
    cases = restore_paths(manifest["cases"], bindings)
    if not manifest.get("identity") or any(not case.get("resource_policy") for case in cases):
        raise ValueError("Prepare a fingerprinted D2 parent manifest with run_slots.py --dry-run first")
    verify_identity(manifest["identity"], ROOT, cases)
    if args.prepare:
        for case in cases:
            case_dir = directory / "cases" / case["name"]
            policy = dict(manifest["execution_policy"], resource_policy=case["resource_policy"])
            with case_lock(case_dir):
                if not (case_dir / "case.json").exists():
                    write_json(case_dir / "case.json", case)
                    write_json(case_dir / "execution.json", policy)
                write_json(case_dir / "case_runtime.json", case)
                write_json(case_dir / "execution_runtime.json", policy)
                _ensure_target_plan(case_dir, case, policy)
        seal_matrix_plan(directory)
    index = verify_matrix_plan(directory)
    counts = {case["name"]: len(read_json(directory / "cases" / case["name"] / "targets.json")) for case in cases}
    plan = plan_chunks(cases, counts, manifest["execution_policy"], max_hours=args.max_hours,
                       overhead_hours=args.overhead_hours)
    plan.update(parent_matrix_hash=manifest["parent_matrix_hash"],
                experiment_identity_hash=manifest["experiment_identity_hash"], matrix_plan_sha256=index["sha256"])
    by_name = {case["name"]: case for case in cases}
    for chunk in plan["chunks"]:
        output = directory / "chunks" / chunk["chunk_id"]
        selected = copy.deepcopy(manifest)
        selected["cases"] = [by_name[name] for name in chunk["cases"]]
        selected["chunk_id"] = chunk["chunk_id"]
        selected["matrix_plan_sha256"] = index["sha256"]
        save_manifest(output, selected)
        write_json(output / "runtime_asset_paths.json", bindings)
        for name in chunk["cases"]:
            # Copy the exact prepared denominator, never re-enumerate during merge.
            source = directory / "cases" / name / "targets.json"
            target = output / "cases" / name / "targets.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.read_bytes() != source.read_bytes():
                raise ValueError("Existing shard target plan changed")
            target.write_bytes(source.read_bytes())
        chunk["manifest"] = f"chunks/{chunk['chunk_id']}/benchmark_manifest.json"
    write_json(directory / "queue_plan.json", plan)
    print(json.dumps({key: plan[key] for key in ("slots", "n_chunks", "estimate_kind")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
