"""Seal finished run inputs independently of report generation.

Each seal is an append-only snapshot. A resume can add first terminal results,
but cannot change previously sealed declarations, terminals or scoring records.
"""

from __future__ import annotations

import json
from pathlib import Path

from kd.benchmark_identity import file_digest
from kd.benchmark_protocol import digest

SCHEMA = "benchmark-freeze-v1"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def contained(directory, relative):
    directory = Path(directory).resolve()
    path = (directory / relative).resolve()
    path.relative_to(directory)
    return path


def verify_freeze(directory, *, allow_summary_update=False):
    """Validate the latest seal without recapturing a new expected hash."""
    directory = Path(directory)
    index = read_json(directory / "benchmark_freeze.json")
    if index.get("schema") != SCHEMA:
        raise ValueError("Unsupported run freeze schema")
    payload = {key: value for key, value in index.items() if key != "sha256"}
    if digest(payload) != index.get("sha256"):
        raise ValueError("Run freeze index digest mismatch")
    for relative, expected in index["input_sha256"].items():
        if allow_summary_update and relative == "benchmark_summary.json":
            continue
        path = contained(directory, relative)
        if not path.is_file() or file_digest(path) != expected:
            raise ValueError(f"Sealed run input changed: {relative}")
    return index


def validate_terminals(directory, manifest, rows):
    """Summary rows must be projections of first terminals plus cached scoring."""
    directory = Path(directory)
    cases = {case["name"]: case for case in manifest["cases"]}
    seen = set()
    for row in rows:
        key = (row.get("name"), row.get("target_id"))
        if key in seen:
            raise ValueError("Duplicate target row in summary")
        seen.add(key)
        if key[0] not in cases:
            raise ValueError("Summary contains a case outside the frozen matrix")
        case_dir = contained(directory, f"cases/{key[0]}")
        terminal = contained(case_dir, f"targets/{key[1]}/result.json")
        if not terminal.is_file():
            raise ValueError(f"Summary lacks a first terminal record: {key}")
        expected = read_json(terminal)
        preparations = sorted((case_dir / "preparation").glob("*/supervision.json"))
        if preparations:
            preparation = read_json(preparations[-1])
            expected.update(preparation_cpu_seconds=preparation.get("cpu_seconds"),
                            preparation_gpu_occupied_seconds=preparation.get("gpu_occupied_seconds"))
        scoring = case_dir / "scoring/ode_rollout/result.json"
        if cases[key[0]]["task"] == "ode" and scoring.is_file():
            data = read_json(scoring)
            metric = data["metrics"].get(row["instance"], {}).get(row["target"])
            if metric is None:
                metric = {"ode_rollout_protocol_version": "ode-rollout-v1",
                          "equation_rollout_status": data.get("failure_status", "abstained_incomplete_system"),
                          "equation_rollout_nrmse": None}
            expected.update(metric)
            expected.update(system_score_seconds=data["times"].get("score_seconds"),
                            system_score_cpu_seconds=data["times"].get("cpu_seconds"),
                            system_score_gpu_occupied_seconds=data["times"].get("gpu_occupied_seconds"))
        if row != expected:
            raise ValueError(f"Summary differs from first terminal/cached scoring: {key}")


def seal_matrix_plan(directory):
    """Freeze all target denominators before queue splitting or any fitting."""
    directory = Path(directory)
    manifest = read_json(directory / "benchmark_manifest.json")
    paths = [directory / "benchmark_manifest.json"]
    targets = 0
    for case in manifest["cases"]:
        path = contained(directory, "cases/" + case["name"] + "/targets.json")
        specs = read_json(path)
        ids = [spec["target_id"] for spec in specs]
        if not ids or len(ids) != len(set(ids)) or any(
                spec.get("preparation_row", {}).get("target_manifest_complete") is False for spec in specs):
            raise ValueError(f"Cannot certify target denominator: {case['name']}")
        targets += len(specs)
        paths.append(path)
    index = {"schema": "benchmark-matrix-plan-v1",
             "parent_matrix_hash": manifest.get("parent_matrix_hash"),
             "experiment_identity_hash": manifest.get("experiment_identity_hash"),
             "n_cases": len(manifest["cases"]), "n_declared_targets": targets,
             "input_sha256": {path.relative_to(directory).as_posix(): file_digest(path) for path in paths}}
    index["sha256"] = digest(index)
    path = directory / "matrix_plan.json"
    if path.exists() and read_json(path) != index:
        raise ValueError("Frozen matrix target plans changed; create a new parent matrix")
    _write(path, index)
    return index


def verify_matrix_plan(directory):
    directory = Path(directory)
    index = read_json(directory / "matrix_plan.json")
    if index.get("schema") != "benchmark-matrix-plan-v1" or digest(
            {key: value for key, value in index.items() if key != "sha256"}) != index.get("sha256"):
        raise ValueError("Matrix target-plan index digest mismatch")
    for relative, expected in index["input_sha256"].items():
        path = contained(directory, relative)
        if not path.is_file() or file_digest(path) != expected:
            raise ValueError(f"Frozen matrix target plan changed: {relative}")
    return index


def freeze_run(directory):
    """Seal complete target plans and the available first results at run end."""
    directory = Path(directory)
    if (directory / "benchmark_freeze.json").exists():
        verify_freeze(directory, allow_summary_update=True)
    manifest = read_json(directory / "benchmark_manifest.json")
    rows = read_json(directory / "benchmark_summary.json")
    validate_terminals(directory, manifest, rows)
    paths = [directory / "benchmark_manifest.json", directory / "benchmark_summary.json"]
    declared_targets = 0
    terminal_keys = set()
    for case in manifest["cases"]:
        case_dir = contained(directory, "cases/" + case["name"])
        plan_path = case_dir / "targets.json"
        if not plan_path.is_file():
            raise ValueError(f"Missing target plan for {case['name']}; prepare the matrix before sealing")
        plan = read_json(plan_path)
        ids = [spec["target_id"] for spec in plan]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("Empty or duplicate target plan")
        if any(spec.get("preparation_row", {}).get("target_manifest_complete") is False for spec in plan):
            raise ValueError("Incomplete target denominator cannot be sealed")
        declared_targets += len(plan)
        paths.append(plan_path)
        for name in ("case.json", "execution.json", "scoring/ode_rollout/result.json"):
            path = case_dir / name
            if path.is_file():
                paths.append(path)
        paths.extend(sorted((case_dir / "preparation").glob("*/supervision.json")))
        for spec in plan:
            terminal = contained(case_dir, "targets/" + spec["target_id"] + "/result.json")
            if terminal.is_file():
                paths.append(terminal)
                terminal_keys.add((case["name"], spec["target_id"]))
    if {(row["name"], row["target_id"]) for row in rows} != terminal_keys:
        raise ValueError("Summary must contain every declared first terminal exactly once")
    inputs = {path.relative_to(directory).as_posix(): file_digest(path) for path in paths}
    # Retain the previous summary snapshot when a partial run is later resumed.
    summary_relative = f"freezes/summary-{inputs['benchmark_summary.json']}.json"
    summary_snapshot = directory / summary_relative
    if not summary_snapshot.exists():
        summary_snapshot.parent.mkdir(parents=True, exist_ok=True)
        summary_snapshot.write_bytes((directory / "benchmark_summary.json").read_bytes())
    inputs[summary_relative] = file_digest(summary_snapshot)
    index = {"schema": SCHEMA, "input_sha256": inputs,
             "parent_matrix_hash": manifest.get("parent_matrix_hash"),
             "experiment_identity_hash": manifest.get("experiment_identity_hash"),
             "n_declared_targets": declared_targets, "n_terminal_rows": len(rows)}
    index["sha256"] = digest(index)
    snapshot = directory / "freezes" / f"{index['sha256']}.json"
    if not snapshot.exists():
        _write(snapshot, index)
    _write(directory / "benchmark_freeze.json", index)
    return index
