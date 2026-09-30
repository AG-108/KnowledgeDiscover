"""Export scientific coverage and expanded fit units without training models."""

import argparse
import csv
import hashlib
import inspect
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_benchmark as benchmark  # noqa: E402
from kd.dataset._domain import constitutive_instance_specs  # noqa: E402
from kd.dataset._metadata import DATASET_PROTOCOL_VERSION, pending_datasets  # noqa: E402


def instance_manifest(name, card):
    """Names/counts from source metadata; does not generate PDEformer solutions."""
    if name.startswith("solid_"):
        return [{**s, "n_targets": 1} for s in constitutive_instance_specs(name)]
    if name == "wave_breaking":
        with (ROOT / "kd/dataset/WaveBreaking.pkl").open("rb") as stream:
            waves = pickle.load(stream)
        return [{"instance": key, "n_targets": 1} for key in sorted(waves)]
    if name == "pdeformer_sinus":
        from kd.dataset._pdeformer_sinus import load_pdeformer_sinus_benchmark

        count = inspect.signature(load_pdeformer_sinus_benchmark).parameters["n_pde"].default
        return [{"instance": f"{name}/{i:04d}", "n_targets": 1} for i in range(count)]
    count = 2 if name == "pyqg_subgrid" else 1
    if name.startswith("ode_core_"):
        from kd.dataset._ode_core import core_equations

        count = len(core_equations(name.removeprefix("ode_core_")))
    elif name == "ball_drop":
        count = 2
    elif card["task"] in {"pde", "steady_pde"}:
        from kd.dataset._advanced import declared_targets
        count = len(declared_targets(name))
    return [{"instance": name, "n_targets": count}]


def coverage_inventory(*, expand_instances=False, sha256=False):
    catalog, _ = benchmark.dataset_catalog()
    data_root = ROOT / "kd/dataset"
    rows = [{"dataset": name, "numerical_task": card["task"], **card}
            for name, card in catalog.items()] + pending_datasets()
    for row in rows:
        sources = []
        missing_patterns = []
        for pattern in row["source_paths"]:
            paths = sorted(p for p in data_root.glob(pattern) if p.is_file())
            if not paths:
                missing_patterns.append(pattern)
            for path in paths:
                item = {"path": path.relative_to(data_root).as_posix(), "bytes": path.stat().st_size}
                if sha256:
                    digest = hashlib.sha256()
                    with path.open("rb") as stream:
                        for block in iter(lambda: stream.read(1024 * 1024), b""):
                            digest.update(block)
                    item["sha256"] = digest.hexdigest()
                sources.append(item)
        row.update(source_files=sources, missing_patterns=missing_patterns,
                   available=bool(sources) and not missing_patterns,
                   schedulable=row["dataset"] in catalog and row.get("role") != "test_only")
        if expand_instances and row["schedulable"]:
            try:
                row["instances"] = instance_manifest(row["dataset"], row)
                row["n_instances"] = len(row["instances"])
                row["n_targets_per_method_seed"] = sum(i["n_targets"] for i in row["instances"])
                row["instance_manifest_complete"] = True
            except (OSError, ValueError) as exc:
                row.update(instance_manifest_complete=False, instance_error=str(exc),
                           n_instances=None, n_targets_per_method_seed=None)
    cases = benchmark.build_experiments(catalog, profile="full")
    counts = Counter(c["dataset"] for c in cases)
    by_name = {row["dataset"]: row for row in rows}
    known_fits = 0
    fits_by_track = Counter()
    if expand_instances:
        for case in cases:
            row = by_name[case["dataset"]]
            instances = row.get("instances") or []
            if case["task"] in {"pde", "steady_pde"} and case["model"] not in {"pdefind_nd", "implicit_pde"}:
                fit_count = len(instances)
            else:
                fit_count = sum(instance["n_targets"] for instance in instances)
            known_fits += fit_count
            fits_by_track[row["scientific_task"]] += fit_count
    opt_in = [name for name,card in catalog.items() if card.get("opt_in")]
    opt_cases = benchmark.build_experiments(catalog,profile="full",datasets=opt_in)
    opt_fits = 0
    if expand_instances:
        for case in opt_cases:
            opt_fits += sum(i["n_targets"] for i in by_name[case["dataset"]].get("instances",[]))
    return {
        "opt_in_catalog_names":len(opt_in),
        "opt_in_capacity_cases_one_seed":len(opt_cases),
        "opt_in_capacity_target_fits_one_seed":opt_fits if expand_instances else None,
        "dataset_protocol_version": DATASET_PROTOCOL_VERSION,
        "scope": "default full profile, all models, seed 0; includes unsupported pairs",
        "catalog_names": len(catalog), "independent_names": len(counts),
        "cases_one_seed": len(cases), "case_task_counts": dict(Counter(c["task"] for c in cases)),
        "expanded_target_fits_one_seed": known_fits if expand_instances else None,
        "target_fits_by_scientific_task": dict(fits_by_track) if expand_instances else None,
        "target_count_complete": expand_instances and all(by_name[n].get("instance_manifest_complete") for n in counts),
        "fit_unit": "scalar SR/ODE; independent equation for pdefind_nd/implicit_pde; legacy PDE joint diagnostic",
        "formal_run_ready": False, "rows": rows,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expand-instances", action="store_true",
                        help="Read source instance identities, including trusted local WaveBreaking pickle")
    parser.add_argument("--sha256", action="store_true", help="Hash referenced assets (can be large)")
    args = parser.parse_args(argv)
    report = coverage_inventory(expand_instances=args.expand_instances, sha256=args.sha256)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    benchmark.write_json(args.output_dir / "dataset_inventory.json", report)
    fields = ["dataset", "numerical_task", "scientific_task", "physical_family", "benchmark_role",
              "integration_status", "available", "schedulable", "target_definition", "split_protocol",
              "instance_policy", "reference_status", "source_paths", "requirements", "missing_patterns",
              "n_instances", "n_targets_per_method_seed", "instance_manifest_complete"]
    with (args.output_dir / "dataset_inventory.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in report["rows"]:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                             for k, v in row.items()})
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
