"""Deterministic tables, seed-cluster bootstrap and plots from frozen results."""

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from kd.benchmark_resources import policy_hash
from kd.benchmark_protocol import benchmark_role, comparison_fields, recovery_eligibility, validate_main_matrix
from kd.benchmark_identity import stable_execution, stable_resources
from kd.benchmark_store import validate_terminals, verify_freeze

PROTOCOL = "frozen-report-v2"
STATUSES = ("ok", "error", "timeout", "skipped", "missing")
METRICS = ("test_nrmse", "equation_residual_nrmse", "equation_rollout_nrmse", "runtime")
GROUP_FIELDS = ("task", "scientific_task", "metric_protocol_version", "dataset_protocol_version",
                "execution_protocol_version", "profile", "device_group", "evaluation_protocol_hash",
                "resource_policy_hash", "method_config_hash", "execution_policy_hash",
                "experiment_identity_hash")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8")


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def load_frozen(directory, expected_hash):
    """Validate matrix membership and reconstruct every declared target denominator."""
    directory = Path(directory)
    manifest_path = directory / "benchmark_manifest.json"
    if sha256(manifest_path) != expected_hash.lower():
        raise ValueError("Manifest SHA256 mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf8"))
    seal = None
    if manifest.get("identity") or manifest.get("benchmark_mode") == "main":
        seal = verify_freeze(directory)
    if manifest.get("benchmark_mode") == "main":
        validate_main_matrix(manifest["cases"], manifest.get("execution_policy", {}),
                             configuration_sources=manifest.get("configuration_sources", {}))
    summary_path = directory / "benchmark_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf8"))
    if not isinstance(summary, list) or not isinstance(manifest.get("cases"), list):
        raise ValueError("Expected a manifest matrix and a target-row summary")
    if seal:
        validate_terminals(directory, manifest, summary)
    cases = {c["name"]: c for c in manifest["cases"]}
    if len(cases) != len(manifest["cases"]):
        raise ValueError("Duplicate cases in manifest")
    indexed = {}
    for row in summary:
        case = cases.get(row.get("name"))
        if case is None:
            raise ValueError("Summary contains a case outside the frozen matrix")
        for field in ("dataset", "model", "task", "seed", "profile", "device", "dataset_protocol_version", "execution_protocol_version"):
            if row.get(field) != case.get(field):
                raise ValueError(f"Summary differs from frozen case at {field}")
        expected_policy = dict(manifest.get("execution_policy", {}))
        if case.get("resource_policy"):
            expected_policy["resource_policy"] = case["resource_policy"]
        if stable_execution(row.get("execution_policy")) != stable_execution(expected_policy):
            raise ValueError("Summary timing budget differs from frozen manifest")
        if case.get("resource_policy") and (
                stable_resources(row.get("resource_policy")) != stable_resources(case["resource_policy"])
                or row.get("resource_policy_hash") != policy_hash(case["resource_policy"])):
            raise ValueError("Summary resource policy differs from frozen manifest")
        key = (row["name"], row.get("target_id"))
        if key in indexed:
            raise ValueError("Duplicate target row in frozen summary")
        if row.get("status") not in STATUSES[:-1]:
            raise ValueError("Invalid terminal target status")
        indexed[key] = dict(row)
    rows, inputs = [], {"benchmark_manifest.json": expected_hash.lower(),
                        "benchmark_summary.json": sha256(summary_path)}
    if seal:
        inputs.update(seal["input_sha256"])
        inputs["benchmark_freeze.json"] = sha256(directory / "benchmark_freeze.json")
    for name, case in cases.items():
        # Case paths must remain under the supplied frozen directory.
        path = (directory / "cases" / name / "targets.json").resolve()
        path.relative_to(directory.resolve())
        if not path.is_file():
            raise ValueError(f"Missing target plan for {name}; denominator cannot be certified")
        inputs[path.relative_to(directory.resolve()).as_posix()] = sha256(path)
        specs = json.loads(path.read_text(encoding="utf8"))
        seen = set()
        for spec in specs:
            target_id = spec["target_id"]
            if target_id in seen:
                raise ValueError("Duplicate target in plan")
            seen.add(target_id)
            key = (name, target_id)
            eligible = recovery_eligibility(case, spec)
            if "recovery_eligible" in spec and spec["recovery_eligible"] != eligible:
                raise ValueError("Target plan grants recovery without a certified reference")
            row = indexed.pop(key, None)
            if row is None:
                row = {**{k: case.get(k) for k in (*GROUP_FIELDS, "dataset", "model", "seed", "device")},
                       "name": name, "target_id": target_id, "instance": spec["instance"],
                       "target": spec["target"], "status": "missing", "family": case.get("physical_family", case.get("family"))}
                row["metric_protocol_version"] = case.get("metric_protocol_version", "2.1")
                row["recovery_eligible"] = spec.get("recovery_eligible", recovery_eligibility(case, spec))
                row["recovery_evaluator_status"] = "not_run"
            if "recovery_eligible" in spec and row.get("recovery_eligible") != spec["recovery_eligible"]:
                raise ValueError("Summary recovery eligibility differs from declared target plan")
            if not eligible:
                row["recovery_eligible"] = False
            if row.get("instance") != spec["instance"] or row.get("target") != spec["target"]:
                raise ValueError("Summary target identity differs from plan")
            if row.get("target_manifest_complete") is False or spec.get("preparation_row", {}).get("target_manifest_complete") is False:
                raise ValueError("Incomplete instance/target denominator; freeze requires a complete target plan")
            row["device_group"] = "gpu" if str(case.get("device", "cpu")).startswith("cuda") else "cpu"
            row["resource_policy_hash"] = policy_hash(case.get("resource_policy"))
            if row.get("evaluation_protocol", case.get("evaluation", {})) != case.get("evaluation", {}):
                raise ValueError("Summary scoring protocol differs from manifest")
            row["family"] = row.get("family") or case.get("physical_family", case.get("family", case["dataset"]))
            row.update(comparison_fields(case, manifest.get("execution_policy", {})))
            row["benchmark_role"] = benchmark_role(case)
            row["implementation_variant"] = case.get("method_declaration", {}).get("implementation_variant", row.get("implementation_variant", "undeclared_legacy_variant"))
            from kd.dataset._overlap import training_overlap
            row["training_data_overlap"] = training_overlap(case["model"], case["dataset"])
            rows.append(row)
        if not specs:
            raise ValueError("Empty target plan")
    if indexed:
        raise ValueError("Summary contains undeclared targets")
    return manifest, rows, inputs


def _macro(rows, metric):
    """Equal targets per instance, instances per dataset, datasets per family, families per seed."""
    values = defaultdict(list)
    for row in rows:
        if metric == "success_rate":
            value = float(row.get("status") == "ok")
        elif metric == "recovery_lower_bound":
            if row.get("recovery_eligible") is not True:
                continue
            if row.get("status") == "skipped":
                continue
            key = "pde_support_recovery" if row.get("task") in {"pde", "steady_pde"} else "exact_recovery"
            value = float(row.get("status") == "ok" and row.get(key) in (True, 1))
        else:
            value = row.get(metric)
            if metric != "runtime" and row.get("status") != "ok":
                continue
            if not _finite(value):
                continue
        values[(row["seed"], row["family"], row["dataset"], row["instance"])].append(value)
    for width in (3, 2, 1):
        buckets = defaultdict(list)
        for key, data in values.items():
            buckets[key[:width]].append(float(np.mean(data)))
        values = buckets
    return {key[0]: float(np.mean(data)) for key, data in values.items()}


def bootstrap(seed_values, rng, *, samples=2000):
    values = np.asarray(list(seed_values.values()), float)
    if not len(values):
        return {"mean": None, "ci_low": None, "ci_high": None, "n_seeds": 0, "ci_status": "no_evaluable_seeds"}
    result = {"mean": float(values.mean()), "ci_low": None, "ci_high": None,
              "n_seeds": len(values), "ci_status": "insufficient_independent_seeds"}
    if len(values) > 1:
        estimates = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
        low, high = np.quantile(estimates, [0.025, 0.975])
        result.update(ci_low=float(low), ci_high=float(high), ci_status="ok")
    return result


def tables(rows, manifest_hash):
    references = [r for r in rows if r["model"] == "reference"]
    groups = defaultdict(list)
    for row in rows:
        if row["model"] != "reference" and row.get("benchmark_role", "main") == "main":
            groups[tuple(row.get(k) for k in GROUP_FIELDS) + (row["model"],)].append(row)
    main, intervals = [], []
    rng = np.random.default_rng(int(manifest_hash[:16], 16))
    for key, group in sorted(groups.items(), key=repr):
        counts = Counter(r["status"] for r in group)
        entry = {**dict(zip(GROUP_FIELDS + ("model",), key)),
                 "benchmark_role": "main",
                 "recovery_metric": "pde_support_recovery" if group[0].get("task") in {"pde", "steady_pde"} else "structural_recovery",
                 "n_target_rows": len(group), "n_cases": len({r["name"] for r in group}),
                 "n_datasets": len({r["dataset"] for r in group}),
                 "n_seeds": len({r["seed"] for r in group}),
                 **{f"n_{s}": counts[s] for s in STATUSES},
                 "success_rate": counts["ok"] / len(group),
                 "runtime_total_seconds": sum(r["runtime"] for r in group if _finite(r.get("runtime"))),
                 "denominator_unit": "declared_target_row"}
        entry["implementation_variant"] = "; ".join(sorted({r.get("implementation_variant", "undeclared_legacy_variant") for r in group}))
        entry["prior_exposure_status_counts"] = json.dumps(dict(sorted(Counter(
            r.get("training_data_overlap", {}).get("status", "not_recorded") for r in group).items())), sort_keys=True)
        entry["generalization_scope"] = "held-out observations in fixed benchmark matrix; no unseen-equation claim"
        rollout_counts = Counter(r.get("equation_rollout_status") or "not_applicable" for r in group)
        entry["rollout_status_counts"] = json.dumps(dict(sorted(rollout_counts.items())), sort_keys=True)
        entry["system_score_seconds_total"] = sum(
            next((r["system_score_seconds"] for r in group if r["name"] == name and
                  _finite(r.get("system_score_seconds"))), 0.0) for name in {r["name"] for r in group})
        for metric in (*METRICS, "success_rate", "recovery_lower_bound"):
            values = _macro(group, metric)
            estimate = bootstrap(values, rng)
            n_evaluable = sum(1 for r in group if _finite(r.get(metric)) and
                             (r["status"] == "ok" or metric == "runtime"))
            if metric == "success_rate":
                n_evaluable = len(group)
            if metric == "recovery_lower_bound":
                n_evaluable = sum(r.get("recovery_eligible") is True and r.get("status") != "skipped" for r in group)
            intervals.append({**dict(zip(GROUP_FIELDS + ("model",), key)), "metric": metric,
                              **estimate, "n_evaluable_targets": n_evaluable,
                              "n_declared_targets": len(group), "bootstrap_unit": "whole_seed",
                              "ci_scope": "repeat-seed variation for fixed matrix; no cross-system uncertainty",
                              "semantics": "available-success macro mean" if metric in METRICS[:-1]
                              else "all eligible targets; unparsed/failed count as zero" if metric == "recovery_lower_bound"
                              else "macro seed mean"})
            entry[f"{metric}_macro_mean"] = estimate["mean"]
            entry[f"{metric}_evaluable_targets"] = n_evaluable
        main.append(entry)
    return main, intervals, references


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plots(output, main):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    groups = defaultdict(list)
    cohort_fields = tuple(field for field in GROUP_FIELDS if field != "method_config_hash")
    for row in main:
        groups[tuple(row[k] for k in cohort_fields)].append(row)
    artifacts = []
    for key, rows in sorted(groups.items(), key=repr):
        name = hashlib.sha256(repr(key).encode()).hexdigest()[:12]
        destination = output / name
        destination.mkdir(exist_ok=True)
        _json(destination / "cohort.json", dict(zip(cohort_fields, key)))
        fig, ax = plt.subplots(figsize=(max(5, len(rows)), 4))
        bottom = np.zeros(len(rows))
        for status in STATUSES:
            values = np.asarray([r[f"n_{status}"] for r in rows])
            ax.bar(np.arange(len(rows)), values, width=0.65, bottom=bottom, label=status)
            bottom += values
        ax.set_ylabel("Declared target rows")
        ax.set_xticks(np.arange(len(rows)), [r["model"] for r in rows])
        ax.set_xlim(-0.7, len(rows) - 0.3)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.tick_params(axis="x", labelrotation=30)
        ax.legend()
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            path = destination / f"status_stacked.{suffix}"
            fig.savefig(path, dpi=160)
            artifacts.append(path.relative_to(output).as_posix())
        plt.close(fig)
        for metric in METRICS[:-1]:
            valid = [r for r in rows if _finite(r.get(f"{metric}_macro_mean")) and
                     _finite(r.get("runtime_macro_mean")) and r["runtime_macro_mean"] > 0]
            if not valid:
                continue
            fig, ax = plt.subplots(figsize=(6, 4))
            x = np.asarray([r["runtime_macro_mean"] for r in valid])
            y = np.asarray([r[f"{metric}_macro_mean"] for r in valid])
            front = [i for i in range(len(valid)) if not any(
                (x[j] <= x[i] and y[j] <= y[i]) and (x[j] < x[i] or y[j] < y[i]) for j in range(len(valid)))]
            ax.scatter(x, y)
            for i, row in enumerate(valid):
                ax.annotate(f"{row['model']} ({row[f'{metric}_evaluable_targets']}/{row['n_target_rows']})", (x[i], y[i]))
            order = sorted(front, key=lambda i: x[i])
            ax.plot(x[order], y[order], linestyle="--", label="Available-success Pareto front")
            ax.set(xlabel="Macro target wall seconds (including failures)", ylabel=metric)
            ax.set_xscale("log")
            ax.legend()
            fig.tight_layout()
            for suffix in ("png", "pdf"):
                path = destination / f"pareto_{metric}.{suffix}"
                fig.savefig(path, dpi=160)
                artifacts.append(path.relative_to(output).as_posix())
            plt.close(fig)
    return artifacts


def generate(directory, manifest_hash):
    directory = Path(directory)
    _manifest, rows, inputs = load_frozen(directory, manifest_hash)
    main, intervals, references = tables(rows, manifest_hash)
    report_hash = inputs.get("benchmark_freeze.json", manifest_hash)
    output = directory / "analysis" / PROTOCOL / report_hash.lower()[:12]
    output.mkdir(parents=True, exist_ok=True)
    provenance_path = output / "provenance.json"
    if provenance_path.exists():
        old = json.loads(provenance_path.read_text(encoding="utf8"))
        if old["input_sha256"] != inputs:
            raise ValueError("Frozen report inputs changed after first publication")
    _json(output / "main_table.json", main)
    write_csv(output / "main_table.csv", main)
    _json(output / "bootstrap_ci.json", intervals)
    write_csv(output / "bootstrap_ci.csv", intervals)
    _json(output / "references.json", references)
    _json(output / "diagnostics.json", [r for r in rows if r.get("benchmark_role") in {"diagnostic", "oracle"}])
    artifacts = plots(output, main)
    _json(provenance_path, {"protocol": PROTOCOL, "input_sha256": inputs,
                           "manifest_sha256": manifest_hash.lower(), "bootstrap_samples": 2000,
                           "bootstrap_unit": "whole_seed", "confidence": 0.95,
                           "n_target_rows": len(rows), "n_reference_rows": len(references),
                           "figures": artifacts})
    return output
