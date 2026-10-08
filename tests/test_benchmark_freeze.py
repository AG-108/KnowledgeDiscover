"""Publication/resume guards tested through real manifests, plans and terminals."""

import copy
import json
import shutil

import pytest

import run_benchmark as benchmark
from kd import benchmark_identity as identity
from kd.benchmark_protocol import comparison_fields, inherited_policy, validate_main_matrix
from kd.benchmark_resources import policy_hash, slot_policy
from kd.benchmark_store import freeze_run, seal_matrix_plan, verify_freeze
from kd.evaluation.frozen_report import load_frozen, tables
from kd.evaluation.reporting import paired_comparison
from scripts.merge_benchmark_results import merge_results
from scripts.run_slots import bind_case

GRAMMAR = ["add", "sub", "mul", "div", "sin", "cos", "exp", "log", "sqrt"]
SOURCES = {"sindy": {"source": "native STLSQ", "implementation_variant": "polynomial_stlsq",
                     "validation_status": "passed"}}


def matrix(root, *, seeds=(0, 1)):
    catalog, _ = benchmark.dataset_catalog()
    cases = benchmark.build_experiments(catalog, models=["sindy"], datasets=["ode_core_damped_oscillator"],
                                        profile="full", seeds=list(seeds))
    for case in cases:
        case.update(task_grammar=GRAMMAR, experiment_identity_hash="experiment")
    manifest = {"cases": cases, "execution_policy": benchmark.execution_policy(5),
                "benchmark_mode": "main", "configuration_sources": SOURCES,
                "identity": {"sha256": "implementation"}, "parent_matrix_hash": "parent",
                "experiment_identity_hash": "experiment"}
    benchmark.save_manifest(root, manifest)
    for case in cases:
        plan = [{"target_id": f"i0000_t{i:04d}", "instance": case["dataset"], "target": target,
                 "ground_truth": truth, "recovery_eligible": True}
                for i, (target, truth) in enumerate(zip(["d(x)/dt", "d(v)/dt"], ["v", "-x-0.2*v"]))]
        benchmark.write_json(root / "cases" / case["name"] / "targets.json", plan)
    seal_matrix_plan(root)
    return manifest


def shard(root, parent, selected, *, omit_last=False):
    manifest = copy.deepcopy(parent)
    manifest["cases"] = selected
    benchmark.save_manifest(root, manifest)
    rows = []
    for case in selected:
        plan = [{"target_id": f"i0000_t{i:04d}", "instance": case["dataset"], "target": target,
                 "ground_truth": truth, "recovery_eligible": True}
                for i, (target, truth) in enumerate(zip(["d(x)/dt", "d(v)/dt"], ["v", "-x-0.2*v"]))]
        case_dir = root / "cases" / case["name"]
        benchmark.write_json(case_dir / "targets.json", plan)
        for spec in plan[:1] if omit_last else plan:
            row = benchmark._base_row(case, spec["instance"], spec["target"])
            row.update(status="ok", stop_reason="native_return", runtime=1.0, exact_recovery=1,
                       target_id=spec["target_id"], execution_policy=parent["execution_policy"])
            row.update(comparison_fields(case, parent["execution_policy"]))
            benchmark.write_json(case_dir / "targets" / spec["target_id"] / "result.json", row)
            rows.append(row)
    benchmark.save_summary(root, rows)
    freeze_run(root)
    return rows


def test_legal_merge_matches_single_run_and_preserves_missing_recovery(tmp_path):
    parent_dir = tmp_path / "parent"
    parent = matrix(parent_dir)
    a, b, direct = [tmp_path / name for name in ("a", "b", "direct")]
    shard(a, parent, parent["cases"][:1])
    shard(b, parent, parent["cases"][1:], omit_last=True)
    shard(direct, parent, parent["cases"][:1])
    # Match the same available terminals in one directory.
    other_rows = json.loads((b / "benchmark_summary.json").read_text())
    shutil.copytree(b / "cases", direct / "cases", dirs_exist_ok=True)
    benchmark.save_manifest(tmp_path / "single", parent)
    shutil.copytree(direct / "cases", tmp_path / "single/cases")
    all_rows = json.loads((a / "benchmark_summary.json").read_text()) + other_rows
    benchmark.save_summary(tmp_path / "single", all_rows)
    freeze_run(tmp_path / "single")
    output = tmp_path / "merged"
    merge_results(output, [a, b], parent_manifest=parent_dir / "benchmark_manifest.json")
    actual = load_frozen(output, identity.file_digest(output / "benchmark_manifest.json"))[1]
    expected = load_frozen(tmp_path / "single", identity.file_digest(tmp_path / "single/benchmark_manifest.json"))[1]
    assert tables(actual, "1" * 64) == tables(expected, "1" * 64)
    assert len(actual) == 4
    assert tables(actual, "1" * 64)[0][0]["recovery_lower_bound_macro_mean"] == 0.75
    missing_output = tmp_path / "missing_shard"
    merge_results(missing_output, [a], parent_manifest=parent_dir / "benchmark_manifest.json")
    missing = load_frozen(missing_output, identity.file_digest(missing_output / "benchmark_manifest.json"))[1]
    assert sum(row["status"] == "missing" for row in missing) == 2
    assert tables(missing, "1" * 64)[0][0]["recovery_lower_bound_macro_mean"] == 0.5


@pytest.mark.parametrize("change", ["duplicate", "budget", "configuration", "parent", "summary", "terminal", "plan"])
def test_invalid_shards_are_rejected_before_output_mutation(tmp_path, change):
    parent_dir = tmp_path / "parent"
    parent = matrix(parent_dir)
    source = tmp_path / "source"
    shard(source, parent, parent["cases"][:1])
    inputs = [source, source] if change == "duplicate" else [source]
    if change in {"budget", "configuration", "parent"}:
        path = source / "benchmark_manifest.json"
        data = json.loads(path.read_text())
        if change == "budget":
            data["execution_policy"]["timeout"] = 100
        elif change == "configuration":
            data["cases"][0]["model_params"]["polynomial_degree"] = 3
        else:
            data["parent_matrix_hash"] = "other"
    elif change in {"summary", "terminal"}:
        path = (source / "benchmark_summary.json" if change == "summary" else
                source / "cases" / parent["cases"][0]["name"] / "targets/i0000_t0000/result.json")
        data = json.loads(path.read_text())
        (data[0] if isinstance(data, list) else data)["exact_recovery"] = 0
    elif change == "plan":
        path = source / "cases" / parent["cases"][0]["name"] / "targets.json"
        data = json.loads(path.read_text())
        data.pop()
    if change != "duplicate":
        benchmark.write_json(path, data)
    output = tmp_path / "invalid"
    with pytest.raises(ValueError):
        merge_results(output, inputs, parent_manifest=parent_dir / "benchmark_manifest.json")
    assert not output.exists()


def test_summary_must_match_first_terminal_even_before_first_seal(tmp_path):
    parent = matrix(tmp_path / "parent", seeds=(0,))
    shard(tmp_path / "run", parent, parent["cases"])
    (tmp_path / "run/benchmark_freeze.json").unlink()
    path = tmp_path / "run/benchmark_summary.json"
    rows = json.loads(path.read_text())
    rows[0]["runtime"] = 100
    benchmark.write_json(path, rows)
    with pytest.raises(ValueError, match="first terminal"):
        freeze_run(tmp_path / "run")


def test_partial_resume_adds_seal_without_rewriting_first_terminal(tmp_path):
    parent = matrix(tmp_path / "parent", seeds=(0,))
    rows = shard(tmp_path / "run", parent, parent["cases"], omit_last=True)
    run = tmp_path / "run"
    first = run / "cases" / parent["cases"][0]["name"] / "targets/i0000_t0000/result.json"
    original = first.read_bytes()
    old = verify_freeze(run)["sha256"]
    new = dict(rows[0], target_id="i0000_t0001", target="d(v)/dt")
    benchmark.write_json(first.parent.parent / "i0000_t0001/result.json", new)
    benchmark.save_summary(run, rows + [new])
    assert freeze_run(run)["sha256"] != old
    assert first.read_bytes() == original
    assert (run / "freezes" / f"{old}.json").is_file()


def test_timing_inherits_strict_control_and_main_override_is_rejected():
    control = {"benchmark_mode": "control", "execution_policy": benchmark.execution_policy(1000, soft_deadline=False)}
    assert inherited_policy([control])["timeout"] == 1000
    assert inherited_policy([control])["soft_deadline"] is False
    with pytest.raises(ValueError, match="different timing"):
        inherited_policy([control, {"execution_policy": benchmark.execution_policy(5)}])
    with pytest.raises(ValueError, match="cannot be overridden"):
        inherited_policy([{**control, "benchmark_mode": "main"}], {"timeout": 5})


def test_uniform_grammar_and_configuration_gates(tmp_path):
    parent = matrix(tmp_path)
    validate_main_matrix(parent["cases"], parent["execution_policy"], configuration_sources=SOURCES)
    other = copy.deepcopy(parent["cases"][0])
    other["dataset"] = "ode_core_pendulum"
    other["task_grammar"] = ["add", "mul"]
    with pytest.raises(ValueError, match="grammar differs"):
        validate_main_matrix(parent["cases"] + [other], parent["execution_policy"])
    other["task_grammar"] = GRAMMAR
    other["model_params"]["polynomial_degree"] = 3
    with pytest.raises(ValueError, match="violates D3"):
        validate_main_matrix(parent["cases"] + [other], parent["execution_policy"])


def test_service_pid_changes_preserve_v2_identity_and_v1_stays_distinct(tmp_path):
    parent = matrix(tmp_path)
    gpu = slot_policy("gpu", service_pids=[101], service_bindings=[{"pid": 101, "create_time": 1.0}])
    restarted = dict(gpu, service_pids=[102], service_bindings=[{"pid": 102, "create_time": 2.0}])
    case = dict(parent["cases"][0], model="llmsr", model_params={})
    assert policy_hash(gpu) == policy_hash(restarted)
    assert bind_case(case, {"gpu": gpu})["name"] == bind_case(case, {"gpu": restarted})["name"]
    assert policy_hash(dict(gpu, version="slots-v1")) != policy_hash(dict(restarted, version="slots-v1"))


def test_portable_slots_preserve_identity_only_for_same_hardware(tmp_path):
    from kd.benchmark_resources import validate_policy

    parent = matrix(tmp_path)
    profile = {"cpu_model": "fixture CPU", "machine": "x86_64",
               "gpu": {"name": "fixture GPU", "memory_mb": 24000, "driver_version": "1"}}
    first = slot_policy("gpu", cpus=[4, 5, 6, 7], gpu="GPU-first", hardware_profile=profile)
    moved = slot_policy("gpu", cpus=[60, 61, 62, 63], gpu="GPU-next", hardware_profile=profile)
    validate_policy(first)
    validate_policy(moved)
    case = dict(parent["cases"][0], model="physo", model_params={})
    assert policy_hash(first) == policy_hash(moved)
    assert bind_case(case, {"gpu": first})["name"] == bind_case(case, {"gpu": moved})["name"]
    changed = copy.deepcopy(moved)
    changed["hardware_profile"]["gpu"]["name"] = "different GPU"
    assert policy_hash(first) != policy_hash(changed)
    legacy_first = slot_policy("gpu", cpus=[4, 5, 6, 7], gpu="GPU-first")
    legacy_moved = slot_policy("gpu", cpus=[60, 61, 62, 63], gpu="GPU-next")
    assert policy_hash(legacy_first) != policy_hash(legacy_moved)


def test_portable_slot_hardware_is_checked_before_launch(monkeypatch):
    import psutil

    import kd.benchmark_resources as resources

    profile = {"cpu_model": "fixture CPU", "machine": "x86_64", "gpu": None}
    policy = slot_policy("cpu", cpus=psutil.Process().cpu_affinity()[:4], hardware_profile=profile)
    monkeypatch.setattr(resources, "host_hardware_profile", lambda *args: profile)
    resources.validate_host(policy)
    monkeypatch.setattr(resources, "host_hardware_profile", lambda *args: dict(profile, cpu_model="different"))
    with pytest.raises(ValueError, match="hardware profile changed"):
        resources.validate_host(policy)


def test_portable_rebinding_keeps_manifest_and_accepts_actual_result_bindings(tmp_path):
    parent_dir, run = tmp_path / "parent", tmp_path / "portable"
    parent = matrix(parent_dir, seeds=(0,))
    profile = {"cpu_model": "fixture CPU", "machine": "x86_64", "gpu": None}
    original = slot_policy("cpu", cpus=[0, 1, 2, 3], hardware_profile=profile)
    moved = slot_policy("cpu", cpus=[40, 41, 42, 43], hardware_profile=profile)
    source = parent["cases"][0]
    first = bind_case(source, {"cpu": original})
    next_case = bind_case(source, {"cpu": moved})
    manifest = dict(parent, cases=[first], resource_slots={"cpu": original})
    benchmark.save_manifest(run, manifest)
    manifest_path = run / "benchmark_manifest.json"
    original_bytes = manifest_path.read_bytes()
    benchmark.save_manifest(run, dict(manifest, cases=[next_case], resource_slots={"cpu": moved}))
    assert manifest_path.read_bytes() == original_bytes
    plan = json.loads((parent_dir / "cases" / source["name"] / "targets.json").read_text())
    case_dir = run / "cases" / first["name"]
    benchmark.write_json(case_dir / "targets.json", plan)
    spec = plan[0]
    row = benchmark._base_row(next_case, spec["instance"], spec["target"])
    policy = dict(parent["execution_policy"], resource_policy=moved)
    row.update(status="ok", target_id=spec["target_id"], resource_policy=moved,
               resource_policy_hash=policy_hash(moved), resource_slot="cpu", execution_policy=policy)
    row.update(comparison_fields(next_case, policy))
    benchmark.write_json(case_dir / "targets" / spec["target_id"] / "result.json", row)
    benchmark.save_summary(run, [row])
    freeze_run(run)
    from kd.benchmark_identity import file_digest

    _, actual, _ = load_frozen(run, file_digest(manifest_path))
    assert actual[0]["resource_policy"]["cpu_affinity"] == [40, 41, 42, 43]


@pytest.mark.parametrize("changed", ["code", "asset", "environment"])
def test_identity_rejects_content_changes_but_accepts_path_migration(tmp_path, monkeypatch, changed):
    root = tmp_path / "checkout"
    (root / "kd/dataset/data").mkdir(parents=True)
    (root / "run_benchmark.py").write_text("implementation = 1\n")
    asset = root / "kd/dataset/data/fixture.csv"
    asset.write_text("1,2\n")
    monkeypatch.setattr(identity, "environment_fingerprint", lambda: {"python": "test", "packages": {"numpy": "1"}})
    case = {"name": "fit", "model": "gplearn", "model_params": {},
            "dataset_metadata": {"source_paths": ["data/fixture.csv"]}}
    manifest = {"cases": [case], "execution_policy": benchmark.execution_policy(5)}
    identity.attach(manifest, root)
    moved = tmp_path / "moved"
    shutil.copytree(root, moved)
    identity.verify(manifest["identity"], moved, manifest["cases"])
    if changed == "code":
        (moved / "run_benchmark.py").write_text("implementation = 2\n")
    elif changed == "asset":
        (moved / "kd/dataset/data/fixture.csv").write_text("1,3\n")
    else:
        monkeypatch.setattr(identity, "environment_fingerprint", lambda: {"python": "test", "packages": {"numpy": "2"}})
    with pytest.raises(ValueError, match="changed"):
        identity.verify(manifest["identity"], moved, manifest["cases"])


def test_different_budgets_do_not_aggregate_or_pair(tmp_path):
    parent = matrix(tmp_path, seeds=(0,))
    case = parent["cases"][0]
    row = benchmark._base_row(case, case["dataset"], "d(x)/dt")
    row.update(status="ok", test_nrmse=0.1)
    a, b = dict(row), dict(row, model="other")
    a.update(comparison_fields(case, benchmark.execution_policy(5)))
    b.update(comparison_fields(case, benchmark.execution_policy(10)))
    assert paired_comparison([a, b], "test_nrmse", "sindy", "other")["n_pairs"] == 0
    assert len([entry for entry in benchmark.aggregate_results([a, dict(a, **comparison_fields(case, benchmark.execution_policy(10)))])
                if entry["model"] == "sindy"]) == 2
