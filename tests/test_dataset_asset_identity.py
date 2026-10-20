"""Input identity follows dataset loader overrides and portable local bindings."""

import copy
import shutil
from pathlib import Path

import pytest

from kd import benchmark_identity as identity
from kd.benchmark_protocol import recovery_eligibility, validate_main_matrix


@pytest.fixture(autouse=True)
def inexpensive_environment(monkeypatch):
    monkeypatch.setattr(identity, "environment_fingerprint", lambda: {"test": True})


def write(path, content=b"original"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def make_case(name, paths, params=None, *, source="registered", case_name="sample"):
    return {
        "name": case_name, "dataset": name, "model": "gplearn", "task": "sr",
        "source": source, "dataset_params": params or {}, "model_params": {},
        "dataset_metadata": {"source_paths": paths},
    }


def attach(root, cases):
    manifest = {"cases": cases, "execution_policy": {}}
    bindings = identity.attach(manifest, root)
    return manifest, bindings


@pytest.mark.parametrize("name,paths", [
    ("fde_pt_fade", ["FDE/data/pt_fade.mat"]),
    ("fde_pt_fade_pde", ["FDE/data/pt_fade.mat"]),
    ("dso_file_Custom-1", ["DeepSymbolicOptimization/data/Custom-1.csv",
                           "DeepSymbolicOptimization/data/Custom-1_test.csv"]),
    ("green_advection", ["Discover_Green_function/data/advection.mat"]),
    ("pyqg_subgrid", ["Discover_Green_function/data/run=*.nc"]),
])
def test_root_override_hashes_actual_files_and_rejects_tampering(tmp_path, name, paths):
    root, data_root = tmp_path / "checkout", tmp_path / "external"
    root.mkdir()
    actual = []
    for pattern in paths:
        relative = pattern.replace("*", "0")
        write(root / "kd/dataset" / relative, b"unused checkout bytes")
        actual.append(write(data_root / relative))
    case = make_case(name, paths, {"root": str(data_root)})
    manifest, bindings = attach(root, [case])
    assets = manifest["identity"]["assets"]
    assert {item["sha256"] for item in assets.values()} == {identity.file_digest(actual[0])}
    assert all(str(data_root) in path for key, path in bindings.items() if key.startswith("dataset:"))
    actual[-1].write_bytes(b"changed external input")
    with pytest.raises(ValueError, match="changed"):
        identity.verify(manifest["identity"], root, manifest["cases"])


@pytest.mark.parametrize("absolute_source", [False, True])
def test_symbolic_overrides_bind_custom_formula_and_function_sets(tmp_path, absolute_source):
    root, data_root = tmp_path / "checkout", tmp_path / "external"
    write(root / "kd/dataset/data/benchmarks.csv", b"unused formula")
    write(root / "kd/dataset/data/function_sets.csv", b"unused grammar")
    formula = write(data_root / "custom.csv", b"custom formula")
    grammar = write(data_root / "function_sets.csv", b"custom grammar")
    params = {"root": str(data_root),
              "benchmark_source": str(formula) if absolute_source else "custom.csv"}
    case = make_case("Nguyen-1", ["data/benchmarks.csv", "data/function_sets.csv"],
                     params, source="symbolic")
    manifest, bindings = attach(root, [case])
    assert {item["sha256"] for item in manifest["identity"]["assets"].values()} == {
        identity.file_digest(formula), identity.file_digest(grammar),
    }
    public = identity.public_manifest(manifest)
    assert all(value.startswith("asset:dataset:")
               for value in public["cases"][0]["dataset_params"].values())
    restored = identity.restore_paths(public["cases"], bindings)
    assert restored[0]["dataset_params"] == {
        "root": str(data_root.resolve()), "benchmark_source": str(formula.resolve()),
    }
    grammar.write_bytes(b"changed grammar")
    with pytest.raises(ValueError, match="changed"):
        identity.verify(manifest["identity"], root, restored)


def test_absolute_formula_override_without_root_uses_default_grammar(tmp_path):
    root = tmp_path / "checkout"
    formula = write(tmp_path / "external/custom.csv", b"custom formula")
    grammar = write(root / "kd/dataset/data/function_sets.csv", b"default grammar")
    case = make_case("Nguyen-1", ["data/benchmarks.csv", "data/function_sets.csv"],
                     {"benchmark_source": str(formula)}, source="symbolic")
    manifest, _ = attach(root, [case])
    assert {item["sha256"] for item in manifest["identity"]["assets"].values()} == {
        identity.file_digest(formula), identity.file_digest(grammar),
    }
    formula.write_bytes(b"changed custom formula")
    with pytest.raises(ValueError, match="changed"):
        identity.verify(manifest["identity"], root, manifest["cases"])


def test_distinct_roots_keep_distinct_asset_records_and_bindings(tmp_path):
    root = tmp_path / "checkout"
    root.mkdir()
    relative = "FDE/data/observed.mat"
    first = write(tmp_path / "first" / relative, b"first observation")
    second = write(tmp_path / "second" / relative, b"second observation")
    cases = [make_case("fde_pt_fade", [relative], {"root": str(path.parent.parent.parent)},
                       case_name=str(index)) for index, path in enumerate((first, second))]
    manifest, bindings = attach(root, cases)
    assert len(manifest["identity"]["assets"]) == 2
    public = identity.public_manifest(manifest)
    assert public["cases"][0]["dataset_params"]["root"] != public["cases"][1]["dataset_params"]["root"]
    restored = identity.restore_paths(public["cases"], bindings)
    assert [case["dataset_params"]["root"] for case in restored] == [
        str(tmp_path / "first"), str(tmp_path / "second"),
    ]
    second.write_bytes(b"modified second observation")
    with pytest.raises(ValueError, match="changed"):
        identity.verify(manifest["identity"], root, restored)


def test_parent_directory_patterns_preserve_expanded_logical_paths(tmp_path):
    root, data_root = tmp_path / "checkout", tmp_path / "external"
    root.mkdir()
    write(data_root / "runs/first/field.nc")
    write(data_root / "runs/second/field.nc")
    case = make_case("pyqg_subgrid", ["runs/*/field.nc"], {"root": str(data_root)})
    manifest, _ = attach(root, [case])
    assert len(manifest["identity"]["assets"]) == 2
    assert all("*" not in key for key in manifest["identity"]["assets"])


@pytest.mark.parametrize("symbolic", [False, True])
def test_identical_inputs_can_move_with_separate_runtime_bindings(tmp_path, symbolic):
    root, original_root, moved_root = (tmp_path / name for name in ("checkout", "original", "moved"))
    root.mkdir()
    if symbolic:
        paths = ["data/benchmarks.csv", "data/function_sets.csv"]
        write(original_root / "custom.csv", b"formula")
        write(original_root / "function_sets.csv", b"grammar")
        case = make_case("Nguyen-1", paths, {"root": str(original_root),
                         "benchmark_source": str(original_root / "custom.csv")}, source="symbolic")
    else:
        paths = ["FDE/data/observed.mat"]
        write(original_root / paths[0])
        case = make_case("fde_pt_fade", paths, {"root": str(original_root)})
    manifest, _ = attach(root, [case])
    public = identity.public_manifest(manifest)
    shutil.copytree(original_root, moved_root)
    moved_case = copy.deepcopy(case)
    moved_case["dataset_params"]["root"] = str(moved_root)
    if symbolic:
        moved_case["dataset_params"]["benchmark_source"] = str(moved_root / "custom.csv")
    moved_manifest, moved_bindings = attach(root, [moved_case])
    assert moved_manifest["identity"]["sha256"] == manifest["identity"]["sha256"]
    assert moved_manifest["experiment_identity_hash"] == manifest["experiment_identity_hash"]
    assert identity.canonical_case(moved_manifest["cases"][0]) == identity.canonical_case(manifest["cases"][0])
    restored = identity.restore_paths(public["cases"], moved_bindings)
    assert Path(restored[0]["dataset_params"]["root"]) == moved_root
    identity.verify(manifest["identity"], root, restored)


def test_missing_dataset_binding_is_rejected(tmp_path):
    root = tmp_path / "checkout"
    root.mkdir()
    case = make_case("fde_pt_fade", ["FDE/data/observed.mat"], {"root": str(tmp_path / "missing")})
    manifest, _ = attach(root, [case])
    public = identity.public_manifest(manifest)
    with pytest.raises(ValueError, match="Missing local asset binding: dataset:root:"):
        identity.restore_paths(public["cases"], {})


@pytest.mark.parametrize("equivalent", [False, True])
def test_formula_override_inherits_certification_only_for_identical_bytes(tmp_path, equivalent):
    root, data_root = tmp_path / "checkout", tmp_path / "external"
    root.mkdir()
    baseline = write(root / "kd/dataset/data/benchmarks.csv", b"certified formula")
    formula = write(data_root / "custom.csv", baseline.read_bytes() if equivalent else b"other formula")
    write(data_root / "function_sets.csv", b"grammar")
    case = make_case("Nguyen-1", ["data/benchmarks.csv", "data/function_sets.csv"],
                     {"root": str(data_root), "benchmark_source": str(formula)}, source="symbolic")
    case.update(profile="full", ground_truth="x1 + x1**2", benchmark_role="main")
    case["dataset_metadata"].update(
        source_status="packaged_definition_verified", reference_status="formula_declared",
        source_evidence={"asset_sha256": {"data/benchmarks.csv": identity.file_digest(baseline)}},
    )
    manifest, bindings = attach(root, [case])
    effective = manifest["cases"][0]
    policy = {"timeout": 60, "soft_deadline": True}
    if equivalent:
        validate_main_matrix([effective], policy)
        assert effective["dataset_metadata"]["source_status"] == "packaged_definition_verified"
        assert recovery_eligibility(effective)
    else:
        with pytest.raises(ValueError, match="source provenance has not been verified"):
            validate_main_matrix([effective], policy)
        assert effective["dataset_metadata"]["reference_status"] == "unknown"
        assert effective["ground_truth"] is None
        assert not recovery_eligibility(effective)
    restored = identity.restore_paths(identity.public_manifest(manifest)["cases"], bindings)
    identity.verify(manifest["identity"], root, restored)
