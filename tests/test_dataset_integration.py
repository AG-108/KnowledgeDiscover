"""Regression cases from the 2026-10-11 data integration review."""

import shutil
from types import SimpleNamespace

import numpy as np
import pytest

import run_benchmark as b
from kd import benchmark_identity as identity
from kd.benchmark_capabilities import PDE_CAPABILITIES, validate_pde
from kd.benchmark_execution import SkipCase
from kd.benchmark_protocol import validate_main_matrix
from kd.dataset import EquationTarget, GridPDEDataset
from kd.dataset._advanced import declared_targets
from kd.dataset._metadata import pending_datasets

LEGACY = tuple(k for k in PDE_CAPABILITIES
               if k not in {"pdefind_nd", "implicit_pde", "pdenet", "reference"})


def case(model="pdefind", dataset="wave", seeds=(0,)):
    catalog, _ = b.dataset_catalog()
    return b.build_experiments(catalog, models=[model], datasets=[dataset], seeds=seeds)[0]


def grid(name="heat"):
    x, t = np.linspace(0, 1, 21), np.linspace(0, 1, 21)
    value = 1 + x[:, None] + t[None, :]
    data = GridPDEDataset(name, None, None, 0, x=x, t=t, usol=value, legacy=True)
    data.equation_targets = declared_targets(name)
    return data


@pytest.mark.parametrize("model", LEGACY)
@pytest.mark.parametrize("dataset", ["wave", "klein_gordon", "implicit_mixed"])
def test_wrong_pde_target_is_rejected_in_plan_preflight_and_execution(monkeypatch, model, dataset):
    data, c = grid(dataset), case(model, dataset)
    monkeypatch.setattr(b, "_load_instances", lambda _: iter([(dataset, data)]))
    with pytest.raises(SkipCase, match="method_unsupported"):
        b.prepare_pde(data, model)
    with pytest.raises(SkipCase, match="method_unsupported"):
        b.run_pde(data, c)
    specs = b._prepare_targets(c)
    assert [s["target"] for s in specs] == ["PDE"]
    assert specs[0]["data_metadata"]["target"] == data.equation_targets[0].to_dict()
    assert all(s["preparation_row"]["status"] == "skipped" for s in specs)
    assert all(s["preparation_row"]["capability_status"] == "method_unsupported" for s in specs)
    report = b.check_compatibility([c], mode="full", check_runtime=False)
    assert report["scope"]["enumeration_complete"]
    assert report["pairs"][0]["capability_status"] == "method_unsupported"
    assert report["pairs"][0]["target"] == "PDE"
    assert report["pairs"][0]["equation_targets"] == specs[0]["equation_targets"]


@pytest.mark.parametrize("model", LEGACY)
def test_first_order_legacy_targets_still_supported(model):
    assert b.prepare_pde(grid(), model).usol.shape == (21, 21)


@pytest.mark.parametrize("dataset,model", [
    ("wave", "pdefind_nd"), ("klein_gordon", "pdefind_nd"),
    ("implicit_mixed", "implicit_pde"),
])
def test_correct_target_routes_remain_eligible(monkeypatch, dataset, model):
    data, c = grid(dataset), case(model, dataset)
    validate_pde(data, model)
    monkeypatch.setattr(b, "_load_instances", lambda _: iter([(dataset, data)]))
    specs = b._prepare_targets(c)
    assert "preparation_row" not in specs[0]
    assert specs[0]["target"] == data.equation_targets[0].key


def test_full_preflight_checks_later_instances_every_seed_and_target(monkeypatch, tmp_path):
    catalog, _ = b.dataset_catalog()
    cases = b.build_experiments(catalog, models=["gplearn"], datasets=["Nguyen-1"], seeds=[0, 1])
    def prepare(dataset, c):
        if c["seed"] == 0 and dataset is first[0]:
            raise SkipCase("Training response is constant")
        return [{"target": key} for key in ("a", "b")]

    first = []

    def instances(_):
        first[:] = [SimpleNamespace(target_name="response")]
        return iter([("first", first[0]), ("second", SimpleNamespace(target_name="response"))])

    monkeypatch.setattr(b, "_load_instances", instances)
    monkeypatch.setattr(b, "_case_problems", prepare)
    report = b.check_compatibility(cases, mode="full", check_runtime=False)
    assert len(report["dataset_loads"]) == 4
    assert len(report["pairs"]) == 7
    ineligible = [p for p in report["pairs"] if p["instance_eligibility"] == "ineligible"]
    assert [(p["seed"], p["instance"]) for p in ineligible] == [(0, "first")]
    assert ineligible[0]["preparation_status"] == "instance_ineligible"
    assert report["scope"]["enumeration_complete"]
    b.save_compatibility_report(tmp_path, report)
    assert "preparation_status" in (tmp_path / "compatibility_pairs.csv").read_text(encoding="utf-8-sig")
    sample = b.check_compatibility(cases, mode="sample", check_runtime=False)
    assert len(sample["dataset_loads"]) == 2
    assert not sample["scope"]["enumeration_complete"]


def test_runtime_failure_does_not_hide_method_mismatch(monkeypatch):
    import kd.benchmark_compatibility as compatibility

    c = case()
    monkeypatch.setattr(b, "_load_instances", lambda _: iter([("wave", grid("wave"))]))
    monkeypatch.setattr(compatibility, "_runtime",
                        lambda *args: {"status": "error", "error": "native unavailable"})
    row = b.check_compatibility([c], mode="full")["pairs"][0]
    assert row["status"] == "incompatible"
    assert row["capability_status"] == "method_unsupported"
    assert row["runtime_status"] == "error"


@pytest.mark.parametrize("changed", [
    "kd/dataset/data/function_sets.csv",
    "kd/dataset/Discovery_of_soild_consititutive/data/data_rubber/test/test.xlsx",
    "examples/_common.py",
    "kd/model/eqgpt/data/PDE_dataset_0725.xlsx",
    "kd/model/eqgpt/data/dict_datas_0725.json",
])
def test_actual_dependencies_reject_tampering_and_allow_checkout_move(tmp_path, monkeypatch, changed):
    root = tmp_path / "checkout"
    paths = [
        "run_benchmark.py", "examples/_common.py", "kd/dataset/data/benchmarks.csv",
        "kd/dataset/data/function_sets.csv",
        "kd/dataset/Discovery_of_soild_consititutive/data/data_rubber/train/train.xlsx",
        "kd/dataset/Discovery_of_soild_consititutive/data/data_rubber/test/test.xlsx",
        "kd/model/eqgpt/data/PDE_dataset_0725.xlsx",
        "kd/model/eqgpt/data/dict_datas_0725.json",
    ]
    for relative in paths:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("original", encoding="utf-8")
    for relative in ("data/benchmarks.csv", "data/function_sets.csv"):
        shutil.copyfile(b.ROOT / "kd/dataset" / relative, root / "kd/dataset" / relative)
    monkeypatch.setattr(identity, "environment_fingerprint", lambda: {"test": True})
    cases = [case("gplearn", "Nguyen-1"), case("gplearn", "rubber_train"), case("eqgpt", "heat")]
    manifest = {"cases": cases, "execution_policy": b.execution_policy(5)}
    identity.attach(manifest, root)
    moved = tmp_path / "moved"
    shutil.copytree(root, moved)
    identity.verify(manifest["identity"], moved, manifest["cases"])
    (moved / changed).write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        identity.verify(manifest["identity"], moved, manifest["cases"])


@pytest.mark.parametrize("exported", ["u_t = 1.0", None])
def test_ordinary_sga_scores_weighted_export_and_never_display(monkeypatch, exported):
    class Model:
        best_pde_ = "( ^3 u)"
        best_score_ = 0.0
        best_expression_ = exported
        equation_export_status_ = "weighted_native_equation" if exported else "abstain: fixture"

        def fit_dataset(self, dataset, **kwargs):
            assert len(dataset.t) < 21

    monkeypatch.setattr(b, "_new_model", lambda *args: Model())
    row = b.run_pde(grid(), case("sga", "heat"))
    assert row["native_expression_display"] == "( ^3 u)"
    assert row["expression"] == exported
    if exported:
        assert row["equation_residual_mse"] < 1e-20
    else:
        assert row["status"] == "error"
        assert row["equation_residual_status"] == "abstain_native_export"
        assert row.get("equation_residual_mse") is None


def test_pdenet_fit_keeps_verified_grid_contract(monkeypatch):
    from kd.evaluation.equation import periodic_grid_options

    x, y, t = np.arange(8) / 8, np.arange(10) / 5, np.arange(21) / 20
    values = np.ones((1, len(x), len(y), len(t)))
    data = GridPDEDataset("analytic", None, None, 0, coords={"x": x, "y": y, "t": t},
                          usol=values)
    data.state_vars = ["u"]
    data.protocol_metadata = {"boundary": {"kind": "periodic", "verified": True,
                                           "endpoint": "excluded", "evidence": "analytic fixture"}}

    class Model:
        def fit(self, training):
            assert training.equation_name == "analytic"
            assert training.state_vars == ["u"]
            assert len(training.t) < len(t)
            np.testing.assert_allclose(periodic_grid_options(training)["dx"], [1 / 8, 1 / 5])

        def print_model(self):
            print("u_t = 0")

    monkeypatch.setattr(b, "_new_model", lambda *args: Model())
    row = b.run_pde(data, case("pdenet", "heat"))
    assert row["expression"] == "u_t = 0"


def test_failure_and_preflight_keep_execution_target_and_recovery_denominator(monkeypatch):
    c, data = case("pdefind", "kdv"), grid("kdv")
    monkeypatch.setattr(b, "_load_instances", lambda _: iter([("kdv", data)]))
    spec = b._prepare_targets(c)[0]
    row = b.check_compatibility([c], mode="full", check_runtime=False)["pairs"][0]
    assert spec["target"] == row["target"] == "PDE"

    def missing(_):
        raise FileNotFoundError("fixture")
        yield

    monkeypatch.setattr(b, "_load_instances", missing)
    failed = b._prepare_targets(c)[0]
    assert failed["target"] == spec["target"]
    assert failed["ground_truth"] == spec["ground_truth"] == c["ground_truth"]
    assert failed["recovery_eligible"] == spec["recovery_eligible"] is True


def test_function_set_dependency_preserves_formal_sr_admission():
    catalog, _ = b.dataset_catalog()
    c = b.build_experiments(catalog, models=["gplearn"], datasets=["Nguyen-1"], profile="full")[0]
    metadata = c["dataset_metadata"]
    assert metadata["source_status"] == "packaged_definition_verified"
    assert set(metadata["source_evidence"]["asset_sha256"]) == {
        "data/benchmarks.csv", "data/function_sets.csv"}
    assert validate_main_matrix([c], b.execution_policy(5))["n_cases"] == 1


@pytest.mark.parametrize("model", ["sindy", "pysindy", "pdefind_nd"])
def test_preflight_detects_model_dependent_insufficient_support(monkeypatch, model):
    data = grid()
    if model == "pdefind_nd":
        data = GridPDEDataset("heat", None, None, 0, x=data.x[:4], t=data.t,
                              usol=data.usol[:4], legacy=True)
    else:
        data = b._slice_pde_time(data, 0, 5)
    monkeypatch.setattr(b, "_load_instances", lambda _: iter([("heat", data)]))
    row = b.check_compatibility([case(model, "heat")], mode="full", check_runtime=False)["pairs"][0]
    assert row["status"] == "incompatible"
    assert row["instance_eligibility"] == "ineligible"
    assert row["preparation_status"] in {"insufficient_support", "instance_ineligible"}


def test_ordinary_rhs_coordinates_and_nonfinite_candidate_have_residual_status():
    data = grid()
    data.usol = 1 + data.x[:, None] * data.t[None, :]
    row = b.pde_equation_residual("u_t = x", data)
    assert row["equation_residual_status"] == "ok"
    assert row["equation_residual_mse"] < 1e-20
    invalid = b.pde_equation_residual("(ux / 0)+(x*x)", data)
    assert invalid["equation_residual_status"] == "nonfinite_candidate"
    assert invalid.get("equation_residual_mse") is None


def test_eleven_frame_equation_view_keeps_three_test_frames_and_support_gap(monkeypatch):
    from kd.evaluation.equation import _temporal_views

    data, c = b._slice_pde_time(grid(), 0, 11), case("pdefind_nd", "heat")
    train, test, split = _temporal_views(data, c)
    assert len(train.t) == 6 and len(test.t) == 3
    assert split["test_frame_start"] - split["train_frame_stop"] == 2
    assert not np.intersect1d(train.t, test.t).size
    monkeypatch.setattr(b, "_load_instances", lambda _: iter([("heat", data)]))
    row = b.check_compatibility([c], mode="full", check_runtime=False)["pairs"][0]
    assert row["status"] == "compatible"


def test_pdenet_rejects_auxiliary_pressure_evolution():
    axis, t = np.arange(8) / 8, np.arange(21) / 20
    data = GridPDEDataset("analytic", None, None, 0, coords={"x": axis, "y": axis, "t": t},
                          usol=np.ones((3, 8, 8, 21)))
    data.state_vars = ["u", "v", "p"]
    data.equation_targets = [EquationTarget("u_t", "u"), EquationTarget("v_t", "v")]
    data.protocol_metadata = {"boundary": {"kind": "periodic", "verified": True,
                                           "endpoint": "excluded"}}
    with pytest.raises(SkipCase, match="auxiliary fields"):
        validate_pde(data, "pdenet")


def test_pending_observations_are_tracked_without_new_fit_weight(tmp_path):
    from scripts.export_asset_manifest import _role, asset_manifest

    cards = {row["dataset"]: row for row in pending_datasets()}
    assert cards["diffusion_raw"]["source_paths"] == ["data/Diffusion_equation.txt"]
    assert cards["dso_fixed_Nguyen-1"]["aliases"] == ["dso_file_Test-1"]
    assert sum(name.startswith("dso_fixed_") for name in cards) == 20
    catalog, _ = b.dataset_catalog()
    assert "diffusion_raw" not in catalog and "dso_fixed_Nguyen-1" not in catalog
    path = tmp_path / "data/Diffusion_equation.txt"
    path.parent.mkdir()
    path.write_text("1 2 3", encoding="utf-8")
    row = asset_manifest(root=tmp_path)["assets"][0]
    assert row["role"] == "observation_or_definition"
    assert row["datasets"] == ["diffusion_raw"]
    assert row["integration"][0]["status"] == "not_integrated"
    assert _role("EqGPT/results_save/awards.csv") == "upstream_result"
    assert _role("DeepSymbolicOptimization/dso/dso/task/control/data/Pendulum-v0/model-td3.pkl") == "training_asset"
    assert _role("data/function_sets.csv") == "grammar_definition"


@pytest.mark.parametrize("model,dataset", [("pdefind", "heat"), ("pdefind_nd", "wave")])
def test_row_inherits_declared_metric_protocol_and_legacy_case_uses_same_fallback(model, dataset):
    declared = case(model, dataset)
    expected = declared["metric_protocol_version"]
    legacy = dict(declared)
    legacy.pop("metric_protocol_version")
    assert b._base_row(legacy)["metric_protocol_version"] == expected
    declared["metric_protocol_version"] = "explicit-test-protocol"
    assert b._base_row(declared)["metric_protocol_version"] == "explicit-test-protocol"
