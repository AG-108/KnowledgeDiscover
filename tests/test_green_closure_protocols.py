"""A5/A6 independent numeric, leakage, source and execution acceptance."""

import copy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import run_benchmark as b
from kd.dataset._closure import PYQG_PATHS, load_pyqg, resolved_features, spectral_derivative
from kd.dataset._file_sr import FILE_SR_NAMES, PREFIX, load_file_sr
from kd.dataset._green import (
    GREEN_SPECS,
    GreenOperatorDataset,
    load_green,
    reference_kernel,
    trapezoid_weights,
)
from kd.evaluation.closure import problems as closure_problems
from kd.evaluation.green import forward_metrics
from kd.evaluation.green import problems as green_problems
from kd.model.kd_green_basis import GreenBasis

ROOT = Path(__file__).resolve().parents[1]


def green_fixture(nxi=29):
    x = np.array([0.02, 0.15, 0.37, 0.68, 0.97])
    xi = np.linspace(0, 1, nxi) ** 1.4
    rng = np.random.default_rng(419)
    # Constant forcings: trapezoid exact for affine xi integrands.
    F = np.broadcast_to(rng.normal(size=(1, 15)), (nxi, 15)).copy()
    K = 2 * x[:, None] + 3 * xi[None, :] + 4 * x[:, None] * xi[None, :]
    hom = 1 + x
    U = (4 * x[:, None] + 1.5) * F[:1] + hom[:, None]
    return GreenOperatorDataset("analytic", x, xi, F, U, hom), K


def case(model="green_basis", name="green_laplace"):
    catalog, _ = b.dataset_catalog()
    return b.build_experiments(catalog, models=[model], datasets=[name], seeds=[0])[0]


def test_nonuniform_orientation_and_homogeneous_exact_integral():
    d, K = green_fixture()
    assert K.shape[0] != K.shape[1]
    np.testing.assert_allclose(d.forward(K, block_rows=2), d.U, rtol=1e-10, atol=1e-12)
    assert not np.allclose(K[:5, :5], K[:5, :5].T)
    with pytest.raises(ValueError, match="orientation"):
        d.forward(K.T)


def test_trapezoid_converges_against_independent_integral():
    errors = []
    for n in (21, 41, 81):
        xi = np.linspace(0, 1, n) ** 1.2
        approx = trapezoid_weights(xi) @ np.exp(xi)
        errors.append(abs(approx - (np.e - 1)))
    assert errors[1] < errors[0] / 3.5
    assert errors[2] < errors[1] / 3.5


@pytest.mark.parametrize("axis", [[0, 0, 1], [1, 0], [0, np.nan], [[0], [1]]])
def test_invalid_quadrature_axes_rejected(axis):
    with pytest.raises(ValueError):
        trapezoid_weights(axis)


@pytest.mark.parametrize(
    "expression", ["__import__('os')", "np.load('secret')", "x.__class__", "[x for x in y]"]
)
def test_reference_parser_rejects_code(expression):
    with pytest.raises(ValueError):
        reference_kernel(expression, np.arange(2), np.arange(3))


def test_restricted_branch_reference_evaluation():
    x, xi = np.array([0.2, 0.8]), np.array([0.1, 0.5, 0.9])
    K = reference_kernel("x*(1-y)*(x<=y) + y*(1-x)*(x>y)", x, xi)
    expected = np.minimum(x[:, None], xi[None, :]) * (1 - np.maximum(x[:, None], xi[None, :]))
    np.testing.assert_allclose(K, expected)


def test_finite_basis_identifiable_response_only_recovery():
    d, _ = green_fixture(47)
    rng = np.random.default_rng(512)
    d.F = rng.normal(size=d.F.shape)
    model = GreenBasis()
    design = model.basis(d.kernel_inputs)
    coef = np.array([0.4, 3, -0.2, 2, 0.9, -0.3, 0.7])
    K = (design @ coef).reshape(len(d.x), len(d.xi))
    d.U = d.forward(K)
    model.fit(d.training_view(d.split()))
    assert model.diagnostics_["finite_basis_identifiable"]
    assert np.linalg.norm(model.coef_ - coef) / np.linalg.norm(coef) < 0.01
    np.testing.assert_allclose(model.kernel(d.kernel_inputs), K, atol=1e-9)


def test_rank_deficient_fit_never_certifies_unique_kernel():
    d, _ = green_fixture()
    model = GreenBasis().fit(d.training_view(d.split()))
    assert not model.diagnostics_["finite_basis_identifiable"]
    assert not model.diagnostics_["nonparametric_kernel_unique"]


def test_test_and_reference_perturbations_do_not_change_training():
    d, _ = green_fixture()
    split = d.split()
    before = d.training_view(split)
    d.reference = "forbidden executable source"
    d.U[:, split["test"]] += 1e8
    d.F[:, split["test"]] -= 1e6
    after = d.training_view(split)
    for key in before:
        np.testing.assert_array_equal(before[key], after[key])
    assert "reference" not in before
    assert not (
        set(split["train"]) & set(split["validation"]) | set(split["train"]) & set(split["test"])
    )


def test_invalid_kernel_has_terminal_error_and_zero_coverage():
    d, K = green_fixture()
    K[0, 0] = np.nan
    result = forward_metrics(d, K, [0, 1], 1.0)
    assert result["status"] == "error" and result["prediction_finite_coverage"] == 0


def test_green_task_loss_and_earlystop_ignore_test_reference():
    from kd.model.dso.task.green import GreenTask

    d, K = green_fixture()
    split = d.split()

    class Candidate:
        invalid = False

        def execute(self, X):
            return K.ravel()

    p = Candidate()
    task = GreenTask(d.training_view(split), ["add", "sub", "mul", "abs", "lt"])
    reward, score = task.reward_function(p), task.evaluate(p)
    assert reward == pytest.approx(1.0)
    d.reference = "bad"
    d.U[:, split["test"]] += 1e9
    d.F[:, split["test"]] -= 1e9
    changed = GreenTask(d.training_view(split), ["add", "sub", "mul", "abs", "lt"])
    assert reward == changed.reward_function(p)
    assert score == changed.evaluate(p)
    assert "lt" in task.library.names


def test_real_dso_search_development_and_ordinary_task_isolation():
    from kd.model.dso.program import Program
    from kd.model.kd_dso import KD_DSO
    from kd.model.kd_green import GreenDSO

    d, _ = green_fixture()
    model = GreenDSO(
        n_samples=30,
        batch_size=10,
        max_length=10,
        function_set=["add", "sub", "mul", "abs"],
        device="cpu",
        seed=7,
        verbose=False,
    )
    model.fit(d.training_view(d.split(7)))
    before = model.kernel(d.kernel_inputs)
    assert np.isfinite(before).all()
    assert model.task_.task_type == "green"
    ordinary = KD_DSO(
        n_samples=20,
        batch_size=10,
        max_length=8,
        function_set=["add", "mul"],
        device="cpu",
        seed=2,
        verbose=False,
    )
    X = np.linspace(0.1, 1, 20)[:, None]
    ordinary.fit(X, 2 * X[:, 0])
    assert Program.task.task_type == "regression"
    np.testing.assert_array_equal(before, model.kernel(d.kernel_inputs))
    assert Program.task.task_type == "regression"
    assert np.isfinite(ordinary.predict(X)).all()


@pytest.mark.external_data("kd/dataset/Discover_Green_function/dso/task/pde/data/*.mat")
@pytest.mark.parametrize("name", list(GREEN_SPECS))
def test_real_green_asset_layout_and_variant_split(name):
    d = load_green(name)
    assert d.F.shape == (200, 100) and d.U.shape == (100, 100)
    assert d.weights.sum() == pytest.approx(d.xi[-1] - d.xi[0])
    assert sum(map(len, (d.split()["train"], d.split()["validation"], d.split()["test"]))) == 100
    assert d.split()["test"] == load_green("green_laplace").split()["test"]
    assert len(list(green_problems(d, case(name=name)))) == 1


def test_periodic_derivatives_and_layer_axes_analytic():
    x = np.arange(48) * 2 * np.pi / 48
    y = np.arange(32) * 2 * np.pi / 32
    Y, X = np.meshgrid(y, x, indexing="ij")
    field = np.stack([np.stack([np.sin(2 * X) + np.cos(3 * Y), 7 * np.sin(X)])] * 3)
    fields = {name: field for name in ("q", "streamfunction", "u", "v")}
    run = {"fields": fields, "x": x, "y": y, "periodic": True}
    values, names = resolved_features(run)
    np.testing.assert_allclose(
        values[..., names.index("q_x_L1")],
        np.broadcast_to(2 * np.cos(2 * X), (3, 32, 48)),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        values[..., names.index("q_y_L1")],
        np.broadcast_to(-3 * np.sin(3 * Y), (3, 32, 48)),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        values[..., names.index("q_x_L2")], np.broadcast_to(7 * np.cos(X), (3, 32, 48)), atol=1e-12
    )
    with pytest.raises(ValueError, match="periodic"):
        spectral_derivative(field, -1, x, periodic=False)


@pytest.mark.external_data(*["kd/dataset/" + p for p in PYQG_PATHS])
def test_real_pyqg_layout_and_leakage_and_balanced_sampling():
    d = load_pyqg()
    c = case("gplearn", "pyqg_subgrid")
    before = list(closure_problems(d, c))
    assert [p["target"] for p in before] == ["q_subgrid_forcing_L1", "q_subgrid_forcing_L2"]
    assert d.runs[0]["fields"]["q"].shape == (87, 2, 64, 64)
    assert np.diff(d.runs[0]["time"])[0] == 3600000
    d.runs[0]["fields"]["ufull"] = np.zeros((87, 2, 64, 64)) + 1e6
    saved_target = d.runs[0]["fields"]["q_subgrid_forcing"].copy()
    d.runs[0]["fields"]["q_subgrid_forcing"] += 0.01
    base, _ = resolved_features(d.runs[0])
    d.runs[0]["fields"]["q_subgrid_forcing"] = saved_target
    original, _ = resolved_features(d.runs[0])
    np.testing.assert_array_equal(base, original)
    d.runs[1]["fields"]["q"] += 1e5
    d.runs[1]["fields"]["q_subgrid_forcing"] += 1e8
    after = list(closure_problems(d, c))
    for a, z in zip(before, after):
        np.testing.assert_array_equal(a["X_train"], z["X_train"])
        np.testing.assert_array_equal(a["y_train"], z["y_train"])
        assert a["metadata"]["normalization"]["x_mean"] == z["metadata"]["normalization"]["x_mean"]
        sample = a["metadata"]["sampling"]
        assert sample["validation_frames"][0] - sample["train_frames"][-1] == 3
        assert len(sample["train_grid_indices_yx"]) * len(sample["train_frames"]) == len(
            a["X_train"]
        )
        assert sample["train_sample_weight"] * len(a["X_train"]) == pytest.approx(1.0)


@pytest.mark.external_data("kd/dataset/DeepSymbolicOptimization/dso/dso/task/regression/data/*.csv")
@pytest.mark.parametrize("label", FILE_SR_NAMES)
def test_all_25_official_file_candidates(label):
    d = load_file_sr("dso_file_" + label)
    assert len(d.X_train) > 0 and len(d.X_test) > 0
    assert d.X_train.shape[1] == d.X_test.shape[1]
    assert not hasattr(d, "sym_true")
    c = case("gplearn", "dso_file_" + label)
    p = next(b.regression_problems(d, c))
    np.testing.assert_array_equal(p["X_train"], d.X_train)
    np.testing.assert_array_equal(p["X_test"], d.X_test)
    assert p["ground_truth"] is None
    if label == "PiecewiseFunction-1":
        assert (
            d.X_train[:, 0].min()
            < d.X_test[:, 0].min()
            < d.X_test[:, 0].max()
            < d.X_train[:, 0].max()
        )


def test_missing_and_invalid_file_sr_are_errors(tmp_path):
    path = tmp_path / PREFIX
    path.mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        load_file_sr("dso_file_Custom-1", root=tmp_path)
    (path / "Custom-1.csv").write_text("header,value\n1,nonnumeric\n")
    with pytest.raises(ValueError):
        load_file_sr("dso_file_Custom-1", root=tmp_path)


def test_missing_green_and_pyqg_preserve_target_denominator(tmp_path):
    c = case()
    c["dataset_params"] = {"root": str(tmp_path)}
    assert [x["target"] for x in b._prepare_targets(c)] == ["kernel"]
    c = case("gplearn", "pyqg_subgrid")
    c["dataset_params"] = {"root": str(tmp_path)}
    specs = b._prepare_targets(c)
    assert [s["target"] for s in specs] == ["q_subgrid_forcing_L1", "q_subgrid_forcing_L2"]


def test_public_star_import_no_external_io_or_model_load():
    code = """import builtins
from pathlib import Path
original_open=builtins.open
original_path_open=Path.open
def guard(file,*a,**k):
    normalized=str(file).replace(chr(92),'/')
    if 'kd/dataset/' in normalized and Path(normalized).suffix in {'.csv','.mat','.npy','.nc','.pkl','.xls','.xlsx'}:
        raise AssertionError('external data read during import')
    return original_open(file,*a,**k)
builtins.open=guard
def path_guard(path,*a,**k):
    guard(path,*a,**k).close() if 'kd/dataset/' in str(path).replace(chr(92),'/') and path.suffix in {'.csv','.mat','.npy','.nc','.pkl','.xls','.xlsx'} else None
    return original_path_open(path,*a,**k)
Path.open=path_guard
import kd.dataset as d
assert all(hasattr(d,n) for n in d.__all__)
def denied(*a,**k): raise AssertionError('data read')
builtins.open=denied
exec('from kd.dataset import *')
import sys
assert 'kd.model.kd_dso' not in sys.modules
assert 'xarray' not in sys.modules
"""
    subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, check=True)


def test_new_protocol_isolation_and_default_queue_preserved():
    c = case()
    modern = b._base_row(c)
    assert modern["metric_protocol_version"] == "2.3"
    assert modern["green_protocol_version"] == "green-v1"
    catalog, _ = b.dataset_catalog()
    cases = b.build_experiments(catalog, profile="full")
    assert len(cases) == 3311
    assert len([c for c in cases if c["model"] != "sisso"]) == 3024
    assert all(not catalog[c["dataset"]].get("opt_in") for c in cases)
    assert [c["model"] for c in b.build_experiments(catalog, datasets=["green_laplace"])] == [
        "green_dso",
        "green_basis",
    ]
    modern.update(status="ok", stop_reason="budget_exhausted")
    other = copy.deepcopy(modern)
    other["metric_protocol_version"] = "2.1"
    assert {r["metric_protocol_version"] for r in b.aggregate_results([modern, other])} == {
        "2.1",
        "2.3",
    }


def test_asset_manifest_rebuild_hash_alias_and_missing_checkout(tmp_path):
    from scripts.export_asset_manifest import asset_manifest

    root = tmp_path / "source"
    root.mkdir()
    (root / "a.csv").write_text("1,2\n3,4\n")
    (root / "b.csv").write_bytes((root / "a.csv").read_bytes())
    original = asset_manifest(root=root)
    assert len(original["assets"]) == 2
    assert original["assets"][0]["byte_aliases"] == ["b.csv"]
    assert original["assets"][0]["license_status"] == "unknown"
    empty = tmp_path / "empty"
    empty.mkdir()
    rebuilt = asset_manifest(root=empty, expected=original)
    assert rebuilt["counts"] == {"local_missing": 2}
    assert all(row["sha256"] for row in rebuilt["assets"])
    assert not any(str(tmp_path) in str(row) for row in rebuilt["assets"])


def test_training_exposure_annotations_do_not_claim_checkpoint_overlap():
    from kd.dataset._overlap import training_overlap

    for name in ("allen_cahn", "convection_diffusion", "burgers2d", "wave", "klein_gordon"):
        row = training_overlap("eqgpt", name)
        assert row["status"] == "confirmed"
        assert row["observation_overlap"] == "unknown"
        assert row["heldout_equation_exclusion"] is False
    assert training_overlap("e2e", "dso_file_Custom-1")["status"] == "unknown"
    assert (
        training_overlap("symbolicgpt", "dso_file_Custom-1")["real_observation_use_in_pretraining"]
        is False
    )


@pytest.mark.parametrize(
    "name,model,targets",
    [
        ("green_laplace", "green_basis", ["kernel"]),
        ("pyqg_subgrid", "gplearn", ["q_subgrid_forcing_L1", "q_subgrid_forcing_L2"]),
    ],
)
def test_new_tracks_timeout_error_and_resume_terminal(monkeypatch, tmp_path, name, model, targets):
    c = case(model, name)
    specs = b._prepare_targets(c)
    calls = []

    def launch(case_path, attempt, policy, spec=None):
        if spec is None:
            b.write_json(attempt / "targets.json", specs)
            return {"returncode": 0}
        calls.append(spec["target"])
        row = b._base_row(c, spec["instance"], spec["target"])
        row.update(
            status="timeout" if spec["target_index"] == 0 else "error",
            error="fixture terminal",
            target_id=spec["target_id"],
            execution_policy=policy,
            attempt=int(attempt.name),
            stop_reason="hard_timeout" if spec["target_index"] == 0 else "error",
            stop_reason_source="supervisor",
        )
        b.write_json(attempt / "result.json", row)
        return {"returncode": 0}

    monkeypatch.setattr(b, "_launch_worker", launch)
    b.execute_case(c, tmp_path, timeout=1)
    assert calls == targets
    b.execute_case(c, tmp_path, timeout=1, resume=True)
    assert calls == targets
