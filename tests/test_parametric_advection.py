"""Source layouts, variable-coefficient semantics and independent discovery checks."""

import copy
import hashlib

import numpy as np
import pytest
from scipy.io import loadmat, savemat

import run_benchmark as b
from kd.benchmark_capabilities import PDE_CAPABILITIES, validate_pde
from kd.benchmark_execution import SkipCase
from kd.dataset import EquationTarget, GridPDEDataset, _advanced, load_dataset, load_pde_grid
from kd.dataset._derivatives import namespace
from kd.dataset._metadata import pending_datasets
from kd.evaluation.equation import _temporal_views, prepare_views
from kd.model.kd_equation import SparseEquationModel, library

NAMES = ("parametric_burgers", "parametric_convection_diffusion", "advection_diffusion")


def configured_cases():
    settings, _ = b.load_benchmark_config("configs/benchmark/parametric_advection.json")
    catalog, _ = b.dataset_catalog()
    b.validate_benchmark_config(settings, catalog)
    return b.build_experiments(
        catalog, models=settings["models"], datasets=settings["datasets"],
        seeds=settings["seeds"], profile=settings["profile"], config=settings["overrides"],
    )


@pytest.mark.parametrize("name,nx,nt,dx,dt", [
    (NAMES[0], 251, 200, .004, .005),
    (NAMES[1], 201, 250, .04, .02),
])
def test_parametric_layout_initial_boundary_and_reference_residual(name, nx, nt, dx, dt):
    data = load_dataset(name)
    assert data.usol.shape == (1, nx, nt)
    np.testing.assert_array_equal(data.x, np.arange(nx) * dx)
    np.testing.assert_array_equal(data.t, np.arange(nt) * dt)
    raw = np.load(_advanced.ROOT / _advanced.EVOLUTION_SPECS[name]["path"], allow_pickle=False)
    np.testing.assert_array_equal(data.usol[0], raw.T)
    initial = ((1.5 - data.x) * np.sin(2 * np.pi * data.x) if name == NAMES[0]
               else (8 - data.x) * np.sin(data.x))
    np.testing.assert_allclose(data.usol[0, :, 0], initial, atol=1e-14)
    np.testing.assert_array_equal(data.usol[0, [0, -1]], 0)
    ns, _ = namespace(data, temporal_order=1)
    rhs = (-ns["u"] * ns["u_x"] + .02 * np.exp(ns["t"]) * ns["u_xx"] if name == NAMES[0]
           else -(1 + .25 * np.sin(ns["x"])) * ns["u_x"] + ns["u_xx"])
    valid = np.isfinite(rhs) & np.isfinite(ns["u_t"])
    assert np.linalg.norm((ns["u_t"] - rhs)[valid]) / np.linalg.norm(ns["u_t"][valid]) < .02
    assert data.protocol_metadata["boundary"]["kind"] == "dirichlet"
    assert data.equation_targets == [EquationTarget("u_t")]


def test_expression1_has_two_spatial_axes_and_no_inferred_reference_or_periodicity():
    data = load_dataset(NAMES[2])
    packed = loadmat(_advanced.ROOT / "data/Advection_diffusion.mat")["Expression1"]
    packed = packed.reshape(51, 51, 61, 4)
    assert data.usol.shape == (1, 51, 51, 61)
    assert data.spatial_vars == ["x", "y"]
    np.testing.assert_array_equal(data.usol[0], packed[..., 3])
    np.testing.assert_allclose(data.t, np.linspace(3, 9, 61))
    for axis in ("x", "y"):
        np.testing.assert_allclose(data.coords_spatial[axis], np.linspace(-5, 5, 51))
    assert data.protocol_metadata["source_reference"] is None
    assert "unknown" in data.protocol_metadata["boundary"]
    assert b.prepare_pde(data, "pdefind_nd") is data
    with pytest.raises(SkipCase, match="method_unsupported"):
        b.prepare_pde(data, "deepmod")
    with pytest.raises(SkipCase, match="boundary"):
        b.prepare_pde(data, "pdenet")


def test_registered_metadata_paths_and_compatibility_loader():
    catalog, excluded = b.dataset_catalog()
    pending = {row["dataset"] for row in pending_datasets()}
    for name in NAMES:
        assert name in catalog and name not in excluded and name not in pending
        card = catalog[name]
        assert card["source_paths"] == [_advanced.EVOLUTION_SPECS[name]["path"]]
        assert card["source_evidence"]["verification_date"] == "2026-10-16"
        np.testing.assert_array_equal(load_pde_grid(name).usol, load_dataset(name).usol)
    assert "upstream_bytes_unverified" in catalog[NAMES[0]]["source_status"]
    assert all(catalog[n]["source_status"] == "upstream_byte_verified" for n in NAMES[1:])


def test_changed_asset_is_rejected_before_assigning_paper_coordinates(tmp_path, monkeypatch):
    spec = _advanced.EVOLUTION_SPECS[NAMES[0]]
    path = tmp_path / spec["path"]
    path.parent.mkdir()
    np.save(path, np.zeros(spec["shape"]))
    monkeypatch.setattr(_advanced, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="source_blocked.*SHA256"):
        load_dataset(NAMES[0])


def test_expression1_rejects_noncartesian_coordinate_channels(tmp_path, monkeypatch):
    mesh = np.meshgrid(np.arange(5.), np.arange(6.), np.arange(7.), indexing="ij")
    packed = np.stack([*mesh, np.ones((5, 6, 7))], axis=-1)
    packed[2, 2, 2, 0] += .5
    path = tmp_path / "data/Advection_diffusion.mat"
    path.parent.mkdir()
    savemat(path, {"Expression1": packed.reshape(-1, 1)})
    spec = {**_advanced.EVOLUTION_SPECS[NAMES[2]], "shape": packed.shape,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    monkeypatch.setitem(_advanced.EVOLUTION_SPECS, NAMES[2], spec)
    monkeypatch.setattr(_advanced, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="source_blocked.*Cartesian"):
        load_dataset(NAMES[2])


@pytest.mark.parametrize("name", NAMES[:2])
def test_constant_coefficient_adapters_and_unconfigured_basis_are_explicitly_rejected(name):
    data = load_dataset(name)
    for model in PDE_CAPABILITIES:
        if model != "pdefind_nd":
            with pytest.raises(SkipCase, match="method_unsupported"):
                validate_pde(data, model)
    validate_pde(data, "pdefind_nd")
    case = next(c for c in configured_cases() if c["dataset"] == name)
    case["model_params"]["coordinate_basis"] = []
    with pytest.raises(SkipCase, match="method_unsupported.*coordinate_basis"):
        prepare_views(data, case)
    report = b.check_compatibility([case], mode="full", check_runtime=False)
    row = report["pairs"][0]
    assert row["status"] == "incompatible"
    assert row["preparation_status"] == "method_unsupported"


def test_config_uses_one_library_for_all_three_and_full_preflight_passes():
    cases = configured_cases()
    assert len(cases) == 3
    bases = [c["model_params"]["coordinate_basis"] for c in cases]
    assert bases == [bases[0]] * 3
    report = b.check_compatibility(cases, mode="full", check_runtime=False)
    assert report["scope"]["enumeration_complete"]
    assert {row["dataset"] for row in report["pairs"]} == set(NAMES)
    assert all(row["status"] == "compatible" for row in report["pairs"])


def test_absolute_time_basis_is_observation_only_and_holdout_cannot_change_training():
    data = load_dataset(NAMES[0])
    case = next(c for c in configured_cases() if c["dataset"] == NAMES[0])
    train, test, split = _temporal_views(data, case)
    ns, _ = namespace(train, temporal_order=1)
    basis = ["x", "t", "exp(t)"]
    matrix, terms = library(ns, ["u"], coordinate_basis=basis)
    assert "u_t" not in terms and "u_xx*exp(t)" in terms
    changed = copy.deepcopy(data)
    changed.usol[..., split["test_frame_start"]:] += 999
    changed.protocol_metadata["source_reference"] = "secret"
    changed.coef_dict = {"secret": 123}
    train2, _, _ = _temporal_views(changed, case)
    matrix2, terms2 = library(namespace(train2, temporal_order=1)[0], ["u"], coordinate_basis=basis)
    np.testing.assert_array_equal(matrix, matrix2)
    assert terms == terms2
    test_ns, _ = namespace(test, temporal_order=1)
    assert test_ns["t"][0, 0] == data.t[split["test_frame_start"]] > 0
    assert not np.intersect1d(train.t, test.t).size
    for forbidden in ("u", "u_t", "secret", "y"):
        with pytest.raises(ValueError, match="observed coordinates"):
            library(ns, ["u"], coordinate_basis=[forbidden])


def test_time_varying_diffusion_is_recovered_from_independent_analytic_modes():
    x, t = np.linspace(-1, 2, 101), np.linspace(.2, 1.2, 151)
    X, T = np.meshgrid(x, t, indexing="ij")
    diffusivity = .17
    u = sum(np.sin(k * X) * np.exp(-diffusivity * k**2 * np.exp(T)) for k in (1., 2., 3.))
    data = GridPDEDataset("analytic", None, None, 0, coords={"x": x, "t": t}, usol=u[None])
    data.state_vars = ["u"]
    ns, _ = namespace(data, temporal_order=1)
    matrix, terms = library(ns, ["u"], coordinate_basis=["exp(t)"])
    model = SparseEquationModel(threshold=.02).fit(matrix, terms, ns["u_t"])
    assert model.diagnostics_["support"] == ["u_xx*exp(t)"]
    assert model.coefficients_[terms.index("u_xx*exp(t)")] == pytest.approx(diffusivity, rel=.01)
