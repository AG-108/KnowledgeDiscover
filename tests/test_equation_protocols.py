"""Independent analytic acceptance for A3/A4, including failure accounting."""

import copy

import numpy as np
import pytest

import run_benchmark as b
from kd.dataset import Derivative, EquationTarget, GridPDEDataset, SteadyPDEDataset, load_dataset
from kd.dataset._advanced import EVOLUTION_SPECS, STEADY_SPECS, declared_targets
from kd.dataset._derivatives import (
    LocalPolynomial,
    axis_derivative,
    grid_derivative,
    namespace,
    segments_valid,
)
from kd.dataset._equations import equation_spec, targets
from kd.evaluation.equation import (
    _spatial_views,
    _temporal_views,
    periodic_grid_options,
    problems,
    run,
)
from kd.model.kd_equation import SparseEquationModel, canonical_equation, library, relative_residual


def grid(x, t, value, y=None, fields=("u",)):
    coords = {"x": x, **({"y": y} if y is not None else {}), "t": t}
    result = GridPDEDataset("analytic", None, None, 0.0, coords=coords, usol=value, time_var="t")
    result.state_vars = list(fields)
    return result


def case(model="pdefind_nd", dataset="wave"):
    catalog, _ = b.dataset_catalog()
    return b.build_experiments(catalog, models=[model], datasets=[dataset])[0]


def test_derivative_ast_and_contract():
    assert Derivative("v", (1, 1), 0).symbol == "v_xy"
    assert Derivative("u", (1,), 1).symbol == "u_xt"
    assert Derivative("u", (0,), 2).to_dict()["time_order"] == 2
    with pytest.raises(ValueError):
        Derivative("u", (1, 1, 1, 1))
    x = np.linspace(0, 1, 11)
    d = SteadyPDEDataset("steady", np.array([x * x]), coords={"x": x})
    assert not hasattr(d, "t")
    assert equation_spec(d, targets(d)[0])["time_axis"] is None


@pytest.mark.parametrize("swap", [False, True])
def test_anisotropic_rectangular_polynomial_and_channel_order(swap):
    x, y, t = np.linspace(-2, 3, 31), np.linspace(-1, 2, 43), np.linspace(0, 1, 9)
    X, Y, T = np.meshgrid(x, y, t, indexing="ij")
    u, v = X**3 + 2 * Y**2 + X * Y + T, 4 * X**2 * Y + 3 * T**2
    d = grid(
        x, t, np.array([v, u]) if swap else np.array([u, v]), y, ("v", "u") if swap else ("u", "v")
    )
    ns, _ = namespace(d, temporal_order=2)
    interior = np.s_[3:-3, 3:-3, 2:-2]
    for key, expected in {
        "u_x": 3 * X**2 + Y,
        "u_yy": 4 + 0 * X,
        "u_xy": 1 + 0 * X,
        "v_xx": 8 * Y,
        "v_tt": 6 + 0 * X,
    }.items():
        np.testing.assert_allclose(ns[key][interior], expected[interior], rtol=1e-8, atol=1e-8)
    assert np.isnan(ns["u_x"][:2]).all()


def test_axis_permutation_and_fourth_order():
    x, y = np.linspace(-1, 1, 51), np.linspace(-2, 2, 31)
    X, Y = np.meshgrid(x, y, indexing="ij")
    v = X**4 + Y**3
    a = grid_derivative(v, [x, y], Derivative("u", (1, 2)))
    z = grid_derivative(v.T, [y, x], Derivative("u", (2, 1))).T
    np.testing.assert_allclose(a[4:-4, 4:-4], z[4:-4, 4:-4], atol=1e-8)
    fourth = axis_derivative(X**4, x, 4, 0)
    np.testing.assert_allclose(fourth[4:-4], 24, atol=1e-7)


@pytest.mark.parametrize("order", [1, 2])
def test_temporal_second_order_convergence(order):
    errors = []
    for n in (41, 81, 161):
        t = np.linspace(0, 2 * np.pi, n)
        expected = np.cos(t) if order == 1 else -np.sin(t)
        actual = axis_derivative(np.sin(t), t, order, stencil=3)
        errors.append(np.sqrt(np.mean((actual[1:-1] - expected[1:-1]) ** 2)))
    assert errors[0] / errors[1] >= 3 and errors[1] / errors[2] >= 3


def test_nonuniform_coordinates():
    x = np.linspace(0.2, 1.4, 41) ** 2
    actual = axis_derivative(x**4, x, 1)
    np.testing.assert_allclose(actual[2:-2], 4 * x[2:-2] ** 3, rtol=1e-8, atol=1e-8)


def test_wave_kg_and_mixed_analytic():
    x, t = np.linspace(-1, 1, 71), np.linspace(0, 1, 91)
    X, T = np.meshgrid(x, t, indexing="ij")
    for speed, mass in [(1.0, 0.0), (0.5, 5.0)]:
        freq = np.sqrt(speed * 4 + mass)
        u = np.sin(2 * X) * np.cos(freq * T)
        d = grid(x, t, u[None])
        d.equation_targets = [EquationTarget("u_tt", time_order=2)]
        ns, _ = namespace(d, temporal_order=2)
        residual = ns["u_tt"] - speed * ns["u_xx"] + mass * ns["u"]
        assert np.nanmax(abs(residual)) < 0.002
    d = grid(x, t, (X**2 * T**2)[None])
    ns, _ = namespace(d, temporal_order=1)
    np.testing.assert_allclose(ns["u_xt"][3:-3, 2:-2], (4 * X * T)[3:-3, 2:-2], atol=1e-8)


def test_local_polynomial_hole_regions_and_degeneracy():
    rng = np.random.default_rng(7)
    p = rng.uniform(-1, 1, (600, 2))
    p = p[np.linalg.norm(p, axis=1) > 0.22]
    geometry = {
        "kind": "box",
        "verified": True,
        "bounds": [[-1, 1], [-1, 1]],
        "holes": [{"center": [0, 0], "radius": 0.2}],
    }
    regions = (p[:, 0] > 0).astype(int)
    solver = LocalPolynomial(p, geometry=geometry, region_ids=regions, neighbors=55)
    values = p[:, 0] ** 3 + 2 * p[:, 1] ** 2 + p[:, 0] * p[:, 1]
    actual = solver.derivative(values, np.array([1, 0]))
    np.testing.assert_allclose(
        actual[solver.support], (3 * p[:, 0] ** 2 + p[:, 1])[solver.support], atol=1e-8
    )
    for i, ids in enumerate(solver.neighborhoods):
        assert segments_valid(p[i], p[ids], geometry).all()
        assert (regions[ids] == regions[i]).all()
    points = np.column_stack([np.linspace(-1, 1, 50), np.zeros(50)])
    bad = LocalPolynomial(points, geometry=geometry)
    assert not bad.support.any()
    assert np.isnan(bad.derivative(np.ones(50), np.array([1, 0]))).all()


def test_unverified_scatter_geometry_rejected():
    with pytest.raises(ValueError, match="verified"):
        LocalPolynomial(np.random.default_rng(0).normal(size=(50, 2)))


def test_segment_crossing_hole_rejected():
    geo = {"kind": "box", "verified": True, "holes": [{"center": [0, 0], "radius": 0.1}]}
    assert segments_valid(np.array([-0.5, 0]), np.array([[0.5, 0], [-0.5, 0.2]]), geo).tolist() == [
        False,
        True,
    ]


def test_library_is_observation_only_and_no_self_target():
    x, t = np.linspace(-1, 1, 31), np.linspace(0, 1, 15)
    X, T = np.meshgrid(x, t, indexing="ij")
    d = grid(x, t, (X**2 + T)[None])
    d.coef_dict = {"secret": 42}
    d.sym_true = "secret"
    ns, _ = namespace(d, temporal_order=1)
    A, terms = library(ns, ["u"], coordinate_basis=["x", "sin(pi*x)"])
    assert "u_t" not in terms and all("secret" not in term for term in terms)
    d.coef_dict = {"secret": -1}
    d.sym_true = "0"
    A2, terms2 = library(
        namespace(d, temporal_order=1)[0], ["u"], coordinate_basis=["x", "sin(pi*x)"]
    )
    np.testing.assert_array_equal(A, A2)
    assert terms == terms2


def test_explicit_discovery_recovers_independent_heat_solution():
    x, t = np.linspace(-1, 2, 81), np.linspace(0, 0.5, 101)
    X, T = np.meshgrid(x, t, indexing="ij")
    # Multiple independent Fourier modes make u and u_xx identifiable.
    u = sum(np.sin(k * X) * np.exp(-0.3 * k * k * T) for k in (1.0, 2.0, 3.0))
    ns, _ = namespace(grid(x, t, u[None]), temporal_order=1)
    A, terms = library(ns, ["u"])
    m = SparseEquationModel(threshold=0.01).fit(A, terms, ns["u_t"])
    assert m.diagnostics_["support"] == ["u_xx"]
    assert abs(m.coefficients_[terms.index("u_xx")] - 0.3) / 0.3 < 0.01


@pytest.mark.parametrize("factor", [1.0, -17.0, 0.001])
def test_implicit_canonical_scale_sign_and_zero(factor):
    eq = canonical_equation(np.array([1.0, 2.0]) * factor, ["u_xx", "u_yy"])
    np.testing.assert_allclose(list(eq.values()), [1 / np.sqrt(5), 2 / np.sqrt(5)])
    metric = relative_residual(
        np.array([[1.0, -1.0], [2.0, -2.0]]), np.array([1.0, 1.0]) * factor, 1.0
    )
    assert metric["equation_residual_nrmse"] == pytest.approx(0.0)
    for coefficients, terms in [
        ([0, 0], ["u_x", "u_y"]),
        ([1, -1], ["u_x", "u_x"]),
        ([1], ["u_x-u_x"]),
    ]:
        with pytest.raises(ValueError):
            canonical_equation(coefficients, terms)
    with pytest.raises(ValueError):
        relative_residual(np.ones((2, 2)), np.zeros(2), 1.0)


def test_implicit_sparse_support_and_ambiguity():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 6))
    X[:, 2] = 2 * X[:, 0] - 3 * X[:, 1]
    terms = ["u_xx", "u_yy", "u", "u_x", "u_y", "1"]
    m = SparseEquationModel(null_tolerance=1e-7).fit(X, terms)
    assert set(m.diagnostics_["support"]) == set(terms[:3])
    assert not m.diagnostics_["ambiguous"]
    expected = canonical_equation([2, -3, -1], terms[:3])
    for key, val in expected.items():
        assert m.canonical_[key] == pytest.approx(val, abs=0.01)
    X[:, 4] = X[:, 3]
    ambiguous = SparseEquationModel(null_tolerance=1e-7).fit(X, terms)
    assert (
        ambiguous.diagnostics_["ambiguous"] and ambiguous.diagnostics_["near_null_dimension"] == 2
    )


def test_temporal_split_precedes_derivatives_and_truth_changes():
    x, t = np.linspace(-1, 1, 31), np.linspace(0, 1, 41)
    X, T = np.meshgrid(x, t, indexing="ij")
    d = grid(x, t, (X * X + T * T)[None])
    c = case()
    train, test, split = _temporal_views(d, c)
    changed = copy.deepcopy(d)
    changed.usol[..., split["test_frame_start"] :] += 999
    train2, _, _ = _temporal_views(changed, c)
    np.testing.assert_array_equal(
        namespace(train, temporal_order=2)[0]["u_tt"],
        namespace(train2, temporal_order=2)[0]["u_tt"],
    )
    assert not np.intersect1d(train.t, test.t).size


def steady_fixture(dimension=2, known=False):
    axes = [np.linspace(-1, 1, 27 + i * 2) for i in range(dimension)]
    mesh = np.meshgrid(*axes, indexing="ij")
    # Harmonic polynomial, not a single separable Fourier eigenmode.
    u = mesh[0] ** 3 - 3 * mesh[0] * mesh[1] ** 2 + mesh[0] * mesh[1]
    if known:
        u += sum(a**2 for a in mesh)
    geometry = {"kind": "box", "verified": True, "bounds": [[-1, 1]] * dimension}
    target = EquationTarget(
        "implicit_relation",
        time_order=0,
        kind="implicit",
        known_source="forcing" if known else None,
    )
    return SteadyPDEDataset(
        "analytic",
        u[None],
        coords=dict(zip("xyz", axes)),
        geometry=geometry,
        known_sources={"forcing": np.full(u.shape, -2.0 * dimension)} if known else {},
        equation_targets=[target],
    )


@pytest.mark.parametrize("dimension,known", [(2, False), (2, True), (3, True)])
def test_steady_derivatives_known_source_2d_3d(dimension, known):
    d = steady_fixture(dimension, known)
    ns, _ = namespace(d)
    laplacian = sum(ns["u_" + axis * 2] for axis in "xyz"[:dimension])
    np.testing.assert_allclose(
        laplacian[np.isfinite(laplacian)], 2.0 * dimension if known else 0.0, atol=1e-8
    )


def test_steady_split_gap_and_leakage():
    d = steady_fixture()
    c = case("implicit_pde", "laplace_disk")
    m = SparseEquationModel()
    train, test, split = _spatial_views(d, c, m)
    d2 = copy.deepcopy(d)
    # Locate the test points exactly in the original grid.
    for p in test.points:
        ids = tuple(np.flatnonzero(a == v)[0] for a, v in zip(d2.coords.values(), p))
        d2.values[(0, *ids)] += 9999
    train2, _, split2 = _spatial_views(d2, c, m)
    assert split == split2 and split["excluded_points"] > 0
    np.testing.assert_array_equal(train.values, train2.values)
    np.testing.assert_array_equal(namespace(train)[0]["u_xx"], namespace(train2)[0]["u_xx"])


def test_steady_fit_score_has_no_fake_prediction_or_zero_r2():
    d = steady_fixture(2, True)
    c = case("implicit_pde", "laplace_disk")
    result = run(d, next(problems(d, c)), c)
    assert result["prediction_status"] == "abstain"
    assert "test_r2" not in result
    assert result["equation_residual_nrmse"] < 0.01
    assert 0 < result["equation_residual_finite_coverage"] <= 1


def test_multifield_pressure_auxiliary_and_gauge():
    x, y, t = np.linspace(-1, 1, 21), np.linspace(-2, 2, 25), np.linspace(0, 1, 11)
    X, Y, T = np.meshgrid(x, y, t, indexing="ij")
    d = grid(x, t, np.array([X + T, Y - T, X * Y + T]), y, ("u", "v", "p"))
    assert [target.key for target in targets(d)] == ["u_t", "v_t"]
    ns, _ = namespace(d, temporal_order=1)
    assert "p_t" not in library(ns, d.state_vars)[1]
    d.usol[2] += 500
    np.testing.assert_allclose(namespace(d)[0]["p_x"][3:-3], ns["p_x"][3:-3], atol=1e-10)


def test_pdenet_anisotropic_rectangular_and_boundary():
    x, y, t = np.arange(21) * 0.07, np.arange(31) * 0.03, np.arange(11) * 0.01
    d = grid(x, t, np.zeros((1, 21, 31, 11)), y)
    with pytest.raises(b.SkipCase, match="boundary"):
        periodic_grid_options(d)
    d.protocol_metadata = {
        "boundary": {"kind": "periodic", "verified": True, "endpoint": "excluded"}
    }
    opts = periodic_grid_options(d)
    assert opts["mesh_size"] == [21, 31]
    assert opts["dx"] == pytest.approx([0.07, 0.03])
    assert opts["mesh_bound"][1] == pytest.approx([21 * 0.07, 31 * 0.03])
    assert b.prepare_pde(d, "pdenet") is d


@pytest.mark.parametrize("name", list(EVOLUTION_SPECS))
def test_real_checked_evolution_load(name):
    d = load_dataset(name)
    assert np.isfinite(d.usol).all()
    assert targets(d) == declared_targets(name)


@pytest.mark.parametrize("name", list(STEADY_SPECS))
def test_all_twelve_steady_files_load_or_explicit_source_block(name):
    try:
        d = load_dataset(name)
    except ValueError as exc:
        assert "source_blocked" in str(exc)
    else:
        assert not hasattr(d, "t")
        assert d.values.shape[1] == d.n_response
        assert targets(d) == declared_targets(name)
        if name == "tlc_steady_lid_driven":
            assert d.protocol_metadata["known_parameters"] == {"Re": 100}
            assert len(d.spatial_vars) == 2


def test_new_task_targets_setup_failure_and_independent_fit_units(monkeypatch):
    c = case("implicit_pde", "tlc_steady_ns2d")
    specs = b._prepare_targets(c)
    assert [s["target"] for s in specs] == ["momentum_u", "momentum_v"]
    assert all(s["data_metadata"]["auxiliary_fields"] == ["p"] for s in specs)

    def bad(_):
        raise FileNotFoundError("asset absent")
        yield

    monkeypatch.setattr(b, "_load_instances", bad)
    specs = b._prepare_targets(c)
    assert len(specs) == 2 and all(s["preparation_row"]["target_manifest_complete"] for s in specs)
    assert all(s["preparation_row"]["status"] == "error" for s in specs)


def test_known_constraint_does_not_become_discovery_target():
    d = load_dataset("tlc_steady_ns2d")
    assert len(targets(d)) == 2 and "continuity" not in [t.key for t in targets(d)]
    assert d.protocol_metadata["known_constraints"]


def test_harmonic_analytic_equation_discovery_not_single_truth_column():
    x, y = np.linspace(-1, 1, 61), np.linspace(-0.8, 1.2, 71)
    X, Y = np.meshgrid(x, y, indexing="ij")
    u = (
        np.exp(0.7 * X) * np.cos(0.7 * Y)
        + np.exp(1.3 * X) * np.sin(1.3 * Y)
        + 0.1 * (X * X - Y * Y)
    )
    d = SteadyPDEDataset("harmonic", u[None], coords={"x": x, "y": y})
    ns, _ = namespace(d)
    A, terms = library(ns, ["u"], implicit=True)
    m = SparseEquationModel(null_tolerance=1e-5).fit(A, terms)
    assert set(m.diagnostics_["support"]) == {"u_xx", "u_yy"}
    assert not m.diagnostics_["ambiguous"]
    coef = m.canonical_
    assert coef["u_xx"] / coef["u_yy"] == pytest.approx(1.0, rel=0.01)


@pytest.mark.parametrize("scheme", ["central", "upwind"])
def test_pdenet_actual_convolution_axis_and_mixed_scaling(scheme):
    import torch

    from kd.model.kd_pdenet import initparameters, setenv

    x, y = np.arange(23) * 0.04, np.arange(31) * 0.09
    X, Y = np.meshgrid(x, y, indexing="ij")
    options = {
        "--eps": 23 * 0.04,
        "--cell_num": 1,
        "--dx": 0.04,
        "--dx_axes": [0.04, 0.09],
        "--mesh_size": [23, 31],
        "--mesh_bound": [[0, 0], [23 * 0.04, 31 * 0.09]],
        "--dtype": "double",
        "--constraint": "frozen",
        "--kernel_size": 5,
        "--dt": 0.01,
        "--max_order": 2,
        "--channel_names": "u",
        "--hidden_layers": 1,
        "--scheme": scheme,
        "--device": "cpu",
        "--npseed": 0,
        "--torchseed": 0,
    }
    # Callback persistence is not needed to inspect physical convolution units.
    from unittest.mock import patch

    with patch.object(setenv.setcallback, "setcallback"):
        _, _, model = setenv._set_model(options)
    initparameters.initkernels(model, scheme)
    value = torch.tensor((X**2 + 3 * Y**2 + X * Y)[None], dtype=torch.float64)
    for fd, expected in [
        (model.fd10, 2 * X + Y),
        (model.fd01, 6 * Y + X),
        (model.fd20, 2 + 0 * X),
        (model.fd02, 6 + 0 * X),
        (model.fd11, 1 + 0 * X),
    ]:
        actual = fd(value).detach().numpy()[0]
        np.testing.assert_allclose(actual[4:-4, 4:-4], expected[4:-4, 4:-4], atol=1e-8)
    assert model.fd11.dx == [0.04, 0.09]


def test_supplied_continuity_is_excluded_from_implicit_discovery():
    rng = np.random.default_rng(12)
    X = rng.normal(size=(600, 5))
    X[:, 1] = -X[:, 0]
    X[:, 4] = 2 * X[:, 2] - 3 * X[:, 3]
    terms = ["u_x", "v_y", "p_x", "u_xx", "u"]
    model = SparseEquationModel(null_tolerance=1e-7).fit(X, terms, excluded_relations=["u_x+v_y"])
    assert set(model.diagnostics_["support"]) == {"p_x", "u_xx", "u"}
    assert model.diagnostics_["ambiguous"]


def test_anchored_poisson_does_not_discover_copy_of_constant_forcing():
    d = steady_fixture(2, True)
    c = case("implicit_pde", "laplace_disk")
    result = run(d, next(problems(d, c)), c)
    assert "1" not in result["details"]["discovery"]["support"]
    assert set(result["details"]["discovery"]["support"]) == {"u_xx", "u_yy"}
    eq = result["equation_expression"]
    coefficients = dict(zip(eq["terms"], eq["coefficients"]))
    assert coefficients["u_xx"] / coefficients["u_yy"] == pytest.approx(1.0, rel=0.01)


def test_new_equation_timeout_keeps_next_rhs_and_resume_terminal(monkeypatch, tmp_path):
    # Use a verified analytic domain: source-blocked observations now terminate in planning.
    c = case("implicit_pde", "tlc_steady_ns2d")
    axis = np.linspace(0, 1, 11)
    data = SteadyPDEDataset(
        "analytic", np.zeros((3, 11, 11)), coords={"x": axis, "y": axis},
        state_vars=("u", "v", "p"), geometry={"verified": True, "kind": "box",
                                             "bounds": [[0, 1], [0, 1]]},
        equation_targets=declared_targets("tlc_steady_ns2d"),
    )
    monkeypatch.setattr(b, "_load_instances", lambda _: iter([("analytic", data)]))
    specs = b._prepare_targets(c)
    assert all("preparation_row" not in spec for spec in specs)
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
    assert calls == ["momentum_u", "momentum_v"]
    b.execute_case(c, tmp_path, timeout=1, resume=True)
    assert calls == ["momentum_u", "momentum_v"]


def test_real_tlc_symbols_units_and_pressure_denominator():
    for name in ("tlc_heat_longtime", "tlc_heat_multiscale_lesspoints"):
        d = load_dataset(name)
        assert d.state_vars == ["u"]
        assert d.protocol_metadata["field_units"] == {"u": "K"}
        assert [t.key for t in targets(d)] == ["u_t"]
    d = load_dataset("tlc_ns_long")
    assert d.state_vars == ["u", "v", "p"]
    assert [t.key for t in targets(d)] == ["u_t", "v_t"]
    assert [t.key for t in targets(load_dataset("wdwake"))] == ["u_t", "v_t"]


def test_metric_versions_do_not_pool_legacy_and_new_residual_normalizations():
    modern = b._base_row(case())
    legacy_case = case("pdefind", "heat")
    legacy = b._base_row(legacy_case)
    assert modern["metric_protocol_version"] == "2.2"
    assert legacy["metric_protocol_version"] == "2.1"
    aggregate = b.aggregate_results([modern, legacy])
    assert {a["metric_protocol_version"] for a in aggregate} == {"2.1", "2.2"}


def test_implicit_pde_rejects_pure_algebraic_near_constant_relation():
    values = np.linspace(.999, 1.001, 500)
    columns = np.column_stack([np.ones(500), values, values**2])
    with pytest.raises(ValueError, match="No nontrivial implicit relation"):
        SparseEquationModel().fit(columns, ["1", "u", "u**2"])


def test_implicit_rejects_common_vanishing_multiplier():
    rng = np.random.default_rng(21)
    u = np.linspace(.999, 1.001, 500)
    ux = rng.normal(size=500)
    X = np.column_stack([ux, u*ux, u*u*ux])
    with pytest.raises(ValueError, match="No nontrivial implicit relation"):
        SparseEquationModel().fit(X, ["u_x", "u*u_x", "u**2*u_x"])
