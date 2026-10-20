"""All PDE adapters are scheduled against the same FDE field observations."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from benchmark_helpers import (
    FDE_FIELD_DATASETS,
    FDE_RESPONSE_DATASETS,
    PDE_METHODS,
    SR_METHODS,
)
from scipy.io import savemat

import run_benchmark as benchmark
from kd.benchmark_execution import SkipCase
from kd.dataset import GridPDEDataset, load_dataset
from kd.dataset._fractional import FDE_PDE_READY_DATASETS, FDE_PDE_SPECS, FDE_SPECS
from kd.evaluation.fractional import method_view, temporal_views


@pytest.fixture
def field_root(tmp_path):
    directory = tmp_path / "FDE/data"
    directory.mkdir(parents=True)
    x, t = np.linspace(0, 3, 40), np.arange(15.0)
    u = np.exp(-t[:, None] / 20) * np.sin(x[None, :]) + 0.03 * t[:, None]
    for response in FDE_PDE_SPECS.values():
        spec = FDE_SPECS[response]
        path = tmp_path / spec["path"]
        if spec["layout"]:
            xkey, tkey, ukey = spec["keys"]
            savemat(path, {xkey: x[None], tkey: t[None], ukey: u})
        else:
            np.savetxt(path, u)
    return tmp_path


def case(model="pdefind", name="fde_pt_fade_pde", **overrides):
    catalog, _ = benchmark.dataset_catalog()
    return benchmark.build_experiments(catalog, models=[model], datasets=[name], config=overrides)[
        0
    ]


def test_every_pde_adapter_is_present_in_both_fde_configs():
    catalog, _ = benchmark.dataset_catalog()
    expected = PDE_METHODS
    for path in ("fde_pde.json", "fde_all_methods.json"):
        config = json.loads((Path("configs/benchmark") / path).read_text(encoding="utf-8"))
        cases = benchmark.build_experiments(
            catalog, models=config["models"], datasets=config["datasets"], tasks=config["tasks"]
        )
        expected_pairs = {(model, dataset) for model in PDE_METHODS for dataset in FDE_FIELD_DATASETS}
        if path == "fde_all_methods.json":
            expected_pairs |= {(model, dataset) for model in SR_METHODS for dataset in FDE_RESPONSE_DATASETS}
        assert {(c["model"], c["dataset"]) for c in cases} == expected_pairs
        assert len(cases) == len(expected_pairs)
        pde = [c for c in cases if c["task"] == "pde"]
        assert {c["model"] for c in pde} == expected
        assert {c["dataset"] for c in pde} == FDE_FIELD_DATASETS
        assert len(pde) == len(expected) * len(FDE_FIELD_DATASETS)
        for c in pde:
            assert c["scientific_task"] == "fractional_field_integer_pde_diagnostic"
            assert c["ground_truth"] is None
            assert c["physical_family"] == FDE_PDE_SPECS[c["dataset"]]
            row = benchmark._base_row(c)
            assert row["fde_pde_protocol_version"] == "fde-pde-diagnostic-v1"
            assert row["metric_target"] == "observed_equation_residual"
            assert "fde_response_protocol_version" not in row
            assert row["fractional_operator_evaluated"] is False
            assert row["benchmark_role"] == "diagnostic"
            assert row["recovery_eligible"] is False
            assert row["metric_protocol_version"] == (
                "2.2" if c["model"] in {"pdefind_nd", "implicit_pde"} else "2.1"
            )
    defaults = benchmark.build_experiments(catalog)
    assert not any(c["dataset"] in FDE_PDE_SPECS for c in defaults)
    assert not any("creep_pde" in n or "silt_pde" in n for n in catalog)


@pytest.mark.parametrize("name", FDE_PDE_READY_DATASETS)
def test_pde_view_preserves_raw_field_and_embedded_coordinates(field_root, name):
    response = load_dataset(FDE_PDE_SPECS[name], root=field_root)
    grid = load_dataset(name, root=field_root)
    assert isinstance(grid, GridPDEDataset)
    assert grid.spatial_vars == ["x"] and grid.n_response == 1
    np.testing.assert_array_equal(grid.usol, response.values.T)
    np.testing.assert_array_equal(grid.x, response.x)
    np.testing.assert_array_equal(grid.t, response.t)
    assert grid.protocol_metadata["boundary"] == "unknown"
    assert grid.protocol_metadata["fractional_orders"] is None


def test_headerless_pde_field_retains_caller_coordinate_evidence(field_root):
    grid = load_dataset(
        "fde_ground_water_flow_pde",
        root=field_root,
        coordinates={
            "x": np.linspace(0, 3, 40),
            "t": np.arange(15.0),
            "layout": "tx",
            "evidence": "synthetic generator",
        },
    )
    assert grid.protocol_metadata["coordinate_status"] == "caller_declared"
    assert grid.protocol_metadata["coordinate_evidence"] == "synthetic generator"


def test_short_field_has_disjoint_derivative_support_and_unchanged_train_data(field_root):
    grid = load_dataset("fde_pt_fde_pde", root=field_root)
    train, test, split = temporal_views(grid, 0.2)
    assert len(train.t) == 10 and len(test.t) == 3
    assert split["support_gap_frames"] == 2
    assert set(train.t).isdisjoint(test.t)
    changed = copy.deepcopy(grid)
    changed.usol[:, split["test_frame_start"] :] += 1e10
    changed_train, changed_test, _ = temporal_views(changed, 0.2)
    np.testing.assert_array_equal(changed_train.usol, train.usol)
    assert np.all(changed_test.usol > test.usol)
    too_short = benchmark._slice_pde_time(grid, 0, 9)
    with pytest.raises(SkipCase, match="insufficient_support"):
        temporal_views(too_short, 0.2)


def test_implicit_representation_uses_same_observations_without_mutating_explicit_target(
    field_root,
):
    from kd.evaluation.equation import problems, validate

    grid = load_dataset("fde_pt_fade_pde", root=field_root)
    view = method_view(grid, "implicit_pde")
    np.testing.assert_array_equal(view.usol, grid.usol)
    assert grid.equation_targets[0].kind == "explicit"
    assert view.equation_targets[0].kind == "implicit"
    validate(grid, "implicit_pde")
    validate(grid, "pdefind_nd")
    problem = next(problems(grid, case("implicit_pde")))
    assert problem["target"] == "u_t"
    assert problem["equation_target"]["kind"] == "implicit"
    assert "fractional_order_unknown" in problem["metadata"]["temporal_order_status"]


@pytest.mark.parametrize("model", ["pdefind", "pdefind_nd", "implicit_pde", "pdenet"])
def test_blocked_or_missing_assets_keep_one_declared_target(field_root, tmp_path, model):
    name = "fde_ground_water_flow_pde"
    c = case(model, name, datasets={name: {"root": str(field_root)}})
    targets = benchmark._prepare_targets(c)
    assert len(targets) == 1
    target = targets[0]
    assert target["target"] == ("u_t" if model in {"pdefind_nd", "implicit_pde"} else "PDE")
    assert target["preparation_row"]["capability_status"] == "source_blocked"
    c["dataset_params"]["root"] = str(tmp_path / "missing")
    target = benchmark._prepare_targets(c)[0]
    assert target["preparation_row"]["status"] == "error"
    assert target["preparation_row"]["capability_status"] == "missing_asset"


def test_pdenet_reports_native_dimension_limit_without_fabricating_a_spatial_axis(field_root):
    grid = load_dataset("fde_pt_fade_pde", root=field_root)
    with pytest.raises(SkipCase, match="two spatial dimensions") as exc:
        benchmark.prepare_pde(grid, "pdenet")
    row = benchmark._failure(case("pdenet"), exc.value, target="PDE")
    assert row["status"] == "skipped" and row["capability_status"] == "method_unsupported"


@pytest.mark.parametrize(
    "model", ["pdefind", "pdefind_nd", "implicit_pde", "sindy", "pysindy", "weakform", "sga"]
)
def test_native_integer_methods_fit_only_the_train_block_and_score_the_heldout_field(
    field_root, model, monkeypatch
):
    def unexpected_rollout(*args, **kwargs):
        pytest.fail("Unverified FDE boundaries cannot be used for a periodic rollout")

    monkeypatch.setattr(benchmark, "pde_rollout_metrics", unexpected_rollout)
    c = case(model)
    grid = load_dataset(c["dataset"], root=field_root)
    problem = next(iter(benchmark._case_problems(grid, c)))
    result = benchmark._run_problem(grid, problem, c, None)
    assert result["expression"]
    assert result["metric_target"] == "observed_equation_residual"
    assert result["split"]["n_train_frames"] == 10
    assert result["split"]["n_test_frames"] == 3
    assert result["equation_residual_points"] > 0
    if model not in {"pdefind_nd", "implicit_pde"}:
        assert result["equation_residual_total_points"] == 40 * 3
    assert result["equation_residual_status"] == "ok"
    assert np.isfinite(result["equation_residual_nrmse"])
    assert result["equation_rollout_status"] == "abstain"
    assert result["recovery_eligible"] is False


def test_fde_residual_scores_all_test_frames_and_observed_coordinate_terms(field_root):
    grid = load_dataset("fde_pt_fade_pde", root=field_root)
    _, test, _ = temporal_views(grid, 0.2)
    expression = "-0.0106 * mul(x1,u1)"
    score = benchmark.pde_equation_residual(expression, test)
    rhs = -0.0106 * test.x[:, None] * test.usol
    ut = np.gradient(test.usol, test.t, axis=-1, edge_order=2)
    assert score["equation_residual_total_points"] == test.usol.size
    assert score["equation_residual_mse"] == pytest.approx(np.mean((ut - rhs) ** 2))
    assert score["equation_residual_status"] == "ok"
    invalid = benchmark.pde_equation_residual("(ux / 0)+(x*x)", test)
    assert invalid["equation_residual_status"] == "nonfinite_candidate"
    assert invalid["equation_residual_finite_coverage"] == 0
    assert (
        benchmark.pde_equation_residual("( ^3 u)", test)["equation_residual_status"]
        == "abstain_unsupported_expression"
    )


def test_sga_export_keeps_default_term_weights_and_derivatives_of_generated_terms():
    from kd.model.sga.adapter import symbolic_rhs

    def node(name, children=0, start=None):
        return SimpleNamespace(name=name, child_num=children, child_st=start)

    # 0.25*u + 0.4*d(u*u)/dx - 2*u**3, including the native default u term.
    derivative = SimpleNamespace(
        tree=[
            [node("d", 2, 0)],
            [node("*", 2, 0), node("x")],
            [node("u"), node("u")],
        ]
    )
    power = SimpleNamespace(tree=[[node("^3", 1, 0)], [node("u")]])
    expression = symbolic_rhs(["u"], [derivative, power], [0.25, 0.4, -2.0])
    values = np.array([0.2, 0.7, 1.1])
    slopes = np.array([1.0, -0.2, 0.5])
    import sympy as sp

    predicted = sp.lambdify(
        [sp.Symbol("u"), sp.Symbol("u_x")], expression.split("=", 1)[1], "numpy"
    )(values, slopes)
    np.testing.assert_allclose(predicted, 0.25 * values + 0.8 * values * slopes - 2 * values**3)
    with pytest.raises(ValueError, match="align"):
        symbolic_rhs(["u"], [power], [1.0])


@pytest.mark.external_data(
    "kd/dataset/FDE/data/pt_fade.mat",
    "kd/dataset/FDE/data/pt_fde.mat",
    "kd/dataset/FDE/data/tsfade_fft.mat",
)
def test_real_archive_fields_have_consistent_response_and_pde_views():
    for name in FDE_PDE_READY_DATASETS:
        response = load_dataset(FDE_PDE_SPECS[name])
        grid = load_dataset(name)
        np.testing.assert_array_equal(grid.usol.T, response.values)
        train, test, _ = temporal_views(grid, 0.2)
        assert train.t[-1] < test.t[0]
        assert len(test.t) >= 3


def test_combined_config_is_valid_and_cli_enumerates_every_pde_pair(tmp_path, lightweight_cli_environment):
    assert (
        benchmark.main(
            [
                "--config",
                "configs/benchmark/fde_all_methods.json",
                "--dry-run",
                "--output-dir",
                str(tmp_path),
            ]
        )
        == 0
    )
    manifest = json.loads((tmp_path / "benchmark_manifest.json").read_text(encoding="utf-8"))
    assert {c["dataset"] for c in manifest["cases"]} == FDE_FIELD_DATASETS | FDE_RESPONSE_DATASETS
    pde = [c for c in manifest["cases"] if c["task"] == "pde"]
    assert {c["model"] for c in pde} == PDE_METHODS
    assert {c["fde_pde_protocol_version"] for c in pde} == {"fde-pde-diagnostic-v1"}
    assert manifest["profile"] == "smoke"
