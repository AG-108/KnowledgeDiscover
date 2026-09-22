"""Benchmark contracts: coverage, independent holdouts, adapters and recovery."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_benchmark as benchmark
from kd.dataset import GridPDEDataset, ODEDataset, TabularRegressionDataset


def make_case(model="gplearn", dataset="Koza-2", **kwargs):
    catalog, _ = benchmark.dataset_catalog()
    return benchmark.build_experiments(catalog, models=[model], datasets=[dataset], **kwargs)[0]


def test_full_matrix_covers_catalog_and_examples():
    from kd.dataset import DATASET_REGISTRY

    catalog, excluded = benchmark.dataset_catalog()
    cases = benchmark.build_experiments(catalog)
    assert set(DATASET_REGISTRY) <= set(catalog)
    assert set(benchmark.BASELINES) == {c["model"] for c in cases}
    assert set(catalog) - {"rubber_test"} == {c["dataset"] for c in cases}
    assert {c["task"] for c in cases} == {"sr", "ode", "pde"}
    assert "wave_breaking" in catalog and "advection_diffusion" in excluded
    assert len({c["name"] for c in cases}) == len(cases)
    for spec in benchmark.BASELINES.values():
        assert (benchmark.ROOT / "examples" / spec["example"]).is_file()


def test_domain_datasets_are_included_in_the_full_matrix():
    catalog, _ = benchmark.dataset_catalog()
    expected = {
        "cyt_flatplate": ("sr", 9),
        "cyt_naca0012": ("sr", 9),
        "solid_dif": ("sr", 9),
        "solid_strain_stress": ("sr", 9),
        "solid_hardening": ("sr", 9),
        "vgs_I_0-100": ("pde", 12),
        "vgs_I_100-200": ("pde", 12),
        "vgs_II_0-1000": ("pde", 12),
        "vgs_II_1000-2000": ("pde", 12),
    }

    cases = benchmark.build_experiments(catalog, datasets=list(expected))
    for dataset, (task, baseline_count) in expected.items():
        selected = [case for case in cases if case["dataset"] == dataset]
        assert len(selected) == baseline_count
        assert {case["task"] for case in selected} == {task}


def test_selection_pairing_and_config_identity():
    catalog, _ = benchmark.dataset_catalog()
    cases = benchmark.build_experiments(
        catalog, models=["gplearn"], datasets=["rubber_*"], seeds=[3, 3, 4]
    )
    assert len(cases) == 2
    assert {c["dataset"] for c in cases} == {"rubber_train"}
    assert [c["model_params"]["random_state"] for c in cases] == [3, 4]
    changed = make_case(config={"models": {"gplearn": {"generations": 7}}})
    assert changed["name"] != make_case()["name"]
    with pytest.raises(ValueError, match="No model matches"):
        benchmark.build_experiments(catalog, models=["typo"])
    evaluated = make_case(config={"evaluation": {"Koza-2": {
        "information_criteria": {"n_free_parameters": 2,
                                 "likelihood_model": "gaussian_mle_variance"}
    }}})
    assert evaluated["evaluation"]["information_criteria"]["n_free_parameters"] == 2


def test_repository_has_one_config_per_baseline():
    settings, path = benchmark.load_benchmark_config()
    configs = benchmark.load_baseline_configs(settings["baseline_config_dir"])
    assert path == benchmark.DEFAULT_CONFIG_FILE
    assert set(configs) == set(benchmark.BASELINES)
    assert all(configs[name]["name"] == name for name in configs)
    full = make_case(model="dso", profile="full")
    assert full["model_params"]["n_samples"] == 2_000_000
    assert full["run_params"]["max_train_samples"] is None


def test_device_assignment_is_recorded_and_cpu_only_models_reject_cuda():
    catalog, _ = benchmark.dataset_catalog()
    case = benchmark.build_experiments(
        catalog, models=["dso"], datasets=["Koza-2"], devices={"dso": "cuda:3"}
    )[0]
    assert case["device"] == "cuda:3"
    assert case["model_params"]["device"] == "cuda:3"
    with pytest.raises(ValueError, match="does not support CUDA"):
        benchmark.build_experiments(
            catalog, models=["gplearn"], datasets=["Koza-2"], devices={"gplearn": "cuda:0"}
        )


def test_total_config_and_cli_device_override(tmp_path):
    config = tmp_path / "server.json"
    config.write_text(
        json.dumps(
            {
                "profile": "full",
                "models": ["dso"],
                "datasets": ["Koza-2"],
                "devices": {"dso": "cuda:7"},
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "output"
    assert (
        benchmark.main(
            [
                "--config",
                str(config),
                "--device",
                "dso=cpu",
                "--output-dir",
                str(output),
                "--dry-run",
            ]
        )
        == 0
    )
    manifest = json.loads((output / "benchmark_manifest.json").read_text(encoding="utf-8"))
    assert manifest["profile"] == "full"
    assert manifest["devices"] == {"dso": "cpu"}
    assert manifest["cases"][0]["model_params"]["n_samples"] == 2_000_000


def test_legacy_override_config_migrates_device(tmp_path):
    config = tmp_path / "legacy.json"
    config.write_text(
        json.dumps({"models": {"dso": {"n_samples": 17, "device": "cuda:2"}}}), encoding="utf-8"
    )
    settings, _ = benchmark.load_benchmark_config(config)
    assert settings["devices"]["dso"] == "cuda:2"
    assert settings["overrides"]["models"]["dso"] == {"n_samples": 17}


def test_tabular_holdout_respects_groups_and_train_cap():
    X = np.arange(80).reshape(40, 2)
    groups = np.repeat(["curve_a", "curve_b", "curve_c", "curve_d"], 10)
    data = TabularRegressionDataset("solid_dif", X, X[:, 0], groups=groups)
    case = make_case(dataset="solid_dif", config={"run": {"max_train_samples": 5}})
    problem = next(benchmark.regression_problems(data, case))
    train_ids = (problem["X_train"][:, 0] / 2).astype(int)
    test_ids = (problem["X_test"][:, 0] / 2).astype(int)
    assert not set(groups[train_ids]) & set(groups[test_ids])
    assert len(train_ids) == 5 and len(test_ids) == 10
    assert problem["split"] == "group_holdout"


def test_regression_metrics_and_exact_symbolic_recovery():
    scores = benchmark.regression_metrics([1, 2, 3], [1, 2, 4])
    assert scores["test_nrmse"] == pytest.approx(np.sqrt(scores["test_mse"] / (2 / 3)))
    assert scores["test_r2"] == pytest.approx(0.5)
    recovered = benchmark.expression_metrics("add(mul(X0, X0), sub(X0, X0))", "x1**2", ["x1"])
    assert recovered["exact_recovery"] == 1
    assert recovered["expression_complexity"] == 7
    partial = benchmark.regression_metrics([1, 2, 3], [1, np.nan, 3])
    assert partial["prediction_finite_coverage"] == pytest.approx(2 / 3)
    assert "test_mse" not in partial


def test_ode_holds_out_whole_trajectories_and_fits_each_derivative():
    trajectories = []
    for i in range(5):
        t = np.linspace(0, 1, 9 + i)
        trajectories.append(
            {
                "id": f"trial{i}",
                "t": t,
                "state": np.array([t * t + i, 2 * t]),
                "params": {"mass": i + 1},
            }
        )
    data = ODEDataset("ball_drop", trajectories, ["h", "v"], param_names=["mass"])
    problems = list(benchmark.regression_problems(data, make_case(dataset="ball_drop")))
    assert [p["target"] for p in problems] == ["d(h)/dt", "d(v)/dt"]
    for p in problems:
        assert p["split"] == "trajectory_holdout"
        assert not set(p["metadata"]["train_trajectories"]) & set(
            p["metadata"]["test_trajectories"]
        )
        assert not set(p["X_train"][:, 2]) & set(p["X_test"][:, 2])
    np.testing.assert_allclose(problems[0]["y_train"], problems[0]["X_train"][:, 1], atol=1e-12)
    np.testing.assert_allclose(problems[1]["y_test"], 2, atol=1e-12)


def test_rubber_uses_official_test_only(monkeypatch):
    train = TabularRegressionDataset("rubber_train", [[0], [1]], [0, 2])
    test = TabularRegressionDataset("rubber_test", [[2], [3]], [4, 6])
    import kd.dataset

    monkeypatch.setattr(
        kd.dataset, "load_dataset", lambda name: test if name == "rubber_test" else None
    )
    p = next(benchmark.regression_problems(train, make_case(dataset="rubber_train")))
    np.testing.assert_array_equal(p["X_train"], train.X)
    np.testing.assert_array_equal(p["X_test"], test.X)
    assert p["metadata"]["test_dataset"] == "rubber_test"


def scalar_grid():
    x, t = np.linspace(1, 2, 8), np.linspace(0, 1, 9)
    u = 10 * x[:, None] + t[None, :]
    return GridPDEDataset("toy", None, None, 0, x=x, t=t, usol=u, legacy=True)


def test_pde_temporal_holdout_residual_structure_and_rollout():
    data = scalar_grid()
    train, evaluation, split = benchmark.temporal_block_holdout(data, 0.2)
    assert split == 7
    np.testing.assert_array_equal(train.t, data.t[:split])
    np.testing.assert_array_equal(evaluation.t, data.t[split - 1 :])
    np.testing.assert_array_equal(evaluation.usol[..., 1:], data.usol[..., split:])

    residual = benchmark.pde_equation_residual("u_t = 1", evaluation)
    assert residual["equation_residual_nrmse"] == pytest.approx(0, abs=1e-12)
    metadata = {"boundary_conditions": {"type": "periodic"},
                "endpoint_convention": "periodic_endpoint_excluded",
                "spatial_discretization": "spectral_fft",
                "solver": {"method": "RK45", "rtol": 1e-8, "atol": 1e-10, "max_step": None},
                "reference_nrmse_tolerance": 1e-8}
    rollout = benchmark.symbolic_pde_rollout_metrics("u_t = 1", data, split, metadata, "1")
    assert rollout["equation_rollout_nrmse"] == pytest.approx(0, abs=1e-12)
    assert rollout["equation_rollout_steps"] == 2

    structure = benchmark.pde_structure_metrics("u_t = -0.9*u*u_x + 0.11*u_xx", "-u*u_x + 0.1*u_xx")
    assert structure["term_support_accuracy"] == 1
    assert structure["pde_support_recovery"] == 1
    assert structure["exact_symbolic_recovery"] == 1
    assert structure["exact_symbolic_recovery_semantics"].startswith("legacy alias")
    assert structure["coefficient_error"] == pytest.approx(0.1)

    wrong_structure = benchmark.pde_structure_metrics(
        "u_t = -u*u_x + 0.1*u_xx + u", "-u*u_x + 0.1*u_xx"
    )
    assert wrong_structure["exact_symbolic_recovery"] == 0


def test_pde_residual_reports_nonfinite_coverage_without_subset_score():
    x, t = np.linspace(0, 1, 8), np.linspace(0, 1, 9)
    field = np.broadcast_to(x[:, None], (len(x), len(t)))
    data = GridPDEDataset("singular", None, None, 0, x=x, t=t, usol=field, legacy=True)
    with np.errstate(all="ignore"):
        residual = benchmark.pde_equation_residual("u_t = 1/u", data)
    assert 0 < residual["equation_residual_points"] < residual["equation_residual_total_points"]
    assert residual["equation_residual_total_points"] > 0
    assert 0 < residual["equation_residual_finite_coverage"] < 1
    assert "equation_residual_mse" not in residual


def test_sindy_pde_adapter_aligns_derivatives_and_discovers_rhs():
    x = np.linspace(0, 1, 21)
    t = np.linspace(0, 1, 21)
    u = x[:, None] + 2 * t[None, :]
    data = GridPDEDataset("toy", None, None, 0, x=x, t=t, usol=u, legacy=True)
    X, y, names = benchmark.pde_sindy_problem(data, derivative_order=2, trim_boundary=2)
    assert names == ["u", "u_x", "u_xx"]
    assert X.shape == (17 * 17, 3)
    np.testing.assert_allclose(y, 2, atol=1e-12)

    case = make_case(model="sindy", dataset="kdv")
    row = benchmark.run_pde(data, case)
    assert row["expression"].startswith("u_t = ")
    assert row["equation_residual_mse"] == pytest.approx(0, abs=1e-8)


def test_deepmod_uses_time_first_and_aligned_targets(monkeypatch):
    class Capture:
        best_equation_ = "u_t = 1"

        def fit(self, X, y):
            np.testing.assert_allclose(y[:, 0], 10 * X[:, 1] + X[:, 0])

    monkeypatch.setattr(benchmark, "_new_model", lambda *args: Capture())
    row = benchmark.run_pde(scalar_grid(), make_case(model="deepmod", dataset="kdv"))
    assert row["expression"] == "u_t = 1"
    assert "test_mse" not in row


def test_multidimensional_fields_are_not_flattened():
    coords = {"x": np.linspace(0, 1, 5), "y": np.linspace(0, 1, 6), "t": np.linspace(0, 1, 7)}
    data = GridPDEDataset("two_fields", None, None, 0, coords=coords, usol=np.zeros((2, 5, 6, 7)))
    with pytest.raises(benchmark.SkipCase, match="scalar field"):
        benchmark.prepare_pde(data, "pdefind")
    assert benchmark.prepare_pde(data, "pdenet").usol.shape == (2, 5, 6, 7)


def test_generated_pde_instances_are_preserved():
    case = make_case(
        model="pdefind",
        dataset="pdeformer_sinus",
        config={"datasets": {"pdeformer_sinus": {"n_pde": 2, "n_x": 16, "n_t": 5}}},
    )
    instances = list(benchmark._load_instances(case))
    assert [name for name, _ in instances] == ["pdeformer_sinus/0000", "pdeformer_sinus/0001"]
    assert all(d.usol.shape == (16, 5) for _, d in instances)


def test_failure_does_not_erase_other_instances(monkeypatch, tmp_path):
    case = make_case(model="pdefind", dataset="pdeformer_sinus")
    monkeypatch.setattr(
        benchmark, "_load_instances", lambda c: iter([("bad", None), ("good", scalar_grid())])
    )

    def run(data, case):
        if data is None:
            raise ValueError("broken instance")
        return {"expression": "u_t = 1"}

    monkeypatch.setattr(benchmark, "run_pde", run)
    rows = benchmark.run_case(case, tmp_path)
    assert [r["status"] for r in rows] == ["error", "ok"]
    assert [r["instance"] for r in rows] == ["bad", "good"]
    # RSS backends are optional; an unavailable measurement stays null rather
    # than being fabricated as zero.
    assert all(r["peak_memory_mb"] is None or r["peak_memory_mb"] > 0 for r in rows)


def test_missing_dependency_is_reported(monkeypatch, tmp_path):
    case = make_case(model="pdefind", dataset="kdv")
    monkeypatch.setattr(benchmark, "_load_instances", lambda c: iter([("kdv", scalar_grid())]))

    def fail(*args):
        raise ModuleNotFoundError("optional_baseline_dependency")

    monkeypatch.setattr(benchmark, "_new_model", fail)
    rows = benchmark.run_case(case, tmp_path)
    assert rows[0]["status"] == "skipped"
    assert "optional_baseline_dependency" in rows[0]["error"]


def test_timeout_preserves_completed_targets(monkeypatch, tmp_path):
    case = make_case()
    directory = tmp_path / "cases" / case["name"]

    def timeout(*args, **kwargs):
        benchmark.write_json(
            directory / "partial.json", [{**benchmark._base_row(case), "status": "ok"}]
        )
        raise subprocess.TimeoutExpired("worker", 1)

    monkeypatch.setattr(benchmark.subprocess, "run", timeout)
    rows = benchmark.execute_case(case, tmp_path, timeout=1)
    assert [r["status"] for r in rows] == ["ok", "timeout"]


def test_resume_skips_successful_worker(monkeypatch, tmp_path):
    case = make_case()
    row = {**benchmark._base_row(case), "status": "ok"}
    benchmark.write_json(tmp_path / "cases" / case["name"] / "result.json", [row])

    def forbidden(*args, **kwargs):
        pytest.fail("A successful case should not launch another worker")

    monkeypatch.setattr(benchmark.subprocess, "run", forbidden)
    assert benchmark.execute_case(case, tmp_path, resume=True) == [row]


def test_strict_json_and_csv_keep_failed_rows(tmp_path):
    row = {**benchmark._base_row(make_case()), "test_mse": float("nan")}
    benchmark.save_summary(tmp_path, [row])
    text = (tmp_path / "benchmark_summary.json").read_text(encoding="utf-8")
    assert "NaN" not in text and json.loads(text)[0]["test_mse"] is None
    assert "error" in (tmp_path / "benchmark_summary.csv").read_text(encoding="utf-8-sig")
    assert (tmp_path / "benchmark_aggregate.json").is_file()
    assert (tmp_path / "benchmark_aggregate.csv").is_file()
    assert (tmp_path / "benchmark_seed_family.json").is_file()
    assert (tmp_path / "benchmark_paired.json").is_file()


def test_aggregate_reports_method_rates_runtime_memory_and_recovery():
    case = make_case()
    rows = [
        {
            **benchmark._base_row(case),
            "status": "ok",
            "runtime": 2.0,
            "peak_memory_mb": 100.0,
            "exact_recovery": 1,
            "exact_symbolic_recovery": 1,
        },
        {
            **benchmark._base_row(case),
            "status": "timeout",
            "runtime": 4.0,
            "peak_memory_mb": 120.0,
            "exact_recovery": 0,
            "exact_symbolic_recovery": 0,
        },
    ]
    group = next(
        item
        for item in benchmark.aggregate_results(rows)
        if item["task"] == "sr" and item["model"] == "gplearn"
    )
    assert group["success_rate"] == 0.5
    assert group["denominator_unit"] == "target_row"
    assert group["n_cases"] == 1 and group["n_target_rows"] == 2
    assert group["timeout_rate"] == 0.5
    assert group["runtime_mean_seconds"] == 3
    assert group["peak_memory_max_mb"] == 120
    assert group["exact_recovery_rate"] == 0.5
    assert group["exact_recovery_eligible"] == 2
    assert group["exact_recovery_parse_coverage"] == 1
    assert group["exact_symbolic_recovery_rate"] is None


def test_pde_recovery_denominator_includes_failed_predeclared_cases():
    case = make_case(model="pdefind", dataset="kdv")
    ok = {
        **benchmark._base_row(case),
        "status": "ok",
        "recovery_evaluator_status": "parsed",
        "pde_support_recovery": 1,
        "exact_symbolic_recovery": 1,
    }
    failed = {**benchmark._base_row(case), "status": "timeout"}
    group = next(
        item
        for item in benchmark.aggregate_results([ok, failed])
        if item["task"] == "pde" and item["model"] == "pdefind"
    )
    assert group["success_rate"] == 0.5
    assert group["pde_support_recovery_eligible"] == 2
    assert group["pde_support_parse_coverage"] == 1
    assert group["pde_support_recovery_rate"] == 0.5
    assert group["exact_symbolic_recovery_rate"] == 0.5


def test_evaluator_failure_is_not_counted_as_algorithm_failure():
    case = make_case()
    parsed = {**benchmark._base_row(case), "status": "ok", "exact_recovery": 1}
    evaluator_bug = {**benchmark._base_row(case), "status": "ok", "recovery_evaluator_status": "error"}
    group = next(item for item in benchmark.aggregate_results([parsed, evaluator_bug])
                 if item["task"] == "sr" and item["model"] == "gplearn")
    assert group["exact_recovery_parse_coverage"] == 0.5
    assert group["exact_recovery_rate"] is None
    assert group["exact_recovery_conditional_rate"] == 1
    assert group["exact_recovery_lower_bound"] == 0.5


def test_skipped_unsupported_case_is_not_recovery_eligible():
    case = make_case()
    group = next(item for item in benchmark.aggregate_results([
        {**benchmark._base_row(case), "status": "skipped"}
    ]) if item["task"] == "sr" and item["model"] == "gplearn")
    assert group["exact_recovery_eligible"] == 0


def test_aggregate_does_not_pool_protocol_versions():
    case = make_case()
    current = {**benchmark._base_row(case), "status": "ok", "exact_recovery": 1}
    legacy = {**current, "metric_protocol_version": None, "exact_recovery": 0}
    groups = [item for item in benchmark.aggregate_results([current, legacy])
              if item["task"] == "sr" and item["model"] == "gplearn"]
    assert {item["metric_protocol_version"] for item in groups} == {"2.0", "legacy/unknown"}


def test_coefficient_recovery_has_declared_tolerances():
    close = benchmark.pde_structure_metrics("u_t=-1.02*u*u_x+0.098*u_xx", "-u*u_x+0.1*u_xx")
    far = benchmark.pde_structure_metrics("u_t=-0.8*u*u_x+0.1*u_xx", "-u*u_x+0.1*u_xx")
    assert close["coefficient_recovery"] == 1
    assert far["coefficient_recovery"] == 0


def test_pic_candidate_parser_handles_burgers_kdv_and_rejects_unknown_terms():
    burgers = benchmark.pde_pic_candidate("u_t=-u*u_x+0.1*u_xx")
    kdv = benchmark.pde_pic_candidate("u_t=-u*u_x-0.0025*u_xxx")
    assert dict(zip(*burgers)) == {"u_xx": 0.1, "u*u_x": -1.0}
    assert dict(zip(*kdv)) == {"u_xxx": -0.0025, "u*u_x": -1.0}
    with pytest.raises(ValueError, match="does not support"):
        benchmark.pde_pic_candidate("u_t=sin(u)")


def test_opt_in_pic_runner_records_components_without_changing_baseline_status(
    tmp_path, monkeypatch
):
    import types
    import kd.metrics as metrics

    dataset = GridPDEDataset(
        equation_name="kdv", pde_data=None, x=np.linspace(-1, 1, 4),
        t=np.linspace(0, 1, 4), usol=np.arange(16.0).reshape(4, 4),
        domain={"x": (-1, 1), "t": (0, 1)}, epi=0.0, legacy=True,
    )
    reference = types.SimpleNamespace(cache_key="cache-key")
    prepared = types.SimpleNamespace(
        reference=reference, reference_train_rmse=0.1,
        reference_train_normalized_rmse=0.01,
    )
    seen = {}

    def fake_prepare(coordinates, values, **kwargs):
        seen["coordinates"] = coordinates
        seen["cache_dir"] = kwargs["cache_dir"]
        return prepared

    def fake_evaluate(_prepared, terms, **kwargs):
        seen["terms"] = tuple(terms)
        return metrics.PICResult(
            "ok", 0.02, 0.2, 0.1, np.ones((2, 2)),
            np.asarray(kwargs["original_coefficients"]), np.array([-0.003, -1.1]),
            np.array([-0.0026, -1.01]), {"reference_cache_hit": 1.0}, "cache-key",
        )

    monkeypatch.setattr(metrics, "prepare_torch_pic_reference", fake_prepare)
    monkeypatch.setattr(metrics, "evaluate_torch_pic", fake_evaluate)
    case = {
        "seed": 7,
        "evaluation": {"physics_informed_pic": {
            "enabled": True, "cache_dir": str(tmp_path), "backend": {
                "reference_epochs": 1, "pinn_epochs": 1, "nx": 4, "nt": 4,
                "n_windows": 2,
            },
        }},
    }
    result = {"expression": "u_t=-u*u_x-0.0025*u_xxx", "status": "ok"}
    benchmark._apply_physics_informed_pic(result, dataset, case)
    assert result["status"] == "ok" and result["pic_status"] == "ok"
    assert result["pic"] == pytest.approx(0.02)
    assert seen["terms"] == ("u_xxx", "u*u_x")
    assert seen["coordinates"].shape == (16, 2)
    assert result["pic_reference_cache_hit"] is True


def test_pic_evaluator_bug_is_isolated_from_baseline_row(tmp_path, monkeypatch):
    import kd.metrics as metrics

    dataset = GridPDEDataset(
        equation_name="burgers", pde_data=None, x=np.linspace(-1, 1, 4),
        t=np.linspace(0, 1, 4), usol=np.arange(16.0).reshape(4, 4),
        domain={"x": (-1, 1), "t": (0, 1)}, epi=0.0, legacy=True,
    )

    def broken_prepare(*args, **kwargs):
        raise KeyError("corrupt evaluator state")

    monkeypatch.setattr(metrics, "prepare_torch_pic_reference", broken_prepare)
    case = {"seed": 0, "evaluation": {"physics_informed_pic": {
        "cache_dir": str(tmp_path), "backend": {
            "reference_epochs": 1, "pinn_epochs": 1, "nx": 4, "nt": 4,
            "n_windows": 2,
        },
    }}}
    result = {"expression": "u_t=-u*u_x+0.1*u_xx", "status": "ok"}
    benchmark._apply_physics_informed_pic(result, dataset, case)
    assert result["status"] == "ok"
    assert result["pic_status"] == "evaluator_error"
    assert "corrupt evaluator state" in result["pic_message"]


def test_information_criteria_require_provenance():
    from kd.evaluation import information_criteria
    with pytest.raises(ValueError, match="declare log_likelihood"):
        information_criteria(n_observations=10, n_free_parameters=2, residual_sum_squares=1)
    result = information_criteria(n_observations=10, n_free_parameters=2,
                                  residual_sum_squares=1,
                                  likelihood_model="gaussian_mle_variance")
    assert result["information_criterion_k"] == 3
    assert result["information_criterion_variance_parameters"] == 1
    row = {"test_mse": 0.5, "n_test": 20}
    benchmark._apply_information_criteria(row, {"likelihood_model": "gaussian_mle_variance"})
    assert row["information_criterion_status"] == "abstained_missing_parameter_count"
    benchmark._apply_information_criteria(
        row, {"n_free_parameters": 2, "likelihood_model": "gaussian_mle_variance",
              "likelihood_data_role": "training", "parameter_estimation": "maximum_likelihood",
              "n_observations": 10, "residual_sum_squares": 1}
    )
    assert row["information_criterion_status"] == "ok" and row["information_criterion_k"] == 3


def test_information_criteria_never_relabel_heldout_error_as_training_mle():
    row = {"test_mse": 0.5, "n_test": 20}
    benchmark._apply_information_criteria(row, {"n_free_parameters": 2,
                                               "likelihood_model": "gaussian_mle_variance"})
    assert row["information_criterion_status"] == "abstained_missing_training_mle_provenance"
    assert "aic" not in row


def test_ode_core_predeclares_truth_and_family_and_e2e_abstains_without_assets():
    case = make_case(model="sindy", dataset="ode_core_population")
    assert case["family"] == "population" and len(case["ground_truth"]) == 2
    assert all("beta" not in equation for equation in case["ground_truth"])
    with pytest.raises(benchmark.SkipCase, match="checkpoint_path"):
        benchmark._new_model("e2e", {})


def test_reporting_uses_exact_pairs_and_seed_denominators():
    from kd.evaluation import paired_comparison, seed_family_summary
    rows = [
        {"model": "a", "dataset": "d", "instance": "i", "target": "y", "seed": 0, "family": "f", "m": 1},
        {"model": "b", "dataset": "d", "instance": "i", "target": "y", "seed": 0, "family": "f", "m": 3},
        {"model": "a", "dataset": "d2", "instance": "j", "target": "y", "seed": 1, "family": "f", "m": 9},
    ]
    paired = paired_comparison(rows, "m", "a", "b")
    assert paired["n_pairs"] == 1 and paired["mean_difference_a_minus_b"] == -2
    summary = seed_family_summary(rows, "m")
    a_summary = next(item for item in summary if item["model"] == "a")
    assert a_summary["n_seeds"] == 2 and a_summary["n_target_rows"] == 2
    with pytest.raises(ValueError, match="duplicate pairing key"):
        paired_comparison([rows[0], dict(rows[0]), rows[1]], "m", "a", "b")


def test_shared_pde_rollout_requires_declared_boundary_conditions():
    from kd.evaluation.pde_rollout import rollout, validate_solver
    x, times = np.linspace(0, 1, 8), np.linspace(0, 0.2, 3)
    initial = np.ones_like(x)
    rhs = lambda state, coordinates, bc: np.zeros_like(state)
    with pytest.raises(ValueError, match="boundary_conditions"):
        rollout(rhs, initial, x, times, boundary_conditions=None)
    result = validate_solver(rhs, initial, x, times,
                             boundary_conditions={"type": "periodic"},
                             reference=np.ones((len(x), len(times))), nrmse_tolerance=1e-10)
    assert result["validation_passed"] is True


@pytest.mark.parametrize("equation,decay", [("-1.25*u_x", 0.0), ("0.08*u_xx", 0.08)])
def test_validated_spectral_equation_rollout_on_periodic_analytic_solution(equation, decay):
    from kd.dataset import GridPDEDataset
    x = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    t = np.linspace(0, 0.2, 9)
    if decay:
        u = np.sin(x[:, None]) * np.exp(-decay * t[None, :])
    else:
        u = np.sin(x[:, None] - 1.25 * t[None, :])
    data = GridPDEDataset("analytic", None, None, 0, x=x, t=t, usol=u, legacy=True)
    metadata = {"boundary_conditions": {"type": "periodic"},
                "endpoint_convention": "periodic_endpoint_excluded",
                "spatial_discretization": "spectral_fft",
                "solver": {"method": "RK45", "rtol": 1e-8, "atol": 1e-10, "max_step": None},
                "reference_nrmse_tolerance": 1e-5}
    result = benchmark.symbolic_pde_rollout_metrics(equation, data, 4, metadata, equation)
    assert result["equation_rollout_status"] == "ok"
    assert result["equation_rollout_nrmse"] < 1e-5


def test_equation_rollout_abstains_without_metadata_or_failed_reference():
    data = scalar_grid()
    assert benchmark.symbolic_pde_rollout_metrics("1", data, 7)["equation_rollout_status"] == "abstained_missing_metadata"
    metadata = {"boundary_conditions": {"type": "periodic"},
                "endpoint_convention": "periodic_endpoint_excluded",
                "spatial_discretization": "spectral_fft",
                "solver": {"method": "RK45", "rtol": 1e-8, "atol": 1e-10, "max_step": None},
                "reference_nrmse_tolerance": 1e-12}
    result = benchmark.symbolic_pde_rollout_metrics("1", data, 7, metadata, "0")
    assert result["equation_rollout_status"] == "abstained_reference_validation_failed"


def test_multitarget_setup_failure_preserves_target_denominator(monkeypatch, tmp_path):
    case = make_case(model="gplearn", dataset="Koza-2")
    case.update(task="ode", ground_truth=["v", "-h"])
    monkeypatch.setattr(benchmark, "_load_instances", lambda c: iter([("ode", object())]))
    monkeypatch.setattr(benchmark, "regression_problems",
                        lambda *args: (_ for _ in ()).throw(ValueError("bad trajectories")))
    rows = benchmark.run_case(case, tmp_path)
    assert len(rows) == 2
    assert [row["ground_truth_expression"] for row in rows] == ["v", "-h"]
    assert all(row["recovery_eligible"] for row in rows)


def test_worker_timeout_preserves_remaining_ode_target_denominator(monkeypatch, tmp_path):
    case = make_case(model="sindy", dataset="ode_core_chaotic")
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("mock-worker", 10)
    monkeypatch.setattr(benchmark.subprocess, "run", timeout)
    rows = benchmark.execute_case(case, tmp_path, timeout=10)
    assert len(rows) == 3
    assert {row["target"] for row in rows} == {"d(x)/dt", "d(y)/dt", "d(z)/dt"}
    assert all(row["status"] == "timeout" and row["recovery_eligible"] for row in rows)


def test_partial_worker_timeout_does_not_duplicate_completed_targets(monkeypatch, tmp_path):
    case = make_case(model="sindy", dataset="ode_core_oscillator")
    directory = tmp_path / "cases" / case["name"]
    def timeout(*args, **kwargs):
        completed = benchmark._base_row(case, case["dataset"], "d(x)/dt")
        completed.update(status="ok", ground_truth_expression=case["ground_truth"][0], runtime=0.1)
        benchmark.write_json(directory / "partial.json", [completed])
        raise subprocess.TimeoutExpired("mock-worker", 10)
    monkeypatch.setattr(benchmark.subprocess, "run", timeout)
    rows = benchmark.execute_case(case, tmp_path, timeout=10)
    assert [(row["target"], row["status"]) for row in rows] == [("d(x)/dt", "ok"), ("d(v)/dt", "timeout")]


def test_imports_do_not_eagerly_load_optional_models():
    code = "import run_benchmark, kd.dataset, kd.model, sys; assert 'deepymod' not in sys.modules; assert 'kd.model.kd_dscv' not in sys.modules; import kd.viz"
    subprocess.run([sys.executable, "-c", code], cwd=benchmark.ROOT, check=True, timeout=30)


def test_failed_evaluation_keeps_expression():
    result = benchmark._score_regression({"expression": "log(x1)"}, [1, 2], lambda: [1, np.nan])
    assert result["status"] == "error"
    assert result["expression"] == "log(x1)"
    assert "Prediction failed" in result["error"]


def test_evaluator_exception_does_not_mark_algorithm_failure(monkeypatch):
    monkeypatch.setattr(benchmark, "regression_metrics",
                        lambda *args: (_ for _ in ()).throw(RuntimeError("evaluator bug")))
    result = benchmark._score_regression({"expression": "x1", "status": "ok"}, [1], lambda: [1])
    assert result["status"] == "ok"
    assert result["evaluation_status"] == "evaluator_error"


def test_pyoperon_adapter_extracts_named_expression(monkeypatch):
    case = make_case(model="pyoperon", dataset="Koza-2")
    captured = {}

    class Capture:
        model_ = object()

        def fit(self, X, y):
            captured["fit_shape"] = (X.shape, y.shape)

        def get_model_string(self, model, precision, names):
            assert model is self.model_
            captured["format"] = (precision, names)
            return "x1 * x1"

        def predict(self, X):
            return X[:, 0] ** 2

    monkeypatch.setattr(benchmark, "_new_model", lambda *args: Capture())
    problem = {
        "X_train": np.array([[0.0], [1.0], [2.0]]),
        "y_train": np.array([0.0, 1.0, 4.0]),
        "X_test": np.array([[3.0], [4.0]]),
        "y_test": np.array([9.0, 16.0]),
        "variable_names": ["x1"],
        "target": "y",
        "split": "holdout",
        "metadata": {},
        "ground_truth": "x1**2",
    }
    row = benchmark.run_regression(problem, case)

    assert row["test_mse"] == pytest.approx(0)
    assert row["exact_recovery"] == 1
    assert captured == {"fit_shape": ((3, 1), (3,)), "format": (8, ["x1"])}


def test_spr_external_data_never_load_builtin_field(monkeypatch):
    from types import SimpleNamespace

    case = make_case(model="spr", dataset="burgers")
    created = []

    class Capture:
        config_pinn = {}
        config_task = {"dataset": "Burgers2"}

        def make_pinn_model(self):
            assert self.config_task["dataset"] is None
            result = SimpleNamespace()
            created.append(result)
            return result

        def import_dataset(self, *args, **kwargs):
            self.make_pinn_model()

        def train(self, **kwargs):
            self.make_pinn_model()  # Real SPR rebuilds its PINN in train().
            assert self.config_task["dataset"] == "Burgers2"
            return {"expression": "u_t = 1"}

    monkeypatch.setattr(benchmark, "_new_model", lambda *args: Capture())
    benchmark.run_pde(scalar_grid(), case)
    assert len(created) == 2
    assert all(p.pretrain_epoch == case["run_params"]["spr_pretrain_epochs"] for p in created)
