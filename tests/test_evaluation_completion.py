"""Independent analytic E1/E2 checks and frozen E3 denominator/provenance checks."""

import copy
import hashlib
import json

import numpy as np
import pytest

import run_benchmark as benchmark
from kd.dataset import ODEDataset
from kd.evaluation.frozen_report import generate, load_frozen, tables
from kd.evaluation.ode_rollout import evaluate, numeric_expression
from kd.evaluation.reference import reference_cases, run


def oscillator():
    times = np.linspace(0, 4, 61)
    trajectories = [{"id": f"traj_{i}", "t": times,
                     "state": np.array([np.cos(times + i / 10), -np.sin(times + i / 10)]),
                     "params": {}} for i in range(4)]
    ds = ODEDataset("analytic_oscillator", trajectories, ["x", "v"])
    ds.sym_true = ["dx/dt = v", "dv/dt = -x"]
    return ds


def ode_case():
    catalog, _ = benchmark.dataset_catalog()
    return benchmark.build_experiments(catalog, tasks=["ode"], models=["sindy"],
                                       datasets=["ode_core_oscillator"])[0]


def analytic_rows(case):
    return [{**benchmark._base_row(case, case["dataset"], f"d({name})/dt"),
             "status": "ok", "expression": expression}
            for name, expression in zip(("x", "v"), ("v", "-x"))]


def test_coupled_ode_matches_independent_analytic_solution_and_whole_holdout():
    ds, case = oscillator(), ode_case()
    result = evaluate(ds, analytic_rows(case), case)
    for value in result.values():
        assert value["equation_rollout_status"] == "ok"
        assert value["equation_rollout_nrmse"] < 1e-6
        assert not set(value["equation_rollout_train_trajectories"]) & set(value["equation_rollout_test_trajectories"])
        assert value["equation_rollout_trajectories"][0]["n_points"] == 60


@pytest.mark.parametrize("expression,status", [("100*x", "diverged"), ("1/(x-x)", "invalid_rhs"), ("unknown", "unparsed_rhs")])
def test_rollout_failures_never_get_a_finite_success_metric(expression, status):
    case = ode_case()
    rows = analytic_rows(case)
    rows[0]["expression"] = expression
    result = evaluate(oscillator(), rows, case)
    assert {r["equation_rollout_status"] for r in result.values()} == {status}
    assert all(r["equation_rollout_nrmse"] is None for r in result.values())


def test_failed_rhs_blocks_entire_system_without_hiding_target():
    case = ode_case()
    rows = analytic_rows(case)
    rows[0]["status"] = "timeout"
    result = evaluate(oscillator(), rows, case)
    assert len(result) == 2
    assert all(r["equation_rollout_status"] == "abstained_incomplete_system" for r in result.values())


def test_wrong_reference_is_rejected_and_missing_parameters_are_not_invented():
    ds, case = oscillator(), ode_case()
    ds.sym_true = ["dx/dt=0", "dv/dt=0"]
    result = evaluate(ds, analytic_rows(case), case)
    assert all(r["equation_rollout_status"] == "abstained_reference_validation_failed" for r in result.values())
    ds.sym_true = ["dx/dt=v", "dv/dt=-omega*x"]
    result = evaluate(ds, analytic_rows(case), case)
    assert all(r["equation_rollout_status"] == "abstained_reference_validation_failed" for r in result.values())


@pytest.mark.parametrize("text", ["__import__('os').system('echo forbidden')", "x.__class__", "[x][0]"])
def test_rhs_parser_rejects_executable_syntax(text):
    with pytest.raises((ValueError, SyntaxError)):
        numeric_expression(text, ["x"])


def test_stiff_reference_uses_radau():
    from kd.dataset._ode_core import generate_ode_core

    ds = generate_ode_core("robertson", n_trajectories=2, n_points=41)
    case = ode_case()
    rows = [{"target": f"d({s})/dt", "expression": e, "status": "ok"}
            for s, e in zip(ds.state_vars, ds.sym_true)]
    values = evaluate(ds, rows, case)
    assert all(v["equation_rollout_status"] == "ok" for v in values.values())
    assert all(v["equation_rollout_integrator"] == "Radau" for v in values.values())


def test_reference_uses_shared_derivative_score_without_training(monkeypatch):
    case = reference_cases([ode_case()])[0]
    problems = list(benchmark.regression_problems(oscillator(), case))
    monkeypatch.setattr(benchmark, "_new_model", lambda *a: pytest.fail("reference constructed a model"))
    result = run(oscillator(), problems[0], case)
    expected = benchmark.regression_metrics(problems[0]["y_test"], problems[0]["X_test"][:, 1])
    assert result["test_mse"] == expected["test_mse"]
    assert result["test_mse"] > 0  # observed finite-difference derivative has a real error floor
    assert result["reference_floor_eligible"]
    assert result["ranking_eligible"] is False
    assert result["recovery_eligible"] is False


def test_missing_reference_has_explicit_abstention():
    case = reference_cases([ode_case()])[0]
    problem = next(benchmark.regression_problems(oscillator(), case))
    problem["ground_truth"] = None
    result = run(oscillator(), problem, case)
    assert result["status"] == "skipped"
    assert result["reference_status"] == "abstained_missing_reference"


def test_reference_rows_do_not_change_method_aggregates_or_pairs(tmp_path):
    case = ode_case()
    methods = analytic_rows(case)
    controls = [{**r, "model": "reference", "ranking_eligible": False} for r in methods]
    assert benchmark.aggregate_results(methods + controls) == benchmark.aggregate_results(methods)
    benchmark.save_summary(tmp_path, methods + controls)
    assert len(json.loads((tmp_path / "benchmark_reference.json").read_text())) == 2
    assert all(r["model"] != "reference" for r in json.loads((tmp_path / "benchmark_seed_family.json").read_text()))


def frozen_fixture(directory, *, seeds=(0, 1), omit=False):
    cases, rows = [], []
    for seed in seeds:
        case = ode_case()
        case.update(name=f"case_s{seed}", seed=seed)
        cases.append(case)
        plan = []
        for i, row in enumerate(analytic_rows(case)):
            spec = {"target_id": f"i0000_t{i:04d}", "target": row["target"], "instance": row["instance"]}
            plan.append(spec)
            row.update(target_id=spec["target_id"], runtime=2.0 + seed,
                       test_nrmse=0.1 + seed, execution_policy=benchmark.execution_policy(5))
            if not (omit and seed == seeds[-1] and i == 1):
                rows.append(row)
        path = directory / "cases" / case["name"]
        path.mkdir(parents=True)
        (path / "targets.json").write_text(json.dumps(plan))
    manifest = {"cases": cases, "execution_policy": benchmark.execution_policy(5)}
    path = directory / "benchmark_manifest.json"
    path.write_text(json.dumps(manifest))
    (directory / "benchmark_summary.json").write_text(json.dumps(rows))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_denominator_includes_missing_targets_and_rejects_hash_or_duplicates(tmp_path):
    digest = frozen_fixture(tmp_path, omit=True)
    _, rows, _ = load_frozen(tmp_path, digest)
    assert len(rows) == 4
    assert sum(r["status"] == "missing" for r in rows) == 1
    main, intervals, _ = tables(rows, digest)
    assert main[0]["success_rate"] == 0.75
    success = next(r for r in intervals if r["metric"] == "success_rate")
    assert success["bootstrap_unit"] == "whole_seed"
    assert success["ci_status"] == "ok"
    with pytest.raises(ValueError, match="SHA256"):
        load_frozen(tmp_path, "0" * 64)
    path = tmp_path / "benchmark_summary.json"
    summary = json.loads(path.read_text())
    path.write_text(json.dumps(summary + [summary[0]]))
    with pytest.raises(ValueError, match="Duplicate"):
        load_frozen(tmp_path, digest)


def test_one_seed_has_no_fake_confidence_interval(tmp_path):
    digest = frozen_fixture(tmp_path, seeds=(0,))
    _, rows, _ = load_frozen(tmp_path, digest)
    _, intervals, _ = tables(rows, digest)
    assert all(r["ci_low"] is None and r["ci_high"] is None for r in intervals)


def test_report_artifacts_and_results_freeze_guard(tmp_path):
    digest = frozen_fixture(tmp_path)
    summary = tmp_path / "benchmark_summary.json"
    original = summary.read_bytes()
    output = generate(tmp_path, digest)
    assert summary.read_bytes() == original
    assert (output / "main_table.csv").is_file()
    assert list(output.glob("*/status_stacked.png"))
    assert list(output.glob("*/pareto_test_nrmse.pdf"))
    rows = json.loads(original)
    rows[0]["runtime"] = 99
    summary.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="inputs changed"):
        generate(tmp_path, digest)


def test_reference_control_is_deduplicated_across_methods():
    case = ode_case()
    other = copy.deepcopy(case)
    other.update(model="pysindy", name="other")
    assert len(reference_cases([case, other])) == 1


def test_equation_reference_scores_exact_coefficients_without_optimizer(monkeypatch):
    from kd.dataset import EquationTarget, GridPDEDataset
    from kd.evaluation.equation import problems
    from kd.model.kd_equation import SparseEquationModel

    x, t = np.linspace(-1, 1, 31), np.linspace(0, 2, 41)
    ds = GridPDEDataset("analytic", None, None, 0.0, coords={"x": x, "t": t},
                        usol=(np.sin(x[:, None]) * np.exp(-t)[None, :])[None], time_var="t")
    ds.state_vars = ["u"]
    ds.equation_targets = [EquationTarget("u_t")]
    ds.protocol_metadata = {"source_reference": "-u"}
    catalog, _ = benchmark.dataset_catalog()
    template = benchmark.build_experiments(catalog, models=["pdefind_nd"], datasets=["wave"])[0]
    case = reference_cases([template])[0]
    monkeypatch.setattr(SparseEquationModel, "fit", lambda *a, **k: pytest.fail("reference optimized"))
    result = run(ds, next(problems(ds, case)), case)
    assert result["metric_target"] == "observed_equation_residual"
    assert result["equation_residual_nrmse"] < 0.01
    assert result["ranking_eligible"] is False


def test_green_reference_uses_whole_forcing_holdout_and_training_scale():
    from kd.dataset._green import GreenOperatorDataset
    from kd.evaluation.green import forward_metrics, problems

    x, xi = np.linspace(0, 1, 11), np.linspace(0, 1, 17)
    forcing = np.broadcast_to(np.arange(1, 16, dtype=float)[None], (17, 15)).copy()
    hom = x * 0
    kernel = x[:, None] + xi[None]
    response = (x[:, None] + 0.5) * forcing[:1]
    ds = GreenOperatorDataset("analytic", x, xi, forcing, response, hom)
    ds.reference = "x+y"
    catalog, _ = benchmark.dataset_catalog()
    template = benchmark.build_experiments(catalog, models=["green_basis"], datasets=["green_laplace"], tasks=["green"])[0]
    case = reference_cases([template])[0]
    problem = next(problems(ds, case))
    result = run(ds, problem, case)
    scale = np.sqrt(np.mean(ds.training_view(problem["split"])["response"] ** 2))
    expected = forward_metrics(ds, kernel, problem["split"]["test"], scale)
    assert result["test_mse"] == pytest.approx(expected["test_mse"])
    assert result["normalization_train_scale"] == scale
    assert result["reference_floor_eligible"] is False


def test_frozen_budgets_and_protocols_cannot_be_relabelled(tmp_path):
    digest = frozen_fixture(tmp_path)
    path = tmp_path / "benchmark_summary.json"
    rows = json.loads(path.read_text())
    rows[0]["execution_policy"]["timeout"] = 100
    path.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="budget"):
        load_frozen(tmp_path, digest)


def test_system_score_timeout_is_terminal_and_does_not_change_fits(monkeypatch, tmp_path):
    case = ode_case()
    rows = analytic_rows(case)
    benchmark.write_json(tmp_path / "partial.json", rows)
    monkeypatch.setattr(benchmark, "supervise", lambda *a, **k: {"timeout_phase": "score", "score_seconds": 1})
    result = benchmark._score_ode_outputs(tmp_path, case, rows, benchmark.execution_policy(score_timeout=1))
    assert all(r["status"] == "ok" and r["equation_rollout_status"] == "score_timeout" for r in result)
    monkeypatch.setattr(benchmark, "supervise", lambda *a, **k: pytest.fail("retried scoring timeout"))
    resumed = benchmark._score_ode_outputs(tmp_path, case, rows, benchmark.execution_policy(score_timeout=1))
    assert resumed == result


def test_system_score_resource_limit_keeps_first_fit_results(monkeypatch, tmp_path):
    case = ode_case()
    rows = analytic_rows(case)
    benchmark.write_json(tmp_path / "partial.json", rows)
    monkeypatch.setattr(benchmark, "supervise", lambda *a, **k: {
        "resource_limit": "rss", "score_seconds": 1, "cpu_seconds": 0.5, "gpu_occupied_seconds": 0})
    result = benchmark._score_ode_outputs(tmp_path, case, rows, benchmark.execution_policy(score_timeout=1))
    assert all(r["status"] == "ok" and r["equation_rollout_status"] == "resource_limit" for r in result)
    assert all(r["system_score_cpu_seconds"] == 0.5 for r in result)
    monkeypatch.setattr(benchmark, "supervise", lambda *a, **k: pytest.fail("retried RSS limit"))
    assert benchmark._score_ode_outputs(tmp_path, case, rows, benchmark.execution_policy(score_timeout=1)) == result


def test_frozen_resource_policy_cannot_be_relabelled(tmp_path):
    from kd.benchmark_resources import policy_hash, slot_policy

    frozen_fixture(tmp_path)
    manifest_path = tmp_path / "benchmark_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    resources = slot_policy("cpu")
    for case in manifest["cases"]:
        case["resource_policy"] = resources
    manifest_path.write_text(json.dumps(manifest))
    summary_path = tmp_path / "benchmark_summary.json"
    rows = json.loads(summary_path.read_text())
    for row in rows:
        row["execution_policy"]["resource_policy"] = resources
        row.update(resource_policy=resources, resource_policy_hash=policy_hash(resources))
    summary_path.write_text(json.dumps(rows))
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    load_frozen(tmp_path, digest)
    rows[0]["resource_policy_hash"] = "changed"
    summary_path.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="resource policy"):
        load_frozen(tmp_path, digest)


def test_reference_worker_failure_remains_recovery_ineligible(tmp_path):
    case = reference_cases([ode_case()])[0]
    spec = {"instance": case["dataset"], "target": "d(x)/dt", "target_id": "i0000_t0000", "ground_truth": "v"}
    row = benchmark._worker_failure(case, spec, {"timeout_phase": "score"}, tmp_path / "run.log")
    assert row["status"] == "timeout"
    assert row["recovery_eligible"] is False and row["reference_floor_eligible"] is False
