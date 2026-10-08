"""Bounded SISSO++ verification included in normal full pytest discovery."""

import json

import numpy as np
import pytest
import sympy as sp

import run_benchmark as benchmark
from kd.model.kd_sisso import SISSOModel

pytestmark = pytest.mark.native_sisso


@pytest.mark.parametrize("fix_intercept", [False, True])
def test_native_linear_fit_and_heldout_predictions(tmp_path, fix_intercept):
    X = np.linspace(-2, 2, 24)[:, None]
    bias = 0 if fix_intercept else 1.25
    model = SISSOModel(max_rung=0, fix_intercept=fix_intercept, workdir=tmp_path / "native")
    model.fit(X, bias + 2 * X[:, 0], variable_names=["v"])
    heldout = np.array([[-1.7], [0.123], [3.1]])
    assert model.predict(heldout) == pytest.approx(bias + 2 * heldout[:, 0], abs=1e-12)
    assert benchmark.expression_metrics(model.best_expression_, f"{bias}+2*v", ["v"])["exact_recovery"] == 1


def test_native_nonlinear_expression_matches_predictions(tmp_path):
    X = np.linspace(0.2, 2, 24)[:, None]
    model = SISSOModel(max_rung=1, allowed_ops=["sq"], workdir=tmp_path / "native")
    model.fit(X, 1.2 + 3 * X[:, 0] ** 2)
    heldout = np.array([[-1.7], [0.123], [3.1]])
    assert model.predict(heldout) == pytest.approx(1.2 + 3 * heldout[:, 0] ** 2, abs=1e-12)
    assert benchmark.expression_metrics(model.best_expression_, "1.2+3*x1**2", ["x1"])["exact_recovery"] == 1


def test_full_profile_fits_two_descriptors(tmp_path):
    params = benchmark.load_baseline_configs()["sisso"]["model"].copy()
    params.update(benchmark.load_baseline_configs()["sisso"]["profiles"]["full"]["model"])
    params["workdir"] = tmp_path / "native"
    X = np.random.default_rng(0).uniform(0.2, 2, (32, 2))
    y = 1.25 + 2 * X[:, 0] - 0.3 * X[:, 1]
    model = SISSOModel(**params).fit(X, y)
    assert len(model.feature_names_) == 2
    heldout = np.array([[0.3, 1.9], [1.5, 0.8]])
    symbolic = benchmark.expression_sympy(model.best_expression_, ["x1", "x2"])
    evaluate = sp.lambdify(sp.symbols("x1 x2"), symbolic, "numpy")
    assert np.isfinite(model.predict(heldout)).all()
    assert model.predict(heldout) == pytest.approx(evaluate(*heldout.T), abs=1e-10)


def test_supervised_sr_and_ode_targets(tmp_path):
    output = tmp_path / "benchmark"
    assert benchmark.main([
        "--models", "sisso", "--datasets", "Koza-2", "ode_core_damped_oscillator",
        "--timeout", "30", "--output-dir", str(output),
    ]) == 0
    rows = json.loads((output / "benchmark_summary.json").read_text())
    assert len(rows) == 3
    assert all(row["status"] == "ok" and row["stop_reason"] == "budget_exhausted" for row in rows)
    assert all(row["stop_reason_source"] == "sissopp_solver_return" for row in rows)
    assert all(np.isfinite(row["test_mse"]) and row["expression"] for row in rows)
    manifest = json.loads((output / "benchmark_manifest.json").read_text())
    assert manifest["identity"]["environment"]["native_runtime"]["sissopp"]["extension_sha256"]


def test_native_search_is_subject_to_supervisor_hard_timeout(tmp_path):
    output = tmp_path / "timeout"
    assert benchmark.main([
        "--models", "sisso", "--datasets", "Koza-2", "--profile", "full",
        "--timeout", "0.000001", "--timeout-grace", "0", "--output-dir", str(output),
    ]) == 1
    rows = json.loads((output / "benchmark_summary.json").read_text())
    assert len(rows) == 1
    assert rows[0]["status"] == "timeout" and rows[0]["stop_reason"] == "hard_timeout"
    assert rows[0]["timeout_phase"] == "fit"
