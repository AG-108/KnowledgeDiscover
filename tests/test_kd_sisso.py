"""SISSO adapter contracts that run without the optional native dependency."""

from types import SimpleNamespace

import numpy as np
import pytest

import run_benchmark as benchmark
from kd.benchmark_resources import slot_for_model
from kd.model.kd_sisso import SISSOModel


def test_native_coefficients_and_column_names_are_retained(monkeypatch):
    calls = {}

    class NativeRegressor:
        def __init__(self, **kwargs):
            calls.update(kwargs)

        def fit(self, X, y, columns):
            calls["columns"] = columns
            self.solver = SimpleNamespace(models=[[SimpleNamespace(
                coefs=[[2.0, -3.0, 1.25]],
                feats=[SimpleNamespace(expr="(v^2)", eval_many=lambda data: data["v"] ** 2),
                       SimpleNamespace(expr="x", eval_many=lambda data: data["x"])],
            )]])

        def predict(self, X):
            return 1.25 + 2 * X[:, 0] ** 2 - 3 * X[:, 1]

    monkeypatch.setattr(SISSOModel, "check_available", staticmethod(lambda: NativeRegressor))
    model = SISSOModel(n_dim=2, random_state=7)
    X = np.array([[1, 2], [3, 4]], dtype=float)
    assert model.fit(X, [0, 1], variable_names=["v", "x"]) is model
    assert calls["columns"] == ["v", "x"]
    assert calls["nlopt_seed"] == 7
    assert calls["allowed_param_ops"] == []
    metrics = benchmark.expression_metrics(model.best_expression_, "1.25+2*v**2-3*x", ["v", "x"])
    assert metrics["exact_recovery"] == 1
    assert model.predict(X) == pytest.approx([-2.75, 7.25])
    assert model.stop_reason_ == "budget_exhausted"


@pytest.mark.parametrize("X,y,names", [
    ([[1], [2]], [[1, 2], [3, 4]], None),
    ([[1], [np.nan]], [1, 2], None),
    ([[1], [2]], [1], None),
    ([[1, 2], [3, 4]], [1, 2], ["x", "x"]),
    ([[1], [2]], [1, 2], ["x (m)"]),
])
def test_bad_training_data_is_rejected_before_native_import(X, y, names):
    with pytest.raises(ValueError):
        SISSOModel().fit(X, y, variable_names=names)


def test_predict_requires_fit():
    with pytest.raises(RuntimeError, match="fit"):
        SISSOModel().predict([[1]])


def test_dimension_selection_uses_training_predictions_only(monkeypatch):
    feature_x = SimpleNamespace(expr="x1", eval_many=lambda data: data["x1"])
    feature_square = SimpleNamespace(expr="x1^2", eval_many=lambda data: data["x1"] ** 2)

    class NativeRegressor:
        def __init__(self, **kwargs):
            pass

        def fit(self, X, y, columns):
            self.solver = SimpleNamespace(models=[
                [SimpleNamespace(coefs=[[2., 1.]], feats=[feature_x])],
                [SimpleNamespace(coefs=[[2., 1e-15, 1.]], feats=[feature_x, feature_square])],
            ])

    monkeypatch.setattr(SISSOModel, "check_available", staticmethod(lambda: NativeRegressor))
    X = np.linspace(-1, 1, 20)[:, None]
    model = SISSOModel(n_dim=2, model_selection="training_tolerance")
    model.fit(X, 2*X[:, 0]+1)
    assert model.selected_dimension_ == 1
    np.testing.assert_allclose(model.predict(X), 2*X[:, 0]+1)


def test_exhausted_native_feature_pool_is_capped_inside_one_fit(monkeypatch):
    selected = []

    class NativeRegressor:
        def __init__(self, **kwargs):
            self.n_sis_select = kwargs["n_sis_select"]

        def fit(self, X, y, columns):
            selected.append(self.n_sis_select)
            if self.n_sis_select > 3:
                raise RuntimeError("SIS went through all features and did not select enough (3 not 20).")
            self.solver = SimpleNamespace(models=[[SimpleNamespace(
                coefs=[[2., 1.]], feats=[SimpleNamespace(expr="x1", eval_many=lambda data: data["x1"])],
            )]])

    monkeypatch.setattr(SISSOModel, "check_available", staticmethod(lambda: NativeRegressor))
    model = SISSOModel(n_sis_select=20).fit([[0.], [1.]], [1., 3.])
    assert selected == [20, 3]
    assert model.effective_n_sis_select_ == 3
    assert model.n_sis_select == 20


def test_declared_sisso_grammar_is_mapped_without_reference_inspection():
    params = benchmark._dataset_search_space(
        {"use_dataset_function_set": True},
        {"benchmark_function_set": ["add", "mul", "n2", "sqrt", "const"]},
        "sisso", reference_expression="tan(x1)",
    )
    assert params["allowed_ops"] == ["add", "mult", "sq", "sqrt"]


def test_registry_full_profile_and_cpu_slot():
    catalog, _ = benchmark.dataset_catalog()
    cases = benchmark.build_experiments(
        catalog, models=["sisso"], datasets=["Koza-2", "ode_core_damped_oscillator", "burgers"],
        profile="full", seeds=[3, 4],
    )
    assert {case["task"] for case in cases} == {"sr", "ode"}
    assert len(cases) == 4
    assert [case["model_params"]["random_state"] for case in cases] == [3, 4, 3, 4]
    assert all(case["model_params"]["max_rung"] == 2 for case in cases)
    assert slot_for_model("sisso") == "cpu"
    with pytest.raises(ValueError, match="does not support CUDA"):
        benchmark.build_experiments(catalog, models=["sisso"], datasets=["Koza-2"],
                                    devices={"sisso": "cuda:0"})


def test_preflight_checks_native_bindings(monkeypatch):
    catalog, _ = benchmark.dataset_catalog()
    cases = benchmark.build_experiments(catalog, models=["sisso"], datasets=["Koza-2"])

    def missing():
        raise ImportError("native SISSO missing")

    monkeypatch.setattr(SISSOModel, "check_available", staticmethod(missing))
    report = benchmark.check_compatibility(cases)
    assert report["runtime_imports"]["sisso"]["status"] == "error"
