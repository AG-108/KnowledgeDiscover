import numpy as np
import pytest
import sympy as sp

from kd.model.kd_wsindy import WSINDyPDEModel


def traveling_wave(speed=0.7):
    x = np.linspace(0, 2 * np.pi, 161)
    t = np.linspace(0, 2.0, 121)
    u = np.sin(x[:, None] - speed * t[None, :])
    return u, x, t


def test_wsindy_recovers_advection_from_convolutional_weak_system():
    u, x, t = traveling_wave()
    model = WSINDyPDEModel(
        terms=[(1, 1)],
        m_x=20,
        m_t=15,
        test_function_power=8,
        threshold=0,
    ).fit(u, x, t)
    np.testing.assert_allclose(model.coefficients_, [-0.7], atol=2e-6)
    assert model.weak_relative_residual_ < 1e-10
    assert model.method_provenance_["method"] == "WSINDy-PDE"
    assert 0 < model.n_weak_samples_ <= model.max_weak_samples
    assert model.query_stride_ == {"x": 2, "t": 2}


def test_wsindy_never_calls_a_pointwise_field_derivative(monkeypatch):
    u, x, t = traveling_wave()

    def forbidden(*args, **kwargs):
        raise AssertionError("observed field derivatives are forbidden")

    monkeypatch.setattr(np, "gradient", forbidden)
    WSINDyPDEModel(
        terms=[(1, 1)], m_x=12, m_t=10, test_function_power=6, threshold=0
    ).fit(u, x, t)


def test_mstls_path_records_model_selection_diagnostics():
    u, x, t = traveling_wave()
    model = WSINDyPDEModel(
        terms=[(0, 0), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1)],
        m_x=20,
        m_t=15,
        test_function_power=8,
        thresholds=[0.0, 0.01, 0.05, 0.1, 0.2],
    ).fit(u, x, t)
    assert model.selected_lambda_ in model.lambda_path_
    assert len(model.loss_path_) == 5
    assert len(model.projection_cost_path_) == 5
    assert len(model.complexity_cost_path_) == 5
    assert np.count_nonzero(model.coefficients_) < len(model.coefficients_)
    assert abs(model.coefficients_[2] + 0.7) < 2e-5


def test_export_expands_conservative_nonlinear_terms():
    x, t = np.linspace(0, 2, 101), np.linspace(0, 1, 81)
    model = WSINDyPDEModel(
        terms=[(2, 1)], m_x=12, m_t=10, test_function_power=6, threshold=0
    ).fit(1 + x[:, None] + t[None, :], x, t)
    u, ux = sp.symbols("u u_x")
    expected = 2 * model.coefficients_[0] * u * ux
    delta = sp.simplify((sp.sympify(model.best_expression_) - expected) / (u * ux))
    assert abs(float(delta)) < 1e-12
    assert "d_x" in model.weak_expression_


def test_weak_system_rejects_nonuniform_or_misaligned_grids():
    model = WSINDyPDEModel()
    with pytest.raises(ValueError, match="shape"):
        model.weak_system(np.zeros((10, 9)), np.arange(9), np.arange(9))
    x = np.linspace(0, 1, 10) ** 2
    with pytest.raises(ValueError, match="uniform spatial"):
        model.weak_system(np.ones((10, 10)), x, np.linspace(0, 1, 10))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_iter": 0},
        {"threshold": -1},
        {"ridge": float("nan")},
        {"m_x": 1.5},
        {"thresholds": []},
        {"sparsity_tradeoff": 1.1},
    ],
)
def test_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        WSINDyPDEModel(**kwargs)


def test_rejects_identically_zero_constant_derivatives():
    u, x, t = traveling_wave()
    with pytest.raises(ValueError, match="constant"):
        WSINDyPDEModel(terms=[(0, 1)]).weak_system(u, x, t)
