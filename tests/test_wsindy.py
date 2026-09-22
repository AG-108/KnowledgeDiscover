import numpy as np
import pytest
import sympy as sp

from kd.model.kd_wsindy import IntegralWeakPDEModel


def test_weak_form_recovers_linear_advection_without_field_derivatives():
    x = np.linspace(0, 2 * np.pi, 161)
    t = np.linspace(0, 2.0, 121)
    speed = 0.7
    u = np.sin(x[:, None] - speed * t[None, :])
    model = IntegralWeakPDEModel(terms=[(1, 1)], n_test_x=6, n_test_t=6,
                                 support_fraction=0.2, threshold=0).fit(u, x, t)
    np.testing.assert_allclose(model.coefficients_, [-speed], atol=2e-3)
    assert model.weak_residual_mse_ < 1e-7


def test_weak_system_rejects_misaligned_grid():
    model = IntegralWeakPDEModel()
    with np.testing.assert_raises(ValueError):
        model.weak_system(np.zeros((10, 9)), np.arange(9), np.arange(9))


def test_export_expands_conservative_nonlinear_terms():
    x, t = np.linspace(0, 2, 101), np.linspace(0, 1, 81)
    model = IntegralWeakPDEModel(terms=[(2, 1)], threshold=0).fit(
        1 + x[:, None] + t[None, :], x, t)
    u, ux = sp.symbols("u u_x")
    expected = 2 * model.coefficients_[0] * u * ux
    delta = sp.simplify((sp.sympify(model.best_expression_) - expected) / (u * ux))
    assert abs(float(delta)) < 1e-12
    assert "d_x" in model.weak_expression_


@pytest.mark.parametrize("kwargs", [{"max_iter": 0}, {"threshold": -1}, {"ridge": float("nan")}, {"n_test_x": 1.5}])
def test_reject_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        IntegralWeakPDEModel(**kwargs)


def test_reject_derivatives_beyond_test_function_smoothness():
    with pytest.raises(ValueError):
        IntegralWeakPDEModel(terms=[(1, 5)]).weak_system(np.ones((10, 10)), np.arange(10), np.arange(10))
