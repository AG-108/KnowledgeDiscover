"""Numerical checks for the internal and PySINDy sparse regressors."""

import numpy as np

from kd.model.kd_sindy import PySINDyModel, SINDyModel


def polynomial_problem():
    X = np.linspace(-2, 2, 201)[:, None]
    y = 1.5 - 2.0 * X[:, 0] + 0.75 * X[:, 0] ** 2
    return X, y


def test_internal_sindy_recovers_polynomial():
    X, y = polynomial_problem()
    model = SINDyModel(threshold=1e-8, alpha=1e-12).fit(X, y, variable_names=["x"])
    np.testing.assert_allclose(model.predict(X), y, atol=1e-8)
    assert "x**2" in model.best_expression_


def test_pysindy_sr3_fits_polynomial():
    X, y = polynomial_problem()
    model = PySINDyModel(reg_weight_lam=1e-8, tolerance=1e-8, max_iter=100).fit(
        X, y, variable_names=["x"]
    )
    np.testing.assert_allclose(model.predict(X), y, atol=1e-5)
    assert "x**2" in model.best_expression_
