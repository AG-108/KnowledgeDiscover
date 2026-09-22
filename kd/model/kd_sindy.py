"""Sparse identification models used by the unified ODE/PDE benchmark.

``SINDyModel`` is a small NumPy implementation of the classic sequentially
thresholded least-squares algorithm. ``PySINDyModel`` uses PySINDy's feature
library and SR3 optimizer. Keeping both behind the same regression interface
lets the benchmark feed precomputed time or spatial derivatives to either
implementation without changing its train/test split.
"""

from __future__ import annotations

from itertools import combinations_with_replacement

import numpy as np


def _validate_training_data(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if X.ndim != 2 or len(X) != len(y):
        raise ValueError("X must be a two-dimensional array aligned with y")
    if len(X) < 2 or X.shape[1] < 1:
        raise ValueError("SINDy needs at least two samples and one input feature")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("SINDy does not accept NaN or infinite values")
    return X, y


def _polynomial_library(X, variable_names, degree, include_bias):
    """Build deterministic monomials without depending on scikit-learn."""
    columns = []
    names = []
    if include_bias:
        columns.append(np.ones(len(X)))
        names.append("1")
    for order in range(1, degree + 1):
        for indices in combinations_with_replacement(range(X.shape[1]), order):
            columns.append(np.prod(X[:, indices], axis=1))
            powers = {index: indices.count(index) for index in set(indices)}
            factors = [
                variable_names[index] if power == 1 else f"{variable_names[index]}**{power}"
                for index, power in sorted(powers.items())
            ]
            names.append("*".join(factors))
    return np.column_stack(columns), names


def _format_expression(feature_names, coefficients, tolerance=1e-12):
    terms = []
    for name, coefficient in zip(feature_names, coefficients):
        if abs(coefficient) <= tolerance:
            continue
        if name == "1":
            terms.append(f"({coefficient:.12g})")
        else:
            terms.append(f"({coefficient:.12g})*({name})")
    return " + ".join(terms) if terms else "0"


class SINDyModel:
    """Classic SINDy with a polynomial library and STLSQ sparse regression."""

    def __init__(
        self,
        polynomial_degree=2,
        include_bias=True,
        threshold=0.05,
        alpha=1e-6,
        max_iter=20,
        normalize_columns=True,
    ):
        if polynomial_degree < 1:
            raise ValueError("polynomial_degree must be at least one")
        if threshold < 0 or alpha < 0 or max_iter < 1:
            raise ValueError("threshold and alpha must be non-negative; max_iter must be positive")
        self.polynomial_degree = int(polynomial_degree)
        self.include_bias = bool(include_bias)
        self.threshold = float(threshold)
        self.alpha = float(alpha)
        self.max_iter = int(max_iter)
        self.normalize_columns = bool(normalize_columns)

    def fit(self, X, y, variable_names=None):
        X, y = _validate_training_data(X, y)
        names = list(variable_names or [f"x{i + 1}" for i in range(X.shape[1])])
        if len(names) != X.shape[1]:
            raise ValueError("variable_names must match the number of input columns")
        theta, feature_names = _polynomial_library(
            X, names, self.polynomial_degree, self.include_bias
        )
        scales = (
            np.linalg.norm(theta, axis=0) if self.normalize_columns else np.ones(theta.shape[1])
        )
        scales[scales == 0] = 1.0
        normalized = theta / scales
        gram = normalized.T @ normalized
        ridge = self.alpha * np.eye(normalized.shape[1])
        coefficients = np.linalg.solve(gram + ridge, normalized.T @ y)
        active = np.ones(len(coefficients), dtype=bool)
        for _ in range(self.max_iter):
            next_active = np.abs(coefficients) >= self.threshold
            if np.array_equal(next_active, active):
                break
            active = next_active
            coefficients[~active] = 0.0
            if not active.any():
                break
            reduced = normalized[:, active]
            coefficients[active] = np.linalg.solve(
                reduced.T @ reduced + self.alpha * np.eye(active.sum()), reduced.T @ y
            )
        self.coefficients_ = coefficients / scales
        self.feature_names_ = feature_names
        self.best_expression_ = _format_expression(feature_names, self.coefficients_)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        if not hasattr(self, "coefficients_"):
            raise RuntimeError("fit must be called before predict")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("X has an incompatible feature shape")
        theta, _ = _polynomial_library(
            X,
            [f"x{i + 1}" for i in range(X.shape[1])],
            self.polynomial_degree,
            self.include_bias,
        )
        return theta @ self.coefficients_


class PySINDyModel:
    """PySINDy polynomial features with SR3 sparse optimization."""

    def __init__(
        self,
        polynomial_degree=2,
        include_bias=True,
        reg_weight_lam=0.005,
        regularizer="L0",
        relax_coeff_nu=1.0,
        tolerance=1e-5,
        max_iter=30,
        normalize_columns=True,
    ):
        self.polynomial_degree = int(polynomial_degree)
        self.include_bias = bool(include_bias)
        self.reg_weight_lam = float(reg_weight_lam)
        self.regularizer = regularizer
        self.relax_coeff_nu = float(relax_coeff_nu)
        self.tolerance = float(tolerance)
        self.max_iter = int(max_iter)
        self.normalize_columns = bool(normalize_columns)

    def fit(self, X, y, variable_names=None):
        import pysindy as ps

        X, y = _validate_training_data(X, y)
        names = list(variable_names or [f"x{i + 1}" for i in range(X.shape[1])])
        if len(names) != X.shape[1]:
            raise ValueError("variable_names must match the number of input columns")
        library = ps.PolynomialLibrary(
            degree=self.polynomial_degree,
            include_bias=self.include_bias,
        )
        theta = library.fit_transform(X)
        optimizer = ps.SR3(
            reg_weight_lam=self.reg_weight_lam,
            regularizer=self.regularizer,
            relax_coeff_nu=self.relax_coeff_nu,
            tol=self.tolerance,
            max_iter=self.max_iter,
            normalize_columns=self.normalize_columns,
            unbias=False,
        )
        optimizer.fit(theta, y[:, None])
        self.coefficients_ = np.asarray(optimizer.coef_[0], dtype=float)
        self.feature_names_ = [
            name.replace(" ", "*").replace("^", "**") for name in library.get_feature_names(names)
        ]
        self.best_expression_ = _format_expression(self.feature_names_, self.coefficients_)
        self.n_features_in_ = X.shape[1]
        self._library = library
        self._optimizer = optimizer
        return self

    def predict(self, X):
        if not hasattr(self, "_library"):
            raise RuntimeError("fit must be called before predict")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("X has an incompatible feature shape")
        return np.asarray(self._optimizer.predict(self._library.transform(X))).reshape(-1)


__all__ = ["PySINDyModel", "SINDyModel"]
