"""Integral weak-form sparse regression for scalar PDEs in one space dimension.

This is a small, native baseline inspired by WSINDy-PDE, not a port of the
authors' complete MATLAB algorithm.  It moves both time and spatial
derivatives onto compactly supported test functions by integration by parts;
no numerical derivative of the observed field is taken.  The deliberately
limited library contains conservative terms ``d_x^q(u^p)`` only.
"""

from __future__ import annotations

import numpy as np


def _poly_weights(grid, center, half_width, derivative, power=4):
    """Values of d^derivative/dgrid^derivative [(1-s^2)^power]_+."""
    if half_width <= 0:
        raise ValueError("test-function half widths must be positive")
    # Polynomial coefficients are ascending, as required by np.polynomial.
    base = np.polynomial.polynomial.polypow(np.array([1.0, 0.0, -1.0]), power)
    coeff = np.polynomial.polynomial.polyder(base, m=derivative)
    s = (np.asarray(grid) - center) / half_width
    values = np.polynomial.polynomial.polyval(s, coeff) / half_width**derivative
    return np.where(np.abs(s) < 1.0, values, 0.0)


def _integrate2(values, x, t):
    trap = getattr(np, "trapezoid", np.trapz)
    return trap(trap(values, t, axis=1), x, axis=0)


class IntegralWeakPDEModel:
    """Fit ``u_t = sum c[p,q] d_x^q(u^p)`` using weak integrals.

    Parameters are intentionally modest: a tensor grid of compact polynomial
    test functions is placed away from the boundary, then STLSQ is applied to
    the resulting weak linear system.  This covers a transparent 1D baseline,
    but omits the upstream WSINDy-PDE adaptive support selection, scaling,
    generalized library construction, and robust generalized least squares.
    """

    def __init__(self, terms=None, n_test_x=7, n_test_t=7, support_fraction=0.22,
                 threshold=1e-3, ridge=1e-10, max_iter=10):
        self.terms = list([(1, 0), (2, 0), (1, 1), (2, 1), (1, 2), (1, 3)] if terms is None else terms)
        if any(int(v) != v or v < 1 for v in (n_test_x, n_test_t, max_iter)):
            raise ValueError("test counts and max_iter must be positive integers")
        if not np.isfinite([threshold, ridge, support_fraction]).all() or threshold < 0 or ridge < 0:
            raise ValueError("threshold and ridge must be finite and non-negative")
        self.n_test_x = int(n_test_x)
        self.n_test_t = int(n_test_t)
        self.support_fraction = float(support_fraction)
        self.threshold = float(threshold)
        self.ridge = float(ridge)
        self.max_iter = int(max_iter)

    def _validate(self, u, x, t):
        u, x, t = np.asarray(u, float), np.asarray(x, float).reshape(-1), np.asarray(t, float).reshape(-1)
        if u.shape != (len(x), len(t)):
            raise ValueError("u must have shape (len(x), len(t))")
        if not np.isfinite(x).all() or not np.isfinite(t).all():
            raise ValueError("x and t must be finite")
        if min(len(x), len(t)) < 9 or not np.all(np.diff(x) > 0) or not np.all(np.diff(t) > 0):
            raise ValueError("x and t must be increasing grids with at least nine points")
        if not np.isfinite(u).all():
            raise ValueError("u must contain only finite values")
        if not self.terms or any(p < 0 or not 0 <= q <= 3 or int(p) != p or int(q) != q for p, q in self.terms):
            raise ValueError("terms require non-negative integer powers and derivative orders 0..3")
        if len({tuple(term) for term in self.terms}) != len(self.terms):
            raise ValueError("duplicate library terms are not identifiable")
        return u, x, t

    def weak_system(self, u, x, t):
        u, x, t = self._validate(u, x, t)
        hx = self.support_fraction * (x[-1] - x[0])
        ht = self.support_fraction * (t[-1] - t[0])
        if not (0 < self.support_fraction < 0.5):
            raise ValueError("support_fraction must lie between zero and one half")
        cx = np.linspace(x[0] + hx, x[-1] - hx, self.n_test_x)
        ct = np.linspace(t[0] + ht, t[-1] - ht, self.n_test_t)
        rows, targets = [], []
        for xc in cx:
            wx = _poly_weights(x, xc, hx, 0)
            for tc in ct:
                wt = _poly_weights(t, tc, ht, 0)
                wtd = _poly_weights(t, tc, ht, 1)
                targets.append(-_integrate2(u * wx[:, None] * wtd[None, :], x, t))
                row = []
                for power, derivative in self.terms:
                    wxd = _poly_weights(x, xc, hx, derivative)
                    value = _integrate2(u**power * wxd[:, None] * wt[None, :], x, t)
                    row.append(((-1) ** derivative) * value)
                rows.append(row)
        return np.asarray(rows), np.asarray(targets)

    def fit(self, u, x, t):
        theta, target = self.weak_system(u, x, t)
        scales = np.linalg.norm(theta, axis=0)
        scales[scales == 0] = 1.0
        design = theta / scales
        active = np.ones(design.shape[1], dtype=bool)
        coef = np.zeros(design.shape[1])
        for _ in range(self.max_iter):
            a = design[:, active]
            coef[active] = np.linalg.lstsq(
                np.vstack((a, np.sqrt(self.ridge) * np.eye(a.shape[1]))),
                np.r_[target, np.zeros(a.shape[1])], rcond=None,
            )[0]
            next_active = active & (np.abs(coef) >= self.threshold)
            if np.array_equal(active, next_active):
                break
            active = next_active
            coef[~active] = 0.0
            if not active.any():
                break
        # Refit the surviving support even when the iteration budget ends on pruning.
        if active.any():
            a = design[:, active]
            coef[active] = np.linalg.lstsq(
                np.vstack((a, np.sqrt(self.ridge) * np.eye(a.shape[1]))),
                np.r_[target, np.zeros(a.shape[1])], rcond=None,
            )[0]
        self.coefficients_ = coef / scales
        self.feature_names_ = [f"u^{p}" if q == 0 else f"d_x^{q}(u^{p})" for p, q in self.terms]
        self.weak_expression_ = " + ".join(
            f"({c:.12g})*{name}" for c, name in zip(self.coefficients_, self.feature_names_) if c
        ) or "0"
        # Export ordinary jet-variable syntax understood by the common evaluator.
        import sympy as sp
        coordinate = sp.Symbol("x")
        field = sp.Function("u")(coordinate)
        replacements = {field: sp.Symbol("u")}
        replacements.update({sp.diff(field, coordinate, q): sp.Symbol("u_" + "x" * q)
                             for q in range(1, 4)})
        expanded = sum(float(c) * sp.diff(field**p, coordinate, q).xreplace(replacements)
                       for c, (p, q) in zip(self.coefficients_, self.terms) if c)
        self.best_expression_ = str(sp.expand(expanded))
        self.weak_residual_mse_ = float(np.mean((theta @ self.coefficients_ - target) ** 2))
        return self


__all__ = ["IntegralWeakPDEModel"]
