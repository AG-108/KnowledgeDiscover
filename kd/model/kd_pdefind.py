# PDE-FIND wrapper for 1D grid PDE datasets.
#
# Usage:
# - Input: `dataset.get_data()` must return a dict with `x`, `t`, `usol`
#   (a regular grid, matching the GridPDEDataset convention used elsewhere
#   in kd/dataset).
# - `fit()` builds a selectable library of terms up to 3rd-order spatial
#   derivatives (u, u_x, u_xx, u_xxx, products, and u^3) and
#   solves for u_t = Theta * coeffs via least squares, then hard-thresholds
#   small coefficients to zero.
# - `predict()` advances u(x, t0) to u(x, t0+dt) via a single explicit
#   forward-Euler step using the discovered PDE (not RK4).
#
# NOTE (known limitation): `fit()` currently solves the library with a
# plain `np.linalg.lstsq` + hard threshold, not PySINDy's actual STLSQ/
# STRidge optimizer. `function_library`, `differentiation_method`, and
# `alpha` are accepted in `__init__` for API compatibility but are not
# used by `fit()`; only `threshold`, `derivative_order`, and `terms` take
# effect. Picking a `threshold` that's too large relative to the
# true coefficients will silently zero out every term (degenerate
# `u_t = 0` result). A future implementation should use a real STRidge/STLSQ solver.

from typing import Any, Dict, Optional, Sequence

import numpy as np
import pysindy as ps
from pysindy.differentiation import FiniteDifference


class PDEFindModel:
    """A stable PDE-FIND wrapper for 1D grid datasets."""

    LIBRARY_NAMES = (
        "1", "u", "u_x", "u_xx", "u_xxx", "u^2", "u*u_x", "u*u_xx",
        "u*u_xxx", "u_x^2", "u_x*u_xx", "u_xx^2", "u^3",
    )
    LIBRARY_ORDERS = (0, 0, 1, 2, 3, 0, 1, 2, 3, 1, 2, 2, 0)

    def __init__(
        self,
        derivative_order: int = 2,
        function_library: Optional[ps.PDELibrary] = None,
        differentiation_method: ps.differentiation.BaseDifferentiation = FiniteDifference,
        threshold: float = 0.05,
        alpha: float = 1e-5,
        max_iter: int = 50,
        terms: Optional[Sequence[str]] = None,
    ):
        if derivative_order not in (0, 1, 2, 3):
            raise ValueError("derivative_order must be 0, 1, 2, or 3")
        if terms is not None:
            terms = tuple(terms)
            unknown = set(terms) - set(self.LIBRARY_NAMES)
            if not terms or unknown:
                raise ValueError(f"terms must be nonempty known library names; unknown={sorted(unknown)}")
            if len(terms) != len(set(terms)):
                raise ValueError("terms must be unique")
        self.derivative_order = derivative_order
        self.terms = terms
        self.function_library = function_library or ps.PolynomialLibrary(
            degree=2, include_bias=False
        )
        self.threshold = threshold
        self.alpha = alpha
        self.max_iter = max_iter

        self._model = None
        self._x = None
        self._t = None
        self._U = None
        self._spatial_grid = None
        self._diff_method = differentiation_method

    def fit(self, dataset: Any) -> None:
        data: Dict[str, Any] = dataset.get_data()
        x = np.asarray(data["x"], dtype=float).flatten()
        t = np.asarray(data["t"], dtype=float).flatten()

        usol = np.real(np.asarray(data["usol"], dtype=float))
        if usol.ndim == 3:
            usol = usol[0]
        if usol.shape != (len(x), len(t)):
            usol = usol.T
        if usol.shape != (len(x), len(t)):
            raise ValueError(f"usol shape {usol.shape} != ({len(x)}, {len(t)})")

        U = usol.astype(float)

        self._x = x
        self._t = t
        self._U = U
        self._spatial_grid = x

        ux = np.gradient(U, x, axis=0, edge_order=2)
        ut = np.gradient(U, t, axis=1, edge_order=2)
        uxx = np.gradient(ux, x, axis=0, edge_order=2)
        uxxx = np.gradient(uxx, x, axis=0, edge_order=2)

        candidate_terms = [
            np.ones_like(U),
            U,
            ux,
            uxx,
            uxxx,
            U * U,
            U * ux,
            U * uxx,
            U * uxxx,
            ux * ux,
            ux * uxx,
            uxx * uxx,
            U * U * U,
        ]
        names = self.LIBRARY_NAMES
        selected = [
            index for index, (name, order) in enumerate(zip(names, self.LIBRARY_ORDERS))
            if order <= self.derivative_order and (self.terms is None or name in self.terms)
        ]
        if not selected:
            raise ValueError("No configured PDE-FIND terms satisfy derivative_order")
        candidate_terms = [candidate_terms[index] for index in selected]
        names = [names[index] for index in selected]

        theta = np.column_stack([term.ravel() for term in candidate_terms])
        target = ut.ravel()
        coeffs, *_ = np.linalg.lstsq(theta, target, rcond=None)
        coeffs[np.abs(coeffs) <= self.threshold] = 0.0

        self._coeffs = coeffs
        self._feature_names = names
        self._model = {"coeffs": coeffs, "names": names}

    def print_model(self) -> None:
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        terms = []
        for name, coeff in zip(self._feature_names, self._coeffs):
            if abs(coeff) < self.threshold:
                continue
            sign = "+" if coeff >= 0 else "-"
            magnitude = abs(coeff)
            coeff_text = "1.0" if magnitude == 1.0 else f"{magnitude:.4g}"
            terms.append(f"{sign} {coeff_text} * {name}")
        expr = " ".join(terms)
        print("u_t =" + (expr if expr else " 0"))

    def coefficients(self) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return np.asarray(self._coeffs, dtype=float)

    def predict(self, U0: np.ndarray, dt: float) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        if self._x is None:
            raise RuntimeError("No spatial grids stored.")

        U0 = np.asarray(U0, dtype=float).flatten()
        nx = len(self._x)
        if U0.shape != (nx,):
            raise ValueError(f"U0 shape {U0.shape} != ({nx},)")

        x = self._x
        ux = np.gradient(U0, x, edge_order=2)
        uxx = np.gradient(ux, x, edge_order=2)
        uxxx = np.gradient(uxx, x, edge_order=2)

        candidate_terms = [
            np.ones_like(U0),
            U0,
            ux,
            uxx,
            uxxx,
            U0 * U0,
            U0 * ux,
            U0 * uxx,
            U0 * uxxx,
            ux * ux,
            ux * uxx,
            uxx * uxx,
            U0 * U0 * U0,
        ]
        terms_by_name = dict(zip(self.LIBRARY_NAMES, candidate_terms))
        ut = sum(
            coeff * terms_by_name[name]
            for coeff, name in zip(self._coeffs, self._feature_names)
        )
        return U0 + dt * ut
