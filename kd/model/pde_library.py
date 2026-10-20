"""Shared, explicit scalar PDE libraries for calibrated discovery adapters.

Library membership depends on configuration and coordinate validity, never on a
reference equation. Derivatives are computed inside the supplied time block.
"""
from __future__ import annotations

import numpy as np


BASE_TERMS = (
    "1", "u", "u**2", "u**3", "u_x", "u_xx", "u_xxx",
    "u*u_x", "u*u_xx", "u_x**2",
)
EXTRA_TERMS = ("u**2*u_x", "u*u_xxx", "u_x*u_xx", "u_xx**2")
RADIAL_TERMS = ("u_x/x", "u/x", "u/x**2", "u_xx/x")


def library_names(kind="compact", coordinate_terms=False, x=None):
    if kind not in {"compact", "extended"}:
        raise ValueError("library kind must be compact or extended")
    names = list(BASE_TERMS + (EXTRA_TERMS if kind == "extended" else ()))
    if coordinate_terms:
        x = np.asarray(x, dtype=float)
        # A radial coordinate chart must not cross or touch its singularity.
        if np.min(x) > 1e-8 or np.max(x) < -1e-8:
            names.extend(RADIAL_TERMS)
    return names


def evaluate_terms(names, values):
    """Evaluate only the declared monomial vocabulary (NumPy or torch arrays)."""
    u, ux, uxx, uxxx = (values[k] for k in ("u", "u_x", "u_xx", "u_xxx"))
    terms = {
        "1": u * 0 + 1, "u": u, "u**2": u**2, "u**3": u**3,
        "u_x": ux, "u_xx": uxx, "u_xxx": uxxx,
        "u*u_x": u * ux, "u*u_xx": u * uxx, "u_x**2": ux**2,
        "u**2*u_x": u**2 * ux, "u*u_xxx": u * uxxx,
        "u_x*u_xx": ux * uxx, "u_xx**2": uxx**2,
    }
    if any(name in RADIAL_TERMS for name in names):
        x = values["x"]
        terms.update({"u_x/x": ux / x, "u/x": u / x,
                      "u/x**2": u / x**2, "u_xx/x": uxx / x})
    return [terms[name] for name in names]


def grid_problem(dataset, kind="compact", coordinate_terms=False, stencil=7):
    from kd.dataset._derivatives import axis_derivative

    data = dataset.get_data()
    x, t = (np.asarray(data[k], dtype=float).reshape(-1) for k in ("x", "t"))
    u = np.asarray(data["usol"], dtype=float).squeeze()
    if u.shape != (len(x), len(t)):
        u = u.T
    names = library_names(kind, coordinate_terms, x)
    values = {"u": u, "x": np.broadcast_to(x[:, None], u.shape)}
    values.update({"u_" + "x" * order: axis_derivative(u, x, order, 0, stencil)
                   for order in (1, 2, 3)})
    target = axis_derivative(u, t, 1, 1, stencil)
    theta = np.column_stack([v.ravel() for v in evaluate_terms(names, values)])
    good = np.isfinite(theta).all(axis=1) & np.isfinite(target.ravel())
    return theta[good], target.ravel()[good], names


def normalized_stlsq(theta, target, threshold=0.01, ridge=1e-6, max_iter=30):
    """Threshold dimensionless contributions; debias the selected support."""
    theta, target = np.asarray(theta, float), np.asarray(target, float).reshape(-1)
    scales = np.linalg.norm(theta, axis=0)
    scale_y = max(np.linalg.norm(target), np.finfo(float).tiny)
    usable = scales > np.finfo(float).eps
    safe = np.where(usable, scales, 1.0)
    design, y = theta / safe, target / scale_y
    active = usable.copy()
    coefficients = np.zeros(theta.shape[1])
    for _ in range(max_iter):
        reduced = design[:, active]
        if not reduced.shape[1]:
            break
        coefficients[:] = 0
        coefficients[active] = np.linalg.lstsq(
            np.vstack((reduced, np.sqrt(ridge) * np.eye(reduced.shape[1]))),
            np.r_[y, np.zeros(reduced.shape[1])], rcond=None,
        )[0]
        next_active = usable & (np.abs(coefficients) >= threshold)
        if np.array_equal(active, next_active):
            break
        active = next_active
    coefficients[:] = 0
    if active.any():
        coefficients[active] = np.linalg.lstsq(design[:, active], y, rcond=None)[0]
    return coefficients * scale_y / safe


def expression(names, coefficients):
    return " + ".join(f"({float(c):.16g})*({name})"
                      for name, c in zip(names, coefficients) if c != 0) or "0"
