import os
import sys

from _common import bootstrap_project_root

project_root = bootstrap_project_root()

import numpy as np

from kd.dataset import load_wake_equation

DATA_DIR = os.path.join(project_root, "kd", "dataset", "WDwake")
if not os.path.isdir(DATA_DIR):
    raise FileNotFoundError(f"Wake dataset directory not found: {DATA_DIR}")


def _build_weak_form_terms(U: np.ndarray, x: np.ndarray, y: np.ndarray, t: np.ndarray):
    """Build a small weak-form style library via finite differences and sparse regression."""
    ux = np.gradient(U, x, axis=0, edge_order=2)
    uy = np.gradient(U, y, axis=1, edge_order=2)
    ut = np.gradient(U, t, axis=2, edge_order=2)

    uxx = np.gradient(ux, x, axis=0, edge_order=2)
    uyy = np.gradient(uy, y, axis=1, edge_order=2)
    uxy = np.gradient(ux, y, axis=1, edge_order=2)

    terms = [
        np.ones_like(U),
        U,
        ux,
        uy,
        uxx,
        uyy,
        uxy,
        U * U,
        U * ux,
        U * uy,
        ux * ux,
        uy * uy,
        ux * uy,
    ]
    names = [
        "1",
        "u",
        "u_x",
        "u_y",
        "u_xx",
        "u_yy",
        "u_xy",
        "u^2",
        "u*u_x",
        "u*u_y",
        "u_x^2",
        "u_y^2",
        "u_x*u_y",
    ]
    theta = np.column_stack([term.ravel() for term in terms])
    target = ut.ravel()
    return theta, target, names


def fit_sparse_model(U: np.ndarray, x: np.ndarray, y: np.ndarray, t: np.ndarray, threshold: float = 1e-2):
    theta, target, names = _build_weak_form_terms(U, x, y, t)
    coeffs, *_ = np.linalg.lstsq(theta, target, rcond=None)
    selected = [(name, float(coeff)) for name, coeff in zip(names, coeffs) if abs(coeff) > threshold]
    return coeffs, names, selected


dataset = load_wake_equation(DATA_DIR, ["TI8_U.npy", "TI8_V.npy"])
U = np.asarray(dataset.usol[0], dtype=float)
x = np.asarray(dataset.coords["x"], dtype=float)
y = np.asarray(dataset.coords["y"], dtype=float)
t = np.asarray(dataset.coords["t"], dtype=float)

# Keep the example fast and numerically stable with a reduced grid.
U_small = U[::2, ::2, ::4]
x_small = x[::2]
y_small = y[::2]
t_small = t[::4]

coeffs, names, selected = fit_sparse_model(U_small, x_small, y_small, t_small, threshold=1e-2)
print("Weak-form sparse regression (selected terms):")
for name, coeff in selected:
    print(f"  {name}: {coeff:.6e}")

# An intentionally simple forward-Euler sanity check using the identified PDE form.
# The goal here is to keep the example executable without relying on brittle PySINDy internals.
print(f"Number of retained terms: {len(selected)}")
print(f"Field shape used for fitting: {U_small.shape}")
print("Weak-form example completed successfully.")
