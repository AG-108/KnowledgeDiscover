"""Shared helpers for the example scripts under examples/.

Every example script that needs to `import kd` while being run directly
(e.g. `python examples/kd_foo_example.py`) used to repeat the same
sys.path bootstrap and, in a couple of cases, the same wave-dataset
helper functions with divergent copies. This module centralizes both so
example scripts stay small and in sync.
"""

import os
import sys

import numpy as np


def bootstrap_project_root():
    """Ensure the repo root is importable as `kd.*` and return its path.

    Safe to call multiple times; only inserts into sys.path once.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    return project_root


def point_cloud_to_regular_grid(
    arr,
    t_col=0,
    x_col=1,
    u_col=2,
    nx=512,
    round_t_decimals=10,
    x_margin_ratio=0.02,
):
    """
    Convert irregular point-cloud wave data [t, x, u]
    into a regular grid [t_grid, x_grid, u_grid].

    Parameters
    ----------
    arr : ndarray, shape [N, 3]
        Point cloud data.

    t_col, x_col, u_col : int
        Column indices for t, x, u.

    nx : int
        Number of points in the regular spatial grid.

    round_t_decimals : int
        Decimals for grouping time values.

    x_margin_ratio : float
        Shrink common x range slightly to avoid edge interpolation artifacts.

    Returns
    -------
    t_grid : ndarray, shape [nt]
    x_grid : ndarray, shape [nx]
    u_grid : ndarray, shape [nt, nx]
    """

    arr = np.asarray(arr, dtype=float)
    arr = arr[np.all(np.isfinite(arr), axis=1)]

    t = np.round(arr[:, t_col], round_t_decimals)
    x = arr[:, x_col]
    u = arr[:, u_col]

    t_grid = np.unique(t)
    nt = len(t_grid)

    # First pass: determine common spatial range across all time steps.
    x_min_list = []
    x_max_list = []

    grouped = {}

    for ti in t_grid:
        mask = t == ti
        x_i = x[mask]
        u_i = u[mask]

        order = np.argsort(x_i)
        x_i = x_i[order]
        u_i = u_i[order]

        # Remove duplicated x within same time by averaging.
        x_unique, inverse = np.unique(x_i, return_inverse=True)
        u_sum = np.zeros_like(x_unique, dtype=float)
        counts = np.zeros_like(x_unique, dtype=float)

        np.add.at(u_sum, inverse, u_i)
        np.add.at(counts, inverse, 1.0)

        u_unique = u_sum / counts

        grouped[ti] = (x_unique, u_unique)

        x_min_list.append(float(np.min(x_unique)))
        x_max_list.append(float(np.max(x_unique)))

    common_x_min = max(x_min_list)
    common_x_max = min(x_max_list)

    if common_x_min >= common_x_max:
        raise ValueError(
            f"No common x range across time steps: "
            f"common_x_min={common_x_min}, common_x_max={common_x_max}"
        )

    # Shrink range slightly to avoid boundary artifacts.
    width = common_x_max - common_x_min
    common_x_min = common_x_min + x_margin_ratio * width
    common_x_max = common_x_max - x_margin_ratio * width

    x_grid = np.linspace(common_x_min, common_x_max, nx)

    u_grid = np.empty((nt, nx), dtype=float)

    for i, ti in enumerate(t_grid):
        x_i, u_i = grouped[ti]

        # np.interp requires sorted x_i.
        u_grid[i, :] = np.interp(x_grid, x_i, u_i)

    return t_grid, x_grid, u_grid


def build_wave_dataset_for_dscv(arr, dx=1.0, dt=1.0):
    """
    Convert wave elevation data to KD_DSCV PDETask format.

    Expected:
        arr.shape = (nt, nx)

    Returns:
        dataset["u"].shape  = (nx, nt)
        dataset["X"]        = [x], x.shape = (nx, 1)
        dataset["ut"].shape = (nx, nt)
    """
    eta = np.asarray(arr, dtype=np.float64)

    if eta.ndim != 2:
        raise ValueError(f"Expected 2D arr, got shape {eta.shape}")

    nt, nx = eta.shape

    # KD_DSCV convention: first axis = space, second axis = time
    u = eta.T.copy()

    x = (np.arange(nx, dtype=np.float64) * dx).reshape(-1, 1)
    t = (np.arange(nt, dtype=np.float64) * dt).reshape(-1, 1)

    # Time derivative along time axis
    ut = np.gradient(u, dt, axis=1)

    return {
        "u": u,
        "X": [x],
        "t": t,
        "ut": ut,
        "n_input_dim": 1,
        "sym_true": "",
    }
