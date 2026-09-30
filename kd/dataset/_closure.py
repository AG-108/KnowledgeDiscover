"""Optional NetCDF observations for offline two-layer PV closure."""

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
PREFIX = "Discover_Green_function/dso/task/pde/data/"
PYQG_PATHS = [PREFIX + f"eddy_forcing1_run={i}_scipy.nc" for i in (0, 1)]
RESOLVED = ("q", "streamfunction", "u", "v")
TARGET = "q_subgrid_forcing"


class ClosureDataset:
    def __init__(self, runs, metadata):
        self.registry_name = "pyqg_subgrid"
        self.runs, self.protocol_metadata = runs, metadata
        self.target_name = TARGET


def _axis(axis, length):
    axis = np.asarray(axis, float)
    if axis.ndim != 1 or len(axis) < 4 or not np.isfinite(axis).all():
        raise ValueError("Invalid pyqg coordinate")
    delta = np.diff(axis)
    if (
        not (delta > 0).all()
        or not np.allclose(delta, delta[0])
        or not np.isclose(len(axis) * delta[0], length)
    ):
        raise ValueError("Pyqg periodic geometry not supported by actual axes and L/W")
    return axis


def load_pyqg(*, root=None):
    import xarray as xr  # Optional dependency, never imported by dataset import.

    root = Path(root) if root is not None else ROOT
    runs, cards = [], []
    for index, path in enumerate(PYQG_PATHS):
        with xr.open_dataset(root / path, decode_times=False) as source:
            if source.sizes.get("lev") != 2:
                raise ValueError("First pyqg closure protocol requires two layers")
            title = str(source.attrs.get("title", ""))
            if "spectral truncation" not in title or "sharp spectral filter" not in title:
                raise ValueError("source_blocked: pyqg coarsening/filter evidence missing")
            x = _axis(source.x.values, float(source.attrs["pyqg:L"]))
            y = _axis(source.y.values, float(source.attrs["pyqg:W"]))
            time = np.asarray(source.time.values, float)
            if len(time) < 8 or not np.isfinite(time).all() or not (np.diff(time) > 0).all():
                raise ValueError("Invalid pyqg observation time")
            fields = {}
            for name in (*RESOLVED, TARGET):
                if tuple(source[name].dims) != ("time", "lev", "y", "x"):
                    raise ValueError("Unexpected pyqg field axes: " + name)
                fields[name] = np.asarray(source[name].values, float)
                if not np.isfinite(fields[name]).all():
                    raise ValueError("Pyqg fields need explicit nonfinite mask protocol")
            if source[TARGET].attrs.get("units") != "s^-2":
                raise ValueError("source_blocked: PV closure units changed")
            runs.append(
                {
                    "time": time,
                    "x": x,
                    "y": y,
                    "lev": np.asarray(source.lev.values),
                    "fields": fields,
                    "periodic": True,
                }
            )
            cards.append(
                {
                    "run": index,
                    "path": path,
                    "shape": list(fields[TARGET].shape),
                    "time_spacing_seconds": np.diff(time).tolist(),
                    "integrator_dt": float(source.attrs["pyqg:dt"]),
                    "layer_ids": source.lev.values.tolist(),
                    "title": title,
                    "variables": {name: dict(source[name].attrs) for name in source.variables},
                    "coordinate_unit_evidence": "x/y grid-point labels; actual spacing times nx equals L/W in meters",
                    "target_sign": "stored q_subgrid_forcing convention from NetCDF long_name; no inferred sign flip",
                    "periodic_evidence": "pyqg spectral x/y coarsening, endpoint-excluded axes matching L/W",
                    "generator_version": "unknown",
                    "filter_cutoff": "not independently specified",
                }
            )
    if any(not np.array_equal(runs[0][a], runs[1][a]) for a in ("x", "y", "lev")):
        raise ValueError("Pyqg heldout run coordinate/layer mismatch")
    return ClosureDataset(
        runs,
        {
            "source_paths": PYQG_PATHS,
            "source_cards": cards,
            "scientific_task": "turbulence_closure",
            "split_protocol": "run0_train_run1_test; internal_temporal_gap2",
            "features": list(RESOLVED),
            "excluded_features": "full fields, forcing/flux/tendency fields and target derivatives",
            "online_status": "not evaluated",
            "reference_status": "unknown",
        },
    )


def spectral_derivative(field, axis, coordinates, *, periodic):
    if periodic is not True:
        raise ValueError("Spectral derivatives require verified periodic geometry")
    coords = np.asarray(coordinates, float)
    dx = np.diff(coords)
    if not (dx > 0).all() or not np.allclose(dx, dx[0]):
        raise ValueError("Spectral derivative requires uniform coordinates")
    wave = 2 * np.pi * np.fft.fftfreq(len(coords), d=dx[0])
    shape = [1] * field.ndim
    shape[axis] = len(coords)
    return np.fft.ifft(np.fft.fft(field, axis=axis) * (1j * wave).reshape(shape), axis=axis).real


def resolved_features(run):
    # No target or non-resolved entry is accessed here.
    columns, names = [], []
    for layer in range(2):
        for name in RESOLVED:
            field = run["fields"][name][:, layer]
            columns += [
                field,
                spectral_derivative(field, -1, run["x"], periodic=run["periodic"]),
                spectral_derivative(field, -2, run["y"], periodic=run["periodic"]),
            ]
            names += [f"{name}_L{layer + 1}", f"{name}_x_L{layer + 1}", f"{name}_y_L{layer + 1}"]
    return np.stack(columns, axis=-1), names
