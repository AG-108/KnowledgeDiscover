"""Checked A3/A4 observations and source declarations; no invented time axes."""

import hashlib
import io
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

from ._base import GridPDEDataset
from ._equations import EquationTarget, SteadyPDEDataset

ROOT = Path(__file__).resolve().parent

EVOLUTION_SPECS = {
    "parametric_burgers": {
        "path": "data/Parametric_Burgers_equation.npy",
        "axes": "tx",
        "shape": (200, 251),
        "grid": {"x": (0.0, 0.004, 251), "t": (0.0, 0.005, 200)},
        "sha256": "4ef1c1f7b4e9878bdde6324c8344047090a3b28b2ebabaca26ff0a7c5eedeecc",
        "order": 1,
        "reference": "-u*u_x+0.02*exp(t)*u_xx",
        "coefficient_dependencies": ["t"],
        "unknown_coefficient_functions": ["viscosity(t)"],
    },
    "parametric_convection_diffusion": {
        "path": "data/Parametric_convection_diffusion.npy",
        "axes": "tx",
        "shape": (250, 201),
        "grid": {"x": (0.0, 0.04, 201), "t": (0.0, 0.02, 250)},
        "sha256": "6d000a7c22e5551e7ab7523aabbcd497d8fd4756d6ce32b53ed692790e164af9",
        "order": 1,
        "reference": "-(1+0.25*sin(x))*u_x+u_xx",
        "coefficient_dependencies": ["x"],
        "unknown_coefficient_functions": ["advection(x)"],
    },
    "advection_diffusion": {
        "path": "data/Advection_diffusion.mat",
        "axes": "xyt",
        "shape": (51, 51, 61, 4),
        "value": "Expression1",
        "sha256": "faf14f3b50500c32f08a910e47a5c0c7d017d641818db4d5bb268dd4153ea6d7",
        "order": 1,
        # Upstream shows the discovered structure, not certified generating coefficients.
        "reference": None,
    },
    "burgers2d": {
        "path": "EqGPT/data/Burgers_2D/Burgers2D.mat",
        "axes": "tyx",
        "value": "u",
        "order": 1,
        "reference": "-u*u_x-u*u_y+0.01*(u_xx+u_yy)",
    },
    "wave": {
        "path": "EqGPT/data/Wave_equation/wave.mat",
        "axes": "xt",
        "value": "u",
        "order": 2,
        "reference": "u_xx",
    },
    "klein_gordon": {
        "path": "data/KG_equation.mat",
        "axes": "xt",
        "value": "usol",
        "order": 2,
        "reference": "0.5*u_xx-5*u",
        "aliases": ["EqGPT/data/KG_equation/KG_Exp.mat"],
    },
    "implicit_mixed": {
        "path": "EqGPT/data/Eq_6_2_12/data_Eq_6_2_12.csv",
        "order": 1,
        "reference": "0.1*u_xt+u_t+0.1*u_x",
    },
}

TLC_STEADY_FILES = {
    "tlc_steady_ns2d": "ns/ns2d.csv",
    "tlc_steady_ns_0_obstacle": "ns/ns_0_obstacle.csv",
    "tlc_steady_ns_4_obstacle": "ns/ns_4_obstacle.csv",
    "tlc_steady_lid_driven": "others/lid_driven.csv",
    "tlc_steady_poisson1_cg_data": "poisson/poisson1_cg_data.csv",
    "tlc_steady_poisson_3d": "poisson/poisson_3d.csv",
    "tlc_steady_poisson_boltzmann2d": "poisson/poisson_boltzmann2d.csv",
    "tlc_steady_poisson_classic": "poisson/poisson_classic.csv",
    "tlc_steady_poisson_manyarea": "poisson/poisson_manyarea.csv",
}
STEADY_SPECS = {
    **{
        name: {
            "path": "TLC/" + path,
            "geometry_status": "source_blocked",
            "reason": "Valid-domain mask, holes and interface labels are not supplied",
        }
        for name, path in TLC_STEADY_FILES.items()
    },
    "laplace_disk": {
        "path": "EqGPT/data/Possion_equation/Possion_x_y.xlsx",
        "reference": "u_xx+u_yy",
        "geometry_status": "polar_observation_domain",
    },
    "laplace_smile": {
        "path": "EqGPT/data/Laplacian_smile/Laplacian_smile.xlsx",
        "geometry_status": "source_blocked",
        "reason": "Validity of exterior zero/string samples and smile mask is unknown",
    },
    "poisson_eitech": {
        "path": "EqGPT/data/Laplacian_EITech/Laplacian_EITech.xlsx",
        "geometry_status": "source_blocked",
        "reason": "EITech mask and exterior sample semantics are unknown; +1 is source-declared",
    },
}


def declared_targets(name):
    """Asset contracts available even if the file is missing."""
    if name in STEADY_SPECS:
        ns = name.startswith("tlc_steady_ns") or name == "tlc_steady_lid_driven"
        return (
            [EquationTarget(f"momentum_{f}", f, 0, "implicit") for f in ("u", "v")]
            if ns
            else [EquationTarget("implicit_relation", time_order=0, kind="implicit")]
        )
    if name in EVOLUTION_SPECS:
        spec = EVOLUTION_SPECS[name]
        return (
            [EquationTarget("implicit_relation", kind="implicit")]
            if name == "implicit_mixed"
            else [EquationTarget("u_" + "t" * spec["order"], time_order=spec["order"])]
        )
    if name in {"tlc_burgers2d", "tlc_ns_long", "wdwake"}:
        return [EquationTarget(f"{f}_t", f) for f in ("u", "v")]
    if name == "tlc_wave_darcy":
        # This declaration is a blocked capability requirement, not verified order.
        return [EquationTarget("wave_relation", time_order=0, kind="implicit")]
    return [EquationTarget("u_t")]


def metadata(name, spec):
    from ._provenance import source_evidence

    evidence = source_evidence(name)
    return {
        "source_paths": [spec["path"]],
        "coordinate_units": "unknown",
        "field_units": "unknown",
        "boundary": evidence.get("boundary", "unknown"),
        "source_status": evidence.get("source_status", "packaged_observation_and_description"),
        "source_evidence": evidence,
        "source_reference": spec.get("reference"),
        "reference_usage": "scoring_only; excluded from model namespace",
        "coefficient_dependencies": list(spec.get("coefficient_dependencies", [])),
        "unknown_coefficient_functions": list(spec.get("unknown_coefficient_functions", [])),
    }


def load_evolution(name):
    spec = EVOLUTION_SPECS[name]
    if spec.get("sha256"):
        raw = (ROOT / spec["path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != spec["sha256"]:
            raise ValueError("source_blocked: checked evolution asset SHA256 changed")
        if "grid" in spec:
            u = np.asarray(np.load(io.BytesIO(raw), allow_pickle=False), dtype=float)
            if u.shape != spec["shape"]:
                raise ValueError("source_blocked: parametric field layout changed")
            axes = {axis: start + step * np.arange(count)
                    for axis, (start, step, count) in spec["grid"].items()}
            u = u.transpose(tuple(spec["axes"].index(axis) for axis in axes))
        else:
            packed = np.asarray(loadmat(io.BytesIO(raw))[spec["value"]], dtype=float)
            if packed.shape != (np.prod(spec["shape"]), 1):
                raise ValueError("source_blocked: Expression1 vector layout changed")
            packed = packed.reshape(spec["shape"], order="C")
            axes = {"x": packed[:, 0, 0, 0], "y": packed[0, :, 0, 1],
                    "t": packed[0, 0, :, 2]}
            mesh = np.meshgrid(*axes.values(), indexing="ij")
            if any(not np.array_equal(packed[..., i], coordinate)
                   for i, coordinate in enumerate(mesh)):
                raise ValueError("source_blocked: Expression1 coordinates are not Cartesian")
            u = packed[..., 3]
    elif name == "implicit_mixed":
        # Verified retained upstream script: header=None, transpose, dx=.01, dt=.02.
        u = pd.read_csv(ROOT / spec["path"], header=None).to_numpy(dtype=float).T
        if u.shape != (501, 501):
            raise ValueError("Mixed derivative asset layout changed")
        axes = {"x": np.arange(501) * 0.01, "t": np.arange(501) * 0.02}
    else:
        payload = loadmat(ROOT / spec["path"])
        raw_axes = spec["axes"]
        axes = {
            axis: np.asarray(payload[axis], dtype=float).reshape(-1)
            for axis in "xyz"
            if axis in raw_axes
        }
        axes["t"] = np.asarray(payload["t"], dtype=float).reshape(-1)
        u = np.asarray(payload[spec["value"]], dtype=float)
        expected = tuple(len(axes[axis]) for axis in raw_axes)
        if u.shape != expected:
            raise ValueError(f"Source axis declaration {expected} disagrees with {u.shape}")
        u = u.transpose(tuple(raw_axes.index(axis) for axis in axes))
    if not np.isfinite(u).all():
        raise ValueError("Nonfinite evolution observations")
    dataset = GridPDEDataset(name, None, None, 0.0, coords=axes, usol=u[None], time_var="t")
    dataset.state_vars = ["u"]
    dataset.equation_targets = declared_targets(name)
    dataset.protocol_metadata = metadata(name, spec)
    return dataset


def load_steady(name):
    spec = STEADY_SPECS[name]
    path = ROOT / spec["path"]
    parameters = {}
    geometry = None
    if path.suffix == ".csv":
        frame = pd.read_csv(path)
        coordinates = ["X", "Y"] + (["Z"] if "Z" in frame else [])
        ns = name.startswith("tlc_steady_ns") or name == "tlc_steady_lid_driven"
        fields = ["u", "v", "p"] if ns else ["u"]
        if name == "tlc_steady_lid_driven":
            columns = ["u @ Re=100", "v @ Re=100", "p @ Re=100"]
            parameters = {"Re": 100}
        else:
            columns = ["u (m/s)", "v (m/s)", "p (Pa)"] if ns else ["u (1)"]
        expected = set(coordinates + columns)
        if set(frame.columns) != expected:
            raise ValueError(
                f"Unrecognized steady columns {list(frame.columns)}; expected {sorted(expected)}"
            )
        points = frame[coordinates].to_numpy(dtype=float)
        values = frame[columns].to_numpy(dtype=float)
    else:
        frame = pd.read_excel(path)
        if frame.shape[1] != (5 if name == "laplace_disk" else 3):
            raise ValueError("Steady XLSX source column count changed")
        numeric = frame.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(numeric).all():
            raise ValueError(
                "source_blocked: nonnumeric/exterior samples need a source validity mask"
            )
        points, values, fields = numeric[:, :2], numeric[:, -1:], ["u"]
        if name == "laplace_disk":
            radii, theta = numeric[:, 2], numeric[:, 3]
            if not (
                np.allclose(points[:, 0], radii * np.cos(theta), atol=1e-10)
                and np.allclose(points[:, 1], radii * np.sin(theta), atol=1e-10)
            ):
                raise ValueError("Polar Cartesian coordinates disagree")
            # Retained observations start at .0001, source prose says .001.
            # Guard the actual observed annulus; do not fill its unobserved center.
            geometry = {
                "kind": "annulus",
                "verified": True,
                "center": [0.0, 0.0],
                "inner_radius": float(radii.min()),
                "outer_radius": float(radii.max()),
                "evidence": "explicit per-row radius/theta and source disk description",
                "source_inner_radius_discrepancy": {
                    "description": 0.001,
                    "observations": float(radii.min()),
                },
            }
    if not np.isfinite(points).all() or not np.isfinite(values).all():
        raise ValueError("Nonfinite steady observations")
    # Polar angle endpoints can produce coordinate duplicates. Only exact identical
    # point AND field rows are deduplicated; conflicting observations fail.
    unique, inverse = np.unique(points, axis=0, return_inverse=True)
    duplicates = len(points) - len(unique)
    if duplicates:
        first = np.unique(points, axis=0, return_index=True)[1]
        if not np.array_equal(values, values[first][inverse]):
            raise ValueError("Duplicate coordinates have conflicting field observations")
        points, values = unique, values[first]
    card = metadata(name, spec)
    card.update(
        known_parameters=parameters,
        duplicate_rows=duplicates,
        geometry_status=spec["geometry_status"],
        source_blocker=spec.get("reason"),
        pressure_gauge="gradient_only; constant shifts equivalent" if len(fields) == 3 else None,
        known_constraints=["u_x+v_y=0; not a discovery achievement"] if len(fields) == 3 else [],
    )
    return SteadyPDEDataset(
        name,
        values,
        points=points,
        state_vars=fields,
        geometry=geometry,
        equation_targets=declared_targets(name),
        protocol_metadata=card,
    )
