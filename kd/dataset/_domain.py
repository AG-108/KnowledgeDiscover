"""Domain protocols: spatial holdouts, constitutive instances and checked fields.

Raw constitutive loaders remain available for inspection. Benchmark instances
must go through ``load_constitutive_instances`` rather than pooling materials.
"""

from copy import deepcopy
from pathlib import Path

import numpy as np

from ._base import GridPDEDataset, TabularRegressionDataset
from ._provenance import source_evidence

ROOT = Path(__file__).resolve().parent
SOLID_SUBDIRS = {
    "solid_dif": ("data_DIF", "*.xlsx"),
    "solid_strain_stress": ("data_strain_stress", "*.xlsx"),
    "solid_hardening": ("saved_data_hardening_strain_rate", "*.pkl"),
}


def cartesian_grid(payload):
    """Losslessly reorder a complete Cartesian point set, or return None.

    No rounding, interpolation, filling or averaging is allowed. Exact unique
    coordinate values determine membership; missing tuples remain scatter data.
    """
    points = np.asarray(payload["points"], dtype=float)
    if points.ndim != 2 or not points.shape[0] or not points.shape[1]:
        raise ValueError("Cartesian conversion requires a nonempty coordinate matrix")
    if not np.isfinite(points).all():
        raise ValueError("Nonfinite spatial coordinates")
    axes = [np.unique(points[:, i], return_inverse=True) for i in range(points.shape[1])]
    shape = tuple(len(axis) for axis, _ in axes)
    if int(np.prod(shape)) != len(points):
        return None
    flat = np.ravel_multi_index(tuple(index for _, index in axes), shape)
    if len(np.unique(flat)) != len(points):
        return None
    values = np.asarray(payload["usol"])
    if values.ndim == 2:
        values = values[..., None]
    expected = (len(points), len(payload["t"]), len(payload["state_vars"]))
    if values.shape != expected:
        raise ValueError(f"Cartesian values/channel layout {values.shape} disagrees with {expected}")
    if len(payload["spatial_vars"]) != points.shape[1]:
        raise ValueError("Spatial names disagree with coordinate dimensions")
    values = values[np.argsort(flat)].reshape(*shape, len(payload["t"]), values.shape[-1])
    field = np.moveaxis(values, -1, 0)
    coords = {name.lower(): axis for name, (axis, _) in zip(payload["spatial_vars"], axes)}
    coords[payload["time_var"]] = payload["t"]
    result = GridPDEDataset(payload["equation_name"], None, payload["domain"], payload["epi"],
                            coords=coords, usol=field, time_var=payload["time_var"])
    result.state_vars = list(payload["state_vars"])
    result.protocol_metadata = {
        **deepcopy(payload.get("protocol_metadata", {})),
        "grid_conversion": "exact_cartesian_reorder",
        "interpolated": False, "state_vars": result.state_vars,
        "coordinate_names": list(coords),
        "coordinate_units": deepcopy(payload.get("coordinate_units", {key: "unknown" for key in coords})),
        "boundary": deepcopy(payload.get("boundary", "unknown")),
    }
    return result


def spatial_blocks(points, depth=4):
    """Partition coordinates into up to 16 contiguous median-split regions.

    Uses coordinates only, independent of targets and random seeds. Identical
    coordinates stay together; a split never divides a tied coordinate value.
    Axes are rescaled by their global extent for balanced geometric partitions.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or not len(points) or not np.isfinite(points).all():
        raise ValueError("Spatial blocks require a nonempty finite coordinate matrix")
    if not isinstance(depth, int) or depth < 1:
        raise ValueError("Spatial partition depth must be a positive integer")
    scale = np.ptp(points, axis=0)
    normalized = (points - points.min(axis=0)) / np.where(scale > 0, scale, 1)
    labels = np.empty(len(points), dtype=int)
    regions = []

    def partition(ids, level):
        extent = np.ptp(normalized[ids], axis=0)
        axis = int(np.argmax(extent))
        if level == depth or extent[axis] == 0:
            labels[ids] = len(regions)
            regions.append({"lower": points[ids].min(axis=0).tolist(),
                            "upper": points[ids].max(axis=0).tolist(), "rows": len(ids)})
            return
        values, counts = np.unique(normalized[ids, axis], return_counts=True)
        cut = int(np.argmin(abs(np.cumsum(counts)[:-1] - len(ids) / 2)))
        left = normalized[ids, axis] <= values[cut]
        partition(ids[left], level + 1)
        partition(ids[~left], level + 1)

    partition(np.arange(len(points)), 0)
    return labels, regions


def constitutive_instance_specs(name):
    """Stable source-based instance identities, available before loading rows."""
    subdir, pattern = SOLID_SUBDIRS[name]
    directory = ROOT / "Discovery_of_soild_consititutive/data" / subdir
    files = sorted(directory.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No constitutive data matching {directory / pattern}")
    instances = {}
    for path in files:
        # The last token in strain-stress filenames is the strain rate.
        key = path.stem.rsplit("_", 1)[0] if name == "solid_strain_stress" else path.stem
        instances.setdefault(key, []).append(path.stem)
    return [{
        "instance": f"{name}/{key}", "source_key": key, "source_stems": stems,
        "source_instance_id": f"{name}/{key}",
        "source_asset_ids": [
            (directory / (stem + (".pkl" if name == "solid_hardening" else ".xlsx"))).relative_to(ROOT).as_posix()
            for stem in stems],
        "material_id": None, "material_study_id": None,
        "identity_status": "source grouping only; physical material/study not reconciled",
    } for key, stems in sorted(instances.items())]


def load_constitutive_instances(name, *, instances=None):
    """One law per source material/condition; never an unidentified global law.

    DIF is split by observed strain-rate values within each source curve.
    Stress and hardening hold out entire rates within each source material.
    Hardening's pre-matched DIF is excluded from predictors: its provenance does
    not establish availability for a held-out rate. The raw loader retains it.
    """
    from ._base import (
        load_solid_dif_dataset,
        load_solid_hardening_dataset,
        load_solid_strain_stress_dataset,
    )

    specs = constitutive_instance_specs(name)
    if instances is not None:
        selected = set(instances)
        unknown = selected - {s["instance"] for s in specs}
        if unknown or not selected:
            raise ValueError(f"Unknown or empty constitutive instance selection: {sorted(unknown)}")
        specs = [s for s in specs if s["instance"] in selected]
    loaders = {"solid_dif": load_solid_dif_dataset,
               "solid_strain_stress": load_solid_strain_stress_dataset,
               "solid_hardening": load_solid_hardening_dataset}
    raw = loaders[name]()
    raw_groups = np.asarray(raw.groups, dtype=str)
    material = (np.array([g.rsplit("_curve", 1)[0] for g in raw_groups])
                if name == "solid_hardening" else raw_groups)
    for spec in specs:
        mask = np.isin(material, spec["source_stems"])
        if not mask.any():
            raise ValueError(f"No rows for {spec['instance']}")
        X = raw.X[mask, :2] if name == "solid_hardening" else raw.X[mask]
        names = raw.variable_names[:X.shape[1]]
        rate_axis = 0 if name == "solid_dif" else 1
        dataset = TabularRegressionDataset(
            spec["instance"], X, raw.y[mask], variable_names=names,
            groups=X[:, rate_axis], descr=raw.descr,
        )
        dataset.target_name = "DIF" if name == "solid_dif" else "stress"
        dataset.split_protocol = ("within_curve_rate_holdout" if name == "solid_dif"
                                  else "within_material_rate_holdout")
        source_curves = sorted(set(raw_groups[mask]))
        curve_ids = [
            (f"{spec['source_asset_ids'][0]}#curve:{curve.rsplit('_curve', 1)[1]}"
             if name == "solid_hardening" else
             f"Discovery_of_soild_consititutive/data/{SOLID_SUBDIRS[name][0]}/{curve}.xlsx#observations")
            for curve in source_curves
        ]
        dataset.protocol_metadata = {
            **spec, "source_curve_ids": curve_ids, "source_evidence": source_evidence(name), "source_curves": sorted(set(raw_groups[mask])),
            "identity_basis": "source_filename_material_and_study; not cross-study reconciliation",
            "DIF_predictor_used": False,
            "feature_scaling": "raw_source_units; no fitted normalization",
        }
        yield spec["instance"], dataset


FIELD_SPECS = {
    "heat": {"path": "data/Heat_equation.mat", "keys": ("x", "t", "usol"),
             "layout": "xt", "rhs": None, "family": "heat"},
    "burgers_sine": {"path": "data/burgers.mat", "keys": ("x", "t", "usol"),
                     "layout": "xt", "rhs": "-u*u_x + 0.1*u_xx", "family": "burgers"},
    "allen_cahn": {"path": "EqGPT/data/Allen_Cahn/Allen_Cahn.mat",
                   "keys": ("x", "t", "usol"), "layout": "xt",
                   "rhs": "0.003*u_xx + u - u**3", "family": "allen_cahn"},
    "convection_diffusion": {"path": "EqGPT/data/Convection_diffusion_equation/data.mat",
                             "keys": ("x", "t", "u"), "layout": "tx",
                             "rhs": "-u_x + 0.25*u_xx", "family": "convection_diffusion"},
}


def load_domain_field(name):
    """Read explicit MAT keys/axis order; never guess orientation or coordinates."""
    from scipy.io import loadmat

    spec = FIELD_SPECS[name]
    raw = loadmat(ROOT / spec["path"])
    x_key, t_key, u_key = spec["keys"]
    x, t, u = (np.asarray(raw[k], dtype=float) for k in (x_key, t_key, u_key))
    x, t = x.reshape(-1), t.reshape(-1)
    if spec["layout"] == "tx":
        u = u.T
    if u.shape != (len(x), len(t)):
        raise ValueError(f"{name}: explicit axis layout disagrees with field shape {u.shape}")
    if (not all(np.isfinite(a).all() for a in (x, t, u))
            or not np.all(np.diff(x) > 0) or not np.all(np.diff(t) > 0)):
        raise ValueError(f"{name}: nonfinite values or unordered coordinates")
    dataset = GridPDEDataset(name, None, None, 0.0, x=x, t=t, usol=u, legacy=True)
    dataset.registry_name = name
    dataset.legacy_name = name
    dataset.sym_true = spec["rhs"] or ""
    dataset.state_vars = ["u"]
    dataset.protocol_metadata = {"source_path": spec["path"], "source_layout": spec["layout"],
                                 "coordinates": "embedded", "lhs": "u_t",
                                 "reference_status": "source_declared" if spec["rhs"] else "unknown",
                                 "source_evidence": source_evidence(name),
                                 "coordinate_names": ["x", "t"],
                                 "coordinate_units": source_evidence(name)["units"],
                                 "boundary": "unknown"}
    return dataset
