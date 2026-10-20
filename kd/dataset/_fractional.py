"""Observed FDE responses and integer PDE diagnostic views of the same fields.

Response fitting uses u(x,t) or ex(t). No view estimates fractional derivatives
or infer equation parameters from the reference plotting script.
"""

from copy import deepcopy
from pathlib import Path

import numpy as np

from ._base import GridPDEDataset, TabularRegressionDataset

ROOT = Path(__file__).resolve().parent
FDE_RESPONSE_PROTOCOL_VERSION = "fde-response-v1"
FDE_PDE_PROTOCOL_VERSION = "fde-pde-diagnostic-v1"
FDE_SPECS = {
    "fde_creep": {"path": "FDE/data/creep.mat", "keys": ("t", "ex"), "layout": "curve"},
    "fde_silt": {"path": "FDE/data/silt.mat", "keys": ("t", "ex"), "layout": "curve"},
    "fde_pt_fade": {
        "path": "FDE/data/pt_fade.mat",
        "keys": ("x_lin", "t_det", "un"),
        "layout": "tx",
    },
    "fde_pt_fde": {
        "path": "FDE/data/pt_fde.mat",
        "keys": ("x_lin", "t_det", "un"),
        "layout": "tx",
    },
    "fde_tsfade_fft": {
        "path": "FDE/data/tsfade_fft.mat",
        "keys": ("x", "t", "Exact"),
        "layout": "tx",
    },
    "fde_convection_diffusion": {"path": "FDE/data/Convection_diffusion.dat", "layout": None},
    "fde_ground_water_flow": {"path": "FDE/data/Ground_water_flow.dat", "layout": None},
}
FDE_READY_DATASETS = tuple(name for name, spec in FDE_SPECS.items() if spec["layout"])
FDE_PDE_SPECS = {
    name + "_pde": name for name, spec in FDE_SPECS.items() if spec["layout"] != "curve"
}
FDE_PDE_READY_DATASETS = tuple(
    name for name, response in FDE_PDE_SPECS.items() if response in FDE_READY_DATASETS
)


def response_target(name):
    return "ex" if FDE_SPECS[name]["layout"] == "curve" else "u"


def response_evidence(name):
    """Source declarations without opening optional assets."""
    spec = FDE_SPECS[name]
    evidence = {
        "archive": "FDE.zip",
        "source_path": spec["path"],
        "coordinate_status": "embedded" if spec["layout"] else "requires_explicit_coordinates",
        "source_layout": spec["layout"],
        "units": "unknown",
        "fractional_orders": None,
        "reference_equation": None,
        "derivatives_computed": False,
        "noise_added": False,
        "response_protocol_version": FDE_RESPONSE_PROTOCOL_VERSION,
    }
    if name == "fde_tsfade_fft":
        evidence["rounded_observation_alias"] = "FDE/data/tsfade_fft.dat"
        evidence["coordinate_conflict"] = (
            "Embedded x,t include endpoints 30,15; plotting-script arange grids exclude them. "
            "Use embedded MAT coordinates without resampling."
        )
    if name == "fde_silt":
        evidence["repeated_time_policy"] = "retain both observations at t=0.02447 in the same split"
    return evidence


def pde_evidence(name):
    evidence = response_evidence(FDE_PDE_SPECS[name])
    evidence.update(
        pde_protocol_version=FDE_PDE_PROTOCOL_VERSION,
        observation_identity=FDE_PDE_SPECS[name],
        derivative_policy="integer derivatives estimated separately within train/test blocks",
        fractional_operator_evaluated=False,
        boundary="unknown",
    )
    return evidence


def _numeric(value, label):
    a = np.asarray(value)
    if not np.issubdtype(a.dtype, np.number) or np.iscomplexobj(a):
        raise ValueError(f"FDE {label} must contain real numeric values")
    a = a.astype(float)
    if not a.size or not np.isfinite(a).all():
        raise ValueError(f"FDE {label} must contain finite, nonempty values")
    return a


def _vector(value, label, *, allow_repeated=False):
    a = _numeric(value, label)
    if a.ndim not in {1, 2} or (a.ndim == 2 and 1 not in a.shape):
        raise ValueError(f"FDE {label} must be a vector")
    a = a.reshape(-1)
    delta = np.diff(a)
    if len(a) < 2 or np.any(delta < 0) or (not allow_repeated and np.any(delta == 0)):
        qualifier = "nondecreasing" if allow_repeated else "strictly increasing"
        raise ValueError(f"FDE {label} requires >=2 {qualifier} coordinates")
    if len(np.unique(a)) < 2:
        raise ValueError(f"FDE {label} requires >=2 distinct coordinates")
    return a


class FractionalResponseDataset(TabularRegressionDataset):
    """A response table retaining its original coordinates and raw observations."""

    x = None
    t = None
    usol = None

    def __init__(self, name, t, values, *, x=None, metadata=None):
        t = _vector(t, "t", allow_repeated=x is None)
        values = _numeric(values, "response")
        if x is None:
            if values.ndim not in {1, 2} or (values.ndim == 2 and 1 not in values.shape):
                raise ValueError("FDE curve response must be a vector")
            values = values.reshape(-1)
            if len(values) != len(t):
                raise ValueError("FDE time and curve response lengths differ")
            X, y, names, groups = t[:, None], values, ["t"], t
        else:
            x = _vector(x, "x")
            if values.shape != (len(t), len(x)):
                raise ValueError("FDE field must match the explicit (t,x) coordinate shape")
            xx, tt = np.meshgrid(x, t, indexing="xy")
            X = np.column_stack((xx.ravel(), tt.ravel()))
            y, names, groups = values.ravel(), ["x", "t"], tt.ravel()
        super().__init__(name, X, y, variable_names=names, groups=groups)
        self.registry_name = name
        self.x, self.t, self.values = x, t, values
        self.target_name = "ex" if x is None else "u"
        self.split_protocol = "temporal_unique_time_holdout"
        self.protocol_metadata = {
            **deepcopy(metadata or {}),
            "response_protocol_version": FDE_RESPONSE_PROTOCOL_VERSION,
            "target_definition": "ex=f(t)" if x is None else "u=f(x,t)",
            "response_axis_order": ["t"] if x is None else ["t", "x"],
            "raw_response_shape": list(values.shape),
            "duplicate_time_rows": int(len(t) - len(np.unique(t))),
            "split_protocol": self.split_protocol,
            "reference_status": "unknown",
            "derivatives_computed": False,
        }

    def temporal_holdout(self, test_size=0.2):
        """Hold out the latest distinct times; all rows at a time stay together."""
        if isinstance(test_size, bool) or not np.isfinite(test_size) or not 0 < test_size < 1:
            raise ValueError("FDE test_size must be finite and in (0,1)")
        times = np.unique(self.groups)
        cut = max(1, min(len(times) - 1, int(np.floor(len(times) * (1 - test_size)))))
        test_mask = self.groups >= times[cut]
        train, test = np.flatnonzero(~test_mask), np.flatnonzero(test_mask)
        return (
            train,
            test,
            {
                "train_times": times[:cut].tolist(),
                "test_times": times[cut:].tolist(),
                "train_time_range": [float(times[0]), float(times[cut - 1])],
                "test_time_range": [float(times[cut]), float(times[-1])],
                "n_train_rows_before_sampling": len(train),
                "n_test_rows": len(test),
                "split_basis": "sorted unique time coordinates only; latest times held out",
                "generalization": "within_record_temporal_extrapolation",
                "preprocessing": "raw coordinates and responses; no fitted scaling or derivatives",
            },
        )

    def train_test_split(self, test_size=0.2, seed=0):
        train, test, _ = self.temporal_holdout(test_size)
        self.X_train, self.y_train = self.X[train], self.y[train]
        self.X_test, self.y_test = self.X[test], self.y[test]

    def get_data(self):
        return {
            **super().get_data(),
            "x": self.x,
            "t": self.t,
            "values": self.values,
            "protocol_metadata": deepcopy(self.protocol_metadata),
        }


def load_fde_response(name, *, root=None, coordinates=None):
    """Load a declared layout; headerless fields require caller-supplied axes.

    ``coordinates`` is only accepted for the two DAT files without coordinates.
    It must contain x, t, layout (tx or xt), and a nonempty evidence description.
    """
    from scipy.io import loadmat

    if name not in FDE_SPECS:
        raise ValueError(f"Unknown FDE response dataset: {name!r}")
    spec = FDE_SPECS[name]
    path = (Path(root) if root is not None else ROOT) / spec["path"]
    metadata = response_evidence(name)
    if spec["layout"] is None:
        values = _numeric(np.loadtxt(path, ndmin=2), "DAT field")
        if coordinates is None:
            raise ValueError(
                f"source_blocked: {name} requires explicit x,t,layout and coordinate evidence"
            )
        if not isinstance(coordinates, dict) or set(coordinates) != {
            "x",
            "t",
            "layout",
            "evidence",
        }:
            raise ValueError("FDE coordinates must contain exactly x, t, layout and evidence")
        if coordinates["layout"] not in {"tx", "xt"}:
            raise ValueError("FDE coordinate layout must be tx or xt")
        if not isinstance(coordinates["evidence"], str) or not coordinates["evidence"].strip():
            raise ValueError("FDE coordinate evidence must be a nonempty description")
        x, t = coordinates["x"], coordinates["t"]
        if coordinates["layout"] == "xt":
            values = values.T
        metadata.update(
            coordinate_status="caller_declared",
            source_layout=coordinates["layout"],
            coordinate_evidence=coordinates["evidence"],
        )
    else:
        if coordinates is not None:
            raise ValueError("FDE embedded coordinates cannot be overridden")
        with path.open("rb") as stream:
            raw = loadmat(stream)
        missing = set(spec["keys"]) - raw.keys()
        if missing:
            raise ValueError(f"FDE {name}: missing MAT fields {sorted(missing)}")
        if spec["layout"] == "curve":
            t, values = (raw[key] for key in spec["keys"])
            x = None
        else:
            x, t, values = (raw[key] for key in spec["keys"])
    return FractionalResponseDataset(name, t, values, x=x, metadata=metadata)


def load_fde_pde(name, *, root=None, coordinates=None):
    """Expose the same observed field as a scalar 1D PDE diagnostic grid.

    No fractional derivative, equation truth, boundary or extra spatial axis is
    supplied. Curve-only records are deliberately absent from this interface.
    """
    from ._equations import EquationTarget

    if name not in FDE_PDE_SPECS:
        raise ValueError(f"Unknown FDE PDE dataset: {name!r}")
    response = load_fde_response(FDE_PDE_SPECS[name], root=root, coordinates=coordinates)
    dataset = GridPDEDataset(
        equation_name=name,
        pde_data=None,
        x=response.x,
        t=response.t,
        usol=response.values.T.copy(),
        epi=0.0,
        domain={
            "x": (float(response.x[0]), float(response.x[-1])),
            "t": (float(response.t[0]), float(response.t[-1])),
        },
        legacy=True,
    )
    dataset.registry_name = name
    dataset.state_vars = ["u"]
    dataset.equation_targets = [EquationTarget("u_t")]
    dataset.protocol_metadata = {
        **pde_evidence(name),
        **deepcopy(response.protocol_metadata),
        "target_definition": "integer evolution RHS or implicit integer derivative relation",
        "temporal_order_status": "diagnostic_integer_order; fractional_order_unknown",
        "split_protocol": "temporal_block_with_support_gap",
        "pde_axis_order": ["x", "t"],
        "reference_status": "unknown",
    }
    return dataset
