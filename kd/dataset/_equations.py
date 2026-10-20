"""Serializable equation targets and a spatial-only steady dataset."""

import itertools
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class Derivative:
    field: str
    spatial_orders: Tuple[int, ...]
    time_order: int = 0

    def __post_init__(self):
        if (
            len(self.spatial_orders) not in {1, 2, 3}
            or not self.field.isidentifier()
            or any(not isinstance(n, int) or n < 0 for n in (*self.spatial_orders, self.time_order))
        ):
            raise ValueError("Invalid derivative multi-index")

    @property
    def symbol(self):
        suffix = (
            "".join(axis * n for axis, n in zip("xyz", self.spatial_orders)) + "t" * self.time_order
        )
        return self.field + ("_" + suffix if suffix else "")

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class EquationTarget:
    key: str
    field: str = "u"
    time_order: int = 1
    kind: str = "explicit"
    known_source: Optional[str] = None
    role: str = "discover"

    def __post_init__(self):
        if self.kind not in {"explicit", "implicit"} or self.time_order not in {0, 1, 2}:
            raise ValueError("Invalid equation target")
        if self.kind == "explicit" and self.time_order == 0:
            raise ValueError("Explicit evolution needs a temporal derivative")

    def to_dict(self):
        return asdict(self)


class SteadyPDEDataset:
    """Observed spatial fields without a time axis, including explicit geometry.

    Grid values have shape (channels, *spatial_shape); scatter values are
    (points, channels). No coordinate, forcing or mask is fabricated.
    """

    def __init__(
        self,
        equation_name,
        values,
        *,
        state_vars=("u",),
        coords=None,
        points=None,
        spatial_vars=None,
        geometry=None,
        known_sources=None,
        equation_targets=None,
        protocol_metadata=None,
    ):
        self.equation_name = equation_name
        self.state_vars = list(state_vars)
        if len(set(self.state_vars)) != len(self.state_vars) or not self.state_vars:
            raise ValueError("State channels must be nonempty and unique")
        self.n_response = len(self.state_vars)
        self.geometry = deepcopy(geometry)
        self.known_sources = deepcopy(known_sources or {})
        self.protocol_metadata = deepcopy(protocol_metadata or {})
        self.equation_targets = equation_targets or [
            EquationTarget("implicit_relation", time_order=0, kind="implicit")
        ]
        if (coords is None) == (points is None):
            raise ValueError("Supply exactly one of grid coords or scatter points")
        self.values = np.asarray(values, dtype=float)
        if coords is not None:
            self.coords = {k: np.asarray(v, dtype=float).reshape(-1) for k, v in coords.items()}
            self.spatial_vars = list(self.coords)
            expected = (self.n_response, *(len(v) for v in self.coords.values()))
            if self.values.shape != expected:
                raise ValueError(f"Steady grid values {self.values.shape} disagree with {expected}")
            if any(
                len(v) < 3 or not np.isfinite(v).all() or not np.all(np.diff(v) > 0)
                for v in self.coords.values()
            ):
                raise ValueError("Steady grid axes must be finite and strictly increasing")
            self.points = None
        else:
            self.points = np.asarray(points, dtype=float)
            if self.points.ndim != 2 or self.points.shape[1] not in {1, 2, 3}:
                raise ValueError("Steady points require one to three spatial coordinates")
            self.spatial_vars = list(spatial_vars or "xyz"[: self.points.shape[1]])
            if len(self.spatial_vars) != self.points.shape[1]:
                raise ValueError("Spatial names disagree with points")
            if self.values.shape != (len(self.points), self.n_response):
                raise ValueError("Steady scatter values must be (points,channels)")
            if not np.isfinite(self.points).all():
                raise ValueError("Nonfinite steady coordinates")
            self.coords = None
        if not np.isfinite(self.values).all():
            raise ValueError("Nonfinite observations require an explicit validity mask")
        if any(
            np.asarray(v).shape != self.values.shape[1:]
            if self.points is None
            else np.asarray(v).shape != (len(self.points),)
            for v in self.known_sources.values()
        ):
            raise ValueError("Known sources must share the spatial observation shape")

    def get_data(self):
        return {
            "coords": self.coords,
            "points": self.points,
            "values": self.values,
            "state_vars": list(self.state_vars),
            "spatial_vars": list(self.spatial_vars),
            "known_sources": self.known_sources,
        }

    def get_size(self):
        return self.values.shape


def equation_spec(dataset, target):
    """JSON-safe observation and equation contract, without scoring truth."""
    metadata = deepcopy(getattr(dataset, "protocol_metadata", {}))
    steady = isinstance(dataset, SteadyPDEDataset)
    fields = state_fields(dataset)
    spatial = list(dataset.spatial_vars)
    return {
        "version": "equation-v1",
        "derivative_ast": [
            Derivative(field, tuple(orders)).to_dict()
            for field in fields
            for orders in itertools.product(range(3), repeat=len(spatial))
            if 0 < sum(orders) <= 2
        ]
        + [
            Derivative(field, (0,) * len(spatial), order).to_dict()
            for field in fields
            if field != "p"
            for order in range(1, target.time_order + 1)
        ],
        "mixed_derivative_ast": [
            Derivative(field, tuple(int(i == axis) for i in range(len(spatial))), 1).to_dict()
            for field in fields if field != "p"
            for axis in range(len(spatial)) if target.time_order > 0
        ],
        "temporal_order_status": metadata.get("temporal_order_status",
            "source_unverified" if dataset.equation_name == "tlc_wave_darcy" else "declared"),
        "target": target.to_dict(),
        "spatial_axes": spatial,
        "coordinate_units": metadata.get("coordinate_units", "unknown"),
        "state_fields": fields,
        "evolution_fields": [t.field for t in targets(dataset) if t.time_order],
        "auxiliary_fields": [f for f in fields if f not in [t.field for t in targets(dataset)]],
        "field_units": metadata.get("field_units", "unknown"),
        "boundary": metadata.get("boundary", "unknown"),
        "valid_domain": deepcopy(getattr(dataset, "geometry", None)),
        "known_parameters": metadata.get("known_parameters", {}),
        "known_sources": list(getattr(dataset, "known_sources", {})),
        "unknown_coefficient_functions": metadata.get("unknown_coefficient_functions", []),
        "coefficient_dependencies": metadata.get("coefficient_dependencies", []),
        "time_axis": None if steady else "t",
        "evaluation": ["observed_equation_residual"],
        "prediction": "abstain",
        "fit_unit": "one equation target with independent budget",
    }


def state_fields(dataset):
    if hasattr(dataset, "state_vars"):
        return list(dataset.state_vars)
    count = getattr(dataset, "n_response", getattr(dataset, "n_state", 1))
    return ["u", "v"][:count] if count <= 2 else [f"u{i}" for i in range(count)]


def targets(dataset):
    """Declared targets; pressure is auxiliary, never implicitly p_t."""
    if hasattr(dataset, "equation_targets"):
        return list(dataset.equation_targets)
    fields = state_fields(dataset)
    return [EquationTarget(f"{f}_t", field=f) for f in fields if f != "p"]
