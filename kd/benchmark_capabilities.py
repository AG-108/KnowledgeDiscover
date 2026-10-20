"""Capabilities of the complete benchmark adapters, not of upstream methods."""

from dataclasses import dataclass

import numpy as np

from kd.benchmark_execution import SkipCase
from kd.dataset._base import GridPDEDataset
from kd.dataset._equations import SteadyPDEDataset, state_fields, targets
from kd.evaluation.fractional import method_view


@dataclass(frozen=True)
class PDECapability:
    kinds: tuple = ("explicit",)
    time_orders: tuple = (1,)
    spatial_dimensions: tuple = (1,)
    max_channels: int | None = 1
    grid_only: bool = True
    uniform_axes: bool = False
    boundary: str | None = None
    scalar_u_target: bool = True
    coefficient_axes: tuple = ()


PDE_CAPABILITIES = {
    name: PDECapability(uniform_axes=name in {"dscv", "spr", "sga", "weakform"})
    for name in ("dscv", "spr", "sga", "dlga", "deepmod", "pdefind",
                 "weakform", "eqgpt", "sindy", "pysindy", "reference")
}
PDE_CAPABILITIES.update(
    pdefind_nd=PDECapability(time_orders=(1, 2), spatial_dimensions=(1, 2, 3),
                            max_channels=None, grid_only=False, scalar_u_target=False,
                            coefficient_axes=("x", "y", "z", "t")),
    implicit_pde=PDECapability(kinds=("implicit",), time_orders=(0, 1, 2),
                              spatial_dimensions=(1, 2, 3), max_channels=None,
                              grid_only=False, scalar_u_target=False),
    pdenet=PDECapability(spatial_dimensions=(2,), max_channels=None,
                        uniform_axes=True, boundary="periodic", scalar_u_target=False),
)


def failure_status(exc):
    """Stable reason codes shared by preparation, preflight and terminal rows."""
    message = str(exc)
    for status in ("source_blocked", "method_unsupported", "insufficient_support"):
        if status in message:
            return status
    if isinstance(exc, FileNotFoundError):
        return "missing_asset"
    if isinstance(exc, ImportError):
        return "missing_dependency"
    if isinstance(exc, SkipCase):
        return "instance_ineligible"
    return "error"


def validate_pde(dataset, model):
    """Check semantics before any adapter can fit or score a different target."""
    if model not in PDE_CAPABILITIES:
        raise SkipCase(f"method_unsupported: no PDE capability declared for {model}")
    capability = PDE_CAPABILITIES[model]
    dataset = method_view(dataset, model)
    name = dataset.equation_name.lower()
    if name in {"wake", "wdwake"}:
        raise SkipCase("source_blocked: WDwake physical coordinates and boundaries are unverified")
    if name == "tlc_wave_darcy":
        raise SkipCase("source_blocked: temporal order, coefficient and interface evidence required")
    dependencies = set(getattr(dataset, "protocol_metadata", {}).get("coefficient_dependencies", []))
    if dependencies - set(capability.coefficient_axes):
        raise SkipCase(f"method_unsupported: {model} adapter cannot discover coefficient "
                       f"functions of {sorted(dependencies)}")
    steady = isinstance(dataset, SteadyPDEDataset)
    if not isinstance(dataset, GridPDEDataset) and not (
        getattr(dataset, "geometry", None) or {}
    ).get("verified"):
        raise SkipCase("source_blocked: scatter valid domain/holes/interfaces are unverified")
    if capability.grid_only and not isinstance(dataset, GridPDEDataset):
        raise SkipCase("method_unsupported: PDE baselines require a regular grid; "
                       "scatter data need an explicit adapter")
    dim = len(dataset.spatial_vars)
    if dim not in capability.spatial_dimensions or (
        capability.max_channels is not None and dataset.n_response > capability.max_channels
    ):
        detail = ("two spatial dimensions" if model == "pdenet"
                  else "a scalar field in one spatial dimension")
        raise SkipCase(f"method_unsupported: {model} adapter requires {detail}")
    # Boundary checks preserve the stronger source-evidence diagnosis for PDE-Net.
    if capability.boundary:
        from kd.evaluation.equation import periodic_grid_options
        try:
            periodic_grid_options(dataset)
        except SkipCase as exc:
            raise SkipCase(f"source_blocked: {exc}") from exc
    declared = targets(dataset)
    if not declared:
        raise SkipCase("method_unsupported: no equation targets declared")
    for target in declared:
        if target.kind not in capability.kinds or target.time_order not in capability.time_orders:
            raise SkipCase(
                f"method_unsupported: {model} adapter cannot bind target {target.key} "
                f"(kind={target.kind}, time_order={target.time_order}); "
                f"supports kinds={capability.kinds}, time_orders={capability.time_orders}"
            )
    if capability.scalar_u_target and (
        len(declared) != 1 or declared[0].field != "u" or declared[0].known_source is not None
    ):
        raise SkipCase(f"method_unsupported: {model} adapter fits one scalar u_t RHS")
    if model == "pdenet" and (
        {target.field for target in declared} != set(state_fields(dataset))
        or any(target.known_source is not None for target in declared)
    ):
        raise SkipCase("method_unsupported: PDE-Net jointly evolves every input channel; "
                       "auxiliary fields and anchored sources need an explicit adapter")
    if steady and model != "implicit_pde":
        raise SkipCase("method_unsupported: explicit adapter requires evolution RHS")
    if isinstance(dataset, GridPDEDataset):
        coords = [np.asarray(v) for v in dataset.coords_spatial.values()] + [np.asarray(dataset.t)]
        if any(len(c) < 3 or not np.isfinite(c).all() or not np.all(np.diff(c) > 0)
               for c in coords):
            raise SkipCase("instance_ineligible: PDE fitting requires >=3 finite, "
                           "strictly increasing points per axis")
        if not np.isfinite(dataset.usol).all():
            raise SkipCase("source_blocked: PDE grid contains NaN/Inf; "
                           "an explicit validity mask adapter is required")
        if capability.uniform_axes and any(
            not np.allclose(np.diff(c), np.diff(c)[0], rtol=1e-3, atol=1e-10) for c in coords
        ):
            raise SkipCase(f"method_unsupported: {model} adapter requires uniform axes")
