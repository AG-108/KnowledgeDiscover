"""Interior finite differences and geometry-guarded local polynomial derivatives.

This module has no benchmark or model dependency and works without a time axis.
Unsupported neighborhoods retain NaNs and explicit diagnostics.
"""

import itertools
import math

import numpy as np
from scipy.spatial import cKDTree

from ._equations import Derivative, state_fields


def default_stencil(order):
    """Spatial centered stencil size used by derivatives and support checks."""
    return max(5, 2 * (order // 2) + 3)


def axis_derivative(values, coordinate, order=1, axis=0, stencil=None):
    """Centered polynomial FD on explicit coordinates; no boundary wrapping.

    At least second order for uniform grids and derivatives up to order four.
    Temporal callers use three points for first/second derivatives.
    """
    values = np.asarray(values, dtype=float)
    coordinate = np.asarray(coordinate, dtype=float).reshape(-1)
    if not isinstance(order, int) or order < 0 or order > 4:
        raise ValueError("Supported derivative orders are 0 through 4")
    if values.shape[axis] != len(coordinate) or not np.isfinite(coordinate).all():
        raise ValueError("Coordinate/value axis mismatch")
    if not np.all(np.diff(coordinate) > 0):
        raise ValueError("Derivative coordinates must be strictly increasing")
    if order == 0:
        return values.copy()
    stencil = stencil or default_stencil(order)
    if stencil % 2 != 1 or stencil <= order:
        raise ValueError("A derivative stencil must be odd and longer than its order")
    result = np.full_like(values, np.nan, dtype=float)
    radius = stencil // 2
    moved, output = np.moveaxis(values, axis, 0), np.moveaxis(result, axis, 0)
    for index in range(radius, len(coordinate) - radius):
        offsets = coordinate[index - radius : index + radius + 1] - coordinate[index]
        scale = np.max(np.abs(offsets))
        powers = np.arange(stencil)
        vandermonde = (offsets[:, None] / scale) ** powers
        target = np.zeros(stencil)
        target[order] = math.factorial(order) / scale**order
        weights = np.linalg.solve(vandermonde.T, target)
        output[index] = np.tensordot(
            weights, moved[index - radius : index + radius + 1], axes=(0, 0)
        )
    return result


def grid_derivative(field, axes, multi_index, t=None):
    if len(axes) != len(multi_index.spatial_orders):
        raise ValueError("Derivative spatial dimension mismatch")
    result = np.asarray(field, dtype=float).copy()
    for axis, (coordinate, order) in enumerate(zip(axes, multi_index.spatial_orders)):
        if order:
            result = axis_derivative(result, coordinate, order, axis)
    if multi_index.time_order:
        if t is None:
            raise ValueError("Temporal derivative requested for a steady field")
        result = axis_derivative(result, t, multi_index.time_order, -1, stencil=3)
    return result


def segments_valid(origin, destinations, geometry):
    """Guard every stencil segment against declared holes/interfaces."""
    destinations = np.asarray(destinations, dtype=float)
    good = np.ones(len(destinations), dtype=bool)
    if geometry is None or geometry.get("verified") is not True:
        raise ValueError("Scatter geometry/regions must be explicitly verified")
    if geometry.get("kind") not in {"box", "disk", "annulus"}:
        raise ValueError("Unsupported geometry guard")
    holes = list(geometry.get("holes", []))
    if geometry["kind"] in {"disk", "annulus"}:
        center = np.asarray(geometry.get("center", [0] * len(origin)))
        outer = geometry["outer_radius"]
        good &= np.linalg.norm(destinations - center, axis=1) <= outer * (1 + 1e-12)
        if np.linalg.norm(origin - center) > outer * (1 + 1e-12):
            good[:] = False
        if geometry.get("inner_radius", 0) > 0:
            holes.append({"center": center, "radius": geometry["inner_radius"]})
    if geometry["kind"] == "box" and "bounds" in geometry:
        bounds = np.asarray(geometry["bounds"])
        good &= np.all((destinations >= bounds[:, 0]) & (destinations <= bounds[:, 1]), axis=1)
    delta = destinations - origin
    denom = np.sum(delta**2, axis=1)
    for hole in holes:
        center = np.asarray(hole["center"])
        fraction = np.sum((center - origin) * delta, axis=1) / np.where(denom > 0, denom, 1)
        nearest = origin + np.clip(fraction, 0, 1)[:, None] * delta
        good &= np.linalg.norm(nearest - center, axis=1) >= hole["radius"] * (1 - 1e-12)
    return good


class LocalPolynomial:
    """Local cubic least squares, with region and segment support restrictions."""

    def __init__(
        self,
        points,
        *,
        degree=3,
        neighbors=40,
        geometry=None,
        region_ids=None,
        condition_limit=1e8,
        max_radius=None,
    ):
        self.points = np.asarray(points, dtype=float)
        if self.points.ndim != 2 or not np.isfinite(self.points).all():
            raise ValueError("Local polynomial needs finite point coordinates")
        self.dimension = self.points.shape[1]
        if degree < 1 or degree > 4:
            raise ValueError("Local polynomial degree must be 1 through 4")
        self.powers = [
            p
            for p in itertools.product(range(degree + 1), repeat=self.dimension)
            if sum(p) <= degree
        ]
        self.powers.sort(key=lambda p: (sum(p), p))
        self.condition = np.full(len(self.points), np.inf)
        self.weights = {}
        self.support = np.zeros(len(self.points), dtype=bool)
        self.neighborhoods = []
        k = min(len(self.points), max(neighbors, len(self.powers) + 1))
        _, neighbors_array = cKDTree(self.points).query(self.points, k=k)
        regions = None if region_ids is None else np.asarray(region_ids)
        if regions is not None and regions.shape != (len(self.points),):
            raise ValueError("Region labels disagree with point count")
        for index, ids in enumerate(neighbors_array):
            ids = np.atleast_1d(ids)
            allowed = segments_valid(self.points[index], self.points[ids], geometry)
            if regions is not None:
                allowed &= regions[ids] == regions[index]
            ids = ids[allowed]
            offsets = self.points[ids] - self.points[index]
            if max_radius is not None:
                keep = np.linalg.norm(offsets, axis=1) <= max_radius
                offsets, ids = offsets[keep], ids[keep]
            self.neighborhoods.append(ids)
            if len(ids) < len(self.powers):
                continue
            scale = np.max(np.abs(offsets), axis=0)
            if np.any(scale == 0):
                continue
            normalized = offsets / scale
            design = np.column_stack([np.prod(normalized**p, axis=1) for p in self.powers])
            singular = np.linalg.svd(design, compute_uv=False)
            self.condition[index] = singular[0] / max(singular[-1], np.finfo(float).tiny)
            if self.condition[index] > condition_limit:
                continue
            inverse = np.linalg.pinv(design, rcond=1 / condition_limit)
            self.weights[index] = (inverse, scale)
            self.support[index] = True

    def derivative(self, values, orders):
        values = np.asarray(values, dtype=float)
        if values.shape[0] != len(self.points) or tuple(orders) not in self.powers:
            raise ValueError("Values/order incompatible with polynomial stencil")
        out = np.full_like(values, np.nan, dtype=float)
        row = self.powers.index(tuple(orders))
        factor = math.prod(math.factorial(n) for n in orders)
        for index, (inverse, scale) in self.weights.items():
            weights = inverse[row] * factor / np.prod(scale**orders)
            out[index] = np.tensordot(weights, values[self.neighborhoods[index]], axes=(0, 0))
        return out

    def diagnostics(self):
        finite = self.condition[np.isfinite(self.condition)]
        return {
            "neighborhood_count": len(self.points),
            "supported_neighborhoods": int(self.support.sum()),
            "condition_limit_failures": int((~self.support).sum()),
            "max_supported_condition": float(self.condition[self.support].max())
            if self.support.any()
            else None,
            "max_observed_condition": float(finite.max()) if finite.size else None,
        }


def namespace(dataset, *, spatial_order=2, temporal_order=0, neighbors=40, degree=3):
    """Observation-only namespace; never reads reference equations/coef_dict."""
    from ._base import GridPDEDataset
    from ._equations import SteadyPDEDataset

    steady = isinstance(dataset, SteadyPDEDataset)
    grid = isinstance(dataset, GridPDEDataset) or (steady and dataset.points is None)
    t = None if steady else np.asarray(dataset.t).reshape(-1)
    if grid:
        coordinates = list((dataset.coords if steady else dataset.coords_spatial).values())
        dimension = len(coordinates)
        values = (
            dataset.values if steady else (dataset.usol[None] if dataset.legacy else dataset.usol)
        )
        coordinate_fields = np.meshgrid(*coordinates, *([] if t is None else [t]), indexing="ij")
        polynomial = None
    else:
        dimension = dataset.points.shape[1]
        values = (
            dataset.values.T
            if steady
            else (
                dataset.usol[None] if dataset.usol.ndim == 2 else np.moveaxis(dataset.usol, -1, 0)
            )
        )
        geometry = getattr(dataset, "geometry", None)
        polynomial = LocalPolynomial(
            dataset.points,
            neighbors=neighbors,
            degree=degree,
            geometry=geometry,
            region_ids=getattr(dataset, "region_ids", None),
            max_radius=getattr(dataset, "derivative_max_radius", None),
        )
        coordinate_fields = [
            dataset.points[:, i]
            if steady
            else np.broadcast_to(dataset.points[:, i, None], values.shape[1:])
            for i in range(dimension)
        ]
        coordinates = None
    fields = state_fields(dataset)
    if len(fields) != len(values) or len(set(fields)) != len(fields):
        raise ValueError("Field/channel declaration disagrees with observed values")
    result = {"1": np.ones(values.shape[1:])}
    for axis, array in zip("xyz", coordinate_fields[:dimension]):
        result[axis] = array
    if t is not None:
        # Preserve absolute observed time across train/test views and coefficient bases.
        result["t"] = np.broadcast_to(t, values.shape[1:])
    for channel, field in enumerate(fields):
        result[field] = values[channel]
        for orders in itertools.product(range(spatial_order + 1), repeat=dimension):
            if not 0 < sum(orders) <= spatial_order:
                continue
            key = Derivative(field, tuple(orders))
            result[key.symbol] = (
                grid_derivative(values[channel], coordinates, key, t)
                if grid
                else polynomial.derivative(values[channel], np.array(orders))
            )
        if temporal_order:
            for order in range(1, temporal_order + 1):
                key = Derivative(field, (0,) * dimension, order)
                result[key.symbol] = axis_derivative(values[channel], t, order, -1, stencil=3)
            for axis in range(dimension):
                orders = tuple(int(i == axis) for i in range(dimension))
                key = Derivative(field, orders, 1)
                result[key.symbol] = axis_derivative(
                    result[Derivative(field, orders).symbol], t, 1, -1, stencil=3
                )
    result.update(getattr(dataset, "known_sources", {}))
    diagnostics = {
        "spatial_dimensions": dimension,
        "fields": fields,
        "grid": grid,
        "derivatives": "interior_polynomial_fd" if grid else "geometry_guarded_local_polynomial",
        "spatial_order": spatial_order,
        "temporal_order": temporal_order,
    }
    if polynomial is not None:
        diagnostics.update(polynomial.diagnostics())
    return result, diagnostics
