"""Equation discovery bridge with splits before derivative estimation."""

import copy

import numpy as np
from scipy.spatial import cKDTree

from kd.dataset._base import GridPDEDataset
from kd.dataset._derivatives import default_stencil, namespace
from kd.dataset._domain import spatial_blocks
from kd.dataset._equations import SteadyPDEDataset, equation_spec, state_fields, targets
from kd.evaluation.fractional import is_pde_diagnostic, method_view, temporal_views
from kd.model.kd_equation import (
    SparseEquationModel,
    coordinate_basis_symbols,
    library,
    relative_residual,
)

METHODS = {"pdefind_nd", "implicit_pde"}


def validate(dataset, model):
    from kd.benchmark_capabilities import validate_pde

    validate_pde(dataset, model)


def problems(dataset, case):
    # Enumeration has no derivative estimation, reference-dependent target selection or fit.
    dataset = method_view(dataset, case.get("reference_adapter", case["model"]))
    for target in targets(dataset):
        yield {
            "target": target.key,
            "equation_target": target.to_dict(),
            "ground_truth": None,
            "metadata": equation_spec(dataset, target),
            "split": "spatial_block_with_support_gap"
            if isinstance(dataset, SteadyPDEDataset)
            else "temporal_block_with_support_gap",
            "X_train": np.empty((0, 0)),
            "X_test": np.empty((0, 0)),
        }


def _spatial_views(dataset, case, model):
    from kd.benchmark_execution import SkipCase

    if dataset.points is None:
        axes = list(dataset.coords.values())
        points = np.column_stack([a.reshape(-1) for a in np.meshgrid(*axes, indexing="ij")])
        values = dataset.values.reshape(len(dataset.state_vars), -1).T
        sources = {k: np.asarray(v).reshape(-1) for k, v in dataset.known_sources.items()}
    else:
        points, values, sources = dataset.points, dataset.values, dataset.known_sources
    labels, regions = spatial_blocks(points)
    count = min(len(regions) - 1, max(1, round(len(regions) * case["run_params"]["test_size"])))
    held = np.random.default_rng(case["seed"]).permutation(len(regions))[:count]
    test_ids, train_ids = (
        np.flatnonzero(np.isin(labels, held)),
        np.flatnonzero(~np.isin(labels, held)),
    )
    # Coordinate-only support radius, fixed before inspecting any field.
    distances = cKDTree(points).query(points, k=min(len(points), model.neighbors))[0]
    radius = float(np.median(distances[:, -1]))
    distance = cKDTree(points[test_ids]).query(points[train_ids], k=1)[0]
    kept = train_ids[distance > radius]
    if len(kept) < 20 or len(test_ids) < 20:
        raise SkipCase("insufficient_support: spatial gap leaves too few points")
    train_ids = kept

    def view(ids):
        result = SteadyPDEDataset(
            dataset.equation_name,
            values[ids],
            points=points[ids],
            state_vars=dataset.state_vars,
            spatial_vars=dataset.spatial_vars,
            geometry=dataset.geometry,
            known_sources={k: np.asarray(v)[ids] for k, v in sources.items()},
            equation_targets=dataset.equation_targets,
            protocol_metadata=dataset.protocol_metadata,
        )
        if hasattr(dataset, "region_ids"):
            result.region_ids = np.asarray(dataset.region_ids)[ids].copy()
        result.derivative_max_radius = radius
        return result

    return (
        view(train_ids),
        view(test_ids),
        {
            "kind": "within_field_spatial_blocks",
            "held_blocks": held.tolist(),
            "support_gap": radius,
            "excluded_points": len(points) - len(train_ids) - len(test_ids),
            "n_train_observations": len(train_ids),
            "n_test_observations": len(test_ids),
            "generalization": "same-field observed residual; not cross-geometry or BVP",
        },
    )


def _temporal_views(dataset, case):
    from kd.benchmark_execution import SkipCase
    from kd.benchmark_preparation import slice_pde_time

    if is_pde_diagnostic(dataset):
        return temporal_views(dataset, case["run_params"]["test_size"])
    nt = len(dataset.t)
    # One frame on the held-out side of cut is the support gap, so reserving
    # four frames leaves at least three actual test observations.
    cut = nt - max(4, round(nt * case["run_params"]["test_size"]))
    # A three-point temporal stencil has radius one. No shared observation frames.
    train_stop, test_start = cut - 1, cut + 1
    if train_stop < 5 or nt - test_start < 3:
        raise SkipCase("insufficient_support: temporal gap requires five training and "
                       "three test frames plus two unused frames")
    if isinstance(dataset, GridPDEDataset):
        train = slice_pde_time(dataset, 0, train_stop)
        test = slice_pde_time(dataset, test_start, nt)
    else:
        train, test = copy.deepcopy(dataset), copy.deepcopy(dataset)
        for result, sl in ((train, slice(0, train_stop)), (test, slice(test_start, nt))):
            result.t = dataset.t[sl].copy()
            result.usol = dataset.usol[:, sl].copy()
    return (
        train,
        test,
        {
            "kind": "temporal_block_with_support_gap",
            "support_gap_frames": 2,
            "train_frame_stop": train_stop,
            "test_frame_start": test_start,
            "n_train_frames": train_stop,
            "n_test_frames": nt - test_start,
        },
    )


def prepare_views(dataset, case, model=None):
    """Choose observation views and check cheap coordinate support before fitting."""
    from kd.benchmark_execution import SkipCase

    if model is None:
        model = SparseEquationModel(**case["model_params"])
    coordinates = set(dataset.spatial_vars)
    if not isinstance(dataset, SteadyPDEDataset):
        coordinates.add("t")
    basis_symbols = coordinate_basis_symbols(model.coordinate_basis, coordinates)
    dependencies = set(getattr(dataset, "protocol_metadata", {}).get("coefficient_dependencies", []))
    if dependencies - basis_symbols:
        raise SkipCase("method_unsupported: variable coefficients require a configured "
                       f"coordinate_basis involving {sorted(dependencies)}")
    if isinstance(dataset, SteadyPDEDataset):
        return _spatial_views(dataset, case, model)
    train, test, split = _temporal_views(dataset, case)
    if isinstance(train, GridPDEDataset):
        stencil = default_stencil(model.spatial_order)
        if any(len(axis) < stencil for axis in train.coords_spatial.values()):
            raise SkipCase("insufficient_support: spatial derivative "
                           f"library requires at least {stencil} points per axis")
    return train, test, split


def run(dataset, problem, case):
    from kd.benchmark_execution import fit_call

    dataset = method_view(dataset, case.get("reference_adapter", case["model"]))
    validate(dataset, case.get("reference_adapter", case["model"]))
    model = SparseEquationModel(**case["model_params"])
    steady = isinstance(dataset, SteadyPDEDataset)
    target = next(t for t in targets(dataset) if t.key == problem["target"])
    train, test, split = prepare_views(dataset, case, model)
    options = dict(
        spatial_order=model.spatial_order,
        temporal_order=0 if steady else target.time_order,
        neighbors=model.neighbors,
        degree=model.degree,
    )
    train_ns, train_diag = namespace(train, **options)
    test_ns, test_diag = namespace(test, **options)
    fields = state_fields(dataset)
    implicit = target.kind == "implicit"
    X, terms = library(train_ns, fields, implicit=implicit, coordinate_basis=model.coordinate_basis)
    Xt, test_terms = library(
        test_ns, fields, implicit=implicit, coordinate_basis=model.coordinate_basis
    )
    if terms != test_terms:
        raise RuntimeError("Observation-independent candidate library changed")
    if target.known_source is not None:
        # Anchored operator track: a free constant would be an algebraic copy of
        # constant forcing. No independent constant offset is in this variant.
        selected = [i for i, term in enumerate(terms) if term != "1"]
        X, Xt = X[:, selected], Xt[:, selected]
        terms = [terms[i] for i in selected]
        if target.known_source not in train_ns:
            raise ValueError("Task-declared known source missing")
        y, yt = (
            -train_ns[target.known_source].reshape(-1),
            -test_ns[target.known_source].reshape(-1),
        )
        # No column equal to an anchor supplied as y; task forcing is excluded from library.
    elif not implicit:
        y, yt = train_ns[target.key].reshape(-1), test_ns[target.key].reshape(-1)
    else:
        y = yt = None
    constraints = []
    for declaration in getattr(dataset, "protocol_metadata", {}).get("known_constraints", []):
        constraints.append(declaration.split("=", 1)[0])
    if case["model"] == "reference":
        from kd.evaluation.ode_rollout import numeric_expression
        from kd.evaluation.reference import abstain
        from run_benchmark import expression_sympy

        reference = getattr(dataset, "protocol_metadata", {}).get("source_reference")
        if isinstance(reference, dict):
            reference = reference.get(target.key)
        if not reference:
            return abstain("abstained_missing_reference")
        names = list(train_ns)
        try:
            numeric_expression(reference, names)
            import sympy as sp

            declared = sp.expand(expression_sympy(reference, names))
            expanded = [sp.expand(expression_sympy(term, names)) for term in terms]
            coefficients = np.zeros(len(terms))
            for component in sp.Add.make_args(declared):
                coefficient, monomial = component.as_coeff_Mul()
                matches = [i for i, basis in enumerate(expanded) if basis == monomial]
                if len(matches) != 1 or not coefficient.is_number:
                    raise ValueError("Reference term is outside the declared scoring library")
                coefficients[matches[0]] += float(coefficient)
        except (ValueError, TypeError, SyntaxError) as exc:
            return {**abstain("abstained_unsupported_reference"), "reference_error": str(exc)}
        if implicit:
            norm = np.linalg.norm(coefficients)
            if norm == 0:
                return abstain("abstained_trivial_reference")
            coefficients /= norm
            good = np.isfinite(X).all(axis=1)
            scales = np.sqrt(np.mean(X[good] ** 2, axis=0))
            model.train_scale_ = float(np.linalg.norm(coefficients * scales))
        model.coefficients_ = coefficients
        model.expression_ = str(reference)
        model.diagnostics_ = {
            "train_rows": int(np.isfinite(X).all(axis=1).sum()),
            "ambiguous": False,
            "role": "source_reference; no fitting",
        }
    else:
        fit_call(model.fit, X, terms, y, excluded_relations=constraints)
    coef = model.coefficients_
    if y is not None:
        full_terms = terms + [target.known_source or target.key]
        full_coef = np.r_[coef, 1.0 if target.known_source else -1.0]
        training = np.column_stack([X, -y if target.known_source else y])
        testing = np.column_stack([Xt, -yt if target.known_source else yt])
        good = np.isfinite(training).all(axis=1)
        scales = np.sqrt(np.mean(training[good] ** 2, axis=0))
        full_coef /= np.linalg.norm(full_coef)
        scale = float(np.sqrt(np.sum((full_coef * scales) ** 2)))
    else:
        full_terms, full_coef, testing = terms, coef, Xt
        scale = model.train_scale_
    metric = relative_residual(testing, full_coef, scale)
    reference = getattr(dataset, "protocol_metadata", {}).get("source_reference")
    if isinstance(reference, dict):
        reference = reference.get(target.key)
    reference_metrics = None
    if reference:
        import sympy as sp

        expression = sp.sympify(reference)
        if not implicit:
            expression = sp.Symbol(target.key) - expression
        symbols = sorted(expression.free_symbols, key=str)
        if all(str(s) in test_ns for s in symbols):
            residual = np.asarray(
                sp.lambdify(symbols, expression, "numpy")(*[test_ns[str(s)] for s in symbols])
            )
            finite = np.isfinite(residual)
            reference_metrics = {
                "mse": float(np.mean(residual[finite] ** 2)) if finite.any() else None,
                "coverage": float(finite.mean()),
                "role": "source numerical diagnostic only",
            }
    return {
        **metric,
        "expression": model.expression_,
        "expression_complexity": int((abs(coef) > 1e-12).sum()),
        "equation_expression": {
            "target": target.to_dict(),
            "terms": full_terms,
            "coefficients": full_coef.tolist(),
        },
        "equation_protocol_version": "equation-v1",
        "metric_target": "observed_equation_residual",
        "equation_residual_normalization": "training_term_RMS_and_coefficient_unit_norm",
        **(
            {
                "equation_residual_status": "ok"
                if metric["equation_residual_points"]
                else "abstain_insufficient_support"
            }
            if is_pde_diagnostic(dataset)
            else {}
        ),
        "normalization_train_scale": scale,
        "split": split,
        "n_train": model.diagnostics_["train_rows"],
        "n_test": metric["equation_residual_points"],
        "ground_truth_expression": reference,
        "recovery_eligible": False,
        "recovery_evaluator_status": (
            "not_applicable"
            if is_pde_diagnostic(dataset)
            else "ambiguous"
            if model.diagnostics_["ambiguous"]
            else "source_reference_not_certified"
        ),
        "equation_rollout_status": "abstain",
        "prediction_status": "abstain",
        "prediction_abstain_reason": "No independently validated boundary/initial-value solver for this subtrack",
        "method_provenance": {
            "method": case["model"],
            "coordinate_basis": model.coordinate_basis,
            "fit_unit": "one equation target, independent fit",
        },
        "details": {
            "equation": equation_spec(dataset, target),
            "train_derivatives": train_diag,
            "test_derivatives": test_diag,
            "discovery": model.diagnostics_,
            "source_reference_diagnostic": reference_metrics,
        },
    }


def periodic_grid_options(dataset):
    """Actual spacing and endpoint convention for PDE-Net periodic convolutions."""
    from kd.benchmark_execution import SkipCase

    if dataset.equation_name.lower() in {"wake", "wdwake"}:
        raise SkipCase("WDwake physical coordinates and boundary conditions are not verified")
    metadata = getattr(dataset, "protocol_metadata", {})
    boundary = metadata.get("boundary", {})
    if (
        not isinstance(boundary, dict)
        or boundary.get("verified") is not True
        or boundary.get("kind") != "periodic"
    ):
        raise SkipCase(
            "PDE-Net anisotropic/rectangular grids require verified periodic boundary evidence"
        )
    if boundary.get("endpoint") != "excluded":
        raise SkipCase(
            "PDE-Net periodic endpoint convention must explicitly exclude duplicate endpoints"
        )
    axes = [np.asarray(a).reshape(-1) for a in dataset.coords_spatial.values()]
    if len(axes) != 2:
        raise SkipCase("PDE-Net requires two spatial axes")
    coordinates = [*axes, np.asarray(dataset.t).reshape(-1)]
    if any(
        len(a) < 3
        or not np.allclose(np.diff(a), np.diff(a)[0], rtol=1e-7, atol=1e-12)
        or np.diff(a)[0] <= 0
        for a in coordinates
    ):
        raise SkipCase("PDE-Net requires uniform increasing space and time coordinates")
    dx = [float(np.diff(a)[0]) for a in axes]
    return {
        "dx": dx,
        "mesh_size": [len(a) for a in axes],
        "mesh_bound": [
            [float(a[0]) for a in axes],
            [float(a[0] + len(a) * h) for a, h in zip(axes, dx)],
        ],
    }
