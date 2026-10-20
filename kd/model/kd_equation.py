"""Observation-only sparse explicit and constrained implicit equation discovery."""

import itertools

import numpy as np
import sympy as sp


def canonical_terms(terms):
    """Merge algebraically equal terms and reject identically zero candidates."""
    canonical = [str(sp.expand(sp.sympify(term))) for term in terms]
    if any(sp.sympify(term) == 0 for term in canonical):
        raise ValueError("Identically zero candidate equation term")
    return list(dict.fromkeys(canonical))


def canonical_equation(coefficients, terms):
    """Canonical coefficient norm/sign after merging and cancellation."""
    merged = {}
    for coefficient, term in zip(coefficients, terms):
        symbolic = sp.expand(sp.sympify(term))
        for component in sp.Add.make_args(symbolic):
            number, factor = component.as_coeff_Mul()
            key = str(factor)
            merged[key] = merged.get(key, 0.0) + float(coefficient) * float(number)
    merged = {k: v for k, v in merged.items() if abs(v) > 1e-12 and k != "0"}
    if not merged:
        raise ValueError("Zero equation or cancelling/self-copy terms")
    keys = sorted(merged)
    coef = np.array([merged[k] for k in keys])
    coef /= np.linalg.norm(coef)
    if coef[0] < 0:
        coef *= -1
    return dict(zip(keys, coef.tolist()))


def coordinate_basis_symbols(coordinate_basis, coordinates):
    """Validate configured coefficient features against observed coordinates only."""
    symbols = set()
    for basis in coordinate_basis:
        current = {str(symbol) for symbol in sp.sympify(basis).free_symbols}
        if current - set(coordinates):
            raise ValueError("Coordinate basis may contain only observed coordinates")
        symbols.update(current)
    return symbols


def library(namespace, fields, *, implicit=False, coordinate_basis=()):
    """Same declared library for every equation in a method/subtrack."""
    coordinate_basis_symbols(coordinate_basis, set(namespace) & {"x", "y", "z", "t"})
    terms = ["1"] + list(fields)
    terms += [f"{field}**{power}" for field in fields if field != "p" for power in (2, 3)]
    derivatives = [
        key
        for key in namespace
        if "_" in key
        and key.split("_")[0] in fields
        and (implicit or "t" not in key.split("_", 1)[1])
    ]
    terms += derivatives
    first = [
        key
        for key in derivatives
        if len(key.split("_", 1)[1]) == 1 and key.split("_", 1)[1] in "xyz"
    ]
    # Pressure has a gauge and only appears through derivatives.
    terms = [term for term in terms if term != "p"]
    terms += [f"{f}*{key}" for f in fields if f != "p" for key in first]
    terms += [
        f"{f}**2*{key}" for f in fields if f != "p" for key in first if key.startswith(f + "_")
    ]
    base_terms = list(terms)
    # Finite coefficient basis comes only from method configuration, never coef_dict.
    for basis in coordinate_basis:
        terms += [f"({basis})*({term})" for term in base_terms]
    terms = canonical_terms(terms)
    symbols = [sp.Symbol(key) for key in namespace if key != "1"]
    arguments = [namespace[str(symbol)] for symbol in symbols]
    columns = []
    for term in terms:
        array = np.asarray(sp.lambdify(symbols, sp.sympify(term), modules="numpy")(*arguments))
        columns.append(np.broadcast_to(array, namespace["1"].shape).reshape(-1))
    return np.column_stack(columns), terms


def nonzero_columns(matrix):
    rms = np.sqrt(np.mean(matrix**2, axis=0))
    threshold = np.finfo(float).eps * max(1.0, float(rms.max())) * 100
    return rms > threshold, rms


def relative_residual(matrix, coefficients, train_scale):
    norm = np.linalg.norm(coefficients)
    if norm <= 1e-14 or train_scale <= 0:
        raise ValueError("Zero coefficient equation/normalization")
    coefficients = coefficients / norm
    valid = np.isfinite(matrix).all(axis=1)
    residual = matrix[valid] @ coefficients
    return {
        "equation_residual_mse": float(np.mean(residual**2)) if valid.any() else None,
        "equation_residual_nrmse": float(np.sqrt(np.mean(residual**2)) / train_scale)
        if valid.any()
        else None,
        "equation_residual_points": int(valid.sum()),
        "equation_residual_total_points": len(matrix),
        "equation_residual_finite_coverage": float(valid.mean()) if len(matrix) else 0.0,
    }


class SparseEquationModel:
    """STLS explicit RHS or sparse constrained null-space subsets."""

    def __init__(
        self,
        threshold=0.02,
        max_iter=30,
        max_terms=3,
        null_tolerance=0.01,
        coordinate_basis=(),
        spatial_order=2,
        neighbors=40,
        degree=3,
        max_samples=12000,
        seed=0,
    ):
        self.threshold, self.max_iter = threshold, max_iter
        self.max_terms, self.null_tolerance = max_terms, null_tolerance
        self.coordinate_basis = list(coordinate_basis)
        self.spatial_order, self.neighbors, self.degree = spatial_order, neighbors, degree
        self.max_samples, self.seed = max_samples, seed

    def fit(self, matrix, terms, y=None, excluded_relations=()):
        matrix = np.asarray(matrix, dtype=float)
        excluded = []
        for expression in excluded_relations:
            parsed = sp.expand(sp.sympify(expression))
            pieces = sp.Add.make_args(parsed)
            excluded.append(
                canonical_equation(
                    [float(piece.as_coeff_Mul()[0]) for piece in pieces],
                    [str(piece.as_coeff_Mul()[1]) for piece in pieces],
                )
            )
        if len(set(canonical_terms(terms))) != len(terms):
            raise ValueError("Duplicate/algebraically equal library columns must be merged")
        finite = np.isfinite(matrix).all(axis=1)
        if y is not None:
            y = np.asarray(y).reshape(-1)
            finite &= np.isfinite(y)
        ids = np.flatnonzero(finite)
        if self.max_samples and len(ids) > self.max_samples:
            ids = np.sort(
                np.random.default_rng(self.seed).choice(ids, self.max_samples, replace=False)
            )
        X = matrix[ids]
        if len(X) < max(20, len(terms) + 1):
            raise ValueError("Insufficient finite equation support")
        keep, scales = nonzero_columns(X)
        if keep.sum() < 2:
            raise ValueError("No nontrivial equation library")
        Z = X[:, keep] / scales[keep]
        singular = np.linalg.svd(Z, compute_uv=False)
        near_null = int((singular <= singular[0] * self.null_tolerance).sum())
        self.diagnostics_ = {
            "train_rows": len(ids),
            "train_finite_coverage": float(finite.mean()),
            "library_columns": len(terms),
            "active_columns": int(keep.sum()),
            "rank": int(np.linalg.matrix_rank(Z)),
            "near_null_dimension": near_null,
            "condition_number": float(singular[0] / max(singular[-1], np.finfo(float).tiny)),
            "ambiguous": near_null > (1 if y is None else 0),
            "normalization": "training_column_RMS",
        }
        coef = np.zeros(len(terms))
        if y is not None:
            response = y[ids]
            fit = np.linalg.lstsq(Z, response, rcond=1e-10)[0]
            for _ in range(self.max_iter):
                active = abs(fit) >= self.threshold * max(abs(fit).max(), 1e-15)
                if not active.any():
                    raise ValueError("Sparse discovery selected a zero RHS")
                new = np.zeros_like(fit)
                new[active] = np.linalg.lstsq(Z[:, active], response, rcond=1e-10)[0]
                if np.allclose(new, fit):
                    break
                fit = new
            coef[keep] = new / scales[keep]
        else:
            gram = Z.T @ Z / len(Z)
            active_terms = np.array(terms)[keep].tolist()
            factors = [
                {factor for factor, power in sp.sympify(term).as_powers_dict().items()
                 if not factor.is_number and power.is_positive}
                for term in active_terms
            ]
            best = None
            for size in range(2, min(self.max_terms, len(gram)) + 1):
                current = None
                for subset in itertools.combinations(range(len(gram)), size):
                    if set.intersection(*(factors[index] for index in subset)):
                        # Common multipliers permit trivial vanishing-factor solutions.
                        # Search the primitive relation directly in the same library.
                        continue
                    eigenvalues, eigenvectors = np.linalg.eigh(gram[np.ix_(subset, subset)])
                    error = float(np.sqrt(max(0.0, eigenvalues[0])))
                    vector = eigenvectors[:, 0]
                    if np.min(abs(vector)) < 1e-6:
                        continue
                    selected_terms = np.array(terms)[keep][list(subset)].tolist()
                    if not any("_" in term for term in selected_terms):
                        # An algebraic fit to nearly constant observations is not a PDE.
                        continue
                    physical = vector / scales[keep][list(subset)]
                    support_terms = np.array(terms)[keep][list(subset)].tolist()
                    candidate = canonical_equation(physical, support_terms)
                    if any(
                        candidate.keys() == known.keys()
                        and np.allclose(
                            list(candidate.values()), [known[k] for k in candidate], atol=0.01
                        )
                        for known in excluded
                    ):
                        continue
                    if current is None or error < current[0]:
                        current = (error, subset, vector)
                if current is not None and (best is None or current[0] < best[0]):
                    best = current
                if best is not None and best[0] <= self.null_tolerance:
                    break
            if best is None:
                raise ValueError("No nontrivial implicit relation")
            _, subset, vector = best
            selected = np.flatnonzero(keep)[list(subset)]
            coef[selected] = vector / scales[selected]
            coef /= np.linalg.norm(coef)
            if coef[np.flatnonzero(abs(coef) > 1e-12)[0]] < 0:
                coef *= -1
        if not np.any(abs(coef) > 1e-12):
            raise ValueError("Zero equation rejected")
        self.coefficients_, self.terms_, self.scales_ = coef, list(terms), scales
        self.expression_ = " + ".join(
            f"({value:.12g})*({term})" for term, value in zip(terms, coef) if abs(value) > 1e-12
        )
        self.diagnostics_["support"] = [
            term for term, value in zip(terms, coef) if abs(value) > 1e-12
        ]
        if y is None:
            self.train_scale_ = float(np.sqrt(np.sum((coef * scales) ** 2)))
            self.canonical_ = canonical_equation(coef, terms)
        return self
