"""Weak-form sparse identification for scalar PDEs in one space dimension.

This module is an independent Python implementation of the central WSINDy-PDE
algorithm described by Messenger and Bortz (2021): compactly supported test
functions, convolutional weak derivatives, scale-normalized sparse regression,
and MSTLS model selection. It does not differentiate the observed field.

The public adapter intentionally has a narrower input surface than the authors'
MATLAB implementation: it accepts one scalar field on a uniform ``(x, t)``
grid and a conservative library ``D_x^q(u^p)``. Within that scope, all fitted
columns and the time-derivative target use the actual weak formulation.

References
----------
* https://github.com/MathBioCU/WSINDy_PDE
* https://arxiv.org/abs/2007.02848
"""

from __future__ import annotations

from math import comb

import numpy as np


def _falling_factorial(value: int, order: int) -> int:
    result = 1
    for offset in range(order):
        result *= value - offset
    return result


def _test_function_weights(
    half_width: int,
    derivative: int,
    spacing: float,
    power: int,
) -> np.ndarray:
    """Sample a physical derivative of ``(1-s**2)**power`` on its support.

    The factored Leibniz formula is used instead of expanding the polynomial.
    This avoids cancellation for the moderately high powers selected by the
    WSINDy endpoint-decay rule.
    """
    offsets = np.arange(-half_width, half_width + 1, dtype=float)
    scaled = offsets / half_width
    values = np.zeros_like(scaled)
    for right_order in range(derivative + 1):
        left_order = derivative - right_order
        coefficient = (
            comb(derivative, right_order)
            * _falling_factorial(power, left_order)
            * _falling_factorial(power, right_order)
            * ((-1) ** right_order)
        )
        values += coefficient * (1 + scaled) ** (power - left_order) * (
            1 - scaled
        ) ** (power - right_order)
    return values / (half_width * spacing) ** derivative


def _default_test_power(half_width: int, derivative: int, tolerance: float) -> int:
    """Choose the WSINDy polynomial power from endpoint decay and smoothness."""
    penultimate_value = (2 * half_width - 1) / half_width**2
    decay_power = int(np.ceil(np.log(tolerance) / np.log(penultimate_value)))
    return max(decay_power, derivative + 1)


def _ridge_least_squares(design: np.ndarray, target: np.ndarray, ridge: float) -> np.ndarray:
    if ridge == 0:
        return np.linalg.lstsq(design, target, rcond=None)[0]
    n_features = design.shape[1]
    augmented_design = np.vstack((design, np.sqrt(ridge) * np.eye(n_features)))
    augmented_target = np.r_[target, np.zeros(n_features)]
    return np.linalg.lstsq(augmented_design, augmented_target, rcond=None)[0]


def _sequential_threshold_least_squares(
    design: np.ndarray,
    target: np.ndarray,
    threshold: float,
    ridge: float,
    max_iter: int,
) -> np.ndarray:
    """Fit one STLSQ model while preventing an empty returned support."""
    coefficients = _ridge_least_squares(design, target, ridge)
    active = np.ones(design.shape[1], dtype=bool)
    for _ in range(max_iter):
        next_active = np.abs(coefficients) >= threshold
        if not next_active.any():
            next_active[np.argmax(np.abs(coefficients))] = True
        if np.array_equal(active, next_active):
            break
        active = next_active
        coefficients = np.zeros(design.shape[1])
        coefficients[active] = _ridge_least_squares(design[:, active], target, ridge)
    return coefficients


class WSINDyPDEModel:
    """Identify ``u_t = sum c[p,q] D_x^q(u^p)`` with WSINDy-PDE.

    ``terms`` optionally supplies explicit ``(power, derivative_order)`` pairs.
    Otherwise a polynomial/derivative tensor library is generated. ``m_x`` and
    ``m_t`` are test-function half-supports in grid points; omitted values come
    from ``support_fraction``. ``thresholds`` defines the dimensionless MSTLS
    path, while ``threshold`` remains a single-value compatibility option.

    This scalar 1D adapter implements the convolutional weak system and MSTLS
    selection used by WSINDy-PDE. Multi-field, multidimensional, trigonometric,
    and mixed-derivative libraries from upstream are outside its declared scope.
    """

    def __init__(
        self,
        terms=None,
        polynomial_degree=3,
        max_derivative=3,
        m_x=None,
        m_t=None,
        support_fraction=0.15,
        test_function_power=None,
        test_function_tolerance=1e-10,
        query_stride=None,
        max_weak_samples=5000,
        thresholds=None,
        n_thresholds=50,
        threshold_min=1e-4,
        threshold_max=1.0,
        sparsity_tradeoff=0.5,
        ridge=0.0,
        max_iter=20,
        threshold=None,
    ):
        self.terms = None if terms is None else [tuple(term) for term in terms]
        self.polynomial_degree = polynomial_degree
        self.max_derivative = max_derivative
        self.m_x = m_x
        self.m_t = m_t
        self.support_fraction = support_fraction
        self.test_function_power = test_function_power
        self.test_function_tolerance = test_function_tolerance
        self.query_stride = query_stride
        self.max_weak_samples = max_weak_samples
        self.thresholds = thresholds
        self.n_thresholds = n_thresholds
        self.threshold_min = threshold_min
        self.threshold_max = threshold_max
        self.sparsity_tradeoff = sparsity_tradeoff
        self.ridge = ridge
        self.max_iter = max_iter
        self.threshold = threshold
        self._validate_configuration()

    def _validate_configuration(self):
        integer_values = {
            "polynomial_degree": self.polynomial_degree,
            "max_derivative": self.max_derivative,
            "max_weak_samples": self.max_weak_samples,
            "n_thresholds": self.n_thresholds,
            "max_iter": self.max_iter,
        }
        for name, value in integer_values.items():
            if int(value) != value or value < 1:
                raise ValueError(f"{name} must be a positive integer")
            setattr(self, name, int(value))
        for name in ("m_x", "m_t", "test_function_power"):
            value = getattr(self, name)
            if value is not None and (int(value) != value or value < 1):
                raise ValueError(f"{name} must be a positive integer or None")
            if value is not None:
                setattr(self, name, int(value))
        finite = np.asarray(
            [
                self.support_fraction,
                self.test_function_tolerance,
                self.threshold_min,
                self.threshold_max,
                self.sparsity_tradeoff,
                self.ridge,
            ],
            dtype=float,
        )
        if not np.isfinite(finite).all():
            raise ValueError("WSINDy numeric configuration must be finite")
        if not 0 < self.support_fraction < 0.5:
            raise ValueError("support_fraction must lie between zero and one half")
        if not 0 < self.test_function_tolerance < 1:
            raise ValueError("test_function_tolerance must lie between zero and one")
        if self.threshold_min < 0 or self.threshold_max < self.threshold_min:
            raise ValueError("threshold bounds must be non-negative and ordered")
        if not 0 <= self.sparsity_tradeoff <= 1:
            raise ValueError("sparsity_tradeoff must lie in [0, 1]")
        if self.ridge < 0:
            raise ValueError("ridge must be non-negative")
        if self.threshold is not None and (not np.isfinite(self.threshold) or self.threshold < 0):
            raise ValueError("threshold must be finite and non-negative")
        if self.thresholds is not None:
            path = np.asarray(self.thresholds, dtype=float).reshape(-1)
            if not len(path) or not np.isfinite(path).all() or np.any(path < 0):
                raise ValueError("thresholds must be a non-empty finite non-negative sequence")
            self.thresholds = path
        if self.query_stride is not None:
            stride = np.asarray(self.query_stride).reshape(-1)
            if len(stride) not in {1, 2} or np.any(stride < 1) or np.any(stride != stride.astype(int)):
                raise ValueError("query_stride must contain one or two positive integers")

    def _library_terms(self):
        if self.terms is None:
            terms = [(0, 0)]
            terms.extend(
                (power, derivative)
                for power in range(1, self.polynomial_degree + 1)
                for derivative in range(self.max_derivative + 1)
            )
        else:
            terms = list(self.terms)
        normalized = []
        for term in terms:
            if len(term) != 2:
                raise ValueError("each WSINDy term must be a (power, derivative) pair")
            power, derivative = term
            if int(power) != power or int(derivative) != derivative or power < 0 or derivative < 0:
                raise ValueError("term powers and derivative orders must be non-negative integers")
            power, derivative = int(power), int(derivative)
            if power == 0 and derivative != 0:
                raise ValueError("spatial derivatives of the constant library term are identically zero")
            normalized.append((power, derivative))
        if not normalized:
            raise ValueError("the WSINDy library must contain at least one term")
        if len(set(normalized)) != len(normalized):
            raise ValueError("duplicate WSINDy library terms are not identifiable")
        return normalized

    def _validate_data(self, u, x, t):
        u = np.asarray(u, dtype=float)
        x = np.asarray(x, dtype=float).reshape(-1)
        t = np.asarray(t, dtype=float).reshape(-1)
        if u.shape != (len(x), len(t)):
            raise ValueError("u must have shape (len(x), len(t))")
        if min(len(x), len(t)) < 9:
            raise ValueError("WSINDy requires at least nine points on each axis")
        if not np.isfinite(u).all() or not np.isfinite(x).all() or not np.isfinite(t).all():
            raise ValueError("u, x and t must contain only finite values")
        if not np.all(np.diff(x) > 0) or not np.all(np.diff(t) > 0):
            raise ValueError("x and t must be strictly increasing")
        if not np.allclose(np.diff(x), np.diff(x)[0], rtol=1e-6, atol=1e-12):
            raise ValueError("WSINDy convolution requires a uniform spatial grid")
        if not np.allclose(np.diff(t), np.diff(t)[0], rtol=1e-6, atol=1e-12):
            raise ValueError("WSINDy convolution requires a uniform time grid")
        return u, x, t

    def _half_support(self, size: int, explicit: int | None) -> int:
        proposed = int(round(self.support_fraction * (size - 1))) if explicit is None else explicit
        maximum = (size - 5) // 2
        if maximum < 2:
            raise ValueError("grid is too short for a compact weak-form support")
        return min(max(proposed, 2), maximum)

    def _query_slices(self, shape):
        if self.query_stride is not None:
            configured = np.asarray(self.query_stride, dtype=int).reshape(-1)
            if len(configured) == 1:
                strides = (int(configured[0]), int(configured[0]))
            else:
                strides = (int(configured[0]), int(configured[1]))
        else:
            stride = max(1, int(np.ceil(np.sqrt(np.prod(shape) / self.max_weak_samples))))
            strides = (stride, stride)
        return (slice(None, None, strides[0]), slice(None, None, strides[1])), strides

    @staticmethod
    def _weak_convolution(values, x_weights, t_weights, dx, dt):
        try:
            from scipy.signal import fftconvolve
        except ImportError as exc:  # pragma: no cover - SciPy is a core project dependency
            raise ImportError("WSINDyPDEModel requires scipy.signal.fftconvolve") from exc
        kernel = np.outer(x_weights, t_weights)
        return dx * dt * fftconvolve(values, kernel[::-1, ::-1], mode="valid")

    def weak_system(self, u, x, t):
        """Return the WSINDy weak library and weak time-derivative target."""
        u, x, t = self._validate_data(u, x, t)
        terms = self._library_terms()
        max_spatial_derivative = max(derivative for _, derivative in terms)
        mx = self._half_support(len(x), self.m_x)
        mt = self._half_support(len(t), self.m_t)
        px = (
            _default_test_power(mx, max_spatial_derivative, self.test_function_tolerance)
            if self.test_function_power is None
            else max(self.test_function_power, max_spatial_derivative + 1)
        )
        pt = (
            _default_test_power(mt, 1, self.test_function_tolerance)
            if self.test_function_power is None
            else max(self.test_function_power, 2)
        )
        dx, dt = float(x[1] - x[0]), float(t[1] - t[0])
        wx = {
            order: _test_function_weights(mx, order, dx, px)
            for order in range(max_spatial_derivative + 1)
        }
        wt = {
            order: _test_function_weights(mt, order, dt, pt)
            for order in range(2)
        }
        target_grid = -self._weak_convolution(u, wx[0], wt[1], dx, dt)
        slices, strides = self._query_slices(target_grid.shape)
        target = target_grid[slices].reshape(-1)
        columns = []
        for power, derivative in terms:
            weak_column = ((-1) ** derivative) * self._weak_convolution(
                u**power, wx[derivative], wt[0], dx, dt
            )
            columns.append(weak_column[slices].reshape(-1))
        theta = np.column_stack(columns)
        if theta.shape[0] < 2:
            raise ValueError("test-function support and query stride leave too few weak samples")
        self.terms_ = terms
        self.support_ = {"m_x": mx, "m_t": mt}
        self.test_function_powers_ = {"p_x": px, "p_t": pt}
        self.query_stride_ = {"x": strides[0], "t": strides[1]}
        self.n_weak_samples_ = int(theta.shape[0])
        return theta, target

    def _threshold_path(self):
        if self.threshold is not None:
            return np.asarray([self.threshold], dtype=float)
        if self.thresholds is not None:
            return np.unique(np.asarray(self.thresholds, dtype=float))
        if self.threshold_min == self.threshold_max:
            return np.asarray([self.threshold_min], dtype=float)
        if self.threshold_min == 0:
            positive = np.geomspace(
                max(np.finfo(float).eps, self.threshold_max * 1e-6),
                self.threshold_max,
                self.n_thresholds - 1,
            )
            return np.r_[0.0, positive]
        return np.geomspace(self.threshold_min, self.threshold_max, self.n_thresholds)

    def fit(self, u, x, t):
        theta, target = self.weak_system(u, x, t)
        column_scales = np.linalg.norm(theta, axis=0)
        usable = column_scales > np.finfo(float).eps
        if not usable.any():
            raise ValueError("all WSINDy library columns vanish on this dataset")
        safe_scales = column_scales.copy()
        safe_scales[~usable] = 1.0
        target_scale = np.linalg.norm(target)
        if target_scale <= np.finfo(float).eps:
            raise ValueError("the weak time-derivative target is numerically zero")
        design = theta / safe_scales
        normalized_target = target / target_scale
        least_squares = _ridge_least_squares(design, normalized_target, self.ridge)
        least_squares[~usable] = 0.0
        projection_denominator = max(
            np.linalg.norm(design @ least_squares), np.finfo(float).eps
        )

        path = self._threshold_path()
        candidates = []
        losses = []
        projection_costs = []
        complexity_costs = []
        for value in path:
            candidate = _sequential_threshold_least_squares(
                design, normalized_target, float(value), self.ridge, self.max_iter
            )
            candidate[~usable] = 0.0
            projection = np.linalg.norm(design @ (candidate - least_squares)) / projection_denominator
            complexity = np.count_nonzero(candidate) / len(candidate)
            loss = (
                2 * self.sparsity_tradeoff * projection
                + 2 * (1 - self.sparsity_tradeoff) * complexity
            )
            candidates.append(candidate)
            projection_costs.append(float(projection))
            complexity_costs.append(float(complexity))
            losses.append(float(loss))
        selected = int(np.argmin(losses))
        normalized_coefficients = candidates[selected]
        coefficients = normalized_coefficients * target_scale / safe_scales
        coefficients[~usable] = 0.0

        self.coefficients_ = coefficients
        self.selected_lambda_ = float(path[selected])
        self.lambda_path_ = path
        self.mstls_loss_ = float(losses[selected])
        self.loss_path_ = np.asarray(losses)
        self.projection_cost_path_ = np.asarray(projection_costs)
        self.complexity_cost_path_ = np.asarray(complexity_costs)
        self.column_scales_ = column_scales
        self.target_scale_ = float(target_scale)
        self.feature_names_ = [self._feature_name(term) for term in self.terms_]
        self.weak_expression_ = " + ".join(
            f"({coefficient:.12g})*{name}"
            for coefficient, name in zip(self.coefficients_, self.feature_names_)
            if coefficient != 0
        ) or "0"
        self.best_expression_ = self._expanded_expression()
        residual = theta @ self.coefficients_ - target
        self.weak_residual_mse_ = float(np.mean(residual**2))
        self.weak_relative_residual_ = float(np.linalg.norm(residual) / target_scale)
        self.method_provenance_ = {
            "method": "WSINDy-PDE",
            "implementation": "independent scalar-1D Python adaptation",
            "weak_discretization": "compact polynomial test functions and valid FFT convolution",
            "sparse_regression": "MSTLS threshold path",
            "upstream": "https://github.com/MathBioCU/WSINDy_PDE",
            "reference_commit": "d9296be4c17c5e0b4df14472f4cd8276a8ae4eed",
            "paper": "https://arxiv.org/abs/2007.02848",
        }
        return self

    @staticmethod
    def _feature_name(term):
        power, derivative = term
        if power == 0:
            base = "1"
        elif power == 1:
            base = "u"
        else:
            base = f"u^{power}"
        return base if derivative == 0 else f"d_x^{derivative}({base})"

    def _expanded_expression(self):
        import sympy as sp

        coordinate = sp.Symbol("x")
        field = sp.Function("u")(coordinate)
        replacements = {field: sp.Symbol("u")}
        replacements.update(
            {
                sp.diff(field, coordinate, order): sp.Symbol("u_" + "x" * order)
                for order in range(1, max(derivative for _, derivative in self.terms_) + 1)
            }
        )
        expression = 0
        for coefficient, (power, derivative) in zip(self.coefficients_, self.terms_):
            if coefficient == 0:
                continue
            base = sp.Integer(1) if power == 0 else field**power
            expression += float(coefficient) * sp.diff(base, coordinate, derivative)
        return str(sp.expand(expression.doit().xreplace(replacements)))


# Compatibility for code written against the temporary pre-WSINDy adapter.
IntegralWeakPDEModel = WSINDyPDEModel


__all__ = ["WSINDyPDEModel", "IntegralWeakPDEModel"]
