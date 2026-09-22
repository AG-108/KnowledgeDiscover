from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Optional, Tuple, Union
import warnings

import numpy as np


class MetaMetricsError(Exception):
    pass


class MetaMetricsConfig:
    """Configure a metric name, numeric dtype, and optimization direction."""

    def __init__(self, greater_is_better: bool = True, name: str = None, dtype: str = None):
        self.greater_is_better = greater_is_better
        self.name = name
        self.dtype = dtype


Number = Union[int, float]
ArrayLike = Any  # Accept NumPy arrays, torch tensors, and ordinary sequences.


class MetaMetrics(ABC):
    """Define the shared batch and streaming interface for prediction metrics."""

    def __init__(self, config: Optional[MetaMetricsConfig] = None):
        self.config = config or MetaMetricsConfig()
        self._is_fitted = False
        self.reset()

    def update(
        self,
        y_true: ArrayLike,
        y_pred: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
        **kwargs,
    ) -> None:
        """Accumulate one batch of observations."""
        y_true_n, y_pred_n, w_n = self._check_and_normalize_inputs(y_true, y_pred, sample_weight)
        self._update_impl(y_true_n, y_pred_n, w_n, **kwargs)

    def compute(self) -> Union[Number, Dict[str, Number]]:
        """Return the metric value without mutating accumulated state."""
        return self._compute_impl()

    def reset(self) -> None:
        """Clear all accumulated metric state."""
        self._reset_impl()

    def __call__(
        self,
        y_true: ArrayLike,
        y_pred: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
        **kwargs,
    ) -> Union[Number, Dict[str, Number]]:
        """Compute the metric for one complete input pair."""
        self.reset()
        self.update(y_true, y_pred, sample_weight, **kwargs)
        return self.compute()

    # Subclasses implement the accumulation contract below.
    @abstractmethod
    def _update_impl(
        self, y_true: ArrayLike, y_pred: ArrayLike, sample_weight: Optional[ArrayLike], **kwargs
    ) -> None:
        pass

    @abstractmethod
    def _compute_impl(self) -> Union[Number, Dict[str, Number]]:
        pass

    @abstractmethod
    def _reset_impl(self) -> None:
        pass

    # Shared lightweight input validation.
    def _check_and_normalize_inputs(
        self, y_true: ArrayLike, y_pred: ArrayLike, sample_weight: Optional[ArrayLike] = None
    ) -> Tuple[ArrayLike, ArrayLike, Optional[ArrayLike]]:
        """Perform lightweight validation shared by all metric implementations."""
        if y_true is None or y_pred is None:
            raise MetaMetricsError("y_true 和 y_pred 不能为空。")

        # Accept NumPy, torch, and sequence inputs without forcing a conversion.
        # Shape compatibility is task-specific; regression may compare (N,) with (N, 1).
        # Subclasses therefore decide which broadcasting rules are valid.
        # _update_impl performs the final task-specific validation.
        n_true = _safe_len(y_true)

        if sample_weight is not None:
            n_w = _safe_len(sample_weight)
            if n_true is not None and n_w is not None and n_true != n_w:
                raise MetaMetricsError("sample_weight 的长度需与 y_true 对齐。")

        # convert y_true and y_pred to self.config.dtype
        if self.config.dtype is not None:
            y_true = np.asarray(y_true, dtype=self.config.dtype)
            y_pred = np.asarray(y_pred, dtype=self.config.dtype)
            if sample_weight is not None:
                sample_weight = np.asarray(sample_weight, dtype=self.config.dtype)
        return y_true, y_pred, sample_weight


def _safe_len(x: Any) -> Optional[int]:
    try:
        return len(x)  # All supported containers implement len().
    except Exception:
        return None


def _accumulate_squared_error(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sample_weight: Optional[ArrayLike],
) -> Tuple[float, float]:
    """Shared building block for MSE and the information-criterion metrics
    below: reduce one batch to (sum of weighted squared errors, sum of
    weights), so callers only need to accumulate two running scalars.
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight).flatten()
        if len(sample_weight) != len(y_true):
            raise MetaMetricsError("sample_weight 长度需与 y_true 对齐。")
    else:
        sample_weight = np.ones_like(y_true)

    squared_errors = (y_true - y_pred) ** 2
    weighted_errors = squared_errors * sample_weight

    return np.sum(weighted_errors), np.sum(sample_weight)


class MSE(MetaMetrics):
    """Mean squared error. Reference implementation for how to add a new metric:
    subclass MetaMetrics and fill in _reset_impl / _update_impl / _compute_impl.
    """

    def __init__(self, config: Optional[MetaMetricsConfig] = None):
        super().__init__(
            config or MetaMetricsConfig(greater_is_better=False, name="MSE", dtype="float64")
        )

    def _reset_impl(self) -> None:
        self._sum_squared_error = 0.0
        self._count = 0.0

    def _compute_impl(self) -> float:
        if self._count == 0:
            raise MetaMetricsError("没有数据可计算指标，请先调用 update()。")
        return self._sum_squared_error / self._count

    def _update_impl(
        self, y_true: ArrayLike, y_pred: ArrayLike, sample_weight: Optional[ArrayLike], **kwargs
    ) -> None:
        sse, w = _accumulate_squared_error(y_true, y_pred, sample_weight)
        self._sum_squared_error += sse
        self._count += w


class _InformationCriterionBase(MetaMetrics):
    """Shared base for AIC / BIC / ParsimonyInformationCriterion.

    All three score a *fitted model* (not just raw predictions): they need
    the model's residual fit (mean squared error over the data) plus one or
    more terms that describe the model's own complexity. The fit part is
    identical across all three and accumulates the same way MSE does, so it
    lives here; each subclass only implements the complexity-penalty side of
    its formula in `_compute_impl`.

    Complexity-related inputs (e.g. `num_params`, `complexity`,
    `physics_penalty`) describe the candidate model being scored, not a
    particular data batch, so they are fixed at construction time rather
    than passed to `update()`.

    This class deliberately still leaves `_compute_impl` abstract, so it
    cannot be instantiated directly -- only its subclasses can.
    """

    def _reset_impl(self) -> None:
        self._sum_squared_error = 0.0
        self._count = 0.0

    def _update_impl(
        self, y_true: ArrayLike, y_pred: ArrayLike, sample_weight: Optional[ArrayLike], **kwargs
    ) -> None:
        sse, w = _accumulate_squared_error(y_true, y_pred, sample_weight)
        self._sum_squared_error += sse
        self._count += w

    def _mse_and_n(self) -> Tuple[float, float]:
        if self._count == 0:
            raise MetaMetricsError("没有数据可计算指标，请先调用 update()。")
        return self._sum_squared_error / self._count, self._count


class AIC(_InformationCriterionBase):
    """Akaike Information Criterion.

    AIC = n * ln(MSE) + 2k

    This is the usual Gaussian-likelihood AIC = n*ln(RSS/n) + 2k rewritten
    with MSE = RSS/n. `num_params` (k) is the number of free parameters of
    the model being scored (e.g. the number of nonzero coefficients in a
    discovered equation) and is fixed for the lifetime of the metric
    instance -- pass a new instance to score a different candidate model.
    Lower is better.
    """

    def __init__(self, num_params: int, config: Optional[MetaMetricsConfig] = None):
        super().__init__(
            config or MetaMetricsConfig(greater_is_better=False, name="AIC", dtype="float64")
        )
        self.num_params = num_params

    def _compute_impl(self) -> float:
        mse, n = self._mse_and_n()
        return float(n * np.log(mse) + 2 * self.num_params)


class BIC(_InformationCriterionBase):
    """Bayesian Information Criterion.

    BIC = n * ln(MSE) + k * ln(n)

    Same fit term as AIC, but penalizes the parameter count k more heavily
    for larger sample sizes n. Lower is better.
    """

    def __init__(self, num_params: int, config: Optional[MetaMetricsConfig] = None):
        super().__init__(
            config or MetaMetricsConfig(greater_is_better=False, name="BIC", dtype="float64")
        )
        self.num_params = num_params

    def _compute_impl(self) -> float:
        mse, n = self._mse_and_n()
        return float(n * np.log(mse) + self.num_params * np.log(n))


class AdditiveParsimonyScore(_InformationCriterionBase):
    """Legacy custom additive fit/complexity/physics score (not Xu et al. PIC).

    PIC = n * ln(MSE) + lambda_complexity * complexity + lambda_physics * physics_penalty

    Same AIC/BIC-style fit term, but generalizes the penalty side beyond a
    plain parameter count: `complexity` is a structural-complexity score for
    the candidate model (e.g. expression-tree node count) and
    `physics_penalty` is a separate, independently-weighted penalty for
    violating physical constraints (e.g. dimensional consistency, known
    conservation laws). Both are computed by the caller for the model being
    scored and passed in fixed at construction time, same as `num_params`
    for AIC/BIC. Lower is better.
    """

    def __init__(
        self,
        complexity: float,
        physics_penalty: float = 0.0,
        lambda_complexity: float = 1.0,
        lambda_physics: float = 1.0,
        config: Optional[MetaMetricsConfig] = None,
    ):
        super().__init__(
            config or MetaMetricsConfig(greater_is_better=False, name="additive_parsimony_v1", dtype="float64")
        )
        self.complexity = complexity
        self.physics_penalty = physics_penalty
        self.lambda_complexity = lambda_complexity
        self.lambda_physics = lambda_physics

    def _compute_impl(self) -> float:
        mse, n = self._mse_and_n()
        return float(
            n * np.log(mse)
            + self.lambda_complexity * self.complexity
            + self.lambda_physics * self.physics_penalty
        )


class ParsimonyInformationCriterion(AdditiveParsimonyScore):
    """Deprecated compatibility alias for :class:`AdditiveParsimonyScore`.

    This historical class is not the physics-informed information criterion
    of Xu et al. (2022). Use ``PhysicsInformedInformationCriterion`` for that.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "ParsimonyInformationCriterion is the legacy additive_parsimony_v1 score, "
            "not Xu et al.'s PIC; use AdditiveParsimonyScore or "
            "PhysicsInformedInformationCriterion explicitly.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


from .physics_informed import (  # noqa: E402
    PIC_IMPLEMENTATION_VERSION,
    CoefficientStability,
    PICError,
    PICResult,
    PINNTrainingRequest,
    PINNTrainingResult,
    PhysicsInformedInformationCriterion,
    PreparedPICReference,
    coefficient_stability,
    fit_tls_coefficients,
    normalized_rmse,
)

from .pic_torch import (  # noqa: E402
    SUPPORTED_TERMS,
    TorchPICConfig,
    TorchPICPrepared,
    evaluate_torch_pic,
    prepare_torch_pic_reference,
)
