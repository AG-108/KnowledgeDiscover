from typing import Any, Dict, Iterable, Optional, Tuple, Union
from abc import ABC, abstractmethod

import numpy as np


class MetaMetricsError(Exception):
    pass

class MetaMetricsConfig:
    """
    通用指标配置。
    - greater_is_better: 指标值越大是否代表越好。
    - name: 指标名称（可选）。
    - dtype: 内部计算的数据类型（可选，如 'float64'）。
    """
    def __init__(self, greater_is_better: bool = True, name: str = None, dtype: str = None):
        self.greater_is_better = greater_is_better
        self.name = name
        self.dtype = dtype

Number = Union[int, float]
ArrayLike = Any  # 实际可为 numpy.ndarray / torch.Tensor / list 等

class MetaMetrics(ABC):
    """
    预测性能评估抽象基类。

    典型用法：
    - 批量：metrics(y_true, y_pred, sample_weight=...)
    - 流式：metrics.update(y_true_batch, y_pred_batch); ... ; value = metrics.compute()

    约定：
    - y_true, y_pred 形状兼容（由子类在 _check_and_normalize_inputs 中具体校验）
    - 可选的 sample_weight 与 y_true 同长度（或可广播）
    """

    def __init__(self, config: Optional[MetaMetricsConfig] = None):
        self.config = config or MetaMetricsConfig()
        self._is_fitted = False
        self.reset()

    def update(self, y_true: ArrayLike, y_pred: ArrayLike, sample_weight: Optional[ArrayLike] = None, **kwargs) -> None:
        """
        累积一批新数据。可多次调用，最终通过 compute() 得到指标。
        """
        y_true_n, y_pred_n, w_n = self._check_and_normalize_inputs(y_true, y_pred, sample_weight)
        self._update_impl(y_true_n, y_pred_n, w_n, **kwargs)

    def compute(self) -> Union[Number, Dict[str, Number]]:
        """
        计算并返回指标值（或多指标字典）。要求幂等，不应改变内部状态。
        """
        return self._compute_impl()

    def reset(self) -> None:
        """
        清空内部累计状态。
        """
        self._reset_impl()

    def __call__(self, y_true: ArrayLike, y_pred: ArrayLike, sample_weight: Optional[ArrayLike] = None, **kwargs) -> \
    Union[Number, Dict[str, Number]]:
        """
        便捷一次性计算：reset -> update -> compute。
        """
        self.reset()
        self.update(y_true, y_pred, sample_weight, **kwargs)
        return self.compute()

    # ============ 需子类实现的核心抽象 ============
    @abstractmethod
    def _update_impl(self, y_true: ArrayLike, y_pred: ArrayLike, sample_weight: Optional[ArrayLike], **kwargs) -> None:
        pass

    @abstractmethod
    def _compute_impl(self) -> Union[Number, Dict[str, Number]]:
        pass

    @abstractmethod
    def _reset_impl(self) -> None:
        pass

    # ============ 输入校验与规范化 ============
    def _check_and_normalize_inputs(
        self,
        y_true: ArrayLike,
        y_pred: ArrayLike,
        sample_weight: Optional[ArrayLike] = None
    ) -> Tuple[ArrayLike, ArrayLike, Optional[ArrayLike]]:
        """
        统一入口的基础校验与轻量规范化。
        子类可 override 以实现更严格的任务特定检查（例如分类标签空间、概率范围等）。
        """
        if y_true is None or y_pred is None:
            raise MetaMetricsError("y_true 和 y_pred 不能为空。")

        # 允许 numpy / torch / list，尽量只做轻量检查。
        # 注意：y_true/y_pred 的长度一致性校验故意不在这里做——例如回归中
        # y_true.shape=[N] 而 y_pred.shape=[N, 1] 这类可广播的情况因任务而异，
        # 交给子类的 _update_impl 自行处理更合适。
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
        return len(x)  # numpy/torch/list 均可
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
        super().__init__(config or MetaMetricsConfig(greater_is_better=False, name="MSE", dtype="float64"))

    def _reset_impl(self) -> None:
        self._sum_squared_error = 0.0
        self._count = 0.0

    def _compute_impl(self) -> float:
        if self._count == 0:
            raise MetaMetricsError("没有数据可计算指标，请先调用 update()。")
        return self._sum_squared_error / self._count

    def _update_impl(self, y_true: ArrayLike, y_pred: ArrayLike, sample_weight: Optional[ArrayLike], **kwargs) -> None:
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

    def _update_impl(self, y_true: ArrayLike, y_pred: ArrayLike, sample_weight: Optional[ArrayLike], **kwargs) -> None:
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
        super().__init__(config or MetaMetricsConfig(greater_is_better=False, name="AIC", dtype="float64"))
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
        super().__init__(config or MetaMetricsConfig(greater_is_better=False, name="BIC", dtype="float64"))
        self.num_params = num_params

    def _compute_impl(self) -> float:
        mse, n = self._mse_and_n()
        return float(n * np.log(mse) + self.num_params * np.log(n))


class ParsimonyInformationCriterion(_InformationCriterionBase):
    """Parsimony Information Criterion (PIC).

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
        super().__init__(config or MetaMetricsConfig(greater_is_better=False, name="PIC", dtype="float64"))
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

