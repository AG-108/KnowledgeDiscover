"""Numerical core and bounded protocol for Xu et al.'s PDE PIC.

This module deliberately does not implement a neural-network architecture.  A caller
prepares one common ANN reference and supplies a trainer which restores that exact
state for every candidate and refits the PDE coefficients during PINN training.
"""

from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Tuple

import numpy as np


PIC_IMPLEMENTATION_VERSION = "pic-xu-2022-torch-v3"


class PICError(ValueError):
    """Invalid numerical input or an unusable coefficient fit."""


@dataclass(frozen=True)
class CoefficientStability:
    coefficients: np.ndarray
    coefficient_cv: np.ndarray
    r_loss: float
    windows: Tuple[Tuple[float, float], ...]


@dataclass(frozen=True)
class PreparedPICReference:
    """Candidate-independent ANN output and derivative data.

    ``cache_key`` must identify data split, configuration, seed, ANN state and
    implementation version.  ``observed_range`` is fitted on training observations,
    never on candidate outputs or held-out targets.
    """

    time: np.ndarray
    lhs: np.ndarray
    ann_output: np.ndarray
    observed_range: Tuple[float, float]
    cache_key: str
    seed: int
    version: str = PIC_IMPLEMENTATION_VERSION


@dataclass(frozen=True)
class PINNTrainingRequest:
    reference: PreparedPICReference
    rhs_terms: np.ndarray
    initial_coefficients: np.ndarray
    candidate_id: str
    seed: int
    budget: Mapping[str, int]
    require_epoch_refit: bool = True


@dataclass(frozen=True)
class PINNTrainingResult:
    output: np.ndarray
    refitted_coefficients: np.ndarray
    converged: bool
    coefficient_refit_count: int
    cost: Mapping[str, float]
    message: str = ""


@dataclass(frozen=True)
class PICResult:
    status: str
    pic: Optional[float]
    r_loss: Optional[float]
    p_loss: Optional[float]
    window_coefficients: Optional[np.ndarray]
    original_coefficients: Optional[np.ndarray]
    reference_fit_coefficients: Optional[np.ndarray]
    refitted_coefficients: Optional[np.ndarray]
    cost: Mapping[str, float]
    cache_key: str
    message: str = ""
    version: str = PIC_IMPLEMENTATION_VERSION


def fit_tls_coefficients(lhs, rhs_terms, *, denominator_tol: float = 1e-12) -> np.ndarray:
    """Fit ``lhs = rhs_terms @ coefficients`` by homogeneous total least squares.

    This matches the author's SVD of ``[lhs, library]``.  The minus sign is required
    when converting the null relation ``v0*lhs + v_rhs*rhs = 0`` to RHS coefficients.
    """

    y = np.asarray(lhs, dtype=float).reshape(-1, 1)
    x = np.asarray(rhs_terms, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.ndim != 2 or y.shape[0] != x.shape[0] or y.shape[0] < 2 or x.shape[1] < 1:
        raise PICError("TLS needs aligned 2D RHS terms, one LHS column, and at least two rows")
    augmented = np.hstack((y, x))
    if augmented.shape[0] < augmented.shape[1]:
        raise PICError("TLS is underdetermined: rows must be at least LHS plus RHS columns")
    if np.linalg.matrix_rank(x) < x.shape[1]:
        raise PICError("TLS RHS library is rank deficient and coefficients are not identifiable")
    if not np.all(np.isfinite(augmented)):
        raise PICError("TLS input contains non-finite values")
    _, _, vh = np.linalg.svd(augmented, full_matrices=False)
    null_vector = vh[-1]
    scale = float(null_vector[0])
    if abs(scale) <= denominator_tol * max(1.0, float(np.linalg.norm(null_vector))):
        raise PICError("TLS solve is singular: LHS component of the null vector is zero")
    coefficients = -null_vector[1:] / scale
    if not np.all(np.isfinite(coefficients)):
        raise PICError("TLS produced non-finite coefficients")
    return coefficients


def coefficient_stability(
    time,
    lhs,
    rhs_terms,
    *,
    n_windows: int = 10,
    window_fraction: float = 0.5,
    mean_tol: float = 1e-12,
) -> CoefficientStability:
    """Compute paper eqs. 12--13 over overlapping temporal windows."""

    t = np.asarray(time, dtype=float).reshape(-1)
    y = np.asarray(lhs, dtype=float).reshape(-1)
    x = np.asarray(rhs_terms, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if t.size != y.size or x.ndim != 2 or x.shape[0] != t.size:
        raise PICError("time, LHS, and RHS terms must have aligned rows")
    if n_windows < 2 or not (0.0 < window_fraction <= 1.0):
        raise PICError("n_windows must be >=2 and window_fraction must be in (0, 1]")
    if not np.all(np.isfinite(t)) or np.ptp(t) <= 0:
        raise PICError("time must be finite and span a non-zero interval")
    time_range = float(np.ptp(t))
    width = time_range * window_fraction
    # Xu et al.'s code uses dk=(t_up-t_low)/20 for ten windows, rather than
    # spreading the starts over every possible position. Generalize that exact
    # half-window overlap convention as range/(2*n_windows).
    shift = time_range / (2.0 * n_windows)
    starts = float(np.min(t)) + np.arange(n_windows) * shift
    if starts[-1] + width > float(np.max(t)) + 1e-12 * max(1.0, time_range):
        raise PICError("window configuration extends beyond the available time domain")
    coefficients, windows = [], []
    for i, start in enumerate(starts):
        end = start + width
        mask = (t >= start) & ((t <= end) if i == n_windows - 1 else (t < end))
        if np.count_nonzero(mask) < x.shape[1] + 1:
            raise PICError(f"window {i} has too few rows for TLS")
        coefficients.append(fit_tls_coefficients(y[mask], x[mask]))
        windows.append((float(start), float(end)))
    coef = np.asarray(coefficients)
    means = np.mean(coef, axis=0)
    scale = np.maximum(1.0, np.max(np.abs(coef), axis=0))
    if np.any(np.abs(means) <= mean_tol * scale):
        raise PICError("coefficient CV is undefined because a windowed coefficient mean is zero")
    cv = np.abs(np.std(coef, axis=0, ddof=0) / means)
    return CoefficientStability(coef, cv, float(np.mean(cv)), tuple(windows))


def normalized_rmse(ann_output, pinn_output, observed_range: Tuple[float, float]) -> float:
    """Paper eqs. 15--16 using one observation-derived min/max for both outputs."""

    ann = np.asarray(ann_output, dtype=float)
    pinn = np.asarray(pinn_output, dtype=float)
    if ann.shape != pinn.shape or ann.size == 0:
        raise PICError("ANN and PINN outputs must be non-empty and have identical shapes")
    if not np.all(np.isfinite(ann)) or not np.all(np.isfinite(pinn)):
        raise PICError("ANN or PINN output contains non-finite values")
    lower, upper = map(float, observed_range)
    if not np.isfinite(lower + upper) or upper <= lower:
        raise PICError("observation min/max must be finite with max > min")
    return float(np.sqrt(np.mean(((pinn - ann) / (upper - lower)) ** 2)))


class PhysicsInformedInformationCriterion:
    """Opt-in evaluator for the complete PIC composition with external PINN training."""

    def __init__(self, trainer: Callable[[PINNTrainingRequest], PINNTrainingResult]):
        if not callable(trainer):
            raise TypeError("trainer must be callable")
        self.trainer = trainer

    def evaluate(self, reference, rhs_terms, *, candidate_id, budget, n_windows=10,
                 window_fraction=0.5, original_coefficients=None):
        stability = None
        original = None
        reference_fit = None
        refitted = None
        cost = {}
        try:
            if reference.version != PIC_IMPLEMENTATION_VERSION or not reference.cache_key:
                raise PICError("reference has an incompatible version or empty cache key")
            x = np.asarray(rhs_terms, dtype=float)
            stability = coefficient_stability(
                reference.time, reference.lhs, x,
                n_windows=n_windows, window_fraction=window_fraction,
            )
            reference_fit = fit_tls_coefficients(reference.lhs, x)
            original = (None if original_coefficients is None else
                        np.asarray(original_coefficients, dtype=float).reshape(-1))
            if original is not None and (original.size != reference_fit.size or
                                         not np.all(np.isfinite(original))):
                raise PICError("submitted original coefficients have the wrong size or are non-finite")
            try:
                trained = self.trainer(PINNTrainingRequest(
                    reference=reference, rhs_terms=x, initial_coefficients=reference_fit,
                    candidate_id=str(candidate_id), seed=reference.seed, budget=dict(budget),
                ))
            except Exception as exc:  # a failed expensive backend is an evaluation status
                return self._failure(reference, "training_failed", str(exc), {},
                                     stability, original, reference_fit, np.empty(0))
            if not isinstance(trained, PINNTrainingResult):
                raise PICError("trainer returned an unsupported result type")
            cost = dict(trained.cost)
            refitted = np.asarray(trained.refitted_coefficients, dtype=float).reshape(-1)
            if not trained.converged:
                return self._failure(reference, "pinn_not_converged", trained.message, cost,
                                     stability, original, reference_fit, refitted)
            required_refits = int(dict(budget).get("epochs", 0))
            if required_refits < 1 or trained.coefficient_refit_count != required_refits:
                return self._failure(reference, "protocol_violation",
                                     "trainer must refit coefficients exactly once per epoch", cost,
                                     stability, original, reference_fit,
                                     refitted)
            if refitted.size != reference_fit.size or not np.all(np.isfinite(refitted)):
                return self._failure(reference, "invalid_training_output",
                                     "refitted coefficients are non-finite or have the wrong size",
                                     cost, stability, original, reference_fit, refitted)
            p_loss = normalized_rmse(reference.ann_output, trained.output, reference.observed_range)
            value = stability.r_loss * p_loss
            if not np.isfinite(value):
                raise PICError("PIC is non-finite")
            return PICResult("ok", float(value), stability.r_loss, p_loss,
                             stability.coefficients, original, reference_fit,
                             refitted, cost,
                             reference.cache_key)
        except (PICError, np.linalg.LinAlgError, TypeError, ValueError) as exc:
            return PICResult(
                "invalid_input", None,
                None if stability is None else stability.r_loss,
                None,
                None if stability is None else stability.coefficients,
                original,
                reference_fit,
                refitted,
                cost,
                getattr(reference, "cache_key", ""),
                str(exc),
            )

    @staticmethod
    def _failure(reference, status, message, cost, stability, original, reference_fit, refitted):
        return PICResult(status, None, stability.r_loss, None, stability.coefficients,
                         original, reference_fit, np.asarray(refitted, dtype=float), dict(cost),
                         reference.cache_key, message)
