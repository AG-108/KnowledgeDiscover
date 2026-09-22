"""
Extraction of the "sinus" (pretraining) random-PDE generation method from
PDEformer-1D, adapted into a self-contained benchmark generator for this
project.

Original source (Apache License 2.0):
    kd/dataset/mindscience-legacy-master/MindFlow/applications/pdeformer1d/
        data_generation/common.py
        data_generation/custom_sinus.py
    Copyright 2023 Huawei Technologies Co., Ltd

The original code generates data for the equation

    u_t + f0(u) + s(x) + d/dx(f1(u) - kappa(x) u_x) = 0,  (t,x) in [0,1]x[-1,1]
    u(0,x) = g(x)

where f_i(u) = c_i1*u + c_i2*u^2 + c_i3*u^3 (optionally plus sinusoidal
terms), s(x)/kappa(x) are independently zero, a scalar, or a spatially
varying random field, and solves it with the Dedalus-v3 spectral PDE solver.

`dedalus` is not installable in this (Windows) environment, so the random
coefficient/field sampling here is re-implemented in plain numpy (faithfully
reproducing the original sampling distributions), while the PDE solve itself
goes through a pluggable `PDESolverBackend`:
  - `NumpySpectralBackend` (default): FFT-based spatial derivatives + an
    implicit (`scipy.integrate.solve_ivp`, method="BDF") time integrator.
    Handles constant AND spatially-varying coefficients uniformly (at the
    cost of being slower than a dedicated constant-coefficient spectral
    method), entirely in numpy/scipy.
  - `DedalusBackend`: reserved interface for a faithful Dedalus-v3 solve.
    `solve()` raises `NotImplementedError` here; intended to be implemented
    on a Linux environment with `dedalus` installed (see class docstring).

Because we generate the data ourselves, the ground-truth coefficients (and a
human-readable `sym_true` string, when they are scalar) are known exactly --
unlike most other datasets in this package.

Not covered (future work): non-periodic boundary conditions (Dirichlet/
Neumann/Robin, which need a Chebyshev basis + tau method), the wave-equation
and multi-component-equation families, and the inverse-problem datasets.
"""

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from ._base import GridPDEDataset
from ._info import DatasetInfo

# Domain, matching the original custom_sinus.py.
X_L = -1.0
X_R = 1.0
STOP_SIM_TIME = 1.0


# ---------------------------------------------------------------------------
# Coefficient / field sampling (adapted from common.py; kept as plain
# functions rather than the original's PDETermBase/... class hierarchy,
# since that hierarchy exists to support many PDE families and would be
# over-engineering for extracting just this one).
# ---------------------------------------------------------------------------


def _random_value(
    rng: np.random.Generator, magnitude: float = 1.0, size: Optional[Tuple[int, ...]] = None
) -> NDArray:
    """Uniform U([-magnitude, magnitude]). Matches PDERandomCoefBase._random_value (distribution='U')."""
    return magnitude * rng.uniform(-1, 1, size=size)


def _random_field(
    rng: np.random.Generator,
    x_coord: NDArray,
    x_len: float,
    k_tot: int = 4,
    num_choice_k: int = 2,
    normalized: bool = False,
    smooth: bool = False,
) -> NDArray:
    """
    Sum-of-sines random field on a periodic domain, optionally passed through
    an abs()/sign-flip and a smooth window restriction (10% chance each),
    or normalized to [0, 1]. Adapted from RandomFieldCoef._random_field
    (periodic=True branch only) and its helpers _apply_abs_fn /
    _apply_window_restriction.
    """
    selected = rng.integers(low=0, high=k_tot, size=num_choice_k)
    one_hot_sum = np.eye(k_tot)[selected].sum(axis=0)  # (k_tot,)
    angular_freq = 2.0 * np.pi * np.arange(1, k_tot + 1) * one_hot_sum / x_len

    amp = rng.uniform(size=(k_tot, 1))
    phase = 2.0 * np.pi * rng.uniform(size=(k_tot, 1))
    field_ = amp * np.sin(angular_freq[:, None] * x_coord[None, :] + phase)
    field_ = field_.sum(axis=0)  # (n_x,)

    if not smooth:
        if rng.random() < 0.1:
            field_ = np.abs(field_)
        sgn = rng.choice([1, -1])
        field_ = field_ * sgn

        if rng.random() < 0.1:
            x_l = rng.uniform(0.1, 0.45)
            x_r = rng.uniform(0.55, 0.9)
            trns = 0.01
            # NB: kept as-is from the original (it compares raw x_coord
            # against thresholds in [0.1, 0.9], implicitly assuming a
            # roughly-unit interval even though our domain is [-1, 1]).
            mask = 0.5 * (np.tanh((x_coord - x_l) / trns) - np.tanh((x_coord - x_r) / trns))
            field_ = field_ * mask

    if normalized:
        field_ = field_ - field_.min()
        field_ = field_ / field_.max()

    return field_


def _sample_initial_condition(rng: np.random.Generator, x_coord: NDArray, x_len: float) -> NDArray:
    """Random-field initial condition g(x). Matches InitialCondition.reset."""
    return _random_field(rng, x_coord, x_len)


def _sample_field_coef(
    rng: np.random.Generator,
    x_coord: NDArray,
    x_len: float,
    magnitude: float,
    zero_prob: float,
    scalar_prob: float,
    field_prob: float,
) -> Tuple[NDArray, bool]:
    """
    Zero / scalar / spatial-field coefficient, e.g. the source term s(x).
    Matches RandomFieldCoef.reset. Returns (full-length array, is_field).
    """
    probs = np.array([zero_prob, scalar_prob, field_prob], dtype=float)
    coef_type = rng.choice(3, p=probs / probs.sum())
    if coef_type == 0:
        return np.zeros_like(x_coord), False
    if coef_type == 1:
        value = _random_value(rng, magnitude)
        return np.full_like(x_coord, value), False
    return _random_field(rng, x_coord, x_len), True


def _sample_nonneg_field_coef(
    rng: np.random.Generator,
    x_coord: NDArray,
    x_len: float,
    min_val: float,
    max_val: float,
    zero_prob: float,
    scalar_prob: float,
    field_prob: float,
) -> Tuple[NDArray, bool]:
    """
    Zero / scalar / spatial-field non-negative coefficient, e.g. the
    diffusion coefficient kappa(x). Matches NonNegativeCoefField.reset.
    Returns (full-length array, is_field).
    """
    log_min, log_max = np.log(min_val), np.log(max_val)
    probs = np.array([zero_prob, scalar_prob, field_prob], dtype=float)
    coef_type = rng.choice(3, p=probs / probs.sum())
    if coef_type == 0:
        return np.zeros_like(x_coord), False
    if coef_type == 1:
        value = np.exp(rng.uniform(log_min, log_max))
        return np.full_like(x_coord, value), False

    raw = _random_field(rng, x_coord, x_len, normalized=True, smooth=True)
    margin_bottom, span, _ = rng.dirichlet([1, 1, 1]) * (log_max - log_min)
    raw = log_min + margin_bottom + span * raw
    raw = raw.clip(log_min, log_max)
    return np.exp(raw), True


def _sample_poly_sinus_coefs(
    rng: np.random.Generator, magnitude: float, num_sinusoid: int
) -> NDArray:
    """
    Coefficients for f_i(u) = c1*u + c2*u^2 + c3*u^3
        + sum_j c_{j0} h_j(c_{j1}*u + c_{j2}*u^2),
    packed as an array of shape (1 + num_sinusoid, 4):
        row 0:   [_, c1, c2, c3]           (polynomial part)
        row j>0: [c_{j0}, c_{j1}, c_{j2}, sign]  (sign>0 -> sin, else cos)
    Matches SinusoidalTermFi.reset.
    """
    coef = _random_value(rng, magnitude, size=(num_sinusoid + 1, 4))

    # polynomial part: each of the 4 entries kept with prob 0.5; constant
    # term (index 0) always dropped.
    mask = rng.choice(2, size=4).astype(bool)
    coef[0, mask] = 0
    coef[0, 0] = 0

    # sinusoidal part: randomly drop the u or u^2 term inside h_j(...).
    for j in range(1, num_sinusoid + 1):
        op_j_type = rng.choice(3)
        if op_j_type == 1:
            coef[j, 1] = 0
        elif op_j_type == 2:
            coef[j, 2] = 0

    return coef


def _eval_fi(u: NDArray, u2: NDArray, coef: NDArray) -> NDArray:
    """Evaluate f_i(u) given coefficients from `_sample_poly_sinus_coefs`."""
    result = coef[0, 1] * u + coef[0, 2] * u2 + coef[0, 3] * u2 * u
    for j in range(1, coef.shape[0]):
        if coef[j, 0] == 0:
            continue
        op_j = coef[j, 1] * u + coef[j, 2] * u2
        g_j = np.sin(op_j) if coef[j, 3] > 0 else np.cos(op_j)
        result = result + coef[j, 0] * g_j
    return result


def _format_fi(coef: NDArray, u_name: str = "u") -> str:
    """Human-readable string for f_i(u)."""
    terms = []
    c1, c2, c3 = coef[0, 1], coef[0, 2], coef[0, 3]
    if c1 != 0:
        terms.append(f"{c1:.3g}*{u_name}")
    if c2 != 0:
        terms.append(f"{c2:.3g}*{u_name}**2")
    if c3 != 0:
        terms.append(f"{c3:.3g}*{u_name}**3")
    for j in range(1, coef.shape[0]):
        if coef[j, 0] == 0:
            continue
        inner_terms = []
        if coef[j, 1] != 0:
            inner_terms.append(f"{coef[j, 1]:.3g}*{u_name}")
        if coef[j, 2] != 0:
            inner_terms.append(f"{coef[j, 2]:.3g}*{u_name}**2")
        inner = " + ".join(inner_terms) if inner_terms else "0"
        fn_name = "sin" if coef[j, 3] > 0 else "cos"
        terms.append(f"{coef[j, 0]:.3g}*{fn_name}({inner})")
    if not terms:
        return "0"
    return " + ".join(terms).replace("+ -", "- ")


# ---------------------------------------------------------------------------
# PDE specification
# ---------------------------------------------------------------------------


@dataclass
class SinusPDESpec:
    """One fully-specified sampled instance of the 'sinus' PDE family."""

    x_coord: NDArray
    x_len: float
    ic: NDArray
    f0_poly: NDArray
    f1_poly: NDArray
    s: NDArray
    s_is_field: bool
    kappa: NDArray
    kappa_is_field: bool

    def sym_true(self) -> str:
        """
        Human-readable ground-truth equation string. Coefficient fields
        (s(x) / kappa(x)) are shown as symbolic placeholders rather than a
        fabricated closed form -- the actual sampled values are available in
        `coef_dict`.
        """
        f0_str = _format_fi(self.f0_poly)
        f1_str = _format_fi(self.f1_poly)
        s_str = "s(x)" if self.s_is_field else f"{float(self.s[0]):.3g}"
        kappa_str = "kappa(x)" if self.kappa_is_field else f"{float(self.kappa[0]):.3g}"
        return f"u_t + ({f0_str}) + {s_str} + d/dx(({f1_str}) - {kappa_str}*u_x) = 0"

    @property
    def coef_dict(self) -> Dict[str, Union[NDArray, bool]]:
        """Raw ground-truth coefficients, including field arrays when applicable."""
        return {
            "f0_poly": self.f0_poly,
            "f1_poly": self.f1_poly,
            "s": self.s,
            "s_is_field": self.s_is_field,
            "kappa": self.kappa,
            "kappa_is_field": self.kappa_is_field,
            "ic": self.ic,
        }


def _sample_spec(
    rng: np.random.Generator,
    x_coord: NDArray,
    x_len: float,
    num_sinusoid: int,
    coef_magnitude: float,
    kappa_range: Tuple[float, float],
    field_prob: float,
) -> SinusPDESpec:
    zero_prob = scalar_prob = (1.0 - field_prob) / 2.0

    ic = _sample_initial_condition(rng, x_coord, x_len)

    f0_num_sinusoid = int(rng.integers(num_sinusoid + 1))
    f1_num_sinusoid = num_sinusoid - f0_num_sinusoid
    f0_poly = _sample_poly_sinus_coefs(rng, coef_magnitude, f0_num_sinusoid)
    f1_poly = _sample_poly_sinus_coefs(rng, coef_magnitude, f1_num_sinusoid)

    s_field, s_is_field = _sample_field_coef(
        rng, x_coord, x_len, coef_magnitude, zero_prob, scalar_prob, field_prob
    )
    kappa_field, kappa_is_field = _sample_nonneg_field_coef(
        rng, x_coord, x_len, kappa_range[0], kappa_range[1], zero_prob, scalar_prob, field_prob
    )

    return SinusPDESpec(
        x_coord=x_coord,
        x_len=x_len,
        ic=ic,
        f0_poly=f0_poly,
        f1_poly=f1_poly,
        s=s_field,
        s_is_field=s_is_field,
        kappa=kappa_field,
        kappa_is_field=kappa_is_field,
    )


# ---------------------------------------------------------------------------
# Solver backends
# ---------------------------------------------------------------------------


class PDESolverBackend(ABC):
    """Abstract interface for solving one `SinusPDESpec` instance."""

    @abstractmethod
    def solve(self, spec: SinusPDESpec, t_eval: NDArray) -> NDArray:
        """
        Solve the PDE for the given spec.

        Returns
        -------
        usol : (n_x, n_t) ndarray
            Solution field, x on the first axis (matching GridPDEDataset's
            legacy convention), evaluated at each time in `t_eval`.
        """


class NumpySpectralBackend(PDESolverBackend):
    """
    Pure numpy/scipy solver: FFT-based spatial derivatives (periodic domain)
    + an implicit ODE integrator (`scipy.integrate.solve_ivp`, default
    method="BDF") for time stepping.

    Unlike a dedicated constant-coefficient spectral method (which could
    diagonalize the diffusion operator in Fourier space via an integrating
    factor), this backend evaluates the full right-hand side -- including
    spatially-varying kappa(x)/s(x) -- explicitly at every step and lets the
    implicit integrator handle the resulting stiffness. This is slower but
    handles both constant and field-valued coefficients uniformly.
    """

    def __init__(self, method: str = "BDF", rtol: float = 1e-6, atol: float = 1e-8):
        self.method = method
        self.rtol = rtol
        self.atol = atol

    def solve(self, spec: SinusPDESpec, t_eval: NDArray) -> NDArray:
        n_x = spec.x_coord.shape[0]
        wavenumber = 2.0 * np.pi * np.fft.rfftfreq(n_x, d=spec.x_len / n_x)

        def dx(v: NDArray) -> NDArray:
            return np.fft.irfft(1j * wavenumber * np.fft.rfft(v), n=n_x)

        def rhs(_t: float, u: NDArray) -> NDArray:
            u2 = u * u
            f0_val = _eval_fi(u, u2, spec.f0_poly)
            f1_val = _eval_fi(u, u2, spec.f1_poly)
            u_x = dx(u)
            flux = f1_val - spec.kappa * u_x
            return -(f0_val + spec.s + dx(flux))

        sol = solve_ivp(
            rhs,
            (float(t_eval[0]), float(t_eval[-1])),
            spec.ic,
            t_eval=t_eval,
            method=self.method,
            rtol=self.rtol,
            atol=self.atol,
        )
        if not sol.success:
            raise RuntimeError(f"solve_ivp failed: {sol.message}")
        return sol.y.astype(np.float32)  # (n_x, n_t)


class DedalusBackend(PDESolverBackend):
    """
    Reserved interface for a faithful Dedalus-v3 solve, matching the
    original `common.py`'s `PDEDataGenBase._get_dedalus_problem` /
    `gen_solution` (see `custom_sinus.py`'s `SinusoidalPDE._get_dedalus_problem`
    for the periodic-BC case specifically).

    Not implemented here: `dedalus` cannot be installed in this (Windows)
    environment. Implement `solve()` on a Linux environment with `dedalus`
    installed; the coefficient sampling in this module (`SinusPDESpec`,
    `_sample_spec`, etc.) can be reused unchanged -- only the numerical
    solve needs to build and run an actual Dedalus IVP problem from the
    sampled coefficients.
    """

    def solve(self, spec: SinusPDESpec, t_eval: NDArray) -> NDArray:
        raise NotImplementedError(
            "DedalusBackend.solve() is a reserved interface, not yet "
            "implemented. Run this on a Linux environment with `dedalus` "
            "installed, following the original data_generation/common.py + "
            "custom_sinus.py as a reference for building the Dedalus IVP "
            "problem from a SinusPDESpec."
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def load_pdeformer_sinus_benchmark(
    n_pde: int = 20,
    seed: int = 0,
    n_x: int = 128,
    n_t: int = 101,
    num_sinusoid: int = 0,
    coef_magnitude: float = 1.0,
    kappa_range: Tuple[float, float] = (1e-3, 1.0),
    field_prob: float = 1.0 / 3.0,
    backend: Optional[PDESolverBackend] = None,
    u_bound: float = 10.0,
    max_trials_per_pde: int = 20,
) -> "list[GridPDEDataset]":
    """
    Generate a PDE-discovery benchmark by re-solving random instances of the
    PDEformer-1D 'sinus' pretraining PDE family:

        u_t + f0(u) + s(x) + d/dx(f1(u) - kappa(x) u_x) = 0,
        (t,x) in [0,1]x[-1,1], periodic boundary conditions.

    Because each instance is generated here (rather than downloaded), the
    exact ground-truth coefficients are known and attached to each returned
    dataset as `.sym_true` (human-readable string; uses "s(x)"/"kappa(x)"
    placeholders when those coefficients are spatial fields rather than
    constants) and `.coef_dict` (raw coefficient arrays/scalars).

    Parameters
    ----------
    n_pde : int
        Number of PDE instances to generate.
    seed : int
        Seed for the random coefficient/field sampling (fully reproducible).
    n_x, n_t : int
        Spatial grid size and number of saved time snapshots.
    num_sinusoid : int
        Total number of sinusoidal terms (split randomly between f0 and f1).
        0 (default) gives pure polynomial nonlinearities.
    coef_magnitude : float
        Magnitude of the U([-m, m]) distribution used for scalar
        coefficients (polynomial terms, scalar s).
    kappa_range : (float, float)
        (min, max) for the log-uniformly sampled positive diffusion
        coefficient kappa, when it is drawn as zero/scalar/field.
    field_prob : float
        Probability that each of s(x)/kappa(x) is sampled as a spatially
        varying field rather than zero or a scalar (the remaining
        probability mass is split evenly between zero and scalar).
    backend : PDESolverBackend, optional
        Solver backend. Defaults to `NumpySpectralBackend()`.
    u_bound : float
        Reject (and resample) solutions whose |u| exceeds this bound, or
        that are non-finite.
    max_trials_per_pde : int
        Maximum resampling attempts per PDE instance before giving up on it
        (a warning is emitted and that instance is skipped, rather than
        looping forever as the original pretraining-data generator does).

    Returns
    -------
    list of GridPDEDataset
        One dataset per successfully generated PDE instance (may be fewer
        than `n_pde` if some instances repeatedly failed to solve within
        `max_trials_per_pde` attempts).
    """
    if backend is None:
        backend = NumpySpectralBackend()

    rng = np.random.default_rng(seed)
    x_coord = np.linspace(X_L, X_R, n_x, endpoint=False)
    x_len = X_R - X_L
    t_eval = np.linspace(0.0, STOP_SIM_TIME, n_t)

    datasets = []
    for i in range(n_pde):
        accepted = None
        for _ in range(max_trials_per_pde):
            spec = _sample_spec(
                rng, x_coord, x_len, num_sinusoid, coef_magnitude, kappa_range, field_prob
            )
            try:
                usol = backend.solve(spec, t_eval)
            except NotImplementedError:
                raise
            except Exception:
                continue
            if np.isfinite(usol).all() and np.max(np.abs(usol)) <= u_bound:
                accepted = (spec, usol)
                break

        if accepted is None:
            warnings.warn(
                f"load_pdeformer_sinus_benchmark: failed to generate PDE "
                f"instance #{i} within {max_trials_per_pde} trials; skipping."
            )
            continue

        spec, usol = accepted
        descr = DatasetInfo(
            description=f"""
            PDEformer-1D 'sinus' pretraining-family PDE instance #{i}
            (u_t + f0(u) + s(x) + d/dx(f1(u) - kappa(x)u_x) = 0, periodic BC),
            re-generated with a lightweight numpy/scipy spectral+BDF solver
            (NOT the original Dedalus-v3 solver -- see DedalusBackend for a
            reserved interface to add that later).
            Resource: MindFlow PDEformer-1D, data_generation/custom_sinus.py.
            """
        )
        dataset = GridPDEDataset(
            equation_name=f"pdeformer_sinus_{i}",
            pde_data=None,
            x=x_coord,
            t=t_eval,
            usol=usol,
            domain={"x": (float(X_L), float(X_R)), "t": (0.0, float(STOP_SIM_TIME))},
            epi=1e-3,
            descr=descr,
            legacy=True,
        )
        dataset.sym_true = spec.sym_true()
        dataset.coef_dict = spec.coef_dict
        datasets.append(dataset)

    return datasets
