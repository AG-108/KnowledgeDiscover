"""Shared, explicit 1-D PDE method-of-lines rollout."""

import numpy as np
from scipy.integrate import solve_ivp


def rollout(rhs, initial_state, x, times, *, boundary_conditions, method="RK45",
            rtol=1e-6, atol=1e-8, max_step=None):
    """Integrate a semidiscrete periodic RHS; the supplied RHS enforces its stencil."""
    if not isinstance(boundary_conditions, dict) or boundary_conditions.get("type") != "periodic":
        raise ValueError("boundary_conditions currently support only explicit periodic problems")
    x, times = np.asarray(x, float), np.asarray(times, float)
    state = np.asarray(initial_state, float)
    if x.ndim != 1 or times.ndim != 1 or state.shape != x.shape or len(times) < 2:
        raise ValueError("rollout requires 1-D x/state and at least two output times")
    if len(x) < 3 or not all(np.isfinite(a).all() for a in (x, times, state)):
        raise ValueError("rollout needs at least three finite spatial points and finite times/state")
    if not np.isfinite([rtol, atol]).all() or min(rtol, atol) <= 0:
        raise ValueError("positive finite solver tolerances are required")
    if not (np.diff(x) > 0).all() or not (np.diff(times) > 0).all():
        raise ValueError("x and times must be strictly increasing")

    def wrapped(_time, values):
        derivative = np.asarray(rhs(values, x, boundary_conditions), float)
        if derivative.shape != values.shape or not np.isfinite(derivative).all():
            raise ValueError("PDE RHS returned invalid shape or non-finite values")
        return derivative

    options = dict(method=method, t_eval=times, rtol=rtol, atol=atol)
    if max_step is not None:
        if not np.isfinite(max_step) or max_step <= 0:
            raise ValueError("max_step must be finite and positive")
        options["max_step"] = float(max_step)
    solution = solve_ivp(wrapped, (times[0], times[-1]), state, **options)
    if not solution.success or solution.y.shape != (len(x), len(times)):
        raise RuntimeError(f"PDE integration failed: {solution.message}")
    if not np.isfinite(solution.y).all():
        raise RuntimeError("PDE integration produced non-finite values")
    return dict(prediction=solution.y, solver=method, rtol=rtol, atol=atol,
                max_step=max_step, boundary_conditions=dict(boundary_conditions),
                n_rhs_evaluations=solution.nfev, status="ok")


def validate_solver(rhs, initial_state, x, times, *, boundary_conditions, reference,
                    nrmse_tolerance, **settings):
    """Check declared solver settings against a ground-truth trajectory."""
    result = rollout(rhs, initial_state, x, times,
                     boundary_conditions=boundary_conditions, **settings)
    reference = np.asarray(reference, float)
    if not np.isfinite(reference).all() or not np.isfinite(nrmse_tolerance) or nrmse_tolerance < 0:
        raise ValueError("reference data and nonnegative validation tolerance must be finite")
    if reference.shape != result["prediction"].shape:
        raise ValueError("reference trajectory shape does not match solver output")
    scale = float(np.std(reference))
    rmse = float(np.sqrt(np.mean((result["prediction"] - reference) ** 2)))
    nrmse = rmse / scale if scale > 0 else (0.0 if rmse == 0 else np.inf)
    result.update(validation_nrmse=nrmse, validation_tolerance=float(nrmse_tolerance),
                  validation_passed=bool(nrmse <= nrmse_tolerance))
    return result
