"""Scoring-only coupled ODE integration on whole held-out trajectories."""

import ast
import re
import time

import numpy as np
from scipy.integrate import solve_ivp

PROTOCOL = "ode-rollout-v1"
DEFAULTS = {
    "protocol": PROTOCOL, "enabled": True, "method": "auto",
    "rtol": 1e-8, "atol": 1e-10, "max_rhs_calls": 100000,
    "max_state_abs": 1e8, "reference_nrmse_tolerance": 0.05,
}


def numeric_expression(text, names):
    """Restrict symbolic input before using the shared expression-format parser."""
    from run_benchmark import _normalize_pretty_expression, expression_sympy

    text = str(text).split("=", 1)[-1].strip()
    text = _normalize_pretty_expression(text)
    if not text or len(text) > 20000:
        raise ValueError("Empty, unsupported or oversized symbolic expression")
    text = re.sub(r"\b(?:np|numpy|math)\.", "", text).replace("^", "**")
    # Comma-separated prefix notation is parsed by the existing bounded parser.
    tree = ast.parse(text, mode="eval")
    functions = {"add", "sub", "mul", "div", "pow", "n2", "n3", "n4", "neg",
                 "inv", "sin", "cos", "tan", "exp", "expneg", "log", "sqrt",
                 "tanh", "abs", "Abs", "min", "max"}
    allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load,
               ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow,
               ast.USub, ast.UAdd, ast.Tuple)
    for node in ast.walk(tree):
        if not isinstance(node, allowed):
            raise ValueError("Unsupported symbolic syntax")
        if isinstance(node, ast.Name) and node.id.startswith("_"):
            raise ValueError("Private symbolic name")
        if isinstance(node, ast.Constant) and (isinstance(node.value, bool) or
                                               not isinstance(node.value, (int, float))):
            raise ValueError("Non-numeric symbolic constant")
        if isinstance(node, ast.Call) and (not isinstance(node.func, ast.Name) or
                                          node.func.id not in functions or node.keywords):
            raise ValueError("Unsupported symbolic function")
    import sympy as sp

    expression = expression_sympy(text, names)
    if expression is None:
        raise ValueError("Unparseable symbolic expression")
    expression = expression.xreplace({sp.Symbol(f"x{i + 1}"): sp.Symbol(name)
                                      for i, name in enumerate(names) if f"x{i + 1}" not in names})
    if expression.free_symbols - {sp.Symbol(n) for n in names}:
        raise ValueError("Unresolved symbols in expression")
    if expression.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
        raise IntegrationFailure("invalid_rhs", "Nonfinite symbolic RHS")
    return sp.lambdify([sp.Symbol(n) for n in names], expression, "numpy")


class IntegrationFailure(RuntimeError):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def integrate(functions, trajectory, state_vars, param_names, settings, deadline):
    times = np.asarray(trajectory["t"], float)
    state = np.asarray(trajectory["state"], float)
    if (times.ndim != 1 or len(times) < 2 or state.shape != (len(state_vars), len(times))
            or not np.isfinite(times).all() or not np.isfinite(state).all()
            or not np.all(np.diff(times) > 0)):
        raise IntegrationFailure("invalid_data", "Invalid trajectory times/state")
    parameters = [trajectory["params"][p] for p in param_names]
    calls = 0

    def rhs(_t, values):
        nonlocal calls
        calls += 1
        if time.perf_counter() > deadline or calls > settings["max_rhs_calls"]:
            raise IntegrationFailure("integration_limit", "ODE scoring limit exceeded")
        if not np.isfinite(values).all() or np.max(np.abs(values)) > settings["max_state_abs"]:
            raise IntegrationFailure("diverged", "State exceeded the declared finite bound")
        with np.errstate(all="ignore"):
            result = np.asarray([f(*values, *parameters) for f in functions], float)
        if result.shape != values.shape or not np.isfinite(result).all():
            raise IntegrationFailure("invalid_rhs", "RHS is nonfinite or has an invalid shape")
        return result

    result = solve_ivp(rhs, (times[0], times[-1]), state[:, 0], t_eval=times,
                       method=settings["method"], rtol=settings["rtol"], atol=settings["atol"])
    if not result.success or len(result.t) != len(times):
        raise IntegrationFailure("solver_failed", result.message)
    if not np.isfinite(result.y).all() or np.max(np.abs(result.y)) > settings["max_state_abs"]:
        raise IntegrationFailure("diverged", "Nonfinite or unbounded integrated state")
    # The supplied initial observation is excluded from prediction metrics.
    truth, prediction = state[:, 1:], result.y[:, 1:]
    scale = np.std(truth, axis=1)
    rmse = np.sqrt(np.mean((truth - prediction) ** 2, axis=1))
    nrmse = np.divide(rmse, scale, out=np.full_like(rmse, np.nan), where=scale > 0)
    return {"trajectory": trajectory["id"], "status": "ok", "n_points": truth.shape[1],
            "mse": np.mean((truth - prediction) ** 2, axis=1).tolist(),
            "nrmse": [float(v) if np.isfinite(v) else None for v in nrmse], "rhs_calls": calls}


def evaluate(dataset, rows, case, *, seconds=300):
    """Return per-target state scores; no candidate selection or model fitting."""
    from run_benchmark import split_indices

    settings = {**DEFAULTS, **case.get("evaluation", {}).get("ode_rollout", {})}
    common = {"ode_rollout_protocol_version": PROTOCOL,
              "equation_rollout_nrmse": None, "equation_rollout_mse": None,
              "equation_rollout_status": "abstained_incomplete_system"}
    expected = [f"d({name})/dt" for name in dataset.state_vars]
    by_target = {r["target"]: r for r in rows}
    if not settings["enabled"]:
        return {target: {**common, "equation_rollout_status": "disabled"} for target in expected}
    if len(rows) != len(expected) or set(by_target) != set(expected) or any(
            by_target[t].get("status") != "ok" or not by_target[t].get("expression") for t in expected):
        return {target: dict(common) for target in expected}
    if settings["method"] == "auto":
        settings["method"] = "Radau" if getattr(dataset, "generation_metadata", {}).get("system") == "robertson" else "DOP853"
    if (settings["method"] not in {"RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"}
            or any(not np.isfinite(settings[k]) or settings[k] <= 0
                   for k in ("rtol", "atol", "max_rhs_calls", "max_state_abs"))
            or not np.isfinite(settings["reference_nrmse_tolerance"])
            or settings["reference_nrmse_tolerance"] < 0):
        raise ValueError("Invalid ODE rollout settings")
    train_ids, test_ids = split_indices(dataset.n_traj, case["run_params"]["test_size"], case["seed"])
    common.update(equation_rollout_solver_settings=settings,
                  equation_rollout_integrator=settings["method"],
                  equation_rollout_test_trajectories=[dataset.trajectories[i]["id"] for i in test_ids],
                  equation_rollout_train_trajectories=[dataset.trajectories[i]["id"] for i in train_ids],
                  equation_rollout_initial_condition="first held-out observed state; excluded from score",
                  equation_rollout_normalization="per-trajectory/state held-out std; macro trajectory mean")
    names = list(dataset.state_vars) + list(dataset.param_names)
    deadline = time.perf_counter() + seconds
    truth = getattr(dataset, "sym_true", None)
    if not truth or len(truth) != len(expected):
        return {t: {**common, "equation_rollout_status": "abstained_missing_reference"} for t in expected}
    try:
        references = [numeric_expression(eq, names) for eq in truth]
        reference_scores = [integrate(references, dataset.trajectories[i], dataset.state_vars,
                                      dataset.param_names, settings, deadline) for i in test_ids]
        values = [v for r in reference_scores for v in r["nrmse"]]
        reference_error = max(values) if all(v is not None for v in values) else None
        common["equation_rollout_reference_nrmse"] = reference_error
        common["equation_rollout_reference_tolerance"] = settings["reference_nrmse_tolerance"]
        if reference_error is None or reference_error > settings["reference_nrmse_tolerance"]:
            return {t: {**common, "equation_rollout_status": "abstained_reference_validation_failed"} for t in expected}
    except Exception as exc:
        return {t: {**common, "equation_rollout_status": "abstained_reference_validation_failed",
                    "equation_rollout_error": str(exc)} for t in expected}
    try:
        functions = [numeric_expression(by_target[t]["expression"], names) for t in expected]
    except Exception as exc:
        return {t: {**common, "equation_rollout_status": getattr(exc, "status", "unparsed_rhs"),
                    "equation_rollout_error": str(exc)} for t in expected}
    scores = []
    for i in test_ids:
        try:
            scores.append(integrate(functions, dataset.trajectories[i], dataset.state_vars,
                                    dataset.param_names, settings, deadline))
        except Exception as exc:
            scores.append({"trajectory": dataset.trajectories[i]["id"],
                           "status": getattr(exc, "status", "solver_failed"), "error": str(exc)})
    failures = [s for s in scores if s["status"] != "ok"]
    result = {}
    for j, target in enumerate(expected):
        values = [s["nrmse"][j] for s in scores if s["status"] == "ok"]
        valid = not failures and all(v is not None for v in values)
        result[target] = {**common, "equation_rollout_status": failures[0]["status"] if failures else
                          "ok" if valid else "undefined_normalization",
                          "equation_rollout_nrmse": float(np.mean(values)) if valid else None,
                          "equation_rollout_mse": float(np.mean([s["mse"][j] for s in scores])) if valid else None,
                          "equation_rollout_trajectory_coverage": (len(scores) - len(failures)) / len(scores),
                          "equation_rollout_trajectories": scores}
    return result
