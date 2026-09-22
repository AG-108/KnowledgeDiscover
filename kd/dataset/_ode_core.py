"""Small reproducible multi-trajectory ODE benchmark core."""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from ._base import ODEDataset
from ._info import DatasetInfo


SYSTEMS = {
    "oscillator": dict(category="oscillatory", state_vars=["x", "v"],
        equations=["dx/dt = v", "dv/dt = -omega**2*x"], params={"omega": 1.3},
        initial=[1.0, 0.0], t_span=(0.0, 12.0)),
    "population": dict(category="population", state_vars=["prey", "predator"],
        equations=["dprey/dt = alpha*prey-beta*prey*predator", "dpredator/dt = delta*prey*predator-gamma*predator"],
        params={"alpha": 1.1, "beta": 0.4, "delta": 0.1, "gamma": 0.4},
        initial=[8.0, 3.0], t_span=(0.0, 14.0)),
    "chaotic": dict(category="chaotic", state_vars=["x", "y", "z"],
        equations=["dx/dt = sigma*(y-x)", "dy/dt = x*(rho-z)-y", "dz/dt = x*y-beta*z"],
        params={"sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3.0},
        initial=[-8.0, 8.0, 27.0], t_span=(0.0, 8.0)),
    "rational": dict(category="rational", state_vars=["s"],
        equations=["ds/dt = -vmax*s/(km+s)"], params={"vmax": 1.5, "km": 0.7},
        initial=[4.0], t_span=(0.0, 3.0)),
}


def _rhs(name, params):
    if name == "oscillator":
        return lambda _, y: np.array([y[1], -(params["omega"] ** 2) * y[0]])
    if name == "population":
        return lambda _, y: np.array([params["alpha"]*y[0]-params["beta"]*y[0]*y[1], params["delta"]*y[0]*y[1]-params["gamma"]*y[1]])
    if name == "chaotic":
        return lambda _, y: np.array([params["sigma"]*(y[1]-y[0]), y[0]*(params["rho"]-y[2])-y[1], y[0]*y[1]-params["beta"]*y[2]])
    if name == "rational":
        return lambda _, y: np.array([-params["vmax"]*y[0]/(params["km"]+y[0])])
    raise KeyError(name)


def core_equations(system, *, expose_parameters=False):
    """Declared equations; unknown fixed parameters are not model input features."""
    import sympy as sp
    spec = SYSTEMS[system]
    if expose_parameters:
        return list(spec["equations"])
    substitutions = {sp.Symbol(k): v for k, v in spec["params"].items()}
    symbols = {name: sp.Symbol(name) for name in [*spec["state_vars"], *spec["params"]]}
    return [eq.split("=", 1)[0] + "= " + str(sp.sympify(eq.split("=", 1)[1], locals=symbols).subs(substitutions))
            for eq in spec["equations"]]


def generate_ode_core(system, *, n_trajectories=6, n_points=241, seed=0,
                      rtol=1e-10, atol=1e-12, method="DOP853",
                      expose_parameters=False, observation_noise=0.0,
                      time_stride=1, initial_condition_scale=1.0):
    """Generate an ODEDataset; trajectory IDs are the required split groups."""
    if system not in SYSTEMS:
        raise ValueError(f"unknown system {system!r}; choose from {sorted(SYSTEMS)}")
    if int(n_trajectories) != n_trajectories or int(n_points) != n_points or n_trajectories < 2 or n_points < 3:
        raise ValueError("at least two trajectories and three time points are required")
    if not np.isfinite([rtol, atol, observation_noise, initial_condition_scale]).all() or min(rtol, atol, initial_condition_scale) <= 0 or observation_noise < 0:
        raise ValueError("positive finite solver tolerances/initial scale and nonnegative noise required")
    if int(time_stride) != time_stride or time_stride < 1 or len(range(0, n_points, time_stride)) < 3:
        raise ValueError("time_stride must be a positive integer retaining at least three points")
    spec = SYSTEMS[system]
    rng = np.random.default_rng(seed)
    noise_rng = np.random.default_rng(np.random.SeedSequence([seed, 917]))
    base = np.asarray(spec["initial"], float) * initial_condition_scale
    trajectories, initials = [], []
    t = np.linspace(*spec["t_span"], n_points)
    for i in range(n_trajectories):
        y0 = base * (1.0 + rng.uniform(-0.12, 0.12, base.shape))
        # Perturb exact zeros additively so oscillator phases differ.
        y0[base == 0] = rng.uniform(-0.2, 0.2, np.count_nonzero(base == 0))
        sol = solve_ivp(_rhs(system, spec["params"]), spec["t_span"], y0, t_eval=t,
                        method=method, rtol=rtol, atol=atol)
        if not sol.success or sol.t.size != n_points or not np.isfinite(sol.y).all():
            raise RuntimeError(f"ODE solve failed for trajectory {i}: {sol.message}")
        initials.append(y0.tolist())
        clean = sol.y[:, ::time_stride]
        noise_scale = np.std(clean, axis=1, keepdims=True)
        observed = clean + observation_noise * noise_scale * noise_rng.standard_normal(clean.shape)
        trajectories.append({"id": f"{system}_traj_{i:02d}", "t": sol.t[::time_stride],
                             "state": observed, "params": dict(spec["params"]) if expose_parameters else {}})
    ds = ODEDataset(system, trajectories, spec["state_vars"], param_names=list(spec["params"]) if expose_parameters else [],
                    domain={"t": spec["t_span"]},
                    descr=DatasetInfo(description=f"Generated {spec['category']} ODE core system."))
    ds.sym_true = core_equations(system, expose_parameters=expose_parameters)
    ds.generation_metadata = {"system": system, "category": spec["category"],
        "equations": list(spec["equations"]), "parameters": dict(spec["params"]),
        "initial_conditions": initials, "solver": {"method": method, "rtol": rtol, "atol": atol},
        "seed": seed, "n_points": n_points, "split_unit": "whole_trajectory",
        "parameter_protocol": "known_fixed_covariates" if expose_parameters else "unknown_fixed_constants",
        "observation_noise": observation_noise, "noise_definition": "Gaussian, fraction of within-trajectory state std, before derivative estimation",
        "time_stride": time_stride, "initial_condition_scale": initial_condition_scale,
        "domain_shift_note": "Changing initial_condition_scale shifts the entire generated cohort, not a train/test OOD split."}
    return ds


__all__ = ["SYSTEMS", "core_equations", "generate_ode_core"]
