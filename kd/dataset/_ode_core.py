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
    "damped_oscillator": dict(category="damped_oscillatory", state_vars=["x", "v"],
        equations=["dx/dt = v", "dv/dt = -2*zeta*omega0*v-omega0**2*x"],
        params={"zeta": 0.1, "omega0": 1.3}, initial=[1.0, 0.0], t_span=(0.0, 12.0)),
    "pendulum": dict(category="nonlinear_oscillatory", state_vars=["theta", "v"],
        equations=["dtheta/dt = v", "dv/dt = -gamma*v-omega0**2*sin(theta)"],
        params={"gamma": 0.08, "omega0": 1.2}, initial=[1.2, 0.0], t_span=(0.0, 18.0)),
    "duffing": dict(category="nonlinear_oscillatory", state_vars=["x", "v"],
        equations=["dx/dt = v", "dv/dt = -delta*v-alpha*x-beta*x**3"],
        params={"delta": 0.12, "alpha": 1.0, "beta": 0.8},
        initial=[1.0, 0.0], t_span=(0.0, 18.0)),
    "van_der_pol": dict(category="nonlinear_oscillatory", state_vars=["x", "v"],
        equations=["dx/dt = v", "dv/dt = mu*(1-x**2)*v-x"],
        params={"mu": 2.0}, initial=[2.0, 0.0], t_span=(0.0, 20.0)),
    "sir": dict(category="epidemic", state_vars=["susceptible", "infected", "recovered"],
        equations=["dsusceptible/dt = -beta*susceptible*infected",
                   "dinfected/dt = beta*susceptible*infected-gamma*infected",
                   "drecovered/dt = gamma*infected"],
        params={"beta": 0.9, "gamma": 0.25}, initial=[0.96, 0.039, 0.001],
        t_span=(0.0, 40.0), conserved_total=True),
    "fitzhugh_nagumo": dict(category="excitable", state_vars=["v", "w"],
        equations=["dv/dt = v-v**3/3-w+current",
                   "dw/dt = epsilon*(v+a-b*w)"],
        params={"current": 0.5, "epsilon": 0.08, "a": 0.7, "b": 0.8},
        initial=[-1.0, 1.0], t_span=(0.0, 60.0)),
    "brusselator": dict(category="chemical", state_vars=["x", "y"],
        equations=["dx/dt = A-(B+1)*x+x**2*y", "dy/dt = B*x-x**2*y"],
        params={"A": 1.0, "B": 3.0}, initial=[1.0, 1.0], t_span=(0.0, 20.0)),
    "robertson": dict(category="stiff_chemical", state_vars=["x", "y", "z"],
        equations=["dx/dt = -k1*x+k3*y*z",
                   "dy/dt = k1*x-k2*y**2-k3*y*z", "dz/dt = k2*y**2"],
        params={"k1": 0.04, "k2": 3e7, "k3": 1e4},
        initial=[1.0, 0.0, 0.0], t_span=(0.0, 100.0),
        method="Radau", time_sampling="log", conserved_total=True,
        zero_initial_scale=1e-8),
}

# Frozen discovery comparison set. The existing rational system implements
# Michaelis--Menten saturation decay; it is not a second, duplicate dataset.
CORE_ODE_TRACK = (
    "damped_oscillator",
    "pendulum",
    "duffing",
    "van_der_pol",
    "sir",
    "fitzhugh_nagumo",
    "rational",
    "robertson",
)
CORE_ODE_DATASETS = tuple(f"ode_core_{name}" for name in CORE_ODE_TRACK)


def _rhs(name, params):
    if name == "oscillator":
        return lambda _, y: np.array([y[1], -(params["omega"] ** 2) * y[0]])
    if name == "population":
        return lambda _, y: np.array([params["alpha"]*y[0]-params["beta"]*y[0]*y[1], params["delta"]*y[0]*y[1]-params["gamma"]*y[1]])
    if name == "chaotic":
        return lambda _, y: np.array([params["sigma"]*(y[1]-y[0]), y[0]*(params["rho"]-y[2])-y[1], y[0]*y[1]-params["beta"]*y[2]])
    if name == "rational":
        return lambda _, y: np.array([-params["vmax"]*y[0]/(params["km"]+y[0])])
    if name == "damped_oscillator":
        return lambda _, y: np.array([y[1], -2*params["zeta"]*params["omega0"]*y[1]-params["omega0"]**2*y[0]])
    if name == "pendulum":
        return lambda _, y: np.array([y[1], -params["gamma"]*y[1]-params["omega0"]**2*np.sin(y[0])])
    if name == "duffing":
        return lambda _, y: np.array([y[1], -params["delta"]*y[1]-params["alpha"]*y[0]-params["beta"]*y[0]**3])
    if name == "van_der_pol":
        return lambda _, y: np.array([y[1], params["mu"]*(1-y[0]**2)*y[1]-y[0]])
    if name == "sir":
        return lambda _, y: np.array([
            -params["beta"]*y[0]*y[1],
            params["beta"]*y[0]*y[1]-params["gamma"]*y[1],
            params["gamma"]*y[1],
        ])
    if name == "fitzhugh_nagumo":
        return lambda _, y: np.array([y[0]-y[0]**3/3-y[1]+params["current"],
                                       params["epsilon"]*(y[0]+params["a"]-params["b"]*y[1])])
    if name == "brusselator":
        return lambda _, y: np.array([params["A"]-(params["B"]+1)*y[0]+y[0]**2*y[1],
                                       params["B"]*y[0]-y[0]**2*y[1]])
    if name == "robertson":
        return lambda _, y: np.array([
            -params["k1"]*y[0]+params["k3"]*y[1]*y[2],
            params["k1"]*y[0]-params["k2"]*y[1]**2-params["k3"]*y[1]*y[2],
            params["k2"]*y[1]**2,
        ])
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
                      rtol=1e-10, atol=1e-12, method=None,
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
    solver_method = method if method is not None else spec.get("method", "DOP853")
    rng = np.random.default_rng(seed)
    noise_rng = np.random.default_rng(np.random.SeedSequence([seed, 917]))
    base = np.asarray(spec["initial"], float) * initial_condition_scale
    trajectories, initials = [], []
    time_sampling = spec.get("time_sampling", "uniform")
    if time_sampling == "log":
        t = np.r_[spec["t_span"][0], np.geomspace(1e-6, spec["t_span"][1], n_points - 1)]
    else:
        t = np.linspace(*spec["t_span"], n_points)
    rhs = _rhs(system, spec["params"])
    for i in range(n_trajectories):
        y0 = base * (1.0 + rng.uniform(-0.12, 0.12, base.shape))
        # Perturb exact zeros additively so oscillator phases differ.
        zero_count = np.count_nonzero(base == 0)
        if spec.get("conserved_total") and zero_count:
            y0[base == 0] = rng.uniform(0.0, spec.get("zero_initial_scale", 1e-8) * initial_condition_scale, zero_count)
        else:
            y0[base == 0] = rng.uniform(-0.2, 0.2, zero_count)
        if spec.get("conserved_total"):
            y0 *= initial_condition_scale / y0.sum()
        sol = solve_ivp(rhs, spec["t_span"], y0, t_eval=t,
                        method=solver_method, rtol=rtol, atol=atol)
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
        "benchmark_track": "core_ode_v1" if system in CORE_ODE_TRACK else "supplementary_ode",
        "equations": list(spec["equations"]), "parameters": dict(spec["params"]),
        "initial_conditions": initials, "solver": {"method": solver_method, "rtol": rtol, "atol": atol},
        "seed": seed, "n_points": n_points, "split_unit": "whole_trajectory",
        "time_sampling": time_sampling,
        "conserved_total": spec.get("conserved_total", False),
        "parameter_protocol": "known_fixed_covariates" if expose_parameters else "unknown_fixed_constants",
        "observation_noise": observation_noise, "noise_definition": "Gaussian, fraction of within-trajectory state std, before derivative estimation",
        "time_stride": time_stride, "initial_condition_scale": initial_condition_scale,
        "domain_shift_note": "Changing initial_condition_scale shifts the entire generated cohort, not a train/test OOD split."}
    return ds


__all__ = ["SYSTEMS", "CORE_ODE_TRACK", "CORE_ODE_DATASETS", "core_equations", "generate_ode_core"]
