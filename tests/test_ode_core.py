import numpy as np

from kd.dataset._ode_core import CORE_ODE_DATASETS, CORE_ODE_TRACK, SYSTEMS, _rhs, generate_ode_core


NEW_SYSTEMS = {
    "damped_oscillator", "pendulum", "duffing", "van_der_pol", "sir",
    "fitzhugh_nagumo", "brusselator", "robertson",
}


def test_frozen_core_ode_track_covers_eight_distinct_structures():
    assert CORE_ODE_TRACK == (
        "damped_oscillator", "pendulum", "duffing", "van_der_pol", "sir",
        "fitzhugh_nagumo", "rational", "robertson",
    )
    assert len(CORE_ODE_TRACK) == len(set(CORE_ODE_TRACK)) == 8
    assert set(CORE_ODE_TRACK) <= set(SYSTEMS)
    assert CORE_ODE_DATASETS == tuple(f"ode_core_{name}" for name in CORE_ODE_TRACK)
    assert set(SYSTEMS) - set(CORE_ODE_TRACK) == {
        "oscillator", "population", "chaotic", "brusselator",
    }


def test_all_core_families_are_finite_and_fully_documented():
    assert len(SYSTEMS) == 12
    assert NEW_SYSTEMS <= set(SYSTEMS)
    for name in SYSTEMS:
        ds = generate_ode_core(name, n_trajectories=3, n_points=31, seed=4)
        assert ds.n_traj == 3
        assert len(set(tr["id"] for tr in ds.trajectories)) == 3
        assert all(np.isfinite(tr["state"]).all() for tr in ds.trajectories)
        meta = ds.generation_metadata
        assert meta["equations"] and meta["parameters"] and meta["initial_conditions"]
        assert meta["solver"] == {
            "method": "Radau" if name == "robertson" else "DOP853",
            "rtol": 1e-10, "atol": 1e-12,
        }
        assert meta["split_unit"] == "whole_trajectory"
        assert meta["benchmark_track"] == (
            "core_ode_v1" if name in CORE_ODE_TRACK else "supplementary_ode")


def test_new_rhs_matches_declared_equations_and_catalog():
    import sympy as sp
    from kd.dataset import list_datasets

    assert {f"ode_core_{name}" for name in NEW_SYSTEMS} <= set(list_datasets("ode"))
    for name in NEW_SYSTEMS:
        spec = SYSTEMS[name]
        state = np.asarray(spec["initial"], dtype=float) + 0.2
        locals_ = {key: sp.Symbol(key) for key in [*spec["state_vars"], *spec["params"]]}
        substitutions = {
            **{locals_[key]: value for key, value in zip(spec["state_vars"], state)},
            **{locals_[key]: value for key, value in spec["params"].items()},
        }
        declared = [float(sp.sympify(eq.split("=", 1)[1], locals=locals_).subs(substitutions))
                    for eq in spec["equations"]]
        np.testing.assert_allclose(_rhs(name, spec["params"])(0, state), declared, rtol=1e-12)


def test_conserved_systems_have_physical_initial_states_and_stiff_sampling():
    for name in ("sir", "robertson"):
        ds = generate_ode_core(name, n_trajectories=3, n_points=75, seed=3)
        assert ds.generation_metadata["conserved_total"] is True
        for trajectory in ds.trajectories:
            np.testing.assert_allclose(trajectory["state"].sum(axis=0), 1.0, atol=1e-8)
            assert trajectory["state"].min() >= -1e-9
    stiff = generate_ode_core("robertson", n_trajectories=2, n_points=75, time_stride=2)
    assert stiff.generation_metadata["time_sampling"] == "log"
    assert stiff.generation_metadata["solver"]["method"] == "Radau"
    assert len(stiff.trajectories[0]["t"]) == 38
    assert np.all(np.diff(stiff.trajectories[0]["t"]) > 0)


def test_generation_is_reproducible_and_initial_conditions_vary():
    a = generate_ode_core("oscillator", n_trajectories=4, n_points=25, seed=12)
    b = generate_ode_core("oscillator", n_trajectories=4, n_points=25, seed=12)
    for ta, tb in zip(a.trajectories, b.trajectories):
        np.testing.assert_array_equal(ta["state"], tb["state"])
    assert len({tuple(v) for v in a.generation_metadata["initial_conditions"]}) == 4


def test_rational_rhs_matches_recorded_equation_at_initial_point():
    import sympy as sp

    ds = generate_ode_core("rational", n_trajectories=2, n_points=501, seed=2)
    assert ds.generation_metadata["benchmark_track"] == "core_ode_v1"
    s = sp.Symbol("s")
    rhs = sp.sympify(ds.sym_true[0].split("=", 1)[1])
    assert sp.simplify(rhs + 1.5*s/(0.7+s)) == 0
    tr = ds.trajectories[0]
    s0 = tr["state"][0, 0]
    numerical = (tr["state"][0, 1] - s0) / (tr["t"][1] - tr["t"][0])
    expected = -1.5 * s0 / (0.7 + s0)
    np.testing.assert_allclose(numerical, expected, rtol=2e-3)


def test_unknown_fixed_coefficients_are_not_input_features():
    from kd.dataset import load_dataset
    ds = load_dataset("ode_core_chaotic", n_trajectories=2, n_points=15)
    assert ds.param_names == []
    assert all(not tr["params"] for tr in ds.trajectories)
    assert all("sigma" not in equation and "beta" not in equation for equation in ds.sym_true)
    assert "sigma" in generate_ode_core("chaotic", expose_parameters=True).sym_true[0]


def test_noise_sparsity_are_applied_and_keep_paired_initial_conditions():
    clean = generate_ode_core("oscillator", n_points=41, seed=8)
    sparse = generate_ode_core("oscillator", n_points=41, seed=8, time_stride=2)
    noisy = generate_ode_core("oscillator", n_points=41, seed=8, observation_noise=0.05)
    for a, b, c in zip(clean.trajectories, sparse.trajectories, noisy.trajectories):
        np.testing.assert_array_equal(a["state"][:, ::2], b["state"])
        assert not np.array_equal(a["state"], c["state"])
    assert clean.generation_metadata["initial_conditions"] == noisy.generation_metadata["initial_conditions"]
