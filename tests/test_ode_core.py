import numpy as np

from kd.dataset._ode_core import SYSTEMS, generate_ode_core


def test_all_core_families_are_finite_and_fully_documented():
    assert {v["category"] for v in SYSTEMS.values()} == {"oscillatory", "population", "chaotic", "rational"}
    for name in SYSTEMS:
        ds = generate_ode_core(name, n_trajectories=3, n_points=31, seed=4)
        assert ds.n_traj == 3
        assert len(set(tr["id"] for tr in ds.trajectories)) == 3
        assert all(np.isfinite(tr["state"]).all() for tr in ds.trajectories)
        meta = ds.generation_metadata
        assert meta["equations"] and meta["parameters"] and meta["initial_conditions"]
        assert meta["solver"] == {"method": "DOP853", "rtol": 1e-10, "atol": 1e-12}
        assert meta["split_unit"] == "whole_trajectory"


def test_generation_is_reproducible_and_initial_conditions_vary():
    a = generate_ode_core("oscillator", n_trajectories=4, n_points=25, seed=12)
    b = generate_ode_core("oscillator", n_trajectories=4, n_points=25, seed=12)
    for ta, tb in zip(a.trajectories, b.trajectories):
        np.testing.assert_array_equal(ta["state"], tb["state"])
    assert len({tuple(v) for v in a.generation_metadata["initial_conditions"]}) == 4


def test_rational_rhs_matches_recorded_equation_at_initial_point():
    ds = generate_ode_core("rational", n_trajectories=2, n_points=501, seed=2)
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
