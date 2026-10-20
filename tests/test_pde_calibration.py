"""Scientific contracts exercised before the separate PDE calibration run."""
import numpy as np
import pytest

from kd.model.pde_library import library_names, normalized_stlsq


def test_small_physical_coefficient_survives_change_of_units():
    rng = np.random.default_rng(8)
    theta = rng.normal(size=(1000, 5))
    theta[:, 1] *= 1000
    target = theta @ np.array([1, .0025, 0, 0, 0])
    first = normalized_stlsq(theta, target, threshold=.03)
    changed = theta.copy()
    changed[:, 1] *= 100
    second = normalized_stlsq(changed, target, threshold=.03)
    assert np.array_equal(first != 0, [True, True, False, False, False])
    np.testing.assert_allclose(theta @ first, target, atol=1e-10)
    np.testing.assert_allclose(changed @ second, target, atol=1e-10)


def test_library_policy_depends_on_coordinates_not_equation_name():
    positive = library_names("compact", True, np.linspace(1, 2, 31))
    crossing = library_names("compact", True, np.linspace(-1, 1, 31))
    assert "u_x**2" in positive and "u_x/x" in positive
    assert "u_x/x" not in crossing


def test_weak_coordinate_product_rule():
    from kd.model.kd_wsindy import WSINDyPDEModel
    x, t = np.linspace(1, 2, 151), np.linspace(0, 1, 101)
    u = x[:, None]**3 * np.exp(t[None, :])
    model = WSINDyPDEModel(coordinate_terms=True, support_fraction=.2)
    theta, _ = model.weak_system(u, x, t)
    # u_x/x = 3*x*exp(t), independently integrated against the test function.
    from kd.model.kd_wsindy import _test_function_weights
    mx, mt = model.support_["m_x"], model.support_["m_t"]
    dx, dt = x[1]-x[0], t[1]-t[0]
    wx = _test_function_weights(mx, 0, dx, model.test_function_powers_["p_x"])
    wt = _test_function_weights(mt, 0, dt, model.test_function_powers_["p_t"])
    exact = model._weak_convolution(3*x[:, None]*np.exp(t[None, :]), wx, wt, dx, dt)
    slices, _ = model._query_slices(exact.shape)
    np.testing.assert_allclose(theta[:, model.terms_.index("u_x/x")], exact[slices].ravel(), rtol=1e-7, atol=1e-10)


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_deepmod_updates_sparsity_and_refreshes_coefficients(precision):
    pytest.importorskip("deepymod")
    from kd.model.kd_deepmod import KD_DeepMoD
    t, x = np.meshgrid(np.linspace(0, 1, 12), np.linspace(1, 2, 16), indexing="ij")
    X = np.column_stack((t.ravel(), x.ravel()))
    y = np.exp(-t.ravel()) * np.sin(x.ravel())
    model = KD_DeepMoD(training_variant="scheduled", hidden_layers=(8, 8),
        max_steps=12, warmup_steps=3, sparsity_interval=2, threshold=.001,
        coordinate_terms=True, batch_size=64, precision=precision, lbfgs_steps=2)
    model.fit(X, y)
    assert model.sparsity_updates_ > 0
    assert "u_x**2" in model.feature_names_ and "u_x/x" in model.feature_names_
    assert np.isfinite(model.coefficients_[0]).all()
    assert model.completed_steps_ == 12


def test_dlga_crossover_uses_population_size():
    pytest.importorskip("torch")
    from kd.model.kd_dlga import KD_DLGA
    model = KD_DLGA(operators=["u", "u_x"], epi=.001, input_dim=2, device="cpu")
    model.Chrom = [[[0]], [[1]], [[0]], [[1]]]
    model.n_generations = 100
    assert len(model.cross_over()) == 4
