import numpy as np
import pytest

from kd.metrics import (
    PICError,
    PINNTrainingResult,
    PhysicsInformedInformationCriterion,
    PreparedPICReference,
    coefficient_stability,
    fit_tls_coefficients,
    normalized_rmse,
)


def _synthetic_terms():
    time = np.repeat(np.linspace(0.0, 1.0, 41), 5)
    space = np.tile(np.linspace(-1.0, 1.0, 5), 41)
    rhs = np.column_stack((1.0 + time, space + 0.3 * time))
    lhs = rhs @ np.array([2.0, -0.75])
    return time, lhs, rhs


def test_tls_recovers_rhs_coefficients_with_correct_sign():
    _, lhs, rhs = _synthetic_terms()
    assert np.allclose(fit_tls_coefficients(lhs, rhs), [2.0, -0.75], atol=1e-10)


def test_overlapping_window_cv_is_zero_for_stable_equation():
    time, lhs, rhs = _synthetic_terms()
    result = coefficient_stability(time, lhs, rhs, n_windows=6)
    assert result.coefficients.shape == (6, 2)
    assert np.allclose(result.coefficients, [2.0, -0.75], atol=1e-9)
    assert result.r_loss < 1e-10
    assert result.windows[0][1] > result.windows[1][0]


def test_windows_match_author_half_range_and_one_twentieth_shifts():
    time, lhs, rhs = _synthetic_terms()
    result = coefficient_stability(time, lhs, rhs, n_windows=10, window_fraction=0.5)
    assert result.windows[0] == pytest.approx((0.0, 0.5))
    assert result.windows[1] == pytest.approx((0.05, 0.55))
    assert result.windows[-1] == pytest.approx((0.45, 0.95))


def test_zero_mean_coefficient_is_not_silently_regularized():
    time, lhs, rhs = _synthetic_terms()
    # The second term alternates sign by horizon, giving an exactly/near-zero mean.
    lhs_varying = 2.0 * rhs[:, 0] + np.where(time < 0.2, -1.0, 1.0) * rhs[:, 1]
    with pytest.raises(PICError, match="mean is zero"):
        coefficient_stability(time, lhs_varying, rhs, n_windows=2, window_fraction=0.1,
                              mean_tol=1e-8)


def test_physical_loss_uses_one_observation_scale_for_both_outputs():
    ann = np.array([10.0, 14.0])
    pinn = np.array([12.0, 18.0])
    expected = np.sqrt(np.mean((np.array([2.0, 4.0]) / 10.0) ** 2))
    assert normalized_rmse(ann, pinn, (5.0, 15.0)) == pytest.approx(expected)
    with pytest.raises(PICError, match="max > min"):
        normalized_rmse(ann, pinn, (3.0, 3.0))


def test_evaluator_composes_pic_and_preserves_coefficient_provenance():
    time, lhs, rhs = _synthetic_terms()
    ann = np.sin(time)
    reference = PreparedPICReference(time, lhs, ann, (-1.0, 1.0), "split-a/cfg-a/seed-7", 7)
    seen = []

    def trainer(request):
        seen.append(request)
        return PINNTrainingResult(
            output=ann + 0.2,
            refitted_coefficients=np.array([2.01, -0.74]),
            converged=True,
            coefficient_refit_count=25,
            cost={"epochs": 25.0},
        )

    result = PhysicsInformedInformationCriterion(trainer).evaluate(
        reference, rhs, candidate_id="u_t=2*a-.75*b", budget={"epochs": 25}, n_windows=6,
        original_coefficients=[1.9, -0.7],
    )
    assert result.status == "ok"
    assert result.pic == pytest.approx(result.r_loss * result.p_loss)
    assert result.p_loss == pytest.approx(0.1)
    assert np.allclose(result.original_coefficients, [1.9, -0.7])
    assert np.allclose(result.reference_fit_coefficients, [2.0, -0.75])
    assert np.allclose(result.refitted_coefficients, [2.01, -0.74])
    assert seen[0].reference is reference
    assert seen[0].require_epoch_refit is True


@pytest.mark.parametrize(
    "training,status",
    [
        (PINNTrainingResult(np.zeros(205), np.ones(2), False, 5, {}, "diverged"),
         "pinn_not_converged"),
        (PINNTrainingResult(np.zeros(205), np.ones(2), True, 0, {}, ""),
         "protocol_violation"),
    ],
)
def test_evaluator_does_not_fabricate_success(training, status):
    time, lhs, rhs = _synthetic_terms()
    reference = PreparedPICReference(time, lhs, np.zeros_like(time), (0.0, 1.0), "key", 3)
    result = PhysicsInformedInformationCriterion(lambda request: training).evaluate(
        reference, rhs, candidate_id="candidate", budget={"epochs": 5}, n_windows=4
    )
    assert result.status == status
    assert result.pic is None


def test_invalid_pinn_output_is_reported_not_dropped():
    time, lhs, rhs = _synthetic_terms()
    reference = PreparedPICReference(time, lhs, np.zeros_like(time), (0.0, 1.0), "key", 3)
    training = PINNTrainingResult(np.full_like(time, np.nan), np.ones(2), True, 2, {})
    result = PhysicsInformedInformationCriterion(lambda request: training).evaluate(
        reference, rhs, candidate_id="candidate", budget={"epochs": 2}, n_windows=4
    )
    assert result.status == "invalid_input"
    assert result.pic is None
    assert result.cost == {}


def test_trainer_exception_is_an_explicit_failure_status():
    time, lhs, rhs = _synthetic_terms()
    reference = PreparedPICReference(time, lhs, np.zeros_like(time), (0.0, 1.0), "key", 3)

    def failed_backend(request):
        raise RuntimeError("optimizer exploded")

    result = PhysicsInformedInformationCriterion(failed_backend).evaluate(
        reference, rhs, candidate_id="candidate", budget={"epochs": 2}, n_windows=4
    )
    assert result.status == "training_failed"
    assert result.pic is None
    assert "optimizer exploded" in result.message


def test_tls_rejects_underdetermined_truncated_svd():
    with pytest.raises(PICError, match="underdetermined"):
        fit_tls_coefficients([1.0, 2.0], [[1.0, 0.0], [0.0, 1.0]])


def test_tls_rejects_nonidentifiable_duplicate_or_proportional_terms():
    x = np.arange(8.0)
    with pytest.raises(PICError, match="rank deficient"):
        fit_tls_coefficients(3 * x, np.column_stack((x, 2 * x)))


def test_torch_backend_runs_real_autodiff_heat_candidate_on_cpu():
    pytest.importorskip("torch")
    from kd.metrics import TorchPICConfig, evaluate_torch_pic, prepare_torch_pic_reference

    x = np.linspace(0.0, np.pi, 12)
    t = np.linspace(0.0, 0.8, 12)
    coords = np.stack(np.meshgrid(x, t, indexing="ij"), axis=-1).reshape(-1, 2)
    values = np.exp(-coords[:, 1]) * np.sin(coords[:, 0])
    cfg = TorchPICConfig(hidden_width=12, hidden_layers=2, reference_epochs=150,
                         pinn_epochs=3, nx=10, nt=10, n_windows=4, seed=17)
    prepared = prepare_torch_pic_reference(coords, values, config=cfg)
    result = evaluate_torch_pic(prepared, ["u_xx"], candidate_id="heat",
                                original_coefficients=[1.0])
    assert result.status == "ok", result.message
    assert np.isfinite(result.pic)
    assert result.cost["pinn_epochs"] == 3
    assert result.cost["reference_epochs"] == 150
    assert result.original_coefficients.tolist() == [1.0]
    assert prepared.reference_train_rmse < 0.5  # diagnostic, not convergence proof


def test_torch_backend_rejects_unsupported_structure():
    from kd.metrics.pic_torch import _library
    torch = pytest.importorskip("torch")
    z = torch.ones((3, 1), dtype=torch.float64)
    with pytest.raises(PICError, match="unsupported"):
        _library(("sin(u)",), z, z, z)


def test_third_derivative_supports_kdv_library_terms():
    from kd.metrics.pic_torch import _derivatives, _library
    torch = pytest.importorskip("torch")

    class Cubic(torch.nn.Module):
        def forward(self, coordinates):
            return coordinates[:, :1] ** 3 + coordinates[:, 1:2]

    coordinates = torch.tensor([[2.0, 0.5], [-1.0, 0.2]], dtype=torch.float64)
    u, ut, ux, uxx, uxxx = _derivatives(Cubic(), coordinates, create_graph=False)
    assert torch.allclose(uxxx, torch.full_like(uxxx, 6.0))
    library = _library(("u_xxx", "u*u_x"), u, ux, uxx, uxxx)
    assert torch.allclose(library[:, 0], torch.full_like(library[:, 0], 6.0))


def test_public_torch_evaluator_returns_status_and_reference_cost_for_unsupported_candidate():
    from kd.metrics import TorchPICConfig, evaluate_torch_pic, prepare_torch_pic_reference
    pytest.importorskip("torch")
    x = np.linspace(0.0, 1.0, 4)
    t = np.linspace(0.0, 1.0, 4)
    coordinates = np.stack(np.meshgrid(x, t, indexing="ij"), axis=-1).reshape(-1, 2)
    values = coordinates[:, 0] + coordinates[:, 1]
    cfg = TorchPICConfig(hidden_width=3, hidden_layers=1, reference_epochs=1,
                         pinn_epochs=1, nx=4, nt=4, n_windows=2)
    prepared = prepare_torch_pic_reference(coordinates, values, config=cfg)
    result = evaluate_torch_pic(prepared, ["sin(u)"], candidate_id="unsupported")
    assert result.status == "unsupported_candidate"
    assert result.pic is None
    assert result.cost["reference_epochs"] == 1


def test_reference_disk_cache_round_trip_avoids_retraining(tmp_path):
    from kd.metrics import TorchPICConfig, prepare_torch_pic_reference
    pytest.importorskip("torch")
    x = np.linspace(0.0, 1.0, 4)
    t = np.linspace(0.0, 1.0, 4)
    coordinates = np.stack(np.meshgrid(x, t, indexing="ij"), axis=-1).reshape(-1, 2)
    values = np.sin(coordinates[:, 0]) + coordinates[:, 1]
    cfg = TorchPICConfig(hidden_width=3, hidden_layers=1, reference_epochs=2,
                         pinn_epochs=1, nx=4, nt=4, n_windows=2, seed=91)
    first = prepare_torch_pic_reference(coordinates, values, config=cfg, cache_dir=tmp_path)
    second = prepare_torch_pic_reference(coordinates, values, config=cfg, cache_dir=tmp_path)
    assert first.reference.cache_key == second.reference.cache_key
    assert first.reference_cost["reference_cache_hit"] == 0
    assert second.reference_cost["reference_cache_hit"] == 1
    assert second.reference_cost["reference_seconds"] == 0
    assert np.array_equal(first.reference.ann_output, second.reference.ann_output)
    assert len(list(tmp_path.glob("*.npz"))) == 1


def test_reference_quality_gate_rejects_inadequate_ann():
    from kd.metrics import TorchPICConfig, prepare_torch_pic_reference
    pytest.importorskip("torch")
    coordinates = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
    values = np.array([0., 1., 1., 2.])
    config = TorchPICConfig(
        hidden_width=3, hidden_layers=1, reference_epochs=1, pinn_epochs=1,
        nx=3, nt=3, n_windows=2, reference_max_normalized_rmse=1e-12,
    )
    with pytest.raises(PICError, match="quality threshold failed"):
        prepare_torch_pic_reference(coordinates, values, config=config)


def test_configured_pinn_convergence_rule_produces_explicit_failure():
    from kd.metrics import TorchPICConfig, evaluate_torch_pic, prepare_torch_pic_reference
    pytest.importorskip("torch")
    x = np.linspace(0.0, np.pi, 8)
    t = np.linspace(0.0, 0.8, 8)
    coordinates = np.stack(np.meshgrid(x, t, indexing="ij"), axis=-1).reshape(-1, 2)
    values = np.exp(-coordinates[:, 1]) * np.sin(coordinates[:, 0])
    cfg = TorchPICConfig(hidden_width=8, hidden_layers=2, reference_epochs=40,
                         pinn_epochs=2, nx=8, nt=8, n_windows=2, seed=19,
                         pinn_min_relative_loss_improvement=2.0)
    prepared = prepare_torch_pic_reference(coordinates, values, config=cfg)
    result = evaluate_torch_pic(prepared, ["u_xx"], candidate_id="forced-nonconvergence")
    assert result.status == "pinn_not_converged"
    assert result.pic is None
    assert result.cost["pinn_relative_loss_improvement"] < 2.0


def test_reference_grid_and_key_ignore_excluded_heldout_rows():
    from kd.metrics import TorchPICConfig, prepare_torch_pic_reference
    pytest.importorskip("torch")
    train = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
    heldout_a = np.array([[2., 2.]])
    heldout_b = np.array([[200., 300.]])
    values_a = np.array([0., 1., 1., 2., 99.])
    values_b = np.array([0., 1., 1., 2., -999.])
    cfg = TorchPICConfig(hidden_width=3, hidden_layers=1, reference_epochs=1,
                         pinn_epochs=1, nx=3, nt=3, n_windows=2)
    a = prepare_torch_pic_reference(np.vstack((train, heldout_a)), values_a,
                                    train_indices=np.arange(4), config=cfg)
    b = prepare_torch_pic_reference(np.vstack((train, heldout_b)), values_b,
                                    train_indices=np.arange(4), config=cfg)
    assert a.reference.cache_key == b.reference.cache_key
    assert np.array_equal(a.evaluation_coordinates, b.evaluation_coordinates)
    assert a.evaluation_coordinates[:, 0].max() == 1
    assert a.evaluation_coordinates[:, 1].max() == 1


def test_config_indices_and_rng_are_strict_and_isolated():
    from kd.metrics import TorchPICConfig, prepare_torch_pic_reference
    torch = pytest.importorskip("torch")
    coordinates = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
    values = np.array([0., 1., 1., 2.])
    with pytest.raises(PICError, match="integer array"):
        prepare_torch_pic_reference(coordinates, values, train_indices=[0.0, 1.5])
    with pytest.raises(PICError, match="reference_epochs"):
        prepare_torch_pic_reference(coordinates, values,
                                    config=TorchPICConfig(reference_epochs=0))
    cfg = TorchPICConfig(hidden_width=3, hidden_layers=1, reference_epochs=1,
                         pinn_epochs=1, nx=3, nt=3, n_windows=2)
    np.random.seed(1234)
    torch.manual_seed(1234)
    expected_np = np.random.random(2)
    expected_torch = torch.rand(2)
    np.random.seed(1234)
    torch.manual_seed(1234)
    prepare_torch_pic_reference(coordinates, values, config=cfg)
    assert np.array_equal(np.random.random(2), expected_np)
    assert torch.equal(torch.rand(2), expected_torch)
