import numpy as np
import pytest
import torch

from kd.model.kd_dlga import KD_DLGA


def test_generate_meta_data_uses_configured_operators(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    operators = ["u", "u_x", "u_xx"]
    model = KD_DLGA(
        operators=operators, epi=0.01, input_dim=2, device="cpu", hidden_layers=1, hidden_width=10
    )
    assert model.user_operators == operators
    assert model.epi == 0.01
    model.best_epoch = 500
    (tmp_path / "model_save").mkdir()
    torch.save(model.Net.state_dict(), tmp_path / "model_save" / "Net_500.pkl")

    model.generate_meta_data(np.random.default_rng(0).random((10, 2)))

    assert model.Theta.shape == (10, len(operators))


@pytest.mark.parametrize("mutate", [False, True], ids=["generation", "mutation"])
def test_genome_respects_operator_bounds(mutate):
    operators = ["u", "u_x"]
    model = KD_DLGA(operators=operators, epi=0.01, input_dim=2, device="cpu")
    if mutate:
        model.mutate_rate = 1.0
        model.pop_size = 1
        model.Chrom = [[[0], [1]]]
        model.mutation()
        genome = model.Chrom[0]
    else:
        genome = model.random_genome()

    assert genome
    assert all(0 <= gene < len(operators) for module in genome for gene in module)


def test_convert_chrom_to_eq_uses_custom_operators():
    model = KD_DLGA(operators=["u_xx", "u", "u_t"], epi=0.01, input_dim=2)
    equation = model.convert_chrom_to_eq([[0], [1]], "u_t", np.array([[2.5], [-1.0]]))
    assert "2.5*u_xx" in equation
    assert "-1.0*u" in equation
    assert "u_t=" in equation


def test_equation_export_preserves_small_coefficients():
    import run_benchmark as benchmark

    model = KD_DLGA(operators=["u_x"], epi=0.01, input_dim=2)
    equation = model.convert_chrom_to_eq([[0]], "u_t", np.array([[-0.7123456789]]))
    metrics = benchmark.expression_metrics(equation, "-0.7123456789*u_x", ["u_x"])
    assert metrics["exact_recovery"] == 1
    assert metrics["coefficient_error"] == 0


def test_double_precision_training_handles_flat_responses(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    X = np.random.default_rng(0).normal(size=(12, 2))
    y = 0.3 * X[:, 0] - 0.7 * X[:, 1]
    models = []
    for response in (y, y[:, None]):
        torch.manual_seed(0)
        np.random.seed(0)
        model = KD_DLGA(
            operators=["u_x"],
            epi=0.01,
            input_dim=2,
            device="cpu",
            precision="float64",
            max_iter=2,
            hidden_layers=1,
            hidden_width=4,
        )
        model.train_NN(X.copy(), response.copy())
        assert next(model.Net.parameters()).dtype == torch.float64
        models.append(model)
    np.testing.assert_allclose(models[0].val_loss_history, models[1].val_loss_history)


def test_neural_least_squares_refines_observations_and_preserves_precision():
    torch.manual_seed(0)
    model = KD_DLGA(
        operators=["u_x"],
        epi=1e-6,
        input_dim=2,
        device="cpu",
        precision="float64",
        hidden_layers=1,
        hidden_width=3,
        least_squares_nfev=200,
    )
    X = torch.tensor(np.random.default_rng(0).uniform(-1, 1, (40, 2)), dtype=torch.float64)
    with torch.no_grad():
        y = model.Net(X).clone()
        for parameter in model.Net.parameters():
            parameter.add_(0.01 * torch.randn_like(parameter))
        before = float(torch.mean((model.Net(X) - y) ** 2))
    model._refine_observations(X, y)
    with torch.no_grad():
        after = float(torch.mean((model.Net(X) - y) ** 2))
    assert after < before * 1e-6
    assert next(model.Net.parameters()).dtype == torch.float64
    np.testing.assert_allclose(
        model.surrogate_optimization_["training_mse"], after, rtol=1e-14, atol=0
    )
