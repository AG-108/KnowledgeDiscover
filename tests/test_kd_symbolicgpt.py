import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

torch = pytest.importorskip("torch")

from kd.model.symbolicgpt.generator import generate_equation, simplify_formula
from kd.model.symbolicgpt.models import GPT, GPTConfig, PointNetConfig
from kd.model.symbolicgpt.utils import (
    CharDataset,
    _constants_loss,
    evaluate_expression,
    fit_constants,
    points_tensor_from_xy,
    sample_from_model,
    sample_points_for_equation,
    set_seed,
)

# ---------------------------------------------------------------------------
# Equation generator
# ---------------------------------------------------------------------------


def test_generate_equation_basic_shape():
    set_seed(0)
    clean_eqn, skeleton_eqn = generate_equation(num_vars=2, op_list=["add", "sub", "mul", "sin"])
    assert isinstance(clean_eqn, str) and len(clean_eqn) > 0
    assert isinstance(skeleton_eqn, str) and "C" in skeleton_eqn
    # skeleton must not leak concrete float constants
    assert not any(ch.isdigit() for ch in skeleton_eqn.replace("x1", "").replace("x2", ""))


def test_simplify_formula_rounds_coefficients_and_removes_tiny_terms():
    assert simplify_formula("0.00001*x1 + 1.23456*x2 - 0.87654", digits=4) == "1.2346*x2-0.8765"


def test_sample_points_for_equation_known_equation():
    X, Y = sample_points_for_equation("2*x1+1", n_points=10, n_vars=1, min_x=-3, max_x=3)
    assert len(X) == len(Y) == 10
    for x, y in zip(X, Y):
        assert np.isclose(y, 2 * x[0] + 1, atol=1e-6)


# ---------------------------------------------------------------------------
# points_tensor_from_xy (shared padding/clipping helper)
# ---------------------------------------------------------------------------


def test_points_tensor_from_xy_padding_and_truncation():
    X = [[1.0], [2.0], [3.0]]
    Y = [10.0, 20.0, 30.0]
    # max_points=5 > len(X): remaining columns stay zero
    pts = points_tensor_from_xy(X, Y, num_vars=1, num_ys=1, max_points=5)
    assert pts.shape == (2, 5)
    np.testing.assert_allclose(pts[:, 0].numpy(), [1.0, 10.0])
    np.testing.assert_allclose(pts[:, 3].numpy(), [0.0, 0.0])

    # max_points=2 < len(X): extra points are dropped
    pts_trunc = points_tensor_from_xy(X, Y, num_vars=1, num_ys=1, max_points=2)
    assert pts_trunc.shape == (2, 2)


def test_points_tensor_from_xy_clips_nonfinite():
    pts = points_tensor_from_xy(
        [[float("nan")]],
        [float("inf")],
        num_vars=1,
        num_ys=1,
        max_points=1,
        threshold=(-1000, 1000),
    )
    assert torch.isfinite(pts).all()


# ---------------------------------------------------------------------------
# Constant fitting
# ---------------------------------------------------------------------------


def test_fit_constants_recovers_linear_coefficients():
    X = list(np.linspace(-2, 2, 20))
    Y = [3.0 * x + (-1.5) for x in X]
    expression, loss = fit_constants("C*x1+C", X, Y)
    assert loss < 1e-6

    # evaluate the fitted expression back at a couple of points
    for x in (-1.0, 0.0, 1.0):
        y_hat = eval(expression.replace("x1", repr(x)))
        assert np.isclose(y_hat, 3.0 * x - 1.5, atol=1e-3)


def test_fit_constants_handles_no_constants():
    expression, loss = fit_constants("x1", [1.0, 2.0], [1.0, 2.0])
    assert expression == "x1"
    assert loss < 1e-9


def test_fitted_negative_constants_preserve_power_precedence(monkeypatch):
    from types import SimpleNamespace

    import kd.model.symbolicgpt.utils as utils

    monkeypatch.setattr(
        utils, "minimize", lambda *args, **kwargs: SimpleNamespace(x=[-2.0], fun=0.0)
    )
    expression, loss = fit_constants("C**2*x1", [[1.0], [2.0]], [4.0, 8.0])
    assert loss == 0.0
    np.testing.assert_allclose(evaluate_expression(expression, [[1.0], [2.0]]), [4.0, 8.0])


def test_fit_constants_rechecks_optimizer_predictions(monkeypatch):
    from types import SimpleNamespace

    import kd.model.symbolicgpt.utils as utils

    monkeypatch.setattr(
        utils, "minimize", lambda *args, **kwargs: SimpleNamespace(x=[1.0], fun=0.0)
    )
    with pytest.raises(ValueError, match="nonfinite"):
        fit_constants("C/x1", [[0.0], [1.0]], [0.0, 1.0])


def test_evaluate_expression_vectorizes_multiple_variables():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [-1.0, 0.5]])
    result = evaluate_expression("2*x1+x2", X)
    np.testing.assert_allclose(result, [4.0, 10.0, -1.5])


@pytest.mark.parametrize(
    "expression",
    ["x1+", "x2", "x1<1", "[x1]", "x1[0]", "x1.real",
     "sin(x1, x1)", "True", "1j*x1"],
)
def test_evaluate_expression_rejects_invalid_grammar(expression):
    with pytest.raises(ValueError):
        evaluate_expression(expression, [[1.0], [2.0]])


def test_evaluate_expression_broadcasts_real_constants():
    np.testing.assert_array_equal(evaluate_expression("2.5", [[1.0], [2.0]]), [2.5, 2.5])


def test_constants_loss_rejects_partial_finite_coverage():
    assert np.isinf(_constants_loss([], "1/x1", [[0.0], [1.0]], [0.0, 1.0]))
    assert np.isinf(_constants_loss([], "x1", [[0.0], [1.0]], [np.nan, 1.0]))


@pytest.mark.parametrize("skeleton", ["x1+", "1/x1", "exp(1000*x1)"])
def test_fit_constants_rejects_invalid_candidates(skeleton):
    with pytest.raises(ValueError):
        fit_constants(skeleton, [[0.0], [1.0]], [0.0, 1.0])


@pytest.mark.parametrize("constants,loss", [([np.nan], 0.0), ([1.0], np.inf), ([1.0], np.nan)])
def test_fit_constants_rejects_nonfinite_optimizer_results(monkeypatch, constants, loss):
    from types import SimpleNamespace

    import kd.model.symbolicgpt.utils as utils

    monkeypatch.setattr(
        utils, "minimize", lambda *args, **kwargs: SimpleNamespace(x=constants, fun=loss)
    )
    with pytest.raises(ValueError):
        fit_constants("C*x1", [[1.0], [2.0]], [1.0, 2.0])


@pytest.fixture
def sampled_model(monkeypatch):
    """Control only training/sampling; exercise the real candidate fitting path."""
    import kd.model.kd_symbolicgpt as adapter

    corpus = [{"X": [[0.0], [1.0]], "Y": [0.0, 1.0],
               "Skeleton": "C*x1+sin(x1)+exp(x1)+x1**2+C/x1"}] * 2
    chars = sorted(set("".join("<" + rec["Skeleton"] + ">" for rec in corpus)) | {"_", ":"})
    stoi = {ch: i for i, ch in enumerate(chars)}
    monkeypatch.setattr(adapter.KD_SymbolicGPT, "_build_pretrain_corpus", lambda *args: corpus)
    monkeypatch.setattr(adapter.Trainer, "train", lambda self: None)

    def install(samples, model=None):
        pending = list(samples)

        def sample(_model, seed_input, *args, **kwargs):
            batch = [pending.pop(0) for _ in range(len(seed_input))]
            width = max(map(len, batch))
            return torch.tensor([[stoi[ch] for ch in s.ljust(width, "_")] for s in batch])

        monkeypatch.setattr(adapter, "sample_from_model", sample)
        if model is None:
            model = adapter.KD_SymbolicGPT(
                embedding_size=8, n_layer=1, n_head=1, device="cpu",
                max_conditioning_points=2, candidate_batch_size=3,
            )
        model.num_candidates = len(samples)
        return model

    return install


def test_fit_keeps_only_valid_finite_candidates(sampled_model):
    model = sampled_model(["<x1+>", "<x1<1>", "<x2>", "<1/x1>", "<C*x1+C>", "<x1>"])
    X = np.array([[-1.0], [0.0], [1.0], [2.0]])
    y = 2 * X[:, 0] + 1
    model.fit(X, y)
    assert len(model.candidates_) == 2
    assert model.train_loss_ < 1e-6
    assert np.isfinite(model.predict(X)).all()
    assert model.candidate_validation_["sampled"] == 6
    assert model.candidate_validation_["accepted"] == 2


@pytest.mark.parametrize("sample", ["<x1+>", "<x1", "<x1_>", "<<x1>", "<>"])
def test_fit_reports_no_valid_candidates(sampled_model, sample):
    model = sampled_model([sample])
    with pytest.raises(RuntimeError, match="No valid SymbolicGPT candidates"):
        model.fit([[0.0], [1.0]], [1.0, 3.0])
    assert model.candidates_ == []
    assert not hasattr(model, "best_expression_")


@pytest.mark.parametrize("expression,loss", [("x1", np.nan), ("x1", np.inf), ("1/x1", 0.0), ("x1+", 0.0)])
def test_fit_validates_fitter_output(sampled_model, monkeypatch, expression, loss):
    import kd.model.kd_symbolicgpt as adapter

    model = sampled_model(["<x1>"])
    monkeypatch.setattr(adapter, "fit_constants", lambda *args: (expression, loss))
    with pytest.raises(RuntimeError, match="No valid SymbolicGPT candidates"):
        model.fit([[0.0], [1.0]], [0.0, 1.0])


def test_failed_refit_clears_previous_solution(sampled_model):
    model = sampled_model(["<C*x1+C>"])
    model.fit([[0.0], [1.0]], [1.0, 3.0])
    sampled_model(["<x1+>"], model=model)
    with pytest.raises(RuntimeError, match="No valid SymbolicGPT candidates"):
        model.fit([[0.0], [1.0]], [1.0, 3.0])
    with pytest.raises(RuntimeError, match="Call fit"):
        model.predict([[0.0]])


def test_fit_dataset_scores_a_valid_candidate(sampled_model):
    from kd.dataset import SymbolicRegressionDataset

    model = sampled_model(["<C*x1+C>"])
    model.fit_dataset(SymbolicRegressionDataset(name="Koza-2"))
    assert np.isfinite(model.test_mse_)



@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_pretraining_cache_preserves_sampling_rng_and_isolates_weights(monkeypatch, device):
    import random

    import kd.model.kd_symbolicgpt as adapter

    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    params = dict(
        embedding_size=8, n_layer=1, n_head=1, device=device, seed=17,
        pretrain_corpus_size=8, pretrain_epochs=1, batch_size=4,
        n_levels=2, op_list=["add", "mul"], num_candidates=4, candidate_batch_size=4,
    )
    X = np.linspace(-1, 1, 8).reshape(-1, 1)
    cache, samples, states = {}, [], []
    original_sample = adapter.sample_from_model

    def record_samples(*args, **kwargs):
        result = original_sample(*args, **kwargs)
        samples.append(result.detach().cpu().clone())
        return result

    monkeypatch.setattr(adapter, "sample_from_model", record_samples)

    def fit(model, y, **kwargs):
        try:
            model.fit(X, y, **kwargs)
        except RuntimeError as exc:
            assert "No valid SymbolicGPT candidates" in str(exc)
        states.append((random.random(), np.random.random(),
                       torch.rand(3), torch.rand(3, device=device).cpu()))

    first = adapter.KD_SymbolicGPT(**params)
    fit(first, X[:, 0], pretrain_cache=cache)
    frozen_weights = {k: v.clone() for k, v in cache["model"].state_dict().items()}
    # Mutation through the first estimator cannot poison the snapshot.
    if hasattr(first, "model_"):
        with torch.no_grad():
            next(first.model_.parameters()).add_(10)
    second = adapter.KD_SymbolicGPT(**params)
    fit(second, 2 * X[:, 0] + 1, pretrain_cache=cache)
    reference = adapter.KD_SymbolicGPT(**params)
    fit(reference, 2 * X[:, 0] + 1)
    assert second.pretraining_["cache_hit"] is True
    assert reference.pretraining_["cache_hit"] is False
    torch.testing.assert_close(samples[1], samples[2], rtol=0, atol=0)
    assert states[1][:2] == states[2][:2]
    for a, b in zip(states[1][2:], states[2][2:]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert second.candidate_validation_ == reference.candidate_validation_
    assert second.candidates_ == reference.candidates_
    for key, weight in cache["model"].state_dict().items():
        torch.testing.assert_close(weight, frozen_weights[key], rtol=0, atol=0)


@pytest.mark.parametrize("change", ["seed", "op_list", "learning_rate", "num_vars", "num_points"])
def test_pretraining_cache_invalidates_changed_parameters_or_shape(monkeypatch, change):
    from kd.model.kd_symbolicgpt import KD_SymbolicGPT

    calls = []
    def pretrain(self, num_vars, num_points, device):
        calls.append((num_vars, num_points))
        return torch.nn.Linear(1, 1), object()

    monkeypatch.setattr(KD_SymbolicGPT, "_pretrain", pretrain)
    monkeypatch.setattr(KD_SymbolicGPT, "_fit_candidates", lambda self, *args: self)
    model = KD_SymbolicGPT(device="cpu")
    cache = {}
    X = np.ones((4, 1))
    model.fit(X, np.zeros(4), pretrain_cache=cache)
    if change in {"seed", "op_list", "learning_rate"}:
        model.set_params(**{change: {"seed": 1, "op_list": ["sin"], "learning_rate": 0.002}[change]})
    elif change == "num_vars":
        X = np.ones((4, 2))
    else:
        X = np.ones((5, 1))
    model.fit(X, np.zeros(len(X)), pretrain_cache=cache)
    assert len(calls) == 2
    assert model.pretraining_["cache_hit"] is False


def test_fit_reuses_only_explicit_pretraining_cache(monkeypatch):
    from kd.model.kd_symbolicgpt import KD_SymbolicGPT

    calls = []

    def fake_pretrain(self, num_vars, num_points, device):
        calls.append((num_vars, num_points, str(device)))
        return object(), object()

    monkeypatch.setattr(KD_SymbolicGPT, "_pretrain", fake_pretrain)
    monkeypatch.setattr(
        KD_SymbolicGPT, "_fit_candidates", lambda self, *args: self
    )
    model = KD_SymbolicGPT(
        embedding_size=8, n_layer=1, n_head=1, device="cpu", seed=0
    )
    cache = {}
    model.fit([[0.0], [1.0]], [0.0, 1.0], pretrain_cache=cache)
    assert model.pretraining_ == {
        "protocol": "synthetic_case_instance_reuse", "cache_hit": False
    }
    model.fit([[0.0], [1.0]], [3.0, 4.0], pretrain_cache=cache)
    assert model.pretraining_ == {
        "protocol": "synthetic_case_instance_reuse", "cache_hit": True
    }
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# CharDataset
# ---------------------------------------------------------------------------


def test_char_dataset_getitem_shapes():
    corpus = [
        {"X": [[0.0], [1.0]], "Y": [1.0, 2.0], "Skeleton": "C*x1+C"},
        {"X": [[0.5], [1.5]], "Y": [1.5, 2.5], "Skeleton": "C*x1+C"},
    ]
    text = "".join("<" + rec["Skeleton"] + ">" for rec in corpus)
    chars = sorted(set(text) | {"_"})
    block_size = max(len(rec["Skeleton"]) for rec in corpus) + 2

    ds = CharDataset(
        corpus, block_size, chars, numVars=1, numYs=1, numPoints=[2, 3], target="Skeleton"
    )
    assert len(ds) == len(corpus) - 1

    inputs, outputs, points, num_vars = ds[0]
    assert inputs.shape == (block_size,)
    assert outputs.shape == (block_size,)
    assert points.shape == (2, 2)  # numVars + numYs, numPoints[1]-1
    assert int(num_vars) == 1
    cached_inputs, cached_outputs, cached_points, cached_num_vars = ds[0]
    assert cached_inputs.data_ptr() == inputs.data_ptr()
    assert cached_outputs.data_ptr() == outputs.data_ptr()
    assert cached_points.data_ptr() == points.data_ptr()
    assert cached_num_vars.data_ptr() == num_vars.data_ptr()


def test_char_dataset_accepts_json_string_entries():
    import json

    rec = {"X": [[0.0]], "Y": [1.0], "Skeleton": "C*x1+C"}
    corpus = [json.dumps(rec), json.dumps(rec)]
    chars = sorted(set("<C*x1+C>") | {"_"})
    ds = CharDataset(
        corpus, block_size=10, chars=chars, numVars=1, numYs=1, numPoints=[1, 2], target="Skeleton"
    )
    inputs, outputs, points, num_vars = ds[0]
    assert points.shape == (2, 1)


# ---------------------------------------------------------------------------
# GPT sampling
# ---------------------------------------------------------------------------


def test_sample_from_model_terminates_with_correct_length():
    set_seed(0)
    chars = list("<>C*x1+_")
    stoi = {ch: i for i, ch in enumerate(chars)}
    block_size = 12

    pconf = PointNetConfig(
        embeddingSize=8, numberofPoints=3, numberofVars=1, numberofYs=1, method="EMB_SUM"
    )
    mconf = GPTConfig(len(chars), block_size, n_layer=1, n_head=1, n_embd=8, padding_idx=stoi["_"])
    model = GPT(mconf, pconf)

    seed_input = torch.tensor([[stoi["<"]]], dtype=torch.long)
    points = torch.zeros(1, 2, 3)
    variables = torch.tensor([[1]], dtype=torch.long)

    steps = 5
    out = sample_from_model(
        model, seed_input, steps, points=points, variables=variables, sample=True, top_p=0.9
    )
    assert out.shape == (1, 1 + steps)


def test_sample_from_model_supports_candidate_batches():
    set_seed(0)
    chars = list("<>C*x1+_")
    stoi = {ch: i for i, ch in enumerate(chars)}
    block_size = 12
    batch_size = 4

    pconf = PointNetConfig(
        embeddingSize=8, numberofPoints=3, numberofVars=1, numberofYs=1, method="EMB_SUM"
    )
    mconf = GPTConfig(
        len(chars), block_size, n_layer=1, n_head=1, n_embd=8, padding_idx=stoi["_"]
    )
    model = GPT(mconf, pconf)

    seed_input = torch.full((batch_size, 1), stoi["<"], dtype=torch.long)
    points = torch.zeros(batch_size, 2, 3)
    variables = torch.ones(batch_size, 1, dtype=torch.long)
    out = sample_from_model(
        model,
        seed_input,
        5,
        points=points,
        variables=variables,
        sample=True,
        top_p=0.9,
    )
    assert out.shape == (batch_size, 6)


# ---------------------------------------------------------------------------
# Full pipeline (KD_SymbolicGPT.fit), tiny settings just to check it runs end to end
# ---------------------------------------------------------------------------


def test_kd_symbolicgpt_fit_returns_valid_candidate_or_explicit_failure():
    from kd.model.kd_symbolicgpt import KD_SymbolicGPT

    np.random.seed(0)
    X = np.linspace(-2, 2, 10).reshape(-1, 1)
    y = 2.0 * X[:, 0] + 1.0

    model = KD_SymbolicGPT(
        embedding_size=8,
        n_layer=1,
        n_head=1,
        pretrain_corpus_size=20,
        pretrain_epochs=1,
        batch_size=16,
        op_list=["add", "sub", "mul"],
        num_candidates=3,
        max_conditioning_points=5,
        candidate_batch_size=3,
        seed=0,
        device="cpu",
        verbose=False,
    )
    try:
        result = model.fit(X, y)
    except RuntimeError as exc:
        assert "No valid SymbolicGPT candidates" in str(exc)
        assert model.candidates_ == []
        assert not hasattr(model, "best_expression_")
        return

    assert result is model
    assert isinstance(model.best_expression_, str)
    assert isinstance(model.best_skeleton_, str)
    assert isinstance(model.train_loss_, float)
    assert np.isfinite(model.train_loss_)
    assert len(model.candidates_) > 0
    assert len(model.conditioning_indices_) == 5

    pred = model.predict(X[:3])
    assert pred.shape == (3,)
    assert np.isfinite(pred).all()


def test_kd_symbolicgpt_fit_dataset_returns_finite_score_or_explicit_failure():
    from kd.dataset import SymbolicRegressionDataset
    from kd.model.kd_symbolicgpt import KD_SymbolicGPT

    dataset = SymbolicRegressionDataset(name="Koza-2")
    model = KD_SymbolicGPT(
        embedding_size=8,
        n_layer=1,
        n_head=1,
        pretrain_corpus_size=20,
        pretrain_epochs=1,
        batch_size=16,
        op_list=["add", "sub", "mul"],
        num_candidates=3,
        seed=0,
        device="cpu",
        verbose=False,
    )
    try:
        model.fit_dataset(dataset)
    except RuntimeError as exc:
        assert "No valid SymbolicGPT candidates" in str(exc)
        assert not hasattr(model, "test_mse_")
        return

    assert isinstance(model.best_expression_, str)
    assert hasattr(model, "test_mse_")
    assert np.isfinite(model.test_mse_)
