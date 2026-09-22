import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

torch = pytest.importorskip("torch")

from kd.model.symbolicgpt.generator import generate_equation
from kd.model.symbolicgpt.models import GPT, GPTConfig, PointNetConfig
from kd.model.symbolicgpt.utils import (
    CharDataset,
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


def test_evaluate_expression_vectorizes_multiple_variables():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [-1.0, 0.5]])
    result = evaluate_expression("2*x1+x2", X)
    np.testing.assert_allclose(result, [4.0, 10.0, -1.5])


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


def test_kd_symbolicgpt_fit_runs_end_to_end():
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
    result = model.fit(X, y)

    assert result is model
    assert isinstance(model.best_expression_, str)
    assert isinstance(model.best_skeleton_, str)
    assert isinstance(model.train_loss_, float)
    assert len(model.candidates_) > 0
    assert len(model.conditioning_indices_) == 5

    pred = model.predict(X[:3])
    assert pred.shape == (3,)


def test_kd_symbolicgpt_fit_dataset_sets_test_mse():
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
    model.fit_dataset(dataset)

    assert isinstance(model.best_expression_, str)
    assert hasattr(model, "test_mse_")
    assert np.isfinite(model.test_mse_) or np.isnan(model.test_mse_)
