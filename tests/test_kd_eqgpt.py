import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

torch = pytest.importorskip("torch")

from kd.model.eqgpt.gpt import GPT, set_device
from kd.model.eqgpt.reward import (
    calculate_reward,
    delete_duplicate,
    delete_duplicate_columns,
    find_min_no_repeat,
    get_mask_invalid,
    sentence_to_str,
)
from kd.model.eqgpt.terms import calculate_terms
from kd.model.eqgpt.vocab import get_train_dataset, id2word, read_dataset, vocab_size, word2id

# ---------------------------------------------------------------------------
# Vocabulary / handbook dataset
# ---------------------------------------------------------------------------


def test_vocab_loaded():
    assert vocab_size == len(id2word)
    assert word2id["<pad>"] == 0
    for token in ("ut", "u", "ux", "uxx", "uxxx"):
        assert token in word2id


def test_read_dataset_leave_one_out():
    full = read_dataset("")
    without_kdv = read_dataset("KdV equation")
    assert len(full) - len(without_kdv) == 1


def test_get_train_dataset_augmentation():
    seqs = get_train_dataset("KdV equation", augment_times=4, seed=0)
    assert len(seqs) > 0
    for seq in seqs:
        assert seq[0] == word2id["S"]


# ---------------------------------------------------------------------------
# calculate_terms (autograd derivatives on a toy analytic surrogate)
# ---------------------------------------------------------------------------


class _ToyNet(torch.nn.Module):
    """u(x, t) = sin(x) * cos(t), so u_xx == -u and u_xxx == -u_x exactly."""

    def forward(self, xt):
        x, t = xt[:, 0], xt[:, 1]
        return (torch.sin(x) * torch.cos(t)).reshape(-1, 1)


def test_calculate_terms_matches_analytic_derivatives():
    net = _ToyNet()
    database = torch.rand(50, 2, requires_grad=True)
    variables = ["x", "t"]

    u = calculate_terms("u", net, database, variables).reshape(-1)
    uxx = calculate_terms("uxx", net, database, variables)
    ux = calculate_terms("ux", net, database, variables).reshape(-1)
    uxxx = calculate_terms("uxxx", net, database, variables)

    np.testing.assert_allclose(uxx, -u, atol=1e-5)
    np.testing.assert_allclose(uxxx, -ux, atol=1e-5)


# ---------------------------------------------------------------------------
# Reward computation
# ---------------------------------------------------------------------------


def test_delete_duplicate_removes_repeated_additive_group():
    sentence = [word2id["u"], word2id["+"], word2id["u"], word2id["E"]]
    result = delete_duplicate(sentence)
    assert sentence_to_str(result) == "u"


def test_delete_duplicate_columns():
    matrix = np.array([[1, 1, 2], [3, 3, 4]])
    result = delete_duplicate_columns(matrix)
    assert result.shape == (2, 2)


def test_get_mask_invalid_masks_unused_variables():
    device = torch.device("cpu")
    mask = get_mask_invalid(["x", "t"], device)
    # tokens mentioning y/z/Laplace/Div should be masked out
    for i, word in enumerate(id2word):
        if "y" in word or "z" in word or "Laplace" in word or "Div" in word:
            assert mask[i] == 0
    # x/t/u-only tokens remain enabled
    assert mask[word2id["ut"]] == 1
    assert mask[word2id["uxxx"]] == 1


def test_calculate_reward_well_formed_output():
    net = _ToyNet()
    database = torch.rand(200, 2, requires_grad=True)
    variables = ["x", "t"]
    mask = get_mask_invalid(variables, torch.device("cpu"))

    sentence = [word2id["ut"], word2id["+"], word2id["uxx"], word2id["E"]]
    reward, words2value, mask = calculate_reward(sentence, net, database, {}, mask, variables)

    assert isinstance(reward, float)
    assert "ut" in words2value and "uxx" in words2value


def test_find_min_no_repeat():
    rewards = torch.tensor([0.1, 0.9, 0.9, 0.5, 0.2])
    index, award = find_min_no_repeat(rewards, k=3)
    assert len(index) == len(award) == 3
    assert award == sorted(award, reverse=True)


# ---------------------------------------------------------------------------
# GPT sampling
# ---------------------------------------------------------------------------


def test_gpt_step_terminates_and_respects_mask():
    set_device("cpu")
    model = GPT()
    mask = get_mask_invalid(["x", "t"], torch.device("cpu"))

    sentence = [word2id["S"]]
    for _ in range(49):
        next_token, prob = model.step(sentence, mask)
        assert mask[next_token] != 0 or next_token == word2id["E"]
        sentence.append(int(next_token))
        if next_token == word2id["E"]:
            break

    assert sentence[-1] == word2id["E"] or len(sentence) == 50


# ---------------------------------------------------------------------------
# Full pipeline (KD_EqGPT.fit), tiny settings just to check it runs end to end
# ---------------------------------------------------------------------------


def test_kd_eqgpt_fit_runs_end_to_end():
    from kd.dataset import load_kdv_equation
    from kd.model.kd_eqgpt import KD_EqGPT

    dataset = load_kdv_equation()
    model = KD_EqGPT(
        gpt_pretrain_epochs=1,
        gpt_batch_size=64,
        augment_times=2,
        surrogate_iters=20,
        choose=200,
        choose_validate=50,
        optimize_epochs=1,
        samples=3,
        meta_nx=5,
        meta_nt=5,
        device="cpu",
        verbose=False,
        seed=0,
    )
    result = model.fit(dataset, equation_name="kdv")

    assert result is model
    assert isinstance(model.best_pde_, str)
    assert isinstance(model.best_award_, float)
    assert len(model.history_) == 1
