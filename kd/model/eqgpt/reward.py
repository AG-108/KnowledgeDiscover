"""
Reward computation for EqGPT's structure-search loop: given a sampled token
sequence, fits a sparse linear combination of the corresponding PDE terms
(evaluated via `terms.calculate_terms` on the surrogate network) against a
designated target term, and scores the fit quality.

Adapted from `kd/dataset/EqGPT/code/continue_train_GPT.py` (original EqGPT
repo): `calculate_reward`, `delete_duplicate`, `delete_dulplicate_A_column`,
`get_mask_invalid`, `find_min_no_repeat`.
"""

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from .terms import calculate_terms
from .vocab import word2id, id2word


def delete_duplicate(sentence: List[int]) -> List[int]:
    """
    Remove duplicate additive groups from a sampled sentence (grouping by
    '+' and comparing the sorted set of term ids within each group), then
    re-append the end token.
    """
    sentence = list(sentence)
    if sentence[-1] == word2id["E"]:
        sentence.pop(-1)

    all_groups, current = [], []
    for token in sentence:
        if token != word2id["+"]:
            current.append(token)
        else:
            all_groups.append(current)
            current = []
    all_groups.append(current)

    seen_signatures, kept_groups = [], []
    for group in all_groups:
        signature = sorted(group[::2])
        if signature not in seen_signatures:
            seen_signatures.append(signature)
            kept_groups.append(group)

    concise = []
    for group in kept_groups:
        concise.extend(group)
        concise.append(word2id["+"])
    concise.pop(-1)
    concise.append(word2id["E"])
    return concise


def delete_duplicate_columns(matrix: np.ndarray) -> np.ndarray:
    """Drop exact-duplicate columns from a design matrix."""
    kept = []
    for i in range(matrix.shape[1]):
        col = matrix[:, i].tolist()
        if col not in kept:
            kept.append(col)
    return np.array(kept).T


def get_mask_invalid(variables: Sequence[str], device: torch.device) -> torch.Tensor:
    """
    A (vocab_size,) mask (1 = allowed, 0 = forbidden) that disables any
    vocabulary token referencing a variable/operator not present in
    `variables` (e.g. all "*y*"/"*z*"/"Laplace"/"BiLaplace"/"Div" tokens
    when the PDE only depends on x and t).
    """
    mask = torch.ones(len(id2word), device=device)
    if "t" not in variables:
        for i, w in enumerate(id2word):
            if "t" in w:
                mask[i] = 0
    if "x" not in variables:
        for i, w in enumerate(id2word):
            if "x" in w:
                mask[i] = 0
    if "y" not in variables:
        for i, w in enumerate(id2word):
            if "y" in w or "Laplace" in w or "BiLaplace" in w or "Div" in w:
                mask[i] = 0
    if "z" not in variables:
        for i, w in enumerate(id2word):
            if "z" in w or "Div" in w:
                mask[i] = 0
    return mask


# Pairs of terms that would make the fitted linear combination degenerate
# (near-perfectly collinear / definitionally redundant); a sampled sentence
# using both members of a pair is rejected outright.
_DEGENERATE_TERM_PAIRS = (
    ("u", "sin(u)"),
    ("u", "sinh(u)"),
    ("sin(u)", "sinh(u)"),
    ("x", "sinx"),
)
_DEGENERATE_TRIPLE = ("u", "u^2", "u^3")


def calculate_reward(sentence: List[int], net: torch.nn.Module, database: torch.Tensor,
                     words2value: Dict[str, np.ndarray], mask_invalid: torch.Tensor,
                     variables: Sequence[str], epi: float = 0.2
                     ) -> Tuple[float, Dict[str, np.ndarray], torch.Tensor]:
    """
    Score one sampled sentence: fit the additive groups after the first
    against the first group (by convention the target/left-hand term, e.g.
    "ut") via least squares, reward = (1 - epi*log10(n_terms)) * R^2.

    Returns (reward, updated words2value cache, updated mask_invalid --
    terms that evaluate to NaN on this surrogate are permanently masked out).
    """
    sentence = delete_duplicate(sentence)
    terms = sentence[::2]
    operators = sentence[1::2]

    if len(operators) == 0 or max(operators) > word2id["S"] - 1 or len(operators) != len(terms):
        return 0.0, words2value, mask_invalid

    groups = []
    column = 1
    divide_next = False
    for term_id, operator in zip(terms, operators):
        word = id2word[term_id]
        if word not in words2value:
            value = calculate_terms(word, net, database, list(variables)).reshape(-1)
            if np.isnan(value).any():
                mask_invalid[term_id] = 0
            words2value[word] = value
        value = words2value[word]

        column = column / value if divide_next else column * value
        divide_next = False

        if operator == word2id["+"]:
            groups.append(column)
            column = 1
        elif operator == word2id["*"]:
            continue
        elif operator == word2id["/"]:
            divide_next = True
        elif operator == word2id["E"]:
            groups.append(column)

    design = np.vstack(groups).T
    design = delete_duplicate_columns(design)

    finite_rows = np.isfinite(design).all(axis=1)
    design = design[finite_rows]

    if np.isnan(design).any() or design.shape[1] < 2:
        reward = 0.0
    else:
        target = design[:, 0].copy()
        try:
            coef, *_ = np.linalg.lstsq(design[:, 1:], -target, rcond=None)
            rhs = design[:, 1:].dot(coef)
            lhs = -design[:, 0]
            ss_res = ((lhs - rhs) ** 2).sum()
            ss_tot = ((lhs - lhs.mean()) ** 2).sum()
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            reward = (1 - epi * np.log10(design.shape[1])) * r2
            if reward > 1e4:
                reward = 0.0
        except np.linalg.LinAlgError:
            reward = 0.0

    term_id_set = set(terms)
    for pair in _DEGENERATE_TERM_PAIRS:
        ids = {word2id[w] for w in pair if w in word2id}
        if ids.issubset(term_id_set):
            reward = 0.0
    triple_ids = {word2id[w] for w in _DEGENERATE_TRIPLE if w in word2id}
    if triple_ids.issubset(term_id_set):
        reward = 0.0

    equation = "".join(id2word[t] for t in sentence)
    for variable in variables:
        if variable not in equation and "Div" not in equation and "Laplace" not in equation:
            reward = 0.0
            break

    return float(reward), words2value, mask_invalid


def sentence_to_str(sentence: Sequence[int]) -> str:
    """Human-readable form of a token-id sentence (e.g. "ut+u*ux+uxxx")."""
    tokens = [id2word[int(t)] for t in sentence]
    return "".join(t for t in tokens if t not in ("S", "E"))


def find_min_no_repeat(all_reward: torch.Tensor, k: int = 10) -> Tuple[List[int], List[float]]:
    """Top-`k` non-duplicate-scoring candidates from one batch of rewards."""
    pool_size = len(all_reward)
    best_index = torch.topk(all_reward, pool_size).indices.data.numpy().tolist()
    best_award = torch.topk(all_reward, pool_size).values.data.numpy().tolist()

    min_index, min_award = [], []
    for index, award in zip(best_index, best_award):
        if award not in min_award:
            min_award.append(award)
            min_index.append(index)
        if len(min_award) == k:
            break
    return min_index, min_award
