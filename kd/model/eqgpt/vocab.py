"""
Vocabulary and PDE-handbook dataset loading for EqGPT.

Adapted from `kd/dataset/EqGPT/code/read_dataset.py` (original EqGPT repo,
"PDEGPT: Learning from Math handbooks for Partial Differential Equation
Discovery"). Ported as plain functions operating on package-relative
resource paths instead of the original script's cwd-relative file access.

The vocabulary (`dict_datas_0725.json`) and PDE handbook
(`PDE_dataset_0725.xlsx`, 221 PDE forms) are vendored under
`kd/model/eqgpt/data/` so this subpackage does not depend on
`kd/dataset/EqGPT/` remaining on disk.
"""

import itertools
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

_DATA_DIR = Path(__file__).resolve().parent / "data"
_VOCAB_PATH = _DATA_DIR / "dict_datas_0725.json"
_DATASET_PATH = _DATA_DIR / "PDE_dataset_0725.xlsx"

with _VOCAB_PATH.open("r", encoding="utf-8") as _f:
    _dict_datas = json.load(_f)

word2id: Dict[str, int] = _dict_datas["word2id"]
id2word: List[str] = _dict_datas["id2word"]
vocab_size: int = len(word2id)


def read_dataset(exclude_equation_name: str = "") -> List[List[str]]:
    """
    Read the 221-PDE handbook dataset, excluding the row whose "Equation
    name" matches `exclude_equation_name` (leave-one-out: keeps the target
    equation's exact form unseen during GPT pretraining).
    """
    df = pd.read_excel(_DATASET_PATH)
    df = df[df["Equation name"] != exclude_equation_name]
    return [term.split(",") for term in df["Terms"]]


def _permutation(li: list) -> list:
    return list(itertools.permutations(li))


def _combine_permutation(li: List[list]) -> list:
    combined = []
    for l in li:
        combined.extend(l)
        combined.append(word2id["+"])
    combined.pop(-1)
    return combined


def _combine_permutation_multiply(li: List[list]) -> list:
    combined = []
    for l in li:
        if isinstance(l, list):
            combined.extend(l)
        else:
            combined.append(l)
        combined.append(word2id["*"])
    combined.pop(-1)
    return combined


def _split_list(data: list, vocab: int) -> List[list]:
    s_list, temp = [], []
    for item in data:
        if item == vocab:
            s_list.append(temp)
            temp = []
        else:
            temp.append(item)
    s_list.append(temp)
    return s_list


def _data_augmentation_plus(data: list) -> List[list]:
    """Swap the '+'-separated additive groups (all permutations)."""
    all_slice, sl = [], []
    for word in data:
        if word != word2id["+"]:
            sl.append(word)
        else:
            all_slice.append(sl)
            sl = []
    all_slice.append(sl)

    augmented = []
    for li in _permutation(all_slice):
        augmented.append(_combine_permutation(list(li)))
    return augmented


def _data_augmentation_multiply(data: list) -> List[list]:
    """Swap the '*'-separated multiplicative factors within each additive group."""
    if word2id["*"] not in data:
        return [data]

    all_slice = _split_list(data, vocab=word2id["+"])
    augment_index, augment_slice = [], []
    for i, sl in enumerate(all_slice):
        if word2id["*"] in sl:
            augment_index.append(i)
            new_slice = _split_list(sl, vocab=word2id["*"])
            combined_all = [_combine_permutation_multiply(list(li)) for li in _permutation(new_slice)]
            augment_slice.append(combined_all)

    all_augmented, index = [], 0
    for i in range(len(all_slice)):
        augmented = list(all_slice)
        if i in augment_index:
            for data_ in augment_slice[index]:
                augmented[i] = data_
                add_augmented = []
                for d in augmented:
                    add_augmented.extend(d)
                    add_augmented.append(word2id["+"])
                add_augmented.pop(-1)
                all_augmented.append(add_augmented)
            index += 1

    return all_augmented


def _expand_to_wanted_size(arr: list, wanted_size: int) -> list:
    result = arr * math.ceil(wanted_size / len(arr))
    return result[:wanted_size]


def get_train_dataset(exclude_equation_name: str = "", augment_times: int = 32,
                      seed: int = None) -> List[List[int]]:
    """
    Build the GPT pretraining corpus: token-id sequences for every PDE form
    in the handbook (except `exclude_equation_name`), each expanded via
    permutation-based data augmentation (swap order of additive terms /
    multiplicative factors) up to `augment_times` variants, prefixed with
    the start token 'S'.
    """
    if seed is not None:
        random.seed(seed)

    dataset = read_dataset(exclude_equation_name)
    train_data = [[word2id[word] for word in data] for data in dataset]

    all_augmented = []
    for data in train_data:
        augmented_plus = []
        for sl in _data_augmentation_plus(data):
            sl = sl + [word2id["E"]]
            augmented_plus.append(sl)

        all_multiply = []
        for aug_data in augmented_plus:
            all_multiply.extend(_data_augmentation_multiply(aug_data))

        if len(all_multiply) > augment_times:
            all_multiply = random.sample(all_multiply, augment_times)
        else:
            all_multiply = _expand_to_wanted_size(all_multiply, augment_times)

        all_augmented.extend(all_multiply)

    return [[word2id["S"]] + seq for seq in all_augmented]
