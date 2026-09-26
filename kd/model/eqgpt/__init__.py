"""
EqGPT: generative PDE-structure discovery via a GPT pretrained on a math
handbook of 221 known PDE forms, fine-tuned per target dataset using
reward signals from a sparse-regression fit over surrogate-network
derivatives.

Adapted from the original EqGPT repository (Apache License 2.0),
optionally stored locally at `kd/dataset/EqGPT/` in this project:
    "PDEGPT: Learning from Math handbooks for Partial Differential
    Equation Discovery" (https://www.nature.com/articles/s41467-025-65114-2)

This subpackage covers the first three stages of the original pipeline
(GPT pretraining -> surrogate network -> reward-guided structure search).
The fourth stage (PINN-based coefficient refinement, a separate ~750-line
module in the original repo) is not covered here.
"""

from .gpt import GPT, set_device
from .surrogate import NN, train_surrogate_model, random_data, generate_meta_grid
from .pretrain import pretrain_gpt
from .reward import calculate_reward, get_mask_invalid, find_min_no_repeat, sentence_to_str
from .vocab import word2id, id2word, vocab_size, get_train_dataset

__all__ = [
    "GPT", "set_device",
    "NN", "train_surrogate_model", "random_data", "generate_meta_grid",
    "pretrain_gpt",
    "calculate_reward", "get_mask_invalid", "find_min_no_repeat", "sentence_to_str",
    "word2id", "id2word", "vocab_size", "get_train_dataset",
]
