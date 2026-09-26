"""SymbolicGPT: a character-level GPT that autoregressively samples symbolic
regression equation skeletons conditioned on a point cloud (via a PointNet
encoder), then fits numeric constants against the target data.

Adapted from the original SymbolicGPT repo (Valipour et al., "SymbolicGPT:
A Generative Transformer Model for Symbolic Regression", 2021), vendored
optionally stored locally at `kd/dataset/SymbolicGPT/`. See each module's docstring here
for exactly what changed during adaptation (`models.py`, `trainer.py`,
`generator.py`, `utils.py`).

Used by `kd.model.kd_symbolicgpt.KD_SymbolicGPT`.
"""

from .models import GPT, GPTConfig, PointNetConfig
from .trainer import Trainer, TrainerConfig
from .generator import generate_equation
from .utils import (
    CharDataset,
    evaluate_expression,
    fit_constants,
    points_tensor_from_xy,
    sample_from_model,
    sample_points_for_equation,
    set_seed,
)

__all__ = [
    "GPT",
    "GPTConfig",
    "PointNetConfig",
    "Trainer",
    "TrainerConfig",
    "generate_equation",
    "CharDataset",
    "evaluate_expression",
    "fit_constants",
    "points_tensor_from_xy",
    "sample_from_model",
    "sample_points_for_equation",
    "set_seed",
]
