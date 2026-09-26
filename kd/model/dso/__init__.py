"""DSO (Deep Symbolic Optimization): symbolic regression via risk-seeking
policy gradients, ported from TensorFlow 1.x to PyTorch.

An RNN policy autoregressively emits pre-order traversals of expression trees;
the top-epsilon quantile of each sampled batch (by reward) trains the policy via
a risk-seeking policy gradient, so the objective targets best-case rather than
average-case performance.

Petersen et al., "Deep symbolic regression: Recovering mathematical expressions
from data via risk-seeking policy gradients", ICLR 2021.
Original repo (BSD-3-Clause): https://github.com/dso-org/deep-symbolic-optimization
optionally stored locally at `kd/dataset/DeepSymbolicOptimization/`.

Port notes
----------
The numpy-only core (program/library/functions/prior/subroutines, the regression
task, and the benchmark loader) is carried over essentially verbatim -- only
import paths changed. What actually required porting was the TensorFlow layer:
the RNN policy, the state manager, and the policy-gradient optimizers, rewritten
in PyTorch following `kd/model/discover/` (itself a PyTorch reimplementation of
DSO, specialized for PDE discovery). See each module's docstring for specifics.

Not ported: the control/ and binding/ task subpackages (episodic RL, needs
gym/pybullet), the PPO optimizer (marked EXPERIMENTAL and commented out in the
original's own configs), the TensorFlow language-model prior (off by default),
and the file-based checkpoint/logging system.

Used by `kd.model.kd_dso.KD_DSO`.
"""

from .program import Program, from_tokens, from_str_tokens
from .library import Library, Token, PlaceholderConstant
from .functions import create_tokens
from .prior import make_prior
from .memory import Batch, make_queue
from .state_manager import make_state_manager, StateManager, HierarchicalStateManager
from .policy import RNNPolicy, safe_cross_entropy
from .train import Trainer
from .core import DeepSymbolicOptimizer, load_default_config
from .task import make_task, set_task, Task, HierarchicalTask
from .task.regression.regression import RegressionTask, make_regression_metric
from .task.regression.dataset import BenchmarkDataset

__all__ = [
    # Top-level entry point
    "DeepSymbolicOptimizer",
    "load_default_config",
    # Training
    "Trainer",
    # Policy (PyTorch)
    "RNNPolicy",
    "safe_cross_entropy",
    "make_state_manager",
    "StateManager",
    "HierarchicalStateManager",
    # Expression machinery
    "Program",
    "from_tokens",
    "from_str_tokens",
    "Library",
    "Token",
    "PlaceholderConstant",
    "create_tokens",
    "make_prior",
    "Batch",
    "make_queue",
    # Tasks
    "make_task",
    "set_task",
    "Task",
    "HierarchicalTask",
    "RegressionTask",
    "make_regression_metric",
    "BenchmarkDataset",
]
