"""Adapted from archive/kd/model/DeepSymbolicOptimization/dso/dso/core.py
(original: https://github.com/dso-org/deep-symbolic-optimization).

Changes vs. the original
------------------------
- No TensorFlow session: `tf.reset_default_graph()`, `tf.Session(config=...)`
  and the single-thread `ConfigProto` are gone. `setup()` just constructs the
  task, prior, state manager, policy and trainer in order.
- `make_policy` / `make_policy_optimizer` collapse into one step -- PG and PQT
  live on the policy in this port (see policy.py).
- Dropped `make_output_file`/`save_config`/`Checkpoint`/`StatsLogger` (the
  file-based logging and checkpoint system was not ported) and the GP-meld
  controller. Results are returned in memory instead.
- Seeding uses `torch.manual_seed` in place of `tf.set_random_seed`.
"""

import os
import random
import zlib
from copy import deepcopy

import commentjson as json
import numpy as np
import torch

from .program import Program
from .prior import make_prior
from .state_manager import make_state_manager
from .policy import RNNPolicy
from .task import set_task
from .train import Trainer


_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "config", "config_regression.json")


def load_default_config():
    """Load the merged default regression config shipped with this package."""
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def _merge(base, update):
    """Recursively merge `update` into a copy of `base`."""
    out = deepcopy(base)
    for k, v in (update or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


class DeepSymbolicOptimizer:
    """
    Deep symbolic optimization model. Holds hyperparameters and training
    configuration.

    Parameters
    ----------
    config : dict or str or None
        Config dict, or path to a JSON(C) config file. Merged over the packaged
        defaults. If None, the defaults are used as-is.
    """

    def __init__(self, config=None):
        self.config = None
        self.set_config(config)
        self.trainer = None
        self.policy = None
        self.pool = None

    # ------------------------------------------------------------------ config

    def set_config(self, config):
        base = load_default_config()
        if config is None:
            merged = base
        elif isinstance(config, str):
            with open(config, encoding="utf-8") as f:
                merged = _merge(base, json.load(f))
        else:
            merged = _merge(base, config)

        self.config = merged
        self.config_task = merged["task"]
        self.config_training = merged["training"]
        self.config_state_manager = merged["state_manager"]
        self.config_policy = merged["policy"]
        self.config_policy_optimizer = merged["policy_optimizer"]
        self.config_prior = merged["prior"]

    def set_seeds(self):
        """Set random seeds, offset by a checksum of the task name so that
        different tasks with the same seed don't share draws (matching the
        original's behaviour)."""
        seed = self.config_training.get("seed", 0)
        task_name = Program.task.name if Program.task is not None else ""
        shifted = (seed + zlib.adler32(str(task_name).encode("utf-8"))) % (2 ** 31)
        random.seed(shifted)
        np.random.seed(shifted)
        torch.manual_seed(shifted)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(shifted)

    # ------------------------------------------------------------------- setup

    def setup(self, device=None):
        # Clear the Program cache and class-level state from any previous run
        Program.clear_cache()

        self.device = device if device is not None else torch.device("cpu")

        # Task must be set first: it builds the Library that everything else
        # (prior, state manager, policy) sizes itself against.
        self.pool = self.make_pool_and_set_task()
        self.set_seeds()  # must follow set_task

        self.prior = self.make_prior()
        self.state_manager = self.make_state_manager()
        self.policy = self.make_policy()
        self.trainer = self.make_trainer()

    def make_pool_and_set_task(self):
        n_cores_batch = self.config_training.get("n_cores_batch", 1)

        # Set complexity and constant optimizer, then the task itself
        Program.set_complexity(self.config_training.get("complexity", "token"))
        const_optimizer = self.config_training.get("const_optimizer", "scipy")
        const_params = self.config_training.get("const_params") or {}
        Program.set_const_optimizer(const_optimizer, **const_params)

        set_task(self.config_task)

        pool = None
        if n_cores_batch is not None and n_cores_batch > 1:
            from multiprocessing import Pool, cpu_count
            if n_cores_batch > cpu_count():
                n_cores_batch = cpu_count()
            pool = Pool(n_cores_batch)
        return pool

    def make_prior(self):
        return make_prior(Program.library, self.config_prior)

    def make_state_manager(self):
        return make_state_manager(self.config_state_manager)

    def make_policy(self):
        cfg_p = dict(self.config_policy)
        cfg_o = dict(self.config_policy_optimizer)

        policy_type = cfg_p.pop("policy_type", "rnn")
        if policy_type != "rnn":
            raise NotImplementedError(
                f"policy_type='{policy_type}' is not supported; only 'rnn' was ported."
            )

        opt_type = cfg_o.get("policy_optimizer_type", "pg")
        if opt_type not in ("pg", "pqt"):
            raise NotImplementedError(
                f"policy_optimizer_type='{opt_type}' is not supported; only "
                "'pg' and 'pqt' were ported (PPO was not)."
            )

        return RNNPolicy(self.prior, self.state_manager,
                         debug=self.config_training.get("debug", 0),
                         device=self.device,
                         **cfg_p, **cfg_o)

    def make_trainer(self):
        cfg = dict(self.config_training)
        # Keys consumed elsewhere (task/policy construction) or not applicable.
        for k in ("const_optimizer", "const_params", "seed", "n_cores_batch"):
            cfg.pop(k, None)
        return Trainer(self.policy, pool=self.pool, **cfg)

    # ---------------------------------------------------------------- training

    def train_one_step(self, override=None):
        """Train one iteration. Returns a summary dict when training completes,
        else None."""
        if self.trainer is None:
            self.setup()
        assert not self.trainer.done, "Training has already completed!"
        self.trainer.run_one_step(override)
        if self.trainer.done:
            return self._result()
        return None

    def train(self):
        """Train until the sample budget is exhausted or early stopping fires."""
        if self.trainer is None:
            self.setup()
        while not self.trainer.done:
            self.trainer.run_one_step()
        return self._result()

    def _result(self):
        t = self.trainer
        p = t.p_r_best
        return {
            "program": p,
            "expression": repr(p.sympy_expr) if p is not None else "",
            "r": float(t.r_best),
            "iterations": t.iteration,
            "nevals": t.nevals,
            "hall_of_fame": t.hall_of_fame,
            "pareto_front": t.pareto_front() if t.save_pareto_front else [],
            "history": t.history,
        }
