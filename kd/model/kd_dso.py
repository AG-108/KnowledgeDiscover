# kd/model/kd_dso.py

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from ..base import BaseEstimator
from ..metrics import MSE
from .dso import DeepSymbolicOptimizer

_DEFAULT_FUNCTION_SET = ("add", "sub", "mul", "div", "sin", "cos")


class KD_DSO(BaseEstimator):
    """
    Deep Symbolic Optimization (DSO) baseline for symbolic regression, ported
    from TensorFlow 1.x to PyTorch (see `kd/model/dso/`).

    An LSTM/GRU policy autoregressively emits pre-order traversals of expression
    trees. Each sampled batch is scored on the data, the top-`epsilon` quantile
    by reward is retained, and the policy is updated by a risk-seeking policy
    gradient -- so the objective optimizes best-case rather than average-case
    performance, which suits symbolic regression where only the single best
    expression matters.

    Petersen et al., "Deep symbolic regression: Recovering mathematical
    expressions from data via risk-seeking policy gradients", ICLR 2021.

    Follows the scikit-learn-style API used by other `kd` baselines
    (see `kd.model.kd_symbolicgpt.KD_SymbolicGPT`): configure via `__init__`,
    run via `fit`/`fit_dataset`, read results from `self.best_expression_`.
    """

    def __init__(
        self,
        # Search budget
        n_samples: int = 20000,
        batch_size: int = 500,
        # Operator library
        function_set: Sequence[str] = _DEFAULT_FUNCTION_SET,
        # Reward
        metric: str = "inv_nrmse",
        metric_params: Sequence[float] = (1.0,),
        threshold: float = 1e-12,
        protected: bool = False,
        # Policy network
        cell: str = "lstm",
        num_layers: int = 1,
        num_units: int = 32,
        initializer: str = "zeros",
        max_length: int = 64,
        learning_rate: float = 0.0005,
        entropy_weight: float = 0.03,
        entropy_gamma: float = 0.7,
        optimizer: str = "adam",
        # Risk-seeking policy gradient
        epsilon: float = 0.05,
        baseline: str = "R_e",
        alpha: float = 0.5,
        # Priority queue training
        policy_optimizer_type: str = "pg",
        pqt_k: int = 10,
        pqt_batch_size: int = 1,
        pqt_weight: float = 200.0,
        pqt_use_pg: bool = False,
        # Constant optimization
        const_optimizer: str = "scipy",
        const_params: Optional[Dict[str, Any]] = None,
        # Misc
        complexity: str = "token",
        hof: int = 100,
        early_stopping: bool = True,
        n_cores_batch: int = 1,
        seed: int = 0,
        device: Optional[str] = None,
        verbose: bool = False,
    ):
        """
        Parameters
        ----------
        n_samples, batch_size :
            Total expressions sampled over the run, and per iteration. The
            iteration count is `n_samples // batch_size`.
        function_set :
            Operators available to the search. Use "const" to add optimizable
            numeric constants (considerably slower), "poly" for a fitted
            polynomial token.
        metric, metric_params :
            Reward metric and its parameters. One of "inv_nrmse", "inv_nmse",
            "inv_mse", "neg_nrmse", "neg_nmse", "neg_mse", "neg_rmse",
            "neglog_mse", "fraction", "pearson", "spearman".
        threshold :
            NMSE below which the task reports success (triggers early stopping
            on noiseless benchmarks).
        protected :
            Use protected operators (e.g. division guarding against /0). When
            False, floating-point errors yield the invalid reward instead.
        cell, num_layers, num_units, initializer, max_length :
            Policy RNN architecture and maximum traversal length.
        learning_rate, entropy_weight, entropy_gamma, optimizer :
            Optimizer settings; `entropy_gamma` decays the entropy bonus by
            `entropy_gamma**t` across timesteps.
        epsilon, baseline, alpha :
            Risk-seeking parameters. `epsilon` is the top quantile kept for
            training; `baseline` is one of "R_e" (quantile, the default),
            "ewma_R", "ewma_R_e", "combined"; `alpha` weights the EWMA variants.
            Set `epsilon=None` or 1.0 for vanilla policy gradient.
        policy_optimizer_type, pqt_* :
            "pg" (policy gradient) or "pqt" (priority queue training). The
            `pqt_*` settings only apply to the latter. PPO was not ported.
        const_optimizer, const_params :
            Optimizer used for "const" tokens.
        complexity :
            Complexity measure for the Pareto front; does not affect the search.
        hof :
            Number of top expressions retained in `self.hall_of_fame_`.
        seed : Random seed (numpy + torch + python `random`).
        device : "cuda"/"cpu"/None (auto-detect).
        """
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.function_set = list(function_set)
        self.metric = metric
        self.metric_params = list(metric_params)
        self.threshold = threshold
        self.protected = protected
        self.cell = cell
        self.num_layers = num_layers
        self.num_units = num_units
        self.initializer = initializer
        self.max_length = max_length
        self.learning_rate = learning_rate
        self.entropy_weight = entropy_weight
        self.entropy_gamma = entropy_gamma
        self.optimizer = optimizer
        self.epsilon = epsilon
        self.baseline = baseline
        self.alpha = alpha
        self.policy_optimizer_type = policy_optimizer_type
        self.pqt_k = pqt_k
        self.pqt_batch_size = pqt_batch_size
        self.pqt_weight = pqt_weight
        self.pqt_use_pg = pqt_use_pg
        self.const_optimizer = const_optimizer
        self.const_params = const_params
        self.complexity = complexity
        self.hof = hof
        self.early_stopping = early_stopping
        self.n_cores_batch = n_cores_batch
        self.seed = seed
        self.device = device
        self.verbose = verbose

    def _resolve_device(self) -> torch.device:
        if self.device is not None:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_config(self, dataset_arg) -> Dict[str, Any]:
        return {
            "task": {
                "task_type": "regression",
                "dataset": dataset_arg,
                "function_set": self.function_set,
                "metric": self.metric,
                "metric_params": self.metric_params,
                "threshold": self.threshold,
                "protected": self.protected,
            },
            "training": {
                "n_samples": self.n_samples,
                "batch_size": self.batch_size,
                "epsilon": self.epsilon,
                "baseline": self.baseline,
                "alpha": self.alpha,
                "complexity": self.complexity,
                "const_optimizer": self.const_optimizer,
                "const_params": self.const_params or {},
                "hof": self.hof,
                "early_stopping": self.early_stopping,
                "n_cores_batch": self.n_cores_batch,
                "verbose": self.verbose,
                "seed": self.seed,
            },
            "policy": {
                "policy_type": "rnn",
                "cell": self.cell,
                "num_layers": self.num_layers,
                "num_units": self.num_units,
                "initializer": self.initializer,
                "max_length": self.max_length,
            },
            "policy_optimizer": {
                "policy_optimizer_type": self.policy_optimizer_type,
                "optimizer": self.optimizer,
                "learning_rate": self.learning_rate,
                "entropy_weight": self.entropy_weight,
                "entropy_gamma": self.entropy_gamma,
                "pqt_k": self.pqt_k,
                "pqt_batch_size": self.pqt_batch_size,
                "pqt_weight": self.pqt_weight,
                "pqt_use_pg": self.pqt_use_pg,
            },
        }

    def _run(self, dataset_arg) -> "KD_DSO":
        model = DeepSymbolicOptimizer(self._build_config(dataset_arg))
        model.setup(device=self._resolve_device())
        result = model.train()

        program = result["program"]
        self.best_program_ = program
        self.best_reward_ = result["r"]
        self.best_expression_ = result["expression"]
        self.hall_of_fame_ = result["hall_of_fame"]
        self.pareto_front_ = result["pareto_front"]
        self.history_ = result["history"]
        self.n_iterations_ = result["iterations"]
        self.nevals_ = result["nevals"]
        self.model_ = model

        if program is not None:
            self.success_ = bool(program.evaluate.get("success", False))
        else:
            self.success_ = False

        if self.verbose:
            print(f"[DSO] best expression: {self.best_expression_}")
            print(f"[DSO] reward: {self.best_reward_:.6f} | success: {self.success_}")

        return self

    def fit(self, X: Any, y: Any) -> "KD_DSO":
        """
        Discover a symbolic expression for (X, y).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.asarray(y, dtype=float).reshape(-1)
        if len(X) != len(y):
            raise ValueError(f"X has {len(X)} rows but y has {len(y)}")

        # DSO's RegressionTask accepts an (X, y) tuple directly (its Case 4
        # branch), so no intermediate file or benchmark lookup is needed.
        return self._run((X, y))

    def fit_benchmark(self, name: str) -> "KD_DSO":
        """
        Discover an expression for a named benchmark from
        `kd/model/dso/task/regression/benchmarks.csv` (e.g. "Keijzer-2",
        "Nguyen-5"). DSO generates the train/test data itself from the
        benchmark's spec, and `self.success_` then reflects exact recovery.
        """
        return self._run(name)

    def fit_dataset(self, dataset: Any) -> "KD_DSO":
        """
        Discover an expression for a `kd.dataset.SymbolicRegressionDataset`
        (e.g. `SymbolicRegressionDataset(name='Keijzer-2')`). Also scores the
        result on the dataset's held-out test split, stored as `self.test_mse_`.
        """
        data = dataset.get_data()
        self.fit(data["X_train"], data["y_train"])

        X_test = np.asarray(data.get("X_test"))
        y_test = np.asarray(data.get("y_test"))
        if X_test.size and y_test.size:
            y_pred = self.predict(X_test)
            finite = np.isfinite(y_pred)
            if finite.all():
                self.test_mse_ = MSE()(y_test.reshape(-1), y_pred)
            else:
                # A discovered expression can be undefined on part of the test
                # domain (e.g. log of a negative outside the training range).
                self.test_mse_ = float("nan")

        return self

    def predict(self, X: Any) -> np.ndarray:
        """Evaluate the discovered expression at new X."""
        if not hasattr(self, "best_program_"):
            raise RuntimeError("Call fit()/fit_dataset()/fit_benchmark() before predict().")
        if self.best_program_ is None:
            raise RuntimeError("No valid expression was found during training.")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return np.asarray(self.best_program_.execute(X), dtype=float).reshape(-1)

    def score(self, X: Any, y: Any) -> float:
        """Coefficient of determination R^2 of the discovered expression."""
        y = np.asarray(y, dtype=float).reshape(-1)
        y_pred = self.predict(X)
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        if ss_tot == 0.0:
            return float("nan")
        return 1.0 - ss_res / ss_tot

    def get_pareto_front(self) -> List[Dict[str, Any]]:
        """Pareto-optimal expressions as dicts of expression/reward/complexity."""
        if not hasattr(self, "pareto_front_"):
            raise RuntimeError("Call fit() before get_pareto_front().")
        return [
            {
                "expression": repr(p.sympy_expr),
                "reward": float(p.r),
                "complexity": float(p.complexity),
            }
            for p in self.pareto_front_
        ]
