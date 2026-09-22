# kd/model/kd_sga.py

from typing import Any, Dict, Optional, Type

from ..base import BaseEstimator
from .sga.sgapde import visualizer as sga_visualizer
from .sga.sgapde.config import SolverConfig
from .sga.sgapde.context import ProblemContext
from .sga.sgapde.solver import SGAPDE_Solver


class KD_SGA(BaseEstimator):
    """Adapt the SGA-PDE solver to the estimator interface used by KD."""

    def __init__(
        self,
        sga_run=100,
        num=20,
        depth=4,
        width=5,
        p_var=0.5,
        p_mute=0.3,
        p_cro=0.5,
        seed=0,
        use_autograd=False,
        max_epoch=100000,
        use_metadata=False,
        delete_edges=False,
    ):
        """Initialize parameters that map directly to SolverConfig."""
        # Assign parameters explicitly so the wrapper remains easy to inspect.
        # These attributes map one-to-one to SolverConfig fields.
        self.sga_run = sga_run
        self.num = num
        self.depth = depth
        self.width = width
        self.p_var = p_var
        self.p_mute = p_mute
        self.p_cro = p_cro
        self.seed = seed
        self.use_autograd = use_autograd
        self.max_epoch = max_epoch
        self.use_metadata = use_metadata
        self.delete_edges = delete_edges

    def fit(self, problem_name: str):
        """Discover a PDE for one of the solver built-in problem names."""
        print(f"--- Starting SGA PDE Discovery for problem: {problem_name} ---")

        # Build SolverConfig from the wrapper parameters and selected problem.
        # Without external arrays, ProblemContext loads the built-in field.
        config = SolverConfig(
            problem_name=problem_name,
            sga_run=self.sga_run,
            num=self.num,
            depth=self.depth,
            width=self.width,
            p_var=self.p_var,
            p_mute=self.p_mute,
            p_cro=self.p_cro,
            seed=self.seed,
            use_autograd=self.use_autograd,
            max_epoch=self.max_epoch,
            use_metadata=self.use_metadata,
            delete_edges=self.delete_edges,
        )

        # Construct the context and preprocess the selected field.
        context = ProblemContext(config)

        # Run the configured symbolic genetic solver.
        solver = SGAPDE_Solver(config)
        best_pde, best_score = solver.run(context)

        # Store fitted attributes with scikit-learn-style trailing underscores.
        self.best_pde_ = best_pde
        self.best_score_ = best_score
        self.context_ = context  # Retain the context for visualization.
        self.config_ = config  # Retain the resolved solver configuration.

        print("\n--- SGA PDE Discovery Finished ---")
        print(f"Best PDE Found: {self.best_pde_}")
        print(f"AIC Score: {self.best_score_}")

        return self

    def fit_dataset(
        self,
        dataset: Any,
        *,
        problem_name: Optional[str] = None,
        context_cls: Optional[Type] = None,
        solver_cls: Optional[Type] = None,
    ):
        """Discover a PDE from a GridPDEDataset instance."""
        from kd.dataset import (
            GridPDEDataset,
        )  # Import lazily to avoid a module-level dependency cycle.

        from .sga.adapter import SGADataAdapter

        if not isinstance(dataset, GridPDEDataset):
            raise TypeError("dataset 必须是 GridPDEDataset 实例")

        adapter = SGADataAdapter(dataset)
        solver_kwargs: Dict[str, Any] = adapter.to_solver_kwargs()

        inferred_name = solver_kwargs.pop("problem_name", None)
        problem_label = problem_name or inferred_name or "custom_dataset"

        actual_context_cls = context_cls or ProblemContext
        actual_solver_cls = solver_cls or SGAPDE_Solver

        print(f"--- Starting SGA PDE Discovery for problem: {problem_label} (dataset mode) ---")

        config = SolverConfig(
            problem_name=problem_label,
            sga_run=self.sga_run,
            num=self.num,
            depth=self.depth,
            width=self.width,
            p_var=self.p_var,
            p_mute=self.p_mute,
            p_cro=self.p_cro,
            seed=self.seed,
            use_autograd=self.use_autograd,
            max_epoch=self.max_epoch,
            use_metadata=self.use_metadata,
            delete_edges=self.delete_edges,
            **solver_kwargs,
        )

        context = actual_context_cls(config)
        solver = actual_solver_cls(config)
        best_pde, best_score = solver.run(context)

        self.best_pde_ = best_pde
        self.best_score_ = best_score
        self.context_ = context
        self.config_ = config
        self.dataset_ = dataset

        print("\n--- SGA PDE Discovery Finished ---")
        print(f"Best PDE Found: {self.best_pde_}")
        print(f"AIC Score: {self.best_score_}")

        return self

    def plot_results(self):
        """Render diagnostics for the most recently fitted solver context."""
        if not hasattr(self, "context_"):
            raise RuntimeError("You must call fit() before plotting results.")

        print("INFO: Generating visualization plots...")
        sga_visualizer.plot_figures(self.context_, self.config_)
