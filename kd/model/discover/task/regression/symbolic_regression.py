import numpy as np

from discover.task import HierarchicalTask
from discover.library import Library
from discover.functions import create_tokens
from discover.task.pde.pde import make_pde_metric


class SymbolicRegressionTask(HierarchicalTask):
    """
    Generic symbolic regression task.

    Target:
        y ≈ Θ(X) w

    Dataset format:
        {
            "X": np.ndarray, shape [n_samples, n_features],
            "y": np.ndarray, shape [n_samples] or [n_samples, 1],
            "variable_names": list[str], optional,
            "n_input_dim": int, optional,
            "sym_true": str, optional
        }

    Notes
    -----
    This task reuses the existing Program.execute() and STRidge pipeline.
    It maps ordinary regression data into the interface expected by Program:

        u  -> [first feature]
        x  -> full feature matrix X
        ut -> target y

    Thus the PDE-style execution pipeline becomes a generic sparse regression:

        ut ≈ Θ(u, x) w
    """

    task_type = "symbolic_regression"

    def __init__(
        self,
        function_set,
        dataset=None,
        metric="inv_nrmse",
        metric_params=(0.01,),
        threshold=1e-8,
        reward_noise=0.0,
        data_noise_level=0,
        max_depth=4,
        protected=False,
        spatial_error=False,
        decision_tree_threshold_set=None,
        cut_ratio=0.03,
        add_const=False,
        eq_num=1,
        normalize_y=False,
        **kwargs
    ):
        super(HierarchicalTask, self).__init__()

        self.add_const = add_const
        self.eq_num = eq_num
        self.name = dataset if dataset is not None else "symbolic_regression"

        # These attributes are accessed by Program/STRidge.
        self.noise_level = data_noise_level
        self.spatial_error = spatial_error
        self.cut_ratio = cut_ratio
        self.max_depth = max_depth

        self.threshold = threshold

        # Reuse PDE metric factory for compatibility.
        # For metric="inv_nrmse", no metric_params are required in some versions;
        # keep your config consistent with make_pde_metric().
        self.metric, self.invalid_reward, self.max_reward = make_pde_metric(
            metric,
            *metric_params
        )

        self.function_set = function_set
        self.decision_tree_threshold_set = decision_tree_threshold_set
        self.protected = protected
        self.stochastic = reward_noise > 0.0

        self.normalize_y = normalize_y
        self.y_mean = 0.0
        self.y_std = 1.0

        self.library = None

        self.X = None
        self.y = None
        self.variable_names = None
        self.n_input_var = None

        self.u = None
        self.x = None
        self.ut = None

    def load_data(self, dataset):
        """
        Load ordinary regression data.

        Expected:
            X: shape [N, d]
            y: shape [N] or [N, 1]
        """

        X = np.asarray(dataset["X"], dtype=float)
        y = np.asarray(dataset["y"], dtype=float).reshape(-1, 1)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have same number of samples, "
                f"but got X.shape[0]={X.shape[0]} and y.shape[0]={y.shape[0]}."
            )

        self.X = X
        self.variable_names = dataset.get(
            "variable_names",
            [f"x{i + 1}" for i in range(X.shape[1])]
        )

        self.n_input_var = dataset.get("n_input_dim", X.shape[1])
        self.sym_true = dataset.get("sym_true", "")

        if self.normalize_y:
            self.y_mean = float(np.mean(y))
            self.y_std = float(np.std(y) + 1e-12)
            y_used = (y - self.y_mean) / self.y_std
        else:
            y_used = y

        self.y = y_used

        # Compatibility mapping for Program.execute(u, x, ut).
        self.u = [X[:, 0].reshape(-1, 1)]
        self.x = X
        self.ut = y_used.reshape(-1, 1)

        tokens = create_tokens(
            n_input_var=self.n_input_var,
            function_set=self.function_set,
            protected=self.protected,
            n_state_var=len(self.u),
            decision_tree_threshold_set=self.decision_tree_threshold_set,
            task_type="pde",
        )

        self.library = Library(tokens)

    def reward_function(self, p):
        """
        Reward for a candidate expression.
        """

        y_hat, y_right, w = p.execute(self.u, self.x, self.ut)

        if p.invalid or y_hat is None:
            return self.invalid_reward, [0], None, None

        r = self.metric(self.ut, y_hat, len(w))

        if not np.isfinite(r):
            r = self.invalid_reward

        return r, w, y_hat, y_right

    def mse_function(self, p):
        """
        MSE objective for constant optimization.
        """

        y_hat, _, _ = p.execute(self.u, self.x, self.ut)

        if p.invalid or y_hat is None:
            return 1e12

        loss = np.mean((y_hat - self.ut) ** 2)

        if not np.isfinite(loss):
            return 1e12

        return float(loss)

    def evaluate(self, p):
        """
        Evaluate discovered expression on the loaded data.
        """

        y_hat, y_right, w = p.execute(self.u, self.x, self.ut)

        if p.invalid or y_hat is None:
            return {
                "mse": None,
                "rmse": None,
                "nrmse": None,
                "nmse_test": None,
                "r2": None,
                "success": False,
            }

        y_true = self.ut

        mse = float(np.mean((y_true - y_hat) ** 2))
        rmse = float(np.sqrt(mse))
        nrmse = float(rmse / (np.std(y_true) + 1e-12))

        ss_res = float(np.sum((y_true - y_hat) ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2) + 1e-12)
        r2 = float(1.0 - ss_res / ss_tot)

        return {
            "mse": mse,
            "rmse": rmse,
            "nrmse": nrmse,
            "nmse_test": mse,
            "r2": r2,
            "success": mse < self.threshold,
        }

    def evaluate_diff(self, p):
        """
        Return residuals.
        """

        y_hat, _, _ = p.execute(self.u, self.x, self.ut)

        if p.invalid or y_hat is None:
            return None

        return self.ut - y_hat

    def terms_values(self, p):
        """
        Return terms and their evaluated values.
        """

        values = p.execute_terms(self.u, self.x)
        return p.STRidge.terms_token, values

    def stability_test(self, p):
        """
        Compatibility hook for Program.execute_stability_test().
        """

        return True
