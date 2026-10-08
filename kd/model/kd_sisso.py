"""SISSO++ regression bridge using the official, optional Python bindings."""

from __future__ import annotations

import re

import numpy as np


class SISSOModel:
    """Fit one dimensionless response with SISSO's highest requested dimension.

    Descriptor selection uses training data only. Native files remain in the
    target's working directory; the benchmark supervisor enforces hard limits.
    """

    def __init__(
        self,
        n_dim=1,
        max_rung=1,
        n_sis_select=1,
        n_residual=1,
        allowed_ops=("add", "sub", "mult", "div", "sin", "cos", "exp", "log", "sq", "cb"),
        fix_intercept=False,
        random_state=0,
        workdir="sisso",
        verbose=False,
        model_selection="highest_dimension",
        selection_tolerance=1e-10,
    ):
        for name, value, minimum in (("n_dim", n_dim, 1), ("max_rung", max_rung, 0),
                                     ("n_sis_select", n_sis_select, 1),
                                     ("n_residual", n_residual, 1)):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        self.n_dim = n_dim
        self.max_rung = max_rung
        self.n_sis_select = n_sis_select
        self.n_residual = n_residual
        self.allowed_ops = list(allowed_ops)
        self.fix_intercept = bool(fix_intercept)
        self.random_state = random_state
        self.workdir = str(workdir)
        self.verbose = bool(verbose)
        if model_selection not in {"highest_dimension", "training_tolerance"}:
            raise ValueError("Unknown SISSO model_selection")
        if not np.isfinite(selection_tolerance) or selection_tolerance < 0:
            raise ValueError("selection_tolerance must be finite and nonnegative")
        self.model_selection = model_selection
        self.selection_tolerance = selection_tolerance

    @staticmethod
    def check_available():
        try:
            from sissopp.sklearn import SISSORegressor
        except (ImportError, OSError) as exc:
            raise ImportError(
                "SISSO++ Python bindings and their native libraries are required; see docs/sisso.md"
            ) from exc
        return SISSORegressor

    def fit(self, X, y, variable_names=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y[:, 0]
        if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
            raise ValueError("X must be a two-dimensional array aligned with a single response y")
        if len(X) < 2 or X.shape[1] < 1 or not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError("SISSO requires at least two finite samples and one feature")
        names = list(variable_names) if variable_names is not None else [
            f"x{i + 1}" for i in range(X.shape[1])
        ]
        if (len(names) != X.shape[1] or len(set(names)) != len(names)
                or any(not re.fullmatch(r"[A-Za-z_]\w*", name) for name in names)):
            raise ValueError("variable_names must be unique identifiers matching the input columns")
        regressor = self.check_available()
        self._model = regressor(
            prop_label="__kd_response", prop_unit="", n_dim=self.n_dim,
            max_rung=self.max_rung, n_sis_select=self.n_sis_select,
            n_residual=self.n_residual, n_models_store=1, allowed_ops=self.allowed_ops,
            allowed_param_ops=[], global_param_opt=False, nlopt_seed=self.random_state,
            fix_intercept=self.fix_intercept, workdir=self.workdir,
            clean_workdir=False, verbose=self.verbose,
        )
        # Native fit returns None; use the retained solver rather than its return value.
        while True:
            try:
                self._model.fit(X, y, columns=names)
                break
            except RuntimeError as exc:
                # n_sis_select is a maximum pool size. The native solver
                # rejects a smaller exhausted feature pool instead of using
                # it; cap only that explicit condition, within the same fit
                # budget. Do not retry unrelated numerical/search failures.
                match = re.search(r"did not select enough \((\d+) not (\d+)\)", str(exc))
                if match is None:
                    raise
                available, requested = map(int, match.groups())
                if not 0 < available < requested:
                    raise
                self._model.n_sis_select = available
        self.effective_n_sis_select_ = getattr(self._model, "n_sis_select", self.n_sis_select)
        native_model = self._model.solver.models[-1][0]
        if self.model_selection == "training_tolerance":
            values = {name: X[:, i] for i, name in enumerate(names)}
            candidates = []
            for dimension_models in self._model.solver.models:
                candidate = dimension_models[0]
                coefs = np.asarray(candidate.coefs, dtype=float)[0]
                features = list(candidate.feats)
                bias = 0.0 if self.fix_intercept else coefs[-1]
                prediction = sum(coef * feature.eval_many(values)
                                 for coef, feature in zip(coefs, features)) + bias
                rmse = float(np.sqrt(np.mean((prediction - y) ** 2)))
                if np.isfinite(rmse):
                    candidates.append((len(features), rmse, candidate))
            if not candidates:
                raise RuntimeError("SISSO returned no finite training prediction")
            tolerance = self.selection_tolerance * max(float(np.std(y)), np.finfo(float).eps)
            best_rmse = min(row[1] for row in candidates)
            native_model = min((row for row in candidates if row[1] <= best_rmse + tolerance),
                               key=lambda row: (row[0], row[1]))[2]
        coefficients = np.asarray(native_model.coefs, dtype=float)
        features = list(native_model.feats)
        if self.fix_intercept and coefficients.shape == (1, len(features)):
            coefficients = np.column_stack((coefficients, np.zeros(1)))
        if (coefficients.shape != (1, len(features) + 1)
                or not np.isfinite(coefficients).all()):
            raise RuntimeError("SISSO returned incompatible or non-finite single-task coefficients")
        self.coefficients_ = coefficients[0].copy()
        self._features = features
        self.variable_names_ = names
        self.feature_names_ = [feature.expr.replace("^", "**") for feature in features]
        self.best_expression_ = " + ".join(
            [f"({self.coefficients_[-1]:.17g})"]
            + [f"({coefficient:.17g})*({feature})" for coefficient, feature in
               zip(self.coefficients_[:-1], self.feature_names_)]
        )
        self.n_features_in_ = X.shape[1]
        self.selected_dimension_ = len(features)
        self.stop_reason_ = "budget_exhausted"
        self.stop_reason_source_ = "sissopp_solver_return"
        return self

    def predict(self, X):
        if not hasattr(self, "best_expression_"):
            raise RuntimeError("fit must be called before predict")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_ or not np.isfinite(X).all():
            raise ValueError("X has incompatible feature shape or non-finite values")
        values = {name: X[:, i] for i, name in enumerate(self.variable_names_)}
        descriptors = np.column_stack([feature.eval_many(values) for feature in self._features])
        # v1.2.6 model.eval_many adds coefs.back() even with fix_intercept=True.
        # Evaluate the original native descriptors and apply the fitted linear
        # coefficients explicitly so prediction and the exported equation agree.
        return descriptors @ self.coefficients_[:-1] + self.coefficients_[-1]


__all__ = ["SISSOModel"]
