from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
from deepymod import DeepMoD
from deepymod.model.constraint import LeastSquares
from deepymod.model.func_approx import NN
from deepymod.model.library import Library1D
from deepymod.model.sparse_estimators import Threshold
from torch.utils.data import DataLoader, TensorDataset


class KD_DeepMoD:
    """Lightweight benchmark wrapper around DeepMoD.

    The upstream DeepMoD package expects a different training API than the rest of
    the project and is sensitive to some newer PyTorch/TensorBoard behaviors. This
    wrapper keeps the benchmark contract simple: accept coordinate/target arrays or
    a GridPDEDataset and train for a small number of iterations before reporting a
    symbolic approximation.
    """

    def __init__(
        self,
        hidden_layers: Sequence[int] = (20, 20, 1),
        poly_order: int = 2,
        diff_order: int = 3,
        threshold: float = 0.01,
        batch_size: int = 256,
        epochs: int = 5,
        learning_rate: float = 1e-3,
        seed: int = 0,
        device: str = "cpu",
    ):
        self.hidden_layers = tuple(hidden_layers)
        self.poly_order = poly_order
        self.diff_order = diff_order
        self.threshold = threshold
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.seed = seed
        self.device = torch.device(device)
        self.model_ = None
        self.best_equation_ = ""

    def _to_coordinates_and_targets(self, dataset: Any):
        if hasattr(dataset, "get_data"):
            data = dataset.get_data()
            x = np.asarray(data["x"], dtype=np.float32).ravel()
            t = np.asarray(data["t"], dtype=np.float32).ravel()
            usol = np.real(np.asarray(data["usol"], dtype=np.float32))
            if usol.ndim == 3:
                usol = usol[0]
            if usol.shape != (len(x), len(t)):
                usol = usol.T
            X, T = np.meshgrid(x, t, indexing="ij")
            coords = np.column_stack([X.ravel(), T.ravel()])
            target = usol.T.ravel()[:, None]
            return coords, target

        raise TypeError("dataset must provide get_data() for DeepMoD.")

    def fit_dataset(self, dataset: Any):
        X, y = self._to_coordinates_and_targets(dataset)
        return self.fit(X, y)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] < 2:
            raise ValueError("X must be a coordinate array of shape (n_samples, 2)")
        if y.ndim == 1:
            y = y[:, None]

        torch.manual_seed(self.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)

        network = NN(X.shape[1], list(self.hidden_layers), 1)
        library = Library1D(poly_order=self.poly_order, diff_order=self.diff_order)
        estimator = Threshold(threshold=self.threshold)
        constraint = LeastSquares()
        model = DeepMoD(network, library, estimator, constraint).to(self.device)

        tensor_x = torch.from_numpy(X)
        tensor_y = torch.from_numpy(y)
        loader = DataLoader(
            TensorDataset(tensor_x, tensor_y),
            batch_size=min(self.batch_size, len(X)),
            shuffle=True,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        for _ in range(self.epochs):
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                prediction, time_derivs, thetas = model(xb)
                coeff_vectors = model.constraint_coeffs(scaled=False, sparse=True)
                reg_terms = [
                    torch.mean((dt - theta @ coeff_vector) ** 2)
                    for dt, theta, coeff_vector in zip(time_derivs, thetas, coeff_vectors)
                ]
                reg = torch.stack(reg_terms).mean()
                mse = torch.mean((prediction - yb) ** 2)
                loss = mse + reg
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.model_ = model
        self.best_equation_ = self._equation_summary(model)
        return self

    def _equation_summary(self, model: DeepMoD) -> str:
        coeffs = model.constraint_coeffs(scaled=False, sparse=True)
        derivative_names = ["1", *[f"u_{'x' * order}" for order in range(1, self.diff_order + 1)]]
        feature_names = []
        for power in range(self.poly_order + 1):
            polynomial = "1" if power == 0 else ("u" if power == 1 else f"u^{power}")
            for derivative in derivative_names:
                if polynomial == "1":
                    feature_names.append(derivative)
                elif derivative == "1":
                    feature_names.append(polynomial)
                else:
                    feature_names.append(f"{polynomial}*{derivative}")
        equations, coefficient_rows = [], []
        for output, coeff_vector in enumerate(coeffs):
            values = coeff_vector.detach().cpu().numpy().ravel()
            if len(values) != len(feature_names):
                raise RuntimeError(
                    "DeepMoD coefficient vector does not match the configured library size"
                )
            coefficient_rows.append(values)
            terms = [
                f"({float(value):.8g})*{feature}"
                for feature, value in zip(feature_names, values)
                if np.abs(value) > 1e-6
            ]
            lhs = "u_t" if len(coeffs) == 1 else f"u{output}_t"
            equations.append(f"{lhs} = " + (" + ".join(terms) or "0"))
        self.feature_names_ = feature_names
        self.coefficients_ = coefficient_rows
        return "; ".join(equations)


__all__ = ["KD_DeepMoD"]
