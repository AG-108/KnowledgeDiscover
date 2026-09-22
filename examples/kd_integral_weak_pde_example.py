"""Bounded synthetic advection example; no external data or full benchmark run."""

import numpy as np
from kd.model import IntegralWeakPDEModel


def main():
    x, t = np.linspace(0, 2 * np.pi, 161), np.linspace(0, 2, 121)
    u = np.sin(x[:, None] - 0.7 * t[None, :])
    model = IntegralWeakPDEModel(terms=[(1, 1)], threshold=0).fit(u, x, t)
    print("u_t =", model.best_expression_)
    print("Weak-integral training MSE:", model.weak_residual_mse_)


if __name__ == "__main__":
    main()
