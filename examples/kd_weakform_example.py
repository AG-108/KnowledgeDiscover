"""Small WSINDy-PDE advection example with no external data dependency."""

from _common import bootstrap_project_root

bootstrap_project_root()

import numpy as np

from kd.model import WSINDyPDEModel


def main():
    x = np.linspace(0, 2 * np.pi, 161)
    t = np.linspace(0, 2, 121)
    u = np.sin(x[:, None] - 0.7 * t[None, :])
    model = WSINDyPDEModel(
        terms=[(1, 1)],
        m_x=20,
        m_t=15,
        test_function_power=8,
        threshold=0,
    ).fit(u, x, t)
    print("u_t =", model.best_expression_)
    print("weak relative residual:", model.weak_relative_residual_)
    print("weak samples:", model.n_weak_samples_)


if __name__ == "__main__":
    main()
