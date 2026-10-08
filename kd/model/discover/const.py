"""Constant optimizers for DISCOVER's symbolic search."""

from functools import partial

import numpy as np
from scipy.optimize import minimize


def make_const_optimizer(name, **kwargs):
    """Return the configured constant optimizer."""
    optimizers = {None: Dummy, "dummy": Dummy, "scipy": ScipyMinimize}
    return optimizers[name](**kwargs)


class ConstOptimizer:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, f, x0):
        """Minimize the negative base reward from the initial constants."""
        raise NotImplementedError


class Dummy(ConstOptimizer):
    def __call__(self, f, x0):
        return x0


class ScipyMinimize(ConstOptimizer):
    def __call__(self, f, x0):
        with np.errstate(divide="ignore"):
            result = partial(minimize, method="BFGS")(f, x0)
        return result["x"]
