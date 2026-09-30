"""DSO discovery using observed integral responses, with no kernel labels."""

import numpy as np

from ..functions import create_tokens
from ..library import Library, Token
from .task import HierarchicalTask


def _less(x, y):
    return (x < y).astype(float)


class GreenTask(HierarchicalTask):
    task_type = "green"
    stochastic = False

    def __init__(self, dataset, function_set, threshold=1e-10, protected=False, **unused):
        super().__init__()
        self.name = "green_observed_integral"
        self.X = np.asarray(dataset["kernel_inputs"], float)
        self.shape = tuple(dataset["kernel_shape"])
        self.WF = np.asarray(dataset["weighted_forcing"], float)
        self.response = np.asarray(dataset["response"], float)
        self.VF = np.asarray(dataset["validation_forcing"], float)
        self.validation = np.asarray(dataset["validation_response"], float)
        self.threshold = threshold
        self.scale = max(float(np.sqrt(np.mean(self.response**2))), 1e-15)
        tokens = create_tokens(2, [f for f in function_set if f != "lt"], protected=protected)
        if "lt" in function_set:
            tokens.append(Token(_less, "lt", arity=2, complexity=2))
        self.library = Library(tokens)
        self.invalid_reward = 0.0
        self.max_reward = 1.0

    def _prediction(self, program, weighted_forcing):
        with np.errstate(all="ignore"):
            values = np.asarray(program.execute(self.X), float)
            values = np.broadcast_to(values, (len(self.X),)).reshape(self.shape)
            if program.invalid or not np.isfinite(values).all():
                raise ValueError("Invalid kernel")
            prediction = values @ weighted_forcing
        if not np.isfinite(prediction).all():
            raise ValueError("Invalid forward response")
        return prediction

    def reward_function(self, program, optimizing=False):
        try:
            error = (
                np.sqrt(np.mean((self._prediction(program, self.WF) - self.response) ** 2))
                / self.scale
            )
        except (ValueError, FloatingPointError, OverflowError):
            return -1e12 if optimizing else self.invalid_reward
        return -float(error) if optimizing else 1 / (1 + float(error))

    def evaluate(self, program):
        # Only internal validation controls success and early stopping.
        try:
            error = float(
                np.mean((self._prediction(program, self.VF) - self.validation) ** 2) / self.scale**2
            )
        except (ValueError, FloatingPointError, OverflowError):
            error = None
        return {
            "nmse_test": error,
            "nmse_test_noiseless": error,
            "success": error is not None and error < self.threshold,
            "data_role": "internal_validation_forcing; never heldout_test",
        }
