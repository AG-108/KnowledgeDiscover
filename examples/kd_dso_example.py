"""Minimal example: DSO baseline on a symbolic regression benchmark.

DSO trains an LSTM policy to emit pre-order traversals of expression trees.
Each iteration samples a batch of expressions, scores them against the data,
keeps the top-`epsilon` quantile by reward, and updates the policy with a
risk-seeking policy gradient -- optimizing best-case rather than average-case
reward, which is what matters when only the single best expression is kept.

This is a TensorFlow 1.x -> PyTorch port of the original DSO release; see
kd/model/dso/ for the ported package and kd/model/kd_dso.py for the wrapper
(the original repo is vendored at kd/dataset/DeepSymbolicOptimization/).

Two runs below:

1. A synthetic `y = 2x + 1`, which DSO recovers exactly in a few iterations --
   a fast confirmation that the search works end to end.
2. Keijzer-2 (`0.3*x1*sin(2*pi*x1)`) from the standard benchmark suite, run on
   a deliberately small budget to keep this example quick (~2-3 min on CPU).
   The published results use ~2,000,000 samples; at the budget used here DSO
   will fit the data only partially and is NOT expected to recover the true
   expression. Note `const` in the function set -- Keijzer-2's 0.3 and 2*pi
   coefficients are unreachable without optimizable constants.
"""

from _common import bootstrap_project_root

project_root = bootstrap_project_root()

import numpy as np

from kd.dataset import SymbolicRegressionDataset
from kd.model.kd_dso import KD_DSO

# Run a synthetic problem that DSO should solve exactly.
print("=" * 68)
print("Run 1: synthetic  y = 2x + 1")
print("=" * 68)

X = np.linspace(-1.0, 1.0, 40).reshape(-1, 1)
y = 2.0 * X[:, 0] + 1.0

model = KD_DSO(
    n_samples=2000,
    batch_size=100,
    function_set=["add", "sub", "mul", "div"],
    max_length=16,
    learning_rate=0.005,
    threshold=1e-10,
    seed=0,
    device="cpu",
    verbose=False,
)
model.fit(X, y)

print(f"discovered : {model.best_expression_}")
print(f"reward     : {model.best_reward_:.6f}")
print(f"train R^2  : {model.score(X, y):.6f}")
print(f"iterations : {model.n_iterations_}  (early-stopped: {model.success_})")
print()


# Run the Keijzer-2 benchmark with a small search budget.
print("=" * 68)
print("Run 2: Keijzer-2 benchmark   0.3*x1*sin(2*pi*x1)")
print("=" * 68)

dataset = SymbolicRegressionDataset(name="Keijzer-2")

model2 = KD_DSO(
    n_samples=6000,
    batch_size=300,
    # "const" adds optimizable numeric constants -- required for Keijzer-2's
    # coefficients, at the cost of a scipy optimization per candidate.
    function_set=["add", "sub", "mul", "div", "sin", "cos", "const"],
    max_length=24,
    learning_rate=0.005,
    entropy_weight=0.03,
    epsilon=0.05,
    baseline="R_e",
    seed=0,
    device="cpu",
    verbose=False,
)
model2.fit_dataset(dataset)

print(f"discovered : {model2.best_expression_}")
print(f"reward     : {model2.best_reward_:.6f}")
print(f"test MSE   : {model2.test_mse_:.6g}")
print(f"iterations : {model2.n_iterations_}")
print()
print("Pareto front (accuracy vs. complexity):")
for entry in model2.get_pareto_front()[:5]:
    print(
        f"  R={entry['reward']:.4f}  complexity={entry['complexity']:.0f}"
        f"  {entry['expression']}"
    )
