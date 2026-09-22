from _common import bootstrap_project_root

bootstrap_project_root()

import numpy as np

from kd.dataset import load_dataset
from kd.model.kd_sindy import SINDyModel

dataset = load_dataset("ball_drop")
trajectory = dataset.trajectories[0]
X = trajectory["state"].T
y = np.gradient(trajectory["state"][1], trajectory["t"], edge_order=2)

model = SINDyModel(polynomial_degree=2, threshold=0.05)
model.fit(X, y, variable_names=dataset.state_vars)

print(f"d(v)/dt = {model.best_expression_}")
print(f"Training MSE: {np.mean((model.predict(X) - y) ** 2):.6e}")
