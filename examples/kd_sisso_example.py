"""Small SISSO++ fit; requires separately installed native bindings."""

from _common import bootstrap_project_root

bootstrap_project_root()

import numpy as np

from kd.model import SISSOModel

x = np.linspace(-2, 2, 32)
X = np.column_stack((x, np.cos(x)))
y = 1.25 + 2 * x
model = SISSOModel(max_rung=0)
model.fit(X, y)
print(model.best_expression_)
print(f"Training RMSE: {np.sqrt(np.mean((model.predict(X) - y) ** 2)):.3g}")
