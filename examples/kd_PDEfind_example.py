from _common import bootstrap_project_root

project_root = bootstrap_project_root()

import numpy as np

from kd.dataset import GridPDEDataset, load_kdv_equation
from kd.model.kd_pdefind import PDEFindModel

data = load_kdv_equation()
x = data.x
t = data.t
u = data.usol

dataset = GridPDEDataset(
    equation_name="KdV",
    pde_data={"x": x, "t": t, "usol": u},
    domain={"x": (x.min(), x.max()), "t": (t.min(), t.max())},
    epi=0.0,
    legacy=True,
)

model = PDEFindModel(
    derivative_order=3,
    threshold=5,
    alpha=1e-5,
    max_iter=500,  # Include second derivatives so the library can represent diffusion.
)

# Fit the PDE from the grid coordinates and solution field.
model.fit(dataset)

# Print the discovered PDE.
model.print_model()

# Evaluate one fitted one-step prediction.
U0 = u[:, 50]  # Use the spatial field at the selected time index.
dt = t[51] - t[50]  # Advance by one neighboring time step.
U1_pred = model.predict(U0, dt)

# sanity check
U1_true = u[:, 51]
mse = np.mean((U1_pred - U1_true) ** 2)
print(f"One-step forecast MSE: {mse:.6e}")
