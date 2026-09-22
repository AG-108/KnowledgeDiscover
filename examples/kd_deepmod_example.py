# General imports
import os

from _common import bootstrap_project_root

project_root = bootstrap_project_root()

import matplotlib.pyplot as plt
import numpy as np
import torch

# DeepMoD functions
from deepymod import DeepMoD
from deepymod.model.constraint import LeastSquares
from deepymod.model.func_approx import NN
from deepymod.model.library import Library1D
from deepymod.model.sparse_estimators import Threshold
from torch.utils.data import DataLoader, TensorDataset

from kd.dataset import load_kdv_equation

# Settings for reproducibility
np.random.seed(42)
torch.manual_seed(0)

data = load_kdv_equation()
u = np.asarray(getattr(data, "u", data.usol), dtype=np.float32)
x = np.asarray(data.x, dtype=np.float32)
t = np.asarray(data.t, dtype=np.float32)

# Reshape data to (n_samples, 2) coordinates and (n_samples, 1) targets.
X, T = np.meshgrid(x, t, indexing="ij")
X_star = np.column_stack([X.ravel(), T.ravel()])
u_star = u.reshape(-1, 1)
print(f"Data shape: X_star={X_star.shape}, u_star={u_star.shape}")

loader = DataLoader(
    TensorDataset(
        torch.tensor(X_star.astype(np.float32)),
        torch.tensor(u_star.astype(np.float32)),
    ),
    batch_size=min(256, len(X_star)),
    shuffle=True,
)

# Quick mode toggle to keep example fast during CI / demos
QUICK = True
EPOCHS = 3 if QUICK else 20

# Define a compact DeepMoD model.
if QUICK:
    network = NN(2, [20, 20, 1], 1)
else:
    network = NN(2, [50, 50, 50, 50, 1], 1)

library = Library1D(poly_order=2, diff_order=3)
constraint = LeastSquares()
sparse_estimator = Threshold(threshold=0.01)
model = DeepMoD(network, library, sparse_estimator, constraint)
optimizer = torch.optim.Adam(model.parameters(), betas=(0.99, 0.99), amsgrad=True, lr=1e-3)

# Train with the current DeepMoD API.
for epoch in range(EPOCHS):
    running_loss = 0.0
    count = 0
    for xb, yb in loader:
        prediction, time_derivs, thetas = model(xb)
        coeff_vectors = model.constraint_coeffs(scaled=False, sparse=True)
        reg_terms = [
            torch.mean((dt - theta @ coeff_vector) ** 2)
            for dt, theta, coeff_vector in zip(time_derivs, thetas, coeff_vectors)
        ]
        reg = torch.stack(reg_terms).mean()
        mse = torch.mean((prediction - yb) ** 2)
        loss = mse + reg
        running_loss += float(loss.item())
        count += 1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"epoch {epoch:02d}: loss={running_loss / count:.6e}")

coeff_vectors = model.constraint_coeffs(scaled=False, sparse=True)
active_terms = []
for idx, coeff_vector in enumerate(coeff_vectors):
    values = coeff_vector.detach().numpy().ravel()
    for j, value in enumerate(values):
        if abs(value) > 1e-3:
            active_terms.append(f"term_{idx}_{j}={value:.4g}")
print("\nActive coefficients:")
print(active_terms)

# Predict on the full grid.
with torch.no_grad():
    pred = model.func_approx(torch.tensor(X_star.astype(np.float32)))[0].numpy()
u_pred = pred.reshape(u.shape)

# Plot the results.
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].pcolormesh(x, t, u.T, shading="auto")
axes[0].set_title("True Solution")
axes[0].set_xlabel("x")
axes[0].set_ylabel("t")
axes[0].figure.colorbar(axes[0].collections[0], ax=axes[0])

axes[1].pcolormesh(x, t, u_pred.T, shading="auto")
axes[1].set_title("Predicted Solution")
axes[1].set_xlabel("x")
axes[1].set_ylabel("t")
axes[1].figure.colorbar(axes[1].collections[0], ax=axes[1])

out_fig = os.path.join(project_root, "results", "deepmod_demo.png")
os.makedirs(os.path.dirname(out_fig), exist_ok=True)
plt.tight_layout()
plt.savefig(out_fig, bbox_inches="tight")
plt.close()
print(f"Saved deepmod demo figure to: {out_fig}")
