import numpy as np
from kd.dataset import load_wake_equation
from kd.model.kd_pdenet import PDENetModel

DATA_DIR = "../kd/dataset/WDwake"

dataset = load_wake_equation(DATA_DIR, ["TI8_U.npy", "TI8_V.npy"])

model = PDENetModel(
    derivative_order=3,
    threshold=5,
    alpha=1e-5,
    max_iter=500,
    device="cuda:0",
    dtype="double",
    steps=1,
    batch_size=8,
)

model.fit(dataset)
model.print_model()

U0 = dataset.usol[:, :, :, 50]          # (C, Nx, Ny)
dt = dataset.t[51] - dataset.t[50]
U1_pred = model.predict(U0, dt)         # (C, Nx, Ny)

U1_true = dataset.usol[:, :, :, 51]
mse = np.mean((U1_pred - U1_true) ** 2)
print(f"One-step forecast MSE: {mse:.6e}")