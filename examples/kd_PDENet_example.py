import os.path

import numpy as np
from kd.dataset import GridPDEDataset
from kd.model.kd_pdenet import PDENetModel

def load_wake_equation(data_path) -> GridPDEDataset:
    U = np.load("../kd/dataset/WDwake/TI8_U.npy")
    V = np.load("../kd/dataset/WDwake/TI8_V.npy")
    # for now, we crop the data to a square domain
    N = min(U.shape[1], U.shape[2])
    U = U[:, :N, :N]
    V = V[:, :N, :N]
    x = np.linspace(-5, 5, U.shape[1])
    y = np.linspace(-5, 5, U.shape[2])
    t = np.linspace(0, 20, U.shape[0])
    coords = {"x": x, "y": y, "t": t}
    usol = np.stack([U, V], axis=0).transpose([0, 2, 3, 1])  # shape: (2, nx, ny, nt)
    return GridPDEDataset(
        equation_name="Wake",
        pde_data={"coords": coords, "usol": usol},
        domain={"x": (x.min(), x.max()), "y": (y.min(), y.max()), "t": (t.min(), t.max())},
        epi=0.0
    )

DATA_DIR = "../kd/dataset"
DATA_PATH = os.path.join(DATA_DIR, "WDwake/TI8_U.npy")

dataset = load_wake_equation(DATA_PATH)

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