import numpy as np
from kd.model.kd_pdenet import PDENetModel
from kd.dataset import PDEDataset, load_kdv_equation

data = load_kdv_equation()
x = data.x
t = data.t
u = data.usol

dataset = PDEDataset(
    equation_name="KdV",
    pde_data={"x": x, "t": t, "usol": u},
    domain={"x": (x.min(), x.max()), "t": (t.min(), t.max())},
    epi=0.0
)

model = PDENetModel(
    derivative_order=3,
    threshold=5,
    alpha=1e-5,
    max_iter=500,
    device="cuda:0",
    dtype="double",
)

model.fit(dataset)
model.print_model()

U0 = dataset.usol[:, 50]
dt = dataset.t[51] - dataset.t[50]
U1_pred = model.predict(U0, dt)

U1_true = dataset.usol[:, 51]
mse = np.mean((U1_pred - U1_true) ** 2)
print(f"One-step forecast MSE: {mse:.6e}")
