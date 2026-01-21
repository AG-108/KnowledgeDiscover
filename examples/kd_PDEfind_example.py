import numpy as np
from kd.model.kd_pdefind import PDEFindModel
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

model = PDEFindModel(
    derivative_order=3,               # 包含到二阶导，适配扩散
    threshold=5,
    alpha=1e-5,
    max_iter=500
)

# 模型训练（根据 PDEDataset 提供的 x,t,usol 拟合 PDE）
model.fit(dataset)

# 打印识别到的 PDE（u_t = ...）
model.print_model()

# 预测
U0 = u[:, 50]            # 取 t=50 对应的空间场
dt = t[51] - t[50]             # 邻近时间步长
U1_pred = model.predict(U0, dt)

# sanity check
U1_true = u[:, 51]
mse = np.mean((U1_pred - U1_true)**2)
print(f"One-step forecast MSE: {mse:.6e}")
