import numpy as np
import pysindy as ps

from kd.dataset import GridPDEDataset

dataset = GridPDEDataset

# dataset.usol: (C, Nx, Ny, Nt)
U = dataset.usol
x = np.asarray(dataset.x, dtype=float)   # (Nx,)
y = np.asarray(dataset.y, dtype=float)   # (Ny,)
t = np.asarray(dataset.t, dtype=float)   # (Nt,)

C, Nx, Ny, Nt = U.shape

# 选一个通道做示例
u = U[0]  # (Nx, Ny, Nt)

dx = x[1] - x[0]
dy = y[1] - y[0]
dt = t[1] - t[0]

# （可选但强烈建议）对噪声做轻度平滑
# ps.SmoothedFiniteDifference / SavitzkyGolay 等都可用，weak form 对噪声更稳，但别完全不管噪声

# 这里用 weak form PDE library（名称依版本而不同）
# 常见参数含义：
# - derivative_order: 空间导数最高阶（例如2或3）
# - poly_order: 非线性多项式阶（例如2表示含 u^2）
# - include_interaction: 是否包含混合项（例如 u*u_x 之类）
# - grid / spacings: 网格信息
# - K / H / test function settings: 测试函数的数量/尺度（弱形式关键超参）
weak_lib = ps.WeakPDELibrary(
    derivative_order=3,
    poly_order=3,
    include_interaction=True,
    is_uniform=True,
    spatial_grid=(x, y),
    temporal_grid=t,
    # 下面这些参数名字各版本略不同，你需要按你安装版本对齐：
    # K=..., Hx=..., Hy=...,  (测试函数数量/支持域宽度)
)

optimizer = ps.STLSQ(threshold=1e-3, alpha=1e-6, normalize=True)

model = ps.SINDy(
    feature_library=weak_lib,
    optimizer=optimizer,
    feature_names=["u"],   # 单通道
)

model.fit(u_txyz, t=dt)
model.print()
