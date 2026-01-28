# - 使用 PySINDy 实现 PDE-FIND 的 Algorithm 1（STRidge/STLSQ 稀疏回归）
# - 面向对象封装一个 PDEFindModel 类
#
# 说明：
# - 这里通过 GridPDEDataset.get_data() 获取 x, t, usol，并将其转换为 PySINDy 期望的格式
# - 对于 PDE-FIND，我们拟合 u_t = Θ(u, u_x, u_xx, ...) ξ，因此不做“点值预测”，而是打印识别出的 PDE 结构
# - predict 方法这里提供一个“时间外推”的简单接口：在已知 u(x, t0) 的情况下，用识别到的 PDE 做一步前向欧拉更新得到 u(x, t0+Δt)（示意）
#   注意：这只是示例用途，真实场景请用更稳定的时间积分（如 RK4），或用 PySINDy 的 simulate 等方法
# 目前这个实现只能支持处理1维数据
# TODO：支持多维数据
# 理论上来说散点也能做，但是很麻烦，后面再处理

from typing import Any, Dict, Optional
import numpy as np
import pysindy as ps
from pysindy.differentiation import FiniteDifference

class PDEFindModel:
    """
    使用 PySINDy 的 PDELibrary + STLSQ（STRidge风格）来进行 PDE-FIND（Algorithm 1）的面向对象封装。
    接口：
      - fit(dataset): 从 GridPDEDataset 中读取 x, t, usol，构造候选库并拟合
      - print_model(): 打印识别到的 PDE
      - predict(U0, dt): 给定某一时刻的空间场 U0(x)，做一次前向欧拉时间推进
    :param derivative_order: 包含到几阶空间导数（默认二阶：支持扩散）
    :param function_library: 候选函数库（默认二次多项式库）
    :param differentiation_method: 差分方法（默认中心差分）
    :param threshold: 稀疏阈值
    :param alpha: 岭回归系数
    :param max_iter: STLSQ 最大迭代次数
    """

    def __init__(
        self,
        derivative_order: int = 2,
        function_library: Optional[ps.PDELibrary] = None,
        differentiation_method: ps.differentiation.BaseDifferentiation = FiniteDifference,
        threshold: float = 0.05,
        alpha: float = 1e-5,
        max_iter: int = 50,
    ):
        self.derivative_order = derivative_order
        self.function_library = function_library or ps.PolynomialLibrary(degree=2, include_bias=False)

        self.threshold = threshold
        self.alpha = alpha
        self.max_iter = max_iter

        # 这些在拟合后赋值
        self._model: Optional[ps.SINDy] = None
        self._x: Optional[np.ndarray] = None
        self._t: Optional[np.ndarray] = None
        self._U: Optional[np.ndarray] = None
        self._spatial_grid: Optional[list] = None
        self._diff_method = differentiation_method

    def fit(self, dataset: Any) -> None:
        """
        从 GridPDEDataset 中读取数据并拟合 PDE。
        要求 dataset.get_data() 返回包含 'x', 't', 'usol' 的字典。
        """
        data: Dict[str, Any] = dataset.get_data()
        x = np.asarray(data["x"], dtype=float).flatten()
        t = np.asarray(data["t"], dtype=float).flatten()

        usol = np.real(np.asarray(data["usol"]))
        if usol.shape != (len(x), len(t)):
            raise ValueError(f"usol shape {usol.shape} != ({len(x)}, {len(t)})")

        U = usol.reshape(len(x), len(t), 1)

        # 保存内部状态
        self._x = x
        self._t = t
        self._U = U
        self._spatial_grid = x

        # 构造 PDE 候选库
        pde_lib = ps.PDELibrary(
            function_library=self.function_library,
            derivative_order=self.derivative_order,
            spatial_grid=self._spatial_grid,
            is_uniform=True,
            differentiation_method=self._diff_method
        )

        # 配置稀疏回归器（STRidge风格）
        optimizer = ps.STLSQ(
            alpha=self.alpha,
            threshold=self.threshold,
            max_iter=self.max_iter,
            normalize_columns=True
        )

        # 组装并拟合
        model = ps.SINDy(
            feature_library=pde_lib,
            optimizer=optimizer
        )
        model.fit(U, t=t)

        # 保存模型
        self._model = model

    def print_model(self) -> None:
        """打印识别到的 PDE（u_t = ...）。"""
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        self._model.print()

    def coefficients(self) -> np.ndarray:
        """返回识别到的系数矩阵。"""
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self._model.coefficients()

    def predict(self, U0: np.ndarray, dt: float) -> np.ndarray:
        """
        做一次简单的时间外推：U(t+dt) ≈ U(t) + dt * f(U, U_x, U_xx, ...)
        说明：
        - 这里用已识别的库和系数，按当前快照 U0 计算右端（u_t），然后做一次前向欧拉。
        - 这是示例性质，真实应用建议用更稳定的积分器或 self._model.simulate。
        输入：
        - U0: 形状 (nx,) 的空间场（对应 self._x）
        - dt: 时间步长
        返回：
        - U1: 形状 (nx,) 的外推结果
        """
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        if self._x is None:
            raise RuntimeError("No spatial grids stored.")
        U0 = np.asarray(U0, dtype=float).flatten()
        nx = len(self._x)
        if U0.shape != (nx,):
            raise ValueError(f"U0 shape {U0.shape} != ({nx},)")

        # 用相同的差分器在空间上计算导数
        # PySINDy 的 PDELibrary 在内部会构造 Θ，但我们这里演示一次性计算 u_t
        # 简化做法：利用已训练模型的 right-hand side 函数估计 u_t
        # 直接调用 model.predict 会给出时间导数估计（对于 SINDyPDE，predict(U) -> dU/dt）
        Ut = np.asarray(self._model.predict(U0.reshape(-1, 1)).ravel())  # 形状 (nx,)

        # 一步前向欧拉
        U1 = U0 + dt * Ut
        return U1
