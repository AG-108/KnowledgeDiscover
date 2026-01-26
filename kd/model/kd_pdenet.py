# PDENetModel.py
from __future__ import annotations

import time, os, sys
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np
import torch

_THIS_FILE = os.path.abspath(__file__)
_THIS_DIR = os.path.dirname(_THIS_FILE)

sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, os.path.join(_THIS_DIR, "PDENet2"))

from aTEAM.optim import NumpyFunctionInterface
from scipy.optimize import fmin_bfgs as bfgs

import conf
import setenv
import initparameters

ArrayLike = Union[np.ndarray, torch.Tensor]


def _is_uniform_grid(x: np.ndarray, rtol=1e-5, atol=1e-8) -> bool:
    if x.ndim != 1 or x.size < 3:
        return True
    dx = np.diff(x)
    return np.allclose(dx, dx.mean(), rtol=rtol, atol=atol)


def _infer_dx_dt_from_dataset(x: np.ndarray, t: np.ndarray) -> Tuple[float, float]:
    if not _is_uniform_grid(x):
        raise ValueError("当前封装假设 x 是均匀网格；否则需要你自定义 dx 处理逻辑。")
    if not _is_uniform_grid(t):
        raise ValueError("当前封装假设 t 是均匀时间步；否则需要你自定义 dt 处理逻辑。")
    dx = float(x[1] - x[0]) if x.size > 1 else 1.0
    dt = float(t[1] - t[0]) if t.size > 1 else 1.0
    return dx, dt


def _infer_eps_from_x(x: np.ndarray) -> float:
    # PDE-Net 原脚本里 eps 常用 2*pi；这里用区间长度近似
    if x.size < 2:
        return 1.0
    dx = float(x[1] - x[0])
    return float((x[-1] - x[0]) + dx)


@dataclass
class FitReport:
    name: str
    blocks: Sequence[int]
    seconds: float


class PDENetModel:
    """
    PDE-Net 2.0 的工程化封装
    - 默认做 1D 单通道（u(x,t)）；
    - 多通道（u,v,...）或 2D 空间网格需要在 _make_u_obs_from_dataset 里扩展。
    """

    def __init__(
            self,
            derivative_order: int = 3,
            threshold: int = 5,
            alpha: float = 1e-5,
            max_iter: int = 500,
            *,
            device: str = "cpu",
            dtype: str = "double",
            kernel_size: int = 5,
            hidden_layers: int = 3,
            scheme: str = "upwind",
            constraint: str = "frozen",
            stablize: float = 0.0,
            sparsity: Optional[float] = None,
            momentsparsity: float = 1e-3,
            blocks: str = "0-6,9,12,15,18",
            name: str = "PDENetModel",
            recordfile: str = "None",
            recordcycle: int = 200,
            savecycle: int = -1,
            npseed: int = -1,
            torchseed: int = -1,
            viscosity: float = 0.0,
            noise: float = 0.0,
            data_start_time: float = 0.0,
    ):
        self.derivative_order = int(derivative_order)
        self.threshold = int(threshold)
        self.alpha = float(alpha)
        self.max_iter = int(max_iter)

        self.device = device
        self.dtype = dtype

        self.kernel_size = int(kernel_size)
        self.hidden_layers = int(hidden_layers)
        self.scheme = scheme
        self.constraint = constraint

        self.stablize = float(stablize)
        # 你的 alpha 更像一个“正则强度”；这里映射成 sparsity（可按你的需求改）
        self.sparsity = float(sparsity) if sparsity is not None else float(alpha)
        self.momentsparsity = float(momentsparsity)

        self.blocks = blocks
        self.name = name
        self.recordfile = recordfile
        self.recordcycle = int(recordcycle)
        self.savecycle = int(savecycle)

        self.npseed = int(npseed)
        self.torchseed = int(torchseed)

        self.viscosity = float(viscosity)
        self.noise = float(noise)
        self.data_start_time = float(data_start_time)

        # 训练后会填充
        self._options = None
        self._globalnames = None
        self._callback = None
        self._model = None
        self._data_model = None
        self._sampling = None
        self._addnoise = None

        self._fitted = False

    # ---------------------------
    # Public API
    # ---------------------------
    def fit(self, dataset) -> FitReport:
        """
        dataset: 你的 PDEDataset 实例，要求提供 x,t,usol 属性，并且 usol shape = (len(x), len(t)).
        """
        t0 = time.time()

        x = np.asarray(dataset.x, dtype=float).flatten()
        t = np.asarray(dataset.t, dtype=float).flatten()
        u = np.asarray(dataset.usol)
        if u.shape != (x.size, t.size):
            raise ValueError(f"dataset.usol shape={u.shape} 与 (len(x),len(t))={(x.size, t.size)} 不一致。")

        dx, dt = _infer_dx_dt_from_dataset(x, t)
        eps = _infer_eps_from_x(x)

        options = self._build_options(dx=dx, dt=dt, eps=eps)
        self._options = options

        globalnames, callback, model, data_model, sampling, addnoise = setenv.setenv(options)

        self._globalnames = globalnames
        self._callback = callback
        self._model = model
        self._data_model = data_model
        self._sampling = sampling
        self._addnoise = addnoise

        # seeds
        torchseed = globalnames.get("torchseed", self.torchseed)
        npseed = globalnames.get("npseed", self.npseed)
        if torchseed is not None and int(torchseed) >= 0:
            torch.cuda.manual_seed_all(int(torchseed))
            torch.manual_seed(int(torchseed))
        if npseed is not None and int(npseed) >= 0:
            np.random.seed(int(npseed))

        # init params（和 train.py 一致）
        initparameters.initkernels(model, scheme=self.scheme)
        initparameters.initexpr(model, viscosity=self.viscosity, pattern="random")

        # blocks（和 train.py 一致）
        blocks = globalnames["blocks"]
        device = globalnames["device"]

        # === 关键：把 dataset 变成 u_obs/u_true/u 供 loss 使用 ===
        # 这里我们对每个 block 复用同一批观测（你也可以按 block 设计不同时间窗）
        # u_obs 的 batch 维度使用 options['batch_size']（下面默认 1）
        for block in blocks:
            print("block:", block, "name:", self.name)

            # stage / frozen
            if block == 0:
                callback.stage = "warmup"
                isfrozen = (False if self.constraint == "free" else True)
            else:
                callback.stage = "block-" + str(block)
                isfrozen = (self.constraint == "frozen")

            stepnum = block if block >= 1 else 1
            layerweight = [1] * stepnum

            u_obs, u_true, u_clean = self._make_u_obs_from_dataset(
                model=model,
                u=u,
                device=device,
                dtype=globalnames["dtype"],
                stepnum=stepnum,
            )

            # NumpyFunctionInterface + forward（照搬 train.py 的结构）
            def forward():
                stableloss, dataloss, sparseloss, momentloss = setenv.loss(
                    model, u_obs, globalnames, block, layerweight
                )
                if block == 0:
                    stableloss = 0
                    sparseloss = 0
                    momentloss = 0
                if self.constraint == "frozen":
                    momentloss = 0

                loss = (
                        self.stablize * stableloss
                        + dataloss
                        + stepnum * self.sparsity * sparseloss
                        + stepnum * self.momentsparsity * momentloss
                )
                if torch.isnan(loss):
                    loss = (torch.ones(1, requires_grad=True) / torch.zeros(1)).to(loss)
                return loss

            nfi = NumpyFunctionInterface(
                [
                    dict(
                        params=model.diff_params(),
                        isfrozen=isfrozen,
                        x_proj=model.diff_x_proj,
                        grad_proj=model.diff_grad_proj,
                    ),
                    dict(params=model.expr_params(), isfrozen=False),
                ],
                forward=forward,
                always_refresh=False,
            )
            callback.nfi = nfi

            def callbackhook(_callback, *args):
                stableloss, dataloss, sparseloss, momentloss = setenv.loss(
                    model, u_obs, globalnames, block, layerweight
                )
                stableloss = float(stableloss.item())
                dataloss = float(dataloss.item())
                sparseloss = float(sparseloss.item())
                momentloss = float(momentloss.item())
                with _callback.open() as output:
                    print(
                        f"stableloss: {stableloss:.2e}  dataloss: {dataloss:.2e}  "
                        f"sparseloss: {sparseloss:.2e}  momentloss: {momentloss:.2e}",
                        file=output,
                    )
                return None

            hook_handle = callback.register_hook(callbackhook)

            if block == 0:
                callback.save(nfi.flat_param, "start")

            try:
                xopt = bfgs(
                    nfi.f,
                    nfi.flat_param,
                    nfi.fprime,
                    gtol=2e-16,
                    maxiter=self.max_iter,
                    callback=callback,
                )
            except RuntimeError as e:
                with callback.open() as output:
                    print(e, file=output)
                xopt = nfi.flat_param
            finally:
                nfi.flat_param = xopt
                callback.save(xopt, "final")
                callback.record(xopt, callback.ITERNUM)
                hook_handle.remove()

                # 每个阶段打印当前表达式（照搬 train.py）
                with callback.open() as output:
                    print("current expression:", file=output)
                    for poly in model.polys:
                        tsym, csym = poly.coeffs()
                        print(tsym[:20], file=output)
                        print(csym[:20], file=output)

        self._fitted = True
        return FitReport(name=self.name, blocks=list(blocks), seconds=time.time() - t0)

    def print_model(self, topk: Optional[int] = None) -> None:
        if not self._fitted:
            raise RuntimeError("请先调用 fit(dataset) 再 print_model().")

        model = self._model
        topk = self.threshold if topk is None else int(topk)

        print("\n=== PDE-Net 2.0 identified model (top terms) ===")
        for i, poly in enumerate(model.polys):
            tsym, csym = poly.coeffs()
            tsym = list(tsym)
            csym = np.asarray(csym)

            # 按系数绝对值排序
            idx = np.argsort(-np.abs(csym))
            idx = idx[: min(topk, len(idx))]

            print(f"\n[Poly {i}]")
            for j in idx:
                print(f"  {tsym[j]}  :  {csym[j]: .6e}")

    def predict(self, U0: ArrayLike, dt: float) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("请先调用 fit(dataset) 再 predict().")

        model = self._model
        device = self._globalnames["device"]
        dtype = self._globalnames["dtype"]

        if isinstance(U0, np.ndarray):
            u0 = torch.from_numpy(np.asarray(U0))
        else:
            u0 = U0

        # 期望形状：batch x channel x Nx x Ny（对 1D 可用 Ny=1）
        if u0.ndim == 1:
            u0 = u0[None, None, :, None]
        elif u0.ndim == 2:
            # 假设 (Nx, C) 或 (C, Nx) 都有人用，这里更保守：当成 (Nx, ) 的扩展不做
            u0 = u0[None, None, :, :]
        elif u0.ndim == 3:
            u0 = u0[None, ...]  # -> batch
        elif u0.ndim != 4:
            raise ValueError(f"U0 维度不支持：{u0.shape}，期望 1D/3D/4D 可转成 (B,C,Nx,Ny).")

        u0 = u0.to(device=device, dtype=dtype)

        with torch.no_grad():
            u1 = model(u0, T=float(dt))

        # 返回 numpy，去掉 batch/channel 维
        u1_np = u1.detach().cpu().numpy()
        # 形状假设 (1,1,Nx,1) -> (Nx,)
        if u1_np.ndim == 4 and u1_np.shape[0] == 1 and u1_np.shape[1] == 1:
            u1_np = u1_np[0, 0, :, 0]
        return u1_np

    # ---------------------------
    # Internals
    # ---------------------------
    def _build_options(self, *, dx: float, dt: float, eps: float) -> Any:
        """
        用 conf.setoptions 构造 options。
        这里尽量沿用 train.py 的字段命名，减少 setenv 侧的改动。
        """
        # derivative_order: 你写“包含到二阶导”，而 PDE-Net 代码里是 max_order=2 表示到二阶空间导
        # 所以这里采用：max_order = derivative_order - 1（更贴近你的注释习惯）
        max_order = max(1, int(self.derivative_order) - 1)

        kw = {
            "--name": self.name,
            "--dtype": self.dtype,
            "--device": self.device,
            "--constraint": self.constraint,

            # computing region
            "--eps": float(eps),
            "--dt": float(dt),
            "--cell_num": 1,
            "--blocks": self.blocks,

            # super parameters of network
            "--kernel_size": int(self.kernel_size),
            "--max_order": int(max_order),
            "--dx": float(dx),
            "--hidden_layers": int(self.hidden_layers),
            "--scheme": self.scheme,

            # data generator (我们不使用 setenv.data 来生成，但 setenv.setenv 仍需要 dataname 等字段)
            "--dataname": "burgers",  # 仅用于走通 setenv.setenv；你的外部数据会覆盖真正的 u_obs
            "--viscosity": float(self.viscosity),
            "--zoom": 1,
            "--max_dt": float(dt),
            "--batch_size": 1,
            "--data_timescheme": "rk2",
            "--channel_names": "u",
            "--freq": 1,
            "--data_start_time": float(self.data_start_time),

            # data transform
            "--start_noise": float(self.noise),
            "--end_noise": float(self.noise),

            # others
            "--stablize": float(self.stablize),
            "--sparsity": float(self.sparsity),
            "--momentsparsity": float(self.momentsparsity),
            "--npseed": int(self.npseed),
            "--torchseed": int(self.torchseed),
            "--maxiter": int(self.max_iter),
            "--recordfile": self.recordfile,
            "--recordcycle": int(self.recordcycle),
            "--savecycle": int(self.savecycle),
            "--start_from": -1,
        }

        # 模拟 train.py 的行为：argv 为空，用 kw 注入
        options = conf.setoptions(argv=[], kw=kw, configfile=None)
        return options

    def _make_u_obs_from_dataset(
            self,
            *,
            model,
            u: np.ndarray,
            device: torch.device,
            dtype: torch.dtype,
            stepnum: int,
    ):
        """
        这是“外部数据”到 PDE-Net 内部 loss 所需格式的适配器。

        经验上 setenv.loss(...) 需要的 u_obs 结构与 setenv.data(...) 产出一致：
        - u_obs 是一个 list/tuple，其中 u_obs[0] 是观测序列张量
        - 张量一般为: (batch, channel, Nx, Ny) 或包含时间展开的额外维
        由于不同 dataname/实现可能略有差异，这里给一个最常见、最小可用的版本。

        当前实现策略（单步监督）：
        - 构造两帧：u(t0) -> u(t0+dt)，并复制为 batch=1
        - 若你的 loss 需要更长序列（由 stepnum 驱动），就在这里扩展更多时间帧。
        """
        # u: (Nx, Nt)
        Nx, Nt = u.shape
        if Nt < 2:
            raise ValueError("dataset.usol 至少需要 2 个时间点才能做一步监督。")

        # 取一个训练窗口：t_idx = 0..stepnum
        # 为了简单，保证至少两帧
        end = min(Nt - 1, max(1, stepnum))
        t0 = 0
        t1 = end

        u0 = u[:, t0]
        u1 = u[:, t1]

        # 变成 torch: (B=1, C=1, Nx, Ny=1)
        u0_t = torch.as_tensor(u0, dtype=dtype, device=device)[None, None, :, None]
        u1_t = torch.as_tensor(u1, dtype=dtype, device=device)[None, None, :, None]

        # 下面这三者结构要尽量贴近 setenv.data 的返回
        # 一个常见做法是：u_obs[0] 里包含“观测到的初值序列/目标序列”
        # 这里用 tuple 表达（你如果发现 loss 期望 list，就改成 list）
        u_obs = (u0_t, u1_t)
        u_true = (u0_t, u1_t)
        u_clean = (u0_t, u1_t)

        # 某些实现会对 u_obs[0] 做 model.UInputs(...)，所以 u_obs[0] 必须是 tensor
        # 如果你跑起来报错（比如 setenv.loss 期望 u_obs[0] 是序列），就从这里开始改
        return u_obs, u_true, u_clean
