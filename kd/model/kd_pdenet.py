# PDENetModel.py
from __future__ import annotations
import os, sys

_THIS_FILE = os.path.abspath(__file__)
_THIS_DIR = os.path.dirname(_THIS_FILE)

sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, os.path.join(_THIS_DIR, "PDENet2"))

import numpy as np
import torch

from aTEAM.optim import NumpyFunctionInterface
from scipy.optimize import fmin_bfgs as bfgs

import conf
import setenv
import initparameters


class PDENetModel:
    """
    Wrapper for PDE-Net-2.0 (POLYPDE2D) trained from a user-provided dataset.

    Expected dataset:
      - dataset.usol: np.ndarray, shape (C, Nx, Ny, Nt)
      - dataset.t:    np.ndarray, shape (Nt,)
      - dataset.x:    np.ndarray, shape (Nx,)  (optional but recommended)
      - dataset.y:    np.ndarray, shape (Ny,)  (optional but recommended)
        OR dataset.domain provides bounds.

    Important constraint from original repo:
      - setenv assumes a square 2D grid and single spacing '--dx' for both dims.
        Practically: Nx == Ny and dx == dy (uniform grids).
    """

    def __init__(
        self,
        derivative_order: int = 3,
        threshold: int = 10,
        alpha: float = 1e-5,
        max_iter: int = 500,
        device: str = "cpu",
        dtype: str = "double",
        kernel_size: int = 5,
        hidden_layers: int = 3,
        scheme: str = "upwind",
        constraint: str = "free",
        stablize: float = 0.0,
        momentsparsity: float = 0.0,
        batch_size: int = 16,
        steps: int = 1,
        seed: int = 0,
        channel_names: str | None = None,
    ):
        self.derivative_order = int(derivative_order)
        self.threshold = int(threshold)
        self.alpha = float(alpha)
        self.max_iter = int(max_iter)

        self.device = device
        self.dtype_name = dtype  # 'float' or 'double'
        self.torch_dtype = torch.float64 if dtype == "double" else torch.float32

        self.kernel_size = int(kernel_size)
        self.hidden_layers = int(hidden_layers)
        self.scheme = str(scheme)
        self.constraint = constraint
        self.stablize = float(stablize)
        self.momentsparsity = float(momentsparsity)

        self.batch_size = int(batch_size)
        self.steps = int(steps)
        self.seed = int(seed)
        self.channel_names = channel_names  # e.g. "u,v"

        self.options = None
        self.globalnames = None
        self.callback = None
        self.model = None
        self._trained = False

    # -------------------------
    # Public API
    # -------------------------
    def fit(self, dataset):
        usol = np.asarray(dataset.usol)
        if usol.ndim != 4:
            raise ValueError(f"Expected dataset.usol shape (C, Nx, Ny, Nt), got {usol.shape}")

        C, Nx, Ny, Nt = usol.shape
        if Nt < (self.steps + 1):
            raise ValueError(f"Need Nt >= steps+1 = {self.steps+1}, got Nt={Nt}")

        # time step
        t = np.asarray(dataset.t, dtype=float).flatten()
        if t.shape[0] != Nt:
            raise ValueError(f"dataset.t length {t.shape[0]} != Nt {Nt}")
        dt_arr = np.diff(t)
        if not np.all(dt_arr > 0):
            raise ValueError("t must be strictly increasing.")
        dt = float(np.median(dt_arr))

        # spatial bounds + dx (repo expects single dx and square mesh)
        x = getattr(dataset, "x", None)
        y = getattr(dataset, "y", None)
        domain = getattr(dataset, "domain", None)

        x_min, x_max, y_min, y_max = self._infer_bounds_xy(x, y, domain)
        x_len = x_max - x_min
        y_len = y_max - y_min
        if x_len <= 0 or y_len <= 0:
            raise ValueError("Invalid spatial bounds.")

        # 这里是为了防止不适应非方阵加的检测
        # if Nx != Ny:
        #     raise ValueError(
        #         f"Original PDE-Net-2.0 setenv assumes square mesh; got Nx={Nx}, Ny={Ny}. "
        #         "You need to modify setenv/polypde to support rectangular grids."
        #     )

        dx_x = x_len / Nx
        # dx_y = y_len / Ny
        # if not np.isclose(dx_x, dx_y, rtol=1e-3, atol=1e-12):
        #     raise ValueError(
        #         f"Original setenv uses a single --dx; need dx==dy. Got dx_x={dx_x}, dx_y={dx_y}. "
        #         "You need to refactor setenv/polypde for anisotropic spacing."
        #     )

        dx = float(dx_x)

        # choose eps so that bound = eps*cell_num equals the "domain size"
        # setenv uses: bound = eps*cell_num, mesh_bound = [[0,0],[bound,bound]]
        # We'll map your [x_min,x_max] and [y_min,y_max] to [0,bound] by assuming periodic/shifted domain.
        # So we use bound = x_len (== y_len).
        bound = float(x_len)

        # channel_names: default "u,v,w,..."
        channel_names = self.channel_names
        if channel_names is None:
            base = "u v w p q r s".split()
            if C <= len(base):
                channel_names = ",".join(base[:C])
            else:
                channel_names = ",".join([f"u{i}" for i in range(C)])

        kw = {
            "--name": "PDENetModel",
            "--dtype": self.dtype_name,
            "--device": self.device,
            "--constraint": self.constraint,

            # region / discretization
            "--eps": bound,
            "--dt": dt,
            "--cell_num": 1,
            "--blocks": "1",

            # network hyperparams
            "--kernel_size": self.kernel_size,
            "--max_order": self.derivative_order,
            "--dx": dx,
            "--hidden_layers": self.hidden_layers,
            "--scheme": self.scheme,

            # required by schema (not used for your dataset-driven training)
            "--dataname": "burgers",
            "--viscosity": 0.0,
            "--zoom": 1,
            "--max_dt": dt,
            "--batch_size": self.batch_size,
            "--data_timescheme": "rk2",
            "--channel_names": channel_names,
            "--freq": 1,
            "--data_start_time": 0.0,

            # noise (off)
            "--start_noise": 0.0,
            "--end_noise": 0.0,

            # regularization weights
            "--stablize": self.stablize,
            "--sparsity": self.alpha,
            "--momentsparsity": self.momentsparsity,

            "--npseed": self.seed,
            "--torchseed": self.seed,
            "--maxiter": self.max_iter,

            "--recordfile": "None",
            "--recordcycle": 200,
            "--savecycle": -1,
            "--start_from": -1,
        }

        options = conf.setoptions(argv=[], kw=kw, configfile=None)
        self.options = options

        globalnames, callback, model, data_model, sampling, addnoise = setenv.setenv(options)
        self.globalnames = globalnames
        self.callback = callback
        self.model = model

        self.callback.stage = "fit"

        torch.cuda.manual_seed_all(globalnames["torchseed"])
        torch.manual_seed(globalnames["torchseed"])
        np.random.seed(globalnames["npseed"])

        initparameters.initkernels(self.model, scheme=self.scheme)
        initparameters.initexpr(self.model, viscosity=0.0, pattern="random")

        # Build u_obs from dataset: list length steps+1, each (B, C, Nx, Ny)
        u_obs = self._make_u_obs_from_dataset(usol, steps=self.steps, batch_size=self.batch_size)

        layerweight = [1] * max(1, self.steps)

        def forward():
            stableloss, dataloss, sparseloss, momentloss = setenv.loss(
                self.model, u_obs, self.globalnames, block=self.steps, layerweight=layerweight
            )
            loss = (
                self.stablize * stableloss
                + dataloss
                + self.steps * self.alpha * sparseloss
                + self.steps * self.momentsparsity * momentloss
            )
            if torch.isnan(loss):
                loss = (torch.ones(1, requires_grad=True) / torch.zeros(1)).to(loss)
            return loss

        isfrozen = (False if self.constraint == "free" else True)

        nfi = NumpyFunctionInterface(
            [
                dict(
                    params=self.model.diff_params(),
                    isfrozen=isfrozen,
                    x_proj=self.model.diff_x_proj,
                    grad_proj=self.model.diff_grad_proj,
                ),
                dict(params=self.model.expr_params(), isfrozen=False),
            ],
            forward=forward,
            always_refresh=False,
        )
        self.callback.nfi = nfi

        def callbackhook(_callback, *args):
            stableloss, dataloss, sparseloss, momentloss = setenv.loss(
                self.model, u_obs, self.globalnames, block=self.steps, layerweight=layerweight
            )
            with _callback.open() as output:
                print(
                    f"stableloss: {stableloss.detach().item():.2e}  "
                    f"dataloss: {dataloss.detach().item():.2e}  "
                    f"sparseloss: {sparseloss.detach().item():.2e}  "
                    f"momentloss: {momentloss.detach().item():.2e}",
                    file=output,
                )
            return None

        hook_handle = self.callback.register_hook(callbackhook)

        xopt = None
        try:
            with self.callback.open() as output:
                print("device:", self.device, file=output)
                print(f"C={C}, Nx={Nx}, Ny={Ny}, Nt={Nt}, dt={dt}, dx={dx}", file=output)
            xopt = bfgs(
                nfi.f,
                nfi.flat_param,
                nfi.fprime,
                gtol=2e-16,
                maxiter=self.max_iter,
                callback=self.callback,
            )
        finally:
            self.callback.stage = "fit"
            if xopt is not None:
                nfi.flat_param = xopt
                self.callback.save(xopt, "final")
                self.callback.record(xopt, self.callback.ITERNUM)
            hook_handle.remove()

        self._trained = True
        return self

    def predict(self, U0: np.ndarray, dt: float) -> np.ndarray:
        """
        One-step forecast.
        U0: shape (C, Nx, Ny)
        returns: shape (C, Nx, Ny)
        """
        self._ensure_ready()
        self.model.eval()

        U0 = np.asarray(U0, dtype=float)
        if U0.ndim != 3:
            raise ValueError(f"Expected U0 shape (C, Nx, Ny), got {U0.shape}")

        u0_t = torch.as_tensor(U0, dtype=self.torch_dtype, device=self.device).unsqueeze(0)  # (1,C,Nx,Ny)

        # 关键：forward 内部用 autograd.grad，所以这里必须开启梯度
        with torch.enable_grad():
            u0_t.requires_grad_(True)
            u1_t = self.model(u0_t, T=float(dt))

        return u1_t.detach().cpu().numpy()[0]

    def print_model(self):
        """
        Print top-k discovered terms for each output channel polynomial.
        """
        self._ensure_ready()

        for i, poly in enumerate(self.model.polys):
            tsym, csym = poly.coeffs()
            cs = np.asarray(csym, dtype=float)
            idx = np.argsort(-np.abs(cs))
            k = min(self.threshold, len(idx))

            print(f"[Poly #{i}] top-{k} terms:")
            for j in idx[:k]:
                print(f"  {cs[j]:+.6e} * {tsym[j]}")

    # -------------------------
    # Internals
    # -------------------------
    def _ensure_ready(self):
        if (self.model is None) or (not self._trained):
            raise RuntimeError("Model is not trained yet. Call model.fit(dataset) first.")

    def _make_u_obs_from_dataset(self, usol: np.ndarray, steps: int, batch_size: int):
        """
        usol: (C, Nx, Ny, Nt)
        returns u_obs list of tensors, each (B, C, Nx, Ny)
        """
        C, Nx, Ny, Nt = usol.shape
        max_start = Nt - (steps + 1)
        if max_start < 0:
            raise ValueError("Not enough time steps for requested 'steps'.")

        starts = np.random.randint(0, max_start + 1, size=batch_size)

        u_obs = []
        for k in range(steps + 1):
            # gather frames at time (start+k) for each batch item
            # result: (B, C, Nx, Ny)
            uk = np.stack([usol[:, :, :, s + k] for s in starts], axis=0)
            uk_t = torch.as_tensor(uk, dtype=self.torch_dtype, device=self.device)
            u_obs.append(uk_t)
        return u_obs

    def _infer_bounds_xy(self, x, y, domain):
        """
        Return x_min,x_max,y_min,y_max.
        Prefer dataset.x/y; fallback to dataset.domain.
        """
        if (x is not None) and (y is not None):
            x = np.asarray(x, dtype=float).flatten()
            y = np.asarray(y, dtype=float).flatten()
            return float(x.min()), float(x.max()), float(y.min()), float(y.max())

        if isinstance(domain, dict) and ("x" in domain) and ("y" in domain):
            x_min, x_max = domain["x"]
            y_min, y_max = domain["y"]
            return float(x_min), float(x_max), float(y_min), float(y_max)

        raise ValueError("Need dataset.x & dataset.y, or dataset.domain with keys 'x' and 'y'.")
