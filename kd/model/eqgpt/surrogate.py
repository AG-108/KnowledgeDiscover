"""
Surrogate neural network (fits noisy/sparse u(x,t) data so that derivatives
can later be obtained via autograd) for EqGPT.

Adapted from `kd/dataset/EqGPT/code/neural_network.py` and the
`train_surrogate_model`/`Generate_meta_data` logic in
`kd/dataset/EqGPT/code/surrogate_model.py` (original EqGPT repo).

Only the "canonical PDE" (regular (x,t) grid, 1 output component) path is
ported -- the original script also handles irregular 2D/3D domains
(Laplacian_smile/EITech/shuttle, Burgers_2D, etc.), which are out of scope
here. Model checkpointing to disk is dropped in favor of keeping the
best-validation-loss state dict in memory, since this is meant to run
end-to-end within a single `fit()` call rather than as a resumable script.
"""

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)


class Rational(nn.Module):
    """Rational-function activation (Boullé, Nakatsukasa & Townsend, 2020)."""

    def __init__(self, dtype=torch.float32, device=torch.device("cpu")):
        super().__init__()
        self.a = nn.Parameter(torch.tensor((1.1915, 1.5957, 0.5, 0.0218), dtype=dtype, device=device))
        self.b = nn.Parameter(torch.tensor((2.3830, 0.0, 1.0), dtype=dtype, device=device))

    def forward(self, x):
        a, b = self.a, self.b
        n_x = a[0] + x * (a[1] + x * (a[2] + a[3] * x))
        d_x = b[0] + x * (b[1] + b[2] * x)
        return n_x / d_x


class NN(nn.Module):
    """Fully-connected surrogate network mapping (x, t) -> u."""

    def __init__(self,
                 num_hidden_layers: int = 5,
                 neurons_per_layer: int = 50,
                 input_dim: int = 2,
                 output_dim: int = 1,
                 dtype: torch.dtype = torch.float32,
                 device: torch.device = torch.device("cpu"),
                 activation: str = "Sin",
                 batch_norm: bool = False):
        super().__init__()
        assert num_hidden_layers > 0 and neurons_per_layer > 0
        assert input_dim > 0 and output_dim > 0

        self.num_hidden_layers = num_hidden_layers
        self.batch_norm = batch_norm

        self.layers = nn.ModuleList()
        if batch_norm:
            self.norm_layer = nn.BatchNorm1d(num_features=input_dim, dtype=dtype, device=device)

        self.layers.append(nn.Linear(input_dim, neurons_per_layer, bias=True).to(dtype=dtype, device=device))
        for _ in range(1, num_hidden_layers):
            self.layers.append(nn.Linear(neurons_per_layer, neurons_per_layer, bias=True).to(dtype=dtype, device=device))
        self.layers.append(nn.Linear(neurons_per_layer, output_dim, bias=True).to(dtype=dtype, device=device))

        if activation in ("Tanh", "Rational"):
            gain = 5.0 / 3.0 if activation == "Tanh" else 1.41
            for i in range(num_hidden_layers + 1):
                nn.init.xavier_normal_(self.layers[i].weight, gain=gain)
                nn.init.zeros_(self.layers[i].bias)
        elif activation == "Sin":
            a = 3.0 / math.sqrt(neurons_per_layer)
            for i in range(num_hidden_layers + 1):
                nn.init.uniform_(self.layers[i].weight, -a, a)
                nn.init.zeros_(self.layers[i].bias)
        else:
            raise ValueError(f"Unknown activation function: {activation!r}")

        self.activations = nn.ModuleList()
        for _ in range(num_hidden_layers):
            if activation == "Tanh":
                self.activations.append(nn.Tanh())
            elif activation == "Sin":
                self.activations.append(Sin())
            elif activation == "Rational":
                self.activations.append(Rational(dtype=dtype, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.batch_norm:
            x = self.norm_layer(x)
        for i in range(self.num_hidden_layers):
            x = self.activations[i](self.layers[i](x))
        return self.layers[self.num_hidden_layers](x)


def random_data(x: np.ndarray, t: np.ndarray, un: np.ndarray, choose: int,
                choose_validate: int, seed: int = 525
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Randomly split a regular (x, t) grid of noisy solution values `un`
    (shape (n_x, n_t)) into a training subset and a held-out validation
    subset, both as flattened (n, 2) coordinate / (n, 1) value tensors.
    """
    x_num, t_num = x.shape[0], t.shape[0]
    total = x_num * t_num

    un_raw = torch.from_numpy(un.astype(np.float32))
    database = torch.zeros([total, 2])
    h_data = torch.zeros([total, 1])
    num = 0
    for j in range(x_num):
        for i in range(t_num):
            database[num, 0] = x[j]
            database[num, 1] = t[i]
            h_data[num] = un_raw[j, i]
            num += 1

    rng = np.random.default_rng(seed)
    order = rng.permutation(total)
    train_idx = order[:choose]
    val_idx = order[choose:choose + choose_validate]

    return (h_data[train_idx], h_data[val_idx], database[train_idx], database[val_idx])


def train_surrogate_model(net: NN, database_choose: torch.Tensor, h_data_choose: torch.Tensor,
                          database_validate: torch.Tensor, h_data_validate: torch.Tensor,
                          device: torch.device, n_iter: int = 20000, eval_every: int = 500,
                          verbose: bool = False) -> NN:
    """
    Fit `net` on (database_choose, h_data_choose) by plain MSE regression,
    tracking the held-out validation loss every `eval_every` iterations and
    restoring the best-validation-loss weights at the end (in memory, no
    checkpoint files).
    """
    net = net.to(device)
    database_choose = database_choose.to(device).requires_grad_(True)
    database_validate = database_validate.to(device).requires_grad_(True)
    h_data_choose = h_data_choose.to(device)
    h_data_validate = h_data_validate.to(device)

    optimizer = torch.optim.Adam(net.parameters())
    mse_loss = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None

    for it in range(n_iter):
        optimizer.zero_grad()
        prediction = net(database_choose)
        loss = mse_loss(h_data_choose, prediction)
        loss.backward()
        optimizer.step()

        if (it + 1) % eval_every == 0:
            with torch.no_grad():
                val_pred = net(database_validate)
                val_loss = torch.mean((h_data_validate - val_pred) ** 2).item()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
            if verbose:
                print(f"iter {it + 1}: train_loss={loss.item():.6g} val_loss={val_loss:.6g}")

    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    return net


def generate_meta_grid(x_low: float, x_up: float, t_low: float, t_up: float,
                       nx: int = 100, nt: int = 100, device: Optional[torch.device] = None
                       ) -> torch.Tensor:
    """Dense (x, t) query grid (with grad tracking) for computing meta-data derivatives."""
    x = torch.linspace(x_low, x_up, nx)
    t = torch.linspace(t_low, t_up, nt)
    database = torch.zeros([nx * nt, 2])
    num = 0
    for j in range(nx):
        for i in range(nt):
            database[num, 0] = x[j]
            database[num, 1] = t[i]
            num += 1
    if device is not None:
        database = database.to(device)
    return database.requires_grad_(True)
