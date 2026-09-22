"""CPU-capable, opt-in PyTorch backend for the Xu et al. PDE PIC.

The supported PDE grammar is deliberately small: scalar ``u_t`` equations whose
RHS is a list drawn from ``1, u, u_x, u_xx, u*u_x, u*u_xx, u^2, u^3``.
Spatial/time coordinates are one-dimensional and derivatives are obtained by
automatic differentiation of a common observation ANN.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Optional, Sequence, Tuple

import numpy as np

from .physics_informed import (
    PIC_IMPLEMENTATION_VERSION, PICError, PICResult, PINNTrainingResult,
    PhysicsInformedInformationCriterion, PreparedPICReference,
    fit_tls_coefficients,
)

SUPPORTED_TERMS = (
    "1", "u", "u_x", "u_xx", "u_xxx", "u*u_x", "u*u_xx", "u*u_xxx", "u^2", "u^3"
)


@dataclass(frozen=True)
class TorchPICConfig:
    hidden_width: int = 24
    hidden_layers: int = 2
    reference_epochs: int = 500
    pinn_epochs: int = 100
    learning_rate: float = 1e-3
    physics_weight: float = 1e-2
    nx: int = 32
    nt: int = 32
    n_windows: int = 10
    window_fraction: float = 0.5
    seed: int = 525
    dtype: str = "float64"
    activation: str = "tanh"
    reference_max_normalized_rmse: Optional[float] = None
    pinn_min_relative_loss_improvement: Optional[float] = None
    pinn_max_final_loss: Optional[float] = None


@dataclass
class TorchPICPrepared:
    reference: PreparedPICReference
    model_state: dict
    config: TorchPICConfig
    observation_coordinates: np.ndarray
    observation_values: np.ndarray
    evaluation_coordinates: np.ndarray
    reference_cost: dict
    reference_train_rmse: float
    reference_train_normalized_rmse: float


def _torch():
    try:
        import torch
    except ImportError as exc:  # optional dependency
        raise PICError("PyTorch is required for the PIC torch backend") from exc
    return torch


def _dtype(torch, name):
    if name == "float64":
        return torch.float64
    if name == "float32":
        return torch.float32
    raise PICError("dtype must be float32 or float64")


def _validate_config(config):
    integer_fields = ("hidden_width", "hidden_layers", "reference_epochs", "pinn_epochs",
                      "nx", "nt", "n_windows", "seed")
    for name in integer_fields:
        value = getattr(config, name)
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise PICError(f"{name} must be an integer")
    for name in integer_fields[:-1]:
        if getattr(config, name) < 1:
            raise PICError(f"{name} must be positive")
    if config.nx < 2 or config.nt < 2:
        raise PICError("nx and nt must each be at least 2")
    if config.n_windows < 2:
        raise PICError("n_windows must be at least 2")
    if not np.isfinite(config.learning_rate) or config.learning_rate <= 0:
        raise PICError("learning_rate must be finite and positive")
    if not np.isfinite(config.physics_weight) or config.physics_weight < 0:
        raise PICError("physics_weight must be finite and nonnegative")
    if not np.isfinite(config.window_fraction) or not (0 < config.window_fraction <= 1):
        raise PICError("window_fraction must be finite and in (0, 1]")
    if config.reference_max_normalized_rmse is not None and (
        not np.isfinite(config.reference_max_normalized_rmse)
        or config.reference_max_normalized_rmse <= 0
    ):
        raise PICError("reference_max_normalized_rmse must be finite and positive or null")
    if config.pinn_min_relative_loss_improvement is not None and (
        not np.isfinite(config.pinn_min_relative_loss_improvement)
        or config.pinn_min_relative_loss_improvement < 0
    ):
        raise PICError(
            "pinn_min_relative_loss_improvement must be finite and nonnegative or null"
        )
    if config.pinn_max_final_loss is not None and (
        not np.isfinite(config.pinn_max_final_loss) or config.pinn_max_final_loss <= 0
    ):
        raise PICError("pinn_max_final_loss must be finite and positive or null")
    if config.activation not in {"tanh", "sin"}:
        raise PICError("activation must be 'tanh' or 'sin'")
    # Author-style shifts need the final window to remain inside the domain.
    if config.window_fraction + (config.n_windows - 1) / (2 * config.n_windows) > 1 + 1e-12:
        raise PICError("window_fraction is too large for author-style shifted windows")
    _dtype(_torch(), config.dtype)


@contextmanager
def _isolated_rng(seed):
    """Use deterministic local work without perturbing caller RNG streams."""
    torch = _torch()
    numpy_state = np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[], enabled=True):
            torch.manual_seed(int(seed))
            np.random.seed(int(seed) % (2 ** 32))
            yield
    finally:
        np.random.set_state(numpy_state)


def _model(config):
    torch = _torch()
    layers = []
    width = 2
    for _ in range(config.hidden_layers):
        linear = torch.nn.Linear(width, config.hidden_width)
        if config.activation == "tanh":
            torch.nn.init.xavier_normal_(linear.weight, gain=5.0 / 3.0)
        else:
            limit = 3.0 / np.sqrt(config.hidden_width)
            torch.nn.init.uniform_(linear.weight, -limit, limit)
        torch.nn.init.zeros_(linear.bias)
        activation = torch.nn.Tanh() if config.activation == "tanh" else _Sine()
        layers.extend((linear, activation))
        width = config.hidden_width
    output = torch.nn.Linear(width, 1)
    if config.activation == "tanh":
        torch.nn.init.xavier_normal_(output.weight, gain=5.0 / 3.0)
    else:
        limit = 3.0 / np.sqrt(config.hidden_width)
        torch.nn.init.uniform_(output.weight, -limit, limit)
    torch.nn.init.zeros_(output.bias)
    layers.append(output)
    return torch.nn.Sequential(*layers).to(dtype=_dtype(torch, config.dtype), device="cpu")


class _Sine:
    """Lazily materialized torch module to keep torch an optional dependency."""

    def __new__(cls):
        torch = _torch()

        class Sine(torch.nn.Module):
            def forward(self, value):
                return torch.sin(value)

        return Sine()


def _validate_coordinates(coordinates, values=None):
    xy = np.asarray(coordinates, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or xy.shape[0] < 2:
        raise PICError("coordinates must have shape (n, 2), ordered as (x, t)")
    if not np.all(np.isfinite(xy)) or np.ptp(xy[:, 0]) <= 0 or np.ptp(xy[:, 1]) <= 0:
        raise PICError("coordinates must be finite and span nonzero x and t intervals")
    if values is None:
        return xy
    u = np.asarray(values, dtype=float).reshape(-1, 1)
    if u.shape[0] != xy.shape[0] or not np.all(np.isfinite(u)):
        raise PICError("observation values must be finite and align with coordinates")
    return xy, u


def _derivatives(model, coordinates, *, create_graph):
    torch = _torch()
    z = coordinates.detach().clone().requires_grad_(True)
    u = model(z)
    grad = torch.autograd.grad(u.sum(), z, create_graph=True)[0]
    ux, ut = grad[:, 0:1], grad[:, 1:2]
    # Keep the second derivative graph until u_xxx is formed.  ``create_graph``
    # controls only whether the final derivatives remain differentiable for PINN
    # back-propagation.
    uxx = torch.autograd.grad(ux.sum(), z, create_graph=True)[0][:, 0:1]
    uxxx = torch.autograd.grad(uxx.sum(), z, create_graph=create_graph)[0][:, 0:1]
    return u, ut, ux, uxx, uxxx


def _library(terms, u, ux, uxx, uxxx=None):
    torch = _torch()
    bad = sorted(set(terms) - set(SUPPORTED_TERMS))
    if bad:
        raise PICError("unsupported scalar 1D PDE term(s): " + ", ".join(bad))
    if uxxx is None and any("u_xxx" in term for term in terms):
        raise PICError("u_xxx was requested but no third derivative was supplied")
    columns = {
        "1": torch.ones_like(u), "u": u, "u_x": ux, "u_xx": uxx,
        "u_xxx": uxxx,
        "u*u_x": u * ux, "u*u_xx": u * uxx, "u*u_xxx": None if uxxx is None else u * uxxx,
        "u^2": u.square(), "u^3": u.pow(3),
    }
    if not terms:
        raise PICError("candidate must contain at least one RHS term")
    return torch.cat([columns[t] for t in terms], dim=1)


def _reference_identity(train_xy, train_u, idx, config):
    payload = {
        "version": PIC_IMPLEMENTATION_VERSION,
        "config": asdict(config),
        "train_indices_sha256": hashlib.sha256(idx.astype("<i8").tobytes()).hexdigest(),
        "observations_sha256": hashlib.sha256(
            np.ascontiguousarray(np.c_[train_xy, train_u]).astype("<f8").tobytes()
        ).hexdigest(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(), payload


def _load_cached_reference(cache_file, cache_key, config):
    torch = _torch()
    try:
        with np.load(cache_file, allow_pickle=False) as cached:
            metadata = json.loads(str(cached["metadata"].item()))
            if metadata.get("cache_key") != cache_key:
                raise PICError("PIC reference cache key mismatch")
            if metadata.get("version") != PIC_IMPLEMENTATION_VERSION:
                raise PICError("PIC reference cache version mismatch")
            model = _model(config)
            state_names = metadata["state_names"]
            template = model.state_dict()
            state = {
                name: torch.as_tensor(cached[f"state_{index}"], dtype=template[name].dtype)
                for index, name in enumerate(state_names)
            }
            model.load_state_dict(state)
            reference = PreparedPICReference(
                time=cached["time"].copy(), lhs=cached["lhs"].copy(),
                ann_output=cached["ann_output"].copy(),
                observed_range=tuple(metadata["observed_range"]), cache_key=cache_key,
                seed=config.seed,
            )
            stored_seconds = float(metadata["reference_training_seconds"])
            cost = {
                "reference_epochs": float(config.reference_epochs),
                "reference_seconds": 0.0,
                "reference_training_seconds_cached": stored_seconds,
                "reference_observations": float(len(cached["observation_coordinates"])),
                "reference_cache_hit": 1.0,
            }
            return TorchPICPrepared(
                reference, {key: value.detach().cpu().clone() for key, value in state.items()},
                config, cached["observation_coordinates"].copy(),
                cached["observation_values"].copy(), cached["evaluation_coordinates"].copy(),
                cost, float(metadata["reference_train_rmse"]),
                float(metadata["reference_train_normalized_rmse"]),
            )
    except PICError:
        raise
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise PICError(f"PIC reference cache is unreadable: {cache_file}: {exc}") from exc


def _store_cached_reference(cache_file, prepared, training_seconds):
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    state_names = list(prepared.model_state)
    metadata = {
        "cache_key": prepared.reference.cache_key,
        "version": PIC_IMPLEMENTATION_VERSION,
        "state_names": state_names,
        "observed_range": list(prepared.reference.observed_range),
        "reference_training_seconds": float(training_seconds),
        "reference_train_rmse": prepared.reference_train_rmse,
        "reference_train_normalized_rmse": prepared.reference_train_normalized_rmse,
    }
    arrays = {
        "metadata": np.asarray(json.dumps(metadata, sort_keys=True)),
        "time": prepared.reference.time,
        "lhs": prepared.reference.lhs,
        "ann_output": prepared.reference.ann_output,
        "observation_coordinates": prepared.observation_coordinates,
        "observation_values": prepared.observation_values,
        "evaluation_coordinates": prepared.evaluation_coordinates,
    }
    arrays.update({
        f"state_{index}": prepared.model_state[name].detach().cpu().numpy()
        for index, name in enumerate(state_names)
    })
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".npz", prefix=cache_file.stem + ".", dir=cache_file.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, cache_file)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def prepare_torch_pic_reference(coordinates, values, *, train_indices=None,
                                config: TorchPICConfig = TorchPICConfig(), cache_dir=None):
    """Fit the shared ANN using only the explicitly selected observations."""
    _validate_config(config)
    torch = _torch()
    xy, u = _validate_coordinates(coordinates, values)
    if train_indices is None:
        idx = np.arange(len(xy), dtype=int)
    else:
        raw_idx = np.asarray(train_indices)
        if raw_idx.ndim != 1 or raw_idx.dtype.kind not in "iu":
            raise PICError("train_indices must be a one-dimensional integer array")
        idx = raw_idx.astype(int, copy=False)
    if idx.size < 2 or np.any(idx < 0) or np.any(idx >= len(xy)) or np.unique(idx).size != idx.size:
        raise PICError("train_indices must be unique, in range, and contain at least two rows")
    train_xy, train_u = xy[idx], u[idx]
    lower, upper = float(train_u.min()), float(train_u.max())
    if upper <= lower:
        raise PICError("training observations must have a nonzero value range")
    cache_key, _ = _reference_identity(train_xy, train_u, idx, config)
    cache_file = None
    if cache_dir is not None:
        cache_file = Path(cache_dir).expanduser().resolve() / f"{cache_key}.npz"
        if cache_file.is_file():
            return _load_cached_reference(cache_file, cache_key, config)
    with _isolated_rng(config.seed):
        model = _model(config)
        dtype = _dtype(torch, config.dtype)
        tx = torch.as_tensor(train_xy, dtype=dtype)
        tu = torch.as_tensor(train_u, dtype=dtype)
        opt = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        start = time.perf_counter()
        for _ in range(config.reference_epochs):
            opt.zero_grad()
            loss = (model(tx) - tu).square().mean()
            if not torch.isfinite(loss):
                raise PICError("reference ANN training became non-finite")
            loss.backward()
            opt.step()
        train_seconds = time.perf_counter() - start
        with torch.no_grad():
            train_pred = model(tx)
            train_rmse = float(torch.sqrt((train_pred - tu).square().mean()))
            train_nrmse = train_rmse / (upper - lower)
            if (config.reference_max_normalized_rmse is not None
                    and train_nrmse > config.reference_max_normalized_rmse):
                raise PICError(
                    "reference ANN quality threshold failed: normalized RMSE "
                    f"{train_nrmse:.6g} > {config.reference_max_normalized_rmse:.6g}"
                )
        xg = np.linspace(train_xy[:, 0].min(), train_xy[:, 0].max(), config.nx)
        tg = np.linspace(train_xy[:, 1].min(), train_xy[:, 1].max(), config.nt)
        eval_xy = np.stack(np.meshgrid(xg, tg, indexing="ij"), axis=-1).reshape(-1, 2)
        ez = torch.as_tensor(eval_xy, dtype=dtype)
        pred, lhs, ux, uxx, uxxx = _derivatives(model, ez, create_graph=False)
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    reference = PreparedPICReference(
        time=eval_xy[:, 1], lhs=lhs.detach().numpy().reshape(-1),
        ann_output=pred.detach().numpy().reshape(-1), observed_range=(lower, upper),
        cache_key=cache_key, seed=config.seed,
    )
    cost = {"reference_epochs": float(config.reference_epochs),
            "reference_seconds": train_seconds, "reference_observations": float(idx.size),
            "reference_cache_hit": 0.0}
    prepared = TorchPICPrepared(reference, state, config, train_xy.copy(), train_u.copy(),
                                eval_xy, cost, train_rmse, train_nrmse)
    if cache_file is not None:
        _store_cached_reference(cache_file, prepared, train_seconds)
    return prepared


def _reference_rhs(prepared, terms):
    torch = _torch()
    with _isolated_rng(prepared.config.seed):
        model = _model(prepared.config)
        model.load_state_dict(prepared.model_state)
        z = torch.as_tensor(prepared.evaluation_coordinates, dtype=_dtype(torch, prepared.config.dtype))
        u, _, ux, uxx, uxxx = _derivatives(model, z, create_graph=False)
        return _library(tuple(terms), u, ux, uxx, uxxx).detach().numpy()


class _CandidateTrainer:
    def __init__(self, prepared, terms):
        self.prepared, self.terms = prepared, tuple(terms)

    def __call__(self, request):
        torch = _torch()
        cfg = self.prepared.config
        epochs = int(request.budget.get("epochs", cfg.pinn_epochs))
        if epochs < 1 or epochs != cfg.pinn_epochs:
            raise PICError("candidate budget must equal the prepared fixed pinn_epochs")
        with _isolated_rng(cfg.seed):
            model = _model(cfg)
            model.load_state_dict(copy.deepcopy(self.prepared.model_state))
            dtype = _dtype(torch, cfg.dtype)
            obs_x = torch.as_tensor(self.prepared.observation_coordinates, dtype=dtype)
            obs_u = torch.as_tensor(self.prepared.observation_values, dtype=dtype)
            collocation = torch.as_tensor(self.prepared.evaluation_coordinates, dtype=dtype)
            opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
            coefficient = np.asarray(request.initial_coefficients)
            start = time.perf_counter()
            initial_loss = None
            final_loss = None
            for epoch in range(epochs):
                opt.zero_grad()
                u, ut, ux, uxx, uxxx = _derivatives(model, collocation, create_graph=True)
                lib = _library(self.terms, u, ux, uxx, uxxx)
                coefficient = fit_tls_coefficients(ut.detach().numpy(), lib.detach().numpy())
                coef_t = torch.as_tensor(coefficient.reshape(-1, 1), dtype=dtype)
                data_loss = (model(obs_x) - obs_u).square().mean()
                physics_loss = (ut - lib @ coef_t).square().mean()
                loss = data_loss + cfg.physics_weight * physics_loss
                if not torch.isfinite(loss):
                    return PINNTrainingResult(np.empty(0), coefficient, False, epoch,
                                              {"pinn_epochs": float(epoch)}, "non-finite PINN loss")
                loss.backward()
                opt.step()
                current_loss = float(loss.detach())
                initial_loss = current_loss if initial_loss is None else initial_loss
                final_loss = current_loss
            seconds = time.perf_counter() - start
            with torch.no_grad():
                output = model(collocation).detach().numpy().reshape(-1)
        relative_improvement = ((initial_loss - final_loss) / max(abs(initial_loss), 1e-15))
        cost = {"pinn_epochs": float(epochs), "pinn_seconds": seconds,
                "collocation_points": float(len(collocation)),
                "pinn_initial_loss": initial_loss, "pinn_final_loss": final_loss,
                "pinn_relative_loss_improvement": relative_improvement}
        if (cfg.pinn_min_relative_loss_improvement is not None
                and relative_improvement < cfg.pinn_min_relative_loss_improvement):
            return PINNTrainingResult(
                output, coefficient, False, epochs, cost,
                "PINN loss did not meet the configured relative-improvement threshold",
            )
        if cfg.pinn_max_final_loss is not None and final_loss > cfg.pinn_max_final_loss:
            return PINNTrainingResult(
                output, coefficient, False, epochs, cost,
                "PINN final loss exceeded the configured maximum",
            )
        return PINNTrainingResult(output, coefficient, True, epochs, cost)


def evaluate_torch_pic(prepared: TorchPICPrepared, terms: Sequence[str], *,
                       candidate_id: str, original_coefficients: Optional[Sequence[float]] = None):
    """Evaluate one candidate using a fresh clone and the fixed prepared budget."""
    terms = tuple(terms)
    try:
        rhs = _reference_rhs(prepared, terms)
        evaluator = PhysicsInformedInformationCriterion(_CandidateTrainer(prepared, terms))
        result = evaluator.evaluate(
            prepared.reference, rhs, candidate_id=candidate_id,
            budget={"epochs": prepared.config.pinn_epochs}, n_windows=prepared.config.n_windows,
            window_fraction=prepared.config.window_fraction,
            original_coefficients=original_coefficients,
        )
    except (PICError, np.linalg.LinAlgError, TypeError, ValueError) as exc:
        result = PICResult(
            status="unsupported_candidate", pic=None, r_loss=None, p_loss=None,
            window_coefficients=None,
            original_coefficients=(None if original_coefficients is None else
                                   np.asarray(original_coefficients, dtype=float).reshape(-1)),
            reference_fit_coefficients=None, refitted_coefficients=None, cost={},
            cache_key=prepared.reference.cache_key, message=str(exc),
        )
    merged_cost = dict(prepared.reference_cost)
    merged_cost.update(result.cost)
    return replace(result, cost=merged_cost)
