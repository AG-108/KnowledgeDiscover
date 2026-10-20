"""Scheduled DeepMoD training with physical derivatives and explicit libraries."""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
from deepymod import DeepMoD
from deepymod.model.deepmod import Constraint, Estimator, Library

from kd.benchmark_deadline import current_deadline
from kd.model.pde_library import evaluate_terms, expression, library_names, normalized_stlsq


class PhysicalNetwork(torch.nn.Module):
    def __init__(self, X, y, hidden, activation="tanh"):
        super().__init__()
        self.register_buffer("center", torch.as_tensor((X.max(0) + X.min(0)) / 2))
        self.register_buffer("scale", torch.as_tensor(np.maximum(np.ptp(X, axis=0) / 2, 1e-8)))
        self.register_buffer("offset", torch.as_tensor(float(np.mean(y))))
        self.register_buffer("amplitude", torch.as_tensor(max(float(np.std(y)), 1e-8)))
        layers = []
        dimensions = [X.shape[1], *hidden, 1]
        for i, (nin, nout) in enumerate(zip(dimensions, dimensions[1:])):
            layers.append(torch.nn.Linear(nin, nout))
            if i < len(dimensions) - 2:
                layers.append(torch.nn.Tanh() if activation == "tanh" else Sine())
        self.network = torch.nn.Sequential(*layers)

    def values(self, coordinates):
        return self.offset + self.amplitude * self.network((coordinates - self.center) / self.scale)

    def forward(self, inputs):
        coordinates = inputs.detach().clone().requires_grad_(True)
        return self.values(coordinates), coordinates


class Sine(torch.nn.Module):
    def forward(self, values):
        return torch.sin(values)


class ExplicitLibrary(Library):
    def __init__(self, names):
        super().__init__()
        self.names = names

    def library(self, inputs):
        u, coords = inputs
        first = torch.autograd.grad(u.sum(), coords, create_graph=True)[0]
        ut, ux = first[:, :1], first[:, 1:2]
        uxx = torch.autograd.grad(ux.sum(), coords, create_graph=True)[0][:, 1:2]
        uxxx = torch.autograd.grad(uxx.sum(), coords, create_graph=True)[0][:, 1:2]
        theta = torch.cat(evaluate_terms(self.names, {
            "u": u, "u_x": ux, "u_xx": uxx, "u_xxx": uxxx, "x": coords[:, 1:2],
        }), dim=1)
        return [ut], [theta]


class NormalizedConstraint(Constraint):
    @staticmethod
    def map_coeffs(mask, coefficients):
        return coefficients.new_zeros((mask.shape[0], 1)).masked_scatter_(mask[:, None], coefficients)

    def fit(self, thetas, derivatives):
        result = []
        for theta, dt in zip(thetas, derivatives):
            if theta.shape[1] == 0:
                result.append(theta.new_empty((0, 1)))
                continue
            scales = torch.linalg.vector_norm(theta, dim=0).detach().clamp_min(1e-12)
            design = theta / scales
            gram = design.T @ design + 1e-6 * torch.eye(theta.shape[1], device=theta.device, dtype=theta.dtype)
            result.append(torch.linalg.solve(gram, design.T @ dt) / scales[:, None])
        return result


class SparseEstimator(Estimator):
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def fit(self, X, y):
        return normalized_stlsq(X, y, self.threshold)


def fit_scheduled(adapter, X, y):
    dtype = getattr(torch, adapter.precision)
    array_dtype = getattr(np, adapter.precision)
    X, y = np.asarray(X, dtype=array_dtype), np.asarray(y, dtype=array_dtype).reshape(-1, 1)
    torch.manual_seed(adapter.seed)
    rng = np.random.default_rng(adapter.seed)
    names = library_names(adapter.library_kind, adapter.coordinate_terms, X[:, 1])
    network = PhysicalNetwork(X, y, adapter.hidden_layers, adapter.activation).to(adapter.device, dtype=dtype)
    model = DeepMoD(network, ExplicitLibrary(names), SparseEstimator(adapter.threshold), NormalizedConstraint())
    model.to(adapter.device, dtype=dtype)
    tx, ty = torch.as_tensor(X, device=adapter.device), torch.as_tensor(y, device=adapter.device)
    permutation = rng.permutation(len(X))
    n_valid = max(1, len(X) // 10)
    valid = torch.as_tensor(permutation[:n_valid], device=adapter.device)
    train = torch.as_tensor(permutation[n_valid:], device=adapter.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=adapter.learning_rate)
    amplitude_sq = network.amplitude**2
    derivative_scale_sq = (network.amplitude / network.scale[0])**2
    best_error, best_state = float("inf"), None
    deadline = current_deadline()
    reserve = min(15.0, deadline.remaining() * 0.12) if deadline else 0
    steps, mask_updates, losses = 0, 0, []
    stopped = "budget_exhausted"
    for step in range(adapter.max_steps):
        if deadline and deadline.remaining() <= reserve:
            stopped = "time_limit"
            break
        ids = train[torch.randint(len(train), (min(adapter.batch_size, len(train)),), device=adapter.device)]
        xb, yb = tx[ids], ty[ids]
        optimizer.zero_grad()
        if step == adapter.warmup_steps and adapter.lbfgs_steps:
            refine_ids = train[:min(len(train), 4096)]
            refine_until = time.perf_counter() + (deadline.remaining() * .35 if deadline else 60)
            refiner = torch.optim.LBFGS(network.parameters(), max_iter=adapter.lbfgs_steps,
                tolerance_grad=1e-12, tolerance_change=1e-15, line_search_fn="strong_wolfe")

            class RefinementDone(Exception):
                pass

            def closure():
                if time.perf_counter() >= refine_until:
                    raise RefinementDone()
                refiner.zero_grad()
                value = torch.mean((network.values(tx[refine_ids]) - ty[refine_ids])**2) / amplitude_sq
                value.backward()
                return value

            try:
                refiner.step(closure)
            except RefinementDone:
                pass
            # A new Adam state avoids carrying moments from before refinement.
            optimizer = torch.optim.Adam(model.parameters(), lr=adapter.learning_rate * .1)
        if step < adapter.warmup_steps:
            prediction = network.values(xb)
            loss = torch.mean((prediction - yb)**2) / amplitude_sq
        else:
            prediction, dts, thetas = model(xb)
            if (step - adapter.warmup_steps) % adapter.sparsity_interval == 0:
                # The upstream sparse estimator must be explicitly invoked.
                masks = model.sparse_estimator(thetas, dts)
                if any(mask.any() for mask in masks):
                    model.constraint.sparsity_masks = masks
                    model.constraint((dts, thetas))
                    mask_updates += 1
            coefficients = model.constraint_coeffs(sparse=True)
            residual = sum(torch.mean((dt - theta @ coef)**2)
                           for dt, theta, coef in zip(dts, thetas, coefficients))
            loss = torch.mean((prediction - yb)**2) / amplitude_sq + adapter.pde_weight * residual / derivative_scale_sq
        if not torch.isfinite(loss):
            stopped = "nonfinite_update"
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 100)
        optimizer.step()
        steps = step + 1
        if step % 100 == 0 or step + 1 == adapter.max_steps:
            with torch.no_grad():
                error = float(torch.mean((network.values(tx[valid]) - ty[valid])**2) / amplitude_sq)
            losses.append({"step": steps, "observation_validation_nmse": error})
            if error < best_error:
                best_error = error
                best_state = copy.deepcopy(network.state_dict())
    if best_state is None:
        raise RuntimeError("DeepMoD did not complete a finite update")
    network.load_state_dict(best_state)
    # Refresh derivatives and coefficients on the training observations, rather
    # than exporting a stale coefficient vector from the last random minibatch.
    matrices, targets = [], []
    for start in range(0, len(tx), 2048):
        dts, thetas = model.library(network(tx[start:start + 2048]))
        matrices.append(thetas[0].detach().cpu().numpy())
        targets.append(dts[0].detach().cpu().numpy())
    theta, target = np.concatenate(matrices), np.concatenate(targets).reshape(-1)
    coefficients = normalized_stlsq(theta, target, adapter.threshold)
    model.constraint.sparsity_masks = [torch.as_tensor(coefficients != 0, device=adapter.device)]
    model.constraint.coeff_vectors = [torch.as_tensor(coefficients[:, None], device=adapter.device, dtype=dtype)]
    adapter.model_ = model
    adapter.feature_names_, adapter.coefficients_ = names, [coefficients]
    adapter.best_equation_ = "u_t = " + expression(names, coefficients)
    adapter.stop_reason_, adapter.completed_steps_ = stopped, steps
    adapter.sparsity_updates_ = mask_updates
    adapter.training_history_ = losses
    adapter.observation_validation_nmse_ = best_error
    return adapter
