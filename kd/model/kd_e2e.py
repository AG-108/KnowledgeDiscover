"""Optional adapter for the official E2E pretrained SymbolicTransformerRegressor.

No downloads or per-case pretraining. The official full-model pickle is executable;
loading requires an explicit trust flag and the expected checkpoint SHA256.
"""

from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
import re
import sys
import time

import numpy as np


class E2EUnavailable(RuntimeError):
    """Required optional source/checkpoint is missing or was not trusted."""


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_official(source_dir, checkpoint, device):
    import torch

    source = str(Path(source_dir).resolve())
    sys.path.insert(0, source)
    try:
        package = importlib.import_module("symbolicregression")
        actual = Path(package.__file__).resolve()
        if not actual.is_relative_to(Path(source)):
            raise E2EUnavailable("another symbolicregression package is already loaded")
        wrapper = importlib.import_module("symbolicregression.model.sklearn_wrapper")
        model = torch.load(str(checkpoint), map_location=device, weights_only=False)
    finally:
        sys.path.remove(source)
    model = model.to(device)
    model.eval()
    return model, wrapper.SymbolicTransformerRegressor


class _TimedModel:
    def __init__(self, model, device):
        self.model, self.device, self.seconds = model, device, 0.0

    def __getattr__(self, name):
        return getattr(self.model, name)

    def __call__(self, *args, **kwargs):
        import torch
        if self.device.startswith("cuda"):
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        try:
            return self.model(*args, **kwargs)
        finally:
            if self.device.startswith("cuda"):
                torch.cuda.synchronize(self.device)
            self.seconds += time.perf_counter() - start


def _tree_expression(tree, names):
    """Translate official tree nodes, rather than substring-replacing operators."""
    import sympy as sp
    if hasattr(tree, "nodes"):
        if len(tree.nodes) != 1:
            raise ValueError("E2E adapter supports scalar outputs only")
        tree = tree.nodes[0]
    value, children = str(tree.value), list(tree.children)
    if not children:
        match = re.fullmatch(r"x_(\d+)", value)
        if match:
            return sp.Symbol(names[int(match[1])])
        if value in {"pi", "e"}:
            return sp.pi if value == "pi" else sp.E
        return sp.Float(value)
    args = [_tree_expression(child, names) for child in children]
    operations = {
        "add": lambda a, b: a + b, "sub": lambda a, b: a - b,
        "mul": lambda a, b: a * b, "div": lambda a, b: a / b,
        "pow": lambda a, b: a ** b, "inv": lambda a: 1 / a,
        "pow2": lambda a: a ** 2, "pow3": lambda a: a ** 3,
        "neg": lambda a: -a, "sin": sp.sin, "cos": sp.cos,
        "id": lambda a: a,
        "tan": sp.tan, "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt,
        "abs": sp.Abs, "arcsin": sp.asin, "arccos": sp.acos,
        "arctan": sp.atan, "sinh": sp.sinh, "cosh": sp.cosh,
        "tanh": sp.tanh, "max": sp.Max, "min": sp.Min,
    }
    if value not in operations:
        raise ValueError(f"unsupported official tree operation: {value}")
    return operations[value](*args)


class E2ETransformerModel:
    def __init__(self, checkpoint_path=None, source_dir=None, checkpoint_sha256=None,
                 trust_checkpoint=False, device="cpu", seed=0, max_input_points=200,
                 n_trees_to_refine=10, stop_refinement_after=1, rescale=True):
        self.checkpoint_path, self.source_dir = checkpoint_path, source_dir
        self.checkpoint_sha256, self.trust_checkpoint = checkpoint_sha256, trust_checkpoint
        self.device, self.seed = device, seed
        self.options = dict(max_input_points=max_input_points,
                            n_trees_to_refine=n_trees_to_refine,
                            stop_refinement_after=stop_refinement_after, rescale=rescale)

    def check_available(self):
        if not self.checkpoint_path or not self.source_dir:
            raise E2EUnavailable("explicit official checkpoint_path and source_dir are required")
        checkpoint, source = Path(self.checkpoint_path).resolve(), Path(self.source_dir).resolve()
        if not checkpoint.is_file() or not (source / "symbolicregression/model/sklearn_wrapper.py").is_file():
            raise E2EUnavailable("official checkpoint or source checkout is missing")
        if self.trust_checkpoint is not True:
            raise E2EUnavailable("full-model pickle requires trust_checkpoint=true after source review")
        expected = str(self.checkpoint_sha256 or "").lower()
        if not re.fullmatch(r"[0-9a-f]{64}", expected) or _sha256(checkpoint) != expected:
            raise E2EUnavailable("checkpoint SHA256 is missing or does not match")
        return checkpoint, source

    def fit(self, X, y, variable_names=None):
        import torch
        X, y = np.asarray(X, float), np.asarray(y, float).reshape(-1)
        if X.ndim != 2 or len(X) != len(y) or len(X) < 2 or not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError("E2E needs aligned finite 2D X and scalar y")
        names = list(variable_names or [f"x{i+1}" for i in range(X.shape[1])])
        if len(names) != X.shape[1] or len(set(names)) != len(names):
            raise ValueError("variable_names must uniquely identify every feature")
        checkpoint, source = self.check_available()
        start = time.perf_counter()
        model, estimator_class = _load_official(source, checkpoint, self.device)
        load_seconds = time.perf_counter() - start
        timed = _TimedModel(model, self.device)
        estimator = estimator_class(model=timed, **self.options)
        numpy_state = np.random.get_state()
        devices = [torch.device(self.device).index or 0] if self.device.startswith("cuda") else []
        try:
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(self.seed)
                np.random.seed(self.seed)
                start = time.perf_counter()
                estimator.fit(X.copy(), y.copy(), verbose=False)
                total_seconds = time.perf_counter() - start
        finally:
            np.random.set_state(numpy_state)
        info = estimator.retrieve_tree(with_infos=True)
        tree = info.get("relabed_predicted_tree")
        if tree is None:
            raise RuntimeError("official E2E estimator returned no relabeled candidate tree")
        self.best_expression_ = str(_tree_expression(tree, names))
        self.predict_function_ = model.env.simplifier.tree_to_numexpr_fn(tree)
        self.n_features_in_ = X.shape[1]
        self.provenance_ = {
            "implementation": "official-e2e-pretrained-adapter-v1",
            "checkpoint_sha256": self.checkpoint_sha256,
            "source_dir": str(source),
            "source_wrapper_sha256": _sha256(source / "symbolicregression/model/sklearn_wrapper.py"),
            "device": self.device, "seed": self.seed, "pretraining_seconds": None,
            "pretraining_protocol": "external shared checkpoint; cost unavailable",
            "checkpoint_load_seconds": load_seconds, "fit_total_seconds": total_seconds,
            "inference_seconds": timed.seconds,
            "non_inference_fit_seconds": max(0.0, total_seconds - timed.seconds),
            "non_inference_scope": "refinement plus preprocessing; not refinement-only",
            "refinement_type": info.get("refinement_type"),
            "training_data_overlap": "unknown", "official_options": self.options,
        }
        return self

    def predict(self, X):
        X = np.asarray(X, float)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("prediction features must match training features")
        values = np.asarray(self.predict_function_(X.copy()))
        if values.shape == (len(X), 1):
            return values[:, 0]
        if values.shape == (len(X),):
            return values
        raise ValueError("official tree returned an unexpected prediction shape")
