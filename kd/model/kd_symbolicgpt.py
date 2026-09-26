# kd/model/kd_symbolicgpt.py

import copy
import pickle
import random
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..base import BaseEstimator
from ..metrics import MSE
from .symbolicgpt import (
    GPT,
    CharDataset,
    GPTConfig,
    PointNetConfig,
    Trainer,
    TrainerConfig,
    evaluate_expression,
    fit_constants,
    generate_equation,
    points_tensor_from_xy,
    sample_from_model,
    sample_points_for_equation,
    set_seed,
)

_DEFAULT_OP_LIST = ("add", "sub", "mul", "div", "sin", "cos", "pow")


class KD_SymbolicGPT(BaseEstimator):
    """
    Generative symbolic-regression baseline (SymbolicGPT): a character-level
    GPT that autoregressively samples equation *skeletons* (C-templated,
    e.g. `"C*sin(x1)+C"`) conditioned on a point-cloud encoding of the data
    (via a PointNet-style encoder), then fits the C constants against the
    real (X, y) by least-squares-style optimization.

    Unlike the original repo (which pretrains once on a large multi-variable
    corpus loaded from disk, then evaluates against held-out equations with
    known ground truth), this wrapper pretrains a small GPT from scratch on
    a synthetic corpus generated for the target's own number of variables,
    every `fit()` call -- the same self-contained, no-external-checkpoint
    approach `kd.model.kd_eqgpt.KD_EqGPT` uses, and for the same reason:
    the original repo's saved checkpoint depends on a character vocabulary
    that was never saved alongside it, so it can't be safely reloaded here.

    Follows the scikit-learn-style API used by other `kd` baselines
    (see `kd.model.kd_sga.KD_SGA`): configure via `__init__`, run via
    `fit`/`fit_dataset`, read results from `self.best_expression_` etc.
    """

    def __init__(
        self,
        embedding_size: int = 64,
        n_layer: int = 4,
        n_head: int = 4,
        method: str = "EMB_SUM",
        variable_embedding: str = "NOT_VAR",
        pretrain_corpus_size: int = 200,
        pretrain_epochs: int = 2,
        batch_size: int = 64,
        learning_rate: float = 6e-4,
        n_levels: int = 3,
        allow_constants: bool = True,
        const_range: Tuple[float, float] = (-2.0, 2.0),
        op_list: Sequence[str] = _DEFAULT_OP_LIST,
        min_x: float = -3.0,
        max_x: float = 3.0,
        num_candidates: int = 10,
        max_conditioning_points: Optional[int] = None,
        candidate_batch_size: int = 1,
        cache_dataset_tensors: bool = True,
        dataloader_workers: int = 0,
        temperature: float = 1.0,
        top_k: float = 0.0,
        top_p: float = 0.7,
        seed: int = 0,
        device: Optional[str] = None,
        verbose: bool = False,
    ):
        """
        Parameters
        ----------
        embedding_size, n_layer, n_head :
            GPT transformer size (also the PointNet encoder's embedding
            width, which must match).
        method :
            How the point-cloud embedding is fused with token/position
            embeddings; one of "EMB_SUM"/"EMB_CAT"/"EMB_CON"/"OUT_SUM"/"OUT_CAT".
        variable_embedding :
            "NOT_VAR" (default, no extra variable-count embedding) or
            "LEA_EMB" (adds a learned embedding of the equation's variable
            count to the point-cloud embedding).
        pretrain_corpus_size, pretrain_epochs, batch_size, learning_rate :
            Synthetic-corpus pretraining settings. The corpus is generated
            fresh for the target's own number of variables (see `fit`).
        n_levels, allow_constants, const_range, op_list :
            Passed to `symbolicgpt.generator.generate_equation` when
            building the synthetic corpus -- controls expression-tree depth
            and which operators/constants can appear.
        min_x, max_x :
            Range used both to sample synthetic corpus points and to
            interpret the real data's x-range for point-cloud encoding.
        num_candidates :
            Number of equation skeletons sampled from the trained GPT at
            inference time; the one whose fitted constants give the lowest
            training error is kept.
        max_conditioning_points :
            Optional deterministic cap on the point cloud used to condition
            the GPT and build its synthetic corpus. Constant fitting and final
            scoring still use every input row. This prevents large tabular
            datasets from multiplying corpus generation and DataLoader work.
        candidate_batch_size :
            Number of candidate skeletons sampled together. Batching turns
            many tiny autoregressive GPU launches into fewer, wider launches.
        cache_dataset_tensors :
            Cache immutable encoded corpus items after their first DataLoader
            visit so subsequent epochs do not rebuild point tensors in Python.
        dataloader_workers :
            Worker processes used by the training DataLoader. Zero works best
            with the in-process tensor cache; positive values can help when
            augmentation is enabled.
        temperature, top_k, top_p :
            Autoregressive sampling controls (see `symbolicgpt.utils.sample_from_model`).
        seed : Random seed (numpy + torch + python `random`).
        device : "cuda"/"cpu"/None (auto-detect).
        """
        self.embedding_size = embedding_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.method = method
        self.variable_embedding = variable_embedding
        self.pretrain_corpus_size = pretrain_corpus_size
        self.pretrain_epochs = pretrain_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_levels = n_levels
        self.allow_constants = allow_constants
        self.const_range = const_range
        self.op_list = list(op_list)
        self.min_x = min_x
        self.max_x = max_x
        self.num_candidates = num_candidates
        self.max_conditioning_points = max_conditioning_points
        self.candidate_batch_size = candidate_batch_size
        self.cache_dataset_tensors = cache_dataset_tensors
        self.dataloader_workers = dataloader_workers
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.seed = seed
        self.device = device
        self.verbose = verbose

        if self.max_conditioning_points is not None and self.max_conditioning_points < 1:
            raise ValueError("max_conditioning_points must be positive or None")
        if self.candidate_batch_size < 1:
            raise ValueError("candidate_batch_size must be positive")
        if self.dataloader_workers < 0:
            raise ValueError("dataloader_workers must be non-negative")

    def _resolve_device(self) -> torch.device:
        if self.device is not None:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_pretrain_corpus(self, num_vars: int, num_points: int) -> List[dict]:
        corpus = []
        attempts = 0
        max_attempts = self.pretrain_corpus_size * 5
        while len(corpus) < self.pretrain_corpus_size and attempts < max_attempts:
            attempts += 1
            try:
                clean_eqn, skeleton_eqn = generate_equation(
                    num_vars,
                    n_levels=self.n_levels,
                    allow_constants=self.allow_constants,
                    const_range=self.const_range,
                    op_list=self.op_list,
                )
                X, Y = sample_points_for_equation(
                    clean_eqn,
                    n_points=num_points,
                    n_vars=num_vars,
                    min_x=self.min_x,
                    max_x=self.max_x,
                )
                if any(np.isnan(Y)) or any(np.isinf(Y)):
                    continue
            except Exception:
                continue  # a small fraction of random equations are degenerate (div-by-zero, etc.)
            corpus.append({"X": X, "Y": Y, "EQ": clean_eqn, "Skeleton": skeleton_eqn})

        if len(corpus) == 0:
            raise RuntimeError(
                "Failed to generate any valid synthetic equations for pretraining "
                f"(num_vars={num_vars}, op_list={self.op_list}); try a smaller n_levels "
                "or a different op_list."
            )
        return corpus

    def fit(self, X: Any, y: Any, *, pretrain_cache=None) -> "KD_SymbolicGPT":
        """
        Discover a symbolic expression for (X, y).

        Parameters
        ----------
        X : array-like, shape (n_samples, n_vars) or (n_samples,)
        y : array-like, shape (n_samples,)
        pretrain_cache : dict or None
            Optional caller-owned, single-entry cache for one dataset instance.
            Reuses only synthetic pretraining with identical parameters and
            shape. Each target still samples and fits constants independently.
            The benchmark creates and discards this cache within one case;
            ordinary fit calls continue to train from scratch.
        """
        # A failed refit must not leave a previous successful solution usable.
        for attribute in (
            "best_expression_", "best_skeleton_", "train_loss_", "test_mse_",
            "model_", "train_dataset_", "conditioning_indices_",
        ):
            self.__dict__.pop(attribute, None)
        self.candidates_ = []
        self.pretraining_ = {
            "protocol": ("synthetic_case_instance_reuse" if pretrain_cache is not None
                         else "synthetic_per_fit"),
            "cache_hit": False,
        }
        self.candidate_validation_ = dict(
            sampled=0, invalid_tokens=0, fit_failed=0,
            nonfinite_loss=0, invalid_predictions=0, accepted=0,
        )
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        set_seed(self.seed)
        device = self._resolve_device()

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.asarray(y, dtype=float).reshape(-1)
        if X.ndim != 2 or not X.size:
            raise ValueError("X must be a nonempty two-dimensional array")
        num_vars = X.shape[1]
        if len(X) != len(y):
            raise ValueError(f"X and y have different lengths: {len(X)} != {len(y)}")
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError("SymbolicGPT requires finite X and y training values")

        if self.max_conditioning_points is not None and len(X) > self.max_conditioning_points:
            # Evenly spaced indices are deterministic and retain coverage of
            # ordered scientific datasets without consuming the global RNG
            # used to construct the seeded synthetic corpus.
            conditioning_indices = np.linspace(
                0, len(X) - 1, self.max_conditioning_points, dtype=int
            )
            conditioning_X = X[conditioning_indices]
            conditioning_y = y[conditioning_indices]
        else:
            conditioning_indices = np.arange(len(X))
            conditioning_X = X
            conditioning_y = y
        num_points = len(conditioning_X)
        self.conditioning_indices_ = conditioning_indices

        # Pretraining uses only configuration, shape and seed, never real X/y.
        # Pickle is used solely to create an exact in-memory key; no data is
        # deserialized. Include every estimator parameter conservatively.
        cache_key = pickle.dumps((
            type(self).__module__, type(self).__qualname__, self.get_params(deep=False),
            num_vars, num_points, str(device), str(torch.get_default_dtype()),
            torch.cuda.current_device() if device.type == "cuda" else None,
        ), protocol=4) if pretrain_cache is not None else None
        if pretrain_cache is not None and pretrain_cache.get("key") == cache_key:
            model = copy.deepcopy(pretrain_cache["model"])
            train_dataset = pretrain_cache["dataset"]
            random.setstate(pretrain_cache["python_rng"])
            np.random.set_state(pretrain_cache["numpy_rng"])
            torch.set_rng_state(pretrain_cache["torch_rng"])
            if pretrain_cache["cuda_rng"] is not None:
                torch.cuda.set_rng_state_all(pretrain_cache["cuda_rng"])
            self.pretraining_["cache_hit"] = True
        else:
            if pretrain_cache is not None:
                pretrain_cache.clear()
            model, train_dataset = self._pretrain(num_vars, num_points, device)
            if pretrain_cache is not None:
                # Snapshot before target-conditioned sampling; inference never
                # mutates the corpus. Copy weights so an estimator cannot alter
                # the cached model through its public model_ attribute.
                pretrain_cache.update(
                    key=cache_key, model=copy.deepcopy(model), dataset=train_dataset,
                    python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                    torch_rng=torch.get_rng_state(),
                    cuda_rng=(torch.cuda.get_rng_state_all()
                              if torch.cuda.is_initialized() else None),
                )

        return self._fit_candidates(
            model, train_dataset, X, y, conditioning_X, conditioning_y, device,
        )

    def _pretrain(self, num_vars, num_points, device):
        """Generate a synthetic corpus and train independently of target values."""
        corpus = self._build_pretrain_corpus(num_vars, num_points)

        text = "".join("<" + str(rec["Skeleton"]) + ">" for rec in corpus)
        chars = sorted(set(text) | {"_", ":"})
        block_size = max(len(str(rec["Skeleton"])) for rec in corpus) + 2

        train_dataset = CharDataset(
            corpus,
            block_size,
            chars,
            numVars=num_vars,
            numYs=1,
            numPoints=[num_points, num_points + 1],
            target="Skeleton",
            cache_tensors=self.cache_dataset_tensors,
        )

        pconf = PointNetConfig(
            embeddingSize=self.embedding_size,
            numberofPoints=num_points,
            numberofVars=num_vars,
            numberofYs=1,
            method=self.method,
            variable_embedding=self.variable_embedding,
        )
        mconf = GPTConfig(
            train_dataset.vocab_size,
            train_dataset.block_size,
            n_layer=self.n_layer,
            n_head=self.n_head,
            n_embd=self.embedding_size,
            padding_idx=train_dataset.paddingID,
        )
        model = GPT(mconf, pconf)

        tconf = TrainerConfig(
            max_epochs=self.pretrain_epochs,
            batch_size=min(self.batch_size, len(train_dataset)),
            learning_rate=self.learning_rate,
            num_workers=self.dataloader_workers,
            show_progress=self.verbose,
        )
        trainer = Trainer(model, train_dataset, None, tconf, device)
        trainer.train()
        return model.eval(), train_dataset

    def _fit_candidates(self, model, train_dataset, X, y,
                        conditioning_X, conditioning_y, device):
        num_vars = X.shape[1]
        num_points = len(conditioning_X)
        # Sample candidate skeletons conditioned on the observed data.
        real_points = points_tensor_from_xy(
            conditioning_X,
            conditioning_y,
            num_vars,
            1,
            num_points,
        ).unsqueeze(0).to(device)
        real_vars = torch.tensor([[num_vars]], dtype=torch.long).to(device)
        seed_input = torch.tensor([[train_dataset.stoi["<"]]], dtype=torch.long).to(device)

        candidates = self.candidates_
        validation = self.candidate_validation_
        remaining = self.num_candidates
        while remaining > 0:
            batch_size = min(self.candidate_batch_size, remaining)
            sampled_batch = sample_from_model(
                model,
                seed_input.expand(batch_size, -1).clone(),
                train_dataset.block_size,
                points=real_points.expand(batch_size, -1, -1).contiguous(),
                variables=real_vars.expand(batch_size, -1).contiguous(),
                temperature=self.temperature,
                sample=True,
                top_k=self.top_k,
                top_p=self.top_p,
            )
            remaining -= batch_size

            for sampled in sampled_batch:
                validation["sampled"] += 1
                decoded = "".join(train_dataset.itos[int(i)] for i in sampled)
                prefix, end_token, _ = decoded.partition(">")
                if not prefix.startswith("<") or not end_token:
                    validation["invalid_tokens"] += 1
                    continue
                skeleton = prefix[1:]
                if skeleton == "" or "x" not in skeleton:
                    validation["invalid_tokens"] += 1
                    continue
                try:
                    # Constant fitting intentionally uses the complete input,
                    # not the capped conditioning point cloud.
                    expression, loss = fit_constants(skeleton, X, y)
                    loss = float(loss)
                except Exception:
                    validation["fit_failed"] += 1
                    continue
                if not np.isfinite(loss):
                    validation["nonfinite_loss"] += 1
                    continue
                try:
                    predictions = evaluate_expression(expression, X)
                    if predictions.shape != y.shape or not np.isfinite(predictions).all():
                        raise ValueError("Invalid training predictions")
                except Exception:
                    validation["invalid_predictions"] += 1
                    continue
                candidates.append((loss, skeleton, expression))
                validation["accepted"] += 1

        if not candidates:
            raise RuntimeError(
                f"No valid SymbolicGPT candidates (0/{validation['sampled']} accepted; "
                f"validation={validation}). Candidates must have valid equation syntax, "
                "finite fitted loss and finite predictions on every training row; "
                "try increasing pretrain_epochs/pretrain_corpus_size or num_candidates."
            )
        candidates.sort(key=lambda c: c[0])
        best_loss, best_skeleton, best_expression = candidates[0]

        self.best_expression_ = best_expression
        self.best_skeleton_ = best_skeleton
        self.train_loss_ = best_loss
        self.candidates_ = candidates
        self.model_ = model
        self.train_dataset_ = train_dataset

        if self.verbose:
            print(f"[SymbolicGPT] best skeleton: {best_skeleton}")
            print(f"[SymbolicGPT] best expression: {best_expression} (train loss={best_loss:.6g})")

        return self

    def fit_dataset(self, dataset: Any) -> "KD_SymbolicGPT":
        """
        Discover a symbolic expression for a `kd.dataset.SymbolicRegressionDataset`
        (e.g. from `SymbolicRegressionDataset(name='Koza-2')`). Also scores
        the result against the dataset's held-out test split, stored as
        `self.test_mse_`.
        """
        data = dataset.get_data()
        self.fit(data["X_train"], data["y_train"])

        X_test = np.asarray(data.get("X_test"))
        y_test = np.asarray(data.get("y_test"))
        if X_test.size and y_test.size:
            y_pred = self.predict(X_test)
            self.test_mse_ = MSE()(y_test.reshape(-1), y_pred)

        return self

    def predict(self, X: Any) -> np.ndarray:
        """Evaluate the discovered `self.best_expression_` at new X."""
        if not hasattr(self, "best_expression_"):
            raise RuntimeError("Call fit()/fit_dataset() before predict().")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # The expression is produced internally by fit_constants. Evaluate it
        # once with vector-valued variables instead of rebuilding and parsing
        # one expression string per row.
        try:
            return evaluate_expression(self.best_expression_, X)
        except Exception:
            return np.full(X.shape[0], np.nan, dtype=float)
