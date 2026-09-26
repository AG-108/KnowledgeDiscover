"""Sampling, dataset, and constant-fitting utilities, adapted from
archive/kd/model/SymbolicGPT/utils.py.

Changes vs. the original:
- Dropped everything only needed for the original repo's file-based
  benchmark-comparison workflow (`plot_and_save_results`,
  `tokenize_predict_and_evaluate`, `generate_sample_and_evaluate`,
  `processDataFiles`) -- `kd_symbolicgpt.py` builds its training corpus
  in-memory and does its own (simpler) predict-and-fit-constants step
  directly, reusing `fit_constants`/`relativeErr` from here.
- Dropped the duplicate `mse` helper -- use `kd.metrics.MSE` instead.
- `generateDataStrEq` renamed to `sample_points_for_equation` (same body).
- `lossFunc` renamed to `_constants_loss` and wrapped by `fit_constants`,
  which does the `scipy.optimize.minimize` call the original repo left
  inlined at every call site.
- `CharDataset.__getitem__` now accepts corpus entries that are already
  `dict`s, not just JSON strings (it still supports JSON strings, e.g. if
  something reads from a `dataset.py`-style file). `kd_symbolicgpt.py`
  builds its synthetic corpus as plain in-memory dicts, so this avoids a
  pointless json.dumps/json.loads roundtrip per example per epoch.

Note on `eval()`: generated and sampled expressions are checked against an
arithmetic AST grammar before evaluation in an explicit NumPy namespace
with no builtins. The original protected sqrt/log/exp behavior is retained;
for example, sqrt uses abs internally. Constant fitting rejects expressions
unless their loss and predictions are finite on every training row.
"""

import ast
import builtins
import json
import math
import random
import re
from functools import lru_cache

import numpy as np
import torch
from numpy import *  # noqa: F401,F403 -- retained for vendored legacy helpers
from scipy.optimize import minimize
from torch.nn import functional as F
from torch.utils.data import Dataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def top_k_top_p_filtering(logits, top_k=0.0, top_p=0.0, filter_value=-float("Inf")):
    """Apply top-k and/or nucleus filtering to one or more token rows.

    ``logits`` may have shape ``[vocab]`` or ``[batch, vocab]``.  Supporting
    the batched form lets SymbolicGPT sample many candidate equations with a
    single set of GPU kernels instead of launching one tiny inference loop per
    candidate.
    """
    if logits.dim() not in (1, 2):
        raise ValueError(f"Expected 1-D or 2-D logits, got shape {tuple(logits.shape)}")
    squeeze = logits.dim() == 1
    if squeeze:
        logits = logits.unsqueeze(0)

    # ``from numpy import *`` below also exports ``min``; call the Python
    # builtin explicitly because this is a comparison, not an axis reduction.
    top_k = int(builtins.min(top_k, logits.size(-1)))
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k, dim=-1)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
        indices_to_remove.scatter_(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits.squeeze(0) if squeeze else logits


@torch.no_grad()
def sample_from_model(
    model,
    x,
    steps,
    points=None,
    variables=None,
    temperature=1.0,
    sample=False,
    top_k=0.0,
    top_p=0.0,
):
    """
    Take a conditioning sequence of indices in x (shape (1, t)) and predict
    the next token, feeding predictions back in each step. Quadratic in
    `steps` (recomputes attention over the growing sequence every step);
    fine at the short equation-string lengths used here.
    """
    block_size = model.get_block_size()
    model.eval()
    for _ in range(steps):
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:]
        logits, _ = model(x_cond, points=points, variables=variables)
        logits = logits[:, -1, :] / temperature
        logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
        probs = F.softmax(logits, dim=-1)
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        x = torch.cat((x, ix), dim=1)

    return x


# ---- safe math overrides used by eval() below (see module docstring) ----


def divide(x, y):
    x = np.nan_to_num(x)
    y = np.nan_to_num(y)
    return np.divide(x, np.maximum(y, 1.0))


def sqrt(x):
    x = np.nan_to_num(x)
    x = np.abs(x)
    return np.sqrt(x)


def log(x, eps=1e-5):
    x = np.nan_to_num(x)
    x = np.sqrt(x * x + eps)
    return np.log(x)


def exp(x, eps=1e-5):
    x = np.nan_to_num(x)
    return np.exp(x)


def relativeErr(y, yHat, eps=1e-5):
    """Relative mean squared error: mean((yHat - y)^2) / ||y||."""
    yHat = np.reshape(yHat, [1, -1])[0]
    y = np.reshape(y, [1, -1])[0]
    if len(y) > 0 and len(y) == len(yHat):
        err = (yHat - y) ** 2 / np.linalg.norm(y + eps)
    else:
        err = 100
    return np.mean(err)


def points_tensor_from_xy(X, Y, num_vars, num_ys, max_points, threshold=(-1000, 1000)):
    """
    Pack a point cloud (X: list of num_vars-length x-vectors, Y: list of
    scalars/num_ys-length y-vectors) into the fixed-width
    `[num_vars+num_ys, max_points]` tensor `tNet` expects: pads/truncates
    each point to (num_vars, num_ys), pads the point count to `max_points`
    with zero-columns (or truncates extra points beyond it), and clips
    NaN/Inf to `threshold`. Shared by `CharDataset.__getitem__` (training)
    and `kd_symbolicgpt.py` (real-data inference), so both build this
    tensor identically.
    """
    points = torch.zeros(num_vars + num_ys, max_points, dtype=torch.float32)
    if max_points <= 0:
        return points

    # The original implementation allocated one tiny tensor per point.  That
    # path dominated DataLoader time for large regression datasets.  Normalize
    # the whole point cloud with NumPy and transfer it to torch once instead.
    try:
        x_array = np.asarray(X, dtype=np.float32)
        y_array = np.asarray(Y, dtype=np.float32)
        if x_array.ndim == 1:
            x_array = x_array.reshape(-1, 1)
        else:
            x_array = x_array.reshape(x_array.shape[0], -1)
        if y_array.ndim == 1:
            y_array = y_array.reshape(-1, 1)
        else:
            y_array = y_array.reshape(y_array.shape[0], -1)

        n_points = builtins.min(len(x_array), len(y_array), max_points)
        packed = np.zeros((n_points, num_vars + num_ys), dtype=np.float32)
        x_width = builtins.min(num_vars, x_array.shape[1])
        y_width = builtins.min(num_ys, y_array.shape[1])
        packed[:, :x_width] = x_array[:n_points, :x_width]
        packed[:, num_vars : num_vars + y_width] = y_array[:n_points, :y_width]
        packed = np.nan_to_num(
            packed,
            nan=float(threshold[1]),
            posinf=float(threshold[1]),
            neginf=float(threshold[0]),
        )
        points[:, :n_points] = torch.from_numpy(packed.T.copy())
        return points
    except (TypeError, ValueError):
        # Preserve support for irregular legacy inputs.  Benchmark datasets
        # use dense numeric arrays and therefore take the vectorized path.
        pass

    for pt_idx, xy in enumerate(zip(X, Y)):
        if pt_idx >= max_points:
            break
        x = [xy[0]] if isinstance(xy[0], (float, np.floating)) else list(xy[0])
        x = x + [0] * builtins.max(num_vars - len(x), 0)
        y = [xy[1]] if isinstance(xy[1], (float, np.floating)) else list(xy[1])
        y = y + [0] * builtins.max(num_ys - len(y), 0)
        p = torch.tensor(x + y, dtype=torch.float32)
        points[:, pt_idx] = torch.nan_to_num(
            p, nan=threshold[1], posinf=threshold[1], neginf=threshold[0]
        )
    return points


class CharDataset(Dataset):
    def __init__(
        self,
        data,
        block_size,
        chars,
        numVars,
        numYs,
        numPoints,
        target="EQ",
        addVars=False,
        const_range=(-0.4, 0.4),
        xRange=(-3.0, 3.0),
        decimals=4,
        augment=False,
        cache_tensors=True,
    ):

        vocab_size = len(chars)

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

        self.numVars = numVars
        self.numYs = numYs
        self.numPoints = numPoints

        # padding token
        self.paddingToken = "_"
        self.paddingID = self.stoi[self.paddingToken]
        self.threshold = [-1000, 1000]

        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data  # list of examples (JSON strings or dicts)
        self.target = target
        self.addVars = addVars

        self.const_range = const_range
        self.xRange = xRange
        self.decimals = decimals
        self.augment = augment
        # The synthetic corpus is immutable when augmentation is disabled.
        # Cache the encoded sequence and packed point cloud after their first
        # use so 50-epoch runs do not rebuild them 50 times in Python.
        self._tensor_cache = [None] * (len(data) - 1) if cache_tensors and not augment else None

    def __len__(self):
        return len(self.data) - 1

    def __getitem__(self, idx):
        if self._tensor_cache is not None and self._tensor_cache[idx] is not None:
            return self._tensor_cache[idx]

        chunk = self.data[idx]

        if isinstance(chunk, str):
            chunk = json.loads(chunk)
        else:
            chunk = dict(chunk)  # copy so augmentation below doesn't mutate the corpus

        eq = chunk[self.target]
        variables = re.finditer(r"x[\d]+", eq)
        numVars = 0
        for v in variables:
            v = int(v.group(0).strip("x"))
            if v > numVars:
                numVars = v

        if self.target == "Skeleton" and self.augment:
            threshold = 5000
            cleanEqn = ""
            for ch in eq:
                if ch == "C":
                    ch = "{}".format(np.random.uniform(self.const_range[0], self.const_range[1]))
                cleanEqn += ch

            nPoints = np.random.randint(*self.numPoints)
            try:
                X, y = sample_points_for_equation(
                    cleanEqn,
                    n_points=nPoints,
                    n_vars=self.numVars,
                    decimals=self.decimals,
                    min_x=self.xRange[0],
                    max_x=self.xRange[1],
                )

                y = [e if abs(e) < threshold else np.sign(e) * threshold for e in y]
                conditions = (
                    (np.isnan(y).any() or np.isinf(y).any())
                    or len(y) == 0
                    or (abs(min(y)) > threshold or abs(max(y)) > threshold)
                )
                if not conditions:
                    chunk["X"], chunk["Y"] = X, y
            except Exception:
                pass  # fall back to the original support points for this example

        # encode every character in the equation to an integer; < is SOS, > is EOS
        if self.addVars:
            dix = [self.stoi[s] for s in "<" + str(numVars) + ":" + eq + ">"]
        else:
            dix = [self.stoi[s] for s in "<" + eq + ">"]
        inputs = dix[:-1]
        outputs = dix[1:]

        paddingSize = builtins.max(self.block_size - len(inputs), 0)
        paddingList = [self.paddingID] * paddingSize
        inputs += paddingList
        outputs += paddingList

        inputs = inputs[: self.block_size]
        outputs = outputs[: self.block_size]

        points = points_tensor_from_xy(
            chunk["X"],
            chunk["Y"],
            self.numVars,
            self.numYs,
            self.numPoints[1] - 1,
            threshold=self.threshold,
        )

        inputs = torch.tensor(inputs, dtype=torch.long)
        outputs = torch.tensor(outputs, dtype=torch.long)
        numVars = torch.tensor(numVars, dtype=torch.long)
        item = (inputs, outputs, points, numVars)
        if self._tensor_cache is not None:
            self._tensor_cache[idx] = item
        return item


def sample_points_for_equation(eq, n_points=2, n_vars=3, decimals=4, min_x=0, max_x=3):
    """Sample `n_points` random x-locations in [min_x, max_x]^n_vars and
    evaluate `eq` (a string using x1..x_n_vars) at each -- used both to
    build the synthetic pretraining corpus and by `CharDataset`'s
    on-the-fly augmentation."""
    if isinstance(min_x, (list, tuple, np.ndarray)):
        lower = np.asarray(min_x, dtype=float)
        upper = np.asarray(max_x, dtype=float)
        interval_indices = np.random.randint(len(lower), size=(n_points, n_vars))
        lows = lower[interval_indices]
        highs = upper[interval_indices]
        X = lows + np.random.random((n_points, n_vars)) * (highs - lows)
    else:
        X = np.random.uniform(min_x, max_x, size=(n_points, n_vars))
    X = np.round(X, decimals)

    Y = evaluate_expression(eq, X)
    Y = np.round(Y, decimals)
    return X.tolist(), Y.tolist()


_EXPRESSION_FUNCTIONS = {
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "sqrt": sqrt, "log": log, "exp": exp,
    "abs": np.abs, "Abs": np.abs,
}


@lru_cache(maxsize=256)
def _compile_expression(expression, num_vars, allow_constants=False):
    """Accept only scalar arithmetic in the generated equation grammar."""
    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, TypeError, ValueError) as exc:
        raise ValueError("Invalid SymbolicGPT expression syntax") from exc
    names = {f"x{i + 1}" for i in range(num_vars)}
    if allow_constants:
        names.add("C")
    pending = [tree.body]
    while pending:
        node = pending.pop()
        if isinstance(node, ast.BinOp) and isinstance(
            node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
        ):
            pending.extend([node.left, node.right])
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            pending.append(node.operand)
        elif isinstance(node, ast.Name) and node.id in names:
            continue
        elif (isinstance(node, ast.Constant)
              and type(node.value) in (builtins.int, builtins.float)
              and math.isfinite(node.value)):
            continue
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
              and node.func.id in _EXPRESSION_FUNCTIONS
              and len(node.args) == 1 and not node.keywords):
            pending.extend(node.args)
        else:
            raise ValueError("Unsupported SymbolicGPT expression grammar or variable")
    return compile(tree, "<SymbolicGPT expression>", "eval")


def _expression_namespace(X):
    """Build the restricted vector namespace used for generated equations."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    namespace = {f"x{i + 1}": X[:, i] for i in range(X.shape[1])}
    namespace.update(_EXPRESSION_FUNCTIONS)
    return X, namespace


def evaluate_expression(expression, X):
    """Evaluate an internally generated expression over all rows of ``X``.

    SymbolicGPT previously converted every row to a new expression string and
    called ``eval`` once per point.  Supplying vector-valued ``x1`` ... ``xn``
    arrays performs the same elementwise arithmetic in one NumPy call.
    """
    X, namespace = _expression_namespace(X)
    code = _compile_expression(expression, X.shape[1])
    with np.errstate(all="ignore"):
        values = eval(code, {"__builtins__": {}}, namespace)
    if np.iscomplexobj(values):
        raise ValueError("SymbolicGPT expressions must return real predictions")
    values = np.asarray(values, dtype=float)
    if values.ndim == 0:
        values = np.full(X.shape[0], float(values), dtype=float)
    if values.shape != (X.shape[0],):
        raise ValueError("SymbolicGPT expressions must return one prediction per input row")
    return values


def _constants_loss(constants, skeleton_eqn, X, Y):
    """Mean relative error of `skeleton_eqn` (with `C` placeholders filled
    in from `constants`, in order) against (X, Y). Minimized by
    `fit_constants` via `scipy.optimize.minimize`."""
    try:
        if not np.isfinite(constants).all():
            return float("inf")
        eq = skeleton_eqn.replace("C", "({})").format(*constants)
        y_hat = evaluate_expression(eq, X)
    except Exception:
        return float("inf")

    y = np.asarray(Y, dtype=float).reshape(-1)
    if len(y_hat) != len(y):
        return float("inf")
    if len(y) == 0 or not np.isfinite(y).all() or not np.isfinite(y_hat).all():
        return float("inf")

    # Match the old point-at-a-time ``relativeErr`` denominator: for a
    # scalar target np.linalg.norm(y + eps) is abs(y + eps).
    denominator = np.abs(y + 1e-5)
    denominator = np.maximum(denominator, np.finfo(float).eps)
    with np.errstate(over="ignore", invalid="ignore"):
        loss = float(np.mean((y_hat - y) ** 2 / denominator))
    return loss if np.isfinite(loss) else float("inf")


def fit_constants(skeleton_eqn, X, Y):
    """
    Fit the `C` placeholders in `skeleton_eqn` to (X, Y) by minimizing
    `_constants_loss`. Returns (concrete_equation_str, final_loss).
    Raises ValueError for invalid grammar, nonfinite fitted constants/loss,
    or predictions that are not finite on every training row.
    """
    X, _ = _expression_namespace(X)
    _compile_expression(skeleton_eqn, X.shape[1], allow_constants=True)
    num_constants = skeleton_eqn.count("C")
    concrete_eqn = skeleton_eqn
    if num_constants:
        c0 = [1.0] * num_constants
        result = minimize(_constants_loss, c0, args=(skeleton_eqn, X, Y))
        if not np.isfinite(result.x).all() or not np.isfinite(result.fun):
            raise ValueError("SymbolicGPT constant fitting returned nonfinite constants or loss")
        concrete_eqn = skeleton_eqn.replace("C", "({})").format(*result.x)
    # Recheck the exported expression over the complete data, not just the
    # optimizer's cached objective or a subset of finite predictions.
    loss = _constants_loss([], concrete_eqn, X, Y)
    if not np.isfinite(loss):
        raise ValueError("SymbolicGPT candidate has nonfinite predictions or loss")
    return concrete_eqn, loss
