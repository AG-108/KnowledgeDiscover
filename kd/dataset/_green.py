"""Observed forcing/response inverse operators; references are scoring only."""

import ast
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
PREFIX = "Discover_Green_function/dso/task/pde/"
FAMILIES = (
    "advection",
    "helmholtz",
    "negative_helmholtz",
    "laplace",
    "periodic_helmholtz",
    "jump_green",
)
GREEN_SPECS = {}
for family in FAMILIES:
    GREEN_SPECS["green_" + family] = {
        "path": PREFIX + ("data2/" if family == "jump_green" else "data/") + family + ".mat",
        "family": "green_" + family,
        "variant": "base",
    }
    if family != "advection":
        for noise in (1, 5, 10):
            GREEN_SPECS[f"green_{family}_noise_{noise}"] = {
                "path": f"{PREFIX}data2/{family}_noise_{noise}.mat",
                "family": "green_" + family,
                "variant": f"noise_label_{noise}",
                "noise_scale": None,
            }
for k in (6, 9):
    GREEN_SPECS[f"green_periodic_helmholtz_k{k}"] = {
        "path": f"{PREFIX}data/periodic_helmholtz_k={k}.mat",
        "family": "green_periodic_helmholtz",
        "variant": f"parameter_k{k}",
        "k": k,
    }

# Uniform declaration, independent of any reference string.
GREEN_GRAMMAR = ("add", "sub", "mul", "div", "sin", "cos", "exp", "abs", "const", "lt")


def trapezoid_weights(axis):
    axis = np.asarray(axis, dtype=float)
    if (
        axis.ndim != 1
        or len(axis) < 2
        or not np.isfinite(axis).all()
        or not (np.diff(axis) > 0).all()
    ):
        raise ValueError("Quadrature requires a finite strictly increasing one-dimensional axis")
    diff = np.diff(axis)
    weights = np.empty_like(axis)
    weights[0], weights[-1] = diff[0] / 2, diff[-1] / 2
    weights[1:-1] = (diff[:-1] + diff[1:]) / 2
    return weights


class GreenOperatorDataset:
    def __init__(self, name, x, xi, forcing, response, u_hom, *, reference=None, metadata=None):
        self.name = self.registry_name = name
        self.x, self.xi = np.asarray(x, float), np.asarray(xi, float)
        trapezoid_weights(self.x)
        self.weights = trapezoid_weights(self.xi)
        self.F, self.U = np.asarray(forcing, float), np.asarray(response, float)
        self.u_hom = np.asarray(u_hom, float)
        if self.F.ndim != 2 or self.F.shape[0] != len(self.xi) or self.F.shape[1] < 5:
            raise ValueError("F must have shape (n_xi, n_forcings>=5)")
        if self.U.shape != (len(self.x), self.F.shape[1]) or self.u_hom.shape != (len(self.x),):
            raise ValueError("U and U_hom must have shapes (n_x, n_forcings) and (n_x,)")
        if not all(np.isfinite(v).all() for v in (self.F, self.U, self.u_hom)):
            raise ValueError("Green observations must be finite")
        self.reference = reference  # Never passed to fit/task/constant optimizer.
        self.protocol_metadata = dict(metadata or {})
        self.target_name = "kernel"

    @property
    def kernel_inputs(self):
        x, xi = np.meshgrid(self.x, self.xi, indexing="ij")
        return np.column_stack((x.ravel(), xi.ravel()))

    def forward(self, kernel, columns=None, *, block_rows=256):
        kernel = np.asarray(kernel, float)
        if kernel.shape != (len(self.x), len(self.xi)):
            raise ValueError("Kernel orientation must be (n_x, n_xi)")
        if not np.isfinite(kernel).all():
            raise ValueError("Nonfinite kernel cannot produce a valid operator")
        forces = self.F if columns is None else self.F[:, columns]
        if not isinstance(block_rows, int) or block_rows < 1:
            raise ValueError("block_rows must be a positive integer")
        weighted = self.weights[:, None] * forces
        prediction = np.empty((len(self.x), forces.shape[1]))
        for start in range(0, len(self.x), block_rows):
            stop = min(start+block_rows, len(self.x))
            prediction[start:stop] = kernel[start:stop] @ weighted + self.u_hom[start:stop,None]
        return prediction

    def split(self, seed=0):
        # Column identity is consistent across clean/noise/parameter variants.
        ids = np.random.default_rng(seed).permutation(self.F.shape[1])
        n_test = max(1, int(np.ceil(len(ids) * 0.2)))
        n_val = max(1, int(np.ceil(len(ids) * 0.2)))
        return {
            "train": ids[n_test + n_val :].tolist(),
            "validation": ids[n_test : n_test + n_val].tolist(),
            "test": ids[:n_test].tolist(),
            "protocol": "whole_forcing_60_20_20",
            "seed": int(seed),
            "forcing_identity": "family_source_column_index; physical equivalence not certified",
            "forcing_ids": [
                self.protocol_metadata.get("family", self.name) + f"/column_{i:04d}"
                for i in range(self.F.shape[1])
            ],
        }

    def training_view(self, split):
        train, validation = split["train"], split["validation"]
        return {
            "kernel_inputs": self.kernel_inputs,
            "kernel_shape": (len(self.x), len(self.xi)),
            "weighted_forcing": self.weights[:, None] * self.F[:, train],
            "response": self.U[:, train] - self.u_hom[:, None],
            "validation_forcing": self.weights[:, None] * self.F[:, validation],
            "validation_response": self.U[:, validation] - self.u_hom[:, None],
        }


def load_green(name, *, root=None):
    from scipy.io import loadmat

    spec = GREEN_SPECS[name]
    d = loadmat((Path(root) if root is not None else ROOT) / spec["path"])
    required = {"X", "Y", "F", "U", "U_hom"}
    if not required <= d.keys():
        raise ValueError("Green MAT missing keys: " + str(sorted(required - d.keys())))
    if any(np.asarray(d[key]).ndim != 2 or np.asarray(d[key]).shape[1] != 1 for key in ("X","Y")):
        raise ValueError("Source Green coordinates must be column vectors, not mesh grids")
    x, xi = np.asarray(d["X"]).reshape(-1), np.asarray(d["Y"]).reshape(-1)
    hom = np.asarray(d["U_hom"])
    if hom.shape != (len(x), 1):
        raise ValueError("Source U_hom layout must be (n_x,1), no free test baseline")
    ref = str(d["ExactGreen"].reshape(-1)[0]) if "ExactGreen" in d else None
    return GreenOperatorDataset(
        name,
        x,
        xi,
        d["F"],
        d["U"],
        hom[:, 0],
        reference=ref,
        metadata={
            **spec,
            "kernel_axes": ["x", "xi"],
            "quadrature": "source_nonuniform_trapezoid",
            "homogeneous_source": "observed_U_hom; fixed for all forcings",
            "reference_status": "source_declared_not_certified" if ref else "missing",
            "units": "unknown",
            "boundary_conditions": "not independently certified",
            "grammar": list(GREEN_GRAMMAR),
            "operator_instance": name,
            "forcing_alignment": "source column index shared within family; noise changes F and U; source generator not retained",
        },
    )


def reference_kernel(expression, x, xi):
    """Small AST interpreter: no eval/sympify, arbitrary attributes or calls."""
    functions = {
        name: getattr(np, name) for name in ("sin", "cos", "exp", "sinh", "cosh", "abs", "sqrt")
    }
    binary = {
        ast.Add: np.add,
        ast.Sub: np.subtract,
        ast.Mult: np.multiply,
        ast.Div: np.divide,
        ast.Pow: np.power,
    }
    compare = {
        ast.Lt: np.less,
        ast.LtE: np.less_equal,
        ast.Gt: np.greater,
        ast.GtE: np.greater_equal,
    }
    grid_x, grid_y = np.meshgrid(x, xi, indexing="ij")

    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.Name) and node.id in {"x", "y"}:
            return grid_x if node.id == "x" else grid_y
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            return -visit(node.operand) if isinstance(node.op, ast.USub) else visit(node.operand)
        if isinstance(node, ast.BinOp) and type(node.op) in binary:
            return binary[type(node.op)](visit(node.left), visit(node.right))
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in compare:
            return compare[type(node.ops[0])](visit(node.left), visit(node.comparators[0]))
        if isinstance(node, ast.Call) and not node.keywords and len(node.args) == 1:
            func = node.func
            name = (
                func.id
                if isinstance(func, ast.Name)
                else (
                    func.attr
                    if isinstance(func, ast.Attribute)
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "np"
                    else None
                )
            )
            if name in functions:
                return functions[name](visit(node.args[0]))
        raise ValueError("Forbidden reference expression AST: " + type(node).__name__)

    tree = ast.parse(expression, mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 300:
        raise ValueError("Reference expression too large")
    with np.errstate(all="ignore"):
        result = np.broadcast_to(visit(tree.body), grid_x.shape).astype(float).copy()
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite reference kernel")
    return result
