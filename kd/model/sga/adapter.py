"""Adapters bridging `GridPDEDataset` with the SGA solver stack."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - purely for type checking
    from kd.dataset import GridPDEDataset


@dataclass
class SGADataAdapter:
    """Translate a :class:`~kd.dataset.GridPDEDataset` into SolverConfig kwargs."""

    dataset: "GridPDEDataset"

    def to_solver_kwargs(self) -> Dict[str, Any]:
        """Return keyword arguments that prime ``SolverConfig`` for in-memory data."""
        # Delayed import keeps the dependency optional at runtime.
        from kd.dataset import GridPDEDataset  # pylint: disable=import-error, cyclic-import

        if not isinstance(self.dataset, GridPDEDataset):
            raise TypeError("SGADataAdapter expects a GridPDEDataset instance")

        x = np.asarray(self.dataset.x, dtype=float).flatten()
        t = np.asarray(self.dataset.t, dtype=float).flatten()
        u = np.asarray(self.dataset.usol, dtype=float)

        if u.shape != (len(x), len(t)):
            raise ValueError(
                "Inconsistent dataset dimensions: expected usol shape "
                f"({len(x)}, {len(t)}) but got {u.shape}"
            )

        return {
            "u_data": u,
            "x_data": x,
            "t_data": t,
            "problem_name": getattr(self.dataset, "equation_name", "custom_dataset")
            or "custom_dataset",
        }


def symbolic_rhs(default_names, trees, coefficients):
    """Export fitted SGA coefficients and spatial tree operators as an RHS.

    The native visualize() string only displays generated terms. Default terms
    and fitted weights must be retained to evaluate the discovered equation.
    Division retains strict symbolic semantics in the common evaluator.
    """
    import sympy as sp

    x = sp.Symbol("x")
    field = sp.Function("u")(x)
    leaves = {
        "u": field,
        "x": x,
        "0": sp.Integer(0),
        "ux": sp.diff(field, x),
        "uxx": sp.diff(field, x, 2),
        "uxxx": sp.diff(field, x, 3),
    }

    def node(tree, depth=0, index=0):
        item = tree[depth][index]
        if item.child_num == 0:
            if item.name not in leaves:
                raise ValueError(f"Unsupported SGA leaf: {item.name}")
            return leaves[item.name]
        args = [node(tree, depth + 1, item.child_st + i) for i in range(item.child_num)]
        if item.name in {"d", "d^2"}:
            if args[1] != x:
                raise ValueError("SGA RHS export supports spatial derivatives only")
            return sp.diff(args[0], x, 1 if item.name == "d" else 2)
        operators = {
            "+": lambda a, b: a + b,
            "-": lambda a, b: a - b,
            "*": lambda a, b: a * b,
            "/": lambda a, b: a / b,
            "^2": lambda a: a**2,
            "^3": lambda a: a**3,
        }
        if item.name not in operators:
            raise ValueError(f"Unsupported SGA operator: {item.name}")
        return operators[item.name](*args)

    terms = [leaves[name] for name in default_names] + [node(tree.tree) for tree in trees]
    coef = np.asarray(coefficients, dtype=float).reshape(-1)
    if len(terms) != len(coef) or not np.isfinite(coef).all():
        raise ValueError("SGA terms and finite fitted coefficients must align")
    expression = sum(
        (sp.Float(c, 17) * term for c, term in zip(coef, terms) if c != 0), sp.Integer(0)
    )
    replacements = {
        derivative: sp.Symbol("u_" + "x" * sum(order for _, order in derivative.variable_count))
        for derivative in expression.atoms(sp.Derivative)
        if derivative.expr == field and all(axis == x for axis, _ in derivative.variable_count)
    }
    expression = expression.xreplace(replacements).subs(field, sp.Symbol("u"))
    if expression.has(sp.Derivative):
        raise ValueError("Unresolved SGA derivative in exported equation")
    return "u_t = " + str(expression)


def fitted_rhs(best_equation, context):
    """Resolve native STRidge weights using the training context only."""
    from copy import deepcopy

    from .sgapde.pde import evaluate_mse

    terms, coefficients = evaluate_mse(deepcopy(best_equation.elements), context, True)
    return symbolic_rhs(context.default_names, terms, coefficients)
