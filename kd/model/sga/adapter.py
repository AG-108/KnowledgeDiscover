"""Adapters bridging `GridPDEDataset` with the SGA solver stack."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, TYPE_CHECKING

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
            "problem_name": getattr(self.dataset, "equation_name", "custom_dataset") or "custom_dataset",
        }
