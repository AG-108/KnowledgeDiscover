"""KD (Knowledge Discovery) package for PDE discovery.

This package provides tools for discovering governing equations of PDEs
using deep learning and symbolic regression approaches.
"""

__version__ = "0.1.0"

__pkg_name__ = "KD"

from .dataset import load_burgers_equation, load_mat_file


# Plotting imports DISCOVER and its optional dependencies. Keep dataset-only
# commands (including benchmark --list/--dry-run) independent of those imports.
def __getattr__(name):
    if name in _submodules:
        from importlib import import_module

        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


_submodules = ["dataset", "model", "viz"]

__all__ = _submodules + [
    "load_burgers_equation",
    "load_mat_file",
    "load_kdv_equation",
    "DLGA",
    "KD_DLGA",
    "KD_DSCV",
    "KD_DSCV_Pinn",
    "viz",
]
