"""KD (Knowledge Discovery) package for PDE discovery.

This package provides tools for discovering governing equations of PDEs
using deep learning and symbolic regression approaches.
"""

__version__ = "0.1.0"

__pkg_name__ = "KD"

from .dataset import load_burgers_equation, load_kdv_equation, load_mat_file


# Plotting imports DISCOVER and its optional dependencies. Keep dataset-only
# commands (including benchmark --list/--dry-run) independent of those imports.
def __getattr__(name):
    if name in _model_exports:
        from importlib import import_module

        value = getattr(import_module(".model", __name__), name)
        globals()[name] = value
        return value
    if name in _submodules:
        from importlib import import_module

        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


_submodules = ["dataset", "model", "viz"]
_model_exports = ["DLGA", "KD_DLGA", "KD_DSCV", "KD_DSCV_SPR"]

__all__ = ["dataset", "model"] + [
    "load_burgers_equation",
    "load_mat_file",
    "load_kdv_equation",
]


def __dir__():
    return sorted(set(globals()) | set(_submodules) | set(_model_exports))
