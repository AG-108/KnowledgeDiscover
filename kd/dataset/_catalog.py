"""
Unified dataset catalog.

Every dataset already wired into `kd.dataset` (PDE / ODE / regression) is
reachable through a single dispatch table, `DATASET_REGISTRY`, and a single
entry point, `load_dataset(name)`, regardless of which underlying loader
function or file format actually produces it.

This module does not implement any parsing itself: it only maps a dataset
name to the existing loader that already knows how to build it (`load_pde_grid`,
`load_csv_tlc`, `load_wake_equation`, `load_ball_drop_dataset`,
`load_rubber_dataset`, `load_cyt_dataset`, the solid-constitutive loaders,
or `load_vgs_dataset`).
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ._base import (
    _CYT_CASE_ALIASES,
    _VGS_CASE_CONFIG,
    load_ball_drop_dataset,
    load_csv_tlc,
    load_cyt_dataset,
    load_rubber_dataset,
    load_solid_dif_dataset,
    load_solid_hardening_dataset,
    load_solid_strain_stress_dataset,
    load_vgs_dataset,
    load_wake_equation,
)
from ._pdeformer_sinus import load_pdeformer_sinus_benchmark
from ._registry import PDE_REGISTRY

# NOTE: `load_pde_grid` lives in `kd/dataset/__init__.py` (not `_base.py`), so it
# is imported lazily inside `_make_pde_registry_loader` below to avoid a circular
# import (this module is itself imported from `__init__.py`).

_TLC_DIR = Path(__file__).resolve().parent / "TLC"
_WDWAKE_DIR = Path(__file__).resolve().parent / "WDwake"

# name -> path (relative to TLC/) for CSVs that carry a "... @ t=..." time axis,
# i.e. the ones `load_csv_tlc` can already parse.
_TLC_TIME_SERIES_FILES: Dict[str, str] = {
    "tlc_burgers1d": "burgers/burgers1d.csv",
    "tlc_burgers2d": "burgers/burgers2d.csv",
    "tlc_heat_complex": "heat/heat_complex.csv",
    "tlc_heat_darcy": "heat/heat_darcy.csv",
    "tlc_heat_longtime": "heat/heat_longtime.csv",
    "tlc_heat_multiscale": "heat/heat_multiscale.csv",
    "tlc_heat_multiscale_lesspoints": "heat/heat_multiscale_lesspoints.csv",
    "tlc_ns_long": "ns/ns_long.csv",
    "tlc_wave_darcy": "others/wave_darcy.csv",
}

# TLC files that only hold a single steady-state / parameter-indexed snapshot
# (e.g. "u @ Re=100" or no "@" suffix at all) instead of a time series.
# GridPDEDataset/ScatterPDEDataset both require a time axis, so these are not
# reachable through `load_dataset` yet.
TLC_UNSUPPORTED_STEADY_STATE = (
    "ns/ns2d.csv",
    "ns/ns_0_obstacle.csv",
    "ns/ns_4_obstacle.csv",
    "others/lid_driven.csv",
    "poisson/poisson1_cg_data.csv",
    "poisson/poisson_3d.csv",
    "poisson/poisson_boltzmann2d.csv",
    "poisson/poisson_classic.csv",
    "poisson/poisson_manyarea.csv",
)

# PDE_REGISTRY entries known to be broken and deliberately excluded from the
# unified catalog: `advection_diffusion` -> Advection_diffusion.mat does not
# actually contain 'x'/'t'/'u' keys (only a single 'Expression1' array of
# shape (634644, 1) whose format has not been reverse-engineered yet), so
# `load_pde_dataset()` raises KeyError internally and silently returns None.
# Remove a name from this set once its underlying loader is fixed.
KNOWN_BROKEN_DATASETS = frozenset({"advection_diffusion"})


def _make_pde_registry_loader(name: str) -> Callable[..., Any]:
    def _loader(**kwargs):
        from . import (
            load_pde_grid,
        )  # deferred: load_pde_grid lives in __init__.py, which imports this module

        return load_pde_grid(name, **kwargs)

    return _loader


def _make_tlc_loader(rel_path: str, equation_name: str) -> Callable[..., Any]:
    def _loader(**kwargs):
        csv_path = _TLC_DIR / rel_path
        return load_csv_tlc(
            str(csv_path), equation_name=equation_name, return_dataset=True, **kwargs
        )

    return _loader


def _load_wdwake(**kwargs):
    return load_wake_equation(str(_WDWAKE_DIR), ["TI8_U.npy", "TI8_V.npy"])


def _load_ball_drop(**kwargs):
    return load_ball_drop_dataset(**kwargs)


def _make_ode_core_loader(system):
    def _loader(**kwargs):
        from ._ode_core import generate_ode_core
        return generate_ode_core(system, **kwargs)
    return _loader


def _load_rubber_train(**kwargs):
    return load_rubber_dataset(split="train", **kwargs)


def _load_rubber_test(**kwargs):
    return load_rubber_dataset(split="test", **kwargs)


def _make_cyt_loader(case_alias: str) -> Callable[..., Any]:
    def _loader(**kwargs):
        return load_cyt_dataset(case=case_alias, **kwargs)

    return _loader


def _load_solid_dif(**kwargs):
    return load_solid_dif_dataset(**kwargs)


def _load_solid_strain_stress(**kwargs):
    return load_solid_strain_stress_dataset(**kwargs)


def _load_solid_hardening(**kwargs):
    return load_solid_hardening_dataset(**kwargs)


def _make_vgs_loader(case: str, window: str) -> Callable[..., Any]:
    def _loader(**kwargs):
        return load_vgs_dataset(case=case, window=window, **kwargs)

    return _loader


DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {}

for _name in PDE_REGISTRY:
    if _name in KNOWN_BROKEN_DATASETS:
        continue
    DATASET_REGISTRY[_name] = {"category": "pde", "loader": _make_pde_registry_loader(_name)}

for _name, _rel_path in _TLC_TIME_SERIES_FILES.items():
    DATASET_REGISTRY[_name] = {"category": "pde", "loader": _make_tlc_loader(_rel_path, _name)}

DATASET_REGISTRY["wdwake"] = {"category": "pde", "loader": _load_wdwake}
DATASET_REGISTRY["ball_drop"] = {"category": "ode", "loader": _load_ball_drop}
for _system, _family in (("oscillator", "oscillatory"), ("population", "population"),
                         ("chaotic", "chaotic"), ("rational", "rational")):
    DATASET_REGISTRY[f"ode_core_{_system}"] = {
        "category": "ode", "family": _family,
        "loader": _make_ode_core_loader(_system),
    }
DATASET_REGISTRY["rubber_train"] = {"category": "regression", "loader": _load_rubber_train}
DATASET_REGISTRY["rubber_test"] = {"category": "regression", "loader": _load_rubber_test}

for _alias in _CYT_CASE_ALIASES:
    DATASET_REGISTRY[f"cyt_{_alias}"] = {
        "category": "regression",
        "loader": _make_cyt_loader(_alias),
    }

DATASET_REGISTRY["solid_dif"] = {"category": "regression", "loader": _load_solid_dif}
DATASET_REGISTRY["solid_strain_stress"] = {
    "category": "regression",
    "loader": _load_solid_strain_stress,
}
DATASET_REGISTRY["solid_hardening"] = {"category": "regression", "loader": _load_solid_hardening}

for _case, _window in _VGS_CASE_CONFIG:
    DATASET_REGISTRY[f"vgs_{_case}_{_window}"] = {
        "category": "pde",
        "loader": _make_vgs_loader(_case, _window),
    }

# NOTE: unlike every other entry, this one returns a *list* of
# GridPDEDataset (one freshly-generated random PDE instance per entry),
# not a single dataset object -- see `load_pdeformer_sinus_benchmark`'s
# docstring. It is also the only entry that performs a nontrivial
# numerical solve at load time rather than just parsing a file.
DATASET_REGISTRY["pdeformer_sinus"] = {
    "category": "pde",
    "loader": load_pdeformer_sinus_benchmark,
}


def load_dataset(name: str, **kwargs) -> Any:
    """
    Unified single entry point for loading any dataset registered in
    `kd.dataset`, regardless of its underlying task category
    (PDE / ODE / regression) or file format.

    Parameters
    ----------
    name : str
        Dataset name. See `list_datasets()` for all available names.
    **kwargs :
        Forwarded to the underlying loader (e.g. `epi=` for PDE datasets,
        `exclude=` for `ball_drop`).

    Returns
    -------
    A single dataset object for every entry EXCEPT `"pdeformer_sinus"`,
    which returns a `list[GridPDEDataset]` (a freshly-generated batch of
    random PDE instances; see `load_pdeformer_sinus_benchmark`).

    Examples
    --------
    >>> load_dataset('kdv')
    >>> load_dataset('tlc_heat_complex')
    >>> load_dataset('ball_drop')
    >>> load_dataset('rubber_train')
    >>> load_dataset('pdeformer_sinus', n_pde=5)  # returns a list
    """
    if name in KNOWN_BROKEN_DATASETS:
        raise ValueError(
            f"Dataset {name!r} is known to be broken and excluded from the unified "
            f"catalog (see KNOWN_BROKEN_DATASETS in kd.dataset._catalog for details)."
        )
    if name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset: {name!r}. Available datasets: {sorted(DATASET_REGISTRY)}"
        )
    return DATASET_REGISTRY[name]["loader"](**kwargs)


def list_datasets(category: Optional[str] = None) -> List[str]:
    """
    List all dataset names registered in the unified catalog.

    Parameters
    ----------
    category : {"pde", "ode", "regression"}, optional
        If given, only return names belonging to that category.
    """
    if category is None:
        return sorted(DATASET_REGISTRY)
    return sorted(n for n, info in DATASET_REGISTRY.items() if info["category"] == category)


def get_dataset_category(name: str) -> str:
    """Return the task category ("pde"/"ode"/"regression") for a registered dataset name."""
    if name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset: {name!r}. Available datasets: {sorted(DATASET_REGISTRY)}"
        )
    return DATASET_REGISTRY[name]["category"]
