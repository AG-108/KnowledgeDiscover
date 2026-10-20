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

import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ._advanced import EVOLUTION_SPECS, STEADY_SPECS, declared_targets, load_evolution, load_steady
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
from ._closure import load_pyqg
from ._domain import FIELD_SPECS, load_domain_field
from ._file_sr import FILE_SR_NAMES, load_file_sr
from ._fractional import FDE_PDE_SPECS, FDE_SPECS, load_fde_pde, load_fde_response
from ._green import GREEN_SPECS, load_green
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
TLC_STEADY_STATE_FILES = (
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

# Steady observations now have their own task; geometry may remain source-blocked.
TLC_UNSUPPORTED_STEADY_STATE = ()

# Kept as an inspection API; Expression1 now uses the checked evolution loader.
KNOWN_BROKEN_DATASETS = frozenset()


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
        dataset = load_csv_tlc(
            str(csv_path), equation_name=equation_name, return_dataset=True,
            **{"grid_if_cartesian": True, **kwargs},
        )
        labels = list(getattr(dataset, "state_vars", ["u"]))
        parsed = [re.fullmatch(r"([A-Za-z_]\w*)(?:\s+\(([^)]*)\))?", label) for label in labels]
        if not all(parsed):
            raise ValueError(f"Unrecognized source field labels: {labels}")
        names = [match.group(1) for match in parsed]
        if len(set(names)) != len(names):
            raise ValueError("Canonical field names collide")
        dataset.state_vars = names
        dataset.equation_targets = declared_targets(equation_name)
        dataset.protocol_metadata = {**getattr(dataset, "protocol_metadata", {}),
            "source_field_labels": labels,
            "field_units": {name: match.group(2) or "unknown" for name, match in zip(names, parsed)}}
        return dataset

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
    if _name in KNOWN_BROKEN_DATASETS or _name in EVOLUTION_SPECS:
        continue
    DATASET_REGISTRY[_name] = {"category": "pde", "loader": _make_pde_registry_loader(_name)}


def _make_domain_field_loader(name):
    def loader(**kwargs):
        if kwargs:
            raise ValueError(f"Unsupported field loader options: {sorted(kwargs)}")
        return load_domain_field(name)
    return loader


for _name, _spec in FIELD_SPECS.items():
    DATASET_REGISTRY[_name] = {"category": "pde", "family": _spec["family"],
                               "loader": _make_domain_field_loader(_name)}

for _name, _rel_path in _TLC_TIME_SERIES_FILES.items():
    DATASET_REGISTRY[_name] = {"category": "pde", "loader": _make_tlc_loader(_rel_path, _name)}

DATASET_REGISTRY["wdwake"] = {"category": "pde", "loader": _load_wdwake}
DATASET_REGISTRY["ball_drop"] = {"category": "ode", "loader": _load_ball_drop}
for _system, _family in (("oscillator", "oscillatory"), ("population", "population"),
                         ("chaotic", "chaotic"), ("rational", "rational"),
                         ("damped_oscillator", "damped_oscillatory"),
                         ("pendulum", "nonlinear_oscillatory"),
                         ("duffing", "nonlinear_oscillatory"),
                         ("van_der_pol", "nonlinear_oscillatory"),
                         ("sir", "epidemic"), ("fitzhugh_nagumo", "excitable"),
                         ("brusselator", "chemical"),
                         ("robertson", "stiff_chemical")):
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


def _advanced_loader(name, steady=False):
    def loader(**kwargs):
        if kwargs:
            raise ValueError(f"Unsupported checked loader options: {sorted(kwargs)}")
        return load_steady(name) if steady else load_evolution(name)
    return loader


for _name in EVOLUTION_SPECS:
    DATASET_REGISTRY[_name] = {"category": "pde", "loader": _advanced_loader(_name)}
for _name in STEADY_SPECS:
    DATASET_REGISTRY[_name] = {"category": "steady_pde", "loader": _advanced_loader(_name, True)}




def _green_loader(name):
    def loader(**kwargs):
        return load_green(name, **kwargs)
    return loader

def _file_loader(name):
    def loader(**kwargs):
        return load_file_sr(name, **kwargs)
    return loader


def _fde_loader(name):
    def loader(**kwargs):
        load = load_fde_pde if name in FDE_PDE_SPECS else load_fde_response
        return load(name, **kwargs)
    return loader


for _name in FDE_SPECS:
    DATASET_REGISTRY[_name] = {
        "category": "regression", "family": _name, "loader": _fde_loader(_name), "opt_in": True,
    }

for _name, _response in FDE_PDE_SPECS.items():
    DATASET_REGISTRY[_name] = {
        "category": "pde", "family": _response, "loader": _fde_loader(_name), "opt_in": True,
    }

for _name, _spec in GREEN_SPECS.items():
    DATASET_REGISTRY[_name] = {"category":"green", "family":_spec["family"],
                              "loader":_green_loader(_name), "opt_in":True}
for _label in FILE_SR_NAMES:
    _name = "dso_file_" + _label
    DATASET_REGISTRY[_name] = {"category":"regression","family":"DSO_file_"+_label.split("-")[0],
                             "loader":_file_loader(_name), "opt_in":True}
DATASET_REGISTRY["pyqg_subgrid"] = {"category":"regression","family":"pyqg",
                                  "loader":load_pyqg, "opt_in":True}

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
