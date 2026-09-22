from importlib import resources
from typing import Optional

import numpy as np

from ._base import (
    GridPDEDataset,
    ODEDataset,
    ScatterPDEDataset,
    SymbolicRegressionDataset,
    TabularRegressionDataset,
    load_ball_drop_dataset,
    load_burgers_equation,
    load_csv_tlc,
    load_cyt_dataset,
    load_cyt_flowfeature_raw,
    load_kdv_equation,
    load_mat_file,
    load_pde_dataset,
    load_rubber_dataset,
    load_solid_dif_dataset,
    load_solid_hardening_dataset,
    load_solid_strain_stress_dataset,
    load_vgs_dataset,
    load_wake_equation,
)
from ._pdeformer_sinus import (
    DedalusBackend,
    NumpySpectralBackend,
    PDESolverBackend,
    SinusPDESpec,
    load_pdeformer_sinus_benchmark,
)
from ._registry import PDE_REGISTRY, get_dataset_info, get_dataset_sym_true, list_available_datasets


def _tag_dataset(dataset: Optional[GridPDEDataset], name: str) -> Optional[GridPDEDataset]:
    """Attach registry metadata (registry_name / legacy_name) to dataset instances."""
    if dataset is None:
        return None

    setattr(dataset, "registry_name", name)
    info = PDE_REGISTRY.get(name, {})
    legacy_name = info.get("legacy_name", getattr(dataset, "legacy_name", None))
    if legacy_name is None:
        legacy_name = name
    setattr(dataset, "legacy_name", legacy_name)
    return dataset


def load_pde_grid(name: str, **kwargs) -> GridPDEDataset:
    """Load a registered PDE dataset through its specialized or declarative loader."""
    # Prefer dedicated loaders for datasets with custom parsing rules.
    if name == "kdv":
        return _tag_dataset(load_kdv_equation(), name)
    elif name == "burgers":
        return _tag_dataset(load_burgers_equation(), name)

    # Fall back to the declarative registry.
    info = get_dataset_info(name)

    # Dispatch according to the registered file layout.
    if "files" in info:  # Some datasets, such as Chafee-Infante, span multiple files.
        data_path = resources.files("kd.dataset.data")
        u = np.asarray(np.load(data_path / info["files"]["u"]), dtype=float)
        x = np.asarray(np.load(data_path / info["files"]["x"]), dtype=float).flatten()
        t = np.asarray(np.load(data_path / info["files"]["t"]), dtype=float).flatten()

        if u.shape == (len(t), len(x)):
            u = u.T

        dataset = GridPDEDataset(
            equation_name=name,
            pde_data=None,
            x=x,
            t=t,
            usol=u,
            domain=info.get("domain"),
            epi=kwargs.get("epi", 1e-3),
            legacy=True,
        )
        return _tag_dataset(dataset, name)
    elif info.get("file", "").endswith(".npy"):
        # Handle raw single-file NPY fields such as PDE_divide and PDE_compound.
        data_path = resources.files("kd.dataset.data")
        data = np.asarray(np.load(data_path / info["file"]), dtype=float)

        # Raw field matrices require coordinates synthesized from registry metadata.
        target_shape = info.get("shape")
        if target_shape:
            # Accept either (nx, nt) or its transpose when matching dimensions.
            if data.shape == tuple(reversed(target_shape)):
                data = data.T
            elif data.shape != target_shape:
                data = data.reshape(target_shape)
        shape = data.shape

        domain = info.get("domain") or {"x": (0.0, 1.0), "t": (0.0, 1.0)}
        x_bounds = domain.get("x", (0.0, 1.0))
        t_bounds = domain.get("t", (0.0, 1.0))

        nx, nt = shape
        x = np.linspace(x_bounds[0], x_bounds[1], nx)
        t = np.linspace(t_bounds[0], t_bounds[1], nt)

        dataset = GridPDEDataset(
            equation_name=name,
            pde_data=None,
            x=x,
            t=t,
            usol=data,
            domain=domain,
            epi=kwargs.get("epi", 1e-3),
            legacy=True,
        )
        return _tag_dataset(dataset, name)
    else:  # Load a single MAT-file dataset through its registered keys.
        dataset = load_pde_dataset(
            info["file"],
            equation_name=name,
            domain=info.get("domain"),
            **info.get("keys", {}),
            **kwargs,
        )
        return _tag_dataset(dataset, name)


from ._catalog import DATASET_REGISTRY, get_dataset_category, list_datasets, load_dataset

__all__ = [
    "GridPDEDataset",
    "ScatterPDEDataset",
    "SymbolicRegressionDataset",
    "ODEDataset",
    "TabularRegressionDataset",
    "load_pde_grid",  # Public compatibility alias for the unified loader.
    "load_burgers_equation",
    "load_mat_file",
    "load_kdv_equation",
    "load_pde_dataset",
    "load_csv_tlc",
    "load_wake_equation",
    "load_ball_drop_dataset",
    "load_rubber_dataset",
    "load_cyt_dataset",
    "load_cyt_flowfeature_raw",
    "load_solid_dif_dataset",
    "load_solid_strain_stress_dataset",
    "load_solid_hardening_dataset",
    "load_vgs_dataset",
    "SinusPDESpec",
    "PDESolverBackend",
    "NumpySpectralBackend",
    "DedalusBackend",
    "load_pdeformer_sinus_benchmark",
    "load_dataset",  # Export the unified dataset entry point.
    "list_datasets",
    "get_dataset_category",
    "DATASET_REGISTRY",
    "list_available_datasets",  # Export registry inspection helpers.
    "get_dataset_sym_true",  # Export registry inspection helpers.
    "Burgers_equation_shock",
    "KdV_equation",
    "KdV_equation_sine",
    "Chaffee_Infante_equation",
    "KG_equation",
    "Allen_Cahn_equation",
    "Convection_diffusion_equation_solution",
    "Convection_diffusion_equation_simulation",
    "Wave_equation",
    "KS_equation",
    "Beam_equation",
    "Heat_equation",
    "Heat_equation_sin",
    "Diffusion_equation",
    "Parametric_convection_diffusion",
    "Parametric_Burgers_equation",
    "Parametric_wave_equation",
    "Burgers_2D",
]
