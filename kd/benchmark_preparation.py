"""Shared PDE adapter preparation and contiguous temporal views."""

import copy

from kd.benchmark_execution import SkipCase


def prepare_pde(dataset, model_name):
    """Keep spatial/channel axes explicit; never silently flatten them."""
    import numpy as np

    from kd.benchmark_capabilities import validate_pde
    from kd.dataset import GridPDEDataset

    validate_pde(dataset, model_name)
    coords = [np.asarray(v) for v in dataset.coords_spatial.values()] + [np.asarray(dataset.t)]
    if model_name in {"pdenet", "pdefind_nd", "implicit_pde"}:
        return dataset
    u = dataset.usol if dataset.legacy else dataset.usol[0]
    result = GridPDEDataset(
        equation_name=dataset.equation_name,
        pde_data=None,
        x=coords[0],
        t=coords[-1],
        usol=u,
        domain={
            "x": (float(coords[0][0]), float(coords[0][-1])),
            "t": (float(coords[-1][0]), float(coords[-1][-1])),
        },
        epi=dataset.epi,
        legacy=True,
    )
    for attr in ("registry_name", "legacy_name", "sym_true", "rollout_metadata", "state_vars", "protocol_metadata", "equation_targets", "geometry", "region_ids"):
        if hasattr(dataset, attr):
            setattr(result, attr, copy.deepcopy(getattr(dataset, attr)))
    return result


def slice_pde_time(dataset, start, stop):
    """Clone a regular-grid dataset over an explicit contiguous time slice."""
    from kd.dataset import GridPDEDataset

    coords = {name: values.copy() for name, values in dataset.coords.items()}
    coords[dataset.time_var] = coords[dataset.time_var][start:stop]
    domain = copy.deepcopy(dataset.domain)
    if isinstance(domain, dict) and len(coords[dataset.time_var]):
        domain[dataset.time_var] = (
            float(coords[dataset.time_var][0]),
            float(coords[dataset.time_var][-1]),
        )
    result = GridPDEDataset(
        equation_name=dataset.equation_name,
        pde_data=None,
        coords=coords,
        usol=dataset.usol[..., start:stop].copy(),
        domain=domain,
        epi=dataset.epi,
        time_var=dataset.time_var,
        legacy=dataset.legacy,
    )
    for attr in ("registry_name", "legacy_name", "sym_true", "rollout_metadata", "state_vars", "protocol_metadata", "equation_targets", "geometry", "region_ids"):
        if hasattr(dataset, attr):
            setattr(result, attr, copy.deepcopy(getattr(dataset, attr)))
    return result


def temporal_block_holdout(dataset, test_size):
    """Use the final contiguous time block only for PDE evaluation."""
    nt = len(dataset.t)
    if not 0 < test_size < 1:
        raise ValueError("PDE temporal holdout needs 0 < test_size < 1")
    if nt < 5:
        raise SkipCase("PDE temporal holdout requires at least five time frames")
    n_test = min(nt - 3, max(2, round(nt * test_size)))
    split = nt - n_test
    # The evaluation view includes the last training frame as the rollout initial condition.
    return slice_pde_time(dataset, 0, split), slice_pde_time(dataset, split - 1, nt), split


