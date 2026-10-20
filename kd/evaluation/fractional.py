"""Observation-only PDE diagnostics for the FDE fields; no fractional truth."""

import copy
from dataclasses import replace

import numpy as np


def is_pde_diagnostic(dataset):
    return bool(getattr(dataset, "protocol_metadata", {}).get("pde_protocol_version"))


def method_view(dataset, model):
    """Let the implicit method search a relation on the same integer derivatives."""
    if model != "implicit_pde" or not is_pde_diagnostic(dataset):
        return dataset
    result = copy.copy(dataset)
    result.equation_targets = [
        replace(target, kind="implicit") for target in dataset.equation_targets
    ]
    return result


def temporal_views(dataset, test_size):
    """Disjoint time blocks, with two unused frames and >=3 test frames.

    All PDE methods use these same observations. Derivatives are estimated only
    after slicing, so neither a stencil nor a rollout initial frame crosses the
    partition. The minimum test support also covers the 15-frame FDE fields.
    """
    from kd.benchmark_execution import SkipCase
    from kd.benchmark_preparation import slice_pde_time

    if isinstance(test_size, bool) or not np.isfinite(test_size) or not 0 < test_size < 1:
        raise ValueError("FDE PDE test_size must be finite and in (0,1)")
    nt = len(dataset.t)
    n_test = max(3, int(np.ceil(nt * test_size)))
    test_start, train_stop = nt - n_test, nt - n_test - 2
    if train_stop < 5:
        raise SkipCase(
            "insufficient_support: FDE PDE needs five train and three test frames plus a two-frame gap"
        )
    train = slice_pde_time(dataset, 0, train_stop)
    test = slice_pde_time(dataset, test_start, nt)
    return (
        train,
        test,
        {
            "kind": "temporal_block_with_support_gap",
            "support_gap_frames": 2,
            "train_frame_stop": train_stop,
            "test_frame_start": test_start,
            "n_train_frames": train_stop,
            "n_test_frames": n_test,
            "train_time_range": [float(train.t[0]), float(train.t[-1])],
            "test_time_range": [float(test.t[0]), float(test.t[-1])],
            "generalization": "within_record_integer_equation_residual; fractional_operator_not_evaluated",
        },
    )
