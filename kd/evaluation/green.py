"""Operator response and scoring-only reference diagnostics."""

import hashlib

import numpy as np

from kd.benchmark_execution import fit_call
from kd.dataset._green import reference_kernel

METHODS = {"green_dso", "green_basis"}


def problems(dataset, case):
    split = dataset.split(case["seed"])
    split["sha256"] = hashlib.sha256(str(split).encode()).hexdigest()
    yield {
        "target": "kernel",
        "split": split,
        "ground_truth": None,
        "X_train": np.empty((len(split["train"]), 0)),
        "X_test": np.empty((len(split["test"]), 0)),
        "metadata": dataset.protocol_metadata,
    }


def forward_metrics(dataset, kernel, columns, scale):
    total = len(dataset.x) * len(columns)
    try:
        prediction = dataset.forward(kernel, columns)
    except (ValueError, FloatingPointError, OverflowError):
        return {
            "prediction_finite_coverage": 0.0,
            "prediction_total_points": total,
            "prediction_finite_points": 0,
            "prediction_status": "invalid",
            "status": "error",
            "error": "Nonfinite or malformed Green kernel/forward response",
        }
    truth = dataset.U[:, columns]
    mse = float(np.mean((prediction - truth) ** 2))
    rows = []
    for i, column in enumerate(columns):
        err = float(np.mean((prediction[:, i] - truth[:, i]) ** 2))
        rows.append(
            {
                "forcing": int(column),
                "mse": err,
                "nrmse": float(np.sqrt(err) / scale),
                "coverage": 1.0,
            }
        )
    return {
        "test_mse": mse,
        "test_rmse": float(np.sqrt(mse)),
        "test_nrmse": float(np.sqrt(mse) / scale),
        "prediction_finite_coverage": 1.0,
        "prediction_finite_points": total,
        "prediction_total_points": total,
        "prediction_status": "ok",
        "forcing_metrics": rows,
    }


def reference_diagnostic(dataset, kernel=None):
    if not dataset.reference:
        return {
            "status": "abstained_missing_reference",
            "reason": "No ExactGreen in original MAT; plotting fits are not ground truth",
        }
    try:
        reference = reference_kernel(dataset.reference, dataset.x, dataset.xi)
        scale = max(float(np.sqrt(np.mean(dataset.U**2))), 1e-15)
        result = {
            "status": "source_declared_not_certified",
            "role": "scoring_only",
            "forward": forward_metrics(dataset, reference, list(range(dataset.F.shape[1])), scale),
        }
        if kernel is not None:
            # Source-independent fixed exclusion of diagonal for kernel diagnostic;
            # forward integration always retains all points, no model-selected mask.
            mask = ~np.isclose(dataset.x[:, None], dataset.xi[None, :], atol=1e-12, rtol=0)
            denom = np.sqrt(np.mean(reference[mask] ** 2))
            result.update(
                kernel_relative_rmse=float(
                    np.sqrt(np.mean((kernel[mask] - reference[mask]) ** 2)) / denom
                )
                if denom > 0
                else None,
                kernel_mask="abs(x-xi)>1e-12; all offdiagonal points",
                excluded_fraction=float(1 - mask.mean()),
            )
        return result
    except (ValueError, SyntaxError, TypeError) as exc:
        return {"status": "reference_conflict", "error": str(exc)}


def run(dataset, problem, case):
    from run_benchmark import _model_stop_reason, _new_model

    model = _new_model(case["model"], case["model_params"])
    view = dataset.training_view(problem["split"])
    fit_call(model.fit, view)
    kernel = model.kernel(dataset.kernel_inputs)
    scale = max(float(np.sqrt(np.mean(view["response"] ** 2))), 1e-15)
    metrics = forward_metrics(dataset, kernel, problem["split"]["test"], scale)
    weighted = view["weighted_forcing"]
    singular = np.linalg.svd(weighted, compute_uv=False)
    rank = int(np.linalg.matrix_rank(weighted))
    diagnostics = {
        "training_forcing_rank": rank,
        "n_xi": len(dataset.xi),
        "n_train_forcings": weighted.shape[1],
        "condition": float(singular[0] / singular[-1]) if singular[-1] > 0 else None,
        "nonparametric_kernel_unique": bool(rank == len(dataset.xi)),
        **getattr(model, "diagnostics_", {}),
    }
    return {
        **metrics,
        **_model_stop_reason(model),
        "expression": getattr(model, "best_expression_", getattr(model, "expression_", None)),
        "metric_target": "heldout_forcing_response",
        "green_protocol_version": "green-v1",
        "split": problem["split"],
        "n_train": len(problem["split"]["train"]),
        "n_test": len(problem["split"]["test"]),
        "normalization_train_scale": scale,
        "recovery_eligible": False,
        "recovery_evaluator_status": "source_reference_not_certified",
        "method_provenance": {
            "fit_unit": "one kernel joint forcing fit",
            "role": "symbolic_discovery"
            if case["model"] == "green_dso"
            else "finite_basis_diagnostic",
        },
        "details": {
            "operator": dataset.protocol_metadata,
            "inverse_diagnostics": diagnostics,
            "reference_diagnostic": reference_diagnostic(dataset, kernel),
        },
    }
