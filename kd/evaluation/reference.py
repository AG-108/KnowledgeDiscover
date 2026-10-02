"""Explicit scoring-only controls, kept outside discovery method rankings."""

import copy
import hashlib
import json

import numpy as np

PROTOCOL = "reference-v1"


def reference_cases(cases):
    """One control per dataset/seed/split/scoring adapter, with its own identity."""
    controls = {}
    for template in cases:
        case = copy.deepcopy(template)
        case.pop("name", None)
        adapter = template["model"] if template["model"] in {"pdefind_nd", "implicit_pde"} else "standard"
        case.update(model="reference", device="cpu", model_params=(copy.deepcopy(template["model_params"])
                    if adapter != "standard" else {}), reference_adapter=adapter,
                    reference_protocol_version=PROTOCOL, benchmark_role="reference")
        case["run_params"] = {k: case["run_params"][k] for k in ("test_size", "max_train_samples")}
        # Training caps affect which observations/scales enter the shared scoring pipeline.
        key = json.dumps(case, sort_keys=True)
        digest = hashlib.sha256(key.encode()).hexdigest()[:12]
        case["name"] = f"{case['task']}_reference_{case['dataset']}_s{case['seed']}_{digest}"
        controls.setdefault(key, case)
    return list(controls.values())


def abstain(reason):
    return {"status": "skipped", "expression": "", "reference_status": reason,
            "reference_floor_eligible": False,
            "prediction_status": "abstain", "recovery_eligible": False,
            "recovery_evaluator_status": "not_applicable", "notes": reason}


def metadata(case):
    source = case.get("dataset_metadata", {}).get("reference_status", "unknown")
    return {"reference_protocol_version": PROTOCOL, "benchmark_role": "reference",
            "ranking_eligible": False, "reference_source_status": source,
            "reference_floor_eligible": source in {"coefficient_declared", "generator_declared", "formula_declared"},
            "recovery_eligible": False, "recovery_evaluator_status": "not_applicable",
            "stop_reason": "converged", "stop_reason_source": "reference_evaluator"}


def run(dataset, problem, case):
    from run_benchmark import _score_regression, expression_metrics, run_pde

    result = metadata(case)
    adapter = case.get("reference_adapter", "standard")
    if adapter in {"pdefind_nd", "implicit_pde"}:
        from kd.evaluation.equation import run as equation_run
        values = equation_run(dataset, problem, case)
        return {**values, **result, "reference_status": values.get("reference_status", "source_declared_diagnostic"),
                "reference_floor_eligible": result["reference_floor_eligible"] and bool(values.get("expression"))}
    if case["task"] == "green":
        from kd.dataset._green import reference_kernel
        from kd.evaluation.green import forward_metrics

        if not dataset.reference:
            return {**result, **abstain("abstained_missing_reference")}
        split = problem["split"]
        view = dataset.training_view(split)
        scale = max(float(np.sqrt(np.mean(view["response"] ** 2))), 1e-15)
        kernel = reference_kernel(dataset.reference, dataset.x, dataset.xi)
        return {**forward_metrics(dataset, kernel, split["test"], scale), **result,
                "expression": dataset.reference, "split": split, "normalization_train_scale": scale,
                "metric_target": "heldout_forcing_response", "reference_status": "source_declared_diagnostic"}
    if problem is None:
        values = run_pde(dataset, case)
        return {**values, **result, "reference_floor_eligible":
                result["reference_floor_eligible"] and bool(values.get("expression"))}
    truth = problem.get("ground_truth")
    if not truth:
        return {**result, **abstain("abstained_missing_reference")}
    from kd.evaluation.ode_rollout import numeric_expression

    try:
        function = numeric_expression(truth, problem["variable_names"])
    except (ValueError, SyntaxError, TypeError) as exc:
        return {**result, **abstain("abstained_unresolved_reference"), "reference_error": str(exc)}
    scores = _score_regression({}, problem["y_test"], lambda: np.broadcast_to(
        function(*problem["X_test"].T), problem["y_test"].shape))
    return {**scores, **expression_metrics(truth, truth, problem["variable_names"]), **result,
            "reference_status": "ok", "expression": str(truth),
            "target": problem["target"], "split": problem["split"],
            "n_train": len(problem["X_train"]), "n_test": len(problem["X_test"]),
            "variable_names": problem["variable_names"], "details": copy.deepcopy(problem["metadata"]),
            "metric_target": "state_time_derivative" if case["task"] == "ode" else "heldout_response"}
