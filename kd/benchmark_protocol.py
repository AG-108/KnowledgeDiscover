"""Shared comparison and recovery declarations, independent of model imports."""

import hashlib
import json
import math
from pathlib import Path

CERTIFIED_REFERENCES = frozenset(
    {"formula_declared", "generator_declared", "coefficient_declared", "certified"}
)
ROLES = frozenset({"main", "diagnostic", "oracle", "reference"})


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")
    ).hexdigest()


def metric_protocol_version(case, default):
    """Derive a new case's metric protocol; legacy cases use the same fallback."""
    if case.get("model") in {"pdefind_nd", "implicit_pde"}:
        return "2.2"
    name, task = case.get("dataset", ""), case.get("task")
    if task == "green" or name == "pyqg_subgrid" or name.startswith("dso_file_") or (name.startswith("fde_") and task == "sr"):
        return "2.3"
    return default


def execution_policy_hash(policy):
    """Timing identity excludes resource placement; callers own missing-policy defaults."""
    return digest({key: value for key, value in policy.items() if key != "resource_policy"})


def recovery_eligibility(case, target=None):
    """Eligibility belongs to the declared target, including missing/failed outputs."""
    if case.get("model") in {"reference", "pdefind_nd", "implicit_pde"}:
        return False
    if case.get("task") == "green" or case.get("metric_protocol_version") == "2.3":
        return False
    metadata = case.get("dataset_metadata", {})
    if metadata.get("reference_status") not in CERTIFIED_REFERENCES:
        return False
    truth = (target or {}).get("ground_truth", case.get("ground_truth"))
    return bool(truth)


def benchmark_role(case):
    """Keep oracle/system/basis diagnostics out of main rankings."""
    if case.get("model") == "reference":
        return "reference"
    params = case.get("model_params", {})
    if params.get("search_space_policy") == "declared_plus_reference":
        return "oracle"
    if case.get("model") in {"pdenet", "green_basis"} or case.get("variant") == "system_diagnostic":
        return "diagnostic"
    role = case.get("benchmark_role")
    if role in ROLES:
        return role
    declared = case.get("dataset_metadata", {}).get("benchmark_role")
    return "main" if declared in {"candidate", "core_main"} else "diagnostic"


def method_configuration(case):
    params = {
        key: value
        for key, value in case.get("model_params", {}).items()
        if key not in {"seed", "random_state", "out_path"}
    }
    for parameter in ("source_dir", "checkpoint_path", "model_path"):
        fingerprint = case.get("asset_fingerprints", {}).get(parameter)
        if fingerprint and parameter in params:
            params[parameter] = "sha256:" + fingerprint
    return {"model": params, "run": case.get("run_params", {})}


def validate_main_matrix(cases, policy, *, configuration_sources=None):
    """Validate D1/D3 and roles before execution and again before publication."""
    if not cases:
        raise ValueError("Formal matrix is empty")
    timeout = policy.get("timeout")
    if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout) or timeout <= 0 or policy.get("soft_deadline") is not True):
        raise ValueError("Formal main track requires a finite T and soft_deadline=true")
    configurations = {}
    ode_grammars = set()
    for case in cases:
        label = f"{case['model']}/{case['dataset']}"
        role = benchmark_role(case)
        if role == "reference":
            continue
        if role != "main":
            raise ValueError(f"{label}: {role} entries require a separate diagnostic matrix")
        if case.get("profile") != "full":
            raise ValueError(f"{label}: formal main track requires profile=full")
        if case.get("model_params", {}).get("search_space_policy", "declared") != "declared":
            raise ValueError(f"{label}: oracle search space is forbidden in the main track")
        key = (case["model"], case["task"])
        configuration = digest(method_configuration(case))
        if key in configurations and configurations[key] != configuration:
            raise ValueError(f"{label}: per-dataset method configuration violates D3")
        configurations[key] = configuration
        metadata = case.get("dataset_metadata", {})
        if metadata.get("source_status") not in {"upstream_byte_verified", "local_generator", "packaged_definition_verified"}:
            raise ValueError(f"{label}: source provenance has not been verified")
        if "source_blocked" in metadata.get("integration_status", ""):
            raise ValueError(f"{label}: source_blocked")
        if metadata.get("reference_status") not in CERTIFIED_REFERENCES:
            raise ValueError(f"{label}: selected main track has no certified reference declaration")
        if "asset_fingerprints" in case and any(value is None for value in case["asset_fingerprints"].values()):
            raise ValueError(f"{label}: a frozen input asset is unavailable")
        if case["dataset"].startswith("ode_core_"):
            grammar = case.get("task_grammar")
            if not isinstance(grammar, list) or not grammar:
                raise ValueError(f"{label}: Core ODE requires a predeclared shared task_grammar")
            ode_grammars.add(tuple(grammar))
        if case["dataset"] == "ode_core_robertson":
            params = case.get("dataset_params", {})
            if params.get("observation_noise", 0) or params.get("time_stride", 1) != 1:
                raise ValueError(f"{label}: noisy/sparse Robertson requires separate validation")
        if configuration_sources is not None:
            source = configuration_sources.get(case["model"], {})
            if not source.get("source") or not source.get("implementation_variant"):
                raise ValueError(f"{label}: missing configuration source/implementation variant")
            if source.get("validation_status") != "passed":
                raise ValueError(f"{label}: minimum method validation has not passed")
    if len(ode_grammars) > 1:
        raise ValueError("Core ODE task_grammar differs between systems")
    return {"n_cases": len(cases), "method_task_configurations": len(configurations),
            "core_ode_grammar": list(next(iter(ode_grammars))) if ode_grammars else None}


def inherited_policy(manifests, overrides=None):
    """Consumers inherit source timing; defaults must never silently change a control."""
    policies = [manifest.get("execution_policy") for manifest in manifests]
    policies = [policy for policy in policies if policy is not None]
    if policies and any(policy != policies[0] for policy in policies[1:]):
        raise ValueError("Source manifests have different timing budgets")
    policy = dict(policies[0]) if policies else {
        "timeout": 21600.0, "soft_deadline": True, "timeout_grace": 5.0,
        "setup_timeout": 300.0, "score_timeout": 300.0,
    }
    policy.pop("version", None)
    policy.pop("resource_policy", None)
    for key, value in (overrides or {}).items():
        if value is not None:
            if policies and any(manifest.get("benchmark_mode") == "main" for manifest in manifests) and value != policy.get(key):
                raise ValueError(f"Frozen main-track timing cannot be overridden at {key}")
            policy[key] = value
    return policy


def validate_shared_manifests(manifests):
    """A queue or merge cannot mix frozen matrices, implementations or modes."""
    for field in ("parent_matrix_hash", "experiment_identity_hash", "configuration_sources", "benchmark_mode"):
        values = [manifest.get(field) for manifest in manifests]
        if values and any(value != values[0] for value in values[1:]):
            raise ValueError(f"Source manifests differ at {field}")
    identities = [(manifest.get("identity") or {}).get("sha256") for manifest in manifests]
    if identities and any(value != identities[0] for value in identities[1:]):
        raise ValueError("Source manifests have different implementation/asset identities")


def read_manifests(paths):
    return [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]


def comparison_fields(case, policy=None):
    return {
        "method_config_hash": digest(method_configuration(case)),
        "evaluation_protocol_hash": digest(case.get("evaluation", {})),
        "execution_policy_hash": execution_policy_hash(policy) if policy is not None else "unbound",
        "experiment_identity_hash": case.get("experiment_identity_hash", "unbound"),
        "benchmark_role": benchmark_role(case),
    }
