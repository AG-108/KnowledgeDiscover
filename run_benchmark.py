"""Unified SR / ODE / PDE benchmark, based on examples/.

    python run_benchmark.py --list
    python run_benchmark.py --dry-run
    python run_benchmark.py --tasks sr --models gplearn --datasets Koza-2
    python run_benchmark.py --tasks ode --models dso gplearn
    python run_benchmark.py --models dso --device dso=cuda:0 --check-devices
    python run_benchmark.py --profile full --seeds 0 1 2 --resume

Both profiles cover all integrated datasets and applicable baselines. ``smoke``
uses demonstration budgets; ``full`` uses longer budgets, not claimed paper
reproduction. See docs/benchmark.md for evaluation and configuration details.
Importing this module does not import models, train, or create output files.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import fnmatch
import hashlib
import importlib
import io
import json
import math
import os
import pickle
import random
import re
import sys
import threading
import time
import traceback
from pathlib import Path

from kd.benchmark_execution import (
    EXECUTION_PROTOCOL_VERSION,
    STOP_REASONS,
    TERMINAL_STATUSES,
    PhaseClock,
    SkipCase,
    atomic_json,
    case_lock,
    execution_policy,
    fit_call,
    reap_orphan,
    score_call,
    supervise,
)
from kd.benchmark_identity import attach as attach_identity
from kd.benchmark_identity import (
    canonical_case,
    public_manifest,
    stable_execution,
    stable_resources,
)
from kd.benchmark_preparation import prepare_pde, slice_pde_time, temporal_block_holdout
from kd.benchmark_protocol import (
    comparison_fields,
    execution_policy_hash,
    metric_protocol_version,
    recovery_eligibility,
    validate_main_matrix,
)
from kd.benchmark_resources import policy_hash
from kd.benchmark_store import freeze_run, verify_freeze

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results" / "benchmark"
CONFIG_DIR = ROOT / "configs" / "benchmark"
DEFAULT_CONFIG_FILE = CONFIG_DIR / "benchmark.json"
TASKS = ("sr", "ode", "pde", "steady_pde", "green")

# Regression methods also fit individual ODE right-hand sides through our bridge.
BASELINES = {
    "green_dso":{"tasks":("green",),"example":"kd_dso_example.py","accelerator":"cuda"},
    "green_basis":{"tasks":("green",),"example":"kd_dso_example.py","accelerator":"cpu"},
    "pdefind_nd": {"tasks": ("pde",), "example": "kd_PDEfind_example.py", "accelerator": "cpu"},
    "implicit_pde": {"tasks": ("pde", "steady_pde"), "example": "kd_PDEfind_example.py", "accelerator": "cpu"},
    "dscv": {"tasks": ("sr", "ode", "pde"), "example": "kd_dscv_example.py", "accelerator": "cpu"},
    "spr": {
        "tasks": ("pde",),
        "example": "kd_dscv_spr_dataset_example.py",
        "accelerator": "cuda-pinn",
    },
    "sga": {"tasks": ("pde",), "example": "kd_sga_dataset_example.py", "accelerator": "cpu"},
    "dlga": {"tasks": ("pde",), "example": "kd_dlga_example.py", "accelerator": "cuda"},
    "deepmod": {"tasks": ("pde",), "example": "kd_deepmod_example.py", "accelerator": "cuda"},
    "pdefind": {"tasks": ("pde",), "example": "kd_PDEfind_example.py", "accelerator": "cpu"},
    "pdenet": {"tasks": ("pde",), "example": "kd_PDENet_example.py", "accelerator": "cuda"},
    "weakform": {"tasks": ("pde",), "example": "kd_weakform_example.py", "accelerator": "cpu"},
    "eqgpt": {"tasks": ("pde",), "example": "kd_eqgpt_example.py", "accelerator": "cuda"},
    "dso": {"tasks": ("sr", "ode"), "example": "kd_dso_example.py", "accelerator": "cuda"},
    "e2e": {"tasks": ("sr", "ode"), "example": "kd_e2e_example.py", "accelerator": "cuda"},
    "symbolicgpt": {
        "tasks": ("sr", "ode"),
        "example": "kd_symbolicgpt_example.py",
        "accelerator": "cuda",
    },
    "gplearn": {"tasks": ("sr", "ode"), "example": "kd_gplearn_example.py", "accelerator": "cpu"},
    "physo": {"tasks": ("sr", "ode"), "example": "kd_physo_example.py", "accelerator": "cuda"},
    "pysr": {"tasks": ("sr", "ode"), "example": "kd_pysr_example.py", "accelerator": "cpu"},
    "sisso": {"tasks": ("sr", "ode"), "example": "kd_sisso_example.py", "accelerator": "cpu"},
    "sindy": {"tasks": ("ode", "pde"), "example": "kd_sindy_example.py", "accelerator": "cpu"},
    "pysindy": {
        "tasks": ("ode", "pde"),
        "example": "kd_pysindy_example.py",
        "accelerator": "cpu",
    },
    "llmsr": {"tasks": ("sr", "ode"), "example": "kd_llmsr_example.py", "accelerator": "cpu"},
    "pyoperon": {
        "tasks": ("sr", "ode"),
        "example": "kd_pyoperon_example.py",
        "accelerator": "cpu",
    },
}
MODEL_CLASSES = {
    "green_dso":("kd.model.kd_green","GreenDSO"),
    "green_basis":("kd.model.kd_green_basis","GreenBasis"),
    "pdefind_nd": ("kd.model.kd_equation", "SparseEquationModel"),
    "implicit_pde": ("kd.model.kd_equation", "SparseEquationModel"),
    "dscv": ("kd.model.kd_dscv", "KD_DSCV"),
    "spr": ("kd.model.kd_dscv", "KD_DSCV_SPR"),
    "sga": ("kd.model.kd_sga", "KD_SGA"),
    "dlga": ("kd.model.kd_dlga", "KD_DLGA"),
    "deepmod": ("kd.model.kd_deepmod", "KD_DeepMoD"),
    "pdefind": ("kd.model.kd_pdefind", "PDEFindModel"),
    "pdenet": ("kd.model.kd_pdenet", "PDENetModel"),
    "eqgpt": ("kd.model.kd_eqgpt", "KD_EqGPT"),
    "dso": ("kd.model.kd_dso", "KD_DSO"),
    "e2e": ("kd.model.kd_e2e", "E2ETransformerModel"),
    "symbolicgpt": ("kd.model.kd_symbolicgpt", "KD_SymbolicGPT"),
    "gplearn": ("gplearn.genetic", "SymbolicRegressor"),
    "pysr": ("pysr", "PySRRegressor"),
    "sisso": ("kd.model.kd_sisso", "SISSOModel"),
    "sindy": ("kd.model.kd_sindy", "SINDyModel"),
    "pysindy": ("kd.model.kd_sindy", "PySINDyModel"),
    "weakform": ("kd.model.kd_wsindy", "WSINDyPDEModel"),
    "llmsr": ("kd.model.kd_llmsr", "KD_LLMSR"),
    "pyoperon": ("pyoperon.sklearn", "SymbolicRegressor"),
}
CUDA_DEVICE = re.compile(r"^cuda(?::(0|[1-9][0-9]*))?$")
RUN_KEYS = {
    "test_size",
    "max_train_samples",
    "dscv_epochs",
    "colloc_num",
    "sample_ratio",
    "spr_pretrain_epochs",
    "spr_pinn_epochs",
    "spr_iterations",
    "dlga_samples",
    "dlga_population",
    "dlga_generations",
}
PDE_REFERENCE_RHS = {
    # These coefficient-bearing forms are documented by the packaged generators/loaders.
    "burgers": "-u*u_x + 0.1*u_xx",
    "kdv": "-u*u_x - 0.0025*u_xxx",
    "chafee-infante": "u_xx - u + u**3",
    "burgers_sine": "-u*u_x + 0.1*u_xx",
    "allen_cahn": "0.003*u_xx + u - u**3",
    "convection_diffusion": "-u_x + 0.25*u_xx",
}
METRIC_PROTOCOL_VERSION = "2.1"
PDE_SUPPORT_METRIC_VERSION = "2.0"
SR_STRUCTURE_METRIC_VERSION = "1.0"


def _deep_update(target, changes):
    """Recursively update dictionaries while replacing scalar and list values."""
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = copy.deepcopy(value)
    return target


def _read_json_object(path, label):
    path = Path(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _project_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_baseline_configs(directory=None):
    """Load one model/run configuration file for every integrated baseline."""
    directory = _project_path(directory or CONFIG_DIR / "baselines")
    configs = {}
    for name in BASELINES:
        path = directory / f"{name}.json"
        if not path.is_file():
            raise ValueError(f"Missing baseline config: {path}")
        value = _read_json_object(path, "Baseline config")
        unknown = set(value) - {"name", "device", "model", "run", "profiles"}
        if unknown:
            raise ValueError(f"Unknown keys in {path}: {sorted(unknown)}")
        if value.get("name", name) != name:
            raise ValueError(f"Baseline config {path} must have name={name!r}")
        for section in ("model", "run", "profiles"):
            if not isinstance(value.get(section, {}), dict):
                raise ValueError(f"{section} in {path} must be a JSON object")
        unknown_run = set(value.get("run", {})) - RUN_KEYS
        if unknown_run:
            raise ValueError(f"Unknown run options in {path}: {sorted(unknown_run)}")
        for profile, profile_value in value.get("profiles", {}).items():
            if profile not in {"smoke", "full"} or not isinstance(profile_value, dict):
                raise ValueError(f"Invalid profile {profile!r} in {path}")
            if set(profile_value) - {"model", "run", "device"}:
                raise ValueError(f"Profile {profile!r} in {path} supports only model, run, device")
            if any(not isinstance(profile_value.get(key, {}), dict) for key in ("model", "run")):
                raise ValueError(f"Profile model/run in {path} must be JSON objects")
            unknown_run = set(profile_value.get("run", {})) - RUN_KEYS
            if unknown_run:
                raise ValueError(f"Unknown profile run options in {path}: {sorted(unknown_run)}")
        configs[name] = value
    return configs


def _normalise_config_shape(value):
    """Accept the former models/datasets/run-only override file as an overlay."""
    value = copy.deepcopy(value)
    explicit = value.pop("overrides", {})
    if not isinstance(explicit, dict):
        raise ValueError("overrides must be a JSON object")
    legacy = {"models": {}, "datasets": {}, "run": {}}
    if isinstance(value.get("models"), dict):
        legacy["models"] = value.pop("models")
    if isinstance(value.get("datasets"), dict):
        legacy["datasets"] = value.pop("datasets")
    if "run" in value:
        legacy["run"] = value.pop("run")
    value["overrides"] = _deep_update(legacy, explicit)
    device_map = value.setdefault("devices", {})
    if not isinstance(device_map, dict):
        raise ValueError("devices must be a JSON object")
    model_overrides = value["overrides"].get("models", {})
    if isinstance(model_overrides, dict):
        for model, params in model_overrides.items():
            if isinstance(params, dict) and "device" in params:
                device_map.setdefault(model, params.pop("device"))
    return value


def load_benchmark_config(path=None):
    """Load the repository defaults and optionally overlay another total config."""
    defaults = _read_json_object(DEFAULT_CONFIG_FILE, "Benchmark config")
    selected_path = Path(path).resolve() if path else DEFAULT_CONFIG_FILE
    value = defaults
    if selected_path != DEFAULT_CONFIG_FILE.resolve():
        value = _deep_update(
            value, _normalise_config_shape(_read_json_object(selected_path, "Benchmark config"))
        )
    else:
        value = _normalise_config_shape(value)
    allowed = {
        "baseline_config_dir",
        "profile",
        "tasks",
        "models",
        "datasets",
        "seeds",
        "output_dir",
        "timeout",
        "timeout_grace",
        "soft_deadline",
        "setup_timeout",
        "score_timeout",
        "resume",
        "devices",
        "profiles",
        "overrides",
        "benchmark_mode",
        "core_ode_grammar",
        "configuration_sources",
    }
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"Unknown benchmark config keys: {sorted(unknown)}")
    if value.get("profile") not in {"smoke", "full"}:
        raise ValueError("profile must be smoke or full")
    for key in ("tasks", "models", "datasets", "seeds"):
        if not isinstance(value.get(key), list):
            raise ValueError(f"{key} must be a JSON array")
    for key in ("devices", "profiles", "overrides"):
        if not isinstance(value.get(key, {}), dict):
            raise ValueError(f"{key} must be a JSON object")
    for profile, profile_value in value["profiles"].items():
        if profile not in {"smoke", "full"} or not isinstance(profile_value, dict):
            raise ValueError(f"Invalid benchmark profile: {profile!r}")
        if set(profile_value) - {"run"} or not isinstance(profile_value.get("run", {}), dict):
            raise ValueError(f"Benchmark profile {profile!r} supports only a run object")
    return value, selected_path


def validate_device_assignment(model, device):
    if not isinstance(device, str) or (device != "cpu" and not CUDA_DEVICE.fullmatch(device)):
        raise ValueError(f"Invalid device for {model}: {device!r}; use cpu, cuda, or cuda:N")
    if device != "cpu" and BASELINES[model]["accelerator"] == "cpu":
        raise ValueError(f"Baseline {model} does not support CUDA in the current integration")
    return device


def check_device_available(device):
    """Validate a configured CUDA device against the current PyTorch runtime."""
    if device == "cpu":
        return "CPU"
    try:
        import torch
    except ImportError as exc:
        raise ValueError(f"Device {device} requires PyTorch: {exc}") from exc
    if not torch.cuda.is_available():
        raise ValueError(f"Device {device} was requested, but torch.cuda.is_available() is false")
    index = torch.cuda.current_device() if device == "cuda" else int(device.split(":", 1)[1])
    if index >= torch.cuda.device_count():
        raise ValueError(
            f"Device {device} does not exist; PyTorch sees {torch.cuda.device_count()} GPU(s)"
        )
    return torch.cuda.get_device_name(index)


def parse_device_overrides(values):
    devices = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"Invalid --device {value!r}; expected BASELINE=DEVICE")
        model, device = value.split("=", 1)
        if model not in BASELINES:
            raise ValueError(f"Unknown baseline in --device: {model!r}")
        if model in devices:
            raise ValueError(f"Duplicate --device assignment for {model}")
        devices[model] = validate_device_assignment(model, device)
    return devices


def validate_benchmark_config(value, catalog):
    execution_policy(value.get("timeout"), timeout_grace=value.get("timeout_grace", 5.0),
                     setup_timeout=value.get("setup_timeout", 300.0),
                     score_timeout=value.get("score_timeout", 300.0),
                     soft_deadline=value.get("soft_deadline", True))
    if not isinstance(value.get("baseline_config_dir"), str) or not value["baseline_config_dir"]:
        raise ValueError("baseline_config_dir must be a non-empty path string")
    for key in ("tasks", "models", "datasets"):
        if not all(isinstance(item, str) and item for item in value[key]):
            raise ValueError(f"{key} must contain non-empty strings")
    if any(task not in TASKS for task in value["tasks"]):
        raise ValueError(f"tasks must contain only: {', '.join(TASKS)}")
    if not all(
        isinstance(seed, int) and not isinstance(seed, bool) and 0 <= seed < 2**32
        for seed in value["seeds"]
    ):
        raise ValueError("seeds must contain integers in [0, 2**32)")
    if value.get("timeout") is not None and (
        isinstance(value["timeout"], bool)
        or not isinstance(value["timeout"], (int, float))
        or value["timeout"] <= 0
    ):
        raise ValueError("timeout must be positive or null")
    if not isinstance(value.get("resume"), bool):
        raise ValueError("resume must be true or false")
    if not isinstance(value.get("output_dir"), str) or not value["output_dir"]:
        raise ValueError("output_dir must be a non-empty path string")
    unknown_devices = set(value["devices"]) - set(BASELINES)
    if unknown_devices:
        raise ValueError(f"Unknown devices baselines: {sorted(unknown_devices)}")
    for model, device in value["devices"].items():
        validate_device_assignment(model, device)
    overrides = value["overrides"]
    if value.get("benchmark_mode", "diagnostic") not in {"main", "diagnostic", "control"}:
        raise ValueError("benchmark_mode must be main, diagnostic or control")
    if value.get("benchmark_mode") == "main" and overrides.get("model_datasets"):
        raise ValueError("Formal main track forbids overrides.model_datasets (D3)")
    grammar = value.get("core_ode_grammar")
    if grammar is not None and (not isinstance(grammar, list) or not grammar or
                                not all(isinstance(token, str) and token for token in grammar)):
        raise ValueError("core_ode_grammar must be a non-empty array of operator names")
    if set(overrides) - {"models", "model_datasets", "datasets", "run", "evaluation"}:
        raise ValueError("overrides supports only models, model_datasets, datasets, run, evaluation")
    for key, choices in (("models", BASELINES), ("datasets", catalog)):
        section = overrides.get(key, {})
        if not isinstance(section, dict) or any(not isinstance(v, dict) for v in section.values()):
            raise ValueError(f"Each overrides.{key} entry must be a JSON object")
        unknown = set(section) - set(choices)
        if unknown:
            raise ValueError(f"Unknown overrides.{key}: {sorted(unknown)}")
    model_datasets = overrides.get("model_datasets", {})
    if not isinstance(model_datasets, dict) or set(model_datasets) - set(BASELINES):
        raise ValueError("overrides.model_datasets must map known models to dataset patterns")
    for model, patterns in model_datasets.items():
        if not isinstance(patterns, dict):
            raise ValueError(f"overrides.model_datasets.{model} must be a JSON object")
        for pattern, model_params in patterns.items():
            if not isinstance(pattern, str) or not any(
                fnmatch.fnmatchcase(name, pattern) for name in catalog
            ):
                raise ValueError(f"No dataset matches overrides.model_datasets.{model}.{pattern}")
            if not isinstance(model_params, dict) or "device" in model_params:
                raise ValueError("Dataset-specific model overrides must be objects without device")
    if not isinstance(overrides.get("run", {}), dict):
        raise ValueError("overrides.run must be a JSON object")
    evaluation = overrides.get("evaluation", {})
    if not isinstance(evaluation, dict) or any(not isinstance(v, dict) for v in evaluation.values()):
        raise ValueError("overrides.evaluation entries must be JSON objects")
    unknown_evaluation = set(evaluation) - ({"*"} | set(catalog))
    if unknown_evaluation:
        raise ValueError(f"Unknown overrides.evaluation datasets: {sorted(unknown_evaluation)}")
    if "rubber_test" in overrides.get("datasets", {}):
        raise ValueError(
            "rubber_test is the fixed official test split; configure rubber_train instead"
        )


def dataset_catalog():
    """All unified loaders and expressions understood by SymbolicRegressionDataset."""
    from kd.dataset import DATASET_REGISTRY
    from kd.dataset._catalog import KNOWN_BROKEN_DATASETS, TLC_UNSUPPORTED_STEADY_STATE
    from kd.dataset._metadata import dataset_metadata, pending_datasets

    catalog = {
        name: {
            "task": "sr" if info["category"] == "regression" else info["category"],
            "source": "catalog",
            "role": "dataset",
            "family": info.get("family", info["category"]),
            "opt_in":info.get("opt_in",False),
            **({"evaluation": copy.deepcopy(info["evaluation"])} if info.get("evaluation") else {}),
        }
        for name, info in sorted(DATASET_REGISTRY.items())
    }
    # Two official splits of ONE experiment, never two training datasets.
    catalog["rubber_train"].update(role="train_test", test_dataset="rubber_test")
    catalog["rubber_test"].update(role="test_only", paired_with="rubber_train")
    with (ROOT / "kd/dataset/data/benchmarks.csv").open(encoding="ISO-8859-1", newline="") as f:
        for row in csv.DictReader(f):
            catalog[row["name"]] = {
                "task": "sr",
                "source": "symbolic",
                "expression": row["expression"],
                "role": "dataset",
            }
    catalog["wave_breaking"] = {"task": "pde", "source": "example", "role": "dataset"}
    for name, info in catalog.items():
        info.update(dataset_metadata(name, info["task"], family=info.get("family")))
    excluded = {n: "Broken loader; excluded by kd.dataset._catalog" for n in KNOWN_BROKEN_DATASETS}
    excluded.update(
        {
            "TLC/" + n: "Steady-state data; no integrated time-dependent loader"
            for n in TLC_UNSUPPORTED_STEADY_STATE
        }
    )
    excluded.update({row["dataset"]: "; ".join(row["requirements"])
                     for row in pending_datasets()})
    return catalog, excluded


def _select(patterns, choices, label):
    selected = set()
    for pattern in patterns or ["*"]:
        matches = fnmatch.filter(list(choices), pattern)
        if not matches:
            raise ValueError(f"No {label} matches {pattern!r}. Choices: {', '.join(choices)}")
        selected.update(matches)
    return [name for name in choices if name in selected]


def build_experiments(
    catalog,
    *,
    tasks=TASKS,
    models=None,
    datasets=None,
    seeds=(0,),
    profile="smoke",
    config=None,
    baseline_configs=None,
    devices=None,
    run_defaults=None,
):
    """Build the complete task-aware matrix without loading data or models."""
    config = config or {}
    if baseline_configs is None or devices is None or run_defaults is None:
        defaults, _ = load_benchmark_config()
        baseline_configs = baseline_configs or load_baseline_configs(
            defaults["baseline_config_dir"]
        )
        devices = defaults["devices"] if devices is None else devices
        if run_defaults is None:
            run_defaults = defaults["profiles"][profile]["run"]
    selected_models = _select(models, BASELINES, "model")
    selected_data = _select(datasets, catalog, "dataset")
    if datasets is None or list(datasets) == ["*"]:
        selected_data = [n for n in selected_data if not catalog[n].get("opt_in")]
    selected_data = sorted({catalog[n].get("paired_with", n) for n in selected_data})
    cases = []
    for name in selected_data:
        task = catalog[name]["task"]
        if task not in tasks:
            continue
        for model in selected_models:
            if task not in BASELINES[model]["tasks"]:
                continue
            baseline_config = baseline_configs[model]
            profile_config = baseline_config.get("profiles", {}).get(profile, {})
            params = _deep_update(
                copy.deepcopy(baseline_config.get("model", {})), profile_config.get("model", {})
            )
            run = _deep_update(copy.deepcopy(run_defaults), baseline_config.get("run", {}))
            _deep_update(run, profile_config.get("run", {}))
            if model in {"dscv", "spr", "sga", "deepmod", "pdenet", "eqgpt", "dso", "symbolicgpt", "e2e"}:
                params["seed"] = 0
            if model in {"dscv", "spr"}:
                params["out_path"] = "./log/"
                suffix = "_t" if model == "spr" else ""
                ops = (
                    ["add", "mul", "div", "diff", "diff2", "diff3"]
                    if task == "pde"
                    else ["add", "sub", "mul", "div"]
                )
                params["binary_operators"] = [op + suffix for op in ops]
                params["unary_operators"] = [op + suffix for op in ("n2", "n3")]
            unknown_run = set(config.get("run", {})) - set(run)
            if unknown_run:
                raise ValueError(f"Unknown run options: {sorted(unknown_run)}")
            _deep_update(run, config.get("run", {}))
            data_params = {}
            if name == "pdeformer_sinus" and profile == "smoke":
                data_params.update(n_pde=1, n_x=32, n_t=21)
            if name == "wave_breaking" and profile == "smoke":
                data_params["keys"] = ["L_G2Tp12A080_broad"]
            data_params.update(config.get("datasets", {}).get(name, {}))
            for seed in dict.fromkeys(seeds):
                model_params = copy.deepcopy(params)
                if "seed" in model_params:
                    model_params["seed"] = seed
                if model in {"gplearn", "llmsr", "pyoperon", "pysr", "sisso"}:
                    model_params["random_state"] = seed
                model_override = copy.deepcopy(config.get("models", {}).get(model, {}))
                override_device = model_override.pop("device", None)
                _deep_update(model_params, model_override)
                for pattern, dataset_override in config.get("model_datasets", {}).get(model, {}).items():
                    if fnmatch.fnmatchcase(name, pattern):
                        _deep_update(model_params, copy.deepcopy(dataset_override))
                if model == "e2e":
                    for key in ("checkpoint_path", "source_dir"):
                        if model_params.get(key):
                            model_params[key] = str(_project_path(model_params[key]).resolve())
                device = profile_config.get("device", baseline_config.get("device", "cpu"))
                if override_device is not None:
                    device = override_device
                if model in devices:
                    device = devices[model]
                device = validate_device_assignment(model, device)
                if BASELINES[model]["accelerator"] != "cpu":
                    model_params["device"] = device
                declared_truth = catalog[name].get("expression")
                if name.startswith("ode_core_"):
                    from kd.dataset._ode_core import core_equations
                    declared_truth = core_equations(name.removeprefix("ode_core_"),
                        expose_parameters=bool(data_params.get("expose_parameters", False)))
                if task == "pde":
                    declared_truth = PDE_REFERENCE_RHS.get(name, declared_truth)
                    if declared_truth is None:
                        try:
                            from kd.dataset import get_dataset_sym_true

                            declared_truth = get_dataset_sym_true(name)
                        except (KeyError, ValueError):
                            pass
                case = {
                    "execution_protocol_version": EXECUTION_PROTOCOL_VERSION,
                    "metric_protocol_version": METRIC_PROTOCOL_VERSION,
                    "task": task,
                    "model": model,
                    "dataset": name,
                    "seed": seed,
                    "profile": profile,
                    "source": catalog[name]["source"],
                    "ground_truth": declared_truth,
                    "family": catalog[name].get("family", task),
                    "dataset_protocol_version": catalog[name]["dataset_protocol_version"],
                    "scientific_task": catalog[name]["scientific_task"],
                    "physical_family": catalog[name]["physical_family"],
                    "dataset_metadata": {key: copy.deepcopy(catalog[name][key]) for key in (
                        "target_definition", "split_protocol", "instance_policy", "benchmark_role",
                        "reference_status", "source_paths", "source_status", "integration_status",
                        "requirements", "source_evidence")},
                    "evaluation": _deep_update(
                        _deep_update(copy.deepcopy(catalog[name].get("evaluation", {})),
                                     config.get("evaluation", {}).get("*", {})),
                        config.get("evaluation", {}).get(name, {}),
                    ),
                    "device": device,
                    "model_params": model_params,
                    "dataset_params": copy.deepcopy(data_params),
                    "run_params": copy.deepcopy(run),
                }
                if model == "green_dso":
                    case["model_params"]["seed"] = seed
                if model in {"pdefind_nd", "implicit_pde"}:
                    case["equation_protocol_version"] = "equation-v1"
                    case["dataset_metadata"]["fit_unit"] = "one independently budgeted equation target"
                    case["dataset_metadata"]["split_protocol"] = (
                        "spatial_blocks_with_support_gap" if task == "steady_pde" else
                        "temporal_blocks_with_support_gap")
                if model == "pdenet":
                    from kd.dataset._advanced import declared_targets
                    case["variant"] = "system_diagnostic"
                    case["dataset_metadata"]["fit_unit"] = "one joint system fit with one target budget"
                    case["dataset_metadata"]["declared_equation_count"] = len(declared_targets(name))
                    case["dataset_metadata"]["comparison_role"] = "diagnostic; excluded from independent equation fairness"
                if task == "ode":
                    from kd.evaluation.ode_rollout import DEFAULTS
                    case["evaluation"]["ode_rollout"] = _deep_update(
                        copy.deepcopy(DEFAULTS), case["evaluation"].get("ode_rollout", {}))
                if name.startswith("cyt_"):
                    target_name = data_params.get("target", "Mut")
                    features = data_params.get("feature_columns", "default_local_flow_features")
                    case["dataset_metadata"]["target_definition"] = f"{target_name}=f({features})"
                if name.startswith("fde_") and task == "pde":
                    from kd.dataset._fractional import FDE_PDE_PROTOCOL_VERSION
                    case["fde_pde_protocol_version"] = FDE_PDE_PROTOCOL_VERSION
                    case["dataset_metadata"]["split_protocol"] = "temporal_block_with_support_gap"
                case["metric_protocol_version"] = metric_protocol_version(case, METRIC_PROTOCOL_VERSION)
                digest = hashlib.sha256(json.dumps(case, sort_keys=True).encode()).hexdigest()[:12]
                case["name"] = f"{task}_{model}_{name}_s{seed}_{digest}"
                cases.append(case)
    return cases


def check_compatibility(cases, excluded=None, progress=False, *, mode="sample", check_runtime=True):
    """Audit selected seeds and targets; full mode enumerates every instance."""
    from kd.benchmark_compatibility import check

    return check(cases, runner=sys.modules[__name__], excluded=excluded,
                 progress=progress, mode=mode, check_runtime=check_runtime)


def save_compatibility_report(output_dir, report):
    """Write detailed compatibility results as JSON and a flat pair table."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "compatibility_report.json", report)
    with (output_dir / "compatibility_pairs.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        fields = ["task", "baseline", "dataset", "case", "instance", "seed", "target",
                  "status", "reason", "capability_status", "load_status", "preparation_status",
                  "asset_status", "instance_eligibility", "runtime_status", "runtime_key", "fit_evidence",
                  "equation_targets"]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict))
                          else value for key, value in row.items()} for row in report["pairs"])


def _load_instances(case):
    from kd.dataset import GridPDEDataset, SymbolicRegressionDataset, load_dataset

    kwargs = dict(case["dataset_params"])
    name = case["dataset"]
    if case["source"] == "symbolic":
        kwargs.setdefault("seed", case["seed"])
        yield name, SymbolicRegressionDataset(name=name, **kwargs)
    elif name in {"solid_dif", "solid_strain_stress", "solid_hardening"}:
        from kd.dataset._domain import load_constitutive_instances

        yield from load_constitutive_instances(name, **kwargs)
    elif name == "wave_breaking":
        from examples._common import point_cloud_to_regular_grid

        keys = kwargs.pop("keys", None)
        kwargs.setdefault("nx", 512)
        with (ROOT / "kd/dataset/WaveBreaking.pkl").open("rb") as f:
            waves = pickle.load(f)
        for key in keys if keys is not None else sorted(waves):
            t, x, u = point_cloud_to_regular_grid(waves[key], **kwargs)
            dataset = GridPDEDataset(
                equation_name="wave_breaking",
                pde_data=None,
                x=x,
                t=t,
                usol=u.T,
                domain=None,
                epi=0.0,
                legacy=True,
            )
            from kd.dataset._provenance import source_evidence

            dataset.protocol_metadata = {
                "source_asset_id": "WaveBreaking.pkl", "source_curve_id": str(key),
                "source_evidence": source_evidence(name),
                "projection_parameters": copy.deepcopy(kwargs),
                "raw_rows": len(waves[key]), "projected_shape": list(u.T.shape),
            }
            yield key, dataset
    else:
        if name == "pdeformer_sinus":
            kwargs.setdefault("seed", case["seed"])
        data = load_dataset(name, **kwargs)
        if data is None:
            raise ValueError(f"Loader returned None for {name}")
        if isinstance(data, list):
            if not data:
                raise ValueError(f"Loader returned an empty batch for {name}")
            for i, item in enumerate(data):
                if name == "pdeformer_sinus":
                    item.protocol_metadata = {**getattr(item, "protocol_metadata", {}),
                        "unknown_coefficient_functions": ["f0(u)", "f1(u)", "source(x)", "diffusivity(x)"],
                        "boundary": {"kind": "periodic", "verified": True, "endpoint": "excluded",
                                     "evidence": "local _pdeformer_sinus.py spectral generator"}}
                yield f"{name}/{i:04d}", item
        else:
            yield name, data


def split_indices(n, test_size, seed, groups=None):
    """Split whole groups when supplied; fail rather than leak a single group."""
    import numpy as np

    if not 0 < test_size < 1 or n < 2:
        raise ValueError("A split needs >=2 samples and 0 < test_size < 1")
    rng = np.random.default_rng(seed)
    if groups is None:
        ids = rng.permutation(n)
        count = min(n - 1, max(1, round(n * test_size)))
        return ids[count:], ids[:count]
    groups = np.asarray(groups)
    unique = np.unique(groups)
    if len(unique) < 2:
        raise SkipCase("Group holdout requires at least two distinct groups")
    shuffled = rng.permutation(unique)
    count = min(len(unique) - 1, max(1, round(len(unique) * test_size)))
    test_mask = np.isin(groups, shuffled[:count])
    return np.flatnonzero(~test_mask), np.flatnonzero(test_mask)


def regression_problems(dataset, case):
    """Independent scalar targets and explicit, reproducible evaluation splits."""
    import numpy as np

    from kd.dataset import (
        FractionalResponseDataset,
        ODEDataset,
        SymbolicRegressionDataset,
        load_dataset,
    )

    seed, options = case["seed"], case["run_params"]
    if isinstance(dataset, ODEDataset):
        train_ids, test_ids = split_indices(dataset.n_traj, options["test_size"], seed)

        def arrays(ids):
            trajectories = []
            for i in ids:
                traj = dataset.trajectories[i]
                if len(traj["t"]) < 3 or not np.all(np.diff(traj["t"]) > 0):
                    raise ValueError("ODE trajectories require >=3 strictly increasing times")
                trajectories.append(traj)
            subset = ODEDataset(
                dataset.equation_name,
                trajectories,
                dataset.state_vars,
                param_names=dataset.param_names,
            )
            X, y, names = subset.to_regression_arrays(order=1, include_params=True)
            # Discard one-sided derivative boundary estimates per trajectory.
            offsets = np.cumsum([0] + [len(t["t"]) for t in trajectories[:-1]])
            keep = np.concatenate(
                [np.arange(1, len(t["t"]) - 1) + offset for t, offset in zip(trajectories, offsets)]
            )
            return X[keep], y[keep], names

        X_train, y_train, names = arrays(train_ids)
        X_test, y_test, _ = arrays(test_ids)
        split = "trajectory_holdout"
        targets = [f"d({state})/dt" for state in dataset.state_vars]
        metadata = {
            "train_trajectories": [dataset.trajectories[i]["id"] for i in train_ids],
            "test_trajectories": [dataset.trajectories[i]["id"] for i in test_ids],
            "generation_metadata": getattr(dataset, "generation_metadata", None),
        }
        if case.get("task_grammar"):
            metadata["benchmark_function_set"] = list(case["task_grammar"]) + ["const"]
            metadata["grammar_source"] = "predeclared_shared_core_ode_task"
        ground_truth = getattr(dataset, "sym_true", None)
    elif case["dataset"] == "pyqg_subgrid":
        from kd.evaluation.closure import problems
        yield from problems(dataset, case)
        return
    elif case["dataset"].startswith("dso_file_"):
        X_train,y_train,X_test,y_test=dataset.X_train,dataset.y_train,dataset.X_test,dataset.y_test
        names=dataset.variable_names
        targets=["y"]
        metadata=copy.deepcopy(dataset.protocol_metadata)
        split=metadata["split_protocol"]
        ground_truth=None
    elif isinstance(dataset, FractionalResponseDataset):
        train, test, split_details = dataset.temporal_holdout(options["test_size"])
        X_train, y_train = dataset.X[train], dataset.y[train]
        X_test, y_test = dataset.X[test], dataset.y[test]
        names, targets = dataset.variable_names, [dataset.target_name]
        split = dataset.split_protocol
        metadata = {**copy.deepcopy(dataset.protocol_metadata), **split_details}
        ground_truth = None
    elif isinstance(dataset, SymbolicRegressionDataset):
        d = dataset.get_data()
        X_train, y_train, X_test, y_test = (
            d[k] for k in ("X_train", "y_train", "X_test", "y_test")
        )
        names = [f"x{i + 1}" for i in range(X_train.shape[1])]
        split, targets, metadata = "provided_benchmark_split", ["y"], {
            "benchmark_function_set": list(dataset.function_set),
        }
        ground_truth = case.get("ground_truth")
    elif case["dataset"] == "rubber_train":
        test = load_dataset("rubber_test")
        if dataset.variable_names != test.variable_names:
            raise ValueError("Rubber train/test feature columns do not match")
        X_train, y_train, X_test, y_test = dataset.X, dataset.y, test.X, test.y
        names = dataset.variable_names
        split, targets, metadata = "official_rubber_split", ["y"], {"test_dataset": "rubber_test"}
        ground_truth = getattr(dataset, "sym_true", None)
    else:
        train, test = split_indices(len(dataset.X), options["test_size"], seed, dataset.groups)
        if case["dataset"] in {"solid_dif", "solid_strain_stress", "solid_hardening"}:
            from kd.dataset._provenance import CONSTITUTIVE_ELIGIBILITY

            rule = CONSTITUTIVE_ELIGIBILITY
            if (len(np.unique(dataset.groups[train])) < rule["min_train_rate_groups"]
                    or len(np.unique(dataset.groups[test])) < rule["min_test_rate_groups"]
                    or len(train) < rule["min_train_samples"] or len(test) < rule["min_test_samples"]):
                raise SkipCase("Constitutive eligibility: requires >=2 training rate groups, "
                               ">=1 test rate group, >=2 training samples and >=1 test sample")
            if np.ptp(dataset.y[train]) == 0:
                raise SkipCase("Constitutive eligibility: constant training target")
        X_train, y_train, X_test, y_test = (
            dataset.X[train],
            dataset.y[train],
            dataset.X[test],
            dataset.y[test],
        )
        names = dataset.variable_names
        split = getattr(dataset, "split_protocol",
                        "group_holdout" if dataset.groups is not None else "random_holdout")
        targets = [getattr(dataset, "target_name", "y")]
        metadata = copy.deepcopy(getattr(dataset, "protocol_metadata", {}))
        if dataset.groups is not None:
            metadata.update(train_groups=np.unique(dataset.groups[train]).tolist(),
                            test_groups=np.unique(dataset.groups[test]).tolist())
        ground_truth = getattr(dataset, "sym_true", None)
    X_train, X_test = np.asarray(X_train), np.asarray(X_test)
    y_train = np.asarray(y_train).reshape(len(X_train), -1)
    y_test = np.asarray(y_test).reshape(len(X_test), -1)
    if not all(np.isfinite(v).all() for v in (X_train, X_test, y_train, y_test)):
        raise ValueError("Regression data contain NaN/Inf; explicit preprocessing is required")
    limit = options["max_train_samples"]
    if limit is not None:
        if limit < 2:
            raise ValueError("max_train_samples must be >=2 or null")
        if len(X_train) > limit:
            ids = np.random.default_rng(seed).choice(len(X_train), limit, replace=False)
            X_train, y_train = X_train[ids], y_train[ids]
    if case["dataset"] in {"solid_dif", "solid_strain_stress", "solid_hardening"}:
        rate_axis = 0 if case["dataset"] == "solid_dif" else 1
        if len(np.unique(X_train[:, rate_axis])) < 2 or np.ptp(y_train) == 0:
            raise SkipCase("Constitutive eligibility: training sample cap removed rate or target variation")
    if isinstance(dataset, FractionalResponseDataset):
        metadata["n_train_rows_used"] = len(X_train)
        metadata["training_sample_policy"] = "seeded_without_replacement; training block only"
    for i, target in enumerate(targets):
        target_truth = ground_truth
        if isinstance(ground_truth, dict):
            target_truth = ground_truth.get(target, ground_truth.get(str(i)))
        elif isinstance(ground_truth, (list, tuple)):
            target_truth = ground_truth[i] if i < len(ground_truth) else None
        yield {
            "X_train": X_train,
            "y_train": y_train[:, i],
            "X_test": X_test,
            "y_test": y_test[:, i],
            "variable_names": names,
            "target": target,
            "split": split,
            "metadata": metadata,
            "ground_truth": target_truth or None,
        }


def regression_metrics(y, prediction):
    import numpy as np

    y = np.asarray(y, dtype=float).reshape(-1)
    prediction = np.asarray(prediction, dtype=float).reshape(-1)
    if prediction.shape != y.shape:
        raise ValueError("Predictions must match target shape")
    finite = np.isfinite(y) & np.isfinite(prediction)
    coverage = float(finite.sum() / y.size) if y.size else None
    coverage_fields = {
        "prediction_total_points": int(y.size),
        "prediction_finite_points": int(finite.sum()),
        "prediction_finite_coverage": coverage,
    }
    if not finite.all():
        return coverage_fields
    mse = float(np.mean((y - prediction) ** 2))
    variance = float(np.var(y))
    return {
        **coverage_fields,
        "test_mse": mse,
        "test_rmse": math.sqrt(mse),
        "test_nrmse": math.sqrt(mse / variance) if variance > 0 else (0.0 if mse == 0 else None),
        "test_r2": 1.0 - mse / variance if variance > 0 else None,
    }


_PREFIX_ARITY = {
    "add": 2,
    "sub": 2,
    "mul": 2,
    "div": 2,
    "pow": 2,
    "n2": 1,
    "n3": 1,
    "n4": 1,
    "neg": 1,
    "sin": 1,
    "cos": 1,
    "tan": 1,
    "exp": 1,
    "expneg": 1,
    "log": 1,
    "sqrt": 1,
    "abs": 1,
    "tanh": 1,
}


def _prefix_sympy(text):
    """Parse the comma-separated prefix notation used by PDE discovery models."""
    import sympy as sp

    tokens = [token.strip() for token in text.split(",") if token.strip()]
    if not tokens or tokens[0].lower() not in {*_PREFIX_ARITY, "diff", "diff2", "diff3"}:
        return None
    position = 0

    def parse():
        nonlocal position
        if position >= len(tokens):
            raise ValueError("Incomplete prefix expression")
        token = tokens[position]
        position += 1
        lower = token.lower()
        if lower in {"diff", "diff2", "diff3"}:
            field, coordinate = parse(), parse()
            order = {"diff": 1, "diff2": 2, "diff3": 3}[lower]
            field_name = str(field).replace("1", "")
            axis = {"x1": "x", "x2": "y", "x3": "z"}.get(str(coordinate), str(coordinate))
            return sp.Symbol(f"{field_name}_{axis * order}")
        if lower in _PREFIX_ARITY:
            args = [parse() for _ in range(_PREFIX_ARITY[lower])]
            operations = {
                "add": lambda a, b: a + b,
                "sub": lambda a, b: a - b,
                "mul": lambda a, b: a * b,
                "div": lambda a, b: a / b,
                "pow": lambda a, b: a**b,
                "n2": lambda a: a**2,
                "n3": lambda a: a**3,
                "n4": lambda a: a**4,
                "neg": lambda a: -a,
                "sin": sp.sin,
                "cos": sp.cos,
                "tan": sp.tan,
                "exp": sp.exp,
                "expneg": lambda a: sp.exp(-a),
                "log": sp.log,
                "sqrt": sp.sqrt,
                "abs": sp.Abs,
                "tanh": sp.tanh,
            }
            return operations[lower](*args)
        if lower == "const":
            return sp.Symbol("const")
        try:
            return sp.Float(token)
        except ValueError:
            aliases = {"u1": "u", "u2": "v", "u3": "w"}
            return sp.Symbol(aliases.get(lower, token))

    result = parse()
    if position != len(tokens):
        raise ValueError("Extra tokens in prefix expression")
    return result


_PRETTY_SUBSCRIPT_DIGITS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
_PRETTY_SUPERSCRIPT_DIGITS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻", "0123456789+-")


def _normalize_pretty_expression(text):
    """Translate unambiguous single-line PhySO display glyphs to infix text.

    Multi-line SymPy pretty output encodes fractions and powers by position;
    flattening it would change the expression, so leave it unparsed.
    """
    if text.isascii():
        return text
    if "\n" in text or "\r" in text:
        return None
    text = text.translate(_PRETTY_SUBSCRIPT_DIGITS)
    text = (
        text.replace("⋅", "*")
        .replace("×", "*")
        .replace("−", "-")
        .replace("π", "pi")
    )
    text = re.sub(
        r"(?<=[\w)])([⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]+)",
        lambda match: f"**({match.group(1).translate(_PRETTY_SUPERSCRIPT_DIGITS)})",
        text,
    )
    return text if text.isascii() else None


def expression_sympy(text, variable_names=()):
    """Convert common baseline expression formats to one SymPy expression."""
    import sympy as sp

    if text is None or not str(text).strip():
        return None
    text = str(text).strip()
    if ";" in text:
        return None
    if "=" in text:
        text = text.split("=", 1)[1].strip()
    text = _normalize_pretty_expression(text)
    if text is None:
        return None
    try:
        prefix = _prefix_sympy(text)
        if prefix is not None:
            return prefix
    except (TypeError, ValueError, IndexError):
        return None
    symbols = {name: sp.Symbol(name) for name in variable_names}
    for index, name in enumerate(variable_names):
        symbols[f"X{index}"] = symbols[name]
        symbols[f"x_{index}"] = symbols[name]
    if len(variable_names) == 1:
        symbols["x"] = symbols[variable_names[0]]
    for compact, expanded in {
        "ut": "u_t",
        "ux": "u_x",
        "uxx": "u_xx",
        "uxxx": "u_xxx",
        "uy": "u_y",
        "uyy": "u_yy",
        "uyyy": "u_yyy",
        "uxy": "u_xy",
    }.items():
        symbols[compact] = symbols.get(expanded, sp.Symbol(expanded))
    functions = {
        "add": lambda a, b: a + b,
        "sub": lambda a, b: a - b,
        "mul": lambda a, b: a * b,
        "div": lambda a, b: a / b,
        "pow": lambda a, b: a**b,
        "n2": lambda a: a**2,
        "n3": lambda a: a**3,
        "n4": lambda a: a**4,
        "neg": lambda a: -a,
        "inv": lambda a: 1 / a,
        "max": sp.Max,
        "min": sp.Min,
        "abs": sp.Abs,
        "Abs": sp.Abs,
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "exp": sp.exp,
        "expneg": lambda a: sp.exp(-a),
        "log": sp.log,
        "sqrt": sp.sqrt,
        "tanh": sp.tanh,
        "pi": sp.pi,
    }
    text = re.sub(r"\b(?:numpy|np|math)\.", "", text).replace("^", "**")
    try:
        return sp.sympify(text, locals={**functions, **symbols})
    except (TypeError, ValueError, SyntaxError, sp.SympifyError):
        return None


def _structural_node(node, variable_symbols):
    """Return a coefficient-free expression signature and aligned coefficients.

    Numeric exponents are retained because they define term form (``x**2`` is
    structurally different from ``x**3``).  Other numeric constants are treated
    as coefficients, including scales and offsets inside function arguments.
    """
    import sympy as sp

    variables = set(variable_symbols)
    if not (node.free_symbols & variables) and node.is_number:
        try:
            value = float(node.evalf())
        except (TypeError, ValueError, OverflowError):
            value = None
        return ("constant",), ([] if value is None else [value])
    if isinstance(node, sp.Symbol):
        return ("symbol", str(node)), []
    if isinstance(node, sp.Add):
        records = [_structural_term_record(arg, variables) for arg in node.args]
        records.sort(key=lambda item: repr(item[0]))
        return (
            ("add", tuple(item[0] for item in records)),
            [value for _, values in records for value in values],
        )
    if isinstance(node, sp.Mul):
        scale = sp.S.One
        children = []
        for arg in node.args:
            if not (arg.free_symbols & variables) and arg.is_number:
                scale *= arg
            else:
                children.append(_structural_node(arg, variables))
        try:
            coefficients = [float(scale.evalf())]
        except (TypeError, ValueError, OverflowError):
            coefficients = []
        children.sort(key=lambda item: repr(item[0]))
        signatures = tuple(item[0] for item in children)
        coefficients.extend(value for _, values in children for value in values)
        if not signatures:
            return ("constant",), coefficients
        if len(signatures) == 1:
            return signatures[0], coefficients
        return ("mul", signatures), coefficients
    if isinstance(node, sp.Pow):
        base_signature, coefficients = _structural_node(node.base, variables)
        # The exponent is part of the term definition, not a fitted coefficient.
        exponent_signature = sp.srepr(sp.simplify(node.exp))
        return ("pow", base_signature, exponent_signature), coefficients
    if isinstance(node, sp.Function):
        arguments = [_structural_term_record(arg, variables) for arg in node.args]
        return (
            (node.func.__name__, tuple(item[0] for item in arguments)),
            [value for _, values in arguments for value in values],
        )
    children = [_structural_node(arg, variables) for arg in node.args]
    return (
        (node.func.__name__, tuple(item[0] for item in children)),
        [value for _, values in children for value in values],
    )


def _structural_term_record(term, variable_symbols):
    """Return one additive term's signature and coefficient vector."""
    import sympy as sp

    signature, coefficients = _structural_node(term, variable_symbols)
    if not isinstance(term, sp.Mul) and not (
        not (term.free_symbols & set(variable_symbols)) and term.is_number
    ):
        # Every nonconstant additive term has an amplitude, even when it is 1.
        coefficients = [1.0, *coefficients]
    return signature, coefficients


def _structural_terms(expression, variable_symbols):
    """Expand algebraic sums and return sorted term records."""
    import sympy as sp

    # Full simplify can be extremely expensive for large GP expressions and is
    # unnecessary for term-support extraction. SymPy parsing has already
    # canonicalized elementary arithmetic; expansion exposes additive support.
    expanded = sp.expand(expression)
    terms = sp.Add.make_args(expanded)
    records = [_structural_term_record(term, set(variable_symbols)) for term in terms]
    return sorted(records, key=lambda item: (repr(item[0]), tuple(item[1])))


def _relative_coefficient_error(discovered_records, truth_records):
    """Return aligned relative L2 error after structural equivalence is known."""
    import numpy as np

    discovered = np.asarray(
        [value for _, values in discovered_records for value in values], dtype=float
    )
    truth = np.asarray([value for _, values in truth_records for value in values], dtype=float)
    if discovered.shape != truth.shape or not np.isfinite(discovered).all() or not np.isfinite(truth).all():
        return None
    denominator = float(np.linalg.norm(truth))
    numerator = float(np.linalg.norm(discovered - truth))
    return numerator / denominator if denominator else numerator


def expression_metrics(expression, ground_truth=None, variable_names=(), evaluate_algebraic=True):
    """Return complexity, structural recovery, and separate coefficient error.

    ``exact_recovery`` is retained as the public compatibility field, but from
    protocol 2.1 it means exact additive-term structural recovery while ignoring
    numeric coefficients.  Strict algebraic equality remains available as
    ``algebraic_exact_recovery``.
    """
    import sympy as sp

    discovered = expression_sympy(expression, variable_names)
    complexity_text = str(expression).split("=", 1)[-1]
    tokens = re.findall(
        r"[A-Za-z_]\w*|(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|\*\*|[-+*/^]",
        complexity_text,
    )
    complexity = len(tokens) if tokens else None
    structural = None
    algebraic = None
    coefficient_error = None
    truth = expression_sympy(ground_truth, variable_names)
    if discovered is not None and truth is not None:
        try:
            algebraic = (
                int(sp.simplify(discovered - truth) == 0)
                if evaluate_algebraic else None
            )
        except (TypeError, ValueError, NotImplementedError):
            algebraic = None
        try:
            symbols = [sp.Symbol(name) for name in variable_names]
            if not symbols:
                symbols = sorted(discovered.free_symbols | truth.free_symbols, key=str)
            discovered_records = _structural_terms(discovered, symbols)
            truth_records = _structural_terms(truth, symbols)
            structural = int(
                [record[0] for record in discovered_records]
                == [record[0] for record in truth_records]
            )
            if structural:
                coefficient_error = _relative_coefficient_error(
                    discovered_records, truth_records
                )
        except (TypeError, ValueError, NotImplementedError, OverflowError):
            structural = None
    return {
        "ground_truth_expression": ground_truth,
        "expression_complexity": complexity,
        "exact_recovery": structural,
        "structural_recovery": structural,
        "exact_recovery_semantics": "additive_term_structure_ignoring_numeric_coefficients",
        "structural_recovery_metric_version": SR_STRUCTURE_METRIC_VERSION,
        "algebraic_exact_recovery": algebraic,
        "coefficient_error": coefficient_error,
        "recovery_evaluator_status": (
            "not_applicable"
            if ground_truth in (None, "")
            else ("evaluated" if structural is not None else "unparsed")
        ),
    }


def _gplearn_protected_exp(values):
    """Finite exponential primitive for gplearn's closed function set."""
    import numpy as np

    return np.exp(np.clip(values, -100.0, 100.0))


def _gplearn_expneg(values):
    import numpy as np

    return np.exp(np.clip(-values, -100.0, 100.0))


def _gplearn_square(values):
    return values * values


def _gplearn_cube(values):
    return values * values * values


def _gplearn_tanh(values):
    import numpy as np

    return np.tanh(values)


def _dataset_search_space(params, metadata, model_name, reference_expression=None):
    """Resolve a declared benchmark grammar, optionally repairing missing unary functions.

    The reference-informed mode is an explicit oracle-assisted diagnostic; its results
    must not be pooled with runs using only the published function-set metadata.
    """
    params = copy.deepcopy(params)
    policy = params.pop("search_space_policy", "declared")
    keep_constant_range = params.pop("keep_constant_range", False)
    keep_free_constants = params.pop("keep_free_constants", False)
    keep_fixed_consts = params.pop("keep_fixed_consts", False)
    if policy not in {"declared", "declared_plus_reference"}:
        raise ValueError(f"Unknown search_space_policy: {policy!r}")
    if not params.pop("use_dataset_function_set", False):
        return params
    declared = metadata.get("benchmark_function_set")
    if not declared:
        return params
    operations, constants = [], []
    has_free_constant = False
    for token in declared:
        token = str(token).strip()
        if token == "const":
            has_free_constant = True
            continue
        if token == "poly":
            # DSO restores this native terminal below. Other baselines must
            # synthesize polynomial terms from their ordinary operators.
            continue
        try:
            constants.append(float(token))
            continue
        except ValueError:
            operations.append(token)
    if policy == "declared_plus_reference" and isinstance(reference_expression, str):
        # The benchmark CSV sometimes omits a function required by its own target
        # (e.g. Nguyen-8 declares Koza operators but has sqrt(x1) as its target).
        for operation in ("sin", "cos", "exp", "log", "sqrt", "tan", "tanh", "abs"):
            if operation not in operations and re.search(
                rf"(?<![A-Za-z]){operation}\s*\(", reference_expression
            ):
                operations.append(operation)
    if model_name == "gplearn":
        supported = {
            "add", "sub", "mul", "div", "sqrt", "log", "abs", "neg", "inv",
            "max", "min", "sin", "cos", "tan", "tanh", "exp", "expneg", "n2", "n3",
        }
        params["function_set"] = [op for op in operations if op in supported]
        if not keep_constant_range and not has_free_constant and not constants:
            params["const_range"] = None
    elif model_name == "physo":
        params["op_names"] = operations
        fixed = list(params.get("fixed_consts", [])) if keep_fixed_consts else []
        params["fixed_consts"] = list(dict.fromkeys(fixed + constants))
        if not has_free_constant and not keep_free_constants:
            params.pop("free_consts_names", None)
            params.pop("free_consts_init_val", None)
            params.pop("free_consts_units", None)
    elif model_name == "pyoperon":
        # Operon uses different names for powers and always needs terminal nodes.
        aliases = {
            "n2": ("square",),
            "n3": ("pow",),
            "inv": ("div",),
            "neg": ("sub",),
            "expneg": ("exp", "sub"),
        }
        supported = {
            "add", "sub", "mul", "div", "sin", "cos", "exp", "log",
            "sqrt", "tan", "tanh", "abs", "square", "pow",
        }
        resolved = []
        for operation in operations:
            for symbol in aliases.get(operation, (operation,)):
                if symbol in supported and symbol not in resolved:
                    resolved.append(symbol)
        params["allowed_symbols"] = ",".join(resolved + ["constant", "variable"])
    elif model_name == "pysr":
        binary_aliases = {
            "add": "+", "sub": "-", "mul": "*", "div": "/",
            "inv": "/", "neg": "-", "expneg": "-",
        }
        unary_aliases = {"n2": "square", "n3": "cube", "expneg": "exp"}
        unary_supported = {
            "sin", "cos", "exp", "log", "sqrt", "tan", "tanh", "abs",
            "square", "cube",
        }
        binary, unary = [], []
        for operation in operations:
            mapped_binary = binary_aliases.get(operation)
            mapped_unary = unary_aliases.get(operation, operation)
            if mapped_binary and mapped_binary not in binary:
                binary.append(mapped_binary)
            if mapped_unary in unary_supported and mapped_unary not in unary:
                unary.append(mapped_unary)
        params["binary_operators"] = binary
        params["unary_operators"] = unary
    elif model_name == "dso":
        # DSO's native vocabulary also accepts fitted constants, fixed numeric
        # terminals, and its optimized polynomial terminal.
        tokens = list(operations)
        tokens.extend(str(value) for value in constants)
        if has_free_constant:
            tokens.append("const")
        if "poly" in declared:
            tokens.append("poly")
        params["function_set"] = list(dict.fromkeys(tokens))
    elif model_name in {"dscv", "spr"}:
        binary_supported = {"add", "sub", "mul", "div"}
        unary_supported = {
            "sin", "cos", "tan", "exp", "log", "sqrt", "n2", "n3",
            "neg", "abs", "tanh", "inv", "expneg",
        }
        suffix = "_t" if model_name == "spr" else ""
        binary = [op + suffix for op in operations if op in binary_supported]
        unary = [
            op + suffix for op in operations
            if op in (unary_supported if model_name == "dscv" else {"n2", "n3"})
        ]
        if not binary and not unary:
            raise SkipCase(f"{model_name} has no supported operators in the declared function set")
        params["binary_operators"] = list(dict.fromkeys(binary))
        params["unary_operators"] = list(dict.fromkeys(unary))
    elif model_name == "sisso":
        aliases = {"mul": ("mult",), "n2": ("sq",), "n3": ("cb",),
                   "neg": ("sub",), "expneg": ("neg_exp",)}
        supported = {"add", "sub", "mult", "div", "inv", "sq", "cb", "sqrt",
                     "sin", "cos", "exp", "neg_exp", "log"}
        params["allowed_ops"] = list(dict.fromkeys(
            symbol for op in operations for symbol in aliases.get(op, (op,)) if symbol in supported
        ))
        if not params["allowed_ops"]:
            raise SkipCase("sisso has no supported operators in the declared function set")
    elif model_name == "symbolicgpt":
        # This wrapper builds a fresh training corpus from these operators.
        # Its generator has no native tan/tanh/abs, nor DSO's poly terminal.
        aliases = {
            "n2": ("pow",), "n3": ("pow",), "inv": ("div",),
            "neg": ("sub",), "expneg": ("exp", "sub"),
        }
        supported = {"add", "sub", "mul", "div", "sin", "cos", "exp", "log", "sqrt", "pow"}
        resolved = []
        for operation in operations:
            for symbol in aliases.get(operation, (operation,)):
                if symbol in supported and symbol not in resolved:
                    resolved.append(symbol)
        params["op_list"] = resolved or ["id"]
    return params


def _new_model(name, params):
    params = copy.deepcopy(params)
    if name == "gplearn":
        # JSON has no tuple type, while gplearn validates these two options as
        # tuples rather than generic two-element sequences.
        for key in ("const_range", "init_depth"):
            if isinstance(params.get(key), list):
                params[key] = tuple(params[key])
        custom_functions = {
            "exp": (_gplearn_protected_exp, 1),
            "expneg": (_gplearn_expneg, 1),
            "n2": (_gplearn_square, 1),
            "n3": (_gplearn_cube, 1),
            "tanh": (_gplearn_tanh, 1),
        }
        requested_custom = set(params.get("function_set", ())) & set(custom_functions)
        if requested_custom:
            from gplearn.functions import make_function

            resolved = {
                function_name: make_function(
                    function=custom_functions[function_name][0],
                    name=function_name,
                    arity=custom_functions[function_name][1],
                    wrap=True,
                )
                for function_name in requested_custom
            }
            params["function_set"] = [
                resolved.get(function, function)
                for function in params["function_set"]
            ]
    module, attr = MODEL_CLASSES[name]
    model = getattr(importlib.import_module(module), attr)(**params)
    if name == "e2e":
        try:
            model.check_available()
        except RuntimeError as exc:
            raise SkipCase(str(exc)) from exc
    return model


def _physo_expression_text(expression, aliases=None):
    """Return a machine-readable PhySO expression with fitted constants.

    ``aliases`` maps the short input names given to PhySO back to the declared
    variable names.
    """
    if getattr(expression, "free_consts", None) is not None:
        expression = copy.copy(expression)
        expression.free_consts = copy.copy(expression.free_consts)
        for name in ("class_values", "spe_values"):
            values = getattr(expression.free_consts, name, None)
            if hasattr(values, "detach"):
                setattr(expression.free_consts, name, values.detach().cpu())
    symbolic = expression.get_infix_sympy(do_simplify=True, evaluate_consts=True)
    if hasattr(symbolic, "flat"):
        symbolic = next(iter(symbolic.flat))
    elif isinstance(symbolic, (list, tuple)):
        symbolic = symbolic[0]
    # Observed inputs are real. Expand only identities justified by those
    # assumptions, so a fitted log(k * exp(a*x)) has an ordinary affine
    # export when k is positive. Never force logarithm branch identities.
    import sympy as sp

    if isinstance(symbolic, sp.Basic):
        if aliases:
            symbolic = symbolic.xreplace({
                symbol: sp.Symbol(aliases[symbol.name], **symbol.assumptions0)
                for symbol in symbolic.free_symbols if symbol.name in aliases
            })
        real_symbols = {
            symbol: sp.Symbol(symbol.name, real=True)
            for symbol in symbolic.free_symbols
            if symbol.is_real is None
        }
        symbolic = sp.expand_log(symbolic.xreplace(real_symbols), force=False)
    return str(symbolic)


def _score_regression(result, target, predict):
    """Retain the discovered expression when held-out evaluation fails."""
    try:
        prediction = predict()
    except Exception as exc:
        result.update(status="error", prediction_status="invalid",
                      error=f"Prediction failed: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return result
    try:
        result.update(regression_metrics(target, prediction))
        coverage = result.get("prediction_finite_coverage")
        if coverage is not None and coverage < 1:
            result.update(
                status="error",
                prediction_status="invalid",
                error=f"Prediction failed: finite prediction coverage is {coverage:.6g}",
            )
    except Exception as exc:
        result.update(evaluation_status="evaluator_error",
                      evaluation_error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
    return result


def _apply_information_criteria(result, configuration):
    """Apply explicitly configured IC evaluation; never infer a parameter count."""
    if not configuration:
        return result
    from kd.evaluation import information_criteria

    if "n_free_parameters" not in configuration:
        result.update(information_criterion_status="abstained_missing_parameter_count")
        return result
    if configuration.get("likelihood_data_role") != "training" or configuration.get("parameter_estimation") != "maximum_likelihood":
        result.update(information_criterion_status="abstained_missing_training_mle_provenance")
        return result
    if "n_observations" not in configuration or not any(key in configuration for key in ("log_likelihood", "residual_sum_squares")):
        result.update(information_criterion_status="abstained_missing_likelihood_inputs")
        return result
    try:
        values = information_criteria(
            n_observations=configuration["n_observations"],
            n_free_parameters=configuration["n_free_parameters"],
            log_likelihood=configuration.get("log_likelihood"),
            residual_sum_squares=configuration.get("residual_sum_squares"),
            likelihood_model=configuration.get("likelihood_model"),
        )
        result.update(values, information_criterion_status="ok")
        result["information_criterion_data_role"] = "training"
        result["information_criterion_provenance"] = dict(configuration)
    except (TypeError, ValueError) as exc:
        result.update(information_criterion_status="abstained_invalid_provenance",
                      information_criterion_error=str(exc))
    return result


def _model_stop_reason(model):
    """Prefer explicit adapter evidence; label uninstrumented native returns."""
    reason = getattr(model, "stop_reason_", None)
    if reason in STOP_REASONS:
        return {"stop_reason": reason,
                "stop_reason_source": getattr(model, "stop_reason_source_", "adapter")}
    details = getattr(model, "run_details_", None)
    if isinstance(details, dict) and details.get("generation"):
        if len(details["generation"]) < getattr(model, "generations", 0):
            return {"stop_reason": "early_stop", "stop_reason_source": "generation_count"}
    return {"stop_reason": "budget_exhausted", "stop_reason_source": "native_return"}


def _finalize_stop_reason(row):
    if row.get("resource_limit") or row.get("stop_reason") == "resource_limit":
        row.update(status="error", stop_reason="resource_limit", stop_reason_source="runner")
    elif row["status"] != "ok":
        row["stop_reason"] = "hard_timeout" if row["status"] == "timeout" else "error"
        row["stop_reason_source"] = "runner"
    elif row.get("stop_reason") not in STOP_REASONS or row.get("stop_reason") == "error":
        row.update(stop_reason="budget_exhausted", stop_reason_source="native_return")
    return row


def _train_dscv_regression(problem, case, params):
    from kd.data import RegularData

    model = _new_model("dscv", params)
    model.data_class = RegularData(case["dataset"])
    model.data_class.load_regression_data(
        X=problem["X_train"],
        y=problem["y_train"],
        variable_names=problem["variable_names"],
        n_input_dim=problem["X_train"].shape[1],
    )
    model.dataset = case["dataset"]
    model.config_task.update(
        task_type="symbolic_regression", dataset=model.dataset, spatial_error=False
    )
    if model.out_path is not None:
        model.out_path = str(Path(model.out_path) / "discover.csv")
    model.setup()
    # DISCOVER's PDE executor indexes x[variable], so its regression bridge
    # must expose a list of feature columns, not rows of the original matrix.
    from kd.model.kd_dscv import Program

    Program.task.x = [problem["X_train"][:, i : i + 1] for i in range(problem["X_train"].shape[1])]
    result = fit_call(model.train, n_epochs=case["run_params"]["dscv_epochs"], verbose=False)
    return {
        "expression": str(result.get("expression", "")),
        "reward": result.get("r"),
        **_model_stop_reason(model),
        "notes": "DISCOVER regression bridge; no fixed-coefficient held-out prediction API. test_mse is unavailable.",
    }


def run_regression(problem, case, *, pretrain_cache=None):
    model = None
    name, params = case["model"], copy.deepcopy(case["model_params"])
    if name in {"gplearn", "physo", "pyoperon", "pysr", "dso", "dscv", "spr", "symbolicgpt", "sisso"}:
        params = _dataset_search_space(
            params, problem["metadata"], name, problem.get("ground_truth")
        )
    if name == "dscv":
        result = _train_dscv_regression(problem, case, params)
    elif name == "physo":
        import physo
        import physo.learn.monitoring as monitoring
        import torch

        options = dict(params)
        # PhySO's upstream documentation recommends CPU expression/constant
        # evaluation. The backend can use CPU inside the allocated GPU slot;
        # it keeps the same four-core limit and reserves that slot's GPU.
        options["device"] = options.pop("execution_device", options.get("device", "cpu"))
        # PhySO rejects token names of 10+ characters (MAX_NAME_SIZE), e.g. the
        # SIR state "susceptible". Alias only those names; the exported
        # expression maps them back so scoring sees the declared names.
        names = list(problem["variable_names"])
        aliases = {}
        for index, variable in enumerate(names):
            if len(variable) >= 10:
                alias = f"x{index}"
                while alias in names or alias in aliases:
                    alias = "_" + alias
                aliases[alias] = variable
        reverse = {variable: alias for alias, variable in aliases.items()}
        options.setdefault("X_names", [reverse.get(variable, variable) for variable in names])
        options.setdefault("y_name", "y")
        options.setdefault("y_units", [0, 0, 0])
        options.setdefault(
            "fixed_consts_units", [[0, 0, 0] for _ in options.get("fixed_consts", [])]
        )
        options.setdefault(
            "free_consts_units", [[0, 0, 0] for _ in options.get("free_consts_names", [])]
        )
        run_config_overrides = options.pop("run_config_overrides", {})
        preset_name = options.pop("run_config_preset", "config0")
        if preset_name not in {"config0", "config1", "config2"}:
            raise ValueError(f"Unknown PhySO run_config_preset: {preset_name!r}")
        preset = getattr(getattr(physo.config, preset_name), preset_name)
        run_config = copy.deepcopy(options.get("run_config", preset))
        _deep_update(run_config, run_config_overrides)
        options["run_config"] = run_config
        options.setdefault(
            "get_run_logger", lambda: monitoring.RunLogger(save_path="physo.log", do_save=True)
        )
        options.setdefault(
            "get_run_visualiser",
            lambda: monitoring.RunVisualiser(
                epoch_refresh_rate=1, do_show=False, do_prints=False, do_save=False
            ),
        )
        expr, logger = fit_call(physo.SR, problem["X_train"].T, problem["y_train"], **options)
        if expr is None:
            raise RuntimeError("PhySO did not return an expression")
        result = _score_regression(
            {
                "expression": _physo_expression_text(expr, aliases),
                "notes": "Dimensionless PhySO configuration; physical units require explicit configuration.",
                **_model_stop_reason(logger),
            },
            problem["y_test"],
            lambda: expr(
                torch.as_tensor(problem["X_test"].T, dtype=torch.float32, device=options["device"])
            )
            .detach()
            .cpu()
            .numpy(),
        )
    elif name == "pysr":
        model = _new_model(name, params)
        fit_call(model.fit,
            problem["X_train"],
            problem["y_train"],
            variable_names=problem["variable_names"],
        )
        result = _score_regression(
            {
                "expression": str(model.sympy()),
                "notes": "PySR uses Julia CPU workers; CUDA is not used by this baseline.",
            },
            problem["y_test"],
            lambda: model.predict(problem["X_test"]),
        )
    else:
        if name in {"sindy", "pysindy"}:
            params.pop("derivative_order", None)
            params.pop("trim_boundary", None)
        if name == "symbolicgpt":
            reuse_pretraining = params.pop("reuse_pretraining_within_case", False)
        model = _new_model(name, params)
        if name == "llmsr":
            fit_call(model.fit,
                problem["X_train"],
                problem["y_train"],
                variable_names=problem["variable_names"],
                target_name=problem["target"],
            )
        elif name in {"sindy", "pysindy", "e2e", "sisso"}:
            fit_call(model.fit,
                problem["X_train"],
                problem["y_train"],
                variable_names=problem["variable_names"],
            )
        elif name == "symbolicgpt" and reuse_pretraining and pretrain_cache is not None:
            fit_call(model.fit, problem["X_train"], problem["y_train"], pretrain_cache=pretrain_cache)
        else:
            fit_call(model.fit, problem["X_train"], problem["y_train"])
        if name == "gplearn":
            expression = str(model._program)
        elif name == "pyoperon":
            expression = model.get_model_string(
                model.model_, precision=8, names=problem["variable_names"]
            )
        else:
            expression = str(model.best_expression_)
        result = _score_regression(
            {
                "expression": expression,
                "notes": (
                    "Classic polynomial-library SINDy with STLSQ."
                    if name == "sindy"
                    else (
                        "PySINDy polynomial library with the SR3 optimizer."
                        if name == "pysindy"
                        else (
                            "LLM-SR program-skeleton search with numerically fitted constants."
                            if name == "llmsr"
                            else (
                                "Operon C++ genetic programming through its scikit-learn API."
                                if name == "pyoperon"
                                else None
                            )
                        )
                    )
                ),
            },
            problem["y_test"],
            lambda: model.predict(problem["X_test"]),
        )
    for key, value in _model_stop_reason(model).items():
        result.setdefault(key, value)
    result.update(
        target=problem["target"],
        split=problem["split"],
        n_train=len(problem["X_train"]),
        n_test=len(problem["X_test"]),
        variable_names=problem["variable_names"],
        details=problem["metadata"],
    )
    if name in {"gplearn", "physo", "pyoperon", "pysr", "dso", "dscv", "spr", "symbolicgpt"}:
        operators = (
            params.get("allowed_symbols", "").split(",")
            if name == "pyoperon"
            else (
                {"binary": params.get("binary_operators", []),
                 "unary": params.get("unary_operators", [])}
                if name in {"pysr", "dscv", "spr"}
                else params.get(
                    "function_set" if name in {"gplearn", "dso"}
                    else "op_list" if name == "symbolicgpt" else "op_names", []
                )
            )
        )
        result["search_space"] = {
            "policy": (
                case["model_params"].get("search_space_policy", "declared")
                if case["model_params"].get("use_dataset_function_set") and
                problem["metadata"].get("benchmark_function_set")
                else "configured_fallback"
            ),
            "operators": operators,
            "declared_function_set": problem["metadata"].get("benchmark_function_set"),
        }
        if name == "physo":
            result["search_space"]["fixed_consts"] = params.get("fixed_consts", [])
            result["search_space"]["free_consts_names"] = params.get("free_consts_names", [])
            result["search_space"]["execution_device"] = options["device"]
    if name == "e2e":
        result["method_provenance"] = model.provenance_
    if name == "sisso":
        result["notes"] = "SISSO++ dimensionless single-task regression; native hard timeout, CPU backend."
        result["search_space"] = {
            "policy": ("declared" if case["model_params"].get("use_dataset_function_set")
                       and problem["metadata"].get("benchmark_function_set") else "configured_fallback"),
            "operators": params["allowed_ops"], "max_rung": params["max_rung"], "n_dim": params["n_dim"],
            "selected_dimension": model.selected_dimension_,
            "effective_n_sis_select": model.effective_n_sis_select_,
        }
    if name == "llmsr":
        result["method_provenance"] = dict(model.search_stats_)
        result["best_program"] = model.best_program_
    if name == "symbolicgpt":
        result["method_provenance"] = dict(model.pretraining_)
        result["candidate_validation"] = dict(model.candidate_validation_)
        result["unique_accepted_skeletons"] = len({row[1] for row in model.candidates_})
    ic_config = case.get("evaluation", {}).get("information_criteria")
    if isinstance(ic_config, dict) and isinstance(ic_config.get("targets"), dict):
        ic_config = ic_config["targets"].get(problem["target"])
    _apply_information_criteria(result, ic_config)
    result.update(
        expression_metrics(
            result.get("expression"),
            problem.get("ground_truth"),
            problem["variable_names"],
        )
    )
    if case["task"] == "ode":
        result["metric_target"] = "state_time_derivative"
    if case["dataset"].startswith("fde_"):
        result["metric_target"] = "observed_response"
    if case["dataset"] == "pyqg_subgrid":
        from kd.evaluation.closure import finish
        result = finish(result, problem, model)
    return result


def _slice_pde_time(dataset, start, stop):
    """Compatibility entrypoint for the shared grid slicing helper."""
    return slice_pde_time(dataset, start, stop)


def _linear_term_map(expression, variable_names=()):
    """Map each additive symbolic term to its numeric coefficient."""
    import sympy as sp

    parsed = _pde_rhs_sympy(expression, variable_names)
    if parsed is None:
        return None
    result = {}
    for term in sp.Add.make_args(sp.expand(parsed)):
        coefficient, symbolic = term.as_coeff_Mul()
        if not coefficient.is_number:
            return None
        key = str(sp.factor(symbolic))
        result[key] = result.get(key, 0.0) + float(coefficient)
    return {key: value for key, value in result.items() if abs(value) > 1e-12}


def _pde_rhs_sympy(expression, variable_names=()):
    """Normalize both explicit RHS and implicit ``F(..., u_t)=0`` equations."""
    import sympy as sp

    if expression is None:
        return None
    text = str(expression).strip()
    if "=" in text:
        left, right = text.split("=", 1)
        if re.fullmatch(r"\s*(?:u_t|ut|d\(u\d*\)/dt)\s*", left):
            parsed = expression_sympy(right, variable_names)
        else:
            parsed = expression_sympy(f"({left}) - ({right})", variable_names)
    else:
        parsed = expression_sympy(text, variable_names)
    if parsed is None:
        return None
    time_derivative = sp.Symbol("u_t")
    if time_derivative in parsed.free_symbols:
        try:
            solutions = sp.solve(sp.Eq(parsed, 0), time_derivative)
            return solutions[0] if len(solutions) == 1 else None
        except (NotImplementedError, TypeError, ValueError):
            return None
    return parsed


def pde_pic_candidate(expression):
    """Translate a linear scalar PDE RHS into the bounded PIC term grammar.

    Coefficients are returned in the exact same order as term names.  The
    conversion is deliberately strict: an unsupported nonlinear composition is
    an evaluator abstention, never a silently simplified or dropped term.
    """
    import sympy as sp

    from kd.metrics import SUPPORTED_TERMS

    parsed = _pde_rhs_sympy(expression)
    if parsed is None:
        raise ValueError("PIC requires a parseable explicit scalar u_t equation")
    symbols = {name: sp.Symbol(name) for name in ("u", "u_x", "u_xx", "u_xxx")}
    templates = {
        "1": sp.Integer(1),
        "u": symbols["u"],
        "u_x": symbols["u_x"],
        "u_xx": symbols["u_xx"],
        "u_xxx": symbols["u_xxx"],
        "u*u_x": symbols["u"] * symbols["u_x"],
        "u*u_xx": symbols["u"] * symbols["u_xx"],
        "u*u_xxx": symbols["u"] * symbols["u_xxx"],
        "u^2": symbols["u"] ** 2,
        "u^3": symbols["u"] ** 3,
    }
    coefficients = {name: 0.0 for name in SUPPORTED_TERMS}
    for additive_term in sp.Add.make_args(sp.expand(parsed)):
        coefficient, symbolic = additive_term.as_coeff_Mul()
        if not coefficient.is_number or not coefficient.is_real:
            raise ValueError(f"PIC term has a non-numeric coefficient: {additive_term}")
        matches = [
            name for name, template in templates.items()
            if sp.simplify(symbolic - template) == 0
        ]
        if len(matches) != 1:
            raise ValueError(f"PIC does not support candidate term: {symbolic}")
        coefficients[matches[0]] += float(coefficient)
    terms = tuple(name for name in SUPPORTED_TERMS if abs(coefficients[name]) > 1e-12)
    if not terms:
        raise ValueError("PIC candidate has no nonzero supported RHS terms")
    return terms, tuple(coefficients[name] for name in terms)


def _pic_training_indices(nx, nt, maximum, seed):
    """Bound ANN observations while retaining every coordinate-domain extreme."""
    import numpy as np

    total = int(nx) * int(nt)
    if maximum is None or int(maximum) >= total:
        return np.arange(total, dtype=int)
    maximum = int(maximum)
    if maximum < 4:
        raise ValueError("physics_informed_pic.max_observations must be at least 4")
    mandatory = np.unique(np.asarray([0, nt - 1, (nx - 1) * nt, total - 1], dtype=int))
    remaining = np.setdiff1d(np.arange(total, dtype=int), mandatory, assume_unique=True)
    count = maximum - len(mandatory)
    sampled = np.random.default_rng(seed).choice(remaining, count, replace=False)
    return np.sort(np.r_[mandatory, sampled])


def _array_list(value):
    return None if value is None else value.tolist()


def _apply_physics_informed_pic(result, train_dataset, case):
    """Apply the costly PDE PIC evaluator only when explicitly configured."""
    import numpy as np

    requested = case.get("evaluation", {}).get("physics_informed_pic")
    if requested is None:
        result["pic_status"] = "not_requested"
        return
    if not isinstance(requested, dict):
        result.update(pic_status="invalid_configuration",
                      pic_message="physics_informed_pic must be a JSON object")
        return
    if not requested.get("enabled", True):
        result["pic_status"] = "disabled"
        return
    try:
        from dataclasses import asdict

        from kd.metrics import TorchPICConfig, evaluate_torch_pic, prepare_torch_pic_reference

        terms, original_coefficients = pde_pic_candidate(result.get("expression"))
        backend = copy.deepcopy(requested.get("backend", {}))
        if not isinstance(backend, dict):
            raise ValueError("physics_informed_pic.backend must be a JSON object")
        backend.setdefault("seed", int(case["seed"]))
        config = TorchPICConfig(**backend)
        if train_dataset.n_response != 1 or len(train_dataset.coords_spatial) != 1:
            raise ValueError("PIC supports only one scalar field on one spatial dimension")
        x = np.asarray(train_dataset.x, dtype=float)
        t = np.asarray(train_dataset.t, dtype=float)
        field = np.asarray(
            train_dataset.usol if train_dataset.legacy else train_dataset.usol[0], dtype=float
        )
        if field.shape != (len(x), len(t)):
            raise ValueError("PIC field shape must be (n_x, n_t)")
        coordinates = np.stack(np.meshgrid(x, t, indexing="ij"), axis=-1).reshape(-1, 2)
        values = field.reshape(-1)
        indices = _pic_training_indices(
            len(x), len(t), requested.get("max_observations"), case["seed"]
        )
        configured_cache = requested.get("cache_dir", "results/benchmark/pic_reference_cache")
        cache_dir = None if configured_cache is None else _project_path(configured_cache)
        prepared = prepare_torch_pic_reference(
            coordinates, values, train_indices=indices, config=config, cache_dir=cache_dir
        )
        evaluated = evaluate_torch_pic(
            prepared, terms, candidate_id=str(result.get("expression")),
            original_coefficients=original_coefficients,
        )
        result.update(
            pic_status=evaluated.status,
            pic=evaluated.pic,
            pic_r_loss=evaluated.r_loss,
            pic_p_loss=evaluated.p_loss,
            pic_message=evaluated.message,
            pic_version=evaluated.version,
            pic_cache_key=evaluated.cache_key,
            pic_terms=list(terms),
            pic_original_coefficients=_array_list(evaluated.original_coefficients),
            pic_reference_fit_coefficients=_array_list(evaluated.reference_fit_coefficients),
            pic_window_coefficients=_array_list(evaluated.window_coefficients),
            pic_refitted_coefficients=_array_list(evaluated.refitted_coefficients),
            pic_cost=dict(evaluated.cost),
            pic_reference_cache_hit=bool(evaluated.cost.get("reference_cache_hit", 0)),
            pic_reference_train_rmse=prepared.reference_train_rmse,
            pic_reference_train_normalized_rmse=prepared.reference_train_normalized_rmse,
            pic_config=asdict(config),
            pic_training_observations=int(len(indices)),
        )
    except Exception as exc:  # PIC is auxiliary; evaluator defects must not fail the baseline row.
        status = "unsupported_candidate" if isinstance(exc, ValueError) and (
            "candidate" in str(exc).lower() or "term" in str(exc).lower()
        ) else "evaluator_error"
        result.update(pic_status=status, pic_message=f"{type(exc).__name__}: {exc}")


def pde_structure_metrics(expression, ground_truth):
    """Compare discovered/true additive supports; coefficients need an explicit true RHS."""
    import numpy as np

    result = {
        "ground_truth_expression": ground_truth,
        "pde_support_recovery": None,
        "exact_symbolic_recovery": None,
        "exact_symbolic_recovery_semantics": "legacy alias of pde_support_recovery",
        "pde_support_metric_version": PDE_SUPPORT_METRIC_VERSION,
        "recovery_evaluator_status": "not_applicable" if not ground_truth else "unparsed",
        "term_support_accuracy": None,
        "term_support_precision": None,
        "term_support_recall": None,
        "coefficient_error": None,
        "coefficient_recovery": None,
        "coefficient_recovery_rtol": 0.05,
        "coefficient_recovery_atol": 1e-8,
    }
    discovered = _linear_term_map(expression)
    truth = _linear_term_map(ground_truth)
    if discovered is None or truth is None:
        return result
    discovered_support, true_support = set(discovered), set(truth)
    union = discovered_support | true_support
    intersection = discovered_support & true_support
    result.update(
        pde_support_recovery=int(discovered_support == true_support),
        exact_symbolic_recovery=int(discovered_support == true_support),
        recovery_evaluator_status="parsed",
        term_support_accuracy=(len(intersection) / len(union) if union else 1.0),
        term_support_precision=(
            len(intersection) / len(discovered_support) if discovered_support else 0.0
        ),
        term_support_recall=(len(intersection) / len(true_support) if true_support else 0.0),
    )
    # Registry prefix strings encode structure but omit physical coefficients.
    if ground_truth and "," not in str(ground_truth):
        keys = sorted(union)
        true_coefficients = np.asarray([truth.get(key, 0.0) for key in keys])
        found_coefficients = np.asarray([discovered.get(key, 0.0) for key in keys])
        denominator = float(np.linalg.norm(true_coefficients))
        if denominator > 0:
            result["coefficient_error"] = float(
                np.linalg.norm(found_coefficients - true_coefficients) / denominator
            )
            result["coefficient_recovery"] = int(
                np.allclose(found_coefficients, true_coefficients, rtol=0.05, atol=1e-8)
            )
    return result


def pde_equation_residual(expression, dataset):
    """Evaluate a scalar discovered RHS against finite-difference u_t on held-out frames."""
    import numpy as np
    import sympy as sp

    if dataset.n_response != 1:
        return {}
    field = np.asarray(dataset.usol if dataset.legacy else dataset.usol[0], dtype=float)
    if field.shape[-1] < 3:
        return {}
    namespace = {"u": field, "u1": field}
    fde = bool(getattr(dataset, "protocol_metadata", {}).get("pde_protocol_version"))
    axes = np.meshgrid(*dataset.coords_spatial.values(), dataset.t, indexing="ij")
    for index, (name, values) in enumerate(zip(dataset.coords_spatial, axes)):
        namespace.update({name: values, "xyz"[index]: values, f"x{index + 1}": values})
    namespace["t"] = axes[-1]
    for axis, (name, coordinate) in enumerate(dataset.coords_spatial.items()):
        suffix = {"x1": "x", "x2": "y", "x3": "z"}.get(name, name)
        first = np.gradient(field, coordinate, axis=axis, edge_order=2)
        second = np.gradient(first, coordinate, axis=axis, edge_order=2)
        third = np.gradient(second, coordinate, axis=axis, edge_order=2)
        namespace[f"u_{suffix}"] = first
        namespace[f"u_{suffix * 2}"] = second
        namespace[f"u_{suffix * 3}"] = third
    if len(dataset.spatial_vars) >= 2:
        first_name = {"x1": "x", "x2": "y"}.get(dataset.spatial_vars[0], dataset.spatial_vars[0])
        second_name = {"x1": "x", "x2": "y"}.get(dataset.spatial_vars[1], dataset.spatial_vars[1])
        namespace[f"u_{first_name}{second_name}"] = np.gradient(
            namespace[f"u_{first_name}"],
            dataset.coords_spatial[dataset.spatial_vars[1]],
            axis=1,
            edge_order=2,
        )
    parsed = _pde_rhs_sympy(expression, tuple(namespace))
    if parsed is None:
        return {"equation_residual_status": "abstain_unsupported_expression"}
    free = sorted(parsed.free_symbols, key=str)
    if any(str(symbol) not in namespace for symbol in free):
        return {"equation_residual_status": "abstain_unknown_symbols"}
    if parsed.has(sp.zoo, sp.oo, -sp.oo, sp.nan):
        return {"equation_residual_status": "nonfinite_candidate",
                "equation_residual_points": 0,
                "equation_residual_total_points": int(field[..., (0 if fde else 1):].size),
                "equation_residual_finite_coverage": 0.0}
    try:
        rhs = sp.lambdify(free, parsed, modules="numpy")(
            *(namespace[str(symbol)] for symbol in free)
        )
        rhs = np.broadcast_to(np.asarray(rhs, dtype=float), field.shape)
    except (TypeError, ValueError, ZeroDivisionError, FloatingPointError):
        return {"equation_residual_status": "abstain_evaluation_error"}
    ut = np.gradient(field, dataset.t, axis=-1, edge_order=2)
    # Legacy views share a rollout initial frame. FDE blocks are disjoint and
    # every test frame belongs to the scoring domain.
    start = 0 if fde else 1
    target, prediction = ut[..., start:].reshape(-1), rhs[..., start:].reshape(-1)
    target_finite = np.isfinite(target)
    prediction_finite = np.isfinite(prediction)
    finite = target_finite & prediction_finite
    total = int(target.size)
    result = {
        "equation_residual_points": int(finite.sum()),
        "equation_residual_total_points": total,
        "equation_residual_finite_coverage": float(finite.sum() / total) if total else None,
    }
    result["equation_residual_status"] = "ok" if finite.all() else "nonfinite_candidate"
    # A partially non-finite candidate has not made predictions over the full
    # declared scoring domain.  Keep diagnostic coverage, but do not award an
    # error score on the easier finite subset.
    if not finite.all():
        return result
    scores = regression_metrics(target[finite], prediction[finite])
    result.update({
        "equation_residual_mse": scores["test_mse"],
        "equation_residual_nrmse": scores["test_nrmse"],
    })
    return result


def pde_rollout_metrics(model, dataset, split):
    """Autoregressively predict every held-out frame from the last training frame."""
    import numpy as np

    state = np.asarray(dataset.usol[..., split - 1], dtype=float)
    truth = np.asarray(dataset.usol[..., split:], dtype=float)
    predictions = []
    for index in range(split, len(dataset.t)):
        dt = float(dataset.t[index] - dataset.t[index - 1])
        state = np.asarray(model.predict(state, dt), dtype=float)
        if state.shape != truth[..., index - split].shape or not np.isfinite(state).all():
            raise ValueError("Rollout prediction has an invalid shape or non-finite values")
        predictions.append(state)
    prediction = np.stack(predictions, axis=-1)
    scores = regression_metrics(truth, prediction)
    return {
        "native_rollout_steps": len(predictions),
        "native_rollout_mse": scores["test_mse"],
        "native_rollout_nrmse": scores["test_nrmse"],
        "native_rollout_r2": scores["test_r2"],
        "native_rollout_source": "model_predict",
    }


def _symbolic_pde_rhs(expression, state, dataset):
    """Evaluate a discovered scalar PDE RHS on one spatial state."""
    import numpy as np
    import sympy as sp

    if dataset.n_response != 1:
        raise ValueError("Symbolic rollout currently supports one response field")
    state = np.asarray(state, dtype=float)
    namespace = {"u": state, "u1": state}
    for axis, (name, coordinate) in enumerate(dataset.coords_spatial.items()):
        suffix = {"x1": "x", "x2": "y", "x3": "z"}.get(name, name)
        first = np.gradient(state, coordinate, axis=axis, edge_order=2)
        second = np.gradient(first, coordinate, axis=axis, edge_order=2)
        third = np.gradient(second, coordinate, axis=axis, edge_order=2)
        namespace[f"u_{suffix}"] = first
        namespace[f"u_{suffix * 2}"] = second
        namespace[f"u_{suffix * 3}"] = third
    if len(dataset.spatial_vars) >= 2:
        first_name = {"x1": "x", "x2": "y"}.get(dataset.spatial_vars[0], dataset.spatial_vars[0])
        second_name = {"x1": "x", "x2": "y"}.get(dataset.spatial_vars[1], dataset.spatial_vars[1])
        namespace[f"u_{first_name}{second_name}"] = np.gradient(
            namespace[f"u_{first_name}"],
            dataset.coords_spatial[dataset.spatial_vars[1]],
            axis=1,
            edge_order=2,
        )
    parsed = _pde_rhs_sympy(expression, tuple(namespace))
    if parsed is None:
        raise ValueError("Expression cannot be converted to an explicit PDE RHS")
    free = sorted(parsed.free_symbols, key=str)
    if any(str(symbol) not in namespace for symbol in free):
        raise ValueError("Expression contains unsupported functions or symbols")
    rhs = sp.lambdify(free, parsed, modules="numpy")(*(namespace[str(symbol)] for symbol in free))
    return np.broadcast_to(np.asarray(rhs, dtype=float), state.shape)


def _spectral_pde_rhs(expression, state, dataset):
    """Evaluate a scalar RHS using periodic FFT derivatives on an endpoint-excluded grid."""
    import numpy as np
    import sympy as sp

    state = np.asarray(state, dtype=float)
    n = len(state)
    dx = float(dataset.x[1] - dataset.x[0])
    wave = 2 * np.pi * np.fft.fftfreq(n, d=dx)
    transform = np.fft.fft(state)
    namespace = {"u": state, "u1": state}
    for order, suffix in ((1, "x"), (2, "xx"), (3, "xxx")):
        namespace[f"u_{suffix}"] = np.fft.ifft((1j * wave) ** order * transform).real
    parsed = _pde_rhs_sympy(expression, tuple(namespace))
    if parsed is None or any(str(symbol) not in namespace for symbol in parsed.free_symbols):
        raise ValueError("expression cannot be evaluated by the periodic scalar PDE evaluator")
    free = sorted(parsed.free_symbols, key=str)
    value = sp.lambdify(free, parsed, modules="numpy")(
        *(namespace[str(symbol)] for symbol in free)
    )
    return np.broadcast_to(np.asarray(value, dtype=float), state.shape)


def symbolic_pde_rollout_metrics(expression, dataset, split, metadata=None,
                                 reference_expression=None):
    """Validated shared method-of-lines equation rollout, or explicit abstention."""
    import numpy as np

    from kd.evaluation.pde_rollout import rollout, validate_solver

    metadata = metadata or getattr(dataset, "rollout_metadata", None)
    if not metadata:
        return {"equation_rollout_status": "abstained_missing_metadata"}
    required = {"boundary_conditions", "endpoint_convention", "spatial_discretization",
                "solver", "reference_nrmse_tolerance"}
    missing = sorted(required - set(metadata))
    if missing:
        return {"equation_rollout_status": "abstained_missing_metadata",
                "equation_rollout_error": f"missing: {', '.join(missing)}"}
    if metadata["boundary_conditions"] != {"type": "periodic"}:
        return {"equation_rollout_status": "abstained_unsupported_boundary"}
    if metadata["endpoint_convention"] != "periodic_endpoint_excluded":
        return {"equation_rollout_status": "abstained_unsupported_endpoint_convention"}
    if metadata["spatial_discretization"] != "spectral_fft":
        return {"equation_rollout_status": "abstained_unsupported_discretization"}
    if len(dataset.spatial_vars) != 1 or dataset.n_response != 1 or not np.allclose(
        np.diff(dataset.x), np.diff(dataset.x)[0], rtol=1e-10, atol=1e-12
    ):
        return {"equation_rollout_status": "abstained_unsupported_grid"}
    if not reference_expression:
        return {"equation_rollout_status": "abstained_missing_reference"}

    state = np.asarray(dataset.usol[..., split - 1], dtype=float)
    truth = np.asarray(dataset.usol[..., split:], dtype=float)
    times = np.asarray(dataset.t[split - 1 :], dtype=float)
    solver = dict(metadata["solver"])
    if not {"method", "rtol", "atol", "max_step"} <= set(solver):
        return {"equation_rollout_status": "abstained_missing_solver_settings"}
    reference = validate_solver(
        lambda values, _x, _bc: _spectral_pde_rhs(reference_expression, values, dataset),
        state, dataset.x, times, boundary_conditions=metadata["boundary_conditions"],
        reference=np.asarray(dataset.usol[..., split - 1 :], dtype=float),
        nrmse_tolerance=float(metadata["reference_nrmse_tolerance"]), **solver
    )
    if not reference["validation_passed"]:
        return {"equation_rollout_status": "abstained_reference_validation_failed",
                "equation_rollout_reference_nrmse": reference["validation_nrmse"]}
    candidate = rollout(
        lambda values, _x, _bc: _spectral_pde_rhs(expression, values, dataset),
        state, dataset.x, times, boundary_conditions=metadata["boundary_conditions"], **solver
    )
    prediction = candidate["prediction"][:, 1:]
    scores = regression_metrics(truth, prediction)
    return {
        "equation_rollout_status": "ok",
        "equation_rollout_steps": truth.shape[-1],
        "equation_rollout_mse": scores["test_mse"],
        "equation_rollout_nrmse": scores["test_nrmse"],
        "equation_rollout_r2": scores["test_r2"],
        "equation_rollout_integrator": candidate["solver"],
        "equation_rollout_spatial_discretization": "spectral_fft",
        "equation_rollout_boundary_conditions": metadata["boundary_conditions"],
        "equation_rollout_reference_nrmse": reference["validation_nrmse"],
        "equation_rollout_solver_settings": solver,
        "equation_rollout_reference_tolerance": metadata["reference_nrmse_tolerance"],
        "equation_rollout_endpoint_convention": metadata["endpoint_convention"],
    }


def pde_coordinates(dataset):
    """Flatten x,t in the same order as u; DeepMoD needs t,x columns later."""
    import numpy as np

    x, t = np.meshgrid(dataset.x, dataset.t, indexing="ij")
    return np.column_stack([x.ravel(), t.ravel()]), dataset.usol.reshape(-1, 1)


def pde_sindy_problem(dataset, derivative_order=3, trim_boundary=2):
    """Create a SINDy regression library input from a scalar one-dimensional grid.

    Spatial and time derivatives are computed before trimming so every retained
    sample stays aligned. The SINDy models then build polynomial interactions
    such as ``u*u_x`` from the returned base columns.
    """
    import numpy as np

    derivative_order = int(derivative_order)
    trim_boundary = int(trim_boundary)
    if derivative_order not in {1, 2, 3}:
        raise ValueError("SINDy PDE derivative_order must be one of 1, 2, or 3")
    if trim_boundary < 0:
        raise ValueError("trim_boundary must be non-negative")
    field = np.asarray(dataset.usol, dtype=float)
    if min(field.shape) <= 2 * trim_boundary:
        raise SkipCase("PDE grid is too small for the configured SINDy boundary trim")

    spatial_derivatives = []
    derivative = field
    for _ in range(derivative_order):
        derivative = np.gradient(derivative, dataset.x, axis=0, edge_order=2)
        spatial_derivatives.append(derivative)
    time_derivative = np.gradient(field, dataset.t, axis=-1, edge_order=2)
    region = (
        slice(trim_boundary, -trim_boundary or None),
        slice(trim_boundary, -trim_boundary or None),
    )
    columns = [field[region], *(value[region] for value in spatial_derivatives)]
    suffixes = ["x" * order for order in range(1, derivative_order + 1)]
    names = ["u", *(f"u_{suffix}" for suffix in suffixes)]
    X = np.column_stack([column.ravel() for column in columns])
    y = time_derivative[region].ravel()
    return X, y, names


def _linear_expression(names, coeffs):
    return " + ".join(f"({float(c):.8g})*{n}" for n, c in zip(names, coeffs) if c != 0) or "0"


def run_pde(dataset, case):
    import numpy as np

    name, params, options = case["model"], copy.deepcopy(case["model_params"]), case["run_params"]
    candidate_space_label = params.pop("candidate_space_label", "configured_fallback")
    dataset = prepare_pde(dataset, name)
    fractional_split = None
    if case.get("fde_pde_protocol_version"):
        from kd.evaluation.fractional import temporal_views
        train_dataset, evaluation_dataset, fractional_split = temporal_views(dataset, options["test_size"])
        split_index = fractional_split["test_frame_start"]
    else:
        train_dataset, evaluation_dataset, split_index = temporal_block_holdout(
            dataset, options["test_size"]
        )
    model = None
    if name == "reference":
        from kd.evaluation.reference import abstain
        truth = PDE_REFERENCE_RHS.get(case["dataset"], getattr(dataset, "sym_true", None))
        if not truth:
            return abstain("abstained_missing_reference")
        # The same derivative/residual and rollout code below scores discovered RHS.
        result = {"expression": str(truth), "reference_status": "ok"}
    elif name == "weakform":
        model = _new_model(name, params)
        fit_call(model.fit, train_dataset.usol, train_dataset.x, train_dataset.t)
        result = {
            "expression": "u_t = " + model.best_expression_,
            "weak_expression": model.weak_expression_,
            "train_residual_mse": float(model.weak_residual_mse_),
            "train_relative_residual": float(model.weak_relative_residual_),
            "wsindy_selected_lambda": float(model.selected_lambda_),
            "wsindy_mstls_loss": float(model.mstls_loss_),
            "wsindy_support": model.support_,
            "wsindy_test_function_powers": model.test_function_powers_,
            "wsindy_query_stride": model.query_stride_,
            "wsindy_n_weak_samples": model.n_weak_samples_,
            "method_provenance": model.method_provenance_,
            "notes": "WSINDy-PDE scalar-1D adaptation: convolutional weak derivatives and MSTLS model selection.",
        }
    else:
        sindy_pde_options = None
        if name in {"sindy", "pysindy"}:
            sindy_pde_options = {
                "derivative_order": params.pop("derivative_order", 3),
                "trim_boundary": params.pop("trim_boundary", 2),
            }
        if name == "eqgpt":
            params["choose_validate"] = min(
                params["choose_validate"], max(1, train_dataset.usol.size // 5)
            )
        model = _new_model(name, params)
        if name in {"dscv", "spr"}:
            if name == "spr":
                model.config_pinn["pinn_epoch"] = options["spr_pinn_epochs"]
                model.config_pinn["iter_num"] = options["spr_iterations"]
                original_factory = model.make_pinn_model

                # train() calls setup() again, so budget every created PINN.
                def make_budgeted_pinn():
                    # External data arrive via import_outter_data(); prevent
                    # the constructor loading a different built-in dataset.
                    label = model.config_task["dataset"]
                    try:
                        model.config_task["dataset"] = None
                        pinn = original_factory()
                    finally:
                        model.config_task["dataset"] = label
                    pinn.pretrain_epoch = options["spr_pretrain_epochs"]
                    pinn.pinn_cv_epoch = options["spr_pinn_epochs"]
                    return pinn

                model.make_pinn_model = make_budgeted_pinn
            kwargs = (
                {}
                if name == "dscv"
                else {
                    "sample_ratio": options["sample_ratio"],
                    "colloc_num": options["colloc_num"],
                    "random_state": case["seed"],
                }
            )
            model.import_dataset(train_dataset, **kwargs)
            found = fit_call(model.train, n_epochs=options["dscv_epochs"], verbose=False)
            result = {"expression": str(found.get("expression", "")), "reward": found.get("r")}
        elif name == "pysindy" and getattr(model, "library_kind", None):
            fit_call(model.fit_dataset, train_dataset)
            result = {
                "expression": "u_t = " + str(model.best_expression_),
                "n_train": model.n_train_,
                "notes": "Explicit PDE library; dimensionless SR3 and selected-support least-squares refit.",
            }
        elif name in {"sindy", "pysindy"}:
            X, y, variable_names = pde_sindy_problem(train_dataset, **sindy_pde_options)
            limit = options["max_train_samples"]
            if limit is not None:
                count = max(2, round(len(X) * limit)) if isinstance(limit, float) else int(limit)
            else:
                count = len(X)
            if count < 2:
                raise ValueError("PDE training sample count must be >=2")
            if len(X) > count:
                indices = np.random.default_rng(case["seed"]).choice(len(X), count, replace=False)
                X, y = X[indices], y[indices]
            fit_call(model.fit, X, y, variable_names=variable_names)
            result = {
                "expression": "u_t = " + str(model.best_expression_),
                "n_train": len(X),
                "notes": (
                    "Classic polynomial-library SINDy with STLSQ; derivatives use finite differences."
                    if name == "sindy"
                    else "PySINDy polynomial library with SR3; derivatives use finite differences."
                ),
            }
        elif name == "sga":
            fit_call(model.fit_dataset, train_dataset, problem_name=case["dataset"])
            result = {
                "expression": model.best_expression_,
                "native_expression_display": str(model.best_pde_),
                "equation_export_status": model.equation_export_status_,
                "aic": float(model.best_score_),
            }
            if model.best_expression_ is None:
                result.update(status="error", error="SGA fitted equation export failed: "
                              + model.equation_export_status_)
        elif name in {"dlga", "deepmod"}:
            X, y = pde_coordinates(train_dataset)
            limit = options["dlga_samples"] if name == "dlga" else options["max_train_samples"]
            if limit is not None:
                count = max(2, round(len(X) * limit)) if isinstance(limit, float) else int(limit)
                if count < 2:
                    raise ValueError("PDE training sample count must be >=2")
                idx = np.random.default_rng(case["seed"]).choice(
                    len(X), min(count, len(X)), replace=False
                )
                X, y = X[idx], y[idx]
            if name == "dlga":
                model.pop_size = options["dlga_population"]
                model.n_generations = options["dlga_generations"]
            else:
                # Library1D differentiates time in column 0 and space in column 1.
                X = X[:, [1, 0]]
            fit_call(model.fit, X, y)
            result = {"expression": str(model.best_equation_), "n_train": len(X)}
        elif name == "eqgpt":
            fit_call(model.fit, train_dataset, equation_name=case["dataset"])
            result = {
                "expression": str(model.best_pde_),
                "reward": float(model.best_award_),
                "notes": "EqGPT handbook pretraining excludes the target row only for recognized equation names.",
            }
        else:
            fit_call(model.fit, train_dataset)
            if name == "pdefind":
                expression = "u_t = " + _linear_expression(
                    model._feature_names, model.coefficients()
                )
            else:
                text_output = io.StringIO()
                with contextlib.redirect_stdout(text_output):
                    model.print_model()
                expression = text_output.getvalue().strip()
            result = {
                "expression": expression,
                "notes": "Model fitted only on the temporal training block.",
            }
            if name == "pdefind":
                result[
                    "notes"
                ] += (" Normalized sequential thresholded ridge regression with support debiasing."
                      if getattr(model, "solver", "legacy") == "stridge" else
                      " Current PDEFindModel uses thresholded least squares, not STRidge.")
            else:
                result[
                    "notes"
                ] += " PDE-Net uses measured per-axis spacing and verified periodic padding."
    result.update(_model_stop_reason(model))
    library_fields = {
        "dscv": ("binary_operators", "unary_operators"),
        "spr": ("binary_operators", "unary_operators"),
        "sga": ("depth", "width"),
        "dlga": ("operators", "library_kind", "coordinate_terms"),
        "deepmod": ("poly_order", "diff_order", "library_kind", "coordinate_terms", "training_variant"),
        "pdefind": ("derivative_order", "terms", "library_kind", "coordinate_terms", "stencil", "solver"),
        "pdenet": ("derivative_order",),
        "weakform": ("terms", "polynomial_degree", "max_derivative", "coordinate_terms"),
        "sindy": ("polynomial_degree", "derivative_order"),
        "pysindy": ("polynomial_degree", "derivative_order", "library_kind", "coordinate_terms", "stencil"),
    }
    library = {
        field: case["model_params"][field]
        for field in library_fields.get(name, ())
        if field in case["model_params"]
    }
    if name == "pdefind":
        library["resolved_terms"] = list(model._feature_names)
    result["search_space"] = {"policy": candidate_space_label, "library": library}
    structural_truth = getattr(dataset, "sym_true", None)
    if not structural_truth:
        try:
            from kd.dataset import get_dataset_sym_true

            structural_truth = get_dataset_sym_true(case["dataset"])
        except (KeyError, ValueError):
            structural_truth = None
    ground_truth = PDE_REFERENCE_RHS.get(case["dataset"], structural_truth)
    _apply_physics_informed_pic(result, train_dataset, case)
    result.update(pde_structure_metrics(result.get("expression"), ground_truth))
    if structural_truth and structural_truth != ground_truth:
        result["ground_truth_structure"] = structural_truth
    result["expression_complexity"] = expression_metrics(result.get("expression"))[
        "expression_complexity"
    ]
    if name == "sga" and model.best_expression_ is None:
        result["equation_residual_status"] = "abstain_native_export"
    else:
        result.update(pde_equation_residual(result.get("expression"), evaluation_dataset))
    try:
        if fractional_split is None and model is not None and name in {"pdefind", "pdenet"}:
            result.update(pde_rollout_metrics(model, dataset, split_index))
    except Exception as exc:
        note = f"Native held-out rollout unavailable: {type(exc).__name__}: {exc}"
        result["notes"] = " ".join(filter(None, [result.get("notes"), note]))
    try:
        if fractional_split is not None:
            result.update(
                equation_rollout_status="abstain", prediction_status="abstain",
                prediction_abstain_reason="FDE source boundary and fractional operator are unverified",
            )
        else:
            rollout_metadata = case.get("evaluation", {}).get("equation_rollout")
            result.update(symbolic_pde_rollout_metrics(
                result.get("expression"), dataset, split_index, rollout_metadata, ground_truth
            ))
    except Exception as exc:
        result.update(equation_rollout_status="evaluator_error",
                      equation_rollout_error=f"{type(exc).__name__}: {exc}")
    result.setdefault("n_train", int(np.prod(train_dataset.usol.shape)))
    result.update(
        split="temporal_block_holdout",
        details=copy.deepcopy(getattr(dataset, "protocol_metadata", {})),
        target="PDE",
        n_test=int(np.prod(dataset.usol[..., split_index:].shape)),
        n_train_time=split_index,
        n_test_time=len(dataset.t) - split_index,
        train_time_range=[float(dataset.t[0]), float(dataset.t[split_index - 1])],
        test_time_range=[float(dataset.t[split_index]), float(dataset.t[-1])],
        field_shape=list(dataset.usol.shape),
    )
    if fractional_split is not None:
        result.update(
            split=fractional_split,
            n_train_time=len(train_dataset.t),
            train_time_range=fractional_split["train_time_range"],
            metric_target="observed_equation_residual",
            fractional_operator_evaluated=False,
            recovery_eligible=False, recovery_evaluator_status="not_applicable",
        )
    return result


@contextlib.contextmanager
def working_directory(path):
    previous = Path.cwd()
    path.mkdir(parents=True, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class PeakMemoryMonitor:
    """Sample the worker process tree RSS and CUDA allocator during one target."""

    def __init__(self, device="cpu", interval=0.05):
        self.device = device
        self.interval = interval
        self.peak_memory_mb = None
        self.peak_gpu_memory_mb = None
        self._peak_bytes = 0
        self._stop = threading.Event()
        self._thread = None
        try:
            import psutil

            self._process = psutil.Process()
        except ImportError:
            self._process = None

    def _rss_bytes(self):
        if self._process is not None:
            try:
                processes = [self._process, *self._process.children(recursive=True)]
                return sum(process.memory_info().rss for process in processes)
            except Exception:
                return 0
        if os.name == "nt":
            try:
                import ctypes
                from ctypes import wintypes

                class Counters(ctypes.Structure):
                    _fields_ = [
                        ("cb", wintypes.DWORD),
                        ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t),
                    ]

                counters = Counters()
                counters.cb = ctypes.sizeof(counters)
                handle = ctypes.windll.kernel32.GetCurrentProcess()
                if ctypes.windll.psapi.GetProcessMemoryInfo(
                    handle, ctypes.byref(counters), counters.cb
                ):
                    return int(counters.WorkingSetSize)
            except (AttributeError, OSError):
                return 0
        try:
            pages = int(Path("/proc/self/statm").read_text(encoding="ascii").split()[1])
            return pages * os.sysconf("SC_PAGE_SIZE")
        except (OSError, ValueError, IndexError, AttributeError):
            return 0

    def _sample(self):
        self._peak_bytes = max(self._peak_bytes, self._rss_bytes())

    def _run(self):
        while not self._stop.wait(self.interval):
            self._sample()

    def __enter__(self):
        self._sample()
        if str(self.device).startswith("cuda"):
            try:
                import torch

                torch.cuda.reset_peak_memory_stats(self.device)
            except (ImportError, RuntimeError, ValueError):
                pass
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback_value):
        self._sample()
        self._stop.set()
        self._thread.join(timeout=max(0.1, self.interval * 2))
        if self._peak_bytes:
            self.peak_memory_mb = self._peak_bytes / (1024**2)
        if str(self.device).startswith("cuda"):
            try:
                import torch

                self.peak_gpu_memory_mb = torch.cuda.max_memory_allocated(self.device) / (1024**2)
            except (ImportError, RuntimeError, ValueError):
                pass


def _base_row(case, instance=None, target=None):
    row = {
        key: case[key] for key in ("name", "task", "model", "dataset", "seed", "profile", "device")
    }
    row.update(
        metric_protocol_version=case.get("metric_protocol_version") or metric_protocol_version(case, METRIC_PROTOCOL_VERSION),
        dataset_protocol_version=case.get("dataset_protocol_version", "legacy/unknown"),
        scientific_task=case.get("scientific_task", "legacy/unknown"),
        physical_family=case.get("physical_family", case.get("family", case["task"])),
        dataset_metadata=copy.deepcopy(case.get("dataset_metadata", {})),
        execution_protocol_version=case.get("execution_protocol_version", "legacy/unknown"),
        instance=instance or case["dataset"],
        family=case.get("physical_family", case.get("family", case.get("task"))),
        target=target,
        runtime=None,
        peak_memory_mb=None,
        peak_gpu_memory_mb=None,
        cpu_seconds=None,
        gpu_occupied_seconds=None,
        resource_slot=case.get("resource_policy", {}).get("slot", "unconstrained"),
        resource_policy=case.get("resource_policy"),
        resource_policy_hash=policy_hash(case.get("resource_policy")),
        expression="",
        ground_truth_expression=case.get("ground_truth"),
        expression_complexity=None,
        exact_recovery=None,
        structural_recovery=None,
        exact_recovery_semantics=(
            "additive_term_structure_ignoring_numeric_coefficients"
            if case.get("task") not in {"pde", "steady_pde"} else None
        ),
        structural_recovery_metric_version=(
            SR_STRUCTURE_METRIC_VERSION if case.get("task") not in {"pde", "steady_pde"} else None
        ),
        algebraic_exact_recovery=None,
        test_mse=None,
        test_rmse=None,
        test_nrmse=None,
        test_r2=None,
        prediction_total_points=None,
        prediction_finite_points=None,
        prediction_finite_coverage=None,
        equation_residual_mse=None,
        equation_residual_nrmse=None,
        equation_residual_points=None,
        equation_residual_total_points=None,
        equation_residual_finite_coverage=None,
        coefficient_error=None,
        coefficient_recovery=None,
        recovery_eligible=recovery_eligibility(case),
        recovery_evaluator_status=("not_certified" if case["model"] in {"pdefind_nd", "implicit_pde"} else
                                   "pending" if case.get("ground_truth") else "not_applicable"),
        pde_support_recovery=None,
        exact_symbolic_recovery=None,
        exact_symbolic_recovery_semantics=(
            "legacy alias of pde_support_recovery" if case.get("task") in {"pde", "steady_pde"} else None
        ),
        pde_support_metric_version=(PDE_SUPPORT_METRIC_VERSION if case.get("task") in {"pde", "steady_pde"} else None),
        term_support_accuracy=None,
        term_support_precision=None,
        term_support_recall=None,
        native_rollout_nrmse=None,
        equation_rollout_nrmse=None,
        equation_rollout_status=None,
        pic_status=("not_requested" if case.get("task") in {"pde", "steady_pde"} else None),
        pic=None,
        pic_r_loss=None,
        pic_p_loss=None,
        pic_message="",
        pic_version=None,
        pic_cache_key=None,
        pic_reference_cache_hit=None,
        status="error",
        stop_reason="error",
        stop_reason_source="runner",
        error="",
    )
    if case["model"] in {"eqgpt","e2e","symbolicgpt", "llmsr"}:
        from kd.dataset._overlap import training_overlap
        row["training_data_overlap"] = training_overlap(case["model"],case["dataset"])
    if case["task"] == "green" or case["dataset"] == "pyqg_subgrid" or case["dataset"].startswith("dso_file_") or (case["dataset"].startswith("fde_") and case["task"] == "sr"):
        row["recovery_eligible"] = False
        row["recovery_evaluator_status"] = "not_applicable"
    if case["task"] == "green":
        row["green_protocol_version"] = "green-v1"
    if case["dataset"] == "pyqg_subgrid":
        row["closure_protocol_version"] = "closure-v1"
    if case["dataset"].startswith("fde_") and case["task"] == "sr":
        from kd.dataset._fractional import FDE_RESPONSE_PROTOCOL_VERSION
        row["fde_response_protocol_version"] = FDE_RESPONSE_PROTOCOL_VERSION
        row["metric_target"] = "observed_response"
    if case.get("fde_pde_protocol_version"):
        row.update(
            fde_pde_protocol_version=case["fde_pde_protocol_version"],
            metric_target="observed_equation_residual", fractional_operator_evaluated=False,
            recovery_eligible=False, recovery_evaluator_status="not_applicable",
            prediction_status="abstain",
            equation_residual_status="not_evaluated",
        )
    if case["model"] == "symbolicgpt":
        row["method_provenance"] = {
            "protocol": (
                "synthetic_case_instance_reuse"
                if case["model_params"].get("reuse_pretraining_within_case")
                else "synthetic_per_fit"
            ),
            "cache_hit": None,
        }
    if case["model"] == "reference":
        from kd.evaluation.reference import metadata
        row.update(metadata(case))
    row["evaluation_protocol"] = copy.deepcopy(case.get("evaluation", {}))
    row["implementation_variant"] = case.get("method_declaration", {}).get("implementation_variant", "undeclared_legacy_variant")
    row["configuration_source"] = case.get("method_declaration", {}).get("source")
    row.update(comparison_fields(case))
    return row


def _failure(case, exc, instance=None, target=None, ground_truth=None):
    row = _base_row(case, instance, target)
    if ground_truth:
        row.update(
            ground_truth_expression=ground_truth,
            recovery_eligible=recovery_eligibility(case, {"ground_truth": ground_truth}),
            recovery_evaluator_status="not_run",
        )
    row.update(
        status=(
            ("error" if case["model"] in {"pdefind_nd", "implicit_pde"} and isinstance(exc, FileNotFoundError)
             else "skipped" if isinstance(exc, (SkipCase, ImportError, FileNotFoundError)) else "error")
        ),
        error=f"{type(exc).__name__}: {exc}",
    )
    if (case["task"] == "green" or case["dataset"] == "pyqg_subgrid" or case["dataset"].startswith(("dso_file_", "fde_"))) and isinstance(exc,FileNotFoundError):
        row.update(status="error")
    if case["dataset"].startswith("fde_") and "source_blocked" in str(exc):
        row.update(status="skipped")
    elif case.get("fde_pde_protocol_version") and isinstance(exc, SkipCase):
        row.update(status="skipped")
    if case["model"] in {"pdefind_nd", "implicit_pde"}:
        row["equation_protocol_version"] = "equation-v1"
        message = str(exc)
        if "source_blocked" in message:
            row["status"] = "skipped"
    from kd.benchmark_capabilities import failure_status
    row["capability_status"] = failure_status(exc)
    traceback.print_exc()
    if case["model"] == "reference":
        row.update(recovery_eligible=False, recovery_evaluator_status="not_applicable",
                   reference_floor_eligible=False, reference_status="unavailable_or_evaluator_error")
    return row


def _case_problems(dataset, case):
    if case["task"] in {"pde", "steady_pde"}:
        from kd.benchmark_capabilities import validate_pde
        validate_pde(dataset, case.get("reference_adapter", case["model"]))
    if case["task"] == "green":
        from kd.evaluation.green import problems
        return problems(dataset,case)
    if case["model"] in {"pdefind_nd", "implicit_pde"} or case.get("reference_adapter") in {"pdefind_nd", "implicit_pde"}:
        from kd.evaluation.equation import problems
        return problems(dataset, case)
    return regression_problems(dataset, case) if case["task"] not in {"pde", "steady_pde"} else [None]


def _run_problem(dataset, problem, case, pretrain_cache):
    if case["model"] == "reference":
        from kd.evaluation.reference import run
        return score_call(run, dataset, problem, case)
    if case["task"] == "green":
        from kd.evaluation.green import run
        return run(dataset,problem,case)
    if case["model"] in {"pdefind_nd", "implicit_pde"}:
        from kd.evaluation.equation import run
        return run(dataset, problem, case)
    return run_pde(dataset, case) if problem is None else run_regression(
        problem, case, pretrain_cache=pretrain_cache)


def run_case(case, work_dir=None, checkpoint=None, *, selection=None, cache_root=None):
    """One matrix entry, with separate rows/errors per generated PDE/ODE target."""
    import numpy as np

    random.seed(case["seed"])
    np.random.seed(case["seed"])
    rows = []
    work_dir = Path(work_dir or RESULTS_DIR / "cases" / case["name"]).resolve()

    def append(row):
        _finalize_stop_reason(row)
        rows.append(row)
        if checkpoint:
            write_json(checkpoint, rows)

    try:
        check_device_available(case["device"])
        for instance_index, (instance, dataset) in enumerate(_load_instances(case)):
            if selection and instance_index != selection["instance_index"]:
                continue
            instance_row_start = len(rows)
            # Explicitly scoped to this instance; never shared between cases,
            # seeds or datasets. An empty cache still counts as enabled.
            pretrain_cache = (
                {} if case["model"] == "symbolicgpt"
                and case["model_params"].get("reuse_pretraining_within_case")
                else None
            )
            cache_path = Path(cache_root or work_dir) / f"instance_{instance_index:04d}" / "pretraining.pkl"
            if selection and pretrain_cache is not None and cache_path.exists():
                with cache_path.open("rb") as stream:
                    pretrain_cache = pickle.load(stream)
            try:
                problems = _case_problems(dataset, case)
                for target_index, problem in enumerate(problems):
                    if selection and target_index != selection["target_index"]:
                        continue
                    target = problem["target"] if problem else "PDE"
                    if selection and (target != selection["target"] or instance != selection["instance"]):
                        raise ValueError("Prepared target identity changed during data reload")
                    start = time.perf_counter()
                    monitor = PeakMemoryMonitor(case["device"])
                    with monitor:
                        try:
                            with working_directory(
                                work_dir
                                / f"instance_{instance_index:04d}"
                                / f"target_{target_index:02d}"
                            ):
                                random.seed(case["seed"])
                                np.random.seed(case["seed"])
                                if case["model"] not in {
                                    "reference",
                                    "pdefind_nd", "implicit_pde",
                                    "gplearn",
                                    "llmsr",
                                    "pdefind",
                                    "pyoperon",
                                    "pysindy",
                                    "pysr",
                                    "sindy",
                                    "sisso",
                                }:
                                    import torch

                                    if os.environ.get("KD_SLOT_THREADS"):
                                        torch.set_num_threads(int(os.environ["KD_SLOT_THREADS"]))
                                    torch.manual_seed(case["seed"])
                                result = _run_problem(dataset, problem, case, pretrain_cache)
                            row = _base_row(case, instance, target)
                            row["status"] = "ok"
                            row.update(result)
                            if row.get("ground_truth_expression") and case["model"] not in {"reference", "pdefind_nd", "implicit_pde"}:
                                row["recovery_eligible"] = recovery_eligibility(case, {"ground_truth": row["ground_truth_expression"]})
                                if case["task"] not in {"pde", "steady_pde"}:
                                    row["recovery_evaluator_status"] = (
                                        "parsed" if row.get("exact_recovery") is not None else "unparsed"
                                    )
                            if row["status"] == "ok" and (not row["expression"] or row["expression"] == "None"):
                                raise RuntimeError(
                                    "Training completed without a discovered expression"
                                )
                            if selection and pretrain_cache is not None:
                                cache_path.parent.mkdir(parents=True, exist_ok=True)
                                temporary = cache_path.with_suffix(".tmp")
                                with temporary.open("wb") as stream:
                                    pickle.dump(pretrain_cache, stream)
                                temporary.replace(cache_path)
                        except Exception as exc:
                            row = _failure(
                                case,
                                exc,
                                instance,
                                target,
                                problem.get("ground_truth") if problem else case.get("ground_truth"),
                            )
                    row["runtime"] = time.perf_counter() - start
                    row["peak_memory_mb"] = monitor.peak_memory_mb
                    row["peak_gpu_memory_mb"] = monitor.peak_gpu_memory_mb
                    append(row)
                    if selection:
                        return rows
            except Exception as exc:
                if selection:
                    return [_failure(case, exc, selection["instance"], selection["target"],
                                     selection.get("ground_truth"))]
                truths = case.get("ground_truth")
                if len(rows) == instance_row_start and case["task"] in {"pde", "steady_pde"}:
                    for target, truth in _pde_target_bindings(case, dataset):
                        append(_failure(case, exc, instance, target, truth))
                elif len(rows) == instance_row_start and isinstance(truths, (list, tuple)):
                    for index, truth in enumerate(truths):
                        append(_failure(case, exc, instance, str(index), truth))
                else:
                    append(_failure(case, exc, instance))
    except Exception as exc:
        if selection:
            return [_failure(case, exc, selection["instance"], selection["target"],
                             selection.get("ground_truth"))]
        targets = _declared_targets(case)
        if targets and not rows:
            for target, truth in targets:
                append(_failure(case, exc, case["dataset"], target, truth))
        else:
            append(_failure(case, exc))
    if selection and not rows:
        rows.append(_failure(case, RuntimeError("Prepared target was not found"),
                             selection["instance"], selection["target"],
                             selection.get("ground_truth")))
    return rows


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    atomic_json(path, _json_safe(value))


CSV_FIELDS = [
    "benchmark_role", "implementation_variant", "configuration_source", "training_data_overlap",
    "method_config_hash", "execution_policy_hash", "evaluation_protocol_hash", "experiment_identity_hash",
    "benchmark_role", "ranking_eligible", "reference_protocol_version", "reference_status",
    "reference_floor_eligible", "reference_source_status", "reference_error", "evaluation_protocol",
    "ode_rollout_protocol_version", "system_score_seconds",
    "equation_rollout_test_trajectories", "equation_rollout_train_trajectories",
    "equation_rollout_initial_condition", "equation_rollout_normalization",
    "equation_rollout_trajectory_coverage", "equation_rollout_trajectories",
    "equation_protocol_version", "equation_expression", "capability_status",
    "green_protocol_version", "forcing_metrics", "closure_protocol_version", "training_data_overlap",
    "fde_response_protocol_version",
    "fde_pde_protocol_version",
    "fractional_operator_evaluated",
    "equation_residual_status",
    "closure_frame_metrics", "closure_layer", "closure_online_status",
    "equation_residual_normalization", "normalization_train_scale",
    "prediction_status", "prediction_abstain_reason",
    "dataset_protocol_version",
    "scientific_task",
    "physical_family",
    "dataset_metadata",
    "details",
    "variable_names",
    "execution_protocol_version",
    "target_id",
    "attempt",
    "stop_reason",
    "stop_reason_source",
    "timeout_phase",
    "fit_seconds",
    "score_seconds",
    "setup_seconds",
    "worker_elapsed_seconds",
    "cpu_seconds", "gpu_occupied_seconds", "gpu_occupied_source",
    "resource_slot", "resource_policy", "resource_policy_hash", "resource_limit",
    "resource_monitor_source", "resource_monitor_error",
    "thread_environment",
    "preparation_cpu_seconds", "preparation_gpu_occupied_seconds",
    "system_score_cpu_seconds", "system_score_gpu_occupied_seconds",
    "execution_policy",
    "target_manifest_complete",
    "name",
    "task",
    "model",
    "dataset",
    "instance",
    "family",
    "target",
    "seed",
    "profile",
    "device",
    "runtime",
    "peak_memory_mb",
    "peak_gpu_memory_mb",
    "metric_protocol_version",
    "expression",
    "ground_truth_expression",
    "expression_complexity",
    "exact_recovery",
    "structural_recovery",
    "exact_recovery_semantics",
    "structural_recovery_metric_version",
    "algebraic_exact_recovery",
    "test_mse",
    "test_rmse",
    "test_nrmse",
    "test_r2",
    "prediction_total_points",
    "prediction_finite_points",
    "prediction_finite_coverage",
    "prediction_status",
    "evaluation_status",
    "evaluation_error",
    "metric_target",
    "train_residual_mse",
    "equation_residual_mse",
    "equation_residual_nrmse",
    "equation_residual_points",
    "equation_residual_total_points",
    "equation_residual_finite_coverage",
    "coefficient_error",
    "coefficient_recovery",
    "coefficient_recovery_rtol",
    "coefficient_recovery_atol",
    "recovery_eligible",
    "recovery_evaluator_status",
    "pde_support_recovery",
    "exact_symbolic_recovery",
    "exact_symbolic_recovery_semantics",
    "pde_support_metric_version",
    "term_support_accuracy",
    "term_support_precision",
    "term_support_recall",
    "native_rollout_steps",
    "native_rollout_mse",
    "native_rollout_nrmse",
    "native_rollout_r2",
    "native_rollout_source",
    "equation_rollout_steps",
    "equation_rollout_mse",
    "equation_rollout_nrmse",
    "equation_rollout_r2",
    "equation_rollout_integrator",
    "equation_rollout_status",
    "equation_rollout_error",
    "equation_rollout_spatial_discretization",
    "equation_rollout_boundary_conditions",
    "equation_rollout_reference_nrmse",
    "equation_rollout_solver_settings",
    "equation_rollout_reference_tolerance",
    "equation_rollout_endpoint_convention",
    "pic_status",
    "pic",
    "pic_r_loss",
    "pic_p_loss",
    "pic_message",
    "pic_version",
    "pic_cache_key",
    "pic_terms",
    "pic_original_coefficients",
    "pic_reference_fit_coefficients",
    "pic_window_coefficients",
    "pic_refitted_coefficients",
    "pic_cost",
    "pic_reference_cache_hit",
    "pic_reference_train_rmse",
    "pic_reference_train_normalized_rmse",
    "pic_config",
    "pic_training_observations",
    "reward",
    "aic",
    "bic",
    "information_criterion_status",
    "information_criterion_data_role",
    "information_criterion_provenance",
    "method_provenance",
    "information_criterion_n",
    "information_criterion_k",
    "information_criterion_structural_k",
    "information_criterion_variance_parameters",
    "likelihood_model",
    "split",
    "n_train",
    "n_test",
    "n_train_time",
    "n_test_time",
    "train_time_range",
    "test_time_range",
    "status",
    "error",
    "notes",
    "log_path",
]


AGGREGATE_FIELDS = [
    "method_config_hash", "execution_policy_hash", "evaluation_protocol_hash", "experiment_identity_hash",
    "benchmark_role", "profile", "regime",
    "resource_policy_hash", "resource_slot", "n_resource_limit",
    "cpu_seconds_total", "cpu_seconds_mean", "cpu_seconds_measured_rows",
    "gpu_occupied_seconds_total", "gpu_occupied_seconds_mean", "gpu_occupied_seconds_measured_rows",
    "worker_elapsed_seconds_total", "preparation_cpu_seconds_total",
    "preparation_gpu_occupied_seconds_total", "system_score_cpu_seconds_total",
    "system_score_gpu_occupied_seconds_total",
    "dataset_protocol_version",
    "scientific_task",
    "execution_protocol_version",
    "stop_reason_counts",
    "stop_reason_rates",
    "stop_reason_source_counts",
    "n_incomplete_target_manifests",
    "metric_protocol_version",
    "task",
    "model",
    "denominator_unit",
    "n_cases",
    "n_target_rows",
    "n_rows",
    "n_success",
    "n_timeout",
    "n_error",
    "n_skipped",
    "success_rate",
    "timeout_rate",
    "runtime_total_seconds",
    "runtime_mean_seconds",
    "runtime_median_seconds",
    "peak_memory_max_mb",
    "peak_memory_mean_mb",
    "peak_gpu_memory_max_mb",
    "exact_recovery_evaluable",
    "exact_recovery_eligible",
    "exact_recovery_parse_coverage",
    "exact_recovery_conditional_rate",
    "exact_recovery_lower_bound",
    "exact_recovery_rate",
    "algebraic_exact_recovery_evaluable",
    "algebraic_exact_recovery_conditional_rate",
    "test_nrmse_mean",
    "test_r2_mean",
    "equation_residual_nrmse_mean",
    "equation_residual_finite_coverage_mean",
    "coefficient_error_mean",
    "exact_symbolic_recovery_evaluable",
    "exact_symbolic_recovery_rate",
    "pde_support_recovery_eligible",
    "pde_support_parse_coverage",
    "pde_support_recovery_conditional_rate",
    "pde_support_recovery_lower_bound",
    "pde_support_recovery_rate",
    "term_support_accuracy_mean",
    "native_rollout_nrmse_mean",
    "equation_rollout_nrmse_mean",
]


def _numeric_values(rows, key):
    return [
        float(row[key])
        for row in rows
        if row.get(key) is not None and math.isfinite(float(row[key]))
    ]


def aggregate_results(rows):
    """Produce method/task rates and metric means from result rows."""
    groups = {}
    for row in rows:
        if row.get("model") == "reference":
            continue
        version = row.get("metric_protocol_version") or "legacy/unknown"
        execution = row.get("execution_protocol_version") or "legacy/unknown"
        data_version = row.get("dataset_protocol_version") or "legacy/unknown"
        scientific_task = row.get("scientific_task") or "legacy/unknown"
        resource_hash = row.get("resource_policy_hash", "unconstrained")
        dimensions = tuple(row.get(key, "legacy/unknown") for key in (
            "benchmark_role", "profile", "regime", "execution_policy_hash", "evaluation_protocol_hash",
            "experiment_identity_hash"))
        # Actual row timing wins over an unbound hash from an adapter-only fixture.
        if row.get("execution_policy"):
            dimensions = (*dimensions[:3], execution_policy_hash(row["execution_policy"]), *dimensions[4:])
        common = (version, execution, data_version, resource_hash, *dimensions)
        groups.setdefault((*common, "all_configurations", "all", "all", "all"), []).append(row)
        groups.setdefault((*common, row.get("method_config_hash", "legacy/unknown"),
                           scientific_task, row["task"], row["model"]), []).append(row)
    aggregates = []
    for (version, execution, data_version, resource_hash, role, profile, regime, timing_hash,
         evaluation_hash, identity_hash, config_hash, scientific_task, task, model), group in sorted(groups.items(), key=repr):
        statuses = [row.get("status") for row in group]
        runtimes = _numeric_values(group, "runtime")
        memories = _numeric_values(group, "peak_memory_mb")
        gpu_memories = _numeric_values(group, "peak_gpu_memory_mb")
        cpu_seconds = _numeric_values(group, "cpu_seconds")
        gpu_seconds = _numeric_values(group, "gpu_occupied_seconds")

        def case_cost(key):
            unique = {row.get("name"): row for row in group if row.get(key) is not None}
            values = _numeric_values(unique.values(), key)
            return sum(values) if values else None
        # New rows explicitly declare the denominator before execution.  For
        # old result files, a present metric is the only safe evidence that the
        # row was eligible; this preserves readability without silently
        # reclassifying old failed rows.
        exact_eligible = [
            row
            for row in group
            if row.get("task") not in {"pde", "steady_pde"}
            and row.get("status") != "skipped"
            and (row.get("recovery_eligible") is True or row.get("exact_recovery") is not None)
        ]
        support_eligible = [
            row
            for row in group
            if row.get("task") in {"pde", "steady_pde"}
            and row.get("status") != "skipped"
            and (
                row.get("recovery_eligible") is True
                or row.get("pde_support_recovery") is not None
                or row.get("exact_symbolic_recovery") is not None
            )
        ]
        exact_valid = [row for row in exact_eligible if row.get("status") == "ok"]
        recovered = _numeric_values(exact_valid, "exact_recovery")
        algebraic_recovered = _numeric_values(exact_valid, "algebraic_exact_recovery")
        support_valid = [row for row in support_eligible if row.get("status") == "ok"]
        support_recovered = [
            float(row.get("pde_support_recovery") if row.get("pde_support_recovery") is not None
                  else row["exact_symbolic_recovery"])
            for row in support_valid
            if row.get("pde_support_recovery") is not None
            or row.get("exact_symbolic_recovery") is not None
        ]

        def mean(key):
            values = _numeric_values(group, key)
            return sum(values) / len(values) if values else None

        ordered_runtime = sorted(runtimes)
        middle = len(ordered_runtime) // 2
        median = None
        if ordered_runtime:
            median = (
                ordered_runtime[middle]
                if len(ordered_runtime) % 2
                else (ordered_runtime[middle - 1] + ordered_runtime[middle]) / 2
            )
        count = len(group)
        aggregates.append(
            {
                "metric_protocol_version": version,
                "execution_protocol_version": execution,
                "dataset_protocol_version": data_version,
                "scientific_task": scientific_task,
                "resource_policy_hash": resource_hash,
                "method_config_hash": config_hash, "execution_policy_hash": timing_hash,
                "evaluation_protocol_hash": evaluation_hash, "experiment_identity_hash": identity_hash,
                "benchmark_role": role, "profile": profile, "regime": regime,
                "resource_slot": group[0].get("resource_slot", "unconstrained"),
                "n_resource_limit": sum(row.get("stop_reason") == "resource_limit" for row in group),
                "cpu_seconds_total": sum(cpu_seconds) if cpu_seconds else None,
                "cpu_seconds_mean": sum(cpu_seconds) / len(cpu_seconds) if cpu_seconds else None,
                "cpu_seconds_measured_rows": len(cpu_seconds),
                "gpu_occupied_seconds_total": sum(gpu_seconds) if gpu_seconds else None,
                "gpu_occupied_seconds_mean": sum(gpu_seconds) / len(gpu_seconds) if gpu_seconds else None,
                "gpu_occupied_seconds_measured_rows": len(gpu_seconds),
                "worker_elapsed_seconds_total": sum(_numeric_values(group, "worker_elapsed_seconds")),
                **{key + "_total": case_cost(key) for key in (
                    "preparation_cpu_seconds", "preparation_gpu_occupied_seconds",
                    "system_score_cpu_seconds", "system_score_gpu_occupied_seconds")},
                "stop_reason_counts": {
                    reason: sum(row.get("stop_reason", "unknown") == reason for row in group)
                    for reason in (*STOP_REASONS, "unknown")
                },
                "stop_reason_rates": {
                    reason: sum(row.get("stop_reason", "unknown") == reason for row in group) / count
                    for reason in (*STOP_REASONS, "unknown")
                },
                "stop_reason_source_counts": {
                    source: sum(row.get("stop_reason_source", "unknown") == source for row in group)
                    for source in sorted({row.get("stop_reason_source", "unknown") for row in group})
                },
                "n_incomplete_target_manifests": len({
                    row.get("name") for row in group if row.get("target_manifest_complete") is False
                }),
                "task": task,
                "model": model,
                "denominator_unit": "target_row",
                "n_cases": len({row.get("name") for row in group}),
                "n_target_rows": count,
                "n_rows": count,
                "n_success": statuses.count("ok"),
                "n_timeout": statuses.count("timeout"),
                "n_error": statuses.count("error"),
                "n_skipped": statuses.count("skipped"),
                "success_rate": statuses.count("ok") / count if count else None,
                "timeout_rate": statuses.count("timeout") / count if count else None,
                "runtime_total_seconds": sum(runtimes),
                "runtime_mean_seconds": sum(runtimes) / len(runtimes) if runtimes else None,
                "runtime_median_seconds": median,
                "peak_memory_max_mb": max(memories) if memories else None,
                "peak_memory_mean_mb": sum(memories) / len(memories) if memories else None,
                "peak_gpu_memory_max_mb": max(gpu_memories) if gpu_memories else None,
                "exact_recovery_evaluable": len(recovered),
                "exact_recovery_eligible": len(exact_eligible),
                "exact_recovery_parse_coverage": (
                    len(recovered) / len(exact_valid) if exact_valid else None
                ),
                "exact_recovery_rate": (
                    sum(recovered) / len(exact_eligible)
                    if exact_eligible and len(recovered) == len(exact_valid) else None
                ),
                "exact_recovery_conditional_rate": (
                    sum(recovered) / len(recovered) if recovered else None
                ),
                "exact_recovery_lower_bound": (
                    sum(recovered) / len(exact_eligible) if exact_eligible else None
                ),
                "algebraic_exact_recovery_evaluable": len(algebraic_recovered),
                "algebraic_exact_recovery_conditional_rate": (
                    sum(algebraic_recovered) / len(algebraic_recovered)
                    if algebraic_recovered else None
                ),
                "test_nrmse_mean": mean("test_nrmse"),
                "test_r2_mean": mean("test_r2"),
                "equation_residual_nrmse_mean": mean("equation_residual_nrmse"),
                "equation_residual_finite_coverage_mean": mean(
                    "equation_residual_finite_coverage"
                ),
                "coefficient_error_mean": mean("coefficient_error"),
                "exact_symbolic_recovery_evaluable": len(support_recovered),
                "exact_symbolic_recovery_rate": (
                    sum(support_recovered) / len(support_eligible)
                    if support_eligible and len(support_recovered) == len(support_valid) else None
                ),
                "pde_support_recovery_eligible": len(support_eligible),
                "pde_support_parse_coverage": (
                    len(support_recovered) / len(support_valid) if support_valid else None
                ),
                "pde_support_recovery_rate": (
                    sum(support_recovered) / len(support_eligible)
                    if support_eligible and len(support_recovered) == len(support_valid) else None
                ),
                "pde_support_recovery_conditional_rate": (
                    sum(support_recovered) / len(support_recovered) if support_recovered else None
                ),
                "pde_support_recovery_lower_bound": (
                    sum(support_recovered) / len(support_eligible) if support_eligible else None
                ),
                "term_support_accuracy_mean": mean("term_support_accuracy"),
                "native_rollout_nrmse_mean": mean("native_rollout_nrmse"),
                "equation_rollout_nrmse_mean": mean("equation_rollout_nrmse"),
            }
        )
    return aggregates


def save_summary(output_dir, rows):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "benchmark_summary.json", rows)
    with (output_dir / "benchmark_summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(_json_safe(rows))
    aggregates = aggregate_results(rows)
    write_json(output_dir / "benchmark_aggregate.json", aggregates)
    with (output_dir / "benchmark_aggregate.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=AGGREGATE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(_json_safe(aggregates))
    from itertools import combinations

    from kd.evaluation import paired_comparison, seed_family_summary

    report_metrics = ("exact_recovery", "test_nrmse", "pde_support_recovery",
                      "equation_rollout_nrmse", "native_rollout_nrmse")
    method_rows = [r for r in rows if r.get("model") != "reference"]
    write_json(output_dir / "benchmark_reference.json", [r for r in rows if r.get("model") == "reference"])
    family_rows = [entry for metric in report_metrics
                   for entry in seed_family_summary(method_rows, metric)]
    paired_rows = []
    models = sorted({row.get("model") for row in rows if row.get("model") and row.get("model") != "reference"})
    for metric in report_metrics:
        for model_a, model_b in combinations(models, 2):
            try:
                entry = paired_comparison(rows, metric, model_a, model_b)
            except ValueError as exc:
                entry = {"metric": metric, "model_a": model_a, "model_b": model_b,
                         "status": "ambiguous_duplicate_keys", "error": str(exc),
                         "n_pairs": None, "n_model_a_evaluable": None,
                         "n_model_b_evaluable": None}
                paired_rows.append(entry)
                continue
            if entry["n_model_a_evaluable"] or entry["n_model_b_evaluable"]:
                paired_rows.append(entry)
    write_json(output_dir / "benchmark_seed_family.json", family_rows)
    write_json(output_dir / "benchmark_paired.json", paired_rows)

    def write_dynamic_csv(path, values):
        fields = list(dict.fromkeys(key for value in values for key in value))
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            if fields:
                writer.writeheader()
                writer.writerows(_json_safe(values))

    write_dynamic_csv(output_dir / "benchmark_seed_family.csv", family_rows)
    write_dynamic_csv(output_dir / "benchmark_paired.csv", paired_rows)


def _pde_target_bindings(case, dataset=None):
    """Execution units: equation adapters fit each target; legacy adapters fit one field."""
    model = case.get("reference_adapter", case["model"])
    if model in {"pdefind_nd", "implicit_pde"}:
        if dataset is None:
            from kd.dataset._advanced import declared_targets
            declared = declared_targets(case["dataset"])
        else:
            from kd.dataset._equations import targets
            declared = targets(dataset)
        return [(target.key, None) for target in declared]
    return [("PDE", case.get("ground_truth"))]


def _declared_targets(case):
    """Target-level failure accounting from predeclared truth, never fitted outputs."""
    if case["dataset"].startswith("fde_"):
        if case["task"] == "pde":
            return _pde_target_bindings(case)
        from kd.dataset._fractional import response_target
        return [(response_target(case["dataset"]), None)]
    if case["task"] in {"pde", "steady_pde"}:
        return _pde_target_bindings(case)
    if case["task"] == "green":
        return [("kernel",None)]
    if case["dataset"] == "pyqg_subgrid":
        return [("q_subgrid_forcing_L1",None),("q_subgrid_forcing_L2",None)]
    truth = case.get("ground_truth")
    if isinstance(truth, dict):
        return list(truth.items())
    if not isinstance(truth, (list, tuple)):
        return []
    result = []
    for index, equation in enumerate(truth):
        lhs = str(equation).split("=", 1)[0].strip() if "=" in str(equation) else ""
        match = re.fullmatch(r"d([A-Za-z_]\w*)/dt", lhs)
        target = f"d({match.group(1)})/dt" if match else str(index)
        result.append((target, equation))
    return result


def _target_spec(instance, instance_index, target, target_index, truth):
    return dict(target_id=f"i{instance_index:04d}_t{target_index:04d}",
                instance=instance, instance_index=instance_index,
                target=target, target_index=target_index, ground_truth=truth)


def _fallback_targets(case, exc, *, instance=None, instance_index=0, declared=None):
    declared = _declared_targets(case) if declared is None else declared
    if not declared:
        target = "PDE" if case["task"] == "pde" else "y"
        if case["dataset"].startswith("cyt_"):
            target = case["dataset_params"].get("target", "Mut")
        elif case["dataset"].startswith("solid_"):
            target = "DIF" if case["dataset"] == "solid_dif" else "stress"
        declared = [(target, case.get("ground_truth"))]
    specs = []
    for index, (target, truth) in enumerate(declared):
        spec = _target_spec(instance or case["dataset"], instance_index, target, index, truth)
        spec["preparation_row"] = _failure(case, exc, spec["instance"], target, truth)
        specs.append(spec)
    return specs


def _case_failure_targets(case, exc):
    """Retain counts known before loading, even if generation fails or times out."""
    params = case["dataset_params"]
    known_instances = True
    if case["dataset"] in {"solid_dif", "solid_strain_stress", "solid_hardening"}:
        from kd.dataset._domain import constitutive_instance_specs

        try:
            declared = constitutive_instance_specs(case["dataset"])
            instances = params.get("instances") or [s["instance"] for s in declared]
        except OSError:
            instances, known_instances = [case["dataset"]], False
    elif case["dataset"] == "pdeformer_sinus":
        import inspect

        from kd.dataset._pdeformer_sinus import load_pdeformer_sinus_benchmark

        default = inspect.signature(load_pdeformer_sinus_benchmark).parameters["n_pde"].default
        instances = [f"pdeformer_sinus/{index:04d}" for index in range(params.get("n_pde", default))]
    elif case["dataset"] == "wave_breaking" and params.get("keys"):
        instances = params["keys"]
    else:
        instances = [case["dataset"]]
    specs = []
    complete = (known_instances and (case["task"] != "ode" or bool(_declared_targets(case)))
                and (case["dataset"] != "wave_breaking" or bool(params.get("keys"))))
    for index, instance in enumerate(instances):
        entries = _fallback_targets(case, exc, instance=instance, instance_index=index)
        for spec in entries:
            spec["preparation_row"]["target_manifest_complete"] = complete
        specs.extend(entries)
    return specs


def _prepare_targets(case):
    """Enumerate instance/target identities before fitting, including setup failures."""
    import numpy as np

    random.seed(case["seed"])
    np.random.seed(case["seed"])
    specs = []
    try:
        for instance_index, (instance, dataset) in enumerate(_load_instances(case)):
            try:
                problems = list(_case_problems(dataset, case))
                for index, problem in enumerate(problems):
                    spec = _target_spec(
                        instance, instance_index, problem["target"] if problem else "PDE", index,
                        problem.get("ground_truth") if problem else case.get("ground_truth"),
                    )
                    if problem:
                        spec.update(split=problem["split"], data_metadata=problem["metadata"],
                                    n_train=len(problem["X_train"]), n_test=len(problem["X_test"]))
                    if case["task"] in {"pde", "steady_pde"}:
                        from kd.dataset._equations import equation_spec, targets
                        actual = targets(dataset)
                        spec["equation_targets"] = [t.to_dict() for t in actual]
                        if not problem and len(actual) == 1:
                            spec["data_metadata"] = equation_spec(dataset, actual[0])
                    specs.append(spec)
            except Exception as exc:
                declared = None
                if case["task"] in {"pde", "steady_pde"}:
                    declared = _pde_target_bindings(case, dataset)
                elif hasattr(dataset, "state_vars"):
                    truths = dict(_declared_targets(case))
                    declared = [(f"d({name})/dt", truths.get(f"d({name})/dt"))
                                for name in dataset.state_vars]
                elif hasattr(dataset, "target_name"):
                    declared = [(dataset.target_name, None)]
                failed = _fallback_targets(case, exc, instance=instance,
                                           instance_index=instance_index, declared=declared)
                for entry in failed:
                    if case["task"] in {"pde", "steady_pde"}:
                        from kd.dataset._equations import equation_spec, targets
                        actual = targets(dataset)
                        entry["equation_targets"] = [t.to_dict() for t in actual]
                        target = next((t for t in actual if t.key == entry["target"]),
                                      actual[0] if len(actual) == 1 else None)
                        if target is not None:
                            entry["data_metadata"] = equation_spec(dataset, target)
                    entry["preparation_row"]["target_manifest_complete"] = True
                    entry["preparation_row"]["details"] = copy.deepcopy(
                        getattr(dataset, "protocol_metadata", {}))
                specs.extend(failed)
    except Exception as exc:
        if specs:
            # The loader failed between instances. Keep its failure visible;
            # the unknown remainder cannot be claimed to have a complete denominator.
            remaining = _fallback_targets(case, exc, instance_index=len(specs))
            for spec in remaining:
                spec["preparation_row"]["target_manifest_complete"] = False
            specs.extend(remaining)
        else:
            specs = _case_failure_targets(case, exc)
    if not specs:
        specs = _fallback_targets(case, ValueError("Dataset produced no targets"))
    for spec in specs:
        spec["recovery_eligible"] = recovery_eligibility(case, spec)
    return specs


def _new_attempt(directory):
    directory.mkdir(parents=True, exist_ok=True)
    index = max((int(path.name) for path in directory.iterdir()
                 if path.is_dir() and path.name.isdigit()), default=0) + 1
    attempt = directory / f"{index:04d}"
    attempt.mkdir()
    return attempt


def _read_terminal(path, case, spec):
    if not path.exists():
        return None
    row = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(row, dict) or row.get("status") not in TERMINAL_STATUSES
            or row.get("stop_reason") not in STOP_REASONS
            or row.get("name") != case["name"] or row.get("target_id") != spec["target_id"]
            or row.get("instance") != spec["instance"] or row.get("target") != spec["target"]
            or row.get("execution_protocol_version") != EXECUTION_PROTOCOL_VERSION):
        raise ValueError(f"Invalid terminal target result; preserved for inspection: {path}")
    return row


def _target_result(directory, case, spec):
    """Recover a worker commit even if its parent died before updating the case."""
    target_dir = directory / "targets" / spec["target_id"]
    for attempt in sorted((target_dir / "attempts").glob("[0-9]*")):
        reap_orphan(attempt)
    canonical = target_dir / "result.json"
    row = _read_terminal(canonical, case, spec)
    if row is not None:
        return row
    for attempt in sorted((target_dir / "attempts").glob("[0-9]*")):
        if not (attempt / "result.json").exists() and (attempt / "supervision.json").exists():
            outcome = json.loads((attempt / "supervision.json").read_text(encoding="utf-8"))
            policy = json.loads((directory / "execution.json").read_text(encoding="utf-8"))
            _commit_target_attempt(case, spec, attempt, policy, outcome)
        row = _read_terminal(attempt / "result.json", case, spec)
        if row is not None:
            write_json(canonical, row)
            return row
    return None


def _commit_target_attempt(case, spec, attempt, policy, outcome):
    """Promote a worker candidate once, after parent resource supervision finishes."""
    existing = _read_terminal(attempt / "result.json", case, spec)
    if existing is not None:
        return existing
    row = _read_terminal(attempt / "candidate.json", case, spec)
    if row is None or outcome.get("timeout_phase") or outcome.get("resource_limit"):
        row = _worker_failure(case, spec, outcome, attempt / "run.log")
    else:
        # The root's final self measurement covers the last fraction of a poll;
        # descendants still come from the parent process-tree sampler.
        sampled_root = outcome.get("worker_cpu_seconds", 0.0)
        own_cpu = row.pop("_self_cpu_seconds", None)
        cpu = outcome.get("cpu_seconds")
        row.update(outcome)
        if own_cpu is not None and cpu is not None:
            row["cpu_seconds"] = cpu + max(0.0, own_cpu - sampled_root)
    row.update(target_id=spec["target_id"], attempt=int(attempt.name), execution_policy=policy)
    row.update(comparison_fields(case, policy))
    _finalize_stop_reason(row)
    write_json(attempt / "result.json", row)
    return row


def _worker_failure(case, spec, outcome, log_path):
    row = _base_row(case, spec["instance"], spec["target"])
    timeout_phase = outcome.get("timeout_phase")
    row.update(
        status="timeout" if timeout_phase else "error",
        error=(f"Hard timeout in {timeout_phase}" if timeout_phase else
               "Process-tree RSS exceeded slot memory limit" if outcome.get("resource_limit") else
               f"Worker exited without a terminal result (code {outcome.get('returncode')})"),
        ground_truth_expression=spec.get("ground_truth"),
        recovery_eligible=recovery_eligibility(case, spec), recovery_evaluator_status="not_run",
        target_id=spec["target_id"], log_path=str(log_path),
        runtime=outcome.get("worker_elapsed_seconds"), **outcome,
    )
    for key in ("split", "n_train", "n_test"):
        if key in spec:
            row[key] = spec[key]
    if "data_metadata" in spec:
        row["details"] = spec["data_metadata"]
    if case["model"] == "reference":
        row.update(recovery_eligible=False, recovery_evaluator_status="not_applicable",
                   reference_floor_eligible=False, reference_status="unavailable_or_evaluator_error")
    return _finalize_stop_reason(row)


def _launch_worker(case_path, attempt, policy, spec=None):
    command = [sys.executable, str(Path(__file__).resolve()),
               "--_case-file", str(case_path), "--_attempt-dir", str(attempt)]
    if spec is None:
        command.append("--_prepare-targets")
    else:
        write_json(attempt / "target.json", spec)
        command.extend(["--_target-file", str(attempt / "target.json")])
    case = json.loads(Path(case_path).read_text(encoding="utf-8"))
    return supervise(command, cwd=ROOT, log_path=attempt / "run.log",
                     phase_path=attempt / "phase.json", policy=policy, device=case["device"], model=case["model"])


def _ensure_target_plan(directory, case, policy):
    for attempt in sorted((directory / "preparation").glob("[0-9]*")):
        reap_orphan(attempt)
    plan_path = directory / "targets.json"
    if plan_path.exists():
        return json.loads(plan_path.read_text(encoding="utf-8"))
    # Preparation completion is also recoverable across a parent interruption.
    for path in sorted((directory / "preparation").glob("*/targets.json")):
        specs = json.loads(path.read_text(encoding="utf-8"))
        write_json(plan_path, specs)
        return specs
    for path in sorted((directory / "preparation").glob("*/supervision.json")):
        outcome = json.loads(path.read_text(encoding="utf-8"))
        candidate = path.parent / "target_candidate.json"
        if candidate.exists() and not outcome.get("timeout_phase") and not outcome.get("resource_limit"):
            specs = json.loads(candidate.read_text(encoding="utf-8"))
            write_json(path.parent / "targets.json", specs)
            write_json(plan_path, specs)
            return specs
    attempt = _new_attempt(directory / "preparation")
    live_case = directory / "case_runtime.json"
    outcome = _launch_worker(live_case if live_case.exists() else directory / "case.json", attempt, policy)
    candidate = attempt / "target_candidate.json"
    if candidate.exists() and not outcome.get("timeout_phase") and not outcome.get("resource_limit"):
        write_json(attempt / "targets.json", json.loads(candidate.read_text(encoding="utf-8")))
    if (attempt / "targets.json").exists() and not outcome.get("timeout_phase") and not outcome.get("resource_limit"):
        specs = json.loads((attempt / "targets.json").read_text(encoding="utf-8"))
    else:
        specs = _case_failure_targets(case, RuntimeError("Target preparation did not complete"))
        for spec in specs:
            complete = spec["preparation_row"].get("target_manifest_complete", False)
            spec["preparation_row"] = _worker_failure(case, spec, outcome, attempt / "run.log")
            spec["preparation_row"]["target_manifest_complete"] = complete
            # Preparation is shared by the case, not one training worker per
            # declared target. Its cost is attached once via preparation_*.
            spec["preparation_row"].update(cpu_seconds=None, gpu_occupied_seconds=None,
                                           worker_elapsed_seconds=None)
    write_json(plan_path, specs)
    return specs


def execute_case(case, output_dir, timeout=None, resume=False, *, timeout_grace=5.0,
                 setup_timeout=300.0, score_timeout=300.0, resource_policy=None, soft_deadline=True):
    """Run isolated targets; terminal first results are immutable, including failures."""
    policy = execution_policy(timeout, timeout_grace=timeout_grace,
                              setup_timeout=setup_timeout, score_timeout=score_timeout,
                              resource_policy=resource_policy or case.get("resource_policy"),
                              soft_deadline=soft_deadline)
    if case.get("execution_protocol_version") != EXECUTION_PROTOCOL_VERSION:
        raise ValueError("Legacy case execution protocol; generate a new manifest/output directory")
    from kd.dataset._metadata import DATASET_PROTOCOL_VERSION

    if case.get("dataset_protocol_version") != DATASET_PROTOCOL_VERSION:
        raise ValueError("Legacy dataset protocol; generate a new manifest/output directory")
    directory = (Path(output_dir) / "cases" / case["name"]).resolve()
    with case_lock(directory):
        request_path, policy_path = directory / "case.json", directory / "execution.json"
        if policy_path.exists():
            if (stable_execution(json.loads(policy_path.read_text(encoding="utf-8"))) != stable_execution(policy)
                    or canonical_case(json.loads(request_path.read_text(encoding="utf-8"))) != canonical_case(case)):
                raise ValueError("Case configuration or timing policy changed; use a new output directory")
            if not resume:
                raise FileExistsError("Case already attempted; use --resume or a new output directory")
        else:
            if (directory / "result.json").exists() or (directory / "partial.json").exists():
                raise ValueError("Unversioned existing results; use a new output directory")
            write_json(request_path, case)
            write_json(policy_path, policy)
        # Preserve first declarations; live PID/path rebinding is separately recorded.
        write_json(directory / "execution_runtime.json", policy)
        write_json(directory / "case_runtime.json", case)
        specs = _ensure_target_plan(directory, case, policy)
        rows = []
        for spec in specs:
            row = _target_result(directory, case, spec)
            if row is None:
                target_dir = directory / "targets" / spec["target_id"]
                attempt = _new_attempt(target_dir / "attempts")
                if "preparation_row" in spec:
                    row = _finalize_stop_reason(dict(spec["preparation_row"]))
                else:
                    outcome = _launch_worker(directory / "case_runtime.json", attempt, policy, spec)
                    row = _commit_target_attempt(case, spec, attempt, policy, outcome)
                row.update(target_id=spec["target_id"], attempt=int(attempt.name),
                           execution_policy=policy)
                row.update(comparison_fields(case, policy))
                if "preparation_row" in spec:
                    resources = policy.get("resource_policy")
                    row.update(resource_policy=resources, resource_policy_hash=policy_hash(resources),
                               resource_slot=resources["slot"] if resources else "unconstrained")
                # Worker commits are already complete; do not rewrite their bytes.
                if not (attempt / "result.json").exists():
                    write_json(attempt / "result.json", row)
                write_json(target_dir / "result.json", row)
            rows.append(row)
            write_json(directory / "partial.json", rows)
        preparations = sorted((directory / "preparation").glob("*/supervision.json"))
        if preparations:
            preparation = json.loads(preparations[-1].read_text(encoding="utf-8"))
            rows = [{**r, "preparation_cpu_seconds": preparation.get("cpu_seconds"),
                     "preparation_gpu_occupied_seconds": preparation.get("gpu_occupied_seconds")} for r in rows]
        if case["task"] == "ode":
            rows = _score_ode_outputs(directory, case, rows, policy)
        write_json(directory / "result.json", rows)
        return rows


def _score_ode_outputs(directory, case, rows, policy):
    """Cache scoring separately; never rewrite a target's first terminal result."""
    cache = directory / "scoring" / "ode_rollout"
    cache.mkdir(parents=True, exist_ok=True)
    result_path = cache / "result.json"
    attempts = cache / "attempts"
    for attempt in sorted(attempts.glob("[0-9]*")):
        reap_orphan(attempt)
        if not (attempt / "result.json").exists() and (attempt / "supervision.json").exists():
            outcome = json.loads((attempt / "supervision.json").read_text(encoding="utf8"))
            _commit_system_score(attempt, outcome)
        if not result_path.exists() and (attempt / "result.json").exists():
            write_json(result_path, json.loads((attempt / "result.json").read_text(encoding="utf8")))
    if not result_path.exists():
        attempt = _new_attempt(attempts)
        outcome = supervise([sys.executable, "-m", "kd.evaluation.ode_worker", str(directory), str(attempt)],
                            cwd=ROOT, log_path=attempt / "run.log", phase_path=attempt / "phase.json", policy=policy,
                            device=case["device"])
        data = _commit_system_score(attempt, outcome)
        write_json(result_path, data)
    data = json.loads(result_path.read_text(encoding="utf8"))
    result = []
    for row in rows:
        metric = data["metrics"].get(row["instance"], {}).get(row["target"])
        if metric is None:
            metric = {"ode_rollout_protocol_version": "ode-rollout-v1",
                      "equation_rollout_status": data.get("failure_status", "abstained_incomplete_system"),
                      "equation_rollout_nrmse": None}
        result.append({**row, **metric, "system_score_seconds": data["times"].get("score_seconds"),
                       "system_score_cpu_seconds": data["times"].get("cpu_seconds"),
                       "system_score_gpu_occupied_seconds": data["times"].get("gpu_occupied_seconds")})
    return result


def _commit_system_score(attempt, outcome):
    path = attempt / "result.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf8"))
    candidate = attempt / "candidate.json"
    if candidate.exists() and not outcome.get("timeout_phase") and not outcome.get("resource_limit"):
        data = json.loads(candidate.read_text(encoding="utf8"))
        own_cpu = data["times"].pop("_self_cpu_seconds", None)
        data["times"].update(outcome)
        if own_cpu is not None and outcome.get("cpu_seconds") is not None:
            data["times"]["cpu_seconds"] += max(0.0, own_cpu - outcome.get("worker_cpu_seconds", 0))
    else:
        data = {"metrics": {}, "times": outcome, "failure_status":
                "resource_limit" if outcome.get("resource_limit") else
                "score_timeout" if outcome.get("timeout_phase") else "evaluator_error"}
    write_json(path, data)
    return data


def save_manifest(output, manifest):
    """Never relabel an existing result directory with another matrix or budget."""
    path = Path(output) / "benchmark_manifest.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        for key in ("execution_protocol_version", "execution_policy", "resource_slots", "cases",
                    "identity", "parent_matrix_hash", "experiment_identity_hash", "benchmark_mode"):
            before, after = previous.get(key), manifest.get(key)
            if key == "execution_policy" and before and after:
                before, after = stable_execution(before), stable_execution(after)
            elif key == "resource_slots" and before and after:
                before = {k: stable_resources(v) for k, v in before.items()}
                after = {k: stable_resources(v) for k, v in after.items()}
            elif key == "cases" and before and after:
                before, after = [canonical_case(c) for c in before], [canonical_case(c) for c in after]
            elif key == "identity" and before and after:
                before, after = before.get("sha256"), after.get("sha256")
            if before != after:
                raise ValueError(f"Existing manifest differs at {key}; use a new output directory")
        return
    write_json(path, public_manifest(manifest) if manifest.get("identity") else manifest)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--tasks", nargs="+", choices=TASKS)
    parser.add_argument("--models", nargs="+", help="Baseline names or quoted glob patterns")
    parser.add_argument("--datasets", nargs="+", help="Dataset names or quoted glob patterns")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--profile", choices=("smoke", "full"))
    parser.add_argument(
        "--search-space-policy",
        choices=("declared", "declared_plus_reference"),
        help="Override the dataset-grammar policy for supported SR baselines",
    )
    parser.add_argument(
        "--config", type=Path, help=f"Total JSON config (default: {DEFAULT_CONFIG_FILE})"
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--include-reference", action="store_true",
                        help="Append scoring-only reference controls; excluded from discovery rankings")
    parser.add_argument(
        "--timeout", type=float, help="Fit seconds per target, excluding setup/scoring (default: unlimited)"
    )
    parser.add_argument("--soft-deadline", action=argparse.BooleanOptionalAction, default=None,
                        help="Return the best completed candidate at T (default enabled)")
    parser.add_argument("--timeout-grace", type=float, help="Grace before hard fit timeout (default: 5s)")
    parser.add_argument("--setup-timeout", type=float, help="Preparation/startup limit (default: 300s)")
    parser.add_argument("--score-timeout", type=float, help="Post-fit scoring limit (default: 300s)")
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        help="Keep terminal targets (including errors/timeouts); resume missing targets only",
    )
    resume.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Ignore resume=true from the total config",
    )
    parser.set_defaults(resume=None)
    parser.add_argument(
        "--device",
        action="append",
        metavar="BASELINE=DEVICE",
        help="Override one assignment, e.g. --device dso=cuda:0 (repeatable)",
    )
    parser.add_argument(
        "--check-devices",
        action="store_true",
        help="Validate selected CUDA assignments, print GPU names, and exit",
    )
    parser.add_argument(
        "--list", action="store_true", help="List all baselines, datasets, and excluded sources"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the full manifest without loading data or fitting",
    )
    parser.add_argument(
        "--check-compatibility",
        action="store_true",
        help="Check dataset/instance/target compatibility without fitting and exit",
    )
    parser.add_argument(
        "--compatibility-mode", choices=("sample", "full"), default="sample",
        help="sample checks the first instance for each selected seed; full checks all instances",
    )
    parser.add_argument("--_case-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--_target-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--_attempt-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--_prepare-targets", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args._case_file:
        case = json.loads(args._case_file.read_text(encoding="utf-8"))
        directory = args._case_file.parent
        attempt = args._attempt_dir
        if attempt is None:
            parser.error("Internal worker requires --_attempt-dir")
        policy_path = directory / "execution_runtime.json"
        policy = json.loads((policy_path if policy_path.exists() else directory / "execution.json").read_text(encoding="utf-8"))
        with PhaseClock(attempt / "phase.json",
                        started=float(os.environ["KD_BENCHMARK_STARTED"]), policy=policy) as clock:
            if args._prepare_targets:
                specs = _prepare_targets(case)
                if clock.times()["setup_seconds"] > policy["setup_timeout"]:
                    raise TimeoutError("Target preparation exceeded setup_timeout")
                write_json(attempt / "target_candidate.json", specs)
                return 0
            if args._target_file is None:
                parser.error("Internal worker requires --_target-file")
            spec = json.loads(args._target_file.read_text(encoding="utf-8"))
            row = run_case(case, attempt / "work", selection=spec,
                           cache_root=directory / "cache")[0]
            row.update(clock.times(), target_id=spec["target_id"], attempt=int(attempt.name),
                       execution_policy=policy, log_path=str(attempt / "run.log"))
            row["worker_elapsed_seconds"] = time.perf_counter() - clock.started
            # A fast phase can end between supervisor polls. Enforce its limit
            # before committing, so a late result is never accepted on resume.
            for phase, limit in (("setup", policy["setup_timeout"]),
                                 ("fit", None if policy["timeout"] is None else
                                  policy["timeout"] + policy["timeout_grace"]),
                                 ("score", policy["score_timeout"])):
                if limit is not None and row[f"{phase}_seconds"] > limit:
                    row.update(status="timeout", timeout_phase=phase,
                               error=f"Hard timeout in {phase}")
                    break
            _finalize_stop_reason(row)
            try:
                import psutil
                cpu = psutil.Process().cpu_times()
                row["_self_cpu_seconds"] = cpu.user + cpu.system
            except ImportError:
                pass
            write_json(attempt / "candidate.json", row)
        return 0
    try:
        catalog, excluded = dataset_catalog()
        settings, config_path = load_benchmark_config(args.config)
        validate_benchmark_config(settings, catalog)
        baseline_configs = load_baseline_configs(settings["baseline_config_dir"])
        profile = args.profile or settings["profile"]
        tasks = args.tasks if args.tasks is not None else settings["tasks"]
        models = args.models if args.models is not None else settings["models"]
        if args.search_space_policy:
            for model in _select(models, BASELINES, "model"):
                if model in {"gplearn", "pyoperon", "pysr", "physo", "dso"}:
                    settings["overrides"].setdefault("models", {}).setdefault(model, {})[
                        "search_space_policy"
                    ] = args.search_space_policy
        datasets = args.datasets if args.datasets is not None else settings["datasets"]
        seeds = args.seeds if args.seeds is not None else settings["seeds"]
        timeout = args.timeout if args.timeout is not None else settings["timeout"]
        timing = {key: getattr(args, key) if getattr(args, key) is not None else
                  settings.get(key, default)
                  for key, default in (("timeout_grace", 5.0), ("setup_timeout", 300.0),
                                       ("score_timeout", 300.0), ("soft_deadline", True))}
        policy = execution_policy(timeout, **timing)
        resume_enabled = args.resume if args.resume is not None else settings["resume"]
        if timeout is not None and timeout <= 0:
            raise ValueError("--timeout must be positive")
        if any(seed < 0 or seed >= 2**32 for seed in seeds):
            raise ValueError("--seeds must be in [0, 2**32)")
        devices = copy.deepcopy(settings["devices"])
        devices.update(parse_device_overrides(args.device))
        if args.list:
            print(
                json.dumps(
                    {
                        "baselines": BASELINES,
                        "configured_devices": devices,
                        "datasets": catalog,
                        "excluded": excluded,
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )
            return 0
        cases = build_experiments(
            catalog,
            tasks=tasks,
            models=models,
            datasets=datasets,
            seeds=seeds,
            profile=profile,
            config=settings["overrides"],
            baseline_configs=baseline_configs,
            devices=devices,
            run_defaults=settings["profiles"][profile]["run"],
        )
        for case in cases:
            case["method_declaration"] = copy.deepcopy(settings.get("configuration_sources", {}).get(case["model"], {}))
            if settings.get("benchmark_mode", "diagnostic") != "main":
                case["benchmark_role"] = "diagnostic"
            if case["dataset"].startswith("ode_core_") and settings.get("core_ode_grammar"):
                case["task_grammar"] = list(settings["core_ode_grammar"])
        if settings.get("benchmark_mode") == "main":
            validate_main_matrix(cases, policy, configuration_sources=settings.get("configuration_sources", {}))
            if not args.dry_run and not args.check_compatibility and not args.check_devices:
                raise ValueError("Formal main execution requires a resource-bound parent manifest and scripts/run_slots.py")
        if not cases:
            raise ValueError("Selection contains no compatible task/model/dataset combinations")
        if args.include_reference:
            if args.check_compatibility or args.check_devices:
                parser.error("--include-reference is for evaluation runs or dry-run manifests")
            from kd.evaluation.reference import reference_cases
            cases.extend(reference_cases(cases))
        assigned = {case["model"]: case["device"] for case in cases}
        if args.check_devices:
            report = {
                model: {
                    "device": device,
                    "accelerator": BASELINES[model]["accelerator"],
                    "runtime": check_device_available(device),
                }
                for model, device in assigned.items()
            }
            print(json.dumps(report, indent=2, ensure_ascii=False))
            return 0
        if not args.dry_run:
            for device in sorted(set(assigned.values())):
                check_device_available(device)
    except (ValueError, ImportError, OSError) as exc:
        parser.error(str(exc))
    output = (
        args.output_dir.resolve()
        if args.output_dir
        else _project_path(settings["output_dir"]).resolve()
    )
    manifest = {
        "benchmark_mode": settings.get("benchmark_mode", "diagnostic"),
        "configuration_sources": settings.get("configuration_sources", {}),
        "execution_protocol_version": EXECUTION_PROTOCOL_VERSION,
        "dataset_protocol_version": next(iter(catalog.values()))["dataset_protocol_version"],
        "execution_policy": policy,
        "config": str(config_path),
        "profile": profile,
        "tasks": tasks,
        "models": models,
        "dataset_patterns": datasets,
        "seeds": seeds,
        "timeout": timeout,
        "resume": resume_enabled,
        "search_space_policy_override": args.search_space_policy,
        "include_reference": args.include_reference,
        "python": sys.executable,
        "devices": assigned,
        "datasets": catalog,
        "baselines": BASELINES,
        "excluded": excluded,
        "cases": cases,
    }
    bindings = attach_identity(manifest, ROOT)
    cases = manifest["cases"]
    if manifest["benchmark_mode"] == "main":
        validate_main_matrix(cases, policy, configuration_sources=manifest["configuration_sources"])
    save_manifest(output, manifest)
    write_json(output / "runtime_asset_paths.json", bindings)
    print(
        f"Selected {len(cases)} cases; profile={profile}; manifest={output / 'benchmark_manifest.json'}",
        flush=True,
    )
    if args.dry_run:
        return 0
    if args.check_compatibility:
        report = check_compatibility(cases, excluded=excluded, progress=True, mode=args.compatibility_mode)
        save_compatibility_report(output, report)
        counts = {
            status: sum(row["status"] == status for row in report["pairs"])
            for status in ("compatible", "incompatible", "error")
        }
        print(f"Compatibility report saved to {output}: {counts}")
        return 1 if counts["error"] else 0
    if (output / "benchmark_freeze.json").exists():
        verify_freeze(output)
    rows = []
    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {case['name']} device={case['device']}", flush=True)
        current = execute_case(case, output, timeout=timeout, resume=resume_enabled, **timing)
        rows.extend(current)
        save_summary(output, rows)
        print(
            "  " + ", ".join(f"{r['instance']}:{r['target']}={r['status']}" for r in current),
            flush=True,
        )
    counts = {
        status: sum(r["status"] == status for r in rows)
        for status in ("ok", "skipped", "error", "timeout")
    }
    freeze_run(output)
    print(f"Saved {len(rows)} rows to {output}: {counts}")
    return 1 if counts["error"] or counts["timeout"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
