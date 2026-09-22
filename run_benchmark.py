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
import gc
import hashlib
import importlib
import io
import json
import math
import os
import pickle
import random
import re
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results" / "benchmark"
CONFIG_DIR = ROOT / "configs" / "benchmark"
DEFAULT_CONFIG_FILE = CONFIG_DIR / "benchmark.json"
TASKS = ("sr", "ode", "pde")

# Regression methods also fit individual ODE right-hand sides through our bridge.
BASELINES = {
    "dscv": {"tasks": TASKS, "example": "kd_dscv_example.py", "accelerator": "cpu"},
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
    "integral_weak_pde": {"tasks": ("pde",), "example": "kd_integral_weak_pde_example.py", "accelerator": "cpu"},
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
    "sindy": ("kd.model.kd_sindy", "SINDyModel"),
    "pysindy": ("kd.model.kd_sindy", "PySINDyModel"),
    "integral_weak_pde": ("kd.model.kd_wsindy", "IntegralWeakPDEModel"),
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
}
METRIC_PROTOCOL_VERSION = "2.0"
PDE_SUPPORT_METRIC_VERSION = "2.0"


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
        "resume",
        "devices",
        "profiles",
        "overrides",
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
    if set(overrides) - {"models", "datasets", "run", "evaluation"}:
        raise ValueError("overrides supports only models, datasets, run, evaluation")
    for key, choices in (("models", BASELINES), ("datasets", catalog)):
        section = overrides.get(key, {})
        if not isinstance(section, dict) or any(not isinstance(v, dict) for v in section.values()):
            raise ValueError(f"Each overrides.{key} entry must be a JSON object")
        unknown = set(section) - set(choices)
        if unknown:
            raise ValueError(f"Unknown overrides.{key}: {sorted(unknown)}")
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


class SkipCase(Exception):
    """Unsupported dataset/model combination or missing resource."""


def dataset_catalog():
    """All unified loaders and expressions understood by SymbolicRegressionDataset."""
    from kd.dataset import DATASET_REGISTRY
    from kd.dataset._catalog import KNOWN_BROKEN_DATASETS, TLC_UNSUPPORTED_STEADY_STATE

    catalog = {
        name: {
            "task": "sr" if info["category"] == "regression" else info["category"],
            "source": "catalog",
            "role": "dataset",
            "family": info.get("family", info["category"]),
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
    excluded = {n: "Broken loader; excluded by kd.dataset._catalog" for n in KNOWN_BROKEN_DATASETS}
    excluded.update(
        {
            "TLC/" + n: "Steady-state data; no integrated time-dependent loader"
            for n in TLC_UNSUPPORTED_STEADY_STATE
        }
    )
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
                if model in {"gplearn", "llmsr", "pyoperon", "pysr"}:
                    model_params["random_state"] = seed
                model_override = copy.deepcopy(config.get("models", {}).get(model, {}))
                override_device = model_override.pop("device", None)
                _deep_update(model_params, model_override)
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
                    "task": task,
                    "model": model,
                    "dataset": name,
                    "seed": seed,
                    "profile": profile,
                    "source": catalog[name]["source"],
                    "ground_truth": declared_truth,
                    "family": catalog[name].get("family", task),
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
                digest = hashlib.sha256(json.dumps(case, sort_keys=True).encode()).hexdigest()[:12]
                case["name"] = f"{task}_{model}_{name}_s{seed}_{digest}"
                cases.append(case)
    return cases


def check_compatibility(cases, excluded=None, progress=False):
    """Load one instance per dataset and apply every selected baseline adapter.

    This preflight checks dataset construction, task-level array conversion,
    baseline imports, temporal holdout construction, and the explicit shape
    constraints enforced by the benchmark. It intentionally does not optimize
    a model, so it is suitable for auditing the complete matrix.
    """
    from kd.dataset import GridPDEDataset

    grouped = {}
    for case in cases:
        grouped.setdefault(case["dataset"], []).append(case)

    runtime = {}
    selected_models = list(dict.fromkeys(case["model"] for case in cases))
    for model in selected_models:
        start = time.perf_counter()
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                io.StringIO()
            ):
                if model == "physo":
                    importlib.import_module("physo")
                elif model == "e2e":
                    _new_model(model, next(c["model_params"] for c in cases if c["model"] == model)).check_available()
                elif model == "pysindy":
                    importlib.import_module("pysindy")
                elif model != "weakform":
                    module, attribute = MODEL_CLASSES[model]
                    getattr(importlib.import_module(module), attribute)
            runtime[model] = {"status": "ok", "seconds": time.perf_counter() - start}
        except Exception as exc:
            runtime[model] = {
                "status": "error",
                "seconds": time.perf_counter() - start,
                "error": f"{type(exc).__name__}: {exc}",
            }

    pair_rows = []
    dataset_rows = []
    for index, (dataset_name, group) in enumerate(sorted(grouped.items()), 1):
        start = time.perf_counter()
        dataset = None
        instance_name = None
        load_error = None
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                io.StringIO()
            ):
                instance_name, dataset = next(_load_instances(group[0]))
            if group[0]["task"] == "pde":
                if isinstance(dataset, GridPDEDataset):
                    details = {
                        "class": type(dataset).__name__,
                        "field_shape": list(dataset.usol.shape),
                        "spatial_dimensions": len(dataset.spatial_vars),
                        "responses": dataset.n_response,
                        "time_points": len(dataset.t),
                    }
                else:
                    details = {
                        "class": type(dataset).__name__,
                        "points_shape": list(dataset.points.shape),
                        "field_shape": list(dataset.usol.shape),
                        "time_points": len(dataset.t),
                    }
            else:
                problems = list(regression_problems(dataset, group[0]))
                if not problems:
                    raise ValueError("Task adapter produced no regression targets")
                details = {
                    "class": type(dataset).__name__,
                    "targets": len(problems),
                    "n_features": int(problems[0]["X_train"].shape[1]),
                    "n_train": int(len(problems[0]["X_train"])),
                    "n_test": int(len(problems[0]["X_test"])),
                }
            load_status = "ok"
        except Exception as exc:
            load_status = "error"
            load_error = f"{type(exc).__name__}: {exc}"
            details = {}

        dataset_rows.append(
            {
                "dataset": dataset_name,
                "task": group[0]["task"],
                "status": load_status,
                "instance": instance_name,
                "seconds": time.perf_counter() - start,
                "error": load_error,
                **details,
            }
        )

        for case in group:
            row = {
                "task": case["task"],
                "baseline": case["model"],
                "dataset": dataset_name,
                "instance": instance_name,
                "status": "compatible",
                "reason": "",
            }
            if load_error:
                row.update(status="error", reason=load_error)
            elif runtime[case["model"]]["status"] != "ok":
                row.update(status="error", reason=runtime[case["model"]]["error"])
            elif case["task"] == "pde":
                try:
                    prepared = prepare_pde(dataset, case["model"])
                    temporal_block_holdout(prepared, case["run_params"]["test_size"])
                    if case["model"] == "sga" and dataset_name not in {
                        "burgers",
                        "kdv",
                        "chafee-infante",
                    }:
                        raise SkipCase(
                            "SGA SolverConfig currently defines only burgers, kdv and "
                            "chafee-infante presets"
                        )
                except SkipCase as exc:
                    row.update(status="incompatible", reason=str(exc))
                except Exception as exc:
                    row.update(status="error", reason=f"{type(exc).__name__}: {exc}")
            pair_rows.append(row)

        if progress and (index % 25 == 0 or index == len(grouped)):
            print(f"[{index}/{len(grouped)}] checked {dataset_name}", flush=True)
        del dataset
        gc.collect()

    return {
        "scope": {
            "datasets": len(grouped),
            "pairs": len(pair_rows),
            "excluded_datasets": excluded or {},
            "level": (
                "dataset load + task adapter + baseline import + explicit compatibility guards"
            ),
        },
        "runtime_imports": runtime,
        "dataset_loads": dataset_rows,
        "pairs": pair_rows,
    }


def save_compatibility_report(output_dir, report):
    """Write detailed compatibility results as JSON and a flat pair table."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "compatibility_report.json", report)
    with (output_dir / "compatibility_pairs.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        fields = ["task", "baseline", "dataset", "instance", "status", "reason"]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(report["pairs"])


def _load_instances(case):
    from kd.dataset import GridPDEDataset, SymbolicRegressionDataset, load_dataset

    kwargs = dict(case["dataset_params"])
    name = case["dataset"]
    if case["source"] == "symbolic":
        kwargs.setdefault("seed", case["seed"])
        yield name, SymbolicRegressionDataset(name=name, **kwargs)
    elif name == "wave_breaking":
        from examples._common import point_cloud_to_regular_grid

        keys = kwargs.pop("keys", None)
        kwargs.setdefault("nx", 512)
        with (ROOT / "kd/dataset/WaveBreaking.pkl").open("rb") as f:
            waves = pickle.load(f)
        for key in keys if keys is not None else sorted(waves):
            t, x, u = point_cloud_to_regular_grid(waves[key], **kwargs)
            yield key, GridPDEDataset(
                equation_name="wave_breaking",
                pde_data=None,
                x=x,
                t=t,
                usol=u.T,
                domain=None,
                epi=0.0,
                legacy=True,
            )
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

    from kd.dataset import ODEDataset, SymbolicRegressionDataset, load_dataset

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
        ground_truth = getattr(dataset, "sym_true", None)
    elif isinstance(dataset, SymbolicRegressionDataset):
        d = dataset.get_data()
        X_train, y_train, X_test, y_test = (
            d[k] for k in ("X_train", "y_train", "X_test", "y_test")
        )
        names = [f"x{i + 1}" for i in range(X_train.shape[1])]
        split, targets, metadata = "provided_benchmark_split", ["y"], {}
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
        X_train, y_train, X_test, y_test = (
            dataset.X[train],
            dataset.y[train],
            dataset.X[test],
            dataset.y[test],
        )
        names = dataset.variable_names
        split = "group_holdout" if dataset.groups is not None else "random_holdout"
        targets, metadata = ["y"], {}
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
    "log": 1,
    "sqrt": 1,
    "abs": 1,
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
                "log": sp.log,
                "sqrt": sp.sqrt,
                "abs": sp.Abs,
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
        "log": sp.log,
        "sqrt": sp.sqrt,
        "pi": sp.pi,
    }
    text = re.sub(r"\b(?:numpy|np|math)\.", "", text).replace("^", "**")
    try:
        return sp.sympify(text, locals={**functions, **symbols})
    except (TypeError, ValueError, SyntaxError, sp.SympifyError):
        return None


def expression_metrics(expression, ground_truth=None, variable_names=()):
    """Return a syntax-tree-like token count and algebraic recovery flag."""
    import sympy as sp

    discovered = expression_sympy(expression, variable_names)
    complexity_text = str(expression).split("=", 1)[-1]
    tokens = re.findall(
        r"[A-Za-z_]\w*|(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|\*\*|[-+*/^]",
        complexity_text,
    )
    complexity = len(tokens) if tokens else None
    exact = None
    truth = expression_sympy(ground_truth, variable_names)
    if discovered is not None and truth is not None:
        try:
            exact = int(sp.simplify(discovered - truth) == 0)
        except (TypeError, ValueError, NotImplementedError):
            exact = None
    return {
        "ground_truth_expression": ground_truth,
        "expression_complexity": complexity,
        "exact_recovery": exact,
    }


def _new_model(name, params):
    module, attr = MODEL_CLASSES[name]
    model = getattr(importlib.import_module(module), attr)(**params)
    if name == "e2e":
        try:
            model.check_available()
        except RuntimeError as exc:
            raise SkipCase(str(exc)) from exc
    return model


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
    result = model.train(n_epochs=case["run_params"]["dscv_epochs"], verbose=False)
    return {
        "expression": str(result.get("expression", "")),
        "reward": result.get("r"),
        "notes": "DISCOVER regression bridge; no fixed-coefficient held-out prediction API. test_mse is unavailable.",
    }


def run_regression(problem, case):
    name, params = case["model"], copy.deepcopy(case["model_params"])
    if name == "dscv":
        result = _train_dscv_regression(problem, case, params)
    elif name == "physo":
        import physo
        import physo.learn.monitoring as monitoring
        import torch

        options = dict(params)
        options.setdefault("X_names", [f"x{i + 1}" for i in range(problem["X_train"].shape[1])])
        options.setdefault("y_name", "y")
        options.setdefault("y_units", [0, 0, 0])
        options.setdefault(
            "fixed_consts_units", [[0, 0, 0] for _ in options.get("fixed_consts", [])]
        )
        options.setdefault("run_config", copy.deepcopy(physo.config.config0.config0))
        options.setdefault(
            "get_run_logger", lambda: monitoring.RunLogger(save_path="physo.log", do_save=True)
        )
        options.setdefault(
            "get_run_visualiser",
            lambda: monitoring.RunVisualiser(
                epoch_refresh_rate=1, do_show=False, do_prints=False, do_save=False
            ),
        )
        expr, _ = physo.SR(problem["X_train"].T, problem["y_train"], **options)
        if expr is None:
            raise RuntimeError("PhySO did not return an expression")
        result = _score_regression(
            {
                "expression": str(expr.get_infix_pretty()),
                "notes": "Dimensionless PhySO configuration; physical units require explicit configuration.",
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
        model.fit(
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
        model = _new_model(name, params)
        if name in {"llmsr", "sindy", "pysindy", "e2e"}:
            model.fit(
                problem["X_train"],
                problem["y_train"],
                variable_names=problem["variable_names"],
            )
        else:
            model.fit(problem["X_train"], problem["y_train"])
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
    result.update(
        target=problem["target"],
        split=problem["split"],
        n_train=len(problem["X_train"]),
        n_test=len(problem["X_test"]),
        variable_names=problem["variable_names"],
        details=problem["metadata"],
    )
    if name == "e2e":
        result["method_provenance"] = model.provenance_
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
    return result


def prepare_pde(dataset, model_name):
    """Keep spatial/channel axes explicit; never silently flatten them."""
    import numpy as np

    from kd.dataset import GridPDEDataset

    if not isinstance(dataset, GridPDEDataset):
        raise SkipCase(
            "PDE baselines require a regular grid; scatter data need an explicit adapter"
        )
    coords = [np.asarray(v) for v in dataset.coords_spatial.values()] + [np.asarray(dataset.t)]
    if any(len(c) < 3 or not np.isfinite(c).all() or not np.all(np.diff(c) > 0) for c in coords):
        raise SkipCase("PDE fitting requires >=3 finite, strictly increasing points per axis")
    if not np.isfinite(dataset.usol).all():
        raise SkipCase(
            "PDE grid contains NaN/Inf (e.g. masked geometry); an explicit mask adapter is required"
        )
    dim = len(dataset.spatial_vars)
    if model_name in {"pdenet", "weakform"}:
        if dim != 2:
            raise SkipCase(f"{model_name} example adapter requires two spatial dimensions")
        if model_name == "pdenet":
            if any(
                not np.allclose(np.diff(c), np.diff(c)[0], rtol=1e-3, atol=1e-10) for c in coords
            ):
                raise SkipCase("PDE-Net requires uniform spatial and time axes")
        return dataset
    if dim != 1 or dataset.n_response != 1:
        raise SkipCase(f"{model_name} adapter supports a scalar field in one spatial dimension")
    if model_name in {"dscv", "spr", "sga"} and any(
        not np.allclose(np.diff(c), np.diff(c)[0], rtol=1e-3, atol=1e-10) for c in coords
    ):
        raise SkipCase(f"{model_name} finite-difference adapter requires uniform axes")
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
    for attr in ("registry_name", "legacy_name", "sym_true", "rollout_metadata"):
        if hasattr(dataset, attr):
            setattr(result, attr, getattr(dataset, attr))
    return result


def _slice_pde_time(dataset, start, stop):
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
        usol=dataset.usol[..., start:stop],
        domain=domain,
        epi=dataset.epi,
        time_var=dataset.time_var,
        legacy=dataset.legacy,
    )
    for attr in ("registry_name", "legacy_name", "sym_true"):
        if hasattr(dataset, attr):
            setattr(result, attr, getattr(dataset, attr))
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
    return _slice_pde_time(dataset, 0, split), _slice_pde_time(dataset, split - 1, nt), split


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
        return {}
    free = sorted(parsed.free_symbols, key=str)
    if any(str(symbol) not in namespace for symbol in free):
        return {}
    try:
        rhs = sp.lambdify(free, parsed, modules="numpy")(
            *(namespace[str(symbol)] for symbol in free)
        )
        rhs = np.broadcast_to(np.asarray(rhs, dtype=float), field.shape)
    except (TypeError, ValueError, ZeroDivisionError, FloatingPointError):
        return {}
    ut = np.gradient(field, dataset.t, axis=-1, edge_order=2)
    # Index zero is the rollout initial condition from the training block.
    target, prediction = ut[..., 1:].reshape(-1), rhs[..., 1:].reshape(-1)
    target_finite = np.isfinite(target)
    prediction_finite = np.isfinite(prediction)
    finite = target_finite & prediction_finite
    total = int(target.size)
    result = {
        "equation_residual_points": int(finite.sum()),
        "equation_residual_total_points": total,
        "equation_residual_finite_coverage": float(finite.sum() / total) if total else None,
    }
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

    from kd.evaluation.pde_rollout import validate_solver, rollout

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


def run_weakform(dataset, params):
    """Example finite-difference/lstsq library, not integral weak form/WSINDy.

    Fit every response independently, without cross-channel terms.
    """
    import numpy as np

    step, time_step = params["spatial_stride"], params["time_stride"]
    if step < 1 or time_step < 1:
        raise ValueError("Weakform strides must be positive")
    spatial = list(dataset.coords_spatial.values())
    x, y, t = spatial[0][::step], spatial[1][::step], dataset.t[::time_step]
    if min(len(x), len(y), len(t)) < 3:
        raise SkipCase("Weakform downsampled axes need >=3 points; reduce the strides")
    fields = dataset.usol[None] if dataset.legacy else dataset.usol
    names = [
        "1",
        "u",
        "u_x",
        "u_y",
        "u_xx",
        "u_yy",
        "u_xy",
        "u^2",
        "u*u_x",
        "u*u_y",
        "u_x^2",
        "u_y^2",
        "u_x*u_y",
    ]
    equations, residuals = [], []
    for c, field in enumerate(fields):
        u = field[::step, ::step, ::time_step]
        ux, uy, ut = np.gradient(u, x, y, t, edge_order=2)
        uxx = np.gradient(ux, x, axis=0, edge_order=2)
        uyy = np.gradient(uy, y, axis=1, edge_order=2)
        uxy = np.gradient(ux, y, axis=1, edge_order=2)
        terms = [
            np.ones_like(u),
            u,
            ux,
            uy,
            uxx,
            uyy,
            uxy,
            u * u,
            u * ux,
            u * uy,
            ux * ux,
            uy * uy,
            ux * uy,
        ]
        theta = np.column_stack([v.ravel() for v in terms])
        coeffs, *_ = np.linalg.lstsq(theta, ut.ravel(), rcond=None)
        coeffs[np.abs(coeffs) <= params["threshold"]] = 0
        equations.append(f"d(u{c})/dt = " + _linear_expression(names, coeffs))
        residuals.append(float(np.mean((theta @ coeffs - ut.ravel()) ** 2)))
    return {
        "expression": "; ".join(equations),
        "train_residual_mse": float(np.mean(residuals)),
        "notes": "Example finite-difference + thresholded least squares, not integral weak form; channels fitted independently.",
    }


def run_pde(dataset, case):
    import numpy as np

    name, params, options = case["model"], copy.deepcopy(case["model_params"]), case["run_params"]
    dataset = prepare_pde(dataset, name)
    train_dataset, evaluation_dataset, split_index = temporal_block_holdout(
        dataset, options["test_size"]
    )
    model = None
    if name == "sga" and case["dataset"] not in {"burgers", "kdv", "chafee-infante"}:
        raise SkipCase(
            "SGA SolverConfig currently defines only burgers, kdv and chafee-infante presets"
        )
    if name == "weakform":
        result = run_weakform(train_dataset, params)
    elif name == "integral_weak_pde":
        model = _new_model(name, params)
        model.fit(train_dataset.usol, train_dataset.x, train_dataset.t)
        result = {
            "expression": "u_t = " + model.best_expression_,
            "weak_expression": model.weak_expression_,
            "train_residual_mse": float(model.weak_residual_mse_),
            "notes": "Native integral weak-form scalar 1D baseline; bounded adaptation, not the complete upstream WSINDy-PDE algorithm.",
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
            found = model.train(n_epochs=options["dscv_epochs"], verbose=False)
            result = {"expression": str(found.get("expression", "")), "reward": found.get("r")}
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
            model.fit(X, y, variable_names=variable_names)
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
            model.fit_dataset(train_dataset, problem_name=case["dataset"])
            result = {"expression": str(model.best_pde_), "aic": float(model.best_score_)}
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
            model.fit(X, y)
            result = {"expression": str(model.best_equation_), "n_train": len(X)}
        elif name == "eqgpt":
            model.fit(train_dataset, equation_name=case["dataset"])
            result = {
                "expression": str(model.best_pde_),
                "reward": float(model.best_award_),
                "notes": "EqGPT handbook pretraining excludes the target row only for recognized equation names.",
            }
        else:
            if name == "pdenet":
                from types import SimpleNamespace

                spatial = list(train_dataset.coords_spatial.values())
                fit_data = SimpleNamespace(
                    usol=(train_dataset.usol[None] if train_dataset.legacy else train_dataset.usol),
                    x=spatial[0],
                    y=spatial[1],
                    t=train_dataset.t,
                    domain=train_dataset.domain,
                )
            else:
                fit_data = train_dataset
            model.fit(fit_data)
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
                ] += " Current PDEFindModel uses thresholded least squares, not STRidge."
            else:
                result[
                    "notes"
                ] += " PDE-Net follows the example wrapper: one x-derived dx for both axes and periodic padding, including rectangular grids."
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
    result.update(pde_equation_residual(result.get("expression"), evaluation_dataset))
    try:
        if model is not None and name in {"pdefind", "pdenet"}:
            result.update(pde_rollout_metrics(model, dataset, split_index))
    except Exception as exc:
        note = f"Native held-out rollout unavailable: {type(exc).__name__}: {exc}"
        result["notes"] = " ".join(filter(None, [result.get("notes"), note]))
    try:
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
        target="PDE",
        n_test=int(np.prod(dataset.usol[..., split_index:].shape)),
        n_train_time=split_index,
        n_test_time=len(dataset.t) - split_index,
        train_time_range=[float(dataset.t[0]), float(dataset.t[split_index - 1])],
        test_time_range=[float(dataset.t[split_index]), float(dataset.t[-1])],
        field_shape=list(dataset.usol.shape),
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
        metric_protocol_version=METRIC_PROTOCOL_VERSION,
        instance=instance or case["dataset"],
        family=case.get("family", case.get("task")),
        target=target,
        runtime=None,
        peak_memory_mb=None,
        peak_gpu_memory_mb=None,
        expression="",
        ground_truth_expression=case.get("ground_truth"),
        expression_complexity=None,
        exact_recovery=None,
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
        recovery_eligible=bool(case.get("ground_truth")),
        recovery_evaluator_status=("pending" if case.get("ground_truth") else "not_applicable"),
        pde_support_recovery=None,
        exact_symbolic_recovery=None,
        exact_symbolic_recovery_semantics=(
            "legacy alias of pde_support_recovery" if case.get("task") == "pde" else None
        ),
        pde_support_metric_version=(PDE_SUPPORT_METRIC_VERSION if case.get("task") == "pde" else None),
        term_support_accuracy=None,
        term_support_precision=None,
        term_support_recall=None,
        native_rollout_nrmse=None,
        equation_rollout_nrmse=None,
        equation_rollout_status=None,
        pic_status=("not_requested" if case.get("task") == "pde" else None),
        pic=None,
        pic_r_loss=None,
        pic_p_loss=None,
        pic_message="",
        pic_version=None,
        pic_cache_key=None,
        pic_reference_cache_hit=None,
        status="error",
        error="",
    )
    return row


def _failure(case, exc, instance=None, target=None, ground_truth=None):
    row = _base_row(case, instance, target)
    if ground_truth:
        row.update(
            ground_truth_expression=ground_truth,
            recovery_eligible=True,
            recovery_evaluator_status="not_run",
        )
    row.update(
        status=(
            "skipped" if isinstance(exc, (SkipCase, ImportError, FileNotFoundError)) else "error"
        ),
        error=f"{type(exc).__name__}: {exc}",
    )
    traceback.print_exc()
    return row


def run_case(case, work_dir=None, checkpoint=None):
    """One matrix entry, with separate rows/errors per generated PDE/ODE target."""
    import numpy as np

    rows = []
    work_dir = Path(work_dir or RESULTS_DIR / "cases" / case["name"]).resolve()

    def append(row):
        rows.append(row)
        if checkpoint:
            write_json(checkpoint, rows)

    try:
        check_device_available(case["device"])
        for instance_index, (instance, dataset) in enumerate(_load_instances(case)):
            instance_row_start = len(rows)
            try:
                problems = regression_problems(dataset, case) if case["task"] != "pde" else [None]
                for target_index, problem in enumerate(problems):
                    target = problem["target"] if problem else "PDE"
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
                                    "gplearn",
                                    "llmsr",
                                    "pdefind",
                                    "pyoperon",
                                    "pysindy",
                                    "pysr",
                                    "sindy",
                                    "weakform",
                                    "integral_weak_pde",
                                }:
                                    import torch

                                    torch.manual_seed(case["seed"])
                                result = (
                                    run_pde(dataset, case)
                                    if problem is None
                                    else run_regression(problem, case)
                                )
                            row = _base_row(case, instance, target)
                            row["status"] = "ok"
                            row.update(result)
                            if row.get("ground_truth_expression"):
                                row["recovery_eligible"] = True
                                if case["task"] != "pde":
                                    row["recovery_evaluator_status"] = (
                                        "parsed" if row.get("exact_recovery") is not None else "unparsed"
                                    )
                            if not row["expression"] or row["expression"] == "None":
                                raise RuntimeError(
                                    "Training completed without a discovered expression"
                                )
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
            except Exception as exc:
                truths = case.get("ground_truth")
                if len(rows) == instance_row_start and isinstance(truths, (list, tuple)):
                    for index, truth in enumerate(truths):
                        append(_failure(case, exc, instance, str(index), truth))
                else:
                    append(_failure(case, exc, instance))
    except Exception as exc:
        targets = _declared_targets(case)
        if targets and not rows:
            for target, truth in targets:
                append(_failure(case, exc, case["dataset"], target, truth))
        else:
            append(_failure(case, exc))
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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(value), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


CSV_FIELDS = [
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
        version = row.get("metric_protocol_version") or "legacy/unknown"
        groups.setdefault((version, "all", "all"), []).append(row)
        groups.setdefault((version, row["task"], row["model"]), []).append(row)
    aggregates = []
    for (version, task, model), group in sorted(groups.items()):
        statuses = [row.get("status") for row in group]
        runtimes = _numeric_values(group, "runtime")
        memories = _numeric_values(group, "peak_memory_mb")
        gpu_memories = _numeric_values(group, "peak_gpu_memory_mb")
        # New rows explicitly declare the denominator before execution.  For
        # old result files, a present metric is the only safe evidence that the
        # row was eligible; this preserves readability without silently
        # reclassifying old failed rows.
        exact_eligible = [
            row
            for row in group
            if row.get("task") != "pde"
            and row.get("status") != "skipped"
            and (row.get("recovery_eligible") is True or row.get("exact_recovery") is not None)
        ]
        support_eligible = [
            row
            for row in group
            if row.get("task") == "pde"
            and row.get("status") != "skipped"
            and (
                row.get("recovery_eligible") is True
                or row.get("pde_support_recovery") is not None
                or row.get("exact_symbolic_recovery") is not None
            )
        ]
        exact_valid = [row for row in exact_eligible if row.get("status") == "ok"]
        recovered = _numeric_values(exact_valid, "exact_recovery")
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
    family_rows = [entry for metric in report_metrics
                   for entry in seed_family_summary(rows, metric)]
    paired_rows = []
    models = sorted({row.get("model") for row in rows if row.get("model")})
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


def _declared_targets(case):
    """Target-level failure accounting from predeclared truth, never fitted outputs."""
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


def execute_case(case, output_dir, timeout=None, resume=False):
    """Separate subprocesses prevent model globals and relative log paths colliding."""
    directory = (output_dir / "cases" / case["name"]).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    result_path, request_path = directory / "result.json", directory / "case.json"
    if resume and result_path.exists():
        previous = json.loads(result_path.read_text(encoding="utf-8"))
        if previous and all(row["status"] == "ok" for row in previous):
            return previous
    result_path.unlink(missing_ok=True)
    partial_path = directory / "partial.json"
    partial_path.unlink(missing_ok=True)
    write_json(request_path, case)
    log_path = directory / "run.log"
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        try:
            env = dict(os.environ, PYTHONIOENCODING="utf-8", MPLBACKEND="Agg")
            process = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--_case-file", str(request_path)],
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout,
            )
            if process.returncode != 0 or not result_path.exists():
                raise RuntimeError(f"Worker exited with code {process.returncode}; see {log_path}")
            rows = json.loads(result_path.read_text(encoding="utf-8"))
        except (subprocess.TimeoutExpired, RuntimeError) as exc:
            rows = (
                json.loads(partial_path.read_text(encoding="utf-8"))
                if partial_path.exists()
                else []
            )
            targets = _declared_targets(case)
            completed = {row.get("target") for row in rows}
            completed_truth = {str(row.get("ground_truth_expression")) for row in rows}
            pending = [(target, truth) for target, truth in targets
                       if target not in completed and str(truth) not in completed_truth]
            if not targets:
                pending = [(None, case.get("ground_truth"))]
            elapsed = time.perf_counter() - start
            unaccounted = max(0.0, elapsed - sum(float(row.get("runtime") or 0) for row in rows))
            for index, (target, truth) in enumerate(pending):
                row = _base_row(case, case["dataset"], target)
                row.update(
                    status="timeout" if isinstance(exc, subprocess.TimeoutExpired) else "error",
                    error=str(exc), runtime=unaccounted if index == 0 else 0.0,
                    ground_truth_expression=truth, recovery_eligible=bool(truth),
                    recovery_evaluator_status="not_run", worker_elapsed_seconds=elapsed,
                )
                rows.append(row)
            if not pending:
                for row in rows:
                    row["worker_completion_error"] = str(exc)
    for row in rows:
        row["log_path"] = str(log_path)
    write_json(result_path, rows)
    return rows


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
        "--config", type=Path, help=f"Total JSON config (default: {DEFAULT_CONFIG_FILE})"
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--timeout", type=float, help="Seconds per model/dataset/seed worker (default: unlimited)"
    )
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        help="Reuse successful cases with identical resolved configuration",
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
        help="Load one minimal dataset instance per selected baseline/dataset pair and exit",
    )
    parser.add_argument("--_case-file", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args._case_file:
        case = json.loads(args._case_file.read_text(encoding="utf-8"))
        directory = args._case_file.parent
        rows = run_case(case, directory, checkpoint=directory / "partial.json")
        write_json(directory / "result.json", rows)
        return 0
    try:
        catalog, excluded = dataset_catalog()
        settings, config_path = load_benchmark_config(args.config)
        validate_benchmark_config(settings, catalog)
        baseline_configs = load_baseline_configs(settings["baseline_config_dir"])
        profile = args.profile or settings["profile"]
        tasks = args.tasks if args.tasks is not None else settings["tasks"]
        models = args.models if args.models is not None else settings["models"]
        datasets = args.datasets if args.datasets is not None else settings["datasets"]
        seeds = args.seeds if args.seeds is not None else settings["seeds"]
        timeout = args.timeout if args.timeout is not None else settings["timeout"]
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
        if not cases:
            raise ValueError("Selection contains no compatible task/model/dataset combinations")
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
        "config": str(config_path),
        "profile": profile,
        "tasks": tasks,
        "models": models,
        "dataset_patterns": datasets,
        "seeds": seeds,
        "timeout": timeout,
        "resume": resume_enabled,
        "python": sys.executable,
        "devices": assigned,
        "datasets": catalog,
        "baselines": BASELINES,
        "excluded": excluded,
        "cases": cases,
    }
    write_json(output / "benchmark_manifest.json", manifest)
    print(
        f"Selected {len(cases)} cases; profile={profile}; manifest={output / 'benchmark_manifest.json'}",
        flush=True,
    )
    if args.dry_run:
        return 0
    if args.check_compatibility:
        report = check_compatibility(cases, excluded=excluded, progress=True)
        save_compatibility_report(output, report)
        counts = {
            status: sum(row["status"] == status for row in report["pairs"])
            for status in ("compatible", "incompatible", "error")
        }
        print(f"Compatibility report saved to {output}: {counts}")
        return 1 if counts["error"] else 0
    rows = []
    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {case['name']} device={case['device']}", flush=True)
        current = execute_case(case, output, timeout=timeout, resume=resume_enabled)
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
    print(f"Saved {len(rows)} rows to {output}: {counts}")
    return 1 if counts["error"] or counts["timeout"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
