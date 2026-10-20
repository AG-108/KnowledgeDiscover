"""Instance/seed/target preflight, without fitting or changing task denominators."""

import contextlib
import gc
import importlib
import io
import json
import time

from kd.benchmark_capabilities import failure_status, validate_pde
from kd.dataset._equations import targets


def _runtime(runner, model, params):
    start = time.perf_counter()
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            if model in {"e2e", "sisso"}:
                runner._new_model(model, params).check_available()
            elif model in {"physo", "pysindy"}:
                importlib.import_module(model)
            else:
                module, attribute = runner.MODEL_CLASSES[model]
                getattr(importlib.import_module(module), attribute)
        return {"status": "ok", "seconds": time.perf_counter() - start}
    except Exception as exc:
        return {"status": "error", "seconds": time.perf_counter() - start,
                "error": f"{type(exc).__name__}: {exc}"}


def check(cases, *, runner, excluded=None, progress=False, mode="sample", check_runtime=True):
    if mode not in {"sample", "full"}:
        raise ValueError("Compatibility mode must be sample or full")
    grouped, runtime, runtime_keys = {}, {}, {}
    for case in cases:
        key = (case["dataset"], case["seed"], json.dumps(case["dataset_params"], sort_keys=True))
        grouped.setdefault(key, []).append(case)
        variant = (case["model"], json.dumps(case["model_params"], sort_keys=True))
        if variant not in runtime_keys:
            model = case["model"]
            runtime_key = model if model not in runtime else model + ":" + case["name"]
            runtime_keys[variant] = runtime_key
            runtime[runtime_key] = (_runtime(runner, model, case["model_params"]) if check_runtime
                                    else {"status": "not_checked"})

    pair_rows, dataset_rows, checked_groups = [], [], []

    def emit(case, instance, target, *, load_status="ok", capability="supported",
             preparation="ok", reason=""):
        variant = (case["model"], json.dumps(case["model_params"], sort_keys=True))
        runtime_key = runtime_keys[variant]
        native = runtime[runtime_key]
        code = load_status if load_status != "ok" else (
            capability if capability != "supported" else preparation)
        status = "compatible"
        if code not in {"ok", "supported"}:
            status = ("error" if code in {"error", "missing_asset", "missing_dependency"}
                      else "incompatible")
        elif native["status"] == "error":
            status, reason = "error", native["error"]
        pair_rows.append({
            "task": case["task"], "baseline": case["model"], "dataset": case["dataset"],
            "case": case["name"], "instance": instance, "seed": case["seed"], "target": target,
            "status": status, "reason": reason, "capability_status": capability,
            "load_status": load_status, "preparation_status": preparation,
            "asset_status": "missing_asset" if load_status == "missing_asset" else (
                "available" if load_status == "ok" else "not_checked"),
            "instance_eligibility": ("eligible" if preparation == "ok" else
                                     "ineligible" if preparation in {
                                         "instance_ineligible", "insufficient_support"} else "not_checked"),
            "runtime_status": native["status"], "runtime_key": runtime_key,
            "fit_evidence": "not_run",
            "equation_targets": [],
        })

    def inspect(instance, dataset, group):
        dataset_rows.append({
            "dataset": group[0]["dataset"], "task": group[0]["task"], "instance": instance,
            "seed": group[0]["seed"], "status": "ok", "class": type(dataset).__name__,
        })
        for case in group:
            pde = case["task"] in {"pde", "steady_pde"}
            declared = ([t for t, _ in runner._pde_target_bindings(case, dataset)] if pde else
                        [t for t, _ in runner._declared_targets(case)])
            declared = declared or [getattr(dataset, "target_name", "y")]
            capability, preparation, reason = "supported", "ok", ""
            if pde:
                try:
                    validate_pde(dataset, case.get("reference_adapter", case["model"]))
                except Exception as exc:
                    capability = failure_status(exc)
                    preparation = "not_checked"
                    if capability in {"instance_ineligible", "insufficient_support"}:
                        preparation, capability = capability, "supported"
                    reason = f"{type(exc).__name__}: {exc}"
            if capability == "supported" and preparation == "ok":
                try:
                    problems = list(runner._case_problems(dataset, case))
                    if not problems:
                        raise ValueError("Task adapter produced no targets")
                    if not pde:
                        declared = [problem["target"] for problem in problems]
                    else:
                        model = case.get("reference_adapter", case["model"])
                        if model in {"pdefind_nd", "implicit_pde"}:
                            from kd.evaluation.equation import prepare_views
                            prepare_views(dataset, case)
                        else:
                            prepared = runner.prepare_pde(dataset, model)
                            if case.get("fde_pde_protocol_version"):
                                from kd.evaluation.fractional import temporal_views
                                train, _, _ = temporal_views(prepared, case["run_params"]["test_size"])
                            else:
                                train, _, _ = runner.temporal_block_holdout(prepared, case["run_params"]["test_size"])
                            if model in {"sindy", "pysindy"}:
                                params = case["model_params"]
                                runner.pde_sindy_problem(
                                    train, derivative_order=params.get("derivative_order", 3),
                                    trim_boundary=params.get("trim_boundary", 2),
                                )
                except Exception as exc:
                    preparation, reason = failure_status(exc), f"{type(exc).__name__}: {exc}"
            for target in declared:
                emit(case, instance, target, capability=capability, preparation=preparation, reason=reason)
                if pde:
                    pair_rows[-1]["equation_targets"] = [t.to_dict() for t in targets(dataset)]

    for index, (key, group) in enumerate(sorted(grouped.items()), 1):
        start, seen, exhausted = time.perf_counter(), [], False
        iterator = runner._load_instances(group[0])
        try:
            while True:
                try:
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        instance, dataset = next(iterator)
                except StopIteration:
                    exhausted = True
                    if not seen:
                        raise ValueError("Dataset produced no instances")
                    break
                seen.append(instance)
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    inspect(instance, dataset, group)
                del dataset
                if mode == "sample":
                    break
        except Exception as exc:
            code, reason = failure_status(exc), f"{type(exc).__name__}: {exc}"
            dataset_rows.append({"dataset": key[0], "task": group[0]["task"], "seed": key[1],
                                 "instance": None, "status": code, "error": reason})
            for case in group:
                with contextlib.redirect_stderr(io.StringIO()):
                    failures = runner._case_failure_targets(case, exc)
                for spec in failures:
                    if spec["instance"] not in seen:
                        emit(case, spec["instance"], spec["target"], load_status=code,
                             capability="not_checked", preparation="not_checked", reason=reason)
        finally:
            if hasattr(iterator, "close"):
                iterator.close()
        checked_groups.append({"dataset": key[0], "seed": key[1],
                               "dataset_params": group[0]["dataset_params"],
                               "instances": seen, "enumeration_complete": exhausted,
                               "seconds": time.perf_counter() - start})
        if progress:
            print(f"[{index}/{len(grouped)}] checked {key[0]} seed={key[1]} "
                  f"instances={len(seen)} mode={mode}", flush=True)
        gc.collect()
    return {
        "scope": {"datasets": len({c["dataset"] for c in cases}), "pairs": len(pair_rows),
                  "mode": mode, "seeds": sorted({c["seed"] for c in cases}),
                  "excluded_datasets": excluded or {}, "checked_groups": checked_groups,
                  "enumeration_complete": all(g["enumeration_complete"] for g in checked_groups),
                  "level": "dataset load + target preparation + adapter capability + runtime import",
                  "fit_evidence": "not_run"},
        "runtime_imports": runtime, "dataset_loads": dataset_rows, "pairs": pair_rows,
    }
