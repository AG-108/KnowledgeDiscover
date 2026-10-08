"""Export candidate configuration/source/capacity checks without data loading or fitting."""

import argparse
import copy
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kd.benchmark_planning import plan_chunks  # noqa: E402
from kd.benchmark_protocol import digest, method_configuration, validate_main_matrix  # noqa: E402
from run_benchmark import (  # noqa: E402
    _declared_targets,
    build_experiments,
    dataset_catalog,
    execution_policy,
    load_baseline_configs,
    load_benchmark_config,
    validate_benchmark_config,
    write_json,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/benchmark/core_main_candidate.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    catalog, _ = dataset_catalog()
    settings, _ = load_benchmark_config(args.config)
    validate_benchmark_config(settings, catalog)
    cases = build_experiments(catalog, tasks=settings["tasks"], models=settings["models"],
                              datasets=settings["datasets"], seeds=settings["seeds"], profile=settings["profile"],
                              config=settings["overrides"], baseline_configs=load_baseline_configs(settings["baseline_config_dir"]),
                              devices=settings["devices"], run_defaults=settings["profiles"][settings["profile"]]["run"])
    for case in cases:
        if case["dataset"].startswith("ode_core_"):
            case["task_grammar"] = settings["core_ode_grammar"]
    policy = execution_policy(settings["timeout"], **{key: settings[key] for key in
                              ("soft_deadline", "timeout_grace", "setup_timeout", "score_timeout")})
    configs, errors = {}, set()
    for case in cases:
        key = f"{case['model']}/{case['task']}"
        configuration = method_configuration(case)
        configs.setdefault(key, {"config_sha256": digest(configuration), "configuration": configuration,
                                 "source": copy.deepcopy(settings["configuration_sources"].get(case["model"], {}))})
        if configs[key]["config_sha256"] != digest(configuration):
            errors.add(f"{key}: per-dataset configuration")
        try:
            validate_main_matrix([case], policy, configuration_sources=settings["configuration_sources"])
        except ValueError as exc:
            errors.add(str(exc))
    counts = {case["name"]: max(1, len(_declared_targets(case))) for case in cases}
    queue = plan_chunks(cases, counts, policy)
    output = {"schema": "main-candidate-audit-v1", "status": "candidate; budget and deployment await server evidence",
              "n_cases": len(cases), "n_declared_fits_estimate": sum(counts.values()),
              "cases_by_task": dict(Counter(case["task"] for case in cases)),
              "datasets_by_task": {task: len({case['dataset'] for case in cases if case['task'] == task}) for task in settings["tasks"]},
              "configurations": configs, "admission_errors": sorted(errors), "budget_estimate": queue,
              "target_count_semantics": "declarations only; certified plans must enumerate actual instances before fitting"}
    write_json(args.output, output)
    print({key: output[key] for key in ("n_cases", "n_declared_fits_estimate", "cases_by_task", "datasets_by_task")})
    print(f"Admission conditions still pending: {len(errors)}; audit={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
