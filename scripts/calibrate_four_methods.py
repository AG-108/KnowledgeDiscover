"""Independent synthetic development checks; never read benchmark test targets."""

import argparse
import json
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def worker(spec_path, output):
    spec = json.loads(Path(spec_path).read_text())
    cpu_slot = spec["method"] == "sisso" or spec.get("diagnostic_cpu", False)
    available = sorted(os.sched_getaffinity(0))
    default_cpus = available[:4] if cpu_slot else available[4:8]
    cpus = spec.get("cpu_affinity", default_cpus)
    if len(cpus) != 4 or not set(cpus).issubset(available):
        raise ValueError("Calibration requires four available cores in each isolated slot")
    os.sched_setaffinity(0, cpus)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "4"
    if cpu_slot:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import numpy as np
    import torch

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    random.seed(spec["seed"])
    np.random.seed(spec["seed"])
    torch.manual_seed(spec["seed"])
    from kd.benchmark_deadline import deadline_scope
    from kd.benchmark_execution import PhaseClock
    from run_benchmark import expression_metrics, run_regression

    method = spec["method"]
    params = spec["parameters"].copy()
    fixture = spec["fixture"]
    start = time.perf_counter()
    if method == "dlga":
        from kd.model.kd_dlga import KD_DLGA

        generator = np.random.default_rng(314159)
        X = generator.uniform([-0.7, 0.0], [0.7, 0.6], (512, 2))
        z = X[:, 0] - 0.7123456789 * X[:, 1]
        y = (z + 0.3 * z * z + 0.2 * z**3).reshape(-1, 1)
        population = params.pop("population", 100)
        generations = params.pop("generations", 20)
        model = KD_DLGA(**params)
        model.pop_size, model.n_generations = population, generations
        with deadline_scope(spec["budget"]):
            model.fit(X.copy(), y.copy())
        expression = model.best_equation_
        result = expression_metrics(expression, "-0.7123456789*u_x", ["u", "u_x", "u_xx", "u_xxx"])
        result["lhs_correct"] = expression.split("=")[0].strip() == "u_t"
        result["expression"] = expression
        result["surrogate_optimization"] = getattr(model, "surrogate_optimization_", {})
        result["best_epoch"] = model.best_epoch
        result["minimum_validation_mse"] = min(model.val_loss_history)
    else:
        rng = np.random.default_rng(314159)
        X = rng.uniform(-1, 1, (64, 1))
        test = rng.uniform(-1, 1, (128, 1))
        power = 1 if fixture == "affine" else 2
        truth = f"0.731 + 1.23456789*x1**{power}"
        y = 0.731 + 1.23456789 * X[:, 0] ** power
        test_y = 0.731 + 1.23456789 * test[:, 0] ** power
        if method == "sisso":
            params["random_state"] = spec["seed"]
        if method == "symbolicgpt":
            params["seed"] = spec["seed"]
        problem = dict(
            X_train=X,
            y_train=y,
            X_test=test,
            y_test=test_y,
            variable_names=["x1"],
            target="y",
            split="independent_synthetic",
            metadata={},
            ground_truth=truth,
        )
        case = dict(
            model=method,
            model_params=params,
            task="sr",
            dataset="development_" + fixture,
            seed=spec["seed"],
            run_params={},
        )
        with PhaseClock(policy={"timeout": spec["budget"], "soft_deadline": True}):
            result = run_regression(problem, case)
    result.update(
        method=method,
        fixture=fixture,
        seed=spec["seed"],
        parameters=spec["parameters"],
        cpu_affinity=cpus,
        diagnostic_cpu=spec.get("diagnostic_cpu", False),
        experiment_role="independent_development",
        budget=spec["budget"],
        elapsed_seconds=time.perf_counter() - start,
    )
    result["passed"] = bool(
        result.get("exact_recovery") == 1
        and result.get("coefficient_error") is not None
        and result["coefficient_error"] < 1e-5
        and result.get("lhs_correct", True)
    )
    Path(output).write_text(json.dumps(result, indent=2, default=str))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec")
    parser.add_argument("--output")
    parser.add_argument("--suite")
    args = parser.parse_args()
    if args.spec:
        worker(args.spec, args.output)
        return
    suite = json.loads(Path(args.suite).read_text())
    directory = Path(args.suite).resolve().parent
    if any(row["method"] != "sisso" and not row.get("diagnostic_cpu") for row in suite):
        import fcntl

        gpu_lock = (directory.parent / ".gpu-calibration.lock").open("a")
        fcntl.flock(gpu_lock, fcntl.LOCK_EX)
    results = []
    for index, spec in enumerate(suite):
        job = directory / f"{index:02d}-{spec['method']}-{spec['fixture']}-seed{spec['seed']}"
        job.mkdir()
        (job / "spec.json").write_text(json.dumps(spec, indent=2))
        with (job / "run.log").open("w") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    __file__,
                    "--spec",
                    str(job / "spec.json"),
                    "--output",
                    str(job / "result.json"),
                ],
                cwd=job,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                code = process.wait(timeout=spec["budget"] + 45)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                code = process.wait()
        if (job / "result.json").exists():
            result = json.loads((job / "result.json").read_text())
        else:
            result = dict(
                method=spec["method"],
                fixture=spec["fixture"],
                seed=spec["seed"],
                passed=False,
                exit_code=code,
                error=(job / "run.log").read_text(errors="replace")[-3500:],
            )
            (job / "result.json").write_text(json.dumps(result, indent=2))
        results.append(result)
        (directory / "summary.json").write_text(json.dumps(results, indent=2, default=str))
        print(
            json.dumps(
                {
                    k: result.get(k)
                    for k in [
                        "method",
                        "fixture",
                        "seed",
                        "passed",
                        "expression",
                        "exact_recovery",
                        "coefficient_error",
                        "elapsed_seconds",
                        "exit_code",
                    ]
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
