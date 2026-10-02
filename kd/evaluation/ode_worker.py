"""Bounded post-fit scoring of immutable target outputs."""

import json
import os
import sys
from pathlib import Path

from kd.benchmark_execution import PhaseClock


def score(case, rows, seconds):
    from kd.evaluation.ode_rollout import evaluate
    from run_benchmark import _load_instances

    metrics = {}
    for instance, dataset in _load_instances(case):
        subset = [r for r in rows if r["instance"] == instance]
        metrics[instance] = evaluate(dataset, subset, case, seconds=seconds)
    return metrics


def main():
    from run_benchmark import write_json

    directory, attempt = [Path(p) for p in sys.argv[1:]]
    case = json.loads((directory / "case.json").read_text(encoding="utf8"))
    rows = json.loads((directory / "partial.json").read_text(encoding="utf8"))
    policy = json.loads((directory / "execution.json").read_text(encoding="utf8"))
    with PhaseClock(attempt / "phase.json", started=float(os.environ["KD_BENCHMARK_STARTED"])) as clock:
        clock.change("score")
        metrics = score(case, rows, policy["score_timeout"])
        if clock.times()["score_seconds"] > policy["score_timeout"]:
            raise TimeoutError("Coupled ODE scoring exceeded score_timeout")
        times = clock.times()
        try:
            import psutil
            cpu = psutil.Process().cpu_times()
            times["_self_cpu_seconds"] = cpu.user + cpu.system
        except ImportError:
            pass
        write_json(attempt / "candidate.json", {"metrics": metrics, "times": times})


if __name__ == "__main__":
    main()
