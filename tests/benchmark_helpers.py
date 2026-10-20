"""Small shared helpers for benchmark cases and worker tests."""

import pytest

import run_benchmark as benchmark


def make_case(model="gplearn", dataset="Koza-2", **kwargs):
    catalog, _ = benchmark.dataset_catalog()
    return benchmark.build_experiments(catalog, models=[model], datasets=[dataset], **kwargs)[0]


def mock_target_workers(monkeypatch, case, statuses, *, interrupt_at=None, commit_first=False,
                        specs=None):
    """A worker transport double; real process/time boundaries have separate tests."""
    calls = []
    if specs is None:
        specs = [benchmark._target_spec(case["dataset"], 0, target, index, truth)
                 for index, (target, truth) in enumerate(benchmark._declared_targets(case)
                 or [("y", case.get("ground_truth"))])]

    def launch(case_path, attempt, policy, spec=None):
        if spec is None:
            benchmark.write_json(attempt / "targets.json", specs)
            return {"returncode": 0, "timeout_phase": None}
        index = len(calls)
        calls.append(spec["target_id"])
        (attempt / "run.log").write_text(f"attempt {attempt.name}", encoding="utf-8")
        if interrupt_at == index and not commit_first:
            raise KeyboardInterrupt
        status = statuses[index]
        outcome = dict(returncode=1 if status == "error" else 0,
                       timeout_phase="fit" if status == "timeout" else None,
                       worker_elapsed_seconds=1.0, fit_seconds=0.8,
                       setup_seconds=0.1, score_seconds=0.1)
        if status != "timeout":
            row = benchmark._base_row(case, spec["instance"], spec["target"])
            row.update(status=status, expression="x1", ground_truth_expression=spec["ground_truth"],
                       target_id=spec["target_id"], attempt=int(attempt.name), execution_policy=policy)
            benchmark.write_json(attempt / "result.json", benchmark._finalize_stop_reason(row))
        if interrupt_at == index:
            raise KeyboardInterrupt
        return outcome

    monkeypatch.setattr(benchmark, "_launch_worker", launch)
    return calls


def host_slots():
    import psutil

    from kd.benchmark_resources import slot_policy

    available = psutil.Process().cpu_affinity()
    if len(available) < 8:
        pytest.skip("Two D2 slots require eight available CPUs")
    return {slot: slot_policy(slot, cpus=cpus) for slot, cpus in
            (("cpu", available[:4]), ("gpu", available[4:8]))}

# Explicit expectations, independent of the production method registry.
SR_METHODS = {"dscv", "dso", "e2e", "symbolicgpt", "gplearn", "physo",
              "pysr", "sisso", "llmsr", "pyoperon"}
PDE_METHODS = {"pdefind_nd", "implicit_pde", "dscv", "spr", "sga", "dlga",
               "deepmod", "pdefind", "pdenet", "weakform", "eqgpt", "sindy", "pysindy"}
FDE_RESPONSE_DATASETS = {"fde_creep", "fde_silt", "fde_pt_fade", "fde_pt_fde", "fde_tsfade_fft"}
FDE_FIELD_DATASETS = {"fde_pt_fade_pde", "fde_pt_fde_pde", "fde_tsfade_fft_pde"}
