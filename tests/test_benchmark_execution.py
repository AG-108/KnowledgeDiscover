"""Exercise real OS processes: deadlines, descendants, locks and native target fits."""

import json
import os
import sys
import time
from pathlib import Path

import pytest

import run_benchmark as benchmark
from kd.benchmark_execution import atomic_json, case_lock, execution_policy, supervise


@pytest.mark.skipif(os.name != "nt", reason="Windows read/replace sharing race")
def test_atomic_phase_write_retries_transient_windows_reader_lock(monkeypatch, tmp_path):
    path = tmp_path / "phase.json"
    atomic_json(path, {"phase": "setup"})
    original = Path.replace
    attempts = []

    def replace(source, destination):
        attempts.append(source)
        if len(attempts) <= 2:
            assert json.loads(path.read_text())["phase"] == "setup"
            raise PermissionError("reader has not closed yet")
        return original(source, destination)

    monkeypatch.setattr(Path, "replace", replace)
    atomic_json(path, {"phase": "fit"})
    assert len(attempts) == 3
    assert json.loads(path.read_text())["phase"] == "fit"


def run_script(tmp_path, body, **limits):
    phase_path = tmp_path / "phase.json"
    script = (
        "import os, time\n"
        "from kd.benchmark_execution import PhaseClock, fit_call\n"
        f"clock = PhaseClock({str(phase_path)!r}, started=float(os.environ['KD_BENCHMARK_STARTED']))\n"
        "with clock:\n" + "\n".join("    " + line for line in body.splitlines())
    )
    policy = execution_policy(timeout=limits.pop("timeout", 0.15), timeout_grace=0,
                              setup_timeout=10, score_timeout=limits.pop("score_timeout", 2), **limits)
    return supervise([sys.executable, "-c", script], cwd=benchmark.ROOT,
                     log_path=tmp_path / "run.log", phase_path=phase_path, policy=policy)


def test_scoring_is_excluded_from_fit_deadline(tmp_path):
    outcome = run_script(tmp_path, "fit_call(time.sleep, 0.02)\ntime.sleep(0.3)")
    assert outcome["returncode"] == 0 and outcome["timeout_phase"] is None
    assert 0 < outcome["fit_seconds"] < 0.15
    assert outcome["score_seconds"] >= 0.3


@pytest.mark.parametrize("phase", ["fit", "score"])
def test_fit_and_scoring_have_separate_hard_limits(tmp_path, phase):
    body = "fit_call(time.sleep, 60)" if phase == "fit" else "fit_call(time.sleep, 0.01)\ntime.sleep(60)"
    outcome = run_script(tmp_path, body, score_timeout=0.1)
    assert outcome["timeout_phase"] == phase
    assert outcome["worker_elapsed_seconds"] < 10


def test_setup_timeout_before_worker_phase_file(tmp_path):
    outcome = supervise([sys.executable, "-c", "import time; time.sleep(60)"],
                        cwd=benchmark.ROOT, log_path=tmp_path / "run.log",
                        phase_path=tmp_path / "missing.json",
                        policy=execution_policy(1, setup_timeout=0.1))
    assert outcome["timeout_phase"] == "setup"


def test_hard_timeout_kills_descendant_process(tmp_path):
    psutil = pytest.importorskip("psutil")
    pid_path = tmp_path / "child.pid"
    body = (
        "import subprocess, sys\n"
        "clock.change('fit')\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], "
        + ("creationflags=subprocess.CREATE_NO_WINDOW" if os.name == "nt" else "start_new_session=False")
        + ")\n"
        f"open({str(pid_path)!r}, 'w').write(str(child.pid))\n"
        "time.sleep(60)"
    )
    outcome = run_script(tmp_path, body, timeout=0.5)
    assert outcome["timeout_phase"] == "fit"
    pid = int(pid_path.read_text())
    deadline = time.monotonic() + 3
    while psutil.pid_exists(pid) and time.monotonic() < deadline:
        if psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
            break
        time.sleep(0.05)
    assert not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE


def test_case_lock_rejects_concurrent_orchestrators(tmp_path):
    with case_lock(tmp_path):
        with pytest.raises(RuntimeError, match="already running"):
            with case_lock(tmp_path):
                pytest.fail("acquired the same case twice")
    with case_lock(tmp_path):
        pass


def test_real_ode_targets_run_and_resume_without_rewriting(tmp_path):
    catalog, _ = benchmark.dataset_catalog()
    case = benchmark.build_experiments(catalog, models=["sindy"],
                                       datasets=["ode_core_oscillator"])[0]
    rows = benchmark.execute_case(case, tmp_path, timeout=10)
    assert [row["status"] for row in rows] == ["ok", "ok"]
    assert all(row["stop_reason"] == "converged" and row["fit_seconds"] > 0 for row in rows)
    directory = tmp_path / "cases" / case["name"]
    workers = [json.loads(path.read_text()) for path in directory.glob("targets/**/worker.json")]
    assert len(workers) == 2 and workers[0]["pid"] != workers[1]["pid"]
    original = {path: path.read_bytes() for path in directory.glob("targets/**/result.json")}
    assert benchmark.execute_case(case, tmp_path, timeout=10, resume=True) == rows
    assert all(path.read_bytes() == value for path, value in original.items())


def test_real_first_target_timeout_still_runs_second_target(monkeypatch, tmp_path):
    catalog, _ = benchmark.dataset_catalog()
    case = benchmark.build_experiments(catalog, models=["sindy"],
                                       datasets=["ode_core_oscillator"])[0]
    wrapper = tmp_path / "slow_first_target.py"
    wrapper.write_text(
        "import json, sys, time\n"
        f"sys.path.insert(0, {str(benchmark.ROOT)!r})\n"
        "import run_benchmark as benchmark\n"
        "spec = json.load(open(sys.argv[sys.argv.index('--_target-file') + 1]))\n"
        "original = benchmark._new_model\n"
        "def create(*args, **kwargs):\n"
        "    model = original(*args, **kwargs)\n"
        "    fit = model.fit\n"
        "    def delayed(*a, **kw):\n"
        "        if spec['target_index'] == 0:\n"
        "            time.sleep(60)\n"
        "        return fit(*a, **kw)\n"
        "    model.fit = delayed\n"
        "    return model\n"
        "benchmark._new_model = create\n"
        "raise SystemExit(benchmark.main(sys.argv[1:]))\n", encoding="utf-8",
    )
    original = benchmark._launch_worker

    def launch(case_path, attempt, policy, spec=None):
        if spec is None:
            return original(case_path, attempt, policy)
        benchmark.write_json(attempt / "target.json", spec)
        return supervise(
            [sys.executable, str(wrapper), "--_case-file", str(case_path),
             "--_attempt-dir", str(attempt), "--_target-file", str(attempt / "target.json")],
            cwd=benchmark.ROOT, log_path=attempt / "run.log",
            phase_path=attempt / "phase.json", policy=policy,
        )

    monkeypatch.setattr(benchmark, "_launch_worker", launch)
    rows = benchmark.execute_case(case, tmp_path / "results", timeout=0.2, timeout_grace=0)
    assert [row["status"] for row in rows] == ["timeout", "ok"]
    assert rows[0]["timeout_phase"] == "fit"
    assert rows[1]["stop_reason"] == "converged"


def test_parallel_reuses_terminal_failures_without_overwriting_destination(monkeypatch, tmp_path):
    from test_run_benchmark import make_case, mock_target_workers

    from scripts import run_parallel_benchmark as parallel

    case = make_case()
    mock_target_workers(monkeypatch, case, ["timeout"])
    rows = benchmark.execute_case(case, tmp_path / "source", timeout=1)
    policy = execution_policy(1)
    assert parallel._reuse_results([case], tmp_path / "copy", [tmp_path / "source"], policy) == 1
    monkeypatch.setattr(benchmark, "_launch_worker", lambda *a, **kw: pytest.fail("reran timeout"))
    assert parallel._run_one(case, tmp_path / "copy", 1) == rows
    assert parallel._reuse_results([case], tmp_path / "copy", [tmp_path / "source"], policy) == 0
    assert parallel._reuse_results([case], tmp_path / "different", [tmp_path / "source"],
                                   execution_policy(2)) == 0
    path = tmp_path / "copy" / "cases" / case["name"] / "result.json"
    original = path.read_bytes()
    with pytest.raises(ValueError, match="timing policy"):
        parallel._run_one(case, tmp_path / "copy", 2)
    assert path.read_bytes() == original


def test_parallel_rejects_automatic_retries(monkeypatch):
    from scripts import run_parallel_benchmark as parallel

    argv = ["runner", "--manifest", "manifest.json", "--compatibility", "compat.json",
            "--output-dir", "output"]
    monkeypatch.setattr(sys, "argv", argv)
    assert parallel._parse_args().retries == 0
    monkeypatch.setattr(sys, "argv", argv + ["--retries", "1"])
    with pytest.raises(SystemExit):
        parallel._parse_args()


def test_cpu_cost_includes_exited_child_and_excludes_sleep(tmp_path):
    body = (
        "import subprocess, sys\n"
        "clock.change('fit')\n"
        "child = subprocess.Popen([sys.executable, '-c', "
        "'import time; end=time.perf_counter()+0.4; exec(\"while time.perf_counter()<end: pass\")'])\n"
        "child.wait()\n"
        "time.sleep(0.3)"
    )
    outcome = run_script(tmp_path, body, timeout=5)
    assert outcome["returncode"] == 0
    assert outcome["cpu_seconds"] >= 0.25
    assert outcome["cpu_seconds"] < outcome["worker_elapsed_seconds"]
    assert outcome["gpu_occupied_seconds"] == 0
    assert outcome["resource_monitor_source"] == "psutil_process_tree_sampled"
    assert json.loads((tmp_path / "supervision.json").read_text())["cpu_seconds"] == outcome["cpu_seconds"]


def host_slots():
    import psutil

    from kd.benchmark_resources import slot_policy

    available = psutil.Process().cpu_affinity()
    if len(available) < 8:
        pytest.skip("Two D2 slots require eight available CPUs")
    return {slot: slot_policy(slot, cpus=cpus) for slot, cpus in
            (("cpu", available[:4]), ("gpu", available[4:8]))}


def test_two_real_slot_workers_have_disjoint_affinity_and_environment(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    slots = host_slots()

    def worker(slot):
        directory = tmp_path / slot
        directory.mkdir()
        report = directory / "environment.json"
        script = (
            "import json, os, psutil, time\n"
            f"json.dump(dict(cpus=psutil.Process().cpu_affinity(), cuda=os.environ['CUDA_VISIBLE_DEVICES'], "
            f"threads=os.environ['OMP_NUM_THREADS']), open({str(report)!r}, 'w'))\n"
            "time.sleep(0.3)\n"
        )
        start = time.perf_counter()
        outcome = supervise([sys.executable, "-c", script], cwd=benchmark.ROOT,
                            log_path=directory / "run.log", phase_path=directory / "phase.json",
                            policy=execution_policy(5, resource_policy=slots[slot]))
        return start, time.perf_counter(), json.loads(report.read_text()), outcome

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(worker, slot) for slot in ("cpu", "gpu")]
        cpu, gpu = [future.result() for future in futures]
    assert max(cpu[0], gpu[0]) < min(cpu[1], gpu[1])
    for slot, result in (("cpu", cpu), ("gpu", gpu)):
        report, outcome = result[2:]
        assert report["cpus"] == slots[slot]["cpu_affinity"]
        assert report["threads"] == "4"
        assert report["cuda"] == ("" if slot == "cpu" else "0")
        assert outcome["returncode"] == 0 and outcome["resource_slot"] == slot
        assert outcome["gpu_occupied_seconds"] == (0 if slot == "cpu" else outcome["worker_elapsed_seconds"])


def test_rss_limit_counts_child_and_kills_whole_tree(tmp_path):
    import psutil

    slots = host_slots()
    resources = dict(slots["cpu"], memory_limit_mb=100.0)
    pid_path = tmp_path / "child.pid"
    script = (
        "import subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; block=bytearray(128*1024*1024); time.sleep(60)'])\n"
        f"open({str(pid_path)!r}, 'w').write(str(child.pid))\n"
        "time.sleep(60)\n"
    )
    outcome = supervise([sys.executable, "-c", script], cwd=benchmark.ROOT,
                        log_path=tmp_path / "run.log", phase_path=tmp_path / "phase.json",
                        policy=execution_policy(5, setup_timeout=10, resource_policy=resources))
    assert outcome["resource_limit"] == "rss"
    assert outcome["timeout_phase"] is None and outcome["peak_memory_mb"] > 100
    pid = int(pid_path.read_text())
    try:
        child = psutil.Process(pid)
        deadline = time.monotonic() + 3
        while child.is_running() and child.status() != psutil.STATUS_ZOMBIE and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not child.is_running() or child.status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        pass
    from test_run_benchmark import make_case

    case = make_case()
    spec = dict(instance=case["dataset"], target="y", target_id="i0000_t0000", ground_truth="x1")
    row = benchmark._worker_failure(case, spec, outcome, tmp_path / "run.log")
    assert row["status"] == "error" and row["stop_reason"] == "resource_limit"
    aggregate = benchmark.aggregate_results([row])[0]
    assert aggregate["n_resource_limit"] == 1 and aggregate["n_error"] == 1
    assert aggregate["cpu_seconds_total"] == row["cpu_seconds"]


def test_slot_resources_in_summary_and_resume_are_immutable(tmp_path):
    from scripts.run_slots import bind_case

    slots = host_slots()
    catalog, _ = benchmark.dataset_catalog()
    source = benchmark.build_experiments(catalog, models=["sindy"], datasets=["ode_core_oscillator"])[0]
    case = bind_case(source, slots)
    assert "resource_policy" not in source
    rows = benchmark.execute_case(case, tmp_path, timeout=10)
    assert [r["status"] for r in rows] == ["ok", "ok"]
    assert all(r["resource_slot"] == "cpu" and r["cpu_seconds"] > 0 for r in rows)
    benchmark.save_summary(tmp_path, rows)
    aggregates = json.loads((tmp_path / "benchmark_aggregate.json").read_text())
    assert all(a["cpu_seconds_total"] > 0 and a["gpu_occupied_seconds_total"] == 0 for a in aggregates)
    directory = tmp_path / "cases" / case["name"]
    original = {p: p.read_bytes() for p in directory.glob("targets/**/result.json")}
    assert benchmark.execute_case(case, tmp_path, timeout=10, resume=True) == rows
    assert all(p.read_bytes() == data for p, data in original.items())
    changed = dict(case["resource_policy"], memory_limit_mb=100.0)
    with pytest.raises(ValueError, match="policy changed"):
        benchmark.execute_case(case, tmp_path, timeout=10, resume=True, resource_policy=changed)
    assert all(p.read_bytes() == data for p, data in original.items())


def test_recover_supervised_candidate_without_rerunning(monkeypatch, tmp_path):
    from test_run_benchmark import make_case

    case = make_case()
    spec = benchmark._target_spec(case["dataset"], 0, "y", 0, "x1")
    directory = tmp_path / "cases" / case["name"]
    attempt = directory / "targets" / spec["target_id"] / "attempts" / "0001"
    row = benchmark._base_row(case, spec["instance"], spec["target"])
    row.update(status="ok", target_id=spec["target_id"], expression="x1", _self_cpu_seconds=0.3)
    benchmark.write_json(attempt / "candidate.json", benchmark._finalize_stop_reason(row))
    benchmark.write_json(attempt / "supervision.json", dict(cpu_seconds=0.4, worker_cpu_seconds=0.2,
                                                           gpu_occupied_seconds=0, returncode=0))
    benchmark.write_json(directory / "execution.json", execution_policy(10))
    monkeypatch.setattr(benchmark, "_launch_worker", lambda *a, **k: pytest.fail("reran supervised target"))
    committed = benchmark._target_result(directory, case, spec)
    assert committed["cpu_seconds"] == pytest.approx(0.5)
    assert "_self_cpu_seconds" not in committed
    assert benchmark._target_result(directory, case, spec) == committed


def test_slot_mapping_uses_method_instead_of_source_device():
    from test_run_benchmark import make_case

    from kd.benchmark_resources import slot_policy
    from scripts.run_slots import bind_case

    slots = {slot: slot_policy(slot) for slot in ("cpu", "gpu")}
    assert slots["cpu"]["memory_limit_mb"] * (1024 ** 2) == 28_000_000_000
    case = make_case(model="dso")
    case["device"] = "cpu"
    bound = bind_case(case, slots)
    assert bound["resource_policy"]["slot"] == "gpu" and bound["device"] == "cuda:0"
    assert bound["model_params"]["device"] == "cuda:0" and case["model_params"]["device"] == "cpu"
    assert bound["name"] != case["name"]


def test_slot_dry_run_does_not_load_data_or_models(monkeypatch, tmp_path):
    from test_run_benchmark import make_case

    from scripts import run_slots

    cases = [make_case(), make_case(model="dso")]
    path = tmp_path / "source.json"
    benchmark.write_json(path, dict(execution_protocol_version=benchmark.EXECUTION_PROTOCOL_VERSION, cases=cases))
    monkeypatch.setattr(run_slots, "execute_case", lambda *a, **k: pytest.fail("fit in dry-run"))
    monkeypatch.setattr(run_slots, "validate_host", lambda *a, **k: pytest.fail("host check in dry-run"))
    output = tmp_path / "output"
    assert run_slots.main(["--manifest", str(path), "--output-dir", str(output), "--dry-run"]) == 0
    manifest = json.loads((output / "benchmark_manifest.json").read_text())
    assert len(manifest["cases"]) == 2
    assert set(manifest["resource_slots"]) == {"cpu", "gpu"}
    assert manifest["source_manifests"][0]["sha256"]
    with pytest.raises(ValueError, match="resource_slots"):
        benchmark.save_manifest(output, {**manifest, "resource_slots": {}})


def test_resource_aggregation_separates_constraints_and_counts_shared_cost_once():
    from test_run_benchmark import make_case

    row = dict(benchmark._base_row(make_case()), status="ok", cpu_seconds=2.0,
               gpu_occupied_seconds=0.0, preparation_cpu_seconds=0.5, system_score_cpu_seconds=0.7)
    groups = benchmark.aggregate_results([row, {**row, "target": "second"},
                                          {**row, "resource_policy_hash": "different", "cpu_seconds": None}])
    all_groups = [g for g in groups if g["model"] == "all"]
    assert len(all_groups) == 2
    measured = next(g for g in all_groups if g["resource_policy_hash"] == "unconstrained")
    assert measured["cpu_seconds_total"] == 4 and measured["cpu_seconds_measured_rows"] == 2
    assert measured["preparation_cpu_seconds_total"] == 0.5
    assert measured["system_score_cpu_seconds_total"] == 0.7
    missing = next(g for g in all_groups if g["resource_policy_hash"] == "different")
    assert missing["cpu_seconds_total"] is None and missing["cpu_seconds_measured_rows"] == 0


@pytest.mark.parametrize("limit", [100, 28 * 1024])
def test_external_gpu_service_is_metered_and_only_killed_on_resource_limit(tmp_path, limit):
    import subprocess

    import psutil

    resources = host_slots()["gpu"]
    service = subprocess.Popen(
        [sys.executable, "-c", "import time; block=bytearray(128*1024*1024); exec('while True: pass')"],
        **({"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {"start_new_session": True}),
    )
    try:
        psutil.Process(service.pid).cpu_affinity(resources["cpu_affinity"])
        time.sleep(0.3)
        resources = dict(resources, service_pids=[service.pid], memory_limit_mb=float(limit))
        outcome = supervise([sys.executable, "-c", "import time; time.sleep(0.4)"], cwd=benchmark.ROOT,
                            log_path=tmp_path / "run.log", phase_path=tmp_path / "phase.json",
                            policy=execution_policy(5, resource_policy=resources))
        assert outcome["peak_memory_mb"] > 100 and outcome["gpu_occupied_seconds"] > 0
        if limit == 100:
            assert outcome["resource_limit"] == "rss"
            service.wait(timeout=3)
        else:
            assert outcome["resource_limit"] is None and service.poll() is None
            assert outcome["cpu_seconds"] >= 0.2
            # Long-lived service CPU before this invocation is excluded.
            assert outcome["cpu_seconds"] < outcome["worker_elapsed_seconds"] + 0.2
    finally:
        if service.poll() is None:
            service.kill()
        service.wait()


def test_slot_scheduler_serializes_each_queue_and_runs_both_queues(monkeypatch, tmp_path):
    import threading

    from test_run_benchmark import make_case

    from scripts import run_slots

    slots = host_slots()
    cases = [make_case(model=model, seeds=[seed]) for model in ("gplearn", "dso") for seed in (0, 1)]
    source = tmp_path / "manifest.json"
    benchmark.write_json(source, dict(execution_protocol_version=benchmark.EXECUTION_PROTOCOL_VERSION, cases=cases))
    barrier, intervals = threading.Barrier(2, timeout=5), {"cpu": [], "gpu": []}

    def execute(case, output, **kwargs):
        slot = case["resource_policy"]["slot"]
        start = time.perf_counter()
        if not intervals[slot]:
            barrier.wait()
        time.sleep(0.04)
        intervals[slot].append((start, time.perf_counter()))
        row = dict(benchmark._base_row(case), status="ok", cpu_seconds=0.02,
                   gpu_occupied_seconds=0.04 if slot == "gpu" else 0)
        return [benchmark._finalize_stop_reason(row)]

    monkeypatch.setattr(run_slots, "execute_case", execute)
    args = ["--manifest", str(source), "--output-dir", str(tmp_path / "run"),
            "--cpu-cpus", ",".join(map(str, slots["cpu"]["cpu_affinity"])),
            "--gpu-cpus", ",".join(map(str, slots["gpu"]["cpu_affinity"]))]
    assert run_slots.main(args) == 0
    for times in intervals.values():
        assert len(times) == 2 and times[0][1] <= times[1][0]
    assert max(intervals["cpu"][0][0], intervals["gpu"][0][0]) < min(intervals["cpu"][0][1], intervals["gpu"][0][1])
    manifest = json.loads((tmp_path / "run" / "benchmark_manifest.json").read_text())
    rows = json.loads((tmp_path / "run" / "benchmark_summary.json").read_text())
    assert len(manifest["cases"]) == len(rows) == 4
    assert {r["resource_slot"] for r in rows} == {"cpu", "gpu"}


@pytest.mark.parametrize("model", ["gplearn", "pyoperon", "pysr", "llmsr", "dso", "sisso"])
def test_method_thread_environment_is_applied_before_numpy_import(tmp_path, model):
    from kd.benchmark_resources import slot_for_model

    resources = host_slots()[slot_for_model(model)]
    report = tmp_path / "threads.json"
    script = (
        "import os, json\n"
        "import numpy as np\n"
        "from threadpoolctl import threadpool_info\n"
        "np.ones((20,20)) @ np.ones((20,20))\n"
        f"json.dump(dict(blas=threadpool_info(), omp=os.environ['OMP_NUM_THREADS'], "
        "julia=os.environ['JULIA_NUM_THREADS'], juliacall=os.environ['PYTHON_JULIACALL_THREADS'], "
        f"torch=os.environ['KD_SLOT_THREADS']), open({str(report)!r}, 'w'))\n"
    )
    outcome = supervise([sys.executable, "-c", script], cwd=benchmark.ROOT,
                        log_path=tmp_path / "run.log", phase_path=tmp_path / "phase.json",
                        policy=execution_policy(10, resource_policy=resources), model=model)
    assert outcome["returncode"] == 0
    actual = json.loads(report.read_text())
    count = 4 if model in {"dso", "sisso"} else 1
    assert actual["omp"] == actual["julia"] == actual["juliacall"] == str(count)
    assert actual["torch"] == "4"
    assert actual["blas"] and all(library["num_threads"] == count for library in actual["blas"])
    assert outcome["thread_environment"]["PYTHON_JULIACALL_THREADS"] == str(count)


def test_torch_uses_four_threads_inside_a_slot_worker(tmp_path):
    resources = host_slots()["gpu"]
    report = tmp_path / "torch_threads.json"
    script = (
        "import json, torch\n"
        "import run_benchmark as benchmark\n"
        "torch.set_num_threads(1)\n"
        "catalog,_=benchmark.dataset_catalog()\n"
        "case=benchmark.build_experiments(catalog,models=['dso'],datasets=['Koza-2'])[0]\n"
        "benchmark._load_instances=lambda case: iter([('tiny',object())])\n"
        "benchmark._case_problems=lambda dataset,case: [dict(target='y')]\n"
        "benchmark._run_problem=lambda *args: dict(expression='x1',threads=torch.get_num_threads())\n"
        f"row=benchmark.run_case(case,{str(tmp_path / 'work')!r})[0]\n"
        f"json.dump(row,open({str(report)!r},'w'))\n"
    )
    outcome = supervise([sys.executable, "-c", script], cwd=benchmark.ROOT,
                        log_path=tmp_path / "run.log", phase_path=tmp_path / "phase.json",
                        policy=execution_policy(10, resource_policy=resources), model="dso")
    assert outcome["returncode"] == 0
    row = json.loads(report.read_text())
    assert row["status"] == "ok" and row["threads"] == 4


def test_four_gplearn_workers_inherit_slot_affinity_and_single_blas_thread(tmp_path):
    resources = host_slots()["cpu"]
    report = tmp_path / "workers.json"
    script = (
        "import json, os, time, psutil, numpy as np\n"
        "from gplearn.genetic import SymbolicRegressor\n"
        "from joblib import Parallel, delayed\n"
        "from threadpoolctl import threadpool_info\n"
        "X=np.linspace(-1,1,40).reshape(-1,1); y=X[:,0]**2\n"
        "model=SymbolicRegressor(n_jobs=4,population_size=32,generations=2,random_state=0)\n"
        "model.fit(X,y)\n"
        "assert np.all(np.isfinite(model.predict(X)))\n"
        "def inspect_worker():\n"
        "    time.sleep(0.15)\n"
        "    return dict(pid=os.getpid(), cpus=psutil.Process().cpu_affinity(), blas=threadpool_info())\n"
        "workers=Parallel(n_jobs=model.n_jobs,batch_size=1)(delayed(inspect_worker)() for _ in range(8))\n"
        f"json.dump(workers,open({str(report)!r},'w'))\n"
    )
    outcome = supervise([sys.executable, "-c", script], cwd=benchmark.ROOT,
                        log_path=tmp_path / "run.log", phase_path=tmp_path / "phase.json",
                        policy=execution_policy(30, setup_timeout=60, resource_policy=resources), model="gplearn")
    assert outcome["returncode"] == 0, (tmp_path / "run.log").read_text()
    workers = json.loads(report.read_text())
    assert len({worker["pid"] for worker in workers}) == 4
    assert all(worker["cpus"] == resources["cpu_affinity"] for worker in workers)
    assert all(library["num_threads"] == 1 for worker in workers for library in worker["blas"])


def test_legacy_c5_parallel_policy_is_preserved_and_separate_from_c7():
    from kd.benchmark_resources import policy_hash, slot_policy, validate_policy, worker_environment

    current = slot_policy("cpu")
    legacy = dict(current)
    legacy.pop("parallelism_protocol")
    assert validate_policy(legacy) == legacy
    assert policy_hash(current) != policy_hash(legacy)
    inherited = {"PYTHON_JULIACALL_THREADS": "auto"}
    assert worker_environment(legacy, inherited, model="pysr")["PYTHON_JULIACALL_THREADS"] == "auto"
    assert worker_environment(current, inherited, model="pysr")["PYTHON_JULIACALL_THREADS"] == "1"


def test_native_operon_uses_four_threads_with_a_small_search(tmp_path):
    import numpy as np
    from test_run_benchmark import make_case

    params = make_case(model="pyoperon", profile="full")["model_params"]
    params.update(population_size=32, generations=2, max_evaluations=2000, optimizer_iterations=2)
    model = benchmark._new_model("pyoperon", params)
    assert model.n_threads == 4
    X = np.linspace(-1, 1, 40).reshape(-1, 1)
    model.fit(X, X[:, 0] ** 2)
    assert np.all(np.isfinite(model.predict(X)))


def test_native_pysr_four_process_search_is_bounded_and_inherits_affinity(tmp_path, monkeypatch):
    import importlib.util

    from kd.benchmark_resources import ProcessTreeMonitor

    if importlib.util.find_spec("pysr") is None or not os.environ.get("PYTHON_JULIAPKG_EXE"):
        pytest.skip("Requires the configured local PySR/Julia environment")
    resources = host_slots()["cpu"]
    samples = []
    original_sample = ProcessTreeMonitor.sample

    def sample(monitor):
        rss = original_sample(monitor)
        processes = []
        for process in monitor.processes.values():
            try:
                if "julia" in process.name().lower():
                    processes.append(dict(pid=process.pid, cpus=process.cpu_affinity()))
            except monitor.psutil.Error:
                pass
        samples.append(processes)
        return rss

    # Observe outside JuliaCall: a Python thread in the worker can be blocked
    # by Julia holding its GIL throughout the search.
    monkeypatch.setattr(ProcessTreeMonitor, "sample", sample)
    report, phase = tmp_path / "pysr.json", tmp_path / "phase.json"
    script = (
        "import os, json, numpy as np\n"
        "from kd.benchmark_execution import PhaseClock, fit_call\n"
        f"clock=PhaseClock({str(phase)!r},started=float(os.environ['KD_BENCHMARK_STARTED']))\n"
        "from pysr import PySRRegressor, jl\n"
        "X=np.linspace(-1,1,24).reshape(-1,1); y=X[:,0]**2\n"
        "model=PySRRegressor(niterations=1,populations=4,population_size=20,maxsize=10, "
        "parallelism='multiprocessing',procs=4,deterministic=False,random_state=0, "
        "binary_operators=['+','*'],unary_operators=[],verbosity=0,progress=False, "
        f"output_directory={str(tmp_path / 'equations')!r})\n"
        "with clock: fit_call(model.fit,X,y)\n"
        "assert np.all(np.isfinite(model.predict(X)))\n"
        f"json.dump(dict(procs=model.procs,threads=int(jl.seval('Threads.nthreads()'))),open({str(report)!r},'w'))\n"
    )
    outcome = supervise([sys.executable, "-c", script], cwd=benchmark.ROOT,
                        log_path=tmp_path / "run.log", phase_path=phase,
                        policy=execution_policy(180, setup_timeout=180, score_timeout=30,
                                                resource_policy=resources), model="pysr")
    assert outcome["returncode"] == 0 and outcome["timeout_phase"] is None, (tmp_path / "run.log").read_text()
    actual = json.loads(report.read_text())
    assert actual["procs"] == 4 and actual["threads"] == 1
    assert max(map(len, samples)) >= 4  # JuliaCall embeds the main runtime in Python.
    assert all(process["cpus"] == resources["cpu_affinity"] for sample in samples for process in sample)
